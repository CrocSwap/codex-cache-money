#!/usr/bin/env python3
"""
cache-money-audit.py — Scan Codex session rollouts to find where the prompt
cache is being thrown away, and what it costs you.

Codex reports caching very differently from Claude Code, and the difference
drives the whole design of this tool:

  * There is no cache_creation signal. OpenAI bills cache writes only for
    explicit cache breakpoints, which Codex doesn't use
    (cache_write_input_tokens is 0 in practice), so "cold turn = big cache
    write" has no analogue.
  * input_tokens INCLUDES cached_input_tokens. Uncached = input - cached.
  * Uncached input on a healthy turn is NOT a cache miss — it is the content
    appended since the last call (new user message, tool output). It is
    irreducible and runs ~3-5% of input on every warm turn.
  * The cache is eviction-driven, not TTL-driven. There is no cliff to race:
    hit rates decay smoothly and never reach zero.

So a real cache loss is detected structurally, not by a token ratio: the
cached prefix comes back SMALLER than the prompt you already sent.

    expected  = floor128(prev.input + prev.output - prev.reasoning)
    expected  = min(expected, floor128(cur.input))
    shortfall = max(0, expected - cur.cached)

Reasoning tokens are subtracted because they are not re-sent on the next
call; the 128-token floor matches the cache's block quantization. Shortfalls
below --tolerance are treated as quantization noise rather than evictions.

Usage:
    python3 cache-money-audit.py                    # last 7 days
    python3 cache-money-audit.py --start 18h        # or 30m, 30d, 2w
    python3 cache-money-audit.py --since-reset     # current primary limit window
    python3 cache-money-audit.py --previous-reset  # the window before that
    python3 cache-money-audit.py --start 7d --end 1d
    python3 cache-money-audit.py --start 2026-09-01 --end 2026-09-15
    python3 cache-money-audit.py --decay            # hit-rate decay by idle gap
    python3 cache-money-audit.py --no-by-project    # skip the per-repo breakdown
    python3 cache-money-audit.py --verbose          # every material eviction
    python3 cache-money-audit.py --json

Reads $CODEX_HOME/sessions/**/rollout-*.jsonl, or
~/.codex/sessions/**/rollout-*.jsonl when CODEX_HOME is unset.
--sessions-dir overrides this location.
"""

import argparse
import functools
import glob
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

# The prefix cache stores whole 128-token blocks; a partial block never hits.
CACHE_BLOCK = 128

# OpenAI public API list prices, USD per 1M tokens, as of 2026-09-29.
# Source: https://developers.openai.com/api/docs/pricing
# Cached input is 10% of input across the GPT-5.x and GPT-6 families,
# except GPT-6.1 Sol at 5%, which makes a miss 20x the cached price.
#
# Standard tier, short-context rates (<=272K input tokens). Long context
# doubles input and is 1.5x output, but Codex caps its window at 258,400
# tokens, so it never gets there.
#
# The 5.6, GPT-6 and GPT-6.1 families also list a cache-write rate (1.25x input).
# It replaces the input rate rather than adding to it, and applies only to
# tokens written through explicit cache breakpoints. Codex reports
# cache_write_input_tokens = 0 on every call, so it isn't used here.
#
# These are list prices for the API. On a ChatGPT/Codex subscription you do
# not pay them — everything derived from this table is a notional
# "what this usage would have cost on the API" figure, not a bill.
PRICING = {
    "gpt-6.1-sol":   {"input":  2.00, "cached": 0.10,  "output": 10.00},
    "gpt-6-astra":   {"input": 10.00, "cached": 1.00,  "output": 50.00},
    "gpt-6-sol":     {"input":  2.00, "cached": 0.20,  "output": 10.00},
    "gpt-6-luna":    {"input":  0.10, "cached": 0.01,  "output":  0.50},
    # Promotional, "available at least through November 21, 2026".
    "gpt-5.6-sol":   {"input": 4.00,  "cached": 0.40,  "output": 20.00},
    "gpt-5.6-terra": {"input": 2.00,  "cached": 0.20,  "output": 12.00},
    "gpt-5.6-luna":  {"input": 0.20,  "cached": 0.02,  "output":  1.20},
    "gpt-5.5":       {"input": 5.00,  "cached": 0.50,  "output": 30.00},
    "gpt-5.4-mini":  {"input": 0.75,  "cached": 0.075, "output":  4.50},
    # Listed so prefix matching doesn't price it as gpt-5.4 (12x too high).
    "gpt-5.4-nano":  {"input": 0.20,  "cached": 0.02,  "output":  1.25},
    "gpt-5.4":       {"input": 2.50,  "cached": 0.25,  "output": 15.00},
    "gpt-5.3-codex": {"input": 1.75,  "cached": 0.175, "output": 14.00},
}
PRICING_AS_OF = "2026-09-29"


def price_for(model: str):
    """Look up list pricing for a model slug, or None if we don't know it.

    Longest-prefix match so variants inherit their base model's price
    (gpt-5.3-codex-spark -> gpt-5.3-codex). Returning None rather than
    guessing keeps unpriced traffic visible instead of silently costed at
    some default rate.
    """
    if not model:
        return None
    if model in PRICING:
        return PRICING[model]
    matches = [k for k in PRICING if model.startswith(k)]
    return PRICING[max(matches, key=len)] if matches else None


def call_cost(call: dict, price: dict) -> dict:
    """API-equivalent cost of one call, split by token type. input_tokens is
    inclusive of cached, so the uncached remainder is billed at the full
    input rate. Output is billed including reasoning tokens."""
    return {
        "uncached": call["uncached"] * price["input"] / 1_000_000,
        "cached": call["cached"] * price["cached"] / 1_000_000,
        "output": call["output"] * price["output"] / 1_000_000,
    }


def cost_row(**extra) -> dict:
    """Accumulator for cost split the way the reports show it. Shortfall is
    the extra paid on uncached input for missed cache, so it sits inside
    cost_uncached and cost, not on top of them."""
    return dict(cost=0.0, cost_uncached=0.0, cost_cached=0.0,
                cost_output=0.0, cost_shortfall=0.0, **extra)


def add_cost(row: dict, parts: dict):
    row["cost"] += sum(parts.values())
    for k, v in parts.items():
        row["cost_" + k] += v


def round_costs(row: dict) -> dict:
    return {k: round(v, 4) if k.startswith("cost") else v for k, v in row.items()}

# Gap buckets for the decay curve. Fine near the minutes that matter,
# coarse out to multi-day so resumed sessions land somewhere sensible.
GAP_EDGES = [0, 60, 300, 600, 1200, 1800, 3600, 7200, 21600, 86400, float("inf")]


# ─────────────────────────── Parsing ───────────────────────────

def iter_rollout_records(filepath: str, contains=None):
    """Yield parsed JSON records from one rollout file, skipping junk lines.

    With contains, skip lines lacking that substring before parsing them.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if contains is not None and contains not in line:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except (IOError, OSError) as e:
        print(f"Warning: could not read {filepath}: {e}", file=sys.stderr)


def parse_timestamp(raw: str):
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_time_arg(raw: str, now: datetime) -> datetime:
    """Parse a --start/--end value into an aware UTC datetime.

    Either a lookback from now (30m, 18h, 7d, 2w) or an ISO date/datetime,
    read as UTC unless it carries its own offset. Raises ValueError.
    """
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhdw])", raw.strip().lower())
    if m:
        return now - timedelta(seconds=float(m.group(1)) * DURATION_UNITS[m.group(2)])
    ts = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def parse_rollout(filepath: str, since: datetime, until: datetime):
    """Parse one rollout file into normalized call records.

    A rollout interleaves three record types we care about:
      session_meta  — once, at the top: session id, cwd, git, cli version
      turn_context  — per turn: model, reasoning effort, cwd
      event_msg/token_count — per API call: usage + live rate limits
    """
    session_id = os.path.basename(filepath).replace(".jsonl", "")
    cwd = git_branch = repo = None
    model = effort = None
    plan_type = None

    for record in iter_rollout_records(filepath):
        rtype = record.get("type")
        payload = record.get("payload") or {}

        if rtype == "session_meta":
            session_id = payload.get("session_id") or session_id
            cwd = payload.get("cwd") or cwd
            git = payload.get("git") or {}
            git_branch = git.get("branch") or git_branch
            repo = git.get("repository_url") or repo

        elif rtype == "turn_context":
            model = payload.get("model") or model
            effort = payload.get("effort") or effort
            cwd = payload.get("cwd") or cwd

        elif rtype == "event_msg" and payload.get("type") == "token_count":
            info = payload.get("info") or {}
            last = info.get("last_token_usage") or {}
            if not last:
                continue
            ts = parse_timestamp(record.get("timestamp", ""))
            if ts is None or ts < since or ts > until:
                continue

            limits = payload.get("rate_limits") or {}
            if limits.get("plan_type"):
                plan_type = limits["plan_type"]

            total_input = last.get("input_tokens", 0) or 0
            cached = last.get("cached_input_tokens", 0) or 0

            yield {
                "ts": ts,
                "ordinal": record.get("ordinal", 0) or 0,
                "session_id": session_id,
                "model": model or "unknown",
                "effort": effort or "unknown",
                "cwd": cwd,
                "repo": repo,
                "branch": git_branch,
                "plan_type": plan_type,
                # input_tokens is inclusive of cached; split it here so no
                # downstream code has to remember that.
                "input": total_input,
                "cached": cached,
                "uncached": max(0, total_input - cached),
                "output": last.get("output_tokens", 0) or 0,
                "reasoning": last.get("reasoning_output_tokens", 0) or 0,
                "ctx_window": info.get("model_context_window"),
                "limits": limits,
                "file": filepath,
            }


def find_rollout_files(root: str):
    base = os.path.expanduser(root)
    if not os.path.isdir(base):
        print(f"No Codex session directory at {base}", file=sys.stderr)
        print("Is Codex installed and has it recorded any sessions?", file=sys.stderr)
        sys.exit(1)
    return sorted(glob.glob(os.path.join(base, "**", "rollout-*.jsonl"), recursive=True))


def latest_limit_snapshot(files, until, name, accept=None):
    """Newest (ts, limit) snapshot at or before until that accept() allows.

    Read limit-only events too: they need not carry token usage.
    """
    latest = None
    dated_files = []
    for path in files:
        try:
            dated_files.append((os.path.getmtime(path), path))
        except OSError:
            continue
    for mtime, path in sorted(dated_files, reverse=True):
        if latest is not None and mtime < latest[0].timestamp():
            break
        for record in iter_rollout_records(path, contains="rate_limits"):
            payload = record.get("payload") or {}
            if record.get("type") != "event_msg" or payload.get("type") != "token_count":
                continue
            ts = parse_timestamp(record.get("timestamp", ""))
            if ts is None or ts.tzinfo is None or ts > until:
                continue
            limits = payload.get("rate_limits") or {}
            if limits.get("limit_id") not in (None, "codex"):
                continue
            lim = limits.get(name)
            if not lim or (latest is not None and ts <= latest[0]):
                continue
            if accept is None or accept(ts, lim):
                latest = (ts, lim)
    return latest


def limit_window(ts, lim, what):
    """(start, reset, minutes) for a snapshot, or ValueError if inconsistent."""
    try:
        minutes = float(lim["window_minutes"])
        reset = datetime.fromtimestamp(float(lim["resets_at"]), timezone.utc)
        start = reset - timedelta(minutes=minutes)
        if not minutes > 0 or start > ts or reset <= ts:
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        raise ValueError(f"{what} snapshot has invalid reset or window data.") from None
    return start, reset, minutes


def reset_window_start(files, until, name):
    """Infer a window start from the newest snapshot at or before until.

    Never roll an expired snapshot forward, since the next window may start
    on activity.
    """
    latest = latest_limit_snapshot(files, until, name)
    if latest is None:
        raise ValueError(f"No {name} rate-limit snapshot found at or before the report end.")
    ts, lim = latest
    start, reset, minutes = limit_window(ts, lim, f"Latest {name}")
    if reset <= until:
        raise ValueError(f"Latest {name} snapshot expired at {reset.isoformat()}; "
                         "record fresh Codex activity or use --start explicitly.")
    return start, {"limit": name, "snapshot_at": ts.isoformat(),
                   "resets_at": reset.isoformat(), "window_minutes": minutes,
                   "inferred": True}


# resets_at jitters by a minute or so between snapshots of the same window.
RESET_JITTER = timedelta(minutes=5)


def previous_reset_window(files, until, name):
    """Infer (start, end, source) for the window before the one current at until.

    Uses the newest snapshot taken before the current window began whose reset
    is distinct from the current one's (beyond jitter). The window ends at its own reset or at
    the current window's start, whichever is first: limits can reset early.
    """
    cur_start, cur = reset_window_start(files, until, name)
    cur_reset = datetime.fromisoformat(cur["resets_at"])

    def earlier_window(ts, lim):
        try:
            reset = datetime.fromtimestamp(float(lim["resets_at"]), timezone.utc)
        except (KeyError, TypeError, ValueError, OverflowError, OSError):
            return True  # let limit_window report it as invalid
        return abs(reset - cur_reset) > RESET_JITTER

    latest = latest_limit_snapshot(files, cur_start, name, earlier_window)
    if latest is None:
        raise ValueError(f"No {name} rate-limit snapshot found from before the current "
                         f"window (started {cur_start.isoformat()}).")
    ts, lim = latest
    start, reset, minutes = limit_window(ts, lim, f"Previous {name}")
    end = min(reset, cur_start)
    return start, end, {"limit": name, "snapshot_at": ts.isoformat(),
                        "window_start": start.isoformat(), "resets_at": reset.isoformat(),
                        "window_minutes": minutes, "ended_at": end.isoformat(),
                        "early_reset": reset > cur_start,
                        "next_window_start": cur_start.isoformat(), "inferred": True}


def collect_sessions(files, since, until):
    """Group calls by session id, ordered, with exact duplicates removed.

    A session can span several rollout files (resume / fork both continue the
    same session_id), and a file can repeat an identical token_count record,
    so dedupe on the full usage fingerprint.
    """
    sessions = defaultdict(list)
    for filepath in files:
        # A file last written before the window opened can't hold anything
        # inside it. Skipping it keeps short windows fast; resumed sessions
        # append to old files, so mtime is the safe test, not the path's date.
        try:
            if os.path.getmtime(filepath) < since.timestamp():
                continue
        except OSError:
            continue
        for call in parse_rollout(filepath, since, until):
            sessions[call["session_id"]].append(call)

    for sid, calls in sessions.items():
        calls.sort(key=lambda c: (c["ts"], c["ordinal"]))
        deduped, seen = [], set()
        for c in calls:
            fp = (c["ts"], c["input"], c["cached"], c["output"])
            if fp in seen:
                continue
            seen.add(fp)
            deduped.append(c)
        sessions[sid] = deduped
    return sessions


@functools.lru_cache(maxsize=None)
def repo_root(cwd: str):
    """The git repository a working directory belongs to, or None.

    Walks up to the nearest .git, so subdirectories and agent worktrees nested
    inside a repo (repo/.fadeno/local/worktrees/x) land on the repo itself,
    even after the worktree is deleted, as most are by the time this runs.
    A linked worktree's .git file points into the main repo's
    .git/worktrees/<name>, so follow it back. Stops below $HOME so a dotfiles
    repo there can't claim everything.
    """
    home = os.path.expanduser("~")
    path = os.path.abspath(cwd)
    while path != home and os.path.dirname(path) != path:
        dotgit = os.path.join(path, ".git")
        if os.path.isdir(dotgit):
            return path
        if os.path.isfile(dotgit):
            try:
                with open(dotgit, encoding="utf-8") as f:
                    line = f.readline().strip()
            except OSError:
                return path
            gitdir = line[len("gitdir:"):].strip() if line.startswith("gitdir:") else ""
            gitdir = os.path.normpath(os.path.join(path, gitdir))
            marker = os.sep + os.path.join(".git", "worktrees") + os.sep
            return gitdir.split(marker)[0] if marker in gitdir else path
        path = os.path.dirname(path)
    return None


@functools.lru_cache(maxsize=None)
def project_of(cwd, repo):
    """(key, name, directory) for the project a call ran in.

    The repo on disk when the working directory resolves to one; otherwise
    the recorded repository URL, then the working directory itself. Keyed by
    full path or URL so two projects sharing a basename stay separate.
    """
    root = repo_root(cwd) if cwd else None
    if root:
        return root, os.path.basename(root), root
    key = repo or cwd or "unknown"
    name = os.path.basename(key.rstrip("/")) or key
    # Repo URLs end in .git and directories don't.
    if name.endswith(".git"):
        name = name[:-4]
    return key, name, cwd


# ─────────────────────────── Shortfall detection ───────────────────────────

def floor_block(n: int) -> int:
    return (n // CACHE_BLOCK) * CACHE_BLOCK


def expected_prefix(parent: dict, cur: dict) -> int:
    """Tokens `cur` should read from cache if it continues `parent`.

    Everything we sent last call plus what the model wrote back, minus
    reasoning (not replayed on the next call), floored to the cache's
    block size.
    """
    carried = max(0, parent["output"] - parent["reasoning"])
    return min(floor_block(parent["input"] + carried), floor_block(cur["input"]))


def analyze_call(history: list, cur: dict, nxt, tolerance_frac: float,
                 tolerance_abs: int, idle_gap: int, compaction_ratio: float):
    """Score one call against recent history in its session.

    Codex runs sub-agents concurrently and they all append token_count events
    to the same rollout, with no agent or turn id attached. So the call
    immediately before this one is frequently a *sibling agent's* call with a
    completely different prefix — comparing against it reports a near-total
    eviction at a two-second gap, which is nonsense.

    So look back over a window of recent calls for a plausible parent: one
    whose prompt fits inside ours (context grows within a stream) and whose
    prefix our cache actually covers. If such a call exists, the cache is
    accounted for and nothing was lost. Only when nothing in the window can
    explain what came back do we count an eviction, measured against the most
    recent plausible parent.

    This under-reports rather than over-reports: an eviction indistinguishable
    from interleaving is not counted.
    """
    if not history:
        return None

    # Compaction first: the prompt shrank, so the old prefix no longer applies
    # and no prior call can be a subset parent. Checking this before the parent
    # search keeps compactions from being silently skipped.
    prev = history[-1]
    if cur["input"] < prev["input"] * compaction_ratio:
        expected = floor_block(cur["input"])
        shortfall = max(0, expected - cur["cached"])
        tolerance = max(tolerance_abs, int(tolerance_frac * expected))
        return {
            "ts": cur["ts"],
            "session_id": cur["session_id"],
            "model": cur["model"],
            "effort": cur["effort"],
            "repo": cur["repo"],
            "cwd": cur["cwd"],
            "gap": max(0, (cur["ts"] - prev["ts"]).total_seconds()),
            "expected": expected,
            "shortfall": shortfall if shortfall > tolerance else 0,
            "material": shortfall > tolerance,
            "cause": "compaction",
        }

    candidates = [h for h in history if h["input"] <= cur["input"]]
    if not candidates:
        return None

    parent = expected = None
    shortfall = 0
    for cand in reversed(candidates):        # most recent first
        exp = expected_prefix(cand, cur)
        if exp <= 0:
            continue
        tol = max(tolerance_abs, int(tolerance_frac * exp))
        if exp - cur["cached"] <= tol:       # this call explains our cache
            parent, expected = cand, exp
            break

    if parent is None:
        parent = candidates[-1]
        expected = expected_prefix(parent, cur)
        if expected <= 0:
            return None
        shortfall = max(0, expected - cur["cached"])

    gap = (cur["ts"] - parent["ts"]).total_seconds()
    if gap < 0:
        return None

    tolerance = max(tolerance_abs, int(tolerance_frac * expected))
    material = shortfall > tolerance

    # A real eviction persists: the next call has to rebuild too. If the very
    # next call reads back a prefix as large as the one we just "lost", the
    # prefix was never gone — that single request just routed to a backend
    # that didn't have it warm. You still paid for it, but nothing you did
    # caused it and nothing you do prevents it.
    recovered = nxt is not None and nxt["cached"] >= expected * 0.9

    if recovered:
        cause = "routing"
    elif gap >= idle_gap:
        cause = "idle"
    else:
        cause = "churn"

    return {
        "ts": cur["ts"],
        "session_id": cur["session_id"],
        "model": cur["model"],
        "effort": cur["effort"],
        "repo": cur["repo"],
        "cwd": cur["cwd"],
        "gap": gap,
        "expected": expected,
        "shortfall": shortfall if material else 0,
        "material": material,
        "cause": cause,
    }


def bucket_index(gap: float) -> int:
    for i in range(len(GAP_EDGES) - 1):
        if GAP_EDGES[i] <= gap < GAP_EDGES[i + 1]:
            return i
    return len(GAP_EDGES) - 2


# ─────────────────────────── Formatting ───────────────────────────

def format_tokens(n) -> str:
    """Three significant figures: 4.27M, 42.7M, 427M."""
    n = int(n or 0)
    if abs(n) < 1_000:
        return str(n)
    # Round before picking the unit, so 999,999 reads 1.00M rather than 1000k.
    n = round(n, 3 - len(str(abs(n))))
    for scale, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if abs(n) >= scale:
            v = n / scale
            decimals = 2 if abs(v) < 10 else 1 if abs(v) < 100 else 0
            return f"{v:.{decimals}f}{suffix}"


def format_pct(frac: float) -> str:
    pct = (frac or 0) * 100
    if pct >= 10:
        return f"{pct:.1f}%"
    if pct >= 1:
        return f"{pct:.2f}%"
    if pct > 0:
        return f"{pct:.3f}%"
    return "0%"


def format_cost(c: float) -> str:
    c = c or 0.0
    if c >= 1000:
        return f"${c:,.0f}"
    if c >= 1:
        return f"${c:,.2f}"
    if c >= 0.01:
        return f"${c:.3f}"
    if c > 0:
        return f"${c:.4f}"
    return "$0"


def compact_path(path: str) -> str:
    """Shorten an absolute path for display: ~ for home, elide the middle."""
    home = os.path.expanduser("~")
    if path.startswith(home):
        path = "~" + path[len(home):]
    if len(path) <= 44:
        return path
    parts = path.split(os.sep)
    if len(parts) <= 3:
        return path
    return os.sep.join([parts[0], "…"] + parts[-2:])


def format_gap(s) -> str:
    if s == float("inf"):
        return "∞"
    if s < 120:
        return f"{int(s)}s"
    if s < 3600:
        return f"{int(s/60)}m"
    if s < 86400:
        return f"{int(s/3600)}h"
    return f"{int(s/86400)}d"


def rule(char="─", width=69):
    return char * width


def header(title, sub=None):
    print()
    print("=" * 69)
    print(f"  {title}")
    if sub:
        print(f"  {sub}")
    print("=" * 69)
    print()


def section(title):
    print(rule())
    print(f"  {title}")
    print(rule())
    print()


def priced_by_cost(rows: dict):
    """(model, cost row) pairs for models with a list price, costliest first."""
    return sorted(((model, m) for model, m in rows.items() if price_for(model)),
                  key=lambda kv: -kv[1]["cost"])


def print_cost_table(title: str, rows):
    """A (label, cost row) table split uncached / cached / output, total
    last, with the shortfall that's already inside uncached alongside."""
    print(f"  {title:<18} {'Uncached':>9} {'Cached':>9} "
          f"{'Shortfall':>9} {'Output':>9} {'Total':>9}")
    print(f"  {'─'*18} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*9}")
    for label, m in rows:
        print(f"  {label[:18]:<18} {format_cost(m['cost_uncached']):>9} "
              f"{format_cost(m['cost_cached']):>9} "
              f"{format_cost(m['cost_shortfall']):>9} "
              f"{format_cost(m['cost_output']):>9} "
              f"{format_cost(m['cost']):>9}")
    print()


def print_token_table(title: str, rows):
    """The token counts behind print_cost_table, same columns, with Lost
    (shortfall tokens) where that table has Shortfall."""
    print(f"  {title:<18} {'Uncached':>9} {'Cached':>9} "
          f"{'Lost':>9} {'Output':>9} {'Total':>9}")
    print(f"  {'─'*18} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*9}")
    for label, m in rows:
        print(f"  {label[:18]:<18} {format_tokens(m['input'] - m['cached']):>9} "
              f"{format_tokens(m['cached']):>9} "
              f"{format_tokens(m['shortfall']):>9} "
              f"{format_tokens(m['output']):>9} "
              f"{format_tokens(m['input'] + m['output']):>9}")
    print()


# ─────────────────────────── Reports ───────────────────────────

def report_decay(buckets, as_json):
    """Hit-rate decay by idle gap. Replaces the TTL-cliff hunt, which has no
    analogue here — OpenAI's cache is evicted, not expired."""
    rows = []
    for i, b in enumerate(buckets):
        if b["expected"] == 0:
            continue
        rows.append({
            "gap_lo_s": GAP_EDGES[i],
            "gap_hi_s": GAP_EDGES[i + 1] if GAP_EDGES[i + 1] != float("inf") else None,
            "calls": b["calls"],
            "events": b["events"],
            "expected_tokens": b["expected"],
            "shortfall_tokens": b["shortfall"],
            "miss_rate": round(b["shortfall"] / b["expected"], 4),
        })

    if as_json:
        print(json.dumps({"mode": "decay", "buckets": rows}, indent=2))
        return

    header("CACHE DECAY BY IDLE GAP",
           "share of the expected prefix that failed to hit cache")
    print(f"  {'Gap':<12} {'Calls':>10} {'Events':>9} {'Expected':>11} "
          f"{'Lost':>10} {'Miss':>8}")
    print(f"  {'─'*12} {'─'*10} {'─'*9} {'─'*11} {'─'*10} {'─'*8}")
    for i, b in enumerate(buckets):
        if b["expected"] == 0:
            continue
        label = f"{format_gap(GAP_EDGES[i])}–{format_gap(GAP_EDGES[i+1])}"
        miss = b["shortfall"] / b["expected"]
        print(f"  {label:<12} {b['calls']:>10,} {b['events']:>9,} "
              f"{format_tokens(b['expected']):>11} {format_tokens(b['shortfall']):>10} "
              f"{format_pct(miss):>8}")
    print()
    print(rule())
    print("  Codex's cache is evicted under memory pressure, not expired on a")
    print("  timer — expect a smooth decay, not a cliff. There is no TTL to set")
    print("  and no deadline to beat; longer gaps are just progressively worse.")
    print(rule())
    print()


def report_main(stats, args):
    s = stats
    span = (s["until"] - s["since"]).total_seconds()
    # Sub-day detail only matters for short windows; times are UTC, like the
    # timestamps in the rollouts.
    fmt = "%Y-%m-%d %H:%M" if span < 2 * 86400 else "%Y-%m-%d"
    tz = " UTC" if span < 2 * 86400 else ""
    header("CODEX CACHE MONEY",
           f"{s['since']:{fmt}} → {s['until']:{fmt}}{tz}  ·  {format_gap(span)}")

    print(f"  Sessions analyzed:        {s['n_sessions']:,}")
    print(f"  API calls:                {s['n_calls']:,}")
    if s["plan_type"]:
        print(f"  Plan:                     {s['plan_type']}")
    if s["models"]:
        top = ", ".join(f"{m} ({c:,} calls)" for m, c in s["models"][:3])
        print(f"  Models:                   {top}")
    if s["efforts"]:
        top = ", ".join(f"{e} ({c:,})" for e, c in s["efforts"][:3])
        print(f"  Reasoning effort:         {top}")
    print()

    # ── token flow ──
    section("TOKEN FLOW")
    total_in = s["total_input"]
    hit = s["total_cached"] / total_in if total_in else 0
    print(f"  Total input:              {format_tokens(total_in):>10}")
    print(f"    ├─ cached (hit)         {format_tokens(s['total_cached']):>10}   {format_pct(hit):>8}")
    print(f"    └─ uncached             {format_tokens(s['total_uncached']):>10}   {format_pct(1-hit):>8}")
    print(f"  Output:                   {format_tokens(s['total_output']):>10}")
    print(f"    └─ of which reasoning   {format_tokens(s['total_reasoning']):>10}")
    print()
    print("  Uncached input is mostly new content appended each turn, not cache")
    print("  misses. The shortfall below is the part that should have hit.")
    print()

    # ── shortfall ──
    section("PREFIX SHORTFALL")
    exp = s["expected"]
    short = s["shortfall"]
    print(f"  Expected cacheable:       {format_tokens(exp):>10}")
    print(f"  Failed to hit:            {format_tokens(short):>10}   "
          f"{format_pct(short/exp if exp else 0):>8}")
    print(f"  Material eviction events: {s['n_events']:,} of {s['n_pairs']:,} call pairs")
    print()

    if s["by_cause"]:
        print("  By cause:")
        print(f"    {'':<22} {'Lost':>10} {'Events':>9} {'Share':>10}")
        print(f"    {'─'*22} {'─'*10} {'─'*9} {'─'*10}")
        labels = {
            "idle": f"Idle gap (>{format_gap(args.idle_gap)})",
            "compaction": "Compaction",
            "churn": "Mid-work churn",
            "routing": "Routing (transient)",
        }
        for cause in ("idle", "compaction", "churn", "routing"):
            c = s["by_cause"].get(cause)
            if not c:
                continue
            print(f"    {labels[cause]:<22} {format_tokens(c['shortfall']):>10} "
                  f"{c['events']:>9,} "
                  f"{format_pct(c['shortfall']/short if short else 0):>10}")
        print()

        def share(cause):
            return s["by_cause"].get(cause, {}).get("shortfall", 0) / short if short else 0

        idle_share = share("idle")
        routing_share = share("routing")

        # Rank the levers honestly: only idle and compaction are things the
        # user can actually act on.
        if routing_share >= 0.4:
            print(f"  >>> {format_pct(routing_share)} of your cache loss is transient routing. <<<")
            print("      Those calls hit a backend without your prefix warm and the")
            print("      next call recovered. Not preventable — discount it and look")
            print("      at the idle and compaction rows for anything actionable.")
        elif idle_share >= 0.25:
            print(f"  >>> {format_pct(idle_share)} of your cache loss comes from idle gaps. <<<")
            print("      Resuming sooner, or starting fresh instead of resuming cold,")
            print("      is a real lever here.")
        else:
            print(f"  >>> Only {format_pct(idle_share)} of your cache loss comes from idle gaps. <<<")
            print("      Waiting less between turns is not your lever — most of the")
            print("      loss happens inside active work.")
        print()

    # ── rate limits ──
    if s["limits"]:
        section("RATE LIMIT")
        for name in ("primary", "secondary"):
            lim = s["limits"].get(name)
            if not lim:
                continue
            window = format_gap(lim["window_minutes"] * 60) if lim["window_minutes"] else "?"
            print(f"  {name:<10} {window:>4} window   {lim['used_percent']:>5.1f}% used "
                  f"as of {lim['ts']:%Y-%m-%d %H:%M}")
        print()
        if s["burn"]:
            for window, b in sorted(s["burn"].items(),
                                    key=lambda kv: kv[0] or 0):
                label = format_gap(window * 60) if window else "?"
                print(f"  Burn rate: ~{format_tokens(b['tokens_per_percent'])} input tokens "
                      f"per 1% of the {label} window")
                print(f"             (median of {b['steps']:,} clean 1% steps)")
            print()
            print("  This is gross input including cache hits. Whether cached tokens")
            print("  are discounted by the limiter is not determinable from these")
            print("  logs — treat it as a burn rate, not a price.")
            print()

    # ── cost ──
    section("API-EQUIVALENT COST")
    print(f"  Cost of this usage at API list prices:  {format_cost(s['total_cost'])}")
    if s["total_cost"] > 0:
        type_labels = (("uncached", "├─", "uncached input"),
                       ("cached", "├─", "cached input"),
                       ("output", "└─", "output (incl. reasoning)"))
        for key, branch, label in type_labels:
            v = s["cost_by_type"][key]
            print(f"    {branch} {label:<24} {format_cost(v):>10}   "
                  f"{format_pct(v / s['total_cost']):>8}")
            if key == "uncached" and s["shortfall_cost"] > 0:
                # Shortfall is billed as uncached input, so its cost sits
                # inside that line rather than on top of the total.
                v = s["shortfall_cost"]
                print(f"    │  └─ {'from prefix shortfall':<21} {format_cost(v):>10}   "
                      f"{format_pct(v / s['total_cost']):>8}")
        print()
    print(f"  Same usage with no cache at all:        {format_cost(s['uncached_cost'])}")
    saved = s["uncached_cost"] - s["total_cost"]
    if s["uncached_cost"] > 0:
        print(f"  Caching saved you:                      {format_cost(saved)}"
              f"   ({format_pct(saved/s['uncached_cost'])})")
    print()
    print(f"  Cost of cache that failed to hit:       {format_cost(s['shortfall_cost'])}")
    if s["by_cause"] and s["shortfall_cost"] > 0:
        labels = {"idle": "idle gaps", "compaction": "compaction",
                  "churn": "mid-work churn", "routing": "routing (transient)"}
        actionable = sum(v["cost"] for k, v in s["by_cause"].items()
                         if k in ("idle", "compaction"))
        for cause in ("idle", "compaction", "churn", "routing"):
            cv = s["by_cause"].get(cause)
            if not cv or cv["cost"] <= 0:
                continue
            print(f"    {'└─' if cause == 'routing' else '├─'} "
                  f"{labels[cause]:<22} {format_cost(cv['cost']):>10}")
        print()
        print(f"  Of that, plausibly actionable:          {format_cost(actionable)}"
              f"   ({format_pct(actionable/s['total_cost']) if s['total_cost'] else '0%'}"
              f" of total spend)")
    print()
    print(f"  List prices as of {PRICING_AS_OF}. On a subscription you don't pay")
    print("  these — this is what the same traffic would have cost on the API.")
    if s["unpriced_calls"]:
        print(f"  {s['unpriced_calls']:,} calls excluded (no public price for "
              f"{', '.join(sorted(s['unpriced_models'])[:3])}).")
    print()

    # ── by model ──
    if s["by_model"]:
        section("BY MODEL")
        print(f"  {'Model':<18} {'Calls':>8} {'Input':>9} {'Output':>8} "
              f"{'Hit':>7} {'Lost':>8} {'Cost':>10}")
        print(f"  {'─'*18} {'─'*8} {'─'*9} {'─'*8} {'─'*7} {'─'*8} {'─'*10}")
        for model, m in sorted(s["by_model"].items(), key=lambda kv: -kv[1]["cost"]):
            h = m["cached"] / m["input"] if m["input"] else 0
            unpriced = "" if price_for(model) else "  (unpriced)"
            print(f"  {model[:18]:<18} {m['calls']:>8,} {format_tokens(m['input']):>9} "
                  f"{format_tokens(m['output']):>8} {format_pct(h):>7} "
                  f"{format_tokens(m['shortfall']):>8} "
                  f"{format_cost(m['cost']):>10}{unpriced}")
        print()

        priced = priced_by_cost(s["by_model"])
        if priced:
            print("  Cost by token type:")
            print_cost_table("Model", priced)
            print("  Shortfall is what missed cache added to Uncached; it is")
            print("  already counted in Total, not on top of it.")
            print()

    # ── by project ──
    if args.by_project and s["by_project"]:
        section("BY PROJECT")
        ranked = sorted(s["by_project"].items(), key=lambda kv: -kv[1]["cost"])
        print(f"  {'Project':<24} {'Sess':>5} {'Calls':>8} {'Input':>9} "
              f"{'Hit':>7} {'Lost':>8} {'Cost':>10}")
        print(f"  {'─'*24} {'─'*5} {'─'*8} {'─'*9} {'─'*7} {'─'*8} {'─'*10}")
        shown = ranked[:args.top_projects]
        for _, p in shown:
            h = p["cached"] / p["input"] if p["input"] else 0
            print(f"  {p['name'][:24]:<24} {len(p['sessions']):>5,} {p['calls']:>8,} "
                  f"{format_tokens(p['input']):>9} {format_pct(h):>7} "
                  f"{format_tokens(p['shortfall']):>8} {format_cost(p['cost']):>10}")
        if len(ranked) > len(shown):
            rest = ranked[len(shown):]
            print(f"  {'… ' + str(len(rest)) + ' more':<24} "
                  f"{sum(len(p['sessions']) for _, p in rest):>5,} "
                  f"{sum(p['calls'] for _, p in rest):>8,} "
                  f"{format_tokens(sum(p['input'] for _, p in rest)):>9} "
                  f"{'':>7} {format_tokens(sum(p['shortfall'] for _, p in rest)):>8} "
                  f"{format_cost(sum(p['cost'] for _, p in rest)):>10}")
        print()
        print("  Directories:")
        for _, p in shown[:8]:
            if p["path"]:
                print(f"    {p['name'][:22]:<22} {compact_path(p['path'])}")
        print()

        # Each project's total, then what each model in it cost.
        rows = []
        for _, p in shown:
            rows.append((p["name"], p))
            rows.extend(("  " + model, m) for model, m in priced_by_cost(p["models"]))
        print("  Spend by model:")
        print_cost_table("Project / model", rows)

        # Tokens need no price, so unpriced models show up here too, after
        # the priced ones.
        rows = []
        for _, p in shown:
            rows.append((p["name"], p))
            rows.extend(("  " + model, m) for model, m in sorted(
                p["models"].items(),
                key=lambda kv: (-kv[1]["cost"], -(kv[1]["input"] + kv[1]["output"]))))
        print("  Tokens by model:")
        print_token_table("Project / model", rows)

    # ── by day ──
    if s["by_day"]:
        section("BY DAY")
        print(f"  {'Date':<12} {'Calls':>9} {'Input':>10} {'Hit':>8} "
              f"{'Lost':>10} {'Events':>8}")
        print(f"  {'─'*12} {'─'*9} {'─'*10} {'─'*8} {'─'*10} {'─'*8}")
        for day in sorted(s["by_day"]):
            d = s["by_day"][day]
            h = d["cached"] / d["input"] if d["input"] else 0
            print(f"  {day:<12} {d['calls']:>9,} {format_tokens(d['input']):>10} "
                  f"{format_pct(h):>8} {format_tokens(d['shortfall']):>10} "
                  f"{d['events']:>8,}")
        print()

    # ── worst sessions ──
    worst = sorted(s["by_session"].items(), key=lambda kv: -kv[1]["shortfall"])[:8]
    if worst and worst[0][1]["shortfall"] > 0:
        section("WORST SESSIONS")
        print(f"  {'Session':<14} {'Calls':>8} {'Input':>10} {'Hit':>8} "
              f"{'Lost':>10} {'Events':>8}")
        print(f"  {'─'*14} {'─'*8} {'─'*10} {'─'*8} {'─'*10} {'─'*8}")
        for sid, v in worst:
            if v["shortfall"] <= 0:
                continue
            h = v["cached"] / v["input"] if v["input"] else 0
            print(f"  {sid[:14]:<14} {v['calls']:>8,} {format_tokens(v['input']):>10} "
                  f"{format_pct(h):>8} {format_tokens(v['shortfall']):>10} "
                  f"{v['events']:>8,}")
        print()

    # ── verbose ──
    if args.verbose and s["events"]:
        section("MATERIAL EVICTIONS")
        print(f"  {'Timestamp':<20} {'Session':<12} {'Gap':>7} "
              f"{'Expected':>10} {'Lost':>10} {'Cost':>9} {'Cause':<11}")
        print(f"  {'─'*20} {'─'*12} {'─'*7} {'─'*10} {'─'*10} {'─'*9} {'─'*11}")
        for e in sorted(s["events"], key=lambda x: -x["shortfall"])[:args.top]:
            print(f"  {e['ts']:%Y-%m-%d %H:%M:%S} {e['session_id'][:12]:<12} "
                  f"{format_gap(e['gap']):>7} {format_tokens(e['expected']):>10} "
                  f"{format_tokens(e['shortfall']):>10} "
                  f"{format_cost(e.get('cost', 0)):>9} {e['cause']:<11}")
        if len(s["events"]) > args.top:
            print(f"\n  ... and {len(s['events'])-args.top:,} more "
                  f"(showing top {args.top} by tokens lost)")
        print()

    print("=" * 69)
    print("  Notes:")
    print("  • Codex reports no cache writes, so a miss costs you the")
    print("    cached-input discount — not a rebuild premium.")
    print("  • 'Lost' counts tokens that should have been read from cache and")
    print("    were not. Its cost is the gap between the full input rate and")
    print("    the cached rate, at API list prices.")
    print("  • Dollar figures are notional: they price your traffic as if it")
    print("    had run on the API. They are not your subscription bill and not")
    print("    a rate-limit figure.")
    print("  • Run with --decay to see how hit rate falls off with idle time.")
    print("=" * 69)
    print()


# ─────────────────────────── Main ───────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Audit Codex session rollouts for prompt-cache loss.")
    window = p.add_mutually_exclusive_group()
    window.add_argument("--start", type=str,
                        help="Window start: a lookback from now (30m, 18h, 7d, 2w) "
                             "or an ISO date/datetime, UTC (default: 7d)")
    window.add_argument("--days", type=int, default=7,
                        help="Analyze last N days; same as --start Nd (default: 7)")
    window.add_argument("--since", type=str,
                        help="Analyze since date (YYYY-MM-DD); same as --start DATE")
    window.add_argument("--since-reset", nargs="?", const="primary",
                        choices=("primary", "secondary"),
                        help="Infer start from the latest limit reset minus its window "
                             "(default: primary; optionally secondary)")
    window.add_argument("--previous-reset", nargs="?", const="primary",
                        choices=("primary", "secondary"),
                        help="Report the limit window before the current one, from its "
                             "start to its reset (default: primary; optionally secondary)")
    p.add_argument("--end", type=str,
                   help="Window end, in the same forms as --start (default: now)")
    codex_home = os.environ.get("CODEX_HOME") or "~/.codex"
    default_sessions_dir = os.path.join(codex_home, "sessions")
    p.add_argument("--sessions-dir", default=default_sessions_dir,
                   help="Rollout directory (default: $CODEX_HOME/sessions, or "
                        "~/.codex/sessions when CODEX_HOME is unset)")
    p.add_argument("--decay", action="store_true",
                   help="Show cache decay by idle gap instead of the main report")
    p.add_argument("--by-project", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Per-repo / per-directory breakdown (default: on)")
    p.add_argument("--top-projects", type=int, default=15,
                   help="Projects to list in the by-project breakdown (default: 15)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="List the largest material eviction events")
    p.add_argument("--top", type=int, default=25,
                   help="How many events to list with --verbose (default: 25)")
    p.add_argument("--json", action="store_true", help="Machine-readable output")
    p.add_argument("--tolerance", type=float, default=0.02,
                   help="Shortfall below this fraction of the expected prefix is "
                        "treated as block-quantization noise (default: 0.02)")
    p.add_argument("--tolerance-abs", type=int, default=2048,
                   help="Absolute floor for the same test, in tokens (default: 2048)")
    p.add_argument("--lookback", type=int, default=8,
                   help="How many recent calls to search for a plausible parent. "
                        "Codex interleaves concurrent sub-agents into one rollout "
                        "with no agent id, so the previous call is often a "
                        "sibling's (default: 8; 1 disables)")
    p.add_argument("--idle-gap", type=int, default=600,
                   help="Gap in seconds above which an eviction is blamed on idling "
                        "(default: 600)")
    p.add_argument("--compaction-ratio", type=float, default=0.75,
                   help="Input shrinking below this fraction of the previous call is "
                        "read as a compaction (default: 0.75)")
    args = p.parse_args()

    # One clock for both ends, so --start 7d --end 1d is exactly six days.
    now = datetime.now(timezone.utc)
    bounds = {}
    for flag, raw in (("--start", args.start or args.since or f"{args.days}d"),
                      ("--end", args.end)):
        if raw is None:
            continue
        try:
            bounds[flag] = parse_time_arg(raw, now)
        except ValueError:
            print(f"Could not parse {flag} {raw!r}; expected a lookback like "
                  f"30m, 18h, 7d, 2w or an ISO date (YYYY-MM-DD[THH:MM])",
                  file=sys.stderr)
            sys.exit(2)
    since = bounds["--start"]
    until = bounds.get("--end", now)

    files = find_rollout_files(args.sessions_dir)
    if not files:
        print(f"No rollout files under {args.sessions_dir}", file=sys.stderr)
        sys.exit(1)

    reset_window = None
    if args.since_reset:
        try:
            since, reset_window = reset_window_start(files, until, args.since_reset)
        except ValueError as e:
            p.error(str(e))
        if not args.json:
            print(f"  Inferred {args.since_reset} window start: {since.isoformat()}")
            print(f"  Next reset: {reset_window['resets_at']} (local logs only)")
    elif args.previous_reset:
        try:
            since, until, reset_window = previous_reset_window(
                files, until, args.previous_reset)
        except ValueError as e:
            p.error(str(e))
        if not args.json:
            print(f"  Inferred previous {args.previous_reset} window: "
                  f"{since.isoformat()} -> {until.isoformat()} (local logs only)")
            if reset_window["early_reset"]:
                print(f"  Reset early: scheduled for {reset_window['resets_at']}")
    if since >= until:
        p.error(f"Window is empty: start {since:%Y-%m-%d %H:%M} is not before "
                f"end {until:%Y-%m-%d %H:%M} (UTC)")

    sessions = collect_sessions(files, since, until)
    n_calls = sum(len(v) for v in sessions.values())
    if not n_calls:
        print(f"No Codex activity between {since:%Y-%m-%d %H:%M} and "
              f"{until:%Y-%m-%d %H:%M} (UTC).", file=sys.stderr)
        sys.exit(0)

    # ── accumulate ──
    totals = dict(input=0, cached=0, uncached=0, output=0, reasoning=0)
    expected_total = shortfall_total = 0
    n_pairs = n_events = 0
    models = defaultdict(int)
    efforts = defaultdict(int)
    plan_type = None
    buckets = [dict(calls=0, events=0, expected=0, shortfall=0)
               for _ in GAP_EDGES[:-1]]
    by_cause = defaultdict(lambda: dict(shortfall=0, events=0, cost=0.0))
    by_model = defaultdict(lambda: cost_row(calls=0, input=0, cached=0, output=0,
                                            shortfall=0))
    by_day = defaultdict(lambda: dict(calls=0, input=0, cached=0,
                                      shortfall=0, events=0, cost=0.0))
    by_session = defaultdict(lambda: dict(calls=0, input=0, cached=0,
                                          shortfall=0, events=0, cost=0.0))
    by_project = defaultdict(lambda: cost_row(calls=0, input=0, cached=0, output=0,
                                              shortfall=0, name=None, path=None,
                                              sessions=set(),
                                              models=defaultdict(lambda: cost_row(
                                                  calls=0, input=0, cached=0,
                                                  output=0, shortfall=0))))
    total_cost = uncached_cost = 0.0
    cost_by_type = dict(uncached=0.0, cached=0.0, output=0.0)
    shortfall_cost = 0.0
    unpriced_models = set()
    unpriced_calls = 0
    events = []
    limits = {}
    burn_steps = defaultdict(list)   # window_minutes -> [input tokens per 1% step]

    for sid, calls in sessions.items():
        for i, c in enumerate(calls):
            day = c["ts"].strftime("%Y-%m-%d")
            for k in totals:
                totals[k] += c[k]
            models[c["model"]] += 1
            efforts[c["effort"]] += 1
            if c["plan_type"]:
                plan_type = c["plan_type"]

            # API-equivalent cost. Unpriced models are tracked separately so
            # they never silently vanish from a dollar total.
            price = price_for(c["model"])
            parts = call_cost(c, price) if price else dict.fromkeys(cost_by_type, 0.0)
            cost = sum(parts.values())
            if price:
                total_cost += cost
                for k, v in parts.items():
                    cost_by_type[k] += v
                # What the same call would have cost with no cache at all —
                # the gap between the two is what caching is worth to you.
                uncached_cost += ((c["input"] * price["input"]
                                   + c["output"] * price["output"]) / 1_000_000)
            else:
                unpriced_models.add(c["model"])
                unpriced_calls += 1

            bm = by_model[c["model"]]
            bm["calls"] += 1; bm["input"] += c["input"]; bm["cached"] += c["cached"]
            bm["output"] += c["output"]
            add_cost(bm, parts)
            bd = by_day[day]
            bd["calls"] += 1; bd["input"] += c["input"]; bd["cached"] += c["cached"]
            bd["cost"] += cost
            bs = by_session[sid]
            bs["calls"] += 1; bs["input"] += c["input"]; bs["cached"] += c["cached"]
            bs["cost"] += cost

            key, name, path = project_of(c["cwd"], c["repo"])
            bp = by_project[key]
            bp["name"] = name
            bp["calls"] += 1; bp["input"] += c["input"]; bp["cached"] += c["cached"]
            bp["output"] += c["output"]
            add_cost(bp, parts)
            bpm = bp["models"][c["model"]]
            bpm["calls"] += 1; bpm["input"] += c["input"]; bpm["cached"] += c["cached"]
            bpm["output"] += c["output"]
            add_cost(bpm, parts)
            bp["sessions"].add(sid)
            if path:
                bp["path"] = path

            # Keep the most RECENT snapshot per limit, not whichever session
            # happened to be iterated last.
            for name in ("primary", "secondary"):
                lim = (c["limits"] or {}).get(name)
                if lim and lim.get("used_percent") is not None:
                    known = limits.get(name)
                    if known is None or c["ts"] > known["ts"]:
                        limits[name] = {
                            "ts": c["ts"],
                            "used_percent": lim["used_percent"],
                            "window_minutes": lim.get("window_minutes"),
                            "resets_at": lim.get("resets_at"),
                        }

            if i == 0:
                continue
            history = calls[max(0, i - args.lookback):i]
            nxt = calls[i + 1] if i + 1 < len(calls) else None
            res = analyze_call(history, c, nxt, args.tolerance, args.tolerance_abs,
                               args.idle_gap, args.compaction_ratio)
            if res is None:
                continue
            n_pairs += 1
            expected_total += res["expected"]
            shortfall_total += res["shortfall"]

            b = buckets[bucket_index(res["gap"])]
            b["calls"] += 1
            b["expected"] += res["expected"]
            b["shortfall"] += res["shortfall"]

            if res["material"]:
                # A missed token is billed at the full input rate instead of
                # the cached rate; the difference is what the miss actually
                # cost. There is no write premium on top — Codex doesn't
                # write through explicit cache breakpoints.
                miss_cost = (res["shortfall"] * (price["input"] - price["cached"])
                             / 1_000_000) if price else 0.0
                res["cost"] = miss_cost
                shortfall_cost += miss_cost

                n_events += 1
                b["events"] += 1
                by_cause[res["cause"]]["shortfall"] += res["shortfall"]
                by_cause[res["cause"]]["events"] += 1
                by_cause[res["cause"]]["cost"] += miss_cost
                bm["shortfall"] += res["shortfall"]; bm["cost_shortfall"] += miss_cost
                bd["shortfall"] += res["shortfall"]; bd["events"] += 1
                bs["shortfall"] += res["shortfall"]; bs["events"] += 1
                bp["shortfall"] += res["shortfall"]; bp["cost_shortfall"] += miss_cost
                bpm["shortfall"] += res["shortfall"]; bpm["cost_shortfall"] += miss_cost
                events.append(res)

        # burn rate: tokens consumed per whole 1% step of the primary window
        acc = 0
        prev_pct = None
        for c in calls:
            lim = (c["limits"] or {}).get("primary") or {}
            pct = lim.get("used_percent")
            acc += c["input"]
            if pct is None:
                continue
            if prev_pct is not None:
                delta = pct - prev_pct
                if delta == 1.0:
                    burn_steps[lim.get("window_minutes")].append(acc)
                    acc = 0
                elif delta != 0:
                    acc = 0
            prev_pct = pct

    if args.decay:
        report_decay(buckets, args.json)
        return

    # Median input tokens per 1% step, computed separately for each window
    # (the 5h and 7d limits burn at very different rates).
    burn = {}
    for window, vals in burn_steps.items():
        if not vals:
            continue
        ordered = sorted(vals)
        burn[window] = {
            "tokens_per_percent": ordered[len(ordered) // 2],
            "steps": len(ordered),
        }

    stats = {
        "since": since,
        "until": until,
        "n_sessions": len(sessions),
        "n_calls": n_calls,
        "n_pairs": n_pairs,
        "n_events": n_events,
        "plan_type": plan_type,
        "models": sorted(models.items(), key=lambda kv: -kv[1]),
        "efforts": sorted(efforts.items(), key=lambda kv: -kv[1]),
        "total_input": totals["input"],
        "total_cached": totals["cached"],
        "total_uncached": totals["uncached"],
        "total_output": totals["output"],
        "total_reasoning": totals["reasoning"],
        "expected": expected_total,
        "shortfall": shortfall_total,
        "by_cause": dict(by_cause),
        "by_model": dict(by_model),
        "by_day": dict(by_day),
        "by_session": dict(by_session),
        "by_project": dict(by_project),
        "events": events,
        "limits": limits,
        "burn": burn,
        "total_cost": total_cost,
        "cost_by_type": cost_by_type,
        "uncached_cost": uncached_cost,
        "shortfall_cost": shortfall_cost,
        "unpriced_models": unpriced_models,
        "unpriced_calls": unpriced_calls,
    }

    if args.json:
        hit = totals["cached"] / totals["input"] if totals["input"] else 0
        out = {
            "period": {"since": since.isoformat(),
                       "until": stats["until"].isoformat()},
            "summary": {
                "sessions": len(sessions),
                "api_calls": n_calls,
                "plan_type": plan_type,
                "total_input_tokens": totals["input"],
                "cached_input_tokens": totals["cached"],
                "uncached_input_tokens": totals["uncached"],
                "output_tokens": totals["output"],
                "reasoning_tokens": totals["reasoning"],
                "cache_hit_rate": round(hit, 4),
                "expected_cacheable_tokens": expected_total,
                "shortfall_tokens": shortfall_total,
                "shortfall_rate": round(
                    shortfall_total / expected_total, 4) if expected_total else 0,
                "material_events": n_events,
                "call_pairs": n_pairs,
            },
            "cost": {
                "currency": "USD",
                "basis": "OpenAI public API list prices",
                "prices_as_of": PRICING_AS_OF,
                "note": "Notional. On a ChatGPT/Codex subscription these are "
                        "not amounts billed to you.",
                "actual": round(total_cost, 4),
                "actual_by_type": {k: round(v, 4) for k, v in cost_by_type.items()},
                "if_nothing_cached": round(uncached_cost, 4),
                "saved_by_cache": round(uncached_cost - total_cost, 4),
                "cache_shortfall": round(shortfall_cost, 4),
                "unpriced_calls": unpriced_calls,
                "unpriced_models": sorted(unpriced_models),
            },
            "by_cause": {k: {**v, "cost": round(v["cost"], 4)}
                         for k, v in by_cause.items()},
            "by_model": {k: round_costs(v) for k, v in by_model.items()},
            "by_day": {k: {**v, "cost": round(v["cost"], 4)}
                       for k, v in sorted(by_day.items())},
            # Keyed by repo path (or URL / directory); "name" is the display name.
            "by_project": {
                k: {**round_costs({kk: vv for kk, vv in v.items()
                                   if kk not in ("sessions", "models")}),
                    "sessions": len(v["sessions"]),
                    "models": {m: round_costs(mv) for m, mv in v["models"].items()}}
                for k, v in by_project.items()
            },
            "rate_limits": {k: {**v, "ts": v["ts"].isoformat()}
                            for k, v in limits.items()},
            "burn_by_window_minutes": burn,
            "worst_sessions": [
                {"session": k, **v} for k, v in
                sorted(by_session.items(), key=lambda kv: -kv[1]["shortfall"])[:10]
            ],
        }
        if args.verbose:
            out["events"] = [
                {**e, "ts": e["ts"].isoformat()}
                for e in sorted(events, key=lambda x: -x["shortfall"])[:args.top]
            ]
        if reset_window is not None:
            out["period"]["reset_window"] = reset_window
        print(json.dumps(out, indent=2, default=str))
        return

    report_main(stats, args)


if __name__ == "__main__":
    main()
