# codex-cache-money

Find out where Codex is throwing away your prompt cache — and whether you
can do anything about it.

```bash
python3 cache-money-audit.py            # last 7 days
```

No dependencies, no install, no config. It reads the session rollouts Codex
already writes to `$CODEX_HOME/sessions/`, falling back to
`~/.codex/sessions/` when `CODEX_HOME` is unset. Use `--sessions-dir` to point
it somewhere else.

## Why this is not just the Claude version with different field names

Codex reports caching differently enough that the usual "cold turn" analysis
gives you nonsense. Four things drive the whole design:

**There is no cache-write signal.** OpenAI bills cache writes only for
tokens written through explicit cache breakpoints, and Codex doesn't use
them — `cache_write_input_tokens` is `0` on every call. So "cold turn = large
cache write" has no analogue, and neither does the idea of a rebuild
*premium*. A miss just forfeits the cached-input discount.

**`input_tokens` includes `cached_input_tokens`.** Anthropic's excludes both
cache fields. Add up Codex's the same way and every total is inflated. Real
uncached input is `input - cached`.

**Uncached input is mostly not a cache miss.** On a perfectly warm turn it's
just the content appended since the last call — your message, tool output.
That's irreducible, and it runs 2–5% of input on every healthy turn. Treating
it as waste would report a permanent 3% loss that no behavior can fix.

**There is no TTL, so there is no cliff.** OpenAI's prefix cache is evicted
under memory pressure, not expired on a timer. Hit rates decay smoothly and
never reach zero — even a day later a third of the prefix still hits. There
is no deadline to beat and nothing to configure.

## How a real cache loss is detected

Structurally, not by a token ratio. A genuine loss is when the cached prefix
comes back **smaller than the prompt you already sent**:

```
expected  = floor128(prev.input + prev.output - prev.reasoning)
expected  = min(expected, floor128(cur.input))
shortfall = max(0, expected - cur.cached)
```

Reasoning tokens are subtracted because they aren't replayed on the next
call. The 128-token floor matches the cache's block quantization. Shortfalls
under `--tolerance` are treated as quantization noise.

Two corrections matter more than the formula:

**Concurrent sub-agents share one rollout.** Codex writes every agent's
`token_count` events into the same file with no agent or turn id attached, so
the call immediately before this one is often a *sibling's*, with a completely
different prefix. Compared naively it looks like a total eviction at a
two-second gap. The tool instead searches back over recent calls
(`--lookback`, default 8) for a plausible parent — one whose prompt fits
inside ours and whose prefix our cache covers. If one exists, nothing was
lost. This deliberately under-reports: a loss indistinguishable from
interleaving isn't counted.

**Most "losses" are transient and self-healing.** A real eviction persists —
the next call has to rebuild too. Very often the cache drops to near zero for
exactly one call and is fully back on the next, which means the prefix was
never gone; that request just routed to a backend that didn't have it warm.
You paid for it, but nothing you did caused it. Those are reported separately
so they don't drown out anything actionable.

## What it tells you

Every loss is attributed to one of four causes, only two of which you control:

| Cause | Meaning | Can you act on it? |
|-------|---------|--------------------|
| **Idle gap** | Came back after a long enough pause that the prefix was evicted | Yes — resume sooner, or start fresh instead of resuming cold |
| **Compaction** | Context was compacted, so the old prefix no longer applies | Somewhat — compact less often, or at better moments |
| **Mid-work churn** | Prefix went away during active work and stayed away | Sometimes — usually a prompt prefix changing underneath you |
| **Routing** | One call missed, the next recovered | No. Transient backend routing |

Expect routing to dominate. On the ~400k-call sample this was built against it
was **~85%** of all cache loss, with idle gaps at ~3%. That's a real finding,
not a failure of the tool — and it's the sort of thing worth knowing before
you spend effort optimizing your habits.

## Usage

```bash
python3 cache-money-audit.py                  # last 7 days
python3 cache-money-audit.py --start 18h      # or 30m, 30d, 2w
python3 cache-money-audit.py --start 7d --end 1d
python3 cache-money-audit.py --start 2026-08-01 --end 2026-08-15
python3 cache-money-audit.py --decay          # hit rate vs. idle gap
python3 cache-money-audit.py --no-by-project  # skip the per-repo breakdown
python3 cache-money-audit.py --verbose        # largest individual losses
python3 cache-money-audit.py --json           # machine-readable
```

`--start` and `--end` each take a lookback from now (`30m`, `18h`, `7d`,
`2w`) or an ISO date or datetime, read as UTC. `--end` defaults to now.
`--days N` and `--since DATE` still work, as shorthands for `--start`.

### By project

On by default; `--no-by-project` hides it. Groups sessions by the git
repository their working directory sits in, ranked by cost. It walks up from
the directory to the nearest `.git`, so subdirectories, linked worktrees, and
agent worktrees nested inside the repo (`repo/.fadeno/local/worktrees/x`) all
count toward the repo itself. That still works after the worktree has been
deleted, which most have been by the time you look. A directory with no repo
on disk falls back to the recorded repository URL, then to the directory
itself.

```
  Project                   Sess    Calls     Input     Hit     Lost       Cost
  ──────────────────────── ───── ──────── ───────── ─────── ──────── ──────────
  polymarket-quoter           11   42,014      5.5B   96.9%      86M     $2,675
  fadeno                      16    6,104      730M   97.4%       6M    $432.97
  aethon                       2    1,647      228M   96.2%       5M    $169.49
  consciousness-controls       2    9,780      1.2B   97.9%       9M     $91.26
```

Ranking by cost rather than call count matters more than it looks:
`consciousness-controls` ran more than twice the calls of `fadeno` but cost a
fifth as much, because it was on Luna at $0.20/MTok instead of Sol at $4.00.
The resolved directory for each project is printed underneath the table.

Under that, each project's spend is split by model and by token type:

```
  Spend by model:
  Project / model     Uncached    Cached Shortfall    Output     Total
  ────────────────── ───────── ───────── ───────── ───────── ─────────
  basanos              $196.01    $1,058    $42.74   $129.94    $1,384
    gpt-6-astra        $129.66   $825.92    $40.87    $87.39    $1,043
    gpt-5.6-sol         $47.63   $153.10    $0.875    $26.85   $227.58
    gpt-5.6-luna        $18.72    $79.17    $0.989    $15.70   $113.60
  igigal                $70.55   $328.54    $29.60    $44.92   $444.01
    gpt-6-astra         $63.49   $310.93    $29.12    $39.76   $414.18
    gpt-5.6-luna         $2.01    $11.28    $0.187     $2.18    $15.48
    gpt-5.6-sol          $5.05     $6.32    $0.294     $2.98    $14.35
```

Then the same split in tokens rather than dollars. Lost is the shortfall
in tokens; Total is all input plus output. Models with no public price
are left out of the spend table but still listed here:

```
  Tokens by model:
  Project / model     Uncached    Cached      Lost    Output     Total
  ────────────────── ───────── ───────── ───────── ───────── ─────────
  basanos                 118M     5.17B     10.3M     16.2M     5.30B
    gpt-6-astra          13.0M      826M     4.54M     1.75M      841M
    gpt-5.6-sol          11.9M      383M      243k     1.34M      396M
    gpt-5.6-luna         93.6M     3.96B     5.50M     13.1M     4.07B
  igigal                 17.6M      891M     4.35M     2.76M      911M
    gpt-6-astra          6.35M      311M     3.24M      795k      318M
    gpt-5.6-luna         10.0M      564M     1.04M     1.82M      576M
    gpt-5.6-sol          1.26M     15.8M     81.5k      149k     17.2M
```

In both tables the project rows are the aggregate, and the indented rows
under each one add up to it.

Tuning, if the defaults misjudge your setup:

```bash
--lookback 8            # calls to search for a parent; 1 disables interleave handling
--tolerance 0.02        # shortfall below this share of the prefix is quantization noise
--tolerance-abs 2048    # absolute floor for the same test
--idle-gap 600          # seconds above which a loss is blamed on idling
--compaction-ratio 0.75 # input shrinking below this share reads as a compaction
--top-projects 15       # rows in the by-project tables
--top 25                # rows in the --verbose table
--sessions-dir PATH     # default $CODEX_HOME/sessions, or ~/.codex/sessions
                        # when CODEX_HOME is unset
```

### `--decay`

Buckets call pairs by the gap since the previous call and reports how much of
the expected prefix failed to hit in each. This replaces the TTL inference the
Claude version does — there's no cliff to find, so the useful output is the
shape of the curve:

```
  Gap               Calls    Events    Expected       Lost     Miss
  ──────────── ────────── ───────── ─────────── ────────── ────────
  0s–60s          367,329     9,850       51.0B       507M   0.994%
  5m–10m              663       115         85M         7M    8.28%
  20m–30m             197        56         26M         5M    21.0%
  30m–1h              267       121         36M        14M    38.5%
  1h–2h               147        99         19M        11M    58.7%
  6h–1d               134        95         18M        12M    64.4%
```

Degradation is already real by 20 minutes and roughly plateaus past two
hours. It never reaches 100%.

## Rate limits

Codex puts a live rate-limit snapshot on *every* API call — `used_percent`,
the window, `resets_at`, and your plan — which Claude Code doesn't expose at
all. The report shows the most recent snapshot per window plus an observed
burn rate in tokens per 1% of the window.

Treat the burn rate as a rate, not a price. Whether cached tokens are
discounted *by the limiter* isn't determinable from these logs:
`used_percent` is quantized to whole integers, `resets_at` rolls
continuously, and hit rates barely vary between sessions. Establishing that
would take a deliberate cold-vs-warm experiment, and until then no number
here should be read as "cache hits extend your limit by X".

## Dollars

Everything is also priced at OpenAI's public API list rates, so you can see
what your traffic is worth even though a subscription doesn't bill you for it:

```
  Cost of this usage at API list prices:  $2,159
    ├─ uncached input              $313.78      14.5%
    │  └─ from prefix shortfall     $83.70      3.88%
    ├─ cached input                 $1,618      74.9%
    └─ output (incl. reasoning)    $227.44      10.5%

  Same usage with no cache at all:        $16,719
  Caching saved you:                      $14,560   (87.1%)

  Cost of cache that failed to hit:       $83.70
    ├─ idle gaps                   $4.20
    ├─ compaction                 $0.523
    ├─ mid-work churn             $0.042
    └─ routing (transient)        $78.94

  Of that, plausibly actionable:          $4.72   (0.219% of total spend)
```

The same split appears per model under `BY MODEL`:

```
  Cost by token type:
  Model               Uncached    Cached Shortfall    Output     Total
  ────────────────── ───────── ───────── ───────── ───────── ─────────
  gpt-6-astra          $233.59    $1,310    $81.11   $173.78    $1,718
  gpt-5.6-sol           $60.43   $222.18     $1.48    $36.58   $319.19
  gpt-5.6-luna          $19.68    $85.22     $1.11    $17.03   $121.93
```

Shortfall is the extra paid because prefix that should have hit cache was
billed as uncached input instead. It's part of the uncached figure and the
total, not an extra charge on top. At 97–98% hit rates, cached input is
usually the largest part of the bill even at a tenth of the price: there's
simply that much more of it.

A miss costs the difference between the full input rate and the cached rate
— there's no write premium on top. OpenAI now lists a cache-write rate
(1.25× input) for the 5.6 and GPT-6 families, but it applies only to explicit
cache breakpoints, which Codex doesn't use. Cached input is 10% of input
across the GPT-5.x and GPT-6 families, so a miss is 10× the token's cached
price.

Prices live in the `PRICING` table at the top of the script, list rates as of
2026-09-23 from [OpenAI's pricing page](https://developers.openai.com/api/docs/pricing).
They're standard-tier, short-context rates. Long context (>272K input)
costs more, but Codex caps its window at 258,400 tokens. GPT-5.6 Sol's $4.00
input is a promotional rate guaranteed only through November 21, 2026. Update the
table when prices move. Models with no published price (Codex's
internal ones, e.g. `codex-auto-review`) are counted in token terms and
excluded from dollar totals, with the excluded call count printed rather than
silently folded in at some default rate.

**These are notional.** On a ChatGPT/Codex subscription you don't pay API
rates — this tells you what the same traffic would have cost, which is a
reasonable proxy for value but is not your bill and is not the same thing as
rate-limit consumption.

## What this deliberately doesn't do

- **No statusline.** Codex's status line is a fixed menu of toggleable items
  with no custom-command slot, and cache metrics aren't among them. There's
  nowhere to render this live without a separate sidecar process, which is a
  different tool than this one.
- **No claim that cache hits extend your rate limit.** See below.

## Requirements

Python 3.8+. Nothing else.

## Related

Sibling project for Claude Code: `claude-cache-money`. Shared report grammar,
almost no shared mechanism — the two providers report caching too differently
for one tool to cover both honestly.
