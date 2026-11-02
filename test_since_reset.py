import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone


SCRIPT = Path(__file__).with_name("cache-money-audit.py")
spec = importlib.util.spec_from_file_location("audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class SinceResetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.end = datetime(2026, 9, 30, 12, tzinfo=timezone.utc)

    def event(self, ts, minutes=60, reset=None, name="primary", usage=None):
        payload = {"type": "token_count", "rate_limits": {
            name: {"window_minutes": minutes, "used_percent": 10,
                   "resets_at": (reset or self.end + timedelta(minutes=30)).timestamp()}}}
        if usage is not None:
            payload["info"] = {"last_token_usage": {
                "input_tokens": usage, "cached_input_tokens": 0, "output_tokens": 1}}
        return {"type": "event_msg", "timestamp": ts.isoformat(), "payload": payload}

    def write(self, name, events):
        path = Path(self.temp.name) / f"rollout-{name}.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in events))
        return str(path)

    def cli(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), "--sessions-dir",
                               self.temp.name, *args], text=True, capture_output=True)

    def test_latest_snapshot_across_files_and_limit_only_events(self):
        old = self.write("old", [self.event(self.end - timedelta(minutes=10), minutes=120)])
        new = self.write("new", [self.event(self.end)])
        start, source = audit.reset_window_start([new, old], self.end, "primary")
        self.assertEqual(start, self.end - timedelta(minutes=30))
        self.assertEqual(source["snapshot_at"], self.end.isoformat())

    def test_historical_end_and_secondary(self):
        path = self.write("history", [self.event(self.end, name="secondary"),
            self.event(self.end + timedelta(minutes=5), minutes=120, name="secondary")])
        start, _ = audit.reset_window_start([path], self.end, "secondary")
        self.assertEqual(start, self.end - timedelta(minutes=30))
        with self.assertRaisesRegex(ValueError, "No primary"):
            audit.reset_window_start([path], self.end, "primary")

    def test_expired_and_invalid_snapshots(self):
        for minutes, reset, message in [
            (60, self.end, "expired"), (0, self.end + timedelta(minutes=30), "invalid"),
            (float("nan"), self.end + timedelta(minutes=30), "invalid"),
        ]:
            with self.subTest(minutes=minutes):
                path = self.write("bad", [self.event(self.end - timedelta(minutes=5),
                                                     minutes=minutes, reset=reset)])
                with self.assertRaisesRegex(ValueError, message):
                    audit.reset_window_start([path], self.end, "primary")

    def test_cli_filters_usage_and_preserves_json(self):
        self.write("usage", [
            self.event(self.end - timedelta(minutes=40), usage=900),
            self.event(self.end - timedelta(minutes=30), usage=100),
            self.event(self.end - timedelta(minutes=10), usage=200),
            self.event(self.end),
            self.event(self.end + timedelta(minutes=5), usage=800),
        ])
        result = self.cli("--since-reset", "--end", self.end.isoformat(), "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["summary"]["total_input_tokens"], 300)
        self.assertEqual(data["summary"]["api_calls"], 2)
        self.assertTrue(data["period"]["reset_window"]["inferred"])
        text = self.cli("--since-reset", "--end", self.end.isoformat())
        self.assertEqual(text.returncode, 0, text.stderr)
        self.assertIn("Inferred primary window start", text.stdout)


    def test_previous_window_after_natural_reset(self):
        # Previous window: 10:00-11:00; current: 11:30-12:30 (started on activity).
        prev_reset = self.end - timedelta(hours=1)
        path = self.write("prev", [
            self.event(self.end - timedelta(hours=2, minutes=30), reset=self.end - timedelta(hours=2)),
            self.event(self.end - timedelta(hours=1, minutes=50), reset=prev_reset),
            self.event(self.end - timedelta(hours=1, minutes=10),
                       reset=prev_reset + timedelta(minutes=1)),  # jitter, same window
            self.event(self.end),
        ])
        start, end, source = audit.previous_reset_window([path], self.end, "primary")
        self.assertEqual(start, prev_reset - timedelta(minutes=59))
        self.assertEqual(end, prev_reset + timedelta(minutes=1))
        self.assertFalse(source["early_reset"])

    def test_previous_window_cut_short_by_early_reset(self):
        # Previous window 10:45-11:45, but the current one began 11:30.
        prev_reset = self.end - timedelta(minutes=15)
        path = self.write("early", [
            self.event(self.end - timedelta(minutes=45), reset=prev_reset),
            self.event(self.end),  # resets 12:30, so started 11:30
        ])
        start, end, source = audit.previous_reset_window([path], self.end, "primary")
        self.assertEqual(start, prev_reset - timedelta(hours=1))
        self.assertEqual(end, self.end - timedelta(minutes=30))
        self.assertTrue(source["early_reset"])

    def test_previous_window_missing(self):
        path = self.write("only", [self.event(self.end)])
        with self.assertRaisesRegex(ValueError, "before the current window"):
            audit.previous_reset_window([path], self.end, "primary")

    def test_cli_previous_reset(self):
        prev_reset = self.end - timedelta(hours=1)
        self.write("usage", [
            self.event(self.end - timedelta(hours=2, minutes=10), reset=prev_reset, usage=900),
            self.event(self.end - timedelta(hours=1, minutes=50), reset=prev_reset, usage=100),
            self.event(self.end - timedelta(hours=1, minutes=20), reset=prev_reset, usage=200),
            self.event(self.end - timedelta(minutes=20), usage=800),
        ])
        result = self.cli("--previous-reset", "--end", self.end.isoformat(), "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["summary"]["total_input_tokens"], 300)
        self.assertEqual(data["period"]["reset_window"]["ended_at"], prev_reset.isoformat())
        text = self.cli("--previous-reset", "--end", self.end.isoformat())
        self.assertEqual(text.returncode, 0, text.stderr)
        self.assertIn("Inferred previous primary window", text.stdout)
        clash = self.cli("--previous-reset", "--since-reset")
        self.assertEqual(clash.returncode, 2)

    def test_conflicting_start_flags(self):
        for flag, value in [("--start", "1d"), ("--days", "1"), ("--since", "2026-09-01")]:
            result = self.cli("--since-reset", flag, value)
            self.assertEqual(result.returncode, 2)
            self.assertIn("not allowed with argument", result.stderr)


if __name__ == "__main__":
    unittest.main()
