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

    def test_conflicting_start_flags(self):
        for flag, value in [("--start", "1d"), ("--days", "1"), ("--since", "2026-09-01")]:
            result = self.cli("--since-reset", flag, value)
            self.assertEqual(result.returncode, 2)
            self.assertIn("not allowed with argument", result.stderr)


if __name__ == "__main__":
    unittest.main()
