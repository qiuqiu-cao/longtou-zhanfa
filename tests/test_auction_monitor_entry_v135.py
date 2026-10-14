import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from auction_monitor_entry_v135 import assess, reachable_points


class AuctionMonitorEntryV135Test(unittest.TestCase):
    def _inputs(self, root: Path, signal_date: str = "2026-09-16"):
        candidates = root / "candidates.csv"
        review = root / "review.json"
        market = root / "market.csv"
        universe = root / "universe.csv"
        pd.DataFrame([
            {"signal_date": signal_date, "code": "600001", "name": "甲", "boards": 1,
             "board_group": "FIRST_BOARD", "pool_assignment": "CLOSING_LIMIT_UP",
             "is_mainboard": True},
            {"signal_date": signal_date, "code": "000002", "name": "乙", "boards": 2,
             "board_group": "CONSECUTIVE_BOARD", "pool_assignment": "CLOSING_LIMIT_UP",
             "is_mainboard": True},
        ]).to_csv(candidates, index=False)
        review.write_text(json.dumps({
            "signal_date": signal_date,
            "roles": {"emotion_anchors": [{"code": "000002", "name": "乙", "boards": 2}]},
        }, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame({"date": [signal_date]}).to_csv(market, index=False)
        pd.DataFrame({"code": ["600001", "000002"], "name": ["甲", "乙"]}).to_csv(
            universe, index=False)
        return candidates, review, market, universe

    def test_full_sequence_ready_before_0920(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = self._inputs(Path(tmp))
            result = assess(inputs[0], inputs[1], "2026-09-17", inputs[2], inputs[3],
                            "2026-09-17T09:18:00+08:00")
            self.assertEqual(result["status"], "READY_FULL_SEQUENCE_PUBLIC_ONLY")
            self.assertEqual(result["candidate_count"], 2)
            self.assertEqual(result["first_board_count"], 1)
            self.assertEqual(result["relay_count"], 1)

    def test_late_start_is_degraded_but_does_not_backfill(self):
        reachable, missed = reachable_points("2026-09-17", "2026-09-17T09:26:00+08:00")
        self.assertEqual(missed, ["09:20", "09:24:50", "09:25"])
        self.assertEqual(reachable, ["09:30", "09:35", "10:00"])

    def test_stale_inputs_block_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = self._inputs(Path(tmp), "2026-09-15")
            pd.DataFrame({"date": ["2026-09-15", "2026-09-16"]}).to_csv(inputs[2], index=False)
            result = assess(inputs[0], inputs[1], "2026-09-17", inputs[2], inputs[3],
                            "2026-09-17T09:18:00+08:00")
            self.assertEqual(result["status"], "BLOCKED")
            self.assertTrue(any("PRIOR_SESSION_BINDING_FAILED" in item for item in result["issues"]))

    def test_after_ten_has_no_reachable_window(self):
        reachable, missed = reachable_points("2026-09-17", "2026-09-17T10:02:00+08:00")
        self.assertEqual(reachable, [])
        self.assertEqual(len(missed), 6)

    def test_invalid_forward_registry_blocks_preflight_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); inputs = self._inputs(root)
            result = assess(inputs[0], inputs[1], "2026-09-17", inputs[2], inputs[3],
                            "2026-09-17T09:18:00+08:00", root/"missing_registry")
            self.assertEqual(result["status"], "BLOCKED")
            self.assertEqual(result["absorption_forward_registry_status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
