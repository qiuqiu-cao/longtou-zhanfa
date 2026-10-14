import json
import tempfile
import unittest
from pathlib import Path

import trade_path_aggregate_v16 as aggregate


def report(code="600001", day="2026-09-07", group="CONSECUTIVE_BOARD", complete=True):
    rows = []
    for offset, value, state in [(1, .01, "NORMAL"), (2, .02, "NORMAL"), (3, -.01, "LOCKED_LIMIT_DOWN")]:
        rows.append({"session_offset": offset, "net_close_return": value if complete else None,
                     "sellability_state": state if complete else None})
    return {
        "version": aggregate.REPORT_VERSION, "code": code, "entry_date_B": day,
        "board_group": group, "marks": rows,
        "B_plus_2_and_B_plus_3_both_positive": False if complete else None,
        "maximum_favourable_excursion_net": .04 if complete else None,
        "maximum_adverse_excursion_net": -.03 if complete else None,
        "window_B1_B3_complete": complete,
        "strategy_path_metric_eligible": complete,
    }


class TradePathAggregateV16Tests(unittest.TestCase):
    def test_aggregate_keeps_unresolved_and_blocked_denominators(self):
        result = aggregate.aggregate(
            [report(), report("600002", "2026-09-08", "FIRST_BOARD", False)], [],
            "2026-09-10T15:05:00+08:00")
        self.assertEqual(result["reports"], 2)
        self.assertEqual(result["horizons"]["1"]["observed"], 1)
        self.assertEqual(result["horizons"]["1"]["unresolved"], 1)
        self.assertEqual(result["horizons"]["3"]["blocked_sellability_count"], 1)
        self.assertIsNone(result["completed_trade_win_rate"])
        self.assertEqual(result["groups"]["FIRST_BOARD"]["complete_windows"], 0)

    def test_duplicate_trade_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate.aggregate([report(), report()], [], "2026-09-10T15:05:00+08:00")

    def test_directory_run_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            output = Path(tmp) / "out"
            source.mkdir()
            (source / "report.json").write_text(json.dumps(report()), encoding="utf-8")
            result = aggregate.run([source], "2026-09-10T15:05:00+08:00", output)
            self.assertTrue(Path(result["report_path"]).exists())
            self.assertEqual(result["status"], "PATH_VALUATION_ONLY")
            with self.assertRaises(FileExistsError):
                aggregate.run([source], "2026-09-10T15:05:00+08:00", output)


if __name__ == "__main__":
    unittest.main()
