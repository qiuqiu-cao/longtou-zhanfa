import json
import tempfile
import unittest
from pathlib import Path

import trade_report_aggregate_v16 as aggregate
import broker_exit_import_v16 as lifecycle


def report(day, code, ret, eligible=True, group="CONSECUTIVE_BOARD"):
    return {
        "version": lifecycle.VERSION,
        "entry_trade_date": day,
        "exit_asof": day + "T15:10:00+08:00",
        "code": code,
        "plan_id": "P-" + code,
        "board_group": group,
        "strategy_complete_trade_eligible": eligible,
        "strategy_net_return": ret if eligible else None,
    }


class TradeReportAggregateV16Tests(unittest.TestCase):
    def test_only_complete_strategy_lifecycles_enter_metrics(self):
        result = aggregate.aggregate(
            [report("2026-09-08", "600002", -.08),
             report("2026-09-07", "600001", .10, group="FIRST_BOARD"),
             report("2026-09-09", "600003", .03, eligible=False)],
            [], "2026-09-10T15:15:00+08:00")
        summary = result["strategy"]
        self.assertEqual(summary["completed_strategy_trades"], 2)
        self.assertEqual(summary["winning_trades"], 1)
        self.assertEqual(summary["large_loss_rate"], .5)
        self.assertEqual(result["groups"]["FIRST_BOARD"]["completed_strategy_trades"], 1)
        self.assertIsNone(summary["account_max_drawdown"])

    def test_duplicate_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate.aggregate([report("2026-09-07", "600001", .1),
                                 report("2026-09-07", "600001", .2)], [], "2026-09-10")

    def test_unresolved_exit_states_are_reported_separately(self):
        open_report = report("2026-09-07", "600001", None, eligible=False)
        open_report.update(unresolved_open_position=True, partial_exit=False, fee_data_complete=False)
        partial_report = report("2026-09-08", "600002", None, eligible=False)
        partial_report.update(unresolved_open_position=True, partial_exit=True, fee_data_complete=True)
        off_plan = report("2026-09-09", "600003", None, eligible=False)
        off_plan.update(unresolved_open_position=False, partial_exit=False, fee_data_complete=True)
        summary = aggregate.aggregate([open_report, partial_report, off_plan], [], "2026-09-10")["strategy"]
        self.assertEqual(summary["unresolved_open_position_reports"], 2)
        self.assertEqual(summary["partial_exit_reports"], 1)
        self.assertEqual(summary["fee_incomplete_reports"], 1)
        self.assertEqual(summary["off_plan_or_ineligible_reports"], 1)

    def test_run_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "reports"
            output = Path(tmp) / "out"
            source.mkdir()
            (source / "report.json").write_text(json.dumps(report("2026-09-07", "600001", .1)), encoding="utf-8")
            result = aggregate.run([source], "2026-09-10", output)
            self.assertTrue(Path(result["report_path"]).exists())
            with self.assertRaises(FileExistsError):
                aggregate.run([source], "2026-09-10", output)


if __name__ == "__main__":
    unittest.main()
