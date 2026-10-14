import copy
import json
import tempfile
import unittest
from pathlib import Path

from account_risk_profile_v143 import audit_profile, run, size_position


ASOF = "2026-09-17T08:30:00+08:00"


def profile():
    return {
        "policy_id": "paper-risk-v143", "account_kind": "paper",
        "source": "research_assumption", "data_status": "verified",
        "frozen_at": "2026-09-17T08:00:00+08:00",
        "available_at": "2026-09-17T08:00:01+08:00",
        "equity_evidence_id": None, "initial_cash": 100000,
        "single_trade_risk_fraction": .005, "gross_exposure_fraction": .40,
        "theme_exposure_fraction": .30, "maximum_drawdown_fraction": .12,
        "minimum_cash_buffer_fraction": .05, "maximum_positions": 1, "lot_size": 100,
        "commission_rate": .00025, "minimum_commission_cash": 5,
        "stamp_tax_rate_sell": .0005, "transfer_fee_rate_both": .00001,
        "other_fee_rate_both": 0,
        "regime_position_caps": {"启动": .25, "升温": .30, "加速": .25,
                                 "一致": .15, "大分歧": .10, "退潮": 0, "修复": .20},
        "stress_loss_fraction_by_group": {"FIRST_BOARD": .10,
                                          "CONSECUTIVE_BOARD": .12},
    }


def plan():
    return {
        "risk_policy_id": "paper-risk-v143", "regime": "升温",
        "board_group": "FIRST_BOARD", "account_equity_at_freeze": 100000,
        "existing_gross_exposure_cash": 0, "existing_theme_exposure_cash": 0,
        "available_cash_at_freeze": 100000, "maximum_acceptable_price": 10,
        "current_drawdown_fraction": 0, "theme_id": "specific-theme-a",
        "equity_snapshot_id": "eq-1", "exposure_snapshot_id": "ex-1",
        "frozen_at": "2026-09-17T08:10:00+08:00",
        "available_at": "2026-09-17T08:10:01+08:00",
    }


class AccountRiskProfileV143Tests(unittest.TestCase):
    def test_missing_profile_fails_closed(self):
        result = audit_profile(None, ASOF)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertFalse(result["can_size_positions"])

    def test_complete_profile_passes_and_matches_ledger_cash(self):
        result = audit_profile(profile(), ASOF, expected_initial_cash=100000)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["maximum_drawdown_fraction"], .12)
        mismatch = audit_profile(profile(), ASOF, expected_initial_cash=90000)
        self.assertIn("INITIAL_CASH_DOES_NOT_MATCH_LEDGER", mismatch["issues"])

    def test_user_account_requires_equity_evidence(self):
        item = profile()
        item["account_kind"] = "user_confirmed"
        result = audit_profile(item, ASOF)
        self.assertIn("USER_ACCOUNT_EQUITY_EVIDENCE_MISSING", result["issues"])

    def test_fee_aware_risk_cap_reduces_one_lot(self):
        result = size_position(profile(), plan(), ASOF)
        self.assertEqual(result["status"], "SIZE_READY")
        self.assertEqual(result["max_shares"], 400)
        self.assertLessEqual(result["estimated_stress_loss_cash"], 500)
        self.assertFalse(result["buy_signal"])

    def test_drawdown_halt_and_retreat_zero_cap(self):
        item = plan()
        item["current_drawdown_fraction"] = -.12
        self.assertEqual(size_position(profile(), item, ASOF)["status"], "DRAWDOWN_HALT")
        item = plan()
        item["regime"] = "退潮"
        self.assertEqual(size_position(profile(), item, ASOF)["max_shares"], 0)

    def test_immutable_cli_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "input.json", Path(folder) / "report.json"
            source.write_text(json.dumps({"profile": profile(), "plan": plan()}), encoding="utf-8")
            result = run(source, target, ASOF)
            self.assertEqual(result["position_size"]["max_shares"], 400)
            self.assertEqual(len(result["source_hashes"]), 2)
            with self.assertRaises(FileExistsError):
                run(source, target, ASOF)


if __name__ == "__main__":
    unittest.main()
