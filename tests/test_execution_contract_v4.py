import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import execution_contract_v4 as execution


SHANGHAI = ZoneInfo("Asia/Shanghai")


class ExecutionContractV4Tests(unittest.TestCase):
    def _candidate(self, boards: int = 3) -> pd.Series:
        return pd.Series({
            "date": "2026-09-01", "code": "600001", "name": "甲", "boards": boards,
            "emotion_state": "升温", "valid_identity": 1, "one_word_proxy": 0,
            "announcement_risk": 0,
        })

    def _snapshot(self, gap: float = 0.03) -> pd.Series:
        return pd.Series({
            "signal_date": "2026-09-01", "trade_date": "2026-09-02", "asof": "09:35",
            "code": "600001", "name": "甲", "auction_gap": gap,
            "auction_amount_ratio": 0.08, "auction_cancel_ratio": 0.10,
            "expectation_gap": 0.01, "vwap_slope_5m": 0.02,
            "vwap_slope_15m": None, "pullback_volume_ratio": 0.60,
            "vwap_reclaimed": 1, "reseal_minute": 4, "theme_sync_open": 0.70,
            "theme_sync_basis": "verified_theme_registry", "theme_evidence_id": "evidence-1",
            "theme_group_id": "theme-1", "theme_member_coverage": 1.0,
            "theme_observed_peers": 2, "rivals_eliminated": 0.5,
            "below_vwap_expanding": 0, "failed_seal_twice": 0,
            "source": "broker", "source_timestamp": "2026-09-02 09:35:00",
            "data_status": "verified_broker_export",
        })

    def _portfolio(self, gross: float = 0, theme: float = 0, positions: int = 0) -> pd.Series:
        return pd.Series({
            "account_equity": 1_000_000, "current_gross_exposure_pct": gross,
            "current_theme_exposure_pct": theme, "open_relay_positions": positions,
            "source": "paper portfolio", "source_timestamp": "2026-09-02 09:35:00",
            "data_status": "verified_import",
        })

    def _capture(self, complete: bool = True, late: bool = False) -> dict:
        return {
            "capture_complete": complete, "late_capture": late,
            "event_hash": "capture-event", "raw_sha256": "capture-file-hash",
        }

    def _contract(self, **kwargs) -> dict:
        return execution.build_contract(
            "session-1", kwargs.get("candidate", self._candidate()),
            kwargs.get("snapshot", self._snapshot()),
            kwargs.get("capture", self._capture()),
            kwargs.get("portfolio", self._portfolio()),
            kwargs.get("generated_at", datetime(2026, 9, 2, 9, 36, tzinfo=SHANGHAI)),
        )

    def _register_contract(self, output: Path, contract: dict) -> None:
        directory = output / contract["session_id"]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{contract['contract_id']}.json"
        path.write_text(json.dumps(contract, ensure_ascii=False), encoding="utf-8")
        execution.append_event(output, {
            "event_type": "CONTRACT", "contract_id": contract["contract_id"],
            "session_id": contract["session_id"], "signal_date": contract["signal_date"],
            "trade_date": contract["trade_date"], "asof": contract["asof"],
            "code": contract["code"], "contract_status": contract["contract_status"],
            "maximum_position_pct": contract["sizing"]["maximum_position_pct"],
            "generated_at": contract["generated_at"],
            "contract_file": str(path.resolve().relative_to(output.resolve())),
            "contract_sha256": execution.sha256_file(path),
        })

    def test_t1_stress_budget_caps_position_near_four_percent(self):
        contract = self._contract()
        self.assertEqual(contract["contract_status"], "PAPER_ENTRY_ALLOWED")
        self.assertAlmostEqual(contract["sizing"]["maximum_position_pct"], 0.005 / 0.12)
        self.assertIn("T+1", contract["same_day_rule"])
        self.assertIn("不能卖出", contract["same_day_rule"])
        self.assertIn("不得仍按同一开盘价", contract[
            "exit_monitoring_contract"]["same_open_prohibition"])
        self.assertIn("不得冒充", contract[
            "exit_monitoring_contract"]["offline_rule"])

    def test_five_board_hot_open_is_capped_at_two_percent(self):
        contract = self._contract(candidate=self._candidate(5), snapshot=self._snapshot(0.08))
        self.assertAlmostEqual(contract["sizing"]["maximum_position_pct"], 0.02)

    def test_first_and_six_board_are_sized_not_mechanically_rejected(self):
        first = self._candidate(1)
        first["competition_identity"] = "pioneer"
        first_contract = self._contract(candidate=first)
        high_contract = self._contract(candidate=self._candidate(6))
        self.assertEqual(first_contract["contract_status"], "PAPER_ENTRY_ALLOWED")
        self.assertEqual(high_contract["contract_status"], "PAPER_ENTRY_ALLOWED")
        self.assertGreater(first_contract["sizing"]["maximum_position_pct"], 0)
        self.assertAlmostEqual(high_contract["sizing"]["maximum_position_pct"], 0.02)
        self.assertIn("B+3", high_contract["holding_period"])

    def test_no_position_slot_forces_no_trade(self):
        contract = self._contract(portfolio=self._portfolio(positions=2))
        self.assertEqual(contract["contract_status"], "NO_TRADE")
        self.assertEqual(contract["sizing"]["maximum_position_pct"], 0)

    def test_cancelled_open_confirmation_cannot_become_contract_entry(self):
        snapshot = self._snapshot()
        snapshot["below_vwap_expanding"] = 1
        contract = self._contract(snapshot=snapshot)
        self.assertEqual(contract["confirmation_status"], "CANCEL")
        self.assertEqual(contract["contract_status"], "NO_TRADE")

    def test_late_contract_cannot_authorize_entry(self):
        contract = self._contract(generated_at=datetime(2026, 9, 2, 9, 41, tzinfo=SHANGHAI))
        self.assertTrue(contract["generated_late"])
        self.assertEqual(contract["contract_status"], "NO_TRADE")

    def test_portfolio_timestamp_cannot_come_from_future(self):
        frame = pd.DataFrame([self._portfolio()])
        frame.loc[0, "source_timestamp"] = "2026-09-02 09:36:00"
        with self.assertRaisesRegex(ValueError, "later than contract"):
            execution.validate_portfolio_state(frame, datetime(2026, 9, 2, 9, 35, tzinfo=SHANGHAI))

    def test_fill_cannot_exceed_frozen_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            contract = self._contract()
            self._register_contract(output, contract)
            with self.assertRaisesRegex(ValueError, "exceeds frozen"):
                execution.record_fill(
                    contract["contract_id"], "VERIFIED_PAPER_FILL", "09:36", 11.0, 4000,
                    "paper", output, datetime(2026, 9, 2, 15, 1, tzinfo=SHANGHAI),
                )

    def test_fill_quantity_must_use_a_share_board_lot(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            contract = self._contract()
            self._register_contract(output, contract)
            with self.assertRaisesRegex(ValueError, "multiple of 100"):
                execution.record_fill(
                    contract["contract_id"], "VERIFIED_PAPER_FILL", "09:36", 11.0, 55,
                    "paper", output, datetime(2026, 9, 2, 15, 1, tzinfo=SHANGHAI),
                )

    def test_blocked_exit_keeps_net_return_unresolved_and_ledger_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            contract = self._contract()
            self._register_contract(output, contract)
            execution.record_fill(
                contract["contract_id"], "VERIFIED_PAPER_FILL", "09:36", 11.0, 3000,
                "paper", output, datetime(2026, 9, 2, 15, 1, tzinfo=SHANGHAI),
            )
            result = execution.record_exit(
                contract["contract_id"], 9.9, None, 0.003, True,
                "MARKET_ERROR", "T+1一字跌停", output,
                datetime(2026, 9, 3, 15, 5, tzinfo=SHANGHAI),
            )
            self.assertIsNone(result["event"]["net_return"])
            self.assertTrue(result["event"]["large_loss"])
            self.assertEqual(result["verification"]["status"], "PASS")

    def test_blocked_exit_can_resolve_on_later_session(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            contract = self._contract()
            self._register_contract(output, contract)
            execution.record_fill(
                contract["contract_id"], "VERIFIED_PAPER_FILL", "09:36", 11.0, 3000,
                "paper", output, datetime(2026, 9, 2, 15, 1, tzinfo=SHANGHAI),
            )
            execution.record_exit(
                contract["contract_id"], 9.9, None, 0.003, True,
                "MARKET_ERROR", "B+1一字跌停", output,
                datetime(2026, 9, 3, 15, 5, tzinfo=SHANGHAI), session_offset=1,
            )
            result = execution.record_exit(
                contract["contract_id"], 9.5, 9.45, 0.003, False,
                "NO_ERROR", "B+2首次可成交退出", output,
                datetime(2026, 9, 4, 15, 5, tzinfo=SHANGHAI), session_offset=2,
            )
            self.assertIsNotNone(result["event"]["net_return"])
            monitor = execution.monitor(output)
            self.assertEqual(monitor["exit_attempts"], 2)
            self.assertEqual(monitor["resolved_exits"], 1)
            self.assertEqual(monitor["sell_blocked_unresolved"], 0)
            self.assertEqual(result["verification"]["status"], "PASS")

    def test_post_b3_exit_requires_prior_sell_blockage(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            contract = self._contract()
            self._register_contract(output, contract)
            execution.record_fill(
                contract["contract_id"], "VERIFIED_PAPER_FILL", "09:36", 11.0, 3000,
                "paper", output, datetime(2026, 9, 2, 15, 1, tzinfo=SHANGHAI),
            )
            with self.assertRaisesRegex(ValueError, "prior sell-blocked"):
                execution.record_exit(
                    contract["contract_id"], 10.5, 10.5, 0.003, False,
                    "NO_ERROR", "late exit", output,
                    datetime(2026, 9, 8, 15, 5, tzinfo=SHANGHAI), session_offset=4,
                )

    def test_contract_file_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            contract = self._contract()
            self._register_contract(output, contract)
            path = output / contract["session_id"] / f"{contract['contract_id']}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["sizing"]["maximum_position_pct"] = 0.5
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(execution.verify_ledger(output)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
