import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import open_confirmation_v4 as opening


class OpenConfirmationV4Tests(unittest.TestCase):
    def _snapshot(self, asof: str = "09:35") -> pd.DataFrame:
        return pd.DataFrame([{
            "signal_date": "2026-08-31", "trade_date": "2026-09-01", "asof": asof,
            "code": "600001", "name": "测试", "auction_gap": 0.03,
            "auction_amount_ratio": 0.08, "auction_cancel_ratio": 0.10,
            "expectation_gap": 0.01, "vwap_slope_5m": 0.02,
            "vwap_slope_15m": math.nan, "pullback_volume_ratio": 0.60,
            "vwap_reclaimed": 1, "reseal_minute": 4, "theme_sync_open": 0.70,
            "rivals_eliminated": 0.50, "below_vwap_expanding": 0,
            "failed_seal_twice": 0, "source": "test",
            "theme_sync_basis": "verified_theme_registry",
            "theme_evidence_id": "evidence-001", "theme_group_id": "theme-001",
            "theme_member_coverage": 1.0, "theme_observed_peers": 2,
            "source_timestamp": f"2026-09-01 {asof}:00" if len(asof) == 5 else f"2026-09-01 {asof}",
            "data_status": "verified_broker_export",
        }])

    def _theme_registry(self, asof: str = "09:35") -> pd.DataFrame:
        snapshot = self._snapshot(asof).iloc[0]
        return pd.DataFrame([{
            "signal_date": snapshot["signal_date"], "trade_date": snapshot["trade_date"],
            "asof": asof, "code": snapshot["code"],
            "theme_sync_open": snapshot["theme_sync_open"],
            "theme_sync_basis": "verified_theme_registry",
            "theme_evidence_id": snapshot["theme_evidence_id"],
            "theme_group_id": snapshot["theme_group_id"],
            "theme_member_coverage": snapshot["theme_member_coverage"],
            "theme_observed_peers": snapshot["theme_observed_peers"],
        }])

    def _candidate(self, eligible: bool = True) -> pd.Series:
        return pd.Series({
            "date": "2026-08-31", "code": "600001", "name": "测试", "boards": 3,
            "valid_identity": 1 if eligible else 0, "one_word_proxy": 0,
            "emotion_state": "升温", "announcement_risk": 0,
        })

    def test_verified_first_board_pioneer_can_enter_open_confirmation(self):
        candidate = self._candidate(False)
        candidate["boards"] = 1
        candidate["competition_identity"] = "pioneer"
        result = opening.evaluate_row(candidate, self._snapshot().iloc[0], "09:35")
        self.assertEqual(result["confirmation_status"], "CONFIRM_RESEAL")
        self.assertEqual(result["board_scope_policy"], "ALL_VERIFIED_FIRST_AND_CONSECUTIVE_BOARDS")

    def test_unverified_random_first_board_remains_ineligible(self):
        candidate = self._candidate(True)
        candidate["boards"] = 1
        result = opening.evaluate_row(candidate, self._snapshot().iloc[0], "09:35")
        self.assertEqual(result["confirmation_status"], "INELIGIBLE")

    def test_verified_six_board_is_not_mechanically_excluded(self):
        candidate = self._candidate(True)
        candidate["boards"] = 6
        result = opening.evaluate_row(candidate, self._snapshot().iloc[0], "09:35")
        self.assertEqual(result["confirmation_status"], "CONFIRM_RESEAL")

    def test_future_open_fields_are_rejected_at_0925(self):
        frame = self._snapshot("09:25")
        frame.loc[0, "source_timestamp"] = "2026-09-01 09:25:00"
        with self.assertRaisesRegex(ValueError, "future leakage"):
            opening.validate_snapshots(frame, self._theme_registry("09:25"))

    def test_source_timestamp_after_asof_is_rejected(self):
        frame = self._snapshot("09:35")
        frame.loc[0, "source_timestamp"] = "2026-09-01 09:36:00"
        with self.assertRaisesRegex(ValueError, "later than asof"):
            opening.validate_snapshots(frame, self._theme_registry())

    def test_missing_required_field_is_data_limited(self):
        snapshot = self._snapshot().iloc[0].copy()
        snapshot["theme_sync_open"] = math.nan
        result = opening.evaluate_row(self._candidate(), snapshot, "09:35")
        self.assertEqual(result["confirmation_status"], "DATA_LIMITED")
        self.assertIn("theme_sync_open", result["missing_fields"])

    def test_below_vwap_expanding_is_hard_cancel(self):
        snapshot = self._snapshot().iloc[0].copy()
        snapshot["below_vwap_expanding"] = 1
        result = opening.evaluate_row(self._candidate(), snapshot, "09:35")
        self.assertEqual(result["confirmation_status"], "CANCEL")

    def test_standard_reseal_chain_confirms_only_paper_execution(self):
        result = opening.evaluate_row(self._candidate(), self._snapshot().iloc[0], "09:35")
        self.assertEqual(result["confirmation_status"], "CONFIRM_RESEAL")
        self.assertEqual(result["execution_mode"], "PAPER_ONLY")

    def test_overheated_open_without_exception_stays_watch(self):
        snapshot = self._snapshot().iloc[0].copy()
        snapshot["auction_gap"] = 0.08
        snapshot["pullback_volume_ratio"] = 0.60
        result = opening.evaluate_row(self._candidate(), snapshot, "09:35")
        self.assertEqual(result["confirmation_status"], "WATCH_OVERHEATED")

    def test_ineligible_candidate_cannot_be_rescued_by_open_strength(self):
        result = opening.evaluate_row(self._candidate(False), self._snapshot().iloc[0], "09:35")
        self.assertEqual(result["confirmation_status"], "INELIGIBLE")

    def test_riskoff_core_can_only_enter_stricter_countertrend_research_path(self):
        candidate = self._candidate(True)
        candidate["emotion_state"] = "大分歧"
        candidate["boards"] = 4
        candidate["competition_identity"] = "height_anchor"
        candidate["theme_status"] = "verified_explicit_source_roster"
        standard = opening.evaluate_row(candidate, self._snapshot().iloc[0], "09:35")
        self.assertEqual(standard["confirmation_status"], "INELIGIBLE")
        research = opening.evaluate_row(
            candidate, self._snapshot().iloc[0], "09:35",
            allow_countertrend_research=True,
        )
        self.assertEqual(research["confirmation_status"],
                         "CONFIRM_COUNTERTREND_RESEAL_RESEARCH")
        self.assertEqual(research["execution_mode"], "PAPER_ONLY")

    def test_countertrend_research_rejects_unverified_theme_or_unknown_one_word(self):
        candidate = self._candidate(True)
        candidate["emotion_state"] = "退潮"
        candidate["boards"] = 3
        candidate["competition_identity"] = "height_anchor"
        candidate["theme_status"] = "unavailable_real_specific_theme"
        candidate["one_word_proxy"] = math.nan
        result = opening.evaluate_row(
            candidate, self._snapshot().iloc[0], "09:35",
            allow_countertrend_research=True,
        )
        self.assertEqual(result["confirmation_status"], "INELIGIBLE")
        self.assertIn("不是一字", result["reasons"])

    def test_snapshot_keys_are_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "snapshot.csv"
            archive = root / "archive.csv"
            self._snapshot().to_csv(source, index=False)
            registry = self._theme_registry()
            opening.import_snapshots(source, archive, registry)
            with self.assertRaisesRegex(ValueError, "immutable snapshot keys"):
                opening.import_snapshots(source, archive, registry)

    def test_industry_proxy_cannot_supply_theme_sync(self):
        frame = self._snapshot()
        frame.loc[0, "theme_sync_basis"] = "industry_proxy"
        with self.assertRaisesRegex(ValueError, "verified theme basis"):
            opening.validate_snapshots(frame, self._theme_registry())

    def test_high_open_without_auction_amount_is_cancelled(self):
        snapshot = self._snapshot().iloc[0].copy()
        snapshot["auction_gap"] = 0.06
        snapshot["auction_amount_ratio"] = 0.01
        result = opening.evaluate_row(self._candidate(), snapshot, "09:35")
        self.assertEqual(result["confirmation_status"], "CANCEL")

    def test_timezone_aware_source_timestamp_is_normalized_to_shanghai(self):
        frame = self._snapshot()
        frame.loc[0, "source_timestamp"] = "2026-09-01T01:35:00Z"
        frame.loc[0, "theme_sync_basis"] = "verified_theme_registry"
        checked = opening.validate_snapshots(frame, self._theme_registry())
        self.assertEqual(len(checked), 1)

    def test_theme_sync_requires_registry_binding(self):
        frame = self._snapshot()
        frame.loc[0, "theme_sync_basis"] = "verified_theme_registry"
        with self.assertRaisesRegex(ValueError, "registry is unavailable"):
            opening.validate_snapshots(frame, pd.DataFrame())

    def test_theme_sync_value_cannot_differ_from_registry(self):
        frame = self._snapshot()
        frame.loc[0, "theme_sync_basis"] = "verified_theme_registry"
        registry = self._theme_registry()
        registry.loc[0, "theme_sync_open"] = 0.9
        with self.assertRaisesRegex(ValueError, "differs from verified registry"):
            opening.validate_snapshots(frame, registry)

    def test_unbound_theme_sync_cannot_confirm_when_evaluate_called_directly(self):
        snapshot = self._snapshot().iloc[0].copy()
        snapshot["theme_sync_basis"] = "user_verified_theme"
        result = opening.evaluate_row(self._candidate(), snapshot, "09:35")
        self.assertEqual(result["confirmation_status"], "DATA_LIMITED")
        self.assertIn("没有绑定冻结成员名册", result["reasons"])


if __name__ == "__main__":
    unittest.main()
