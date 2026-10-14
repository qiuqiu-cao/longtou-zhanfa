"""Synthetic price/cohort/source contracts; no live signals or historical fits."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import model_upgrade_inputs as inputs
from emotion_three_regime_challenger_v86 import FEATURES as MARKET_BASE
from next_session_forecast_lab_v271 import IDENTITY, EMOTION, PRICE_VOLUME
from public_outcome_bridge_v293 import candidate_truth


SIGNAL, TARGET = "2026-09-30", "2026-10-08"


def pool_row(code="000001", boards=2, cents=1100, turnover=2.5):
    return {"code": code, "name": "synthetic", "boards": boards,
            "price_cents": cents, "turnover_pct": turnover, "amount": 100000.}


def day_bar(close=1100, flat=True):
    return {"open": close if flat else close-10, "close": close,
            "high": close, "low": close if flat else close-20, "volume_native": 100.}


def pool(rows, day=TARGET):
    return {"date": day, "rows": rows, "available_at": day+"T15:10:00+08:00"}


class UpgradeInputsSynthetic(unittest.TestCase):
    def test_cent_exact_and_true_daily_one_word_separate_from_legacy_proxy(self):
        rows = [pool_row("000001", turnover=2.5), pool_row("000002", turnover=4.),
                pool_row("000003", cents=1099)]
        daily = {r["code"]: {"days": {SIGNAL: day_bar(1000), TARGET: day_bar()},
                                    "quote": {"status": "UNAVAILABLE"}} for r in rows}
        observed = inputs._pool_frame(pool(rows), daily, TARGET)
        self.assertEqual(observed.loc[0, "actual_one_word_daily"], 1.)
        self.assertEqual(observed.loc[0, "one_word_proxy"], 1.)
        self.assertEqual(observed.loc[1, "actual_one_word_daily"], 1.)
        self.assertEqual(observed.loc[1, "one_word_proxy"], 0.)
        # A one-cent disagreement cannot become close agreement or non-one-word.
        self.assertTrue(observed.loc[2, ["actual_one_word_daily", "one_word_proxy", "amplitude"]].isna().all())
        self.assertEqual(observed.loc[2, "ohlc_status"], "UNKNOWN")

    def test_missing_and_invalid_turnover_amount_remain_unknown(self):
        for turnover, amount in [(None, None), (-1., -2.), (np.inf, np.inf)]:
            r = pool_row(turnover=turnover)
            r["amount"] = amount
            daily = {r["code"]: {"days": {TARGET: day_bar()}}}
            with self.subTest(turnover=turnover):
                observed = inputs._pool_frame(pool([r]), daily, TARGET)
                self.assertTrue(pd.isna(observed.loc[0, "turnover"]))
                self.assertTrue(pd.isna(observed.loc[0, "amount"]))
                self.assertTrue(pd.isna(observed.loc[0, "one_word_proxy"]))

    def test_invalid_fields_record_errors_and_cannot_complete_market_status(self):
        invalid = pool_row(boards=3, turnover=np.inf)
        invalid["amount"] = -1.
        current = inputs._pool_frame(pool([invalid]), {"000001": {"days": {TARGET: day_bar()}}}, TARGET)
        self.assertEqual(set(current.loc[0, "field_errors"]), {"turnover", "amount"})
        previous = pd.DataFrame({"code": ["000001"], "boards": [2]})
        pools = {(TARGET, "failed_board"): pool([]), (TARGET, "limit_down"): pool([]),
                 (TARGET, "previous_limit"): pool([{"code": "000001", "pct": 0.}])}
        with patch("leader_system_v3.emotion_dimensions", return_value={
                "emotion_quality_score": .5, "emotion_data_completeness": 1.}):
            result = inputs._market_row(TARGET, current, previous,
                {"max_board": 2, "limit_count": 1, "emotion_state": "退潮"}, [], pools)
        self.assertEqual(result["data_status"], "partial")

    def _market(self, feedback=None, missing_failed=False):
        previous = pd.DataFrame({"code": ["000001", "000002", "000003", "000004", "000005"],
                                 "boards": [1, 2, 3, 4, 5]})
        current = pd.DataFrame({"code": ["000001", "000002", "000004", "000005"],
                                "boards": [2, 3, 5, 5], "one_word_proxy": [0.] * 4})
        feedback = ([{"code": c, "pct": 0.} for c in previous.code] +
                    [{"code": "000999", "pct": 10000.}]) if feedback is None else feedback
        pools = {(TARGET, "previous_limit"): pool(feedback),
                 (TARGET, "limit_down"): pool([])}
        if not missing_failed:
            pools[(TARGET, "failed_board")] = pool([])
        with patch("leader_system_v3.emotion_dimensions", return_value={
                "emotion_quality_score": .5, "emotion_data_completeness": 1.}):
            return inputs._market_row(TARGET, current, previous,
                {"max_board": 5, "limit_count": 5, "emotion_state": "退潮"}, [], pools)

    def test_original_v3_mid_rate_and_full_previous_feedback_denominator(self):
        observed = self._market()
        self.assertAlmostEqual(observed["promotion_mid"], 2/3)
        self.assertAlmostEqual(observed["promotion_overall"], 3/5)
        self.assertEqual(observed["yesterday_premium"], 0.)
        self.assertEqual(observed["previous_limit_feedback_coverage"], 1.)
        self.assertEqual(observed["data_status"], "verified")
        missing = self._market(missing_failed=True)
        self.assertTrue(pd.isna(missing["failed_rate"]))
        self.assertEqual(missing["data_status"], "partial")

    def test_feedback_missing_member_or_nonfinite_never_verified(self):
        for values in [[{"code": "000001", "pct": 0.}],
                       [{"code": "00000"+str(i), "pct": np.inf if i==3 else 0.} for i in range(1, 6)]]:
            with self.subTest(values=values):
                observed = self._market(values)
                self.assertTrue(pd.isna(observed["yesterday_premium"]))
                self.assertEqual(observed["data_status"], "partial")

    def test_quote_missing_cannot_be_promotion_failure_or_success(self):
        candidate = {"code": "000001", "boards": 2}
        original = pool([pool_row(boards=2, cents=1000)], SIGNAL)
        future = pool([pool_row(boards=3, cents=1100)])
        feedback = pool([{"code": "000001", "price_cents": 1100, "upper_cents_explicit": 1100}])
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / "feedback.json"
            raw.write_text(json.dumps({"data": {"pool": [{"c": "000001", "ylbc": 2}]}}), encoding="utf-8")
            feedback["raw_path"] = str(raw)
            daily = {"000001": {"days": {SIGNAL: day_bar(1000), TARGET: day_bar(1100)},
                                  "quote": {"status": "UNAVAILABLE"}}}
            observed = candidate_truth(candidate, original, future, feedback, daily, SIGNAL, TARGET)
            self.assertIsNone(observed["promoted"])
            self.assertEqual(observed["status"], "unavailable")
            self.assertEqual(observed["reason"], "QUOTE_SNAPSHOT_MISSING")

    def _derive_synthetic(self, sessions, raw_failure=False, include_previous=True, require_next=True, all_first_boards=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        folder = Path(temp.name)
        protocol_path, calendar_path = folder / "protocol.json", folder / "calendar.json"
        protocol_path.write_text("{}", encoding="utf-8")
        calendar = {"status": "verified", "source": "SYNTHETIC_OFFICIAL_CALENDAR", "sessions": sessions}
        calendar_path.write_text(json.dumps(calendar), encoding="utf-8")
        values = {k: .1 for k in MARKET_BASE}
        values.update({"date": SIGNAL, "emotion_state": "退潮", "data_status": "verified",
                       "emotion_quality_score": .4, "promotion_mid": .75, "failed_rate": .1,
                       "max_board": 2, "limit_count": 2, "space_feedback": .5})
        pd.DataFrame([values]).to_csv(folder / "combined_market_input.csv", index=False)
        previous = pool([pool_row("000001", 2, 1000), pool_row("000002", 2, 1000)], SIGNAL)
        current = pool([pool_row("000001", 1 if all_first_boards else 3),
                        pool_row("000002", 1 if all_first_boards else 2)])
        all_pools = {(TARGET, "limit_up"): current,
                     (TARGET, "failed_board"): pool([{"code": "000999"}]),
                     (TARGET, "limit_down"): pool([]),
                     (TARGET, "previous_limit"): pool([{"code": "000001", "pct": 0.}, {"code": "000002", "pct": 0.}])}
        if include_previous:
            all_pools[(SIGNAL, "limit_up")] = previous
        daily = {c: {"available_at": TARGET+"T15:10:00+08:00", "days": {
            SIGNAL: day_bar(1000), TARGET: day_bar(1100, c=="000001")}} for c in ["000001", "000002"]}
        relay_fields = list(dict.fromkeys(IDENTITY + EMOTION + PRICE_VOLUME)) + [
            k+"_same_height_percentile" for k in ["price", "signal_amount", "turnover", "turnover_vs_prev"]] + [
            "signal_one_word", "log_signal_amount"]
        prepared = {"market": pd.DataFrame({"date": [SIGNAL], "sh_a_return_pct": [.1], "sz_a_return_pct": [.1]}),
            "relay": pd.DataFrame({"date": [SIGNAL], "code": ["000001"], "turnover": [.025]}),
            "relay_fields": relay_fields, "provenance": {"source_hashes": {}, "dependency_hashes": {}}}
        manifest = {"target_date": TARGET, "recorded_at": TARGET+"T15:12:00+08:00"}
        with patch.object(inputs, "OLD", folder), patch.object(inputs, "verify_protocol", return_value={"calendar_path": str(calendar_path)}), \
                patch.object(inputs, "_closed", return_value=True), patch.object(inputs, "_source_bindings", return_value={}), \
                patch("model_upgrade_history.prepare_datasets", return_value=prepared), \
                patch.object(inputs, "_old_pool", return_value=None), \
                patch.object(inputs, "reconstruct", side_effect=ValueError("SYNTHETIC_RAW_HASH_CONFLICT") if raw_failure else None,
                             return_value=(manifest, all_pools, daily, [])) as reconstruction:
            result = inputs._derive(protocol_path, folder / "raw", calendar_path, [],
                                    "2026-10-08T15:20:00+08:00", require_next=require_next)
            reconstruction.assert_called_once()
            return result

    def test_derive_rebuilds_real_api_and_preserves_full_current_cohort(self):
        result = self._derive_synthetic([SIGNAL, TARGET, "2026-10-09"])
        self.assertEqual([c["code"] for c in result["relay"]], ["000001", "000002"])
        self.assertEqual(result["relay"][0]["actual_one_word_daily"], 1.)
        self.assertEqual(result["relay"][1]["actual_one_word_daily"], 0.)
        self.assertEqual(result["market"]["features"]["promotion_mid"], .5)
        self.assertEqual(result["market"]["features"]["nonpositive_premium_promotion_down"], 1.)
        self.assertIsNone(result["market"]["features"]["sh_a_return_pct"])
        self.assertIn("sh_a_return_pct", result["market"]["unknown_fields"])

    def test_derive_missing_official_previous_pool_never_crosses_gap(self):
        result = self._derive_synthetic([SIGNAL, "2026-10-07", TARGET, "2026-10-09"], include_previous=False)
        self.assertIsNone(result["market"]["features"]["nonpositive_premium_promotion_down"])
        self.assertIsNone(result["market"]["features"]["promotion_mid"])
        self.assertEqual(result["market"]["status"], "partial")

    def test_raw_reconstruction_failure_prevents_hand_written_verified(self):
        with self.assertRaisesRegex(ValueError, "RAW_HASH_CONFLICT"):
            self._derive_synthetic([SIGNAL, TARGET, "2026-10-09"], raw_failure=True)

    def test_matured_calendar_last_target_needs_no_later_session(self):
        result = self._derive_synthetic([SIGNAL, TARGET], require_next=False)
        self.assertEqual(result["signal_date"], TARGET)
        self.assertIsNone(result["target_date"])
        with self.assertRaisesRegex(ValueError, "extend official calendar"):
            self._derive_synthetic([SIGNAL, TARGET], require_next=True)

    def test_all_first_boards_keep_market_and_empty_relay_without_relay_helper(self):
        with patch("september_forecast_holdout_v278.relay_features",
                   side_effect=AssertionError("empty original relay group must not invoke relay helper")) as helper:
            result = self._derive_synthetic([SIGNAL, TARGET, "2026-10-09"], all_first_boards=True)
            helper.assert_not_called()
        self.assertEqual(result["relay"], [])
        self.assertEqual(result["market"]["features"]["limit_count"], 2.)
        self.assertEqual(result["market"]["features"]["max_board"], 1.)
        self.assertEqual(result["signal_date"], TARGET)
        self.assertEqual(result["target_date"], "2026-10-09")


if __name__ == "__main__":
    unittest.main()
