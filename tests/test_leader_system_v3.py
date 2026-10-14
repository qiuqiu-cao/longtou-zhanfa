import tempfile
import unittest
import json
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd

import leader_system_v3 as v3


class LeaderSystemV3Tests(unittest.TestCase):
    def test_limit_price_uses_half_up_rounding(self):
        self.assertEqual(v3.round_limit_price(10.05), 11.06)
        self.assertEqual(v3.round_limit_price(9.99), 10.99)

    def test_mainboard_filter(self):
        self.assertTrue(v3.is_mainboard("600000", "浦发银行"))
        self.assertTrue(v3.is_mainboard("000001", "平安银行"))
        self.assertFalse(v3.is_mainboard("300001", "特锐德"))
        self.assertFalse(v3.is_mainboard("600001", "*ST测试"))
        self.assertFalse(v3.is_mainboard("600001", "测试", new_flag=True))

    def test_daily_bars_reconstruct_true_streak(self):
        rows = []
        closes = [10.0, 11.0, 12.1, 11.8, 12.98]
        previous = [9.09, 10.0, 11.0, 12.1, 11.8]
        for i, (close, pre) in enumerate(zip(closes, previous), 1):
            rows.append({"date": f"2026-01-0{i}", "code": "600001", "name": "测试",
                         "open": close, "high": close, "low": close, "close": close,
                         "pre_close": pre, "turnover": 10, "amount": 1e8,
                         "float_cap": 1e9, "industry": "测试"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bars.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            pools = v3.pools_from_daily_bars(path)
        self.assertEqual(int(pools["2026-01-03"].iloc[0]["boards"]), 3)
        self.assertEqual(int(pools["2026-01-05"].iloc[0]["boards"]), 1)

    def test_daily_bars_mark_one_word_as_unbuyable_proxy(self):
        rows = [{"date": "2026-01-01", "code": "600001", "name": "测试",
                 "open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0,
                 "pre_close": 10.0, "turnover": 0.5, "amount": 1e7,
                 "float_cap": 1e9, "industry": "测试", "st": False, "new_listing": False}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bars.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            prepared = v3.prepare_daily_bars(path)
            pools = v3.pools_from_prepared_daily_bars(prepared)
        self.assertTrue(bool(prepared.iloc[0]["one_word_daily"]))
        self.assertEqual(str(pools["2026-01-01"].iloc[0]["first_seal"]), "092500")

    def test_recent_pool_append_cannot_rewrite_frozen_historical_dates(self):
        old = pd.DataFrame({"code": ["600001"], "boards": [2]})
        rewritten = pd.DataFrame({"code": ["600999"], "boards": [9]})
        fresh = pd.DataFrame({"code": ["600002"], "boards": [3]})
        result = v3.append_only_recent_pools(
            {"2026-08-28": old}, {"2026-08-28": rewritten, "2026-08-31": fresh})
        self.assertEqual(result["2026-08-28"].iloc[0]["code"], "600001")
        self.assertEqual(result["2026-08-31"].iloc[0]["code"], "600002")

    def test_suspension_gap_resets_continuous_board_streak(self):
        rows = [
            {"date": "2026-01-01", "code": "600001", "name": "甲", "open": 11, "high": 11, "low": 10, "close": 11, "pre_close": 10},
            {"date": "2026-01-03", "code": "600001", "name": "甲", "open": 12.1, "high": 12.1, "low": 11, "close": 12.1, "pre_close": 11},
            {"date": "2026-01-01", "code": "600002", "name": "乙", "open": 10, "high": 10, "low": 10, "close": 10, "pre_close": 10},
            {"date": "2026-01-02", "code": "600002", "name": "乙", "open": 10, "high": 10, "low": 10, "close": 10, "pre_close": 10},
            {"date": "2026-01-03", "code": "600002", "name": "乙", "open": 10, "high": 10, "low": 10, "close": 10, "pre_close": 10},
        ]
        frame = pd.DataFrame(rows)
        for field, value in (("turnover", 5), ("amount", 1e8), ("float_cap", 1e9),
                             ("industry", "测试"), ("st", False), ("new_listing", False)):
            frame[field] = value
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bars.csv"
            frame.to_csv(path, index=False)
            prepared = v3.prepare_daily_bars(path)
        stock = prepared[prepared["code"] == "600001"].sort_values("date")
        self.assertEqual(stock["boards"].tolist(), [1, 1])

    def test_daily_context_derives_failed_down_and_holder_feedback(self):
        rows = [
            {"date": "2026-01-01", "code": "600001", "name": "甲", "open": 10, "high": 10, "low": 9.8, "close": 10, "pre_close": 9.09},
            {"date": "2026-01-02", "code": "600001", "name": "甲", "open": 10.2, "high": 11, "low": 10, "close": 10.5, "pre_close": 10},
            {"date": "2026-01-01", "code": "600002", "name": "乙", "open": 10, "high": 10, "low": 10, "close": 10, "pre_close": 10},
            {"date": "2026-01-02", "code": "600002", "name": "乙", "open": 9.5, "high": 9.5, "low": 9, "close": 9, "pre_close": 10},
        ]
        frame = pd.DataFrame(rows)
        for field, value in (("turnover", 5), ("amount", 1e8), ("float_cap", 1e9),
                             ("industry", "测试"), ("st", False), ("new_listing", False)):
            frame[field] = value
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bars.csv"
            frame.to_csv(path, index=False)
            context = v3.derive_daily_market_context(v3.prepare_daily_bars(path)).set_index("date")
        day = context.loc["2026-01-02"]
        self.assertEqual(int(day["failed_count"]), 1)
        self.assertEqual(int(day["down_limit_count"]), 1)
        self.assertAlmostEqual(float(day["previous_limit_mean_return"]), 0.05)
        self.assertAlmostEqual(float(day["previous_limit_median_return"]), 0.05)
        self.assertAlmostEqual(float(day["yesterday_premium"]), 0.05)
        self.assertEqual(int(day["previous_limit_feedback_count"]), 1)

    def test_raw_previous_feedback_keeps_mean_and_median_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            old_raw = v3.RAW
            try:
                v3.RAW = Path(directory)
                target = v3.RAW / "20260102"
                target.mkdir()
                pd.DataFrame([
                    {"代码": "600001", "名称": "甲", "涨跌幅": 10.0},
                    {"代码": "600002", "名称": "乙", "涨跌幅": -8.0},
                    {"代码": "600003", "名称": "丙", "涨跌幅": -2.0},
                ]).to_csv(target / "previous_limit.csv", index=False)
                mean_return, median_return, tail_return = v3.raw_previous_feedback("2026-01-02")
            finally:
                v3.RAW = old_raw
        self.assertAlmostEqual(mean_return, 0.0)
        self.assertAlmostEqual(median_return, -0.02)
        self.assertLess(tail_return, median_return)

    def test_promotion_label_requires_exact_next_board(self):
        previous = pd.DataFrame({"code": ["600001", "600002"], "boards": [2, 3]})
        current = pd.DataFrame({"code": ["600001", "600002"], "boards": [3, 3]})
        rates = v3.promotion_rates(previous, current)
        self.assertEqual(rates["promotion_2to3"], 1.0)
        self.assertEqual(rates["promotion_3to4"], 0.0)

    def test_emotion_retreat_on_height_collapse(self):
        state = v3.classify_emotion({"height_change": -2, "limit_change": -20,
                                     "promotion_mid": 0.1, "space_feedback": 0,
                                     "failed_rate": 0.3, "down_limit_count": 8,
                                     "limit_count": 20, "emotion_score": 0.2})
        self.assertEqual(state, "退潮")

    def test_high_failed_rate_alone_is_not_market_divergence(self):
        row = {"height_change": 0, "limit_change": 5, "max_board": 5,
               "limit_count": 55, "seal_rate": 0.73, "promotion_overall": 0.45,
               "promotion_mid": 0.50, "space_feedback": 1, "space_feedback_3d": 0.70,
               "failed_rate": 0.27, "down_limit_count": 1,
               "previous_limit_tail_loss": -0.01, "yesterday_premium": 0.025,
               "middle_density": 0.55, "ladder_gap_count": 0, "unique_space": 1,
               "one_word_share": 0.15, "breadth_height_divergence": 0,
               "emotion_score": 0.70}
        self.assertNotEqual(v3.classify_emotion(row), "大分歧")

    def test_height_drop_with_healthy_middle_is_rotation_not_retreat(self):
        row = {"height_change": -3, "limit_change": 5, "max_board": 5,
               "limit_count": 55, "seal_rate": 0.80, "promotion_overall": 0.45,
               "promotion_mid": 0.50, "space_feedback": 1, "space_feedback_3d": 0.70,
               "failed_rate": 0.18, "down_limit_count": 1,
               "previous_limit_tail_loss": -0.01, "yesterday_premium": 0.025,
               "middle_density": 0.55, "ladder_gap_count": 0, "unique_space": 1,
               "one_word_share": 0.15, "breadth_height_divergence": 0,
               "emotion_score": 0.70}
        self.assertNotEqual(v3.classify_emotion(row), "退潮")

    def test_rolling_emotion_percentiles_use_only_previous_rows(self):
        history = [{"failed_rate": value, "down_limit_count": value * 10,
                    "promotion_mid": 1 - value, "yesterday_premium": value,
                    "previous_limit_tail_loss": value - 0.5}
                   for value in np.linspace(0.05, 0.45, 30)]
        current = {"failed_rate": 0.40, "down_limit_count": 4,
                   "promotion_mid": 0.20, "yesterday_premium": 0.10,
                   "previous_limit_tail_loss": -0.10}
        first = v3.rolling_emotion_context(history, current)
        changed_future = history + [{"failed_rate": 0.99, "down_limit_count": 99,
                                     "promotion_mid": 0, "yesterday_premium": -1,
                                     "previous_limit_tail_loss": -1}]
        second = v3.rolling_emotion_context(changed_future[:-1], current)
        self.assertEqual(first, second)
        self.assertGreater(first["failed_rate_percentile_60d"], 0.75)
        self.assertLess(first["promotion_mid_percentile_60d"], 0.25)

    def test_emotion_repairs_after_retreat(self):
        state = v3.classify_emotion({"height_change": 0, "limit_change": 15,
                                     "promotion_mid": 0.3, "space_feedback": 1,
                                     "failed_rate": 0.1, "down_limit_count": 1,
                                     "limit_count": 55, "emotion_score": 0.7}, "退潮")
        self.assertEqual(state, "修复")

    def test_false_acceleration_is_divergence_not_acceleration(self):
        row = {"height_change": 1, "limit_change": 20, "max_board": 7,
               "limit_count": 75, "seal_rate": 0.72, "promotion_overall": 0.18,
               "promotion_mid": 0.12, "space_feedback": 1, "space_feedback_3d": 0.30,
               "failed_rate": 0.28, "down_limit_count": 6, "failed_tail_loss": -0.07,
               "yesterday_premium": -0.01, "middle_density": 0.35,
               "ladder_gap_count": 2, "unique_space": 1, "one_word_share": 0.50,
               "breadth_height_divergence": 0}
        profile = v3.emotion_dimensions(row)
        self.assertEqual(profile["false_acceleration"], 1.0)
        self.assertEqual(profile["emotion_risk_level"], "HIGH")
        self.assertEqual(v3.classify_emotion(row), "大分歧")

    def test_healthy_acceleration_requires_relay_and_loss_confirmation(self):
        row = {"height_change": 1, "limit_change": 18, "max_board": 6,
               "limit_count": 68, "seal_rate": 0.88, "promotion_overall": 0.40,
               "promotion_mid": 0.38, "space_feedback": 1, "space_feedback_3d": 0.75,
               "failed_rate": 0.12, "down_limit_count": 1, "failed_tail_loss": -0.02,
               "yesterday_premium": 0.025, "middle_density": 0.50,
               "ladder_gap_count": 0, "unique_space": 1, "one_word_share": 0.18,
               "breadth_height_divergence": 0}
        profile = v3.emotion_dimensions(row)
        self.assertEqual(profile["false_acceleration"], 0.0)
        self.assertGreater(profile["relay_appetite"], 0.65)
        self.assertGreater(profile["loss_effect_health"], 0.65)
        self.assertEqual(v3.classify_emotion(row), "加速")

    def test_emotion_missing_fields_reduce_completeness_not_health_to_zero(self):
        profile = v3.emotion_dimensions({"limit_count": 55, "height_change": 1,
                                         "max_board": 5, "promotion_mid": 0.35,
                                         "ladder_gap_count": 0, "middle_density": 0.4,
                                         "unique_space": 1, "one_word_share": 0.1})
        self.assertLess(profile["emotion_data_completeness"], 0.60)
        self.assertTrue(np.isnan(profile["loss_effect_health"]))
        self.assertGreater(profile["ladder_health"], 0.5)

    def test_transition_evidence_does_not_overstate_small_samples(self):
        market = pd.DataFrame({"date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                               "emotion_state": ["启动", "升温", "加速"]})
        evidence = v3.transition_evidence(market)
        self.assertEqual(evidence["启动"]["observed_transitions"], 1)
        self.assertEqual(evidence["启动"]["confidence"], "LOW")

    def test_tiny_validation_window_forces_threshold_abstention(self):
        frame = pd.DataFrame({"date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                              "promoted_next_day": [1, 1, 1], "boards": [3, 3, 3],
                              "valid_identity": [1, 1, 1], "buyable_eod": [1, 1, 1],
                              "emotion_state": ["升温", "升温", "升温"],
                              "announcement_risk": [np.nan, np.nan, np.nan]})
        policy = v3.choose_threshold(frame, np.array([0.9, 0.8, 0.7]))
        self.assertFalse(policy["operational"])
        self.assertIsNone(policy["deployment_threshold"])
        self.assertIn("validation dates", policy["abstain_reason"])

    def test_four_way_time_split_has_three_embargo_dates_and_no_overlap(self):
        dates = pd.date_range("2025-01-01", periods=40, freq="D").strftime("%Y-%m-%d")
        frame = pd.DataFrame({"date": dates, "promoted_next_day": np.tile([0, 1], 20)})
        train, calibration, validation, blind, meta = v3.chronological_split(frame)
        groups = [set(part["date"]) for part in (train, calibration, validation, blind)]
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                self.assertFalse(groups[left] & groups[right])
        embargoes = {meta["embargo_1"], meta["embargo_2"], meta["embargo_3"]}
        self.assertEqual(len(embargoes), 3)
        self.assertFalse(embargoes & set().union(*groups))
        self.assertLess(max(groups[0]), min(groups[1]))
        self.assertLess(max(groups[1]), min(groups[2]))
        self.assertLess(max(groups[2]), min(groups[3]))

    def test_sufficient_validation_evidence_can_activate_threshold(self):
        dates = pd.date_range("2026-01-01", periods=20, freq="D").strftime("%Y-%m-%d")
        labels = np.array([1] * 15 + [0] * 5)
        frame = pd.DataFrame({"date": dates, "promoted_next_day": labels,
                              "boards": 3, "valid_identity": 1, "buyable_eod": 1,
                              "emotion_state": "升温", "announcement_risk": np.nan})
        probabilities = np.array([0.9] * 15 + [0.8] * 5)
        policy = v3.choose_threshold(frame, probabilities)
        self.assertTrue(policy["operational"])
        self.assertIsNotNone(policy["deployment_threshold"])
        self.assertGreaterEqual(policy["precision_wilson_95"][0], 0.50)

    def test_feature_distribution_flags_extreme_live_sample(self):
        train = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=40).strftime("%Y-%m-%d"),
                              "feature": np.linspace(0, 1, 40)})
        profile = v3.build_distribution_profile(train, ["feature"])
        results = v3.distribution_shift(profile, pd.DataFrame({"feature": [0.5, 10.0]}))
        self.assertEqual(results[0]["status"], "IN_DISTRIBUTION")
        self.assertEqual(results[1]["status"], "OUT_OF_DISTRIBUTION")
        self.assertIn("feature", results[1]["extreme_features"])

    def test_date_block_bootstrap_returns_ordered_probability_interval(self):
        train = pd.DataFrame({"date": np.repeat(pd.date_range("2026-01-01", periods=8).strftime("%Y-%m-%d"), 4),
                              "x": np.tile([0.0, 0.3, 0.7, 1.0], 8),
                              "promoted_next_day": np.tile([0, 0, 1, 1], 8)})
        validation = pd.DataFrame({"date": np.repeat(["2026-02-01", "2026-02-02"], 4),
                                   "x": np.tile([0.0, 0.3, 0.7, 1.0], 2),
                                   "promoted_next_day": np.tile([0, 0, 1, 1], 2)})
        for frame in (train, validation):
            frame["boards"] = 3
            frame["valid_identity"] = 1
            frame["buyable_eod"] = 1
            frame["emotion_state"] = "升温"
            frame["announcement_risk"] = np.nan
        base = v3.fit_candidate_model(train, validation, validation, ["x"])
        ensemble = v3.fit_bootstrap_ensemble(train, validation, base, ["x"], replicates=3, seed=7)
        lower, median, upper = v3.predict_bootstrap_ensemble(ensemble, validation)
        self.assertEqual(ensemble["replicates_fitted"], 3)
        self.assertTrue(np.all(lower <= median))
        self.assertTrue(np.all(median <= upper))

    def test_ece_is_zero_for_perfect_binary_probabilities(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(v3.expected_calibration_error(y, y), 0.0)

    def test_json_output_uses_null_for_unavailable_market_data(self):
        payload = json.loads(v3.json_text({"known": 1.0, "missing": np.nan}))
        self.assertEqual(payload["known"], 1.0)
        self.assertIsNone(payload["missing"])

    def test_previous_limit_tail_loss_is_named_precisely(self):
        profile = v3.emotion_dimensions({"previous_limit_tail_loss": -0.08,
                                         "height_change": 0, "max_board": 4,
                                         "ladder_gap_count": 0})
        self.assertLess(profile["loss_effect_health"], 0.3)

    def test_outcome_import_rejects_inconsistent_large_loss(self):
        frame = pd.DataFrame({"promoted_next_day": [0], "entry_price": [10.0],
                              "t1_close": [9.0], "large_loss": [0], "sell_blocked": [0]})
        with self.assertRaises(ValueError):
            v3.validate_outcomes_import(frame)

    def test_blocked_outcome_can_keep_mark_to_market_loss_but_not_net_return(self):
        frame = pd.DataFrame({"promoted_next_day": [0], "entry_price": [10.0],
                              "t1_close": [9.0], "large_loss": [1], "sell_blocked": [1],
                              "actual_exit_price": [np.nan], "net_return": [np.nan]})
        checked = v3.validate_outcomes_import(frame)
        self.assertLess(float(checked.iloc[0]["mark_to_market_return"]), -0.07)

    def test_free_history_collection_resumes_completed_codes(self):
        class FakeAk:
            calls = 0

            @staticmethod
            def stock_zh_a_spot_em():
                return pd.DataFrame({"代码": ["600001"], "名称": ["测试"]})

            @classmethod
            def stock_zh_a_hist(cls, **kwargs):
                cls.calls += 1
                return pd.DataFrame({"日期": ["2025-12-31", "2026-01-02"],
                                     "开盘": [10, 10.2], "收盘": [10, 11],
                                     "最高": [10.1, 11], "最低": [9.9, 10.1],
                                     "成交量": [100, 200], "成交额": [1000, 2200],
                                     "换手率": [1, 2]})

        args = SimpleNamespace(history_start="2026-01-01", history_end="2026-01-31",
                               codes_file=None, max_codes=None, workers=1, refresh_history=False)
        with tempfile.TemporaryDirectory() as directory:
            imported = Path(directory) / "imported"
            imported.mkdir()
            with mock.patch.object(v3, "IMPORTED", imported):
                v3.collect_free_history(FakeAk, args)
                v3.collect_free_history(FakeAk, args)
                progress = pd.read_csv(imported / "history-progress.csv", dtype={"code": str})
        self.assertEqual(FakeAk.calls, 1)
        self.assertEqual(progress.iloc[0]["status"], "success")

    def test_current_universe_falls_back_when_spot_provider_fails(self):
        class FakeAk:
            @staticmethod
            def stock_zh_a_spot_em():
                raise ConnectionError("spot unavailable")

            @staticmethod
            def stock_info_a_code_name():
                return pd.DataFrame({"code": ["600001", "300001", "000002"],
                                     "name": ["主板甲", "创业板", "*ST主板"]})

        with tempfile.TemporaryDirectory() as directory:
            imported = Path(directory) / "imported"
            imported.mkdir()
            with mock.patch.object(v3, "IMPORTED", imported):
                universe, source = v3.current_mainboard_universe(FakeAk)
                cached = pd.read_csv(imported / "current-universe.csv", dtype={"code": str})
        self.assertEqual(universe, [("600001", "主板甲")])
        self.assertIn("fallback", source)
        self.assertEqual(cached["code"].tolist(), ["600001"])

    def test_free_history_falls_back_to_sina_and_normalizes_turnover(self):
        class FakeAk:
            @staticmethod
            def stock_zh_a_spot_em():
                return pd.DataFrame({"代码": ["600001"], "名称": ["测试"]})

            @staticmethod
            def stock_zh_a_hist(**kwargs):
                raise ConnectionError("Eastmoney unavailable")

            @staticmethod
            def stock_zh_a_daily(**kwargs):
                return pd.DataFrame({
                    "date": ["2025-12-31", "2026-01-02"], "open": [10, 10.2],
                    "high": [10.1, 11], "low": [9.9, 10.1], "close": [10, 11],
                    "volume": [100, 200], "amount": [1000, 2200],
                    "outstanding_share": [1e8, 1e8], "turnover": [0.01, 0.02],
                })

        args = SimpleNamespace(history_start="2026-01-01", history_end="2026-01-31",
                               codes_file=None, max_codes=None, workers=1, refresh_history=False)
        with tempfile.TemporaryDirectory() as directory:
            imported = Path(directory) / "imported"
            imported.mkdir()
            with mock.patch.object(v3, "IMPORTED", imported):
                v3.collect_free_history(FakeAk, args)
                bars = pd.read_csv(imported / "daily-bars.csv", dtype={"code": str})
        self.assertAlmostEqual(float(bars.iloc[0]["turnover"]), 2.0)
        self.assertAlmostEqual(float(bars.iloc[0]["float_cap"]), 1.1e9)
        self.assertIn("Sina", bars.iloc[0]["source"])

    def test_missing_values_have_explicit_indicators(self):
        frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, np.nan, np.nan]})
        processor = v3.MatrixProcessor(["a", "b"]).fit(frame)
        transformed = processor.transform(frame)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(transformed.shape[1], 4)
        self.assertEqual(transformed[1, 2], 1.0)
        self.assertEqual(transformed[0, 3], 1.0)

    def test_announcement_respects_asof_cutoff(self):
        announcements = pd.DataFrame([
            {"code": "600001", "published_at": "2026-08-28 14:00:00", "risk_type": "减持"},
            {"code": "600001", "published_at": "2026-08-28 18:00:00", "risk_type": "停牌核查"},
        ])
        features = v3.announcement_features(announcements, "600001", "2026-08-28")
        self.assertEqual(features["reduction_risk"], 1.0)
        self.assertEqual(features["announcement_risk"], 0.8)

    @staticmethod
    def _execution_candidate(**overrides):
        row = {
            "date": "2026-01-01", "trade_date": "2026-01-02", "code": "600001",
            "name": "测试", "promoted_next_day": 0.0, "false_strength": np.nan,
            "large_loss": np.nan, "next_open_gap": np.nan, "entry_price": np.nan,
            "t1_close": np.nan, "actual_exit_price": np.nan,
            "mark_to_market_return": np.nan, "net_return": np.nan,
            "round_trip_cost": 0.003, "sell_blocked": np.nan,
            "execution_status": "unavailable", "execution_source": "unavailable",
            "execution_assumption": "", "false_strength_label_status": "unavailable",
            "large_loss_label_status": "unavailable",
        }
        row.update(overrides)
        return pd.DataFrame([row])

    @staticmethod
    def _execution_bars(one_word_trade=False, one_word_exit_down=False):
        trade = {"date": "2026-01-02", "code": "600001", "open": 10.6,
                 "high": 11.0, "low": 10.3, "close": 10.8, "pre_close": 10.0,
                 "limit_price": 11.0, "down_limit_price": 9.0,
                 "one_word_daily": False, "touched_limit": True}
        if one_word_trade:
            trade.update({"open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0,
                          "one_word_daily": True})
        exit_bar = {"date": "2026-01-03", "code": "600001", "open": 10.1,
                    "high": 10.2, "low": 9.7, "close": 9.8, "pre_close": 10.8,
                    "limit_price": 11.88, "down_limit_price": 9.72,
                    "one_word_daily": False, "touched_limit": False}
        if one_word_exit_down:
            exit_bar.update({"open": 9.72, "high": 9.72, "low": 9.72, "close": 9.72})
        return pd.DataFrame([trade, exit_bar])

    def test_daily_execution_proxy_requires_touch_and_prices_entry_at_limit(self):
        result = v3.attach_daily_execution_outcomes(
            self._execution_candidate(), self._execution_bars()).iloc[0]
        self.assertEqual(result["execution_status"], "inferred_daily_limit_touch_fill")
        self.assertAlmostEqual(result["entry_price"], 11.0)
        self.assertAlmostEqual(result["mark_to_market_return"], 9.8 / 11.0 - 1 - 0.003)
        self.assertEqual(result["large_loss"], 1.0)
        self.assertEqual(result["large_loss_label_status"], "inferred_from_unadjusted_daily_ohlc")

    def test_daily_execution_proxy_rejects_one_word_limit_up(self):
        result = v3.attach_daily_execution_outcomes(
            self._execution_candidate(), self._execution_bars(one_word_trade=True)).iloc[0]
        self.assertEqual(result["execution_status"], "unbuyable_one_word_up")
        self.assertTrue(pd.isna(result["entry_price"]))
        self.assertTrue(pd.isna(result["large_loss"]))

    def test_daily_execution_proxy_keeps_one_word_limit_down_unresolved(self):
        result = v3.attach_daily_execution_outcomes(
            self._execution_candidate(), self._execution_bars(one_word_exit_down=True)).iloc[0]
        self.assertEqual(result["execution_status"], "sell_blocked_one_word_down")
        self.assertEqual(result["sell_blocked"], 1.0)
        self.assertTrue(pd.isna(result["net_return"]))
        self.assertLess(result["mark_to_market_return"], -0.10)

    def test_daily_execution_proxy_does_not_overwrite_verified_import(self):
        candidate = self._execution_candidate(
            entry_price=10.5, t1_close=10.7, actual_exit_price=10.6,
            mark_to_market_return=0.016, net_return=0.006,
            execution_status="verified_import", execution_source="user_import")
        result = v3.attach_daily_execution_outcomes(candidate, self._execution_bars()).iloc[0]
        self.assertEqual(result["execution_status"], "verified_import")
        self.assertAlmostEqual(result["entry_price"], 10.5)
        self.assertAlmostEqual(result["net_return"], 0.006)

    def test_terminal_height_label_follows_streak_and_right_censors_live_tail(self):
        frame = pd.DataFrame([
            {"date": "2026-01-01", "trade_date": "2026-01-02", "code": "600001",
             "boards": 2, "promoted_next_day": 1},
            {"date": "2026-01-02", "trade_date": "2026-01-03", "code": "600001",
             "boards": 3, "promoted_next_day": 1},
            {"date": "2026-01-03", "trade_date": "2026-01-06", "code": "600001",
             "boards": 4, "promoted_next_day": 0},
            {"date": "2026-01-03", "trade_date": "", "code": "600002",
             "boards": 3, "promoted_next_day": np.nan},
        ])
        labeled = v3.attach_terminal_height_labels(frame)
        self.assertEqual(labeled.loc[0, "terminal_board"], 4)
        self.assertEqual(labeled.loc[0, "additional_boards"], 2)
        self.assertEqual(labeled.loc[0, "terminal_date"], "2026-01-03")
        self.assertEqual(labeled.loc[2, "terminal_board"], 4)
        self.assertTrue(pd.isna(labeled.loc[3, "terminal_board"]))
        self.assertEqual(labeled.loc[3, "terminal_date"], "")
        self.assertEqual(labeled.loc[3, "terminal_height_censored"], 1)

    @staticmethod
    def _joint_policy_frame(large_losses=0):
        n = 30
        return pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=n, freq="B").strftime("%Y-%m-%d"),
            "boards": [3]*n, "promoted_next_day": [1.0]*n,
            "valid_identity": [1.0]*n, "buyable_eod": [1.0]*n,
            "emotion_state": ["加速"]*n, "announcement_risk": [0.0]*n,
            "entry_price": [10.0]*n,
            "large_loss": [1.0]*large_losses + [0.0]*(n-large_losses),
        })

    def test_joint_execution_threshold_requires_both_promotion_and_loss_targets(self):
        probabilities = np.full(30, 0.9)
        passed = v3.choose_joint_execution_threshold(self._joint_policy_frame(0), probabilities)
        failed = v3.choose_joint_execution_threshold(self._joint_policy_frame(4), probabilities)
        self.assertTrue(passed["operational"])
        self.assertFalse(failed["operational"])
        self.assertIn("large-loss rate", failed["abstain_reason"])

    def test_terminal_height_distribution_is_normalized_and_uses_next_probability(self):
        model = {"maximum_reported_board": 6, "global_promotion_rate": 0.3,
                 "hazards": {"3": {"probability": 0.5}, "4": {"probability": 0.4},
                             "5": {"probability": 0.2}}}
        forecast = v3.terminal_height_distribution(3, 0.8, model)
        self.assertAlmostEqual(sum(forecast["distribution"].values()), 1.0)
        self.assertAlmostEqual(forecast["probability_at_least_one_more"], 0.8)
        self.assertAlmostEqual(forecast["probability_at_least_two_more"], 0.32)
        self.assertEqual(len(forecast["terminal_board_interval_80"]), 2)
        self.assertLessEqual(forecast["terminal_board_interval_80"][0],
                             forecast["terminal_board_interval_80"][1])

    def test_height_training_purges_labels_that_end_after_train_boundary(self):
        frame = pd.DataFrame([
            {"date": "2026-01-01", "terminal_date": "2026-01-03", "boards": 2,
             "promoted_next_day": 1},
            {"date": "2026-01-02", "terminal_date": "2026-01-06", "boards": 3,
             "promoted_next_day": 1},
        ])
        selected = v3.select_height_training_rows(frame, ["2026-01-01", "2026-01-02", "2026-01-03"])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["terminal_date"], "2026-01-03")

    def test_large_loss_is_entry_to_t1_close_after_costs(self):
        frame = pd.DataFrame({"promoted_next_day": [0], "large_loss": [1], "net_return": [-0.083],
                              "date": ["2026-01-01"]})
        metrics = v3.execution_metrics(frame)
        self.assertEqual(metrics["large_loss_rate"], 1.0)
        self.assertLess(metrics["average_trade"], -0.07)

    def test_false_strength_heuristic_rejects_hot_weak_open(self):
        row = pd.Series({"auction_gap": 0.085, "auction_cancel_ratio": 0.6,
                         "one_word_streak": 3, "bomb_count": 2, "theme_sync_open": 0.2,
                         "below_vwap_expanding": 1, "failed_seal_twice": 1,
                         "crowding_score": 0.9, "emotion_state": "大分歧", "boards": 5})
        trap, loss, reasons = v3.heuristic_risks(row)
        self.assertGreater(trap, 0.8)
        self.assertGreater(loss, 0.3)
        self.assertIn("两次冲板不封", reasons)

    def test_unknown_open_data_does_not_become_low_risk(self):
        trap, loss, reasons = v3.heuristic_risks(pd.Series({"auction_gap": np.nan}))
        self.assertIsNone(trap)
        self.assertIsNone(loss)
        self.assertTrue(reasons)

    def test_rejected_risk_model_cannot_drive_live_probability(self):
        artifact = {"status": "REJECTED_NO_OUT_OF_SAMPLE_SKILL_DAILY_PROXY",
                    "model": {"kind": "invalid_should_not_be_loaded"}}
        self.assertIsNone(v3.optional_risk_probability(artifact, pd.DataFrame([{}])))

    def test_training_snapshot_uses_daily_open_only_as_open_stage_proxy(self):
        frame = pd.DataFrame({"date": ["2026-01-01"], "code": ["600001"],
                              "next_open_gap": [0.045]})
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(v3, "DATA", Path(directory)):
                merged = v3.merge_training_snapshots(frame)
        self.assertAlmostEqual(float(merged.iloc[0]["auction_gap"]), 0.045)
        self.assertEqual(merged.iloc[0]["auction_gap_status"], "inferred_from_official_daily_open")
        self.assertTrue(pd.isna(merged.iloc[0]["vwap_reclaimed"]))

    def test_nested_walk_forward_partitions_are_ordered_and_disjoint(self):
        dates = pd.date_range("2025-01-01", periods=60, freq="B").strftime("%Y-%m-%d")
        frame = pd.DataFrame({"date": dates})
        partitions = v3.walk_forward_partitions(frame)
        self.assertGreaterEqual(len(partitions), 2)
        previous_test_end = None
        for partition in partitions:
            train = set(partition["train_dates"])
            calibration = set(partition["calibration_dates"])
            validation = set(partition["validation_dates"])
            test = set(partition["test_dates"])
            self.assertFalse(train & calibration or train & validation or train & test)
            self.assertFalse(calibration & validation or calibration & test or validation & test)
            self.assertLess(max(train), partition["embargo_1"])
            self.assertLess(partition["embargo_1"], min(calibration))
            self.assertLess(max(calibration), partition["embargo_2"])
            self.assertLess(partition["embargo_2"], min(validation))
            self.assertLess(max(validation), partition["embargo_3"])
            self.assertLess(partition["embargo_3"], min(test))
            if previous_test_end is not None:
                self.assertGreater(max(test), previous_test_end)
            previous_test_end = max(test)

    def test_segment_reliability_marks_small_groups_low_confidence(self):
        frame = pd.DataFrame({"boards": [2, 2, 3, 3], "emotion_state": ["升温"] * 4,
                              "promoted_next_day": [0, 1, 0, 1]})
        report = v3.reliability_by_segment(frame, np.array([0.2, 0.8, 0.3, 0.7]))
        self.assertTrue(all(row["confidence"] == "LOW" for row in report["boards"]))
        gate = v3.segment_reliability_gate(report)
        self.assertFalse(gate["passed"])

    def test_online_emotion_forecast_does_not_use_future_states(self):
        states = ["升温", "加速", "加速", "一致", "大分歧", "修复", "升温", "加速"]
        frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=len(states)).strftime("%Y-%m-%d"),
                              "emotion_state": states, "emotion_data_completeness": 1.0})
        changed = frame.copy()
        changed.loc[len(changed) - 1, "emotion_state"] = "退潮"
        first = v3.attach_online_emotion_forecast(frame)
        second = v3.attach_online_emotion_forecast(changed)
        columns = [*v3.STATE_PROBABILITY_COLUMNS.values(), "emotion_transition_evidence",
                   "emotion_transition_entropy", "emotion_state_age"]
        pd.testing.assert_frame_equal(first.iloc[:-1][columns], second.iloc[:-1][columns])

    def test_online_emotion_forecast_exposes_adverse_probability_and_evidence(self):
        frame = pd.DataFrame({"date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                              "emotion_state": ["升温", "加速", "升温"],
                              "emotion_data_completeness": [1.0, 1.0, 1.0]})
        result = v3.attach_online_emotion_forecast(frame)
        last = result.iloc[-1]
        adverse = last["emotion_next_divergence_probability"] + last["emotion_next_retreat_probability"]
        self.assertAlmostEqual(float(last["emotion_adverse_next_probability"]), float(adverse))
        self.assertEqual(int(last["emotion_transition_evidence"]), 1)
        self.assertEqual(last["emotion_forecast_status"], "LOW_EVIDENCE")

    def test_emotion_forecast_backtest_scores_next_state_not_current_state(self):
        states = ["升温", "大分歧", "修复", "退潮", "启动"]
        market = v3.attach_online_emotion_forecast(pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=len(states)).strftime("%Y-%m-%d"),
            "emotion_state": states, "emotion_data_completeness": 1.0,
        }))
        report = v3.emotion_forecast_backtest(market)
        self.assertEqual(report["n"], len(states) - 1)
        self.assertAlmostEqual(report["adverse_rate"], 0.5)
        self.assertEqual(report["status"], "LOW_EVIDENCE")

    def test_supervised_emotion_challenger_waits_for_causal_train_and_calibration_history(self):
        length = 85
        states = ["大分歧" if index % 4 == 1 else "升温" for index in range(length)]
        market = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=length, freq="B").strftime("%Y-%m-%d"),
            "emotion_state": states, "emotion_data_completeness": 1.0,
            "breadth_health": [0.3 if index % 4 == 0 else 0.8 for index in range(length)],
            "relay_appetite": [0.3 if index % 4 == 0 else 0.8 for index in range(length)],
            "ladder_health": 0.6, "loss_effect_health": 0.6, "crowding_risk": 0.3,
            "regime_fragility": [0.8 if index % 4 == 0 else 0.2 for index in range(length)],
            "false_acceleration": 0.0, "promotion_overall": 0.4, "promotion_mid": 0.3,
            "space_feedback": 1.0, "space_feedback_3d": 0.7, "failed_rate": 0.1,
            "down_limit_count": 1.0, "previous_limit_tail_loss": -0.01,
            "yesterday_premium": 0.02, "height_change": 0.0, "limit_change": 2.0,
            "max_board": 5.0, "middle_density": 0.5, "ladder_gap_count": 0.0,
            "one_word_share": 0.1,
        })
        result = v3.attach_online_emotion_forecast(market)
        self.assertTrue(result.iloc[:80]["emotion_adverse_challenger_probability"].isna().all())
        self.assertTrue(result.iloc[80:]["emotion_adverse_challenger_probability"].notna().all())
        self.assertTrue((result.iloc[80:]["emotion_challenger_status"] == "TRAINED_CAUSAL").all())
        self.assertTrue((result.iloc[80:]["emotion_forecast_source"] == "MARKOV_BASELINE").all())
        changed = market.copy()
        changed.loc[length - 1, "emotion_state"] = "退潮"
        changed_result = v3.attach_online_emotion_forecast(changed)
        columns = ["emotion_adverse_challenger_probability", "emotion_forecast_source",
                   "emotion_champion_evaluation_n", "emotion_challenger_vs_markov_brier_skill"]
        pd.testing.assert_frame_equal(result.iloc[:-1][columns], changed_result.iloc[:-1][columns])

    def test_reused_blind_window_is_downgraded_to_development_holdout(self):
        blind = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"],
                              "code": ["600001", "600002"], "boards": [2, 3],
                              "promoted_next_day": [1.0, 0.0]})
        split = {"blind": ["2026-01-01", "2026-01-02"]}
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            (results / "models").mkdir()
            with mock.patch.object(v3, "RESULTS", results):
                fresh = v3.blind_reuse_audit(split, blind)
                (results / "models" / "model_prior.json").write_text(
                    json.dumps({"split": split}), encoding="utf-8")
                reused = v3.blind_reuse_audit(split, blind)
        self.assertEqual(fresh["status"], "FRESH_BLIND")
        self.assertTrue(fresh["deployment_eligible"])
        self.assertEqual(reused["status"], "REUSED_DEVELOPMENT_HOLDOUT")
        self.assertEqual(reused["reuse_count_including_current"], 2)
        self.assertFalse(reused["deployment_eligible"])

    def test_forward_ledger_hash_chain_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / "ledger.jsonl"
            event = v3.append_forward_event({"event_type": "TEST", "value": 1}, ledger, root)
            self.assertEqual(v3.verify_forward_ledger(ledger, root)["status"], "PASS")
            self.assertEqual(event["sequence"], 1)
            ledger.write_text(ledger.read_text(encoding="utf-8").replace('"value":1', '"value":2'),
                              encoding="utf-8")
            verification = v3.verify_forward_ledger(ledger, root)
            self.assertEqual(verification["status"], "FAIL")
            self.assertIn("event hash mismatch at event 1", verification["errors"])

    def test_forward_snapshot_requires_frozen_future_latest_date_and_is_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forward = root / "forward"
            active = forward / "active_freeze.json"
            ledger = forward / "ledger.jsonl"
            model = root / "model.json"
            code = root / "model.py"
            score_json = root / "score.json"
            score_csv = root / "score.csv"
            signals_path = root / "signals.csv"
            outcomes_path = root / "outcomes.csv"
            forward.mkdir()
            model.write_text('{"version":"test"}', encoding="utf-8")
            code.write_text("print('frozen')\n", encoding="utf-8")
            score_json.write_text("{}", encoding="utf-8")
            score_csv.write_text("code,decision\n600001,WATCH\n", encoding="utf-8")
            manifest = {
                "freeze_id": "freeze-test", "data_end": "2026-08-28",
                "candidate_data_end_at_freeze": "2026-08-28", "created_at": "2026-08-29T18:00:00",
                "model_sha256": v3.sha256_file(model), "code_sha256": v3.sha256_file(code),
                "model_deployment_status": "DIAGNOSTIC_ONLY",
            }
            active.write_text(v3.json_text(manifest), encoding="utf-8")
            sealed = forward / "freeze_freeze-test.json"
            sealed.write_text(v3.json_text(manifest), encoding="utf-8")
            v3.append_forward_event({
                "event_type": "MODEL_FREEZE", "freeze_id": "freeze-test",
                "manifest_file": str(sealed.relative_to(root)),
                "manifest_sha256": v3.sha256_file(sealed),
            }, ledger, root)
            candidates = pd.DataFrame([{
                "code": "600001", "name": "甲", "decision": "WATCH",
                "promotion_probability": 0.6, "model_status": "DIAGNOSTIC_ONLY",
            }])
            with self.assertRaisesRegex(ValueError, "later than frozen candidate data end"):
                v3.register_forward_snapshot(
                    "2026-08-28", "09:35", candidates, score_json, score_csv,
                    active, ledger, forward, root, model, code, "2026-08-28",
                    signals_path, outcomes_path)
            result = v3.register_forward_snapshot(
                "2026-08-31", "09:35", candidates, score_json, score_csv,
                active, ledger, forward, root, model, code, "2026-08-31",
                signals_path, outcomes_path)
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(v3.verify_forward_ledger(ledger, root)["status"], "PASS")
            registered = pd.read_csv(signals_path, dtype={"code": str})
            self.assertEqual(registered.iloc[0]["decision"], "WATCH")
            with self.assertRaisesRegex(ValueError, "already preregistered"):
                v3.register_forward_snapshot(
                    "2026-08-31", "09:35", candidates, score_json, score_csv,
                    active, ledger, forward, root, model, code, "2026-08-31",
                    signals_path, outcomes_path)
            active.write_text(v3.json_text({**manifest, "data_end": "2026-01-01"}), encoding="utf-8")
            self.assertEqual(v3.verify_active_freeze_binding(manifest, active, ledger, root)["status"], "FAIL")

    def test_forward_monitor_counts_only_preregistered_executed_enter_signals(self):
        freeze_id = "freeze-test"
        signals = pd.DataFrame({
            "freeze_id": [freeze_id] * 62,
            "decision": ["ENTER"] * 60 + ["WATCH", "REJECT"],
        })
        outcomes = pd.DataFrame({
            "freeze_id": [freeze_id] * 62,
            "decision": ["ENTER"] * 60 + ["WATCH", "REJECT"],
            "execution_status": ["verified_fill"] * 60 + ["verified_fill", "no_trade"],
            "promoted_next_day": [1] * 45 + [0] * 15 + [1, 0],
            "large_loss": [1] * 6 + [0] * 54 + [0, np.nan],
        })
        manifest = {"freeze_id": freeze_id, "model_deployment_status": "HISTORICAL_GATE_PASSED"}
        monitor = v3.forward_monitor(manifest, signals, outcomes, {"status": "PASS"})
        self.assertEqual(monitor["registered_enter_signals"], 60)
        self.assertEqual(monitor["resolved_executable_enter_signals"], 60)
        self.assertAlmostEqual(monitor["promotion_precision"], 0.75)
        self.assertAlmostEqual(monitor["large_loss_rate"], 0.10)
        self.assertTrue(monitor["forward_gate_passed"])
        manifest["model_deployment_status"] = "DIAGNOSTIC_ONLY"
        self.assertFalse(v3.forward_monitor(manifest, signals, outcomes, {"status": "PASS"})["forward_gate_passed"])


if __name__ == "__main__":
    unittest.main()
