import unittest

import pandas as pd

from full_pool_proxy_backtest import EMOTION, candidate_summary, prepare, select_top1, summarize


class FullPoolProxyBacktestTests(unittest.TestCase):
    def features(self):
        base = dict(
            signal_date="2025-01-02", episode_id="e", feature_status="historical_reconstruction_not_forward_archive",
            feature_available_not_before="2025-01-02T15:05:00+08:00", candidate_pool_count=2,
            unknown_height_pool_count=0, known_pool_max_boards=2, height_gap_to_known_top=1,
            same_height_known_count=1, sole_known_top_proxy=0, open_gap_pct=1, amplitude_pct=2,
            amount_to_float_cap=.1, turnover=3, one_price_day_proxy=0, market_limit_count=30,
            market_consecutive_count=5, market_failed_rate=.2, market_seal_rate=.8,
            market_yesterday_premium=.01, market_previous_limit_tail_loss=-.02,
            market_promotion_overall=.3, market_promotion_mid=.2, market_space_feedback=.01,
            market_middle_density=.1, market_ladder_gap_count=0,
        )
        for column in EMOTION:
            base.setdefault(column, .1)
        return pd.DataFrame([
            dict(base, code="1", boards=1, board_group="FIRST_BOARD"),
            dict(base, code="2", boards=2, board_group="CONSECUTIVE_BOARD", episode_id="f",
                 height_gap_to_known_top=0, sole_known_top_proxy=1),
            dict(base, code="3", boards=None, board_group="UNKNOWN_LEFT_CENSORED",
                 episode_id="g", feature_status="left_censored_identity"),
        ])

    def outcomes(self, one_price=False):
        return pd.DataFrame([
            dict(code="000001", signal_date="2025-01-02", exit_date_LABEL_ONLY="2025-01-06",
                 price_path_state="PRICE_OPPORTUNITY_PROXY_ONLY", net_close_mark_LABEL_ONLY=.03,
                 entry_one_price_LABEL_ONLY=one_price),
            dict(code="000002", signal_date="2025-01-02", exit_date_LABEL_ONLY="2025-01-06",
                 price_path_state="PRICE_OPPORTUNITY_PROXY_ONLY", net_close_mark_LABEL_ONLY=None,
                 entry_one_price_LABEL_ONLY=False),
            dict(code="000003", signal_date="2025-01-02", exit_date_LABEL_ONLY="2025-01-06",
                 price_path_state="PRICE_OPPORTUNITY_PROXY_ONLY", net_close_mark_LABEL_ONLY=.05,
                 entry_one_price_LABEL_ONLY=False),
        ])

    def test_prepare_excludes_left_censored_identity(self):
        result = prepare(self.features(), self.outcomes())
        self.assertEqual(set(result.code), {"000001"})

    def test_missing_touch_evidence_is_unknown(self):
        result = prepare(self.features(), self.outcomes())
        self.assertNotIn("000002", set(result.code))

    def test_explicit_no_touch_is_retained_as_negative_full_pool_candidate(self):
        outcomes = self.outcomes()
        outcomes["b1_net_close_mark_LABEL_ONLY"] = [.01, None, None]
        outcomes["b2_net_close_mark_LABEL_ONLY"] = [.03, None, None]
        outcomes["next_session_touched_limit_LABEL_ONLY"] = [True, False, None]
        result = prepare(self.features(), outcomes)
        no_touch = result[result.code.eq("000002")].iloc[0]
        self.assertEqual(no_touch.opportunity_proxy_label, 0)
        self.assertFalse(no_touch.next_session_touched_limit_LABEL_ONLY)
        self.assertTrue(result.attrs["explicit_touch_field_used"])
        self.assertTrue(result.attrs["no_touch_rows_retained_as_negative"])

    def test_missing_b2_mark_is_not_a_negative_label(self):
        outcomes = self.outcomes()
        outcomes["b1_net_close_mark_LABEL_ONLY"] = [.01, .02, None]
        outcomes["b2_net_close_mark_LABEL_ONLY"] = [None, .04, None]
        result = prepare(self.features(), outcomes)
        self.assertEqual(set(result.code), {"000002"})
        self.assertEqual(result.iloc[0].opportunity_target_horizon, "B+2")
        self.assertEqual(result.attrs["target_unknown_rows_excluded"], 1)
        self.assertEqual(result.attrs["target_known_rows"], 1)
        self.assertEqual(
            result.attrs["target_unknown_rows_excluded"] + result.attrs["target_known_rows"],
            result.attrs["identity_resolved_rows"],
        )

    def test_missing_touch_mark_is_not_a_negative_label(self):
        outcomes = self.outcomes()
        outcomes["b1_net_close_mark_LABEL_ONLY"] = [None, .02, .05]
        outcomes["b2_net_close_mark_LABEL_ONLY"] = [.04, .04, .06]
        result = prepare(self.features(), outcomes)
        self.assertEqual(set(result.code), {"000002"})
        self.assertEqual(result.attrs["target_unknown_rows_excluded"], 1)

    def test_one_price_touch_is_not_positive(self):
        result = prepare(self.features(), self.outcomes(one_price=True))
        self.assertEqual(result[result.code.eq("000001")].iloc[0].opportunity_proxy_label, 0)

    def test_selects_exactly_one_per_day_deterministically(self):
        frame = pd.DataFrame([
            {"signal_date": "2025-01-02", "code": "000002", "boards": 2, "score": .7},
            {"signal_date": "2025-01-02", "code": "000001", "boards": 1, "score": .7},
            {"signal_date": "2025-01-03", "code": "000003", "boards": 1, "score": .6},
        ])
        result = select_top1(frame, "score")
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0].code, "000002")

    def test_future_feature_is_rejected(self):
        features = self.features()
        features.loc[0, "feature_available_not_before"] = "2025-01-03T09:00:00+08:00"
        with self.assertRaisesRegex(ValueError, "availability"):
            prepare(features, self.outcomes())

    def test_candidate_summary_is_not_complete_trade_metric(self):
        rows = pd.DataFrame({"opportunity_proxy_label": [0, 1], "score": [.2, .8]})
        result = candidate_summary(rows, "score")
        self.assertEqual(result["proxy_base_rate"], .5)
        self.assertEqual(result["candidate_auc"], 1.)

    def test_path_risk_summary_has_uncertainty_intervals(self):
        rows = pd.DataFrame({
            "opportunity_proxy_label": [0, 1], "score": [.2, .8],
            "b1_net_close_mark_LABEL_ONLY": [-.08, .02],
            "b2_net_close_mark_LABEL_ONLY": [-.09, .03],
            "b3_net_close_mark_LABEL_ONLY": [-.1, .04],
            "board_group": ["FIRST_BOARD", "CONSECUTIVE_BOARD"],
            "boards": [1, 2], "signal_date": ["2025-01-02", "2025-01-03"],
            "next_session_one_price_LABEL_ONLY": [False, False],
        })
        result = summarize(rows, "score")
        self.assertEqual(len(result["path_proxy"]["b2"]["large_loss_rate_wilson_95"]), 2)

    def test_breakdowns_preserve_first_and_relay_board_levels(self):
        rows = pd.DataFrame({
            "opportunity_proxy_label": [0, 1, 1],
            "score": [.2, .8, .7],
            "board_group": ["FIRST_BOARD", "CONSECUTIVE_BOARD", "CONSECUTIVE_BOARD"],
            "boards": [1, 2, 3],
            "signal_date": ["2025-01-02"] * 3,
            "code": ["000001", "000002", "000003"],
            "next_session_one_price_LABEL_ONLY": [False] * 3,
        })
        candidate = candidate_summary(rows, "score")
        selected = summarize(rows, "score")
        self.assertEqual(set(candidate["by_board_group"]), {"FIRST_BOARD", "CONSECUTIVE_BOARD"})
        self.assertEqual(set(candidate["by_board_level"]), {"1", "2", "3"})
        self.assertEqual(selected["by_board_group"]["FIRST_BOARD"]["selected_sessions"], 1)


if __name__ == "__main__":
    unittest.main()
