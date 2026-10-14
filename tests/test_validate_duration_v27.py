import unittest
import pandas as pd
from validate_duration_v27 import horizon_label, metrics, score_pool


class DurationValidationTests(unittest.TestCase):
    def test_break_on_horizon_is_failure(self):
        self.assertEqual(horizon_label(dict(duration_sessions=3, broken_observed=True), 3), 0)

    def test_censor_on_horizon_is_known_success(self):
        self.assertEqual(horizon_label(dict(duration_sessions=3, broken_observed=False), 3), 1)

    def test_censor_before_horizon_remains_unknown(self):
        self.assertIsNone(horizon_label(dict(duration_sessions=2, broken_observed=False), 3))
        self.assertIsNone(horizon_label(dict(duration_sessions=None), 3))

    def fixture(self):
        obs = pd.DataFrame([dict(code="000001", signal_date="2025-07-28", boards=2, start_known=True),
                            dict(code="000002", signal_date="2025-07-28", boards=2, start_known=False)])
        model = dict(training_exclusive_cutoff="2025-06-16", validation_start="2025-07-28",
                     per_height={"2": {"curve": [dict(future_session=1, relay_survival_estimate=.3)]}})
        return obs, model

    def test_unknown_candidates_not_dropped_or_scored(self):
        obs, model = self.fixture()
        pool = score_pool(obs, model, "2025-07-28", "2025-12-26")
        self.assertEqual(len(pool), 2)
        self.assertEqual(pool.iloc[0].p_continue_1_NOT_LIVE, .3)
        self.assertTrue(pd.isna(pool.iloc[1].p_continue_1_NOT_LIVE))

    def test_unknown_is_not_failure_in_metrics(self):
        frame = pd.DataFrame(dict(p_continue_1_NOT_LIVE=[.3, .3], label_continue_1=[1, None]))
        result = metrics(frame, 1)
        self.assertEqual(result["evaluated_rows"], 1)
        self.assertEqual(result["all_candidate_rate_lower"], .5)
        self.assertEqual(result["all_candidate_rate_upper"], 1)

    def test_training_overlap_rejected(self):
        obs, model = self.fixture()
        model["training_exclusive_cutoff"] = "2025-07-29"
        with self.assertRaises(ValueError):
            score_pool(obs, model, "2025-07-28", "2025-12-26")

    def test_unknown_height_is_not_extrapolated(self):
        obs, model = self.fixture()
        obs.loc[0, "boards"] = 25
        self.assertTrue(score_pool(obs, model, "2025-07-28", "2025-12-26").p_continue_1_NOT_LIVE.isna().all())


if __name__ == "__main__":
    unittest.main()
