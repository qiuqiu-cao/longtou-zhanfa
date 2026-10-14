import unittest
import pandas as pd
from leader_duration_baseline_v27 import duration_rows, survival_curve


class DurationTests(unittest.TestCase):
    def fixture(self):
        dates = pd.bdate_range("2025-01-01", periods=6).strftime("%Y-%m-%d").tolist()
        obs = pd.DataFrame([dict(code="000001", episode_id="a", signal_date=dates[1], boards=2, start_known=True)])
        eps = pd.DataFrame([dict(code="000001", episode_id="a", last_limit_date=dates[3], terminal_date=dates[4], termination="BROKEN_BOARD")])
        return obs, eps, dates

    def test_failure_included(self):
        obs, eps, dates = self.fixture()
        rows, _ = duration_rows(obs, eps, dates, dates[5])
        self.assertEqual(rows.iloc[0].duration_sessions, 3)
        self.assertTrue(rows.iloc[0].broken_observed)
        self.assertEqual(survival_curve(rows, 4)["restricted_mean_future_limit_sessions"], 2)

    def test_boundary_is_censor_not_failure(self):
        obs, eps, dates = self.fixture()
        rows, _ = duration_rows(obs, eps, dates, dates[3])
        self.assertFalse(rows.iloc[0].broken_observed)
        self.assertEqual(rows.iloc[0].duration_sessions, 1)
        self.assertIsNone(survival_curve(rows, 3)["restricted_mean_future_limit_sessions"])

    def test_future_event_change_does_not_change_training(self):
        obs, eps, dates = self.fixture()
        left, _ = duration_rows(obs, eps, dates, dates[3])
        eps.loc[0, "last_limit_date"] = dates[4]
        eps.loc[0, "terminal_date"] = dates[5]
        right, _ = duration_rows(obs, eps, dates, dates[3])
        pd.testing.assert_frame_equal(left, right)

    def test_unknown_not_zero_duration(self):
        obs, eps, dates = self.fixture()
        obs.loc[0, "start_known"] = False
        rows, excluded = duration_rows(obs, eps, dates, dates[5])
        self.assertTrue(rows.empty)
        self.assertEqual(excluded, 1)

    def test_tied_censor_remains_in_risk_set(self):
        rows = pd.DataFrame(dict(episode_id=["a", "b"], duration_sessions=[1, 1], broken_observed=[True, False]))
        self.assertEqual(survival_curve(rows, 1)["curve"][0]["relay_survival_estimate"], .5)

    def test_duplicate_episode_rejected(self):
        rows = pd.DataFrame(dict(episode_id=["a", "a"], duration_sessions=[1, 2], broken_observed=[True, True]))
        with self.assertRaises(ValueError):
            survival_curve(rows)


if __name__ == "__main__":
    unittest.main()
