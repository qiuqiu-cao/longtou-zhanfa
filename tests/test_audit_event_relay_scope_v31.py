import unittest
import pandas as pd
from audit_event_relay_scope_v31 import audit


class ScopeTests(unittest.TestCase):
    def fixture(self):
        days = pd.bdate_range("2024-01-01", periods=8).strftime("%Y-%m-%d").tolist()
        events = [dict(code="000001", event_date=days[7], target_pct=200, official_confirmed=True,
                       official_evidence=[dict(start=days[0])], evidence_available_at="2024-01-11T23:59:59+08:00")]
        obs = pd.DataFrame([dict(code="000001", signal_date=days[1], boards=2, episode_id="old", start_known=True),
                            dict(code="000001", signal_date=days[6], boards=2, episode_id="new", start_known=True),
                            dict(code="000001", signal_date=days[7], boards=3, episode_id="new", start_known=True)])
        eps = pd.DataFrame([dict(code="000001", episode_id="old", start_date=days[0], last_limit_date=days[1]),
                            dict(code="000001", episode_id="new", start_date=days[5], last_limit_date=days[7])])
        return events, obs, eps, days

    def test_disconnected_not_success(self):
        rows = audit(*self.fixture())["rows"]
        self.assertEqual(rows[0]["scope_status"], "DIFFERENT_RELAY_NOT_SUCCESS_TO_THIS_EVENT")
        self.assertTrue(rows[1]["scope_status"].startswith("SAME_RELAY_POTENTIAL"))
        self.assertEqual(rows[2]["scope_status"], "EVENT_DAY_NOT_FUTURE_TARGET")
        self.assertTrue(all(r["training_label"] is None for r in rows))

    def test_unverified_event_not_promoted(self):
        args = self.fixture()
        args[0][0]["official_confirmed"] = False
        self.assertEqual(audit(*args)["rows"], [])

    def test_unknown_start_retained(self):
        events, obs, eps, days = self.fixture()
        obs.loc[1, "start_known"] = False
        self.assertEqual(audit(events, obs, eps, days)["rows"][1]["scope_status"], "UNKNOWN_RELAY_START")

    def test_overlapping_episodes_rejected(self):
        events, obs, eps, days = self.fixture()
        eps.loc[0, "last_limit_date"] = days[7]
        with self.assertRaises(ValueError):
            audit(events, obs, eps, days)


if __name__ == "__main__":
    unittest.main()
