import copy
import unittest
from leader_competition_v38 import compare


def snapshot(time="09:35", sealed="sealed"):
    return dict(observed_at="2026-09-04T"+time+":00+08:00", available_at="2026-09-04T"+time+":00+08:00",
                scope_id="synthetic-specific-theme", source="synthetic-test", data_status="verified", universe_complete=True,
                candidates=[dict(code="000001", boards=4, seal_state=sealed), dict(code="000002", boards=3, seal_state="sealed")])


class CompetitionTests(unittest.TestCase):
    def test_earlier_observation_received_after_decision_rejected(self):
        now = snapshot("09:40")
        previous = snapshot("09:39")
        previous["available_at"] = snapshot("09:41")["available_at"]
        with self.assertRaises(ValueError):
            compare("000001", now, now["available_at"], previous)

    def test_opened_not_identity_override(self):
        now = snapshot("09:40", "open")
        r = compare("000001", now, now["available_at"], snapshot())
        self.assertEqual(r["events"][0]["type"], "SUBJECT_OPENED")
        self.assertIsNone(r["leader_identity_override"])
        self.assertFalse(r["automatic_trade_action"])

    def test_future_rejected(self):
        with self.assertRaises(ValueError):
            compare("000001", snapshot("09:40"), snapshot()["available_at"])

    def test_stale_limited(self):
        self.assertEqual(compare("000001", snapshot(), snapshot("09:41")["available_at"])["evidence_state"], "DATA_LIMITED")

    def test_incomplete_limited(self):
        now = snapshot(); now["universe_complete"] = False
        self.assertEqual(compare("000001", now, now["available_at"])["evidence_state"], "DATA_LIMITED")

    def test_missing_peer_not_failed(self):
        now = snapshot("09:40"); now["candidates"].pop()
        r = compare("000001", now, now["available_at"], snapshot())
        self.assertEqual(r["events"][0]["type"], "PEER_LEFT_SCOPE")

    def test_new_higher(self):
        now = snapshot("09:40"); now["candidates"][1]["boards"] = 5
        r = compare("000001", now, now["available_at"], snapshot())
        self.assertEqual(r["events"][0]["type"], "NEW_HIGHER_BOARD_PEER")

    def test_scope_changed(self):
        now = snapshot("09:40"); now["scope_id"] = "different"
        self.assertEqual(compare("000001", now, now["available_at"], snapshot())["evidence_state"], "SCOPE_CHANGED_NOT_COMPARABLE")
