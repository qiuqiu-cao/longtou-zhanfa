import unittest
from leader_holding_review_v21 import review


class HoldingTests(unittest.TestCase):
    def fixture(self):
        position = dict(position_id="paper-1", code="000017", entry_at="2026-09-03T09:35:00+08:00",
                        entry_price=10, shares=100)
        snapshot = dict(code="000017", observed_at="2026-09-04T09:35:00+08:00",
                        available_at="2026-09-04T09:35:10+08:00", data_status="verified",
                        source="test_fixture", price=11, leader_identity="intact", indicator_status="verified",
                        deviation_10d_pct=90, deviation_30d_pct=180, sellability="sellable",
                        emotion_state="退潮", theme_feedback="weakening", notice_summary="普通异动")
        return position, snapshot, "2026-09-04T09:36:00+08:00"

    def test_ordinary_notice_does_not_exit_but_environment_is_recorded(self):
        result = review(*self.fixture())
        self.assertEqual(result["decision"], "HOLD_REVIEW")
        self.assertIn("market_negative_feedback_expectation_downgrade", result["reasons"])
        self.assertIsNone(result["expected_remaining_holding_sessions"])

    def test_target_exit_not_filled_when_blocked(self):
        p, s, t = self.fixture()
        s.update(deviation_10d_pct=100, sellability="locked_limit_down")
        result = review(p, s, t)
        self.assertEqual(result["decision"], "EXIT_PLAN")
        self.assertEqual(result["execution_state"], "SELL_BLOCKED")
        self.assertFalse(result["order_submitted"])

    def test_same_day_exit_is_t1_locked(self):
        p, s, t = self.fixture()
        p["entry_at"] = "2026-09-04T09:30:00+08:00"
        s["leader_identity"] = "lost"
        result = review(p, s, t)
        self.assertEqual(result["execution_state"], "T1_LOCKED")

    def test_stale_data_does_not_repeat_hold(self):
        p, s, t = self.fixture()
        self.assertEqual(review(p, s, "2026-09-04T10:00:00+08:00")["decision"], "DATA_LIMITED")

    def test_future_data_rejected(self):
        p, s, t = self.fixture()
        s["available_at"] = "2026-09-04T10:00:00+08:00"
        with self.assertRaises(ValueError):
            review(p, s, t)


if __name__ == "__main__":
    unittest.main()
