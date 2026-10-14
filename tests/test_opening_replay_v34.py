import unittest

from opening_replay_v34 import observe, next_bar_diagnostic


def bar(time, low=10, high=10, close=10, volume=100):
    return dict(time=time, open=close, high=high, low=low, close=close, volume=volume)


class OpeningReplayTests(unittest.TestCase):
    def test_future_cannot_change_observation(self):
        prefix = [bar("0935")]
        expected = observe(prefix, "0935", 10)
        for future in (bar("0940", 8, 10, 8), bar("0940"), bar("0940", close=float("nan"))):
            self.assertEqual(observe(prefix+[future], "0935", 10), expected)

    def test_not_available_before_bar_end(self):
        self.assertEqual(observe([bar("0935")], "0934", 10)["state"], "NO_COMPLETED_BAR")

    def test_touch_is_not_close_at_limit(self):
        self.assertEqual(observe([bar("0935", 9, 10, 9)], "0935", 10)["state"], "TOUCHED_THEN_BELOW_PROXY")

    def test_observation_is_not_buy_permission(self):
        self.assertEqual(observe([bar("0935")], "0935", 10)["decision"], "WATCH")

    def test_duplicate_observed_times_rejected(self):
        with self.assertRaises(ValueError):
            observe([bar("0935"), bar("0935")], "0935", 10)

    def test_next_bar_only(self):
        rows = [bar("0935", 8, 10, 10), bar("0940"), bar("0945", 8, 10, 8)]
        result = next_bar_diagnostic(rows, "0935", 10)
        self.assertEqual(result["state"], "LIMIT_QUEUE_UNRESOLVED")
        self.assertFalse(result["fill_verified"])

    def test_price_cross_does_not_assign_fill(self):
        result = next_bar_diagnostic([bar("0940", 9, 10, 9)], "0935", 10)
        self.assertEqual(result["state"], "BELOW_LIMIT_TRADES_OPPORTUNITY_ONLY")
        self.assertIsNone(result["simulated_entry_price"])
        self.assertTrue(result["label_only"])

    def test_no_volume_and_missing(self):
        self.assertEqual(next_bar_diagnostic([bar("0940", volume=0)], "0935", 10)["state"], "NO_RECORDED_VOLUME")
        self.assertEqual(next_bar_diagnostic([], "0935", 10)["state"], "NO_NEXT_BAR")
