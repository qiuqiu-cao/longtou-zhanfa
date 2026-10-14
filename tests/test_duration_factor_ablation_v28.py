import unittest
import pandas as pd
from duration_factor_ablation_v28 import GROUPS, top1_without_labels, selected_summary


class DurationFactorTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame(dict(code=["000001", "000002", "000001"],
                                 signal_date=["2025-07-28", "2025-07-28", "2025-07-29"],
                                 episode_id=["a", "b", "a"], label_continue_1=[None, 1, 1]))

    def test_unresolved_top_not_replaced_by_known_winner(self):
        pool = self.fixture()
        top = top1_without_labels(pool, [.9, .5, .8])
        self.assertEqual(top.iloc[0].code, "000001")
        summary = selected_summary(top, pool, 1)
        self.assertEqual(summary["distinct_selected_episodes"], 1)
        self.assertEqual(summary["unresolved_episodes"], 1)
        self.assertEqual(summary["hits"], 0)

    def test_labels_do_not_change_ranking(self):
        pool = self.fixture()
        first = top1_without_labels(pool, [.9, .5, .8])
        pool["label_continue_1"] = [0, 0, 0]
        pd.testing.assert_frame_equal(first, top1_without_labels(pool, [.9, .5, .8]))

    def test_ties_use_code(self):
        self.assertEqual(top1_without_labels(self.fixture(), [.5, .5, .5]).iloc[0].code, "000001")

    def test_feature_groups_exclude_future_and_old_forecasts(self):
        forbidden = ("label", "duration", "target", "probability", "termination")
        self.assertFalse(any(any(x in column for x in forbidden) for columns in GROUPS.values() for column in columns))
        self.assertEqual(GROUPS["identity_plus_emotion"][:len(GROUPS["identity_proxy"])], GROUPS["identity_proxy"])


if __name__ == "__main__":
    unittest.main()
