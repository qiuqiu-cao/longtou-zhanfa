import unittest
from unittest.mock import patch

import numpy as np

import nonword_pairwise_ranker_v282 as ranker
from test_nonword_specialist_v281 import history


def mixed_history():
    rows, cutoff = history()
    rows["daily_true_one_word"] = 0
    rows.loc[rows.code.eq("000002"), "target"] = 1-rows.loc[rows.code.eq("000002"), "target"]
    return rows, cutoff


class PairwiseTests(unittest.TestCase):
    def test_learns_prior_within_date_order(self):
        rows, cutoff = mixed_history()
        rows["target"] = rows.code.eq("000001").astype(int)
        rows["signal_feature"] = np.where(rows.code.eq("000001"), .8, .2)
        known = rows[rows.target_date.lt(cutoff)]
        model = ranker.Ranker(["signal_feature"], .25).fit(known)
        scores = model.score(rows[rows.date.eq(cutoff)])
        self.assertGreater(scores[0], scores[1])
        self.assertEqual(model.training_audit["mixed_dates"], known.date.nunique())
        self.assertLess(model.training_audit["max_label_date"], cutoff)

    def test_symmetric_pairs_keep_each_date_total_weight_one(self):
        rows, cutoff = mixed_history()
        rows = rows[rows.target_date.lt(cutoff)]
        model = ranker.Ranker(["signal_feature"], .25)
        with patch.object(model.model, "fit") as mocked:
            model.fit(rows)
        kwargs = mocked.call_args.kwargs
        weight = kwargs["sample_weight"]
        self.assertAlmostEqual(weight.sum(), rows.date.nunique())
        y = mocked.call_args.args[1]
        self.assertAlmostEqual(weight[y == 1].sum(), weight[y == 0].sum())

    def test_no_word_domain_removal_or_old_probability_inheritance(self):
        rows, cutoff = mixed_history()
        model = ranker.Ranker(["signal_feature"], .25).fit(rows[rows.target_date.lt(cutoff)])
        test = rows[rows.date.eq(cutoff)].copy()
        test["probability"] = .99
        test.loc[test.code.eq("000002"), "daily_true_one_word"] = 1
        p = ranker.predict(model, test)
        self.assertEqual(len(p), len(test))
        self.assertNotIn("probability", p)
        self.assertTrue(p.loc[p.code.eq("000002"), "ranking_score"].isna().all())
        self.assertEqual(p.ranking_first.sum(), 1)

    def test_future_fields_rejected_and_unknown_labels_not_trainable(self):
        with self.assertRaisesRegex(ValueError, "future feature"):
            ranker.Ranker(["net_return"], .25)
        rows, _ = mixed_history()
        rows.loc[0, "target"] = np.nan
        with self.assertRaisesRegex(ValueError, "known nonword"):
            ranker.Ranker(["signal_feature"], .25).fit(rows)

    def test_post_cutoff_labels_do_not_change_selection_or_refit(self):
        rows, cutoff = mixed_history()
        changed = rows.copy()
        changed.loc[changed.target_date.ge(cutoff), "target"] = 1-changed.loc[changed.target_date.ge(cutoff), "target"]
        first, first_log = ranker.frozen(rows, ["signal_feature"], cutoff)
        second, second_log = ranker.frozen(changed, ["signal_feature"], cutoff)
        self.assertEqual(first_log, second_log)
        np.testing.assert_allclose(first.model.coef_, second.model.coef_)

    def test_ranking_choice_uses_signal_score_not_target(self):
        rows, cutoff = mixed_history()
        rows["target"] = rows.code.eq("000001").astype(int)
        rows["signal_feature"] = np.where(rows.code.eq("000001"), .8, .2)
        model = ranker.Ranker(["signal_feature"], .25).fit(rows[rows.target_date.lt(cutoff)])
        test = rows[rows.date.eq(cutoff)].copy()
        before = ranker.predict(model, test)
        test["target"] = 1-test.target
        after = ranker.predict(model, test)
        self.assertEqual(before.loc[before.ranking_first, "code"].tolist(),
                         after.loc[after.ranking_first, "code"].tolist())


if __name__ == "__main__":
    unittest.main()
