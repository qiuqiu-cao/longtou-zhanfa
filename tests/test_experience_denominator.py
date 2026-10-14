"""Boundary checks for the fixed denominator-state research comparison."""
import unittest

import numpy as np
import pandas as pd

from research.experience_denominator import context_features, sanitize_ratio


class DenominatorStateTests(unittest.TestCase):
    sessions = ["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-08", "2026-10-09"]

    @staticmethod
    def observations(records):
        return pd.DataFrame(records, columns=["date", "code", "turnover", "one_word_proxy"])

    @staticmethod
    def candidates(records):
        rows = pd.DataFrame(records)
        rows["turnover_vs_prev_same_height_percentile"] = .123
        return rows

    def test_last_finite_turnover_skips_missing_and_infinite_records(self):
        observations = self.observations([
            ["2026-09-30", "000001", np.inf, 1.],
            ["2026-09-28", "000001", .1, 0.],
            ["2026-09-29", "000001", np.nan, 1.],
            ["2026-09-30", "000002", .9, 1.],
        ])
        rows = pd.DataFrame([{"date": "2026-10-08", "code": "000001"}])
        row = context_features(rows, observations, self.sessions).iloc[0]
        self.assertEqual(row.prior_date, "2026-09-28")
        self.assertAlmostEqual(row.prior_turnover, .1)
        self.assertEqual(row.prior_one_word_proxy, 0.)
        self.assertFalse(row.prior_is_previous_session)

    def test_current_and_future_observations_cannot_change_prior_context(self):
        rows = pd.DataFrame([
            {"date": "2026-09-30", "code": "000001"},
            {"date": "2026-09-30", "code": "000003"},
        ])
        past = self.observations([["2026-09-29", "000001", .1, 0.]])
        contaminated = pd.concat([past, self.observations([
            ["2026-09-30", "000001", .8, 1.],
            ["2026-10-08", "000001", .9, 1.],
            ["2026-09-30", "000003", .7, 1.],
            ["2026-10-09", "000003", .6, 0.],
        ])], ignore_index=True)
        a = context_features(rows, past, self.sessions)
        b = context_features(rows, contaminated, self.sessions)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a), 2)
        self.assertTrue(pd.isna(a.iloc[1].prior_date))
        self.assertTrue(pd.isna(a.iloc[1].prior_turnover))

    def test_official_previous_session_crosses_the_holiday(self):
        rows = self.candidates([
            {"date": "2026-10-08", "code": "000001", "boards": 3,
             "turnover": .2, "turnover_vs_prev": 2.},
            {"date": "2026-10-08", "code": "000002", "boards": 3,
             "turnover": .2, "turnover_vs_prev": 2.},
        ])
        observations = self.observations([
            ["2026-09-30", "000001", .1, 0.],
            ["2026-09-29", "000002", .1, 0.],
        ])
        context = context_features(rows, observations, self.sessions)
        self.assertTrue(context.official_previous_date.eq("2026-09-30").all())
        result = sanitize_ratio(rows, context)
        self.assertEqual(result.loc[result.code.eq("000001"), "turnover_vs_prev"].item(), 2.)
        self.assertTrue(pd.isna(result.loc[result.code.eq("000002"), "turnover_vs_prev"].item()))

    def test_word_variant_uses_prior_state_and_excludes_unknown(self):
        rows = self.candidates([
            {"date": "2026-09-30", "code": "000001", "boards": 2,
             "turnover": .2, "turnover_vs_prev": 2., "one_word_proxy": 1.},
            {"date": "2026-09-30", "code": "000002", "boards": 2,
             "turnover": .2, "turnover_vs_prev": 2., "one_word_proxy": 0.},
            {"date": "2026-09-30", "code": "000003", "boards": 2,
             "turnover": .2, "turnover_vs_prev": 2., "one_word_proxy": 0.},
        ])
        observations = self.observations([
            ["2026-09-29", "000001", .1, 0.],
            ["2026-09-29", "000002", .1, 1.],
            ["2026-09-29", "000003", .1, np.nan],
        ])
        context = context_features(rows, observations, self.sessions)
        a = sanitize_ratio(rows, context, exclude_word=False)
        b = sanitize_ratio(rows, context, exclude_word=True)
        self.assertTrue(a.turnover_vs_prev.notna().all())
        self.assertEqual(b.loc[b.code.eq("000001"), "turnover_vs_prev"].item(), 2.)
        self.assertTrue(b.loc[b.code.isin(["000002", "000003"]), "turnover_vs_prev"].isna().all())

    def test_zero_turnover_updates_context_but_cannot_be_a_ratio_denominator(self):
        rows = self.candidates([
            {"date": "2026-09-30", "code": "000001", "boards": 2,
             "turnover": .2, "turnover_vs_prev": 2.},
        ])
        observations = self.observations([
            ["2026-09-28", "000001", .1, 0.],
            ["2026-09-29", "000001", 0., 1.],
        ])
        context = context_features(rows, observations, self.sessions)
        self.assertEqual(context.iloc[0].prior_date, "2026-09-29")
        self.assertEqual(context.iloc[0].prior_turnover, 0.)
        result = sanitize_ratio(rows, context)
        self.assertTrue(result.turnover_vs_prev.isna().all())
        self.assertTrue(result.turnover_vs_prev_same_height_percentile.isna().all())

    def test_missing_or_infinite_original_ratio_remains_unknown(self):
        rows = self.candidates([
            {"date": "2026-09-30", "code": "000001", "boards": 2,
             "turnover": .2, "turnover_vs_prev": np.nan},
            {"date": "2026-09-30", "code": "000002", "boards": 2,
             "turnover": .2, "turnover_vs_prev": np.inf},
        ])
        observations = self.observations([
            ["2026-09-29", "000001", .1, 0.],
            ["2026-09-29", "000002", .1, 0.],
        ])
        result = sanitize_ratio(rows, context_features(rows, observations, self.sessions))
        self.assertEqual(len(result), len(rows))
        self.assertTrue(result.turnover_vs_prev.isna().all())
        self.assertTrue(result.turnover_vs_prev_same_height_percentile.isna().all())

    def test_percentile_uses_only_valid_same_height_rows_and_preserves_candidates(self):
        rows = self.candidates([
            {"date": "2026-09-30", "code": "000003", "boards": 2,
             "turnover": .3, "turnover_vs_prev": 3.},
            {"date": "2026-09-30", "code": "000002", "boards": 2,
             "turnover": .2, "turnover_vs_prev": 2.},
            {"date": "2026-09-30", "code": "000001", "boards": 2,
             "turnover": .1, "turnover_vs_prev": 1.},
            {"date": "2026-09-30", "code": "000004", "boards": 3,
             "turnover": .2, "turnover_vs_prev": 2.},
        ])
        for i in range(27):
            rows[f"untouched_feature_{i}"] = np.arange(len(rows)) + i / 10
        rows["target"] = [1., np.nan, 0., 1.]
        rows["label_valid"] = [True, False, True, True]
        before = rows.copy(deep=True)
        observations = self.observations([
            ["2026-09-29", "000001", .1, 0.],
            ["2026-09-28", "000002", .1, 0.],
            ["2026-09-29", "000003", .1, 0.],
            ["2026-09-29", "000004", .1, 0.],
        ])
        result = sanitize_ratio(rows, context_features(rows, observations, self.sessions))
        ranked = result.set_index("code").turnover_vs_prev_same_height_percentile
        self.assertEqual(ranked["000001"], .5)
        self.assertEqual(ranked["000003"], 1.)
        self.assertEqual(ranked["000004"], 1.)
        self.assertTrue(pd.isna(ranked["000002"]))
        protected = [c for c in rows if c not in {
            "turnover_vs_prev", "turnover_vs_prev_same_height_percentile"}]
        pd.testing.assert_frame_equal(result[protected].reset_index(drop=True),
                                      before[protected].reset_index(drop=True))
        pd.testing.assert_frame_equal(rows, before)

    def test_valid_original_ratio_is_retained_without_reconstruction_rounding(self):
        original = 2.0000001
        rows = self.candidates([
            {"date": "2026-09-30", "code": "000001", "boards": 2,
             "turnover": .2, "turnover_vs_prev": original},
        ])
        observations = self.observations([["2026-09-29", "000001", .1, 0.]])
        result = sanitize_ratio(rows, context_features(rows, observations, self.sessions))
        self.assertEqual(result.turnover_vs_prev.item(), original)


if __name__ == "__main__":
    unittest.main()
