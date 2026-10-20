"""Relevant checks for denominator and calendar-axis corrections; no fitting."""
import importlib.util
import unittest
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('saved_review', HERE / 'review_saved.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class SavedReviewTests(unittest.TestCase):
    def test_unknown_top_pick_does_not_get_replaced_by_known_winner(self):
        rows = pd.DataFrame([
            dict(date='2026-09-01', code='000001', target=None, label_valid=False, new_score=.9),
            dict(date='2026-09-01', code='000002', target=1., label_valid=True, new_score=.8),
        ])
        top = review.top(rows, 'new_score', 1, 1)
        self.assertEqual(top['selected'], 1)
        self.assertEqual(top['unknown'], 1)
        self.assertEqual(top['hits'], 0)
        self.assertIsNone(top['known_pick_hit_rate'])

    def test_calendar_positions_include_unknown_and_absent_signal_days(self):
        sessions = ['2026-09-01', '2026-09-02', '2026-09-03', '2026-09-04', '2026-09-07']
        rows = pd.DataFrame([
            dict(date=sessions[0], target=1., label_valid=True, new_score=.8, roll_score=.6),
            dict(date=sessions[2], target=None, label_valid=False, new_score=.1, roll_score=.9),
            dict(date=sessions[4], target=0., label_valid=True, new_score=.1, roll_score=.1),
        ])
        result = review.bootstrap_official(rows, sessions, sessions[0], sessions[-1])
        self.assertEqual(result['official_sessions'], 5)
        self.assertEqual(result['candidate_signal_dates'], 3)
        self.assertEqual(result['known_contribution_dates'], 2)
        self.assertEqual(result['dates_without_known_contribution'], 3)
        self.assertEqual(result['lower'], 0.)
        self.assertEqual(result['upper'], 0.)

    def test_all_unknown_outcomes_are_not_failures_or_zero_rate(self):
        rows = pd.DataFrame([dict(date='2026-09-01', target=None, label_valid=False, new_score=.9, roll_score=.1)])
        result = review.metric(rows, 'new_score')
        self.assertEqual(result['unknown'], 1)
        self.assertEqual(result['TN'], 0)
        self.assertIsNone(result['Accuracy'])
        interval = review.bootstrap_official(rows, ['2026-09-01'], '2026-09-01', '2026-09-01')
        self.assertIsNone(interval['lower'])
        self.assertEqual(interval['undefined_draws'], 2000)


if __name__ == '__main__':
    unittest.main()
