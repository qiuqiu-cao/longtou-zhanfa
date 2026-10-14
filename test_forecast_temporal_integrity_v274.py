import unittest

import numpy as np
import pandas as pd

from rolling_forecast_challenger_v274 import known_before, trailing, fit
from reported_seal_archive_v275 import clock, sessions, normalize


class CausalForecastTests(unittest.TestCase):
    def test_label_on_freeze_date_is_purged(self):
        frame = pd.DataFrame({"date": ["2025-06-27", "2025-06-30", "2025-07-01"],
            "target_date": ["2025-06-30", "2025-07-01", "2025-07-02"], "label_valid": [True]*3, "target": [1, 0, 1]})
        known = known_before(frame, "2025-07-01")
        self.assertEqual(known.date.tolist(), ["2025-06-27"])
        frame.loc[frame.date >= "2025-06-30", "target"] = 0
        self.assertTrue(known.equals(known_before(frame, "2025-07-01")))

    def test_trailing_window_is_dates_not_rows(self):
        frame = pd.DataFrame({"date": ["2025-01-01", "2025-01-02", "2025-01-02", "2025-01-03"]})
        self.assertEqual(trailing(frame, 2).date.tolist(), ["2025-01-02", "2025-01-02", "2025-01-03"])

    def test_future_field_cannot_be_fit(self):
        with self.assertRaises(ValueError):
            fit(pd.DataFrame(), ["next_open_gap"], "forest", "relay")

    def test_zero_placeholder_and_lunch_clock_rejected(self):
        for value in [0, "000000", "122500", "092700", "250000", "093099", None]:
            self.assertTrue(np.isnan(clock(value)), value)
        self.assertEqual(clock("092500"), 9*3600+25*60)
        self.assertEqual(clock("14:59:59"), 14*3600+59*60+59)

    def test_holidays_are_not_skipped_to_observed_snapshot(self):
        cal = sessions()
        self.assertEqual(cal[cal.index("2026-09-24")+1], "2026-09-28")
        self.assertEqual(cal[cal.index("2026-09-30")+1], "2026-10-08")
        self.assertNotIn("2026-09-25", cal)


if __name__ == "__main__":
    unittest.main()
