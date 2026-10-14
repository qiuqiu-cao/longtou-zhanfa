import unittest
from build_fixed_cohort_v40 import mapping


class CalendarTests(unittest.TestCase):
    def test_weekend_and_boundary(self):
        self.assertEqual(mapping(["2024-01-11","2024-01-12","2024-01-15"],"2024-01-12","2024-01-14"),{"2024-01-12":"2024-01-15"})

    def test_missing_next_session(self):
        with self.assertRaises(ValueError):
            mapping(["2024-01-12"],"2024-01-12","2024-01-12")

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            mapping(["2024-01-12","2024-01-12","2024-01-15"],"2024-01-12","2024-01-12")
