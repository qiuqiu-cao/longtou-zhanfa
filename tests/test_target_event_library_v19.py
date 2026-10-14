import unittest
import pandas as pd
from target_event_library_v19 import merge_events, known_official_events


class EventLibraryTests(unittest.TestCase):
    def fixture(self):
        scan = pd.DataFrame([dict(code="000017", date="2024-01-17", cross_100_proxy=True, cross_200_proxy=False)])
        case = dict(code="000017", start="2024-01-04", end="2024-01-17", publication_date="2024-01-20",
                    threshold_pct=100, result="THRESHOLD_DIRECTION_MATCH",
                    evidence_status="official_company_full_text_read", source="https://example.test/official",
                    calculated_deviation_pct=100.07)
        return scan, case

    def test_confirmation_does_not_duplicate_or_create_trade_label(self):
        scan, case = self.fixture()
        events = merge_events(scan, [case])
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["official_confirmed"])
        self.assertFalse(events[0]["training_label_ready"])

    def test_second_round_missing_from_scan_is_added(self):
        scan, case = self.fixture()
        case.update(code="000062", start="2024-08-26", end="2024-09-04", publication_date="2024-09-05")
        events = merge_events(scan, [case])
        self.assertEqual(len(events), 2)
        self.assertFalse(events[-1]["scanned"])

    def test_publication_does_not_leak_to_event_date(self):
        scan, case = self.fixture()
        events = merge_events(scan, [case])
        self.assertEqual(known_official_events(events, "2024-01-17T16:00:00+08:00"), [])
        self.assertEqual(len(known_official_events(events, "2024-01-21T08:00:00+08:00")), 1)

    def test_other_trigger_cannot_be_promoted(self):
        scan, case = self.fixture()
        case["threshold_pct"] = 4
        with self.assertRaises(ValueError):
            merge_events(scan, [case])


if __name__ == "__main__":
    unittest.main()
