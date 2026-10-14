import unittest
from catalyst_evidence_v29 import snapshot


class CatalystTests(unittest.TestCase):
    def fixture(self):
        return dict(records=[dict(evidence_id="a1", event_id="a", code="603137", publication_date="2026-07-01",
                                  captured_at="2026-09-04T17:00:00+08:00", evidence_status="verified", source="fixture")])

    def test_date_only_not_in_auction(self):
        result = snapshot(self.fixture(), "603137", "2026-07-01T09:25:00+08:00", "historical_reconstruction")
        self.assertIsNone(result["observed_event_count"])

    def test_reconstruction_not_forward_archive(self):
        ledger = self.fixture()
        asof = "2026-07-03T09:25:00+08:00"
        self.assertEqual(snapshot(ledger, "603137", asof, "historical_reconstruction")["observed_event_count"], 1)
        self.assertIsNone(snapshot(ledger, "603137", asof)["observed_event_count"])

    def test_does_not_infer_theme_or_buy(self):
        result = snapshot(self.fixture(), "603137", "2026-09-04T18:00:00+08:00")
        self.assertIsNone(result["theme_peer_count"])
        self.assertFalse(result["buy_signal"])

    def test_unverified_not_used(self):
        ledger = self.fixture()
        ledger["records"][0]["evidence_status"] = "mirror_observed"
        self.assertIsNone(snapshot(ledger, "603137", "2026-09-04T18:00:00+08:00")["observed_event_count"])

    def test_duplicate_rejected(self):
        ledger = self.fixture()
        ledger["records"].append(ledger["records"][0].copy())
        with self.assertRaises(ValueError):
            snapshot(ledger, "603137", "2026-09-04T18:00:00+08:00")

    def test_future_revision_does_not_replace_old_fact(self):
        ledger = self.fixture()
        record = dict(ledger["records"][0], evidence_id="a2", publication_date="2026-07-10", event_stage="cancelled")
        ledger["records"].append(record)
        result = snapshot(ledger, "603137", "2026-07-03T09:25:00+08:00", "historical_reconstruction")
        self.assertEqual(result["records"][0]["evidence_id"], "a1")


if __name__ == "__main__":
    unittest.main()
