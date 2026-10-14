import unittest
import pandas as pd
from theme_identity_review import review


def fixture():
    return pd.DataFrame([dict(signal_date="2026-09-01",code=code,theme_group_id=group,
                             theme=group,theme_granularity="细分题材",source_ref="synthetic_not_market",
                             source_published_at="2026-09-01 12:27:00",collected_at="2026-09-01 22:20:00",
                             timestamp_precision="exact",evidence_status="verified",evidence_kind="MEMBERSHIP",
                             membership_basis="verified_concept_membership")
                         for code,group in [("000001","subtheme_a"),("000002","subtheme_a"),("000003","subtheme_b")]])


class ThemeIdentityTests(unittest.TestCase):
    def test_after_close_capture_not_intraday_knowledge(self):
        result=review(fixture(),"000001","2026-09-01","2026-09-01T15:00:00+08:00")
        self.assertEqual(result["groups"],[])
        self.assertEqual(result["excluded_candidate_evidence"][0]["reason"],"NOT_KNOWN_AT_DECISION")
        historical=review(fixture(),"000001","2026-09-01","2026-09-01T15:00:00+08:00","historical_reconstruction")
        self.assertEqual(len(historical["groups"]),1)

    def test_candidate_specific_no_cross_theme_support(self):
        result=review(fixture(),"000001","2026-09-01","2026-09-02T09:25:00+08:00")
        self.assertEqual(result["groups"][0]["observed_peers"],["000002"])
        self.assertIsNone(result["groups"][0]["current_support_rate"])
        self.assertFalse(result["groups"][0]["roster_complete"])

    def test_broad_membership_not_narrow(self):
        frame=fixture()
        frame["theme_granularity"]="行业"
        self.assertFalse(review(frame,"000001","2026-09-01","2026-09-02T09:25:00+08:00")["groups"])

    def test_duplicate_evidence_does_not_inflate_peers(self):
        frame=pd.concat([fixture(),fixture()],ignore_index=True)
        result=review(frame,"000001","2026-09-01","2026-09-02T09:25:00+08:00")
        self.assertEqual(result["groups"][0]["observed_peer_count"],1)

    def test_future_source_not_backfilled(self):
        frame=fixture()
        frame["source_published_at"]="2026-09-03 12:00:00"
        frame["collected_at"]="2026-09-03 13:00:00"
        self.assertFalse(review(frame,"000001","2026-09-01","2026-09-02T09:25:00+08:00")["groups"])
