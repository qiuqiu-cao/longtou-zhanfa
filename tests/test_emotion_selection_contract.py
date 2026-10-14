import copy
import unittest

from emotion_selection_contract import audit_outlook


def fixture():
    rule = dict(condition_id="r", metric="theme_sync", unit="ratio", definition_id="sync-v1",
                operator=">=", threshold=.6, window_seconds=300, max_age_seconds=60)
    evidence = dict(evidence_id="e", metric="theme_sync", value=.5, source="fixture",
                    observed_at="2026-09-16T15:00:00+08:00", available_at="2026-09-16T15:01:00+08:00",
                    data_status="verified", definition_id="sync-v1", denominator=10,
                    observed_count=10, missing_count=0)
    negative = dict(evidence, evidence_id="n", value=.2)
    link = dict(code="000001", board_group="FIRST_BOARD", continuation_driver="catalyst",
                exhaustion_risk="crowding", relative_advantage="pioneer", main_path_action="watch",
                alternative_path_action="cancel", support_evidence_ids=["e"], opposing_evidence_ids=["n"],
                entry_condition_ids=["r"], cancel_condition_ids=["r"])
    return dict(asof="2026-09-16T16:00:00+08:00", report_date="2026-09-16",
                focus_candidates=[dict(code="000001", board_group="FIRST_BOARD",
                                       opening_conditions=[rule], cancel_conditions=[rule])],
                next_session_outlook=dict(target_session="2026-09-17", main_path="DIVERGENCE_TO_REPAIR",
                    alternative_path="DIVERGENCE_CONTINUES", evidence=[evidence, negative],
                    support_evidence_ids=["e"], opposing_evidence_ids=["n"],
                    confirmation_conditions=[rule], invalidation_conditions=[rule], selection_links=[link]))


class EmotionSelectionContractTest(unittest.TestCase):
    def test_valid_never_authorizes_entry(self):
        result = audit_outlook(fixture())
        self.assertEqual(result["issues"], [])
        self.assertFalse(result["entry_authorized"])

    def test_future_evidence_rejected(self):
        report = fixture()
        report["next_session_outlook"]["evidence"][0]["available_at"] = "2026-09-17T10:00:00+08:00"
        self.assertIn("INVALID_OR_DUPLICATE_OUTLOOK_EVIDENCE", audit_outlook(report)["issues"])

    def test_all_candidates_need_links_and_real_rules(self):
        report = fixture()
        report["focus_candidates"].append(dict(code="600001", board_group="CONSECUTIVE_BOARD"))
        self.assertIn("SELECTION_LINK_COVERAGE_MISMATCH", audit_outlook(report)["issues"])
        report = fixture()
        report["next_session_outlook"]["selection_links"][0]["entry_condition_ids"] = ["invented"]
        self.assertIn("UNLINKED_ENTRY_CONDITION_IDS", audit_outlook(report)["issues"])

    def test_no_fake_probability_or_missing_opposition(self):
        report = fixture()
        report["next_session_outlook"]["probability"] = .9
        report["next_session_outlook"]["opposing_evidence_ids"] = []
        issues = audit_outlook(report)["issues"]
        self.assertIn("UNCALIBRATED_OUTLOOK_PROBABILITY", issues)
        self.assertIn("INVALID_OPPOSING_EVIDENCE_IDS", issues)
