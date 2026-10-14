import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import leader_role_engine as e
import leader_role_review as r
import stock_judgement_review as original


def facts(root):
    feature = root/"features.json"
    e.write_new(feature, {"fixture": True})
    scopes = {name: {"declared_peer_codes": ["000002"], "missing_or_incomplete_peer_codes": ["000002"],
        "pairs": [{"peer_code": "000002", "peer_ahead_at_some_sample": None}]} for name in ("current", "prior")}
    rows = []
    for code in ("000001", "000002"):
        rows.append({"code": code, "name": "fixture"+code, "boards": 2,
            "same_height_peers": ["000002" if code == "000001" else "000001"],
            "catalyst_context": {"narrative_presence": "UNKNOWN", "earnings_impact": "UNKNOWN"},
            "reviewed_narrative": {"current_trade_narrative_presence": "UNKNOWN"},
            "short_term_context": {"pairwise_price_competition": {"scopes": scopes}},
            "leader_role_evidence": {"roles": {"HEIGHT": {"raw_score": .99, "model_call": True}}}})
    return {"signal_date": "2026-10-12", "target_date": "2026-10-13",
        "feature_archive": str(root), "source_bindings": e.bind(feature), "rows": rows}


class DraftTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.root = Path(self.tmp.name)
        self.facts = facts(self.root);self.path = self.root/"facts.json";e.write_new(self.path,self.facts)

    def tearDown(self):self.tmp.cleanup()

    def test_complete_modules_and_pool_including_unselected_stocks(self):
        draft = r._draft_template(self.facts, self.path)
        self.assertEqual([x["code"] for x in draft["rows"]], ["000001", "000002"])
        for row in draft["rows"]:
            self.assertEqual(set(row["role_assessment"]), set(e.ROLES))
            self.assertEqual(set(row["short_term_assessment"]["checks"]), set(r.short_term.CHECKS))
            self.assertEqual(set(row["pairwise_competition_assessment"]["scopes"]), {"current", "prior"})
            self.assertEqual(row["comparison_peers"], self.facts["rows"][draft["rows"].index(row)]["same_height_peers"])

    def test_learned_score_and_call_do_not_become_template_direction(self):
        draft = r._draft_template(self.facts, self.path)
        self.assertIsNone(draft["model_exposure"])
        for row in draft["rows"]:
            self.assertIsNone(row["promotion_judgement"])
            self.assertTrue(all(x["state"] is None for x in row["role_assessment"].values()))
            self.assertNotIn("raw_score", row)
            self.assertNotIn("model_call", row)

    def test_new_template_plans_have_three_actual_target_windows_without_guessed_conditions(self):
        draft = r._draft_template(self.facts, self.path)
        plans = draft["rows"][0]["short_term_assessment"]["confirmation_plan"]
        self.assertEqual(len(plans), 3)
        for plan in plans:
            self.assertTrue(plan["planned_at"].startswith("2026-10-13T"))
            self.assertEqual(plan["status"], "PENDING")
            self.assertTrue(all(x["conditions"] == [] and x["reason"] == "" for x in plan["paths"].values()))

    def test_diagnostic_template_does_not_reconstruct_old_expectations(self):
        draft = r._draft_template(self.facts, self.path, diagnostic=True)
        self.assertEqual(draft["status"], "DIAGNOSTIC_TEMPLATE_NO_FORECAST")
        self.assertTrue(all(x["short_term_assessment"]["confirmation_plan"] == [] for x in draft["rows"]))

    def test_pending_template_is_rejected_before_original_freeze_or_registration(self):
        draft = r._draft_template(self.facts, self.path)
        path = self.root/"pending.json";e.write_new(path,draft)
        with patch.object(r.catalyst, "freeze") as freeze:
            with self.assertRaisesRegex(ValueError,"pending or diagnostic"):
                r.freeze(self.path,path,self.root/"out.json")
            freeze.assert_not_called()
        self.assertFalse((self.root/"out.json").exists())

    def test_changed_status_does_not_complete_author_review(self):
        draft = r._draft_template(self.facts, self.path);draft["status"] = "AUTHOR_COMPLETED_ROLE_REVIEW"
        path = self.root/"not_reviewed.json";e.write_new(path,draft)
        with self.assertRaisesRegex(ValueError,"not been completed"):
            r.freeze(self.path,path,self.root/"out.json")

    def test_expired_target_cannot_generate_new_forecast_draft(self):
        actual = {**self.facts,"sessions":["2026-10-12","2026-10-13"]}
        path = self.root/"expired.json";e.write_new(path,actual)
        with patch.object(original,"now",return_value=e.stamp("2026-10-13T13:00:00+08:00")):
            with self.assertRaisesRegex(ValueError,"before target auction"):
                r.create_draft(path,self.root/"out.json")
        self.assertFalse((self.root/"out.json").exists())

    def test_blank_direction_cannot_pass_original_judgement_validation(self):
        draft = r._draft_template(self.facts, self.path);draft["model_exposure"] = "MODEL_OUTPUT_ALREADY_SEEN"
        with self.assertRaisesRegex(ValueError,"PROMOTE/FAIL/UNKNOWN"):
            original.validate_judgements(self.facts,draft)


if __name__ == "__main__":unittest.main()
