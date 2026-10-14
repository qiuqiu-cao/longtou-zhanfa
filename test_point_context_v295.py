import copy
import json
from pathlib import Path
import unittest

from point_context_v295 import parse_categories,parse_latest,parse_history,asof_view,normalize,scope_from_pool
from context_draft_bridge_v295 import enrich
from point_context_reconciled_v295 import latest

PROBE=Path(__file__).resolve().parent/"data/v295/context_source_probe_20261002_v1"


class PointContextTests(unittest.TestCase):
    def data(self,name):return json.loads((PROBE/(name+".response")).read_bytes())

    def test_actual_categories_reconcile(self):
        result=parse_categories(self.data("concept_page"),self.data("concept_datacenter"),"600825")
        self.assertEqual(len(result["rows"]),23)
        self.assertFalse(result["active_trading_theme_verified"])
        self.assertFalse(result["historical_membership_verified"])

    def test_category_wrong_security_rejected(self):
        with self.assertRaisesRegex(ValueError,"security identity"):
            parse_categories(self.data("concept_page"),self.data("concept_datacenter"),"600241")

    def test_category_count_mismatch(self):
        value=self.data("concept_datacenter");value["result"]["count"]+=1
        with self.assertRaisesRegex(ValueError,"count incomplete"):
            parse_categories(self.data("concept_page"),value,"600825")

    def test_category_disagreement_not_guessed(self):
        value=self.data("concept_page");value["ssbk"][0]["BOARD_NAME"]="changed"
        with self.assertRaisesRegex(ValueError,"endpoints disagree"):
            parse_categories(value,self.data("concept_datacenter"),"600825")

    def test_actual_rank_has_source_clock(self):
        result=parse_latest(self.data("popularity"),"600825","2026-10-02T22:27:00+08:00")
        self.assertEqual(result["rank"],16)
        self.assertEqual(result["calculated_at"],"2026-10-02T22:24:00+08:00")
        self.assertIsNone(result["bullishness"])

    def test_rank_wrong_security(self):
        with self.assertRaisesRegex(ValueError,"security identity"):
            parse_latest(self.data("popularity"),"600241","2026-10-02T22:27:00+08:00")

    def test_rank_future_clock(self):
        with self.assertRaisesRegex(ValueError,"future"):
            parse_latest(self.data("popularity"),"600825","2026-10-02T22:23:00+08:00")

    def test_rank_universe_bounds(self):
        value=self.data("popularity");value["data"]["rank"]=value["data"]["marketAllCount"]+1
        with self.assertRaisesRegex(ValueError,"outside provider universe"):
            parse_latest(value,"600825","2026-10-02T22:27:00+08:00")

    def test_history_never_becomes_original_live_fact(self):
        result=parse_history(self.data("popularity_history"),"600825","2026-10-02T22:27:00+08:00")
        self.assertEqual(len(result["rows"]),366)
        self.assertFalse(result["original_eod_availability_verified"])
        self.assertFalse(result["forward_feature_authorized"])

    def test_history_duplicate_date(self):
        value=self.data("popularity_history");value["data"].append(copy.deepcopy(value["data"][-1]))
        with self.assertRaisesRegex(ValueError,"duplicate"):
            parse_history(value,"600825","2026-10-02T22:27:00+08:00")

    def test_no_backfill_to_old_cutoff(self):
        value={"candidates":[{"code":"600825","errors":{},"categories":{"available_at":"2026-10-02T22:27:00+08:00"},"attention":None,"attention_history":None}]}
        result=asof_view(value,"2026-10-02T16:43:38+08:00")
        self.assertIsNone(result["candidates"][0]["categories"])
        self.assertIn("AFTER_KNOWLEDGE",result["candidates"][0]["unavailable"]["categories"])

    def test_future_day_cannot_reuse_old_context_as_fresh(self):
        value={"candidates":[{"code":"600825","errors":{},"categories":{"available_at":"2026-10-02T22:27:00+08:00"},"attention":{"available_at":"2026-10-02T22:27:00+08:00","calculated_at":"2026-10-02T22:24:00+08:00"},"attention_history":None}]}
        result=asof_view(value,"2026-10-08T19:30:00+08:00")
        self.assertIsNone(result["candidates"][0]["categories"])
        self.assertIsNone(result["candidates"][0]["attention"])

    def test_missing_request_preserves_error(self):
        with self.assertRaisesRegex(ValueError,"request scope"):
            normalize({"codes":["600825"]},[],"2026-10-02T22:27:00+08:00")

    def test_real_scope_is_complete_provider_pool(self):
        scope=scope_from_pool(Path(__file__).resolve().parent/"data/v293/public_outcome_sources_20261002_v1","2026-09-30")
        self.assertEqual(len(scope["codes"]),48)
        self.assertFalse(scope["official_full_market"])

    def test_reviewed_or_registered_plan_cannot_be_changed(self):
        with self.assertRaisesRegex(ValueError,"unreviewed new daily draft"):
            enrich({"daily_research":{"draft_status":"AUTHOR_REVIEWED"}},None,None)

    def test_draft_cannot_use_new_capture_as_old_theme(self):
        context={"scope":{"price_signal_date":"2026-09-30"},"candidates":[{"code":"600825","errors":{},"categories":{"available_at":"2026-10-02T22:27:00+08:00"},"attention":None,"attention_history":None}]}
        plan={"signal_date":"2026-09-30","evidence_cutoff":"2026-10-02T22:28:00+08:00","candidates":[{"code":"600825"}],"daily_research":{"draft_status":"NEEDS_FACTS_AND_RESEARCH_CALLS"}}
        with self.assertRaisesRegex(ValueError,"signal-day membership"):
            enrich(plan,context,None)

    def test_internal_identifier_is_opaque(self):
        value=self.data("popularity");value["data"]["innerCode"]="gs80534114"
        result=latest(value,"600825","2026-10-02T22:27:00+08:00")
        self.assertEqual(result["rank"],16)
        self.assertEqual(result["provider_internal_identifier"],"gs80534114")

    def test_explicit_security_code_still_required(self):
        value=self.data("popularity");value["data"]["innerCode"]="gs80534114";value["data"]["srcSecurityCode"]="SH600241"
        with self.assertRaisesRegex(ValueError,"explicit security"):
            latest(value,"600825","2026-10-02T22:27:00+08:00")

    def test_enrichment_preserves_forecasts_and_requires_review(self):
        source=str(PROBE/"popularity.response")
        categories={"available_at":"2026-10-02T22:27:00+08:00","rows":[{"board_code":"486","board_name":"传媒"}]}
        attention={"available_at":"2026-10-02T22:27:00+08:00","calculated_at":"2026-10-02T22:24:00+08:00","rank":16,"provider_universe_count":5572}
        context={"scope":{"price_signal_date":"2026-10-02"},"candidates":[{"code":"600825","errors":{},"categories":categories,"attention":attention,"attention_history":None}]}
        plan={"signal_date":"2026-10-02","evidence_cutoff":"2026-10-02T22:28:00+08:00","candidates":[{"code":"600825","promotion_call":"PROMOTE"}],
            "daily_research":{"draft_status":"NEEDS_FACTS_AND_RESEARCH_CALLS","candidates":[{"code":"600825","events":{"promotion":{"call":True,"probability":None}}}],"consensus":{"attention":None},"leaders":[{"identity_status":"CANDIDATE"}]}}
        result=enrich(plan,context,source)
        self.assertEqual(result["candidates"],plan["candidates"])
        self.assertEqual(result["daily_research"]["leaders"],plan["daily_research"]["leaders"])
        self.assertEqual(result["daily_research"]["candidates"][0]["events"],plan["daily_research"]["candidates"][0]["events"])
        self.assertEqual(result["daily_research"]["draft_status"],"NEEDS_FACTS_AND_RESEARCH_CALLS")
        self.assertIsNone(result["daily_research"]["candidates"][0]["theme"]["value"]["active_trading_theme"])


if __name__=="__main__":unittest.main()
