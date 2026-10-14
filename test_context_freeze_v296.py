from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from context_freeze_contract_v296 import validate_context_binding
from context_draft_bridge_v295 import enrich
from daily_research_template_v292 import draft_extension
from point_context_v295 import sha

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/"data/v295/provider_context_reconciled_20261002_v1/context.json"


class ContextFreezeTests(unittest.TestCase):
    def fixture(self):
        # Isolated binding fixture only. This closed-day date is never registered
        # or counted as a genuine market forecast; production calendar rejects it.
        context={"recorded_at":"2026-10-02T22:27:00+08:00","scope":{"price_signal_date":"2026-10-02"},"candidates":[{"code":"600825","errors":{},
            "categories":{"available_at":"2026-10-02T22:27:00+08:00","rows":[{"board_code":"486","board_name":"传媒"}]},
            "attention":{"available_at":"2026-10-02T22:27:00+08:00","calculated_at":"2026-10-02T22:24:00+08:00","rank":16,"provider_universe_count":5572},"attention_history":None}]}
        plan={"signal_date":"2026-10-02","evidence_cutoff":"2026-10-02T22:28:00+08:00","candidates":[{"code":"600825","promotion_call":"PROMOTE"}],
            "daily_research":{"draft_status":"NEEDS_FACTS_AND_RESEARCH_CALLS","candidates":[{"code":"600825","theme":{"status":"UNKNOWN","value":None,"reason":"missing"},"events":{"promotion":{"call":True,"probability":None}}}],"consensus":{"attention":None},"leaders":[{"identity_status":"CANDIDATE"}]}}
        enriched=enrich(plan,context,str(SOURCE))
        enriched["daily_research"]["draft_status"]="AUTHOR_REVIEWED"
        return enriched,deepcopy(context)

    def verify(self,plan,context):
        with patch("context_freeze_contract_v296.reconstruct",return_value=context):
            return validate_context_binding(plan)

    def test_actual_legacy_original_remains_unchanged(self):
        path=ROOT/"data/v271/expectation_registry/2026-09-30_2c74dc406eef7dcf.json"
        original=json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(validate_context_binding(original["plan"])["status"],"LEGACY_NO_CONTEXT_FIELDS")

    def test_reviewed_original_source_projection_allowed(self):
        plan,context=self.fixture()
        result=self.verify(plan,context)
        self.assertEqual(result["attention_codes"],1)
        self.assertFalse(result["probability_authorized_by_context"])

    def test_rank_tamper_rejected(self):
        plan,context=self.fixture()
        plan["daily_research"]["consensus"]["attention"]["value"]["600825"]["rank"]=1
        with self.assertRaisesRegex(ValueError,"attention fact differs"):self.verify(plan,context)

    def test_category_tamper_rejected(self):
        plan,context=self.fixture()
        plan["daily_research"]["candidates"][0]["theme"]["value"]["provider_board_memberships"][0]["board_name"]="fictional"
        with self.assertRaisesRegex(ValueError,"category fact differs"):self.verify(plan,context)

    def test_current_category_cannot_be_declared_active_theme(self):
        plan,context=self.fixture()
        plan["daily_research"]["candidates"][0]["theme"]["value"]["active_trading_theme"]="confirmed leader theme"
        with self.assertRaisesRegex(ValueError,"category fact differs"):self.verify(plan,context)

    def test_clock_tamper_rejected(self):
        plan,context=self.fixture()
        plan["daily_research"]["consensus"]["attention"]["available_at"]="2026-09-30T15:00:00+08:00"
        with self.assertRaisesRegex(ValueError,"attention fact differs"):self.verify(plan,context)

    def test_binding_cannot_be_removed(self):
        plan,context=self.fixture();plan.pop("context_binding")
        with self.assertRaisesRegex(ValueError,"require a source binding"):self.verify(plan,context)

    def test_binding_flags_cannot_enable_history(self):
        plan,context=self.fixture();plan["context_binding"]["history_used_in_live_features"]=True
        with self.assertRaisesRegex(ValueError,"binding scope/flags"):self.verify(plan,context)

    def test_source_hash_tamper_rejected(self):
        plan,context=self.fixture();plan["context_binding"]["source_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"source missing or hash"):self.verify(plan,context)

    def test_future_cutoff_cannot_admit_new_facts(self):
        plan,context=self.fixture();plan["evidence_cutoff"]="2026-10-02T16:43:38+08:00"
        with self.assertRaisesRegex(ValueError,"snapshot arrived after knowledge"):self.verify(plan,context)

    def test_no_context_gaps_are_not_filled(self):
        plan={"daily_research":{"candidates":[],"consensus":{"attention":{"status":"UNKNOWN","value":None,"reason":"missing"}}}}
        self.assertEqual(validate_context_binding(plan)["status"],"NO_PROVIDER_CONTEXT_BOUND")

    def test_actual_later_snapshot_cannot_bind_original(self):
        original=json.loads((ROOT/"data/v271/expectation_registry/2026-09-30_2c74dc406eef7dcf.json").read_text(encoding="utf-8"))
        plan=deepcopy(original["plan"]);plan["daily_research"]=draft_extension(plan)
        plan["context_binding"]={"source":str(SOURCE),"source_sha256":sha(SOURCE),"category_codes_filled":[],"attention_codes_filled":[],"history_used_in_live_features":False,"author_review_still_required":True}
        with self.assertRaisesRegex(ValueError,"snapshot arrived after knowledge"):
            validate_context_binding(plan)

    def test_freeze_cli_rejects_before_register_write(self):
        import context_bound_daily_runner_v296 as runner
        plan,context=self.fixture()
        plan["daily_research"]["consensus"]["attention"]["value"]["600825"]["rank"]=1
        with patch("context_bound_daily_runner_v296.validate_contract",return_value={}),patch("context_freeze_contract_v296.reconstruct",return_value=context),patch("pathlib.Path.read_text",return_value=json.dumps(plan)),patch("sys.argv",["runner","freeze","--plan","ISOLATED_TEST_FIXTURE_NOT_FORECAST.json"]),patch("context_bound_daily_runner_v296.register") as registration:
            with self.assertRaisesRegex(ValueError,"attention fact differs"):
                runner.main()
            registration.assert_not_called()


if __name__=="__main__":unittest.main()
