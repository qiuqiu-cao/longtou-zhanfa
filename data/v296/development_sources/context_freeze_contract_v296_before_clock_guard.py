"""Rebuild V295-derived facts at freeze/read time; preserve independent claims."""
from copy import deepcopy
from pathlib import Path

from context_draft_bridge_v295 import enrich
from point_context_reconciled_v295 import reconstruct
from point_context_v295 import sha
from next_session_expectation_contract_v271 import digest

VERSION="v296-context-freeze-binding-1"


def uses_context(fact):
    if not isinstance(fact,dict) or fact.get("status")=="UNKNOWN":return False
    source=fact.get("source","")
    value=fact.get("value")
    if isinstance(source,str) and Path(source).name=="context.json":return True
    if isinstance(value,dict):
        if "provider_board_memberships" in value:return True
        if any(isinstance(row,dict) and {"rank","provider_universe_count"}.issubset(row) for row in value.values()):return True
    return False


def validate_context_binding(plan):
    ext=plan.get("daily_research")
    if not ext:return {"status":"LEGACY_NO_CONTEXT_FIELDS","backfilled":False}
    binding=plan.get("context_binding")
    attention=ext.get("consensus",{}).get("attention")
    provider_fields=[c.get("theme") for c in ext.get("candidates",[])]+[attention]
    if binding is None:
        if any(uses_context(f) for f in provider_fields):raise ValueError("provider context fields require a source binding")
        return {"status":"NO_PROVIDER_CONTEXT_BOUND","backfilled":False}
    if not isinstance(binding,dict):raise ValueError("invalid context binding")
    source=Path(binding.get("source",""))
    if source.name!="context.json" or not source.is_file() or sha(source)!=binding.get("source_sha256"):
        raise ValueError("context source missing or hash mismatch")
    context=reconstruct(source.parent)
    probe=deepcopy(plan)
    probe["daily_research"]["draft_status"]="NEEDS_FACTS_AND_RESEARCH_CALLS"
    expected=enrich(probe,context,str(source))
    if digest(binding)!=digest(expected["context_binding"]):raise ValueError("context binding scope/flags differ from raw reconstruction")
    actual={c["code"]:c for c in ext["candidates"]}
    rebuilt={c["code"]:c for c in expected["daily_research"]["candidates"]}
    if len(actual)!=len(ext["candidates"]) or set(actual)!=set(c["code"] for c in plan["candidates"]):raise ValueError("daily context candidate denominator changed")
    for code in binding["category_codes_filled"]:
        if digest(actual[code]["theme"])!=digest(rebuilt[code]["theme"]):
            raise ValueError("provider category fact differs from raw reconstruction: "+code)
    if digest(attention)!=digest(expected["daily_research"]["consensus"]["attention"]):
        raise ValueError("provider attention fact differs from raw reconstruction")
    # Unfilled candidate facts must not masquerade as successful context data.
    for code,candidate in actual.items():
        if code not in binding["category_codes_filled"] and uses_context(candidate.get("theme")):
            raise ValueError("unavailable provider category fabricated: "+code)
    return {"version":VERSION,"status":"CONTEXT_FACTS_REBUILT_BEFORE_REGISTRATION_OR_SCORING",
        "category_codes":len(binding["category_codes_filled"]),"attention_codes":len(binding["attention_codes_filled"]),
        "historical_dates_used_in_live_features":False,"probability_authorized_by_context":False,
        "leader_confirmation_authorized_by_context":False,"source_sha256":sha(source)}
