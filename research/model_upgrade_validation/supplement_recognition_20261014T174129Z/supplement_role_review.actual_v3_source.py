"""Additive forum -> cross-day relation -> independent role review entry point.

Old role APIs and source hashes remain intact. Context is also available when
formal feature inputs are unavailable; that route cannot register forecasts.
"""
import argparse
from pathlib import Path

import leader_role_review as legacy
import stock_catalyst_context as catalyst
import stock_forum_preread as preread
import stock_judgement_review as original
import supplement_relation_review as relation


def _match(left,right):
    a,b = original.unique_rows(left),original.unique_rows(right)
    if set(a) != set(b) or any((a[c]["name"],a[c]["boards"]) != (b[c]["name"],b[c]["boards"]) for c in a):
        raise ValueError("supplement context must retain the exact original full cohort")


def _preliminary(pool,bank,relation_bank,calendar,out,exposure):
    # The order is intentional. No role/model output is prepared before pre-read.
    forum = preread.build(pool,bank,calendar,out/"forum_preread",exposure)
    context = relation.build(pool,relation_bank,out/"relations",exposure=exposure)
    _match(forum["rows"],context["rows"])
    forum["checked_input_paths"] = {"pool":str(Path(pool).resolve()),"bank":str(Path(bank).resolve()),"calendar":str(Path(calendar).resolve())}
    return forum,context


def _merge(facts,forum,context,out,diagnostic):
    _match(facts["rows"],context["rows"])
    bycode = original.unique_rows(context["rows"])
    forum_bycode = original.unique_rows(forum["rows"])
    result = {**facts,"rows":[{**row,"forum_preread_context":forum_bycode[row["code"]],
        "supplement_relation_context":bycode[row["code"]]} for row in facts["rows"]]}
    forum_path = str((out/"forum_preread"/"report.json").resolve())
    context_path = str((out/"relations"/"relation_context.json").resolve())
    result.update({"forum_preread_context_path":forum_path,"supplement_relation_context_path":context_path,
        "supplement_review_required":True,"supplement_context_diagnostic_only":diagnostic,
        "supplement_model_exposure":context["model_exposure"],
        "forum_preread_checked_input_paths":forum["checked_input_paths"],
        "preparation_sequence":["FULL_POOL","FORUM_PREREAD","CROSS_DAY_RELATIONS","INDEPENDENT_ROLE_REVIEW"],
        "supplement_role_truth_certified":False,"supplement_prediction_changed":False})
    result["source_bindings"] = facts.get("source_bindings",{}) | forum["source_bindings"] | context["source_bindings"] | original.bind(forum_path) | original.bind(context_path) | original.bind(__file__) | original.bind(preread.__file__)
    return result


def _draft(facts,path,diagnostic):
    if diagnostic:
        draft = {"status":"DIAGNOSTIC_SUPPLEMENT_TEMPLATE_NO_FORECAST",
            "formed_at":original.now().isoformat(),"signal_date":facts["signal_date"],"target_date":facts["target_date"],
            "rows":[{"code":r["code"],"name":r["name"],"boards":r["boards"],"promotion_judgement":None} for r in facts["rows"]],
            "model_exposure":facts["supplement_model_exposure"],"author_review_completed":False,
            "directions_roles_and_reasons_autofilled":False,"original_prediction_changed":False}
    else:
        draft = legacy._draft_template(facts,path)
        draft["model_exposure"] = facts["supplement_model_exposure"]
    bycode = original.unique_rows(facts["rows"])
    for row in draft["rows"]:
        row["forum_preread_assessment"] = {"reviewed":False,"reason":"","unknown_not_negative":True}
        row["supplement_assessment"] = relation.assessment_template(bycode[row["code"]]["supplement_relation_context"])
    draft.update({"facts_path":str(Path(path).resolve()),"facts_sha256":original.sha(path),
        "supplement_review_required":True,"source_bindings":facts["source_bindings"] | original.bind(path)})
    return draft


def context(pool,bank,relation_bank,calendar,output_dir,exposure="MODEL_OUTPUT_ALREADY_SEEN"):
    """Complete diagnostic append-only route; no source/forecast gate bypass."""
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError("new immutable supplement output required")
    # preread itself verifies the official signal/target sessions and bank.
    forum,relations = _preliminary(pool,bank,relation_bank,calendar,out,exposure)
    original_pool = original.read(pool)
    rows = original_pool.get("rows",original_pool.get("candidates"))
    facts = {"status":"DIAGNOSTIC_FULL_POOL_SUPPLEMENT_CONTEXT_NO_FORECAST",
        "signal_date":original_pool["signal_date"],"target_date":original_pool["target_date"],
        "prepared_at":original.now().isoformat(),"pool_size":len(rows),"rows":rows,
        "model_scores_masked":True,"source_bindings":original.bind(pool),
        "new_fits":0,"new_formal_forward_samples":0,"forecast_registration_allowed":False}
    facts = _merge(facts,forum,relations,out,True)
    path = out/"facts_with_supplement.json"
    original.write_new(path,facts)
    original.write_new(out/"author_supplement_PENDING.json",_draft(facts,path,True))
    checked(facts)
    return facts


def prepare(features,theme_report,mainline_report,compatibility,bank,calendar,output_dir,
            pool,relation_bank,exposure="MODEL_OUTPUT_ALREADY_SEEN",role_evidence=None,
            history_datasets=(),mainline_history=(),narrative_reviews=None):
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError("new immutable supplement output required")
    forum,relations = _preliminary(pool,bank,relation_bank,calendar,out,exposure)
    facts = legacy.prepare(features,theme_report,mainline_report,compatibility,bank,calendar,out/"legacy_role_review",
        role_evidence,history_datasets,mainline_history,narrative_reviews)
    facts = _merge(facts,forum,relations,out,False)
    path = out/"facts_with_supplement.json"
    original.write_new(path,facts)
    original.write_new(out/"author_draft_PENDING.json",_draft(facts,path,False))
    checked(facts)
    return facts


def checked(facts):
    if facts.get("supplement_review_required") is not True or facts.get("preparation_sequence") != ["FULL_POOL","FORUM_PREREAD","CROSS_DAY_RELATIONS","INDEPENDENT_ROLE_REVIEW"]:
        raise ValueError("forum pre-read and cross-day relation stage required")
    original.check_sources(facts["source_bindings"])
    forum_path = facts["forum_preread_context_path"]
    if facts["source_bindings"].get(forum_path) != original.sha(forum_path):
        raise ValueError("forum pre-read is not source bound")
    forum = original.read(forum_path)
    inputs = facts["forum_preread_checked_input_paths"]
    for path in inputs.values():
        if facts["source_bindings"].get(path) != original.sha(path):
            raise ValueError("forum preparation input is not source bound")
    pool = original.read(inputs["pool"])
    _,sources,claims,_ = catalyst.validate_bank(inputs["bank"],pool.get("rows",pool.get("candidates")),original.stamp(forum["formed_at"]))
    regenerated = preread.project(pool.get("rows",pool.get("candidates")),claims,sources,pool["signal_date"])
    if forum["rows"] != regenerated:
        raise ValueError("forum pre-read does not match checked original evidence")
    _match(facts["rows"],forum["rows"])
    bycode = original.unique_rows(forum["rows"])
    for row in facts["rows"]:
        if row.get("forum_preread_context") != bycode[row["code"]]:
            raise ValueError("forum pre-read row projection changed")
    return relation.checked(facts)


def validate_review(facts,draft):
    relations = checked(facts)
    if draft.get("model_exposure") != facts["supplement_model_exposure"]:
        raise ValueError("actual declared model exposure cannot change")
    if draft.get("author_review_completed") is not True:
        raise ValueError("supplement draft is pending author review")
    relation.validate_assessments(relations,draft)
    for row in draft["rows"]:
        review = row.get("forum_preread_assessment",{})
        if review.get("reviewed") is not True or review.get("unknown_not_negative") is not True or not str(review.get("reason","")).strip():
            raise ValueError("all candidates require forum pre-read review or explicit unknown reason")
    return relations


def review_context(facts_path,draft_path,output):
    """Leave a completed diagnostic review on file without recording a forecast."""
    facts,draft = original.read(facts_path),original.read(draft_path)
    if facts.get("supplement_context_diagnostic_only") is not True or draft.get("facts_path") != str(Path(facts_path).resolve()) or draft.get("facts_sha256") != original.sha(facts_path):
        raise ValueError("source-bound diagnostic facts required")
    validate_review(facts,draft)
    if any(r.get("promotion_judgement") is not None for r in draft["rows"]):
        raise ValueError("diagnostic review cannot introduce a retroactive forecast")
    result = {**draft,"status":"COMPLETED_APPEND_ONLY_SUPPLEMENT_REVIEW_NO_FORECAST",
        "reviewed_at":original.now().isoformat(),"forecast_registered":False,
        "source_bindings":facts["source_bindings"] | original.bind(facts_path) | original.bind(draft_path)}
    original.write_new(output,result)
    return result


def freeze(facts_path,judgements_path,output):
    facts,draft = original.read(facts_path),original.read(judgements_path)
    if facts.get("supplement_context_diagnostic_only") is not False or draft.get("status","").startswith("DIAGNOSTIC"):
        raise ValueError("diagnostic supplement context cannot register a forecast")
    if draft.get("facts_path") != str(Path(facts_path).resolve()) or draft.get("facts_sha256") != original.sha(facts_path):
        raise ValueError("supplement judgement must bind its exact facts")
    validate_review(facts,draft)
    return legacy.freeze(facts_path,judgements_path,output)


def compare(record,baseline,outcome_archive,output):
    r = original.read(record)
    facts = original.read(r["facts_path"])
    if facts.get("supplement_review_required"):
        validate_review(facts,{**r,"author_review_completed":True})
    return legacy.compare(record,baseline,outcome_archive,output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command",required=True)
    for name in ("context","prepare"):
        q = sub.add_parser(name)
        for field in ("pool","bank","relation-bank","calendar","output-dir"):
            q.add_argument("--"+field,required=True)
        q.add_argument("--model-exposure",default="MODEL_OUTPUT_ALREADY_SEEN")
        if name == "prepare":
            for field in ("features","theme-report","mainline-report","compatibility"):
                q.add_argument("--"+field,required=True)
            q.add_argument("--role-evidence")
    for name in ("freeze","review-context"):
        q = sub.add_parser(name)
        for field in ("facts","judgements","output"):
            q.add_argument("--"+field,required=True)
    q = sub.add_parser("compare")
    for field in ("record","baseline","outcome-archive","output"):
        q.add_argument("--"+field,required=True)
    a = p.parse_args()
    if a.command in ("context","prepare"):
        kw = dict(pool=a.pool,bank=a.bank,relation_bank=a.relation_bank,calendar=a.calendar,
                  output_dir=a.output_dir,exposure=a.model_exposure)
        if a.command == "prepare":
            kw.update(features=a.features,theme_report=a.theme_report,mainline_report=a.mainline_report,
                      compatibility=a.compatibility,role_evidence=a.role_evidence)
        r = globals()[a.command](**kw)
    elif a.command == "compare":
        r = compare(a.record,a.baseline,a.outcome_archive,a.output)
    else:
        r = globals()[a.command.replace("-","_")](a.facts,a.judgements,a.output)
    print({"status":r.get("status"),"pool_size":r.get("pool_size",len(r.get("rows",[])))})
