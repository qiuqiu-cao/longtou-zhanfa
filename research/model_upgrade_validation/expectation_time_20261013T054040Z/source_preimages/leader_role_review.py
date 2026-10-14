"""Optional role layer around unchanged catalyst/discretionary production APIs."""
import argparse
from datetime import datetime, time
import json
from pathlib import Path

import leader_role_engine as e
import leader_role_research as research
import stock_catalyst_context as catalyst
import stock_judgement_review as original
import stock_short_term_context as short_term
import stock_expectation_review as expectation
import stock_narrative_review as narrative


def prepare(features, theme_report, mainline_report, compatibility, bank, calendar, output_dir, role_evidence=None, history_datasets=(), mainline_history=(), narrative_reviews=None):
    out = Path(output_dir)
    facts = catalyst.prepare(features, theme_report, mainline_report, compatibility, bank, out/"catalysts")
    dataset = e.from_archive(features, calendar, theme_report, out/"role_dataset.original.json")
    dataset = e.augment(out/"role_dataset.original.json", out/"role_dataset.json", evidence_path=role_evidence)
    roles = e.analyse(dataset, "STRICT")
    e.write_new(out/"role_facts.json", roles)
    short_context_path = out/"short_term_context.json"
    short_context = short_term.build(out/"role_dataset.json", facts["catalyst_context_path"], short_context_path, "STRICT", history_datasets, [*mainline_history, mainline_report])
    narrative_path = out/"narrative_review_context.json"
    narrative_context = narrative.build(facts["catalyst_context_path"], bank, narrative_reviews, narrative_path)
    packet_path = out/"blind_identity_packet.json"
    e.write_new(packet_path, research.masked_packet(dataset))
    e.write_new(out/"identity_labels_PENDING.json", research.label_template(dataset, packet_path))
    merged = dict(facts)
    bycode = {r["code"]: r for r in roles["rows"]}
    short_bycode = {r["code"]: r for r in short_context["rows"]}
    narrative_bycode = {r["code"]: r for r in narrative_context["rows"]}
    merged["rows"] = [{**r, "leader_role_evidence": bycode[r["code"]],
                       "short_term_context": short_bycode[r["code"]],
                       "reviewed_narrative": narrative_bycode[r["code"]]} for r in facts["rows"]]
    merged["role_dataset_path"] = str((out/"role_dataset.json").resolve())
    merged["role_dataset_sha256"] = e.sha(out/"role_dataset.json")
    merged["short_term_context_path"] = str(short_context_path.resolve())
    merged["short_term_context_sha256"] = e.sha(short_context_path)
    merged["narrative_review_context_path"] = str(narrative_path.resolve())
    merged["narrative_review_context_sha256"] = e.sha(narrative_path)
    merged["source_bindings"] = facts["source_bindings"] | e.bind(out/"role_dataset.json") | e.bind(out/"role_facts.json") | short_context["source_bindings"] | e.bind(short_context_path) | e.bind(expectation.__file__)
    merged["source_bindings"] |= narrative_context["source_bindings"] | e.bind(narrative_path)
    merged["source_bindings"] |= e.bind(__file__)
    merged["prepared_at"] = e.now()
    merged["role_identity_verified_by_this_tool"] = False
    e.write_new(out/"facts_with_roles.json", merged)
    create_draft(out/"facts_with_roles.json", out/"author_draft_PENDING.json")
    return merged


def _draft_template(facts, facts_path, diagnostic=False):
    """Formatting scaffold only: never selects a promotion direction or role."""
    rows = []
    refs = [str((Path(facts["feature_archive"])/"features.json").resolve())]
    for key in ("theme_report", "mainline_report", "catalyst_context_path"):
        if facts.get(key) in facts["source_bindings"]:
            refs.append(facts[key])
    for fact in facts["rows"]:
        catalyst_fact = fact["catalyst_context"]
        narrative_fact = fact["reviewed_narrative"]
        pair_fact = fact["short_term_context"]["pairwise_price_competition"]
        scopes = {}
        for scope, data in pair_fact["scopes"].items():
            scopes[scope] = {"state": None, "reason": "",
                "comparison_peer_codes": data["declared_peer_codes"],
                "incomplete_peer_codes": data["missing_or_incomplete_peer_codes"],
                "stronger_observed_peer_codes": sorted(p["peer_code"] for p in data["pairs"] if p["peer_ahead_at_some_sample"] is True),
                "next_session_paths": {key: {"state": None, "reason": ""}
                    for key in ("joint_promotion", "joint_failure", "divergence")}}
        plans = []
        if not diagnostic:
            for clock, start in (("09:25", "09:20"), ("09:35", "09:30"), ("10:00", "09:35")):
                def at(value):
                    return datetime.combine(datetime.fromisoformat(facts["target_date"]).date(),
                        time.fromisoformat(value), original.SHANGHAI).isoformat()
                plans.append({"checkpoint_id": "opening_"+clock.replace(":", ""),
                    "baseline_kind": "AUTHOR_PREDECLARED", "planned_at": at(clock),
                    "status": "PENDING", "requirement": "",
                    "observation_window": {"start_at": at(start), "end_at": at(clock)},
                    "paths": {name: {"state": "UNKNOWN", "reason": "", "conditions": []}
                        for name in expectation.PATHS}})
        rows.append({"code": fact["code"], "name": fact["name"], "boards": fact["boards"], "promotion_judgement": None,
            "role_hypothesis": "", "relative_thesis": "", "support": "", "rebuttal": "",
            "cancel_condition": "", "missing": [], "comparison_peers": fact["same_height_peers"],
            "fact_references": refs, "opening_confirmation": {k: "" for k in ("09:25", "09:35", "10:00")},
            "catalyst_assessment": {"narrative_presence": catalyst_fact["narrative_presence"],
                "earnings_impact": catalyst_fact["earnings_impact"], "event_and_business_reviewed": False,
                "negative_reason_codes": []},
            "turnover_comparison_claim": "LAST_OBSERVED_RELAY_POOL",
            "role_assessment": {role: {"state": None, "reason": ""} for role in e.ROLES},
            "narrative_type_assessment": {"current_trade_narrative_presence": narrative_fact["current_trade_narrative_presence"],
                "content_types_reviewed": False, "unknown_not_negative": True,
                "reason": "", "fund_causality_confirmed": False},
            "short_term_assessment": {"checks": {key: {"state": None, "reason": "", "evidence_refs": [], "counter_claims": []} for key in short_term.CHECKS},
                "alternative_explanations": [], "counterevidence": [], "cancel_conditions": [],
                "confirmation_plan": plans, "role_identity_confirmed": False, "fund_causality_confirmed": False},
            "pairwise_competition_assessment": {"scopes": scopes,
                "price_rank_is_not_core_identity": True, "role_causality_or_entry_confirmed": False}})
    return {"status": "DIAGNOSTIC_TEMPLATE_NO_FORECAST" if diagnostic else "PENDING_AUTHOR_REVIEW_TEMPLATE",
        "formed_at": e.now(), "signal_date": facts["signal_date"], "target_date": facts["target_date"],
        "facts_path": str(Path(facts_path).resolve()), "facts_sha256": e.sha(facts_path),
        "model_exposure": None, "rows": rows, "author_review_completed": False,
        "directions_roles_and_reasons_autofilled": False, "original_prediction_changed": False,
        "diagnostic_is_not_retroactive_expectation": diagnostic,
        "instructions": ["先核完整事实和主线，填写每股方向及支持/反证/取消条件，再声明真实模型暴露。",
            "UNKNOWN需作者写明原因；模板未填，不是已形成的弃判。",
            "角色、催化、逐对竞争、预期差分别审阅；缺资料不能单独看空。",
            "新日期实际审阅完成后改status为AUTHOR_COMPLETED_ROLE_REVIEW、author_review_completed=true并调用原freeze；工具不会自动认证作者或盲测。",
            "诊断模板无事前观察计划，不得用于原日期登记或回填预测。"]}


@e.verified_session
def create_draft(facts_path, output, diagnostic=False):
    facts = e.read(facts_path)
    if not diagnostic:
        original.pre_auction(facts["signal_date"], facts["target_date"], facts["sessions"], original.now())
    e.verify(facts["source_bindings"])
    original.validate_fact_projection(facts)
    catalyst.checked_context(facts)
    check_role_facts(facts)
    if short_term.checked_projection(facts, required=True) is None or narrative.checked(facts) is None:
        raise ValueError("complete short-term and typed narrative preparation required")
    if facts.get("model_scores_masked") is not True:
        raise ValueError("author draft requires original score-masked facts")
    result = _draft_template(facts, facts_path, diagnostic)
    result["source_bindings"] = facts["source_bindings"] | e.bind(facts_path) | e.bind(__file__)
    e.write_new(output, result)
    return result


def check_role_facts(facts):
    e.verify({facts["role_dataset_path"]: facts["role_dataset_sha256"]})
    d = e.load_dataset(facts["role_dataset_path"])
    roles = e.analyse(d, "STRICT")
    bycode = {r["code"]: r for r in roles["rows"]}
    if d["rows"] and {r["signal_date"] for r in d["rows"]} != {facts["signal_date"]}:
        raise ValueError("role facts date mismatch")
    for row in facts["rows"]:
        if row["leader_role_evidence"] != bycode[row["code"]]:
            raise ValueError("role evidence projection changed")
    return roles


def freeze(facts_path, judgements_path, output):
    facts, draft = e.read(facts_path), e.read(judgements_path)
    if draft.get("status") in {"PENDING_AUTHOR_REVIEW_TEMPLATE", "DIAGNOSTIC_TEMPLATE_NO_FORECAST"}:
        raise ValueError("pending or diagnostic template is not an actual completed author judgement")
    if "author_review_completed" in draft and draft["author_review_completed"] is not True:
        raise ValueError("template author review has not been completed")
    roles = check_role_facts(facts)
    short_context = short_term.checked_projection(facts, required=True)
    short_term.validate_assessments(short_context, draft)
    expectation.validate_plans(facts, draft)
    narrative_context = narrative.checked(facts)
    if narrative_context is None:
        raise ValueError("new role freeze requires typed narrative preparation")
    narrative.validate_assessments(narrative_context, draft)
    rows = original.unique_rows(draft["rows"])
    expected = {r["code"]: r for r in facts["rows"]}
    if set(rows) != set(expected):
        raise ValueError("full original judgement cohort required")
    for code, row in rows.items():
        states = row.get("role_assessment", {})
        if set(states) != set(e.ROLES):
            raise ValueError("eleven role assessments including unknown required")
        for role, value in states.items():
            if value.get("state") not in ("CANDIDATE", "EVIDENCE_SUPPORTED", "COMPETING", "INVALIDATED", "UNKNOWN"):
                raise ValueError("unsupported role state")
            evidence = expected[code]["leader_role_evidence"]["roles"][role]
            if evidence["missing"] and value["state"] != "UNKNOWN":
                raise ValueError("missing required role facts cannot be certified")
            if not isinstance(value.get("reason"), str) or not value["reason"].strip():
                raise ValueError("each role needs explicit evidence or unknown reason")
    # Original independent registry, score masking and all catalyst gates remain intact.
    return catalyst.freeze(facts_path, judgements_path, output)


def compare(record, baseline, outcome_archive, output):
    r = e.read(record)
    facts = e.read(r["facts_path"])
    check_role_facts(facts)
    short_context = short_term.checked_projection(facts)
    if short_context is not None:
        short_term.validate_assessments(short_context, r)
    narrative_context = narrative.checked(facts)
    if narrative_context is not None:
        narrative.validate_assessments(narrative_context, r)
    return catalyst.compare(record, baseline, outcome_archive, output)


def confirm(record_path, confirmation_path, output):
    rec, c = e.read(record_path), e.read(confirmation_path)
    if c["signal_date"] != rec["signal_date"] or c["target_date"] != rec["target_date"]:
        raise ValueError("confirmation date mismatch")
    when = e.stamp(c["observed_at"])
    if when.astimezone(original.SHANGHAI).date().isoformat() != c["target_date"] or when > e.stamp(e.now()) or when <= e.stamp(rec["formed_at"]):
        raise ValueError("confirmation must be an actual later target observation")
    receipt = [r for r in original.registrations() if r["path"] == str(Path(record_path).resolve())]
    if len(receipt) != 1 or receipt[0]["sha256"] != e.sha(record_path):
        raise ValueError("independent registration missing or original judgement changed")
    e.verify(rec["source_bindings"])
    if e.sha(rec["facts_path"]) != rec["facts_sha256"]:
        raise ValueError("original fact record changed")
    reviewed = expectation.evaluate_confirmation(rec, c)
    result = {"status": "APPENDED_ROLE_CONFIRMATION_NOT_ORIGINAL_JUDGEMENT_REVISION", "formed_at": e.now(),
        "record_path": str(Path(record_path).resolve()), "record_sha256": e.sha(record_path),
        "confirmation": c, "expectation_review": reviewed,
        "source_bindings": c["source_bindings"] | e.bind(record_path) | e.bind(confirmation_path) | e.bind(expectation.__file__),
        "original_prediction_changed": False}
    e.write_new(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    for k in ("features", "theme-report", "mainline-report", "compatibility", "bank", "calendar", "output-dir"):
        a.add_argument("--"+k, required=True)
    a.add_argument("--role-evidence")
    a.add_argument("--history-dataset", action="append", default=[])
    a.add_argument("--mainline-history", action="append", default=[])
    a.add_argument("--narrative-reviews")
    a = sub.add_parser("draft")
    for k in ("facts", "output"): a.add_argument("--"+k, required=True)
    a.add_argument("--diagnostic", action="store_true")
    a = sub.add_parser("freeze")
    for k in ("facts", "judgements", "output"): a.add_argument("--"+k, required=True)
    a = sub.add_parser("compare")
    for k in ("record", "baseline", "outcome-archive", "output"): a.add_argument("--"+k, required=True)
    a = sub.add_parser("confirm")
    for k in ("record", "confirmation", "output"): a.add_argument("--"+k, required=True)
    a = p.parse_args()
    if a.command == "prepare": r = prepare(a.features, a.theme_report, a.mainline_report, a.compatibility, a.bank, a.calendar, a.output_dir, a.role_evidence, a.history_dataset, a.mainline_history, a.narrative_reviews)
    elif a.command == "draft": r = create_draft(a.facts, a.output, a.diagnostic)
    elif a.command == "freeze": r = freeze(a.facts, a.judgements, a.output)
    elif a.command == "compare": r = compare(a.record, a.baseline, a.outcome_archive, a.output)
    else: r = confirm(a.record, a.confirmation, a.output)
    print(json.dumps({"status": r.get("status", "ROLE_AUGMENTED_FACTS"), "production_changed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
