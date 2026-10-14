"""Read-only, full-cohort feedback for saved forum argument reviews.

Promotion labels can check directional compatibility, never role identity,
argument truth, funding causality, or the causal benefit of forum reading.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import forum_reasoning_review as forum
import stock_judgement_review as original


VERSION = "forum-reasoning-outcome-context-1"
CALLS = {"PROMOTE": True, "FAIL": False, "UNKNOWN": None}


def _bound_review(facts_path, review_path):
    facts, review = original.read(facts_path), original.read(review_path)
    path = str(Path(facts_path).resolve())
    if review.get("facts_path") != path or review.get("facts_sha256") != original.sha(path):
        raise ValueError("review must bind its exact forum facts")
    bindings = review.get("source_bindings", {})
    if bindings.get(path) != original.sha(path) or any(bindings.get(p) != h for p, h in facts["source_bindings"].items()):
        raise ValueError("review must retain every original source binding")
    original.check_sources(bindings)
    if any(review.get(k) != facts[k] for k in ("signal_date", "target_date")):
        raise ValueError("review dates differ from exact facts")
    if review.get("model_exposure") != facts["supplement_model_exposure"]:
        raise ValueError("review model exposure changed")
    context = forum.checked(facts)
    forum.validate(context, review, bindings)
    forum.supplement.validate_review(facts, {**review, "author_review_completed": True})
    return facts, review, context


def _baseline(comparison, record, facts):
    matches = []
    for filename in comparison["source_bindings"]:
        if Path(filename).suffix.lower() != ".json":
            continue
        value = original.read(filename)
        if isinstance(value, dict) and isinstance(value.get("relay"), dict) and isinstance(value["relay"].get("candidates"), list) and value.get("feature_sha256"):
            if all(value.get(k) == record[k] for k in ("signal_date", "target_date")):
                matches.append((filename, value))
    if len(matches) != 1:
        raise ValueError("comparison needs one bound original same-date BASE output")
    path, baseline = matches[0]
    if baseline["feature_sha256"] != original.sha(Path(record["feature_archive"]) / "features.json"):
        raise ValueError("BASE uses different original features")
    original.check_sources(baseline["source_hashes"])
    formed = original.stamp(baseline["generated_at"])
    original.pre_auction(record["signal_date"], record["target_date"], facts["sessions"], formed)
    if original.stamp(baseline["knowledge_cutoff"]) > formed or formed > original.stamp(comparison["formed_at"]) or formed > original.now():
        raise ValueError("BASE must use actual pre-auction knowledge and formation times")
    return path, baseline


def _read_outcomes(comparison, record):
    from model_upgrade_compat import runtime
    matches = []
    for filename in comparison["source_bindings"]:
        if Path(filename).name != "manifest.json":
            continue
        manifest = original.read(filename)
        if manifest.get("version") == "v293-public-outcome-receipts-1" and all(manifest.get(k) == record[k] for k in ("signal_date", "target_date")):
            matches.append(Path(filename).parent)
    if len(matches) != 1:
        raise ValueError("comparison needs one bound original target outcome archive")
    inputs, _ = runtime(record["compatibility_policy"])
    outcomes = inputs.build_outcomes(record, matches[0])
    original.check_sources(outcomes["sources"])
    if any(comparison["source_bindings"].get(p) != h for p, h in outcomes["sources"].items()):
        raise ValueError("comparison lost reconstructed outcome source bindings")
    return outcomes["candidates"]


def _comparison(record_path, record, facts, comparison_path):
    comparison = original.read(comparison_path)
    if comparison.get("version") != original.VERSION or comparison.get("status") != "SAME_POOL_SOURCE_BOUND_COMPARISON_ONLY":
        raise ValueError("original full-pool comparison required")
    if any(comparison.get(k) != record[k] for k in ("signal_date", "target_date", "model_exposure")):
        raise ValueError("comparison dates or model exposure differ")
    bindings = comparison.get("source_bindings", {})
    if bindings.get(str(Path(record_path).resolve())) != original.sha(record_path) or any(bindings.get(p) != h for p, h in record["source_bindings"].items()):
        raise ValueError("comparison is not bound to this exact registered record")
    original.check_sources(bindings)
    moment = original.stamp(comparison["formed_at"])
    if moment > original.now() or moment < original.stamp(record["formed_at"]):
        raise ValueError("comparison must be an actual later saved result")
    rows = original.unique_rows(comparison["rows"])
    reviews = original.unique_rows(record["rows"])
    if set(rows) != set(reviews):
        raise ValueError("comparison must retain the entire registered cohort")
    _, baseline = _baseline(comparison, record, facts)
    base = original.unique_rows(baseline["relay"]["candidates"])
    labels = _read_outcomes(comparison, record)
    if set(base) != set(rows) or set(labels) != set(rows):
        raise ValueError("BASE or outcomes lost original candidates")
    independent, base_calls = {}, {}
    for code, row in rows.items():
        independent[code] = CALLS[reviews[code]["promotion_judgement"]]
        base_calls[code] = base[code]["predicted_positive"]
        if type(base_calls[code]) is not bool or (labels[code] is not None and type(labels[code]) is not bool):
            raise ValueError("original boolean directions and labels required")
        if any(row.get(k) is not value for k, value in (("independent_call", independent[code]), ("BASE_call", base_calls[code]), ("actual_promoted", labels[code]))):
            raise ValueError("comparison direction or label differs from bound originals")
    wanted = {"independent": original.metrics(independent, labels), "BASE": original.metrics(base_calls, labels)}
    if comparison.get("metrics") != wanted:
        raise ValueError("comparison metrics disagree with its complete rows")
    expected_breakdown = {}
    for field in ("boards", "actual_one_word_daily"):
        groups = {}
        for row in facts["rows"]:
            groups.setdefault(str(row.get(field)), []).append(row["code"])
        expected_breakdown[field] = {group: {arm: original.metrics({c: calls[c] for c in codes}, {c: labels[c] for c in codes}) for arm, calls in (("independent", independent), ("BASE", base_calls))} for group, codes in groups.items()}
    if comparison.get("breakdown") != expected_breakdown:
        raise ValueError("comparison strata differ from original facts")
    return comparison, rows, baseline["source_hashes"]


def _errors(call, label):
    return {"false_positive": call is True and label is False,
        "false_negative": call is False and label is True,
        "abstention": call is None, "positive_abstention": call is None and label is True,
        "label_unknown": label is None}


def _compatibility(effect, label):
    if effect in {"ROLE_ONLY", "NEUTRAL"}:
        return "NOT_APPLICABLE_ROLE_OR_NEUTRAL"
    if label is None or effect not in {"SUPPORT_PROMOTE", "SUPPORT_FAIL"}:
        return "UNKNOWN"
    return "DIRECTION_COMPATIBLE" if (effect == "SUPPORT_PROMOTE") is label else "DIRECTION_INCOMPATIBLE"


def _rows(facts, review, context, outcomes=None):
    fact_rows, inputs = original.unique_rows(facts["rows"]), original.unique_rows(context["rows"])
    result = []
    for row in review["rows"]:
        code, assessment = row["code"], row["forum_reasoning_assessment"]
        outcome = outcomes[code] if outcomes is not None else {"independent_call": None, "BASE_call": None, "actual_promoted": None}
        arguments = {p["opinion_id"]: p for p in inputs[code]["opinions"]}
        opinions = []
        for opinion in assessment["opinions"]:
            source = arguments[opinion["opinion_id"]]
            author_call = CALLS.get(opinion["author_call"])
            author_applies = opinion["author_call"] != "UNSPECIFIED" and opinion["author_expectation_target_date"] == facts["target_date"]
            same_call = "UNKNOWN_OR_NOT_CURRENT_TARGET"
            if outcomes is not None and author_applies and outcome["independent_call"] is not None:
                same_call = "SAME_DIRECTION" if author_call is outcome["independent_call"] else "DIFFERENT_DIRECTION"
            opinions.append({**deepcopy(opinion), "author": source.get("author"), "source_id": source["source_id"],
                "source_section": source["source_section"], "discussion_published_at": source.get("discussion_published_at"),
                "available_at": source.get("available_at"), "original_excerpt": source["excerpt"],
                **outcome,
                "independent_error_flags": None if outcomes is None else _errors(outcome["independent_call"], outcome["actual_promoted"]),
                "BASE_error_flags": None if outcomes is None else _errors(outcome["BASE_call"], outcome["actual_promoted"]),
                "author_independent_direction_comparison": same_call,
                "direction_compatibility": _compatibility(opinion["independent_effect"], outcome["actual_promoted"]),
                "argument_truth": "UNKNOWN", "role_identity_from_promotion_label": False,
                "condition_effectiveness": "UNKNOWN", "cancellation_effectiveness": "UNKNOWN",
                "confirmation_reason": "NO_INDEPENDENT_SOURCE_BOUND_OPENING_CONFIRMATION_INPUT",
                "forum_causal_benefit_verified": False})
        result.append({"code": code, "name": fact_rows[code]["name"], "boards": fact_rows[code]["boards"],
            "forum_coverage": inputs[code].get("coverage"), "agreement": assessment["agreement"],
            "agreement_reason": assessment["agreement_reason"], "independent_synthesis": assessment["synthesis"],
            **outcome, "independent_error_flags": None if outcomes is None else _errors(outcome["independent_call"], outcome["actual_promoted"]),
            "BASE_error_flags": None if outcomes is None else _errors(outcome["BASE_call"], outcome["actual_promoted"]),
            "outcome_pending": outcomes is None, "error_evaluation_status": "NOT_EVALUATED" if outcomes is None else "SOURCE_BOUND_LABEL_COMPARISON", "opinions": opinions})
    return result


def markdown(report):
    text = ["# 股吧逐观点结算反馈", "", f"{report['signal_date']} → {report['target_date']}；完整 {report['pool_size']} 只、{report['opinion_count']} 条观点。", "",
        "待结算：本页没有新增方向或结果。" if report["outcome_pending"] else "仅诊断观点方向与晋级结果是否兼容，不认证观点真值、角色或论坛阅读的因果增益。",
        "条件与取消条件没有独立开盘确认输入，全部保持 UNKNOWN。", ""]
    for row in report["rows"]:
        text.extend([f"## {row['name']} {row['code']}", "", f"作者分歧：{row['agreement']}；{row['agreement_reason']}", "", row["independent_synthesis"], ""])
        if not report["outcome_pending"]:
            text.extend([f"独立方向 {row['independent_call']}；BASE {row['BASE_call']}；实际晋级 {row['actual_promoted']}；误差 {json.dumps(row['independent_error_flags'], ensure_ascii=False)}。", ""])
        for opinion in row["opinions"]:
            text.extend([f"- {opinion.get('author') or '作者未核'}：{opinion['author_claim']}；{opinion['decision']} / {opinion['independent_effect']} / {opinion['direction_compatibility']}。", f"  可检验预期：{opinion['testable_expectation']}"])
        if not row["opinions"]:
            text.append("限定来源未覆盖观点，保留该股完整分母。")
        text.append("")
    return "\n".join(text)


def _save(facts, review, context, bindings, out, outcomes=None, metrics=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError("new immutable feedback output directory required")
    rows = _rows(facts, review, context, outcomes)
    report = {"version": VERSION, "status": "PENDING_FULL_POOL_ARGUMENT_FEEDBACK_NO_FORECAST" if outcomes is None else "SOURCE_BOUND_ARGUMENT_DIRECTION_FEEDBACK_ONLY",
        "formed_at": original.now().isoformat(), "signal_date": facts["signal_date"], "target_date": facts["target_date"],
        "pool_size": len(rows), "opinion_count": sum(len(r["opinions"]) for r in rows), "rows": rows,
        "outcome_pending": outcomes is None, "original_comparison_metrics": metrics,
        "source_bindings": bindings | original.bind(__file__), "new_fits": 0,
        "forecast_registered": False, "original_prediction_changed": False,
        "argument_truth_verified": False, "forum_causal_benefit_verified": False}
    out.mkdir(parents=True)
    original.write_new(out / "report.json", report)
    with (out / "review.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(markdown(report))
    return report


def project(record_path, comparison_path, output_dir):
    record = original.read(record_path)
    if record.get("version") != original.VERSION or record.get("status") != "INDEPENDENT_PREAUCTION_JUDGEMENT_NOT_PRODUCTION_FREEZE":
        raise ValueError("source-bound registered independent record required")
    receipts = [r for r in original.registrations() if r["path"] == str(Path(record_path).resolve())]
    if len(receipts) != 1 or receipts[0]["sha256"] != original.sha(record_path) or any(receipts[0].get(k) != record[k] for k in ("signal_date", "target_date", "pool_size")):
        raise ValueError("exact original registration receipt required")
    facts, record, context = _bound_review(record["facts_path"], record_path)
    if record.get("pool_size") != len(facts["rows"]):
        raise ValueError("registered full-pool count changed")
    original.validate_fact_projection(facts)
    original.validate_judgements(facts, record)
    original.pre_auction(record["signal_date"], record["target_date"], facts["sessions"], original.stamp(record["formed_at"]))
    comparison, outcomes, baseline_sources = _comparison(record_path, record, facts, comparison_path)
    return _save(facts, record, context, comparison["source_bindings"] | baseline_sources | original.bind(record_path) | original.bind(comparison_path), output_dir, outcomes, deepcopy(comparison["metrics"]))


def pending(facts_path, review_path, output_dir):
    facts, review, context = _bound_review(facts_path, review_path)
    if facts.get("supplement_context_diagnostic_only") is not True or review.get("author_review_completed") is not True or review.get("forecast_registered") is not False or review.get("status") != "COMPLETED_APPEND_ONLY_SUPPLEMENT_REVIEW_NO_FORECAST":
        raise ValueError("completed diagnostic argument review required")
    if any(row.get("promotion_judgement") is not None for row in review["rows"]):
        raise ValueError("pending diagnostic feedback cannot import forecast directions")
    return _save(facts, review, context, review["source_bindings"] | original.bind(facts_path) | original.bind(review_path), output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("project")
    for field in ("record", "comparison", "output-dir"):
        p.add_argument("--" + field, required=True)
    p = sub.add_parser("pending")
    for field in ("facts", "review", "output-dir"):
        p.add_argument("--" + field, required=True)
    a = parser.parse_args()
    result = project(a.record, a.comparison, a.output_dir) if a.command == "project" else pending(a.facts, a.review, a.output_dir)
    print(json.dumps({k: result[k] for k in ("status", "pool_size", "opinion_count", "new_fits", "forecast_registered")}, ensure_ascii=False))
