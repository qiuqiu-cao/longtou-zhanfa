"""Fifteen named hypotheses, bound to a completed V297 research receipt.

This does not change calls, calibrate probabilities, or promote a rule. The
receipt, bank, experiment inputs and artifacts must all remain available.
"""
from copy import deepcopy
import csv
import json
from pathlib import Path

from daily_research_contract_v292 import stamp
from experience_hypotheses_v297 import HYPOTHESES, TESTED, classified
from next_session_expectation_contract_v271 import digest
from point_context_v295 import sha

VERSION = "v298-fifteen-hypothesis-review-1"
IDS = [item[0] for item in HYPOTHESES]
RECEIPT_STATUS = "FIXED_PROXY_MASKS_LABELS_DATE_BOOTSTRAPS_AND_GRADES_REPRODUCED"


def pending_hypotheses():
    return [{"id": key, "name": name, "definition": definition,
        "grade": "C" if key in {"symbolic_name", "zodiac"} else "D" if key in {"stock_character", "monster_memory"} else "B",
        "status": "UNTESTED", "core_rule": False, "backtest_completed": False,
        "candidate_for_future_confirmation": False,
        "reason": "尚未绑定在知识截止前可获得的完整研究验收；缺数据不等于没有效果"}
        for key, name, definition in HYPOTHESES]


def check_hashes(mapping, base=None):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("research source hashes missing")
    for name, wanted in mapping.items():
        path = Path(name)
        if base is not None and not path.is_absolute():
            path = Path(base) / path
        if not path.is_file() or sha(path) != wanted:
            raise ValueError("research source missing or changed: " + str(path))


def check_coverage(items):
    if not isinstance(items, list) or len(items) != len(IDS) or [r.get("id") for r in items] != IDS:
        raise ValueError("all fifteen hypotheses required in canonical order; no omissions or duplicates")


def load_review(receipt_path):
    receipt_path = Path(receipt_path).resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != RECEIPT_STATUS or (receipt.get("named_hypotheses"), receipt.get("tested_proxies"), receipt.get("unavailable_hypotheses")) != (15, 7, 8):
        raise ValueError("completed fifteen-hypothesis V297 reconstruction receipt required")
    if receipt.get("core_rule_count") != 0 or receipt.get("forward_samples_added") != 0 or receipt.get("stable_60_verified") is not False:
        raise ValueError("research receipt cannot authorize a core upgrade or forward performance claim")
    check_hashes(receipt.get("source_hashes"))
    bank_path = receipt_path.parent / "hypothesis_bank_NOT_LIVE_RULES.json"
    if receipt["source_hashes"].get(str(bank_path)) != sha(bank_path):
        raise ValueError("bank is not bound by the reconstruction receipt")
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("version") != "v297-research-hypothesis-bank-1" or bank.get("core_model_upgrade_allowed") is not False:
        raise ValueError("unsupported bank or trading permissions")
    source = Path(bank["source"]).resolve()
    if receipt["source_hashes"].get(str(source)) != bank.get("source_sha256") or sha(source) != bank.get("source_sha256"):
        raise ValueError("bank experiment source differs from reconstruction receipt")
    experiment = json.loads(source.read_text(encoding="utf-8"))
    check_hashes(experiment.get("source_hashes"))
    check_hashes(experiment.get("artifact_hashes"), source.parent)
    check_coverage(experiment.get("results"))
    check_coverage(bank.get("hypotheses"))
    if experiment.get("core_model_upgrade") is not False or experiment.get("original_predictions_or_labels_rewritten") is not False:
        raise ValueError("experiment cannot change calls or labels")
    expected = []
    for result, (key, name, definition) in zip(experiment["results"], HYPOTHESES):
        if (result.get("id"), result.get("name"), result.get("definition")) != (key, name, definition):
            raise ValueError("experiment definitions differ from fixed protocol")
        if result.get("backtest_completed") is not (key in TESTED):
            raise ValueError("tested versus missing-evidence status changed")
        if key in TESTED:
            views = result["views"]["target_original"]
            grade = classified(views["development"], views["confirmation"], name)
            for field, value in grade.items():
                if result.get(field) != value:
                    raise ValueError("research grade or evidence gate differs: " + key)
        expected.append({field: result[field] for field in ["id", "name", "definition", "grade", "status", "core_rule", "reason", "backtest_completed"]} | {
            "statistical_evidence": {"source": str(source), "source_sha256": sha(source),
                "scope": "original provider >=2-board cohort; historical viewed diagnostic, not forward", "official_labels_certified": False},
            "candidate_for_future_confirmation": result.get("candidate_for_future_confirmation", False)})
    if digest(expected) != digest(bank["hypotheses"]):
        raise ValueError("bank entries differ from audited statistical results")
    if any(r["core_rule"] is not False for r in expected):
        raise ValueError("unpromoted research cannot become core rules")
    if stamp(experiment["recorded_at"]) > stamp(bank["available_at"]) or stamp(bank["available_at"]) > stamp(receipt["recorded_at"]):
        raise ValueError("research availability clocks inconsistent")
    with (source.parent / "cohort_LABEL_ONLY.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != experiment["candidate_rows"] or len({r["date"] for r in rows}) != experiment["candidate_dates"]:
        raise ValueError("research denominator differs from audited cohort")
    known = [r for r in rows if r["label_valid"] == "True" and r["target_original"] in {"0.0", "1.0"}]
    if not known:
        raise ValueError("research has no realized labels")
    return {"receipt": str(receipt_path), "receipt_sha256": sha(receipt_path),
        "bank": str(bank_path), "bank_sha256": sha(bank_path), "available_at": receipt["recorded_at"],
        "max_label_date": max(r["target_date"] for r in known), "hypotheses": expected}


def expected_binding(review):
    return {"version": VERSION, **{k: review[k] for k in ["receipt", "receipt_sha256", "bank", "bank_sha256", "available_at", "max_label_date"]},
        "core_rules_allowed": False, "probability_authorized": False,
        "historical_statistics_are_forward_results": False}


def check_available(plan, review):
    cutoff = stamp(plan.get("knowledge_cutoff", plan["evidence_cutoff"]))
    if stamp(review["available_at"]) > cutoff:
        raise ValueError("hypothesis research receipt arrived after knowledge cutoff")
    if review["max_label_date"] > plan["signal_date"] or review["max_label_date"] >= plan["target_date"]:
        raise ValueError("hypothesis research contains labels after signal date or in target session")


def attach(plan, receipt_path):
    result = deepcopy(plan)
    ext = result.get("daily_research")
    if not ext or ext.get("draft_status") != "NEEDS_FACTS_AND_RESEARCH_CALLS":
        raise ValueError("only an unreviewed new draft can receive hypothesis research")
    if "hypothesis_binding" in result:
        raise ValueError("do not overwrite an existing hypothesis binding")
    if ext.get("hypothesis_policy") != VERSION or digest(ext.get("hypotheses")) != digest(pending_hypotheses()):
        raise ValueError("start from the complete untested V298 template; preserve existing research separately")
    review = load_review(receipt_path)
    check_available(result, review)
    ext["hypotheses"] = deepcopy(review["hypotheses"])
    result["hypothesis_binding"] = expected_binding(review)
    return result


def validate_hypotheses(plan, require_complete=False):
    ext = plan.get("daily_research")
    binding = plan.get("hypothesis_binding")
    if not ext:
        if binding is not None or require_complete:
            raise ValueError("hypothesis binding requires the full daily research extension")
        return {"status": "LEGACY_NO_HYPOTHESIS_EXTENSION", "backfilled": False}
    strict = require_complete or ext.get("hypothesis_policy") == VERSION or binding is not None
    if not strict:
        if any(r.get("id") in IDS or r.get("backtest_completed") or r.get("statistical_evidence") for r in ext.get("hypotheses", [])):
            raise ValueError("statistical hypothesis claims require the complete policy and source binding")
        return {"status": "LEGACY_HYPOTHESES_UNCHANGED", "backfilled": False}
    if ext.get("hypothesis_policy") != VERSION:
        raise ValueError("new freezes require the fifteen-hypothesis policy")
    check_coverage(ext.get("hypotheses"))
    if binding is None:
        if digest(ext["hypotheses"]) != digest(pending_hypotheses()):
            raise ValueError("without an available research binding every hypothesis must remain untested")
        return {"status": "FIFTEEN_UNTESTED_HYPOTHESES_REPRESENTED", "backfilled": False}
    if not isinstance(binding, dict):
        raise ValueError("invalid hypothesis binding")
    receipt_path = Path(binding.get("receipt", ""))
    if not receipt_path.is_file() or sha(receipt_path) != binding.get("receipt_sha256"):
        raise ValueError("hypothesis receipt missing or hash mismatch")
    review = load_review(receipt_path)
    check_available(plan, review)
    if digest(binding) != digest(expected_binding(review)):
        raise ValueError("hypothesis binding clocks, scope or permissions differ from source")
    if digest(ext["hypotheses"]) != digest(review["hypotheses"]):
        raise ValueError("hypothesis grades or evidence differ from audited source")
    return {"version": VERSION, "status": "FIFTEEN_HYPOTHESES_REBUILT_FROM_AUDITED_SOURCE",
        "tested_proxies": 7, "unavailable_hypotheses": 8, "core_rules": 0,
        "probability_authorized": False, "backfilled": False}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.plan)
    if "expectation_registry" in source.parts:
        raise ValueError("never attach new evidence to a registered original")
    plan = json.loads(source.read_text(encoding="utf-8"))
    value = attach(plan, args.receipt)
    validate_hypotheses(value, require_complete=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    print(json.dumps({"status": "DRAFT_ONLY_NOT_REGISTERED", "hypotheses": 15, "core_rules": 0, "output": str(output.resolve())}, ensure_ascii=True))


if __name__ == "__main__":
    main()
