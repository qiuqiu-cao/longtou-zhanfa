"""Read-only validation status: original ledgers, full denominators, no upgrade.

Never call shadow review_node, predict, rollback or registration mutations.
Pending forum interpretation is not a forecast, abstention or mature sample.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import source_bound_daily_runner_v293 as runner
import stock_judgement_review as judgement

ROOT = Path(__file__).resolve().parent
VERSION = "relay-validation-status-1"


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def promotion_metrics(results):
    calls, labels = {}, {}
    for result in results:
        registration = result["registration_hash"]
        for row in result["rows"]:
            if row["layer"] != "promotion":
                continue
            key = registration + ":" + row["key"]
            if key in calls:
                raise ValueError("latest complete promotion results contain duplicate candidate")
            calls[key] = row["call"] if row["forecast_made"] else None
            labels[key] = row["truth"] if row["outcome_known"] else None
            if row["outcome_known"] and type(row["truth"]) is not bool:
                raise ValueError("known promotion truth must be boolean")
            if row["forecast_made"] and type(row["call"]) is not bool:
                raise ValueError("promotion forecast must be boolean")
    return judgement.metrics(calls, labels)


def latest_events(events):
    latest, revisions = {}, {}
    for event in events:
        key = event["result"]["registration_hash"]
        revisions[key] = revisions.get(key, 0) + 1
        if event["revision"] != revisions[key]:
            raise ValueError("original settlement revision order changed")
        latest[key] = event
    return latest


def formal_status(registry, ledger):
    errors, registrations = [], {}
    for path in sorted(Path(registry).glob("*.json")):
        try:
            wrapped = runner.load_registered(path)
            if wrapped["sha256"] in registrations:
                raise ValueError("duplicate formal registration")
            registrations[wrapped["sha256"]] = (str(path.resolve()), wrapped)
        except Exception as exc:
            errors.append({"scope": "FORMAL_REGISTRATION", "path": str(path), "error": str(exc)})
    try:
        if not Path(ledger).is_file():
            raise ValueError("original formal settlement ledger unavailable")
        events = runner.read_chain(ledger)
        latest = latest_events(events)
    except Exception as exc:
        return {"status": "INVALID_ORIGINAL_LEDGER", "errors": errors + [{"scope": "FORMAL_LEDGER", "path": str(ledger), "error": str(exc)}],
                "aggregate_withheld": True, "new_mature_samples": 0}
    checked, valid = [], {}
    for event in events:
        result = event["result"]
        item = {"registration_hash": result["registration_hash"], "revision": event["revision"],
                "signal_date": result["signal_date"], "target_date": result["target_date"],
                "archived_event_and_source_hashes_valid": True}
        try:
            _, wrapped = registrations[result["registration_hash"]]
            regenerated = runner.score(wrapped, judgement.read(event["source"]), result["recorded_at"])
            if regenerated != result:
                raise ValueError("original source_score full result differs from saved settlement")
            if {(r["key"]) for r in result["rows"] if r["layer"] == "promotion"} != {r["code"] for r in wrapped["plan"]["candidates"]}:
                raise ValueError("formal promotion layer lost original complete candidates")
            item["status"] = "ORIGINAL_SOURCE_SCORE_RECONSTRUCTION_PASS"
            valid[(result["registration_hash"], event["revision"])] = result
        except Exception as exc:
            item.update(status="INVALID_NEW_RECONSTRUCTION", error=str(exc))
            errors.append({"scope": "FORMAL_RESULT", **item})
        checked.append(item)
    archived = [e["result"] for e in latest.values()]
    qualified = [valid[(key, e["revision"])] for key, e in latest.items() if (key, e["revision"]) in valid]
    withheld = bool(errors) or len(qualified) != len(latest)
    return {"status": "ORIGINAL_FULL_POOL_REVERIFIED" if not withheld else "ATTENTION_REQUIRED_AGGREGATE_WITHHELD",
            "registration_count": len(registrations), "ledger_events": len(events), "latest_settled_signal_days": len(latest),
            "latest_revision_only": True, "result_checks": checked, "errors": errors,
            "archived_hash_bound_metrics": promotion_metrics(archived),
            "reverified_promotion_metrics": promotion_metrics(qualified) if not withheld else None,
            "aggregate_withheld": withheld,
            "pending_registered_signals": [{"signal_date": wrapped["plan"]["signal_date"], "target_date": wrapped["plan"]["target_date"],
                                            "status": "NO_SETTLEMENT_IN_ORIGINAL_LEDGER_SOURCE_STILL_REQUIRED"}
                                           for key, (_, wrapped) in registrations.items() if key not in latest],
            "new_mature_samples": 0, "market_and_execution_layers_not_merged": True}


def independent_status(registry, ledger):
    import forum_reasoning_outcome_context as feedback
    errors, records = [], {}
    for entry in read_lines(registry):
        path = entry["path"]
        item = {"signal_date": entry["signal_date"], "path": path}
        try:
            if judgement.sha(path) != entry["sha256"]:
                raise ValueError("independent registered record hash changed")
            record = judgement.read(path)
            if len(record["rows"]) != entry["pool_size"] or any(record[k] != entry[k] for k in ("signal_date", "target_date", "model_exposure")):
                raise ValueError("independent registration dates/exposure/full pool mismatch")
            judgement.check_sources(record["source_bindings"])
            facts = judgement.read(record["facts_path"])
            if judgement.sha(record["facts_path"]) != record["facts_sha256"]:
                raise ValueError("independent facts hash changed")
            judgement.validate_fact_projection(facts)
            judgement.validate_judgements(facts, record)
            judgement.pre_auction(record["signal_date"], record["target_date"], facts["sessions"], judgement.stamp(record["formed_at"]))
            if path in records:
                raise ValueError("duplicate independent record")
            records[path] = (entry, record, facts)
        except Exception as exc:
            errors.append({"scope": "INDEPENDENT_REGISTRATION", **item, "error": str(exc)})
    checks, latest = [], {}
    for event in read_lines(ledger):
        item = {"signal_date": event["signal_date"], "target_date": event["target_date"], "comparison_path": event["comparison_path"]}
        try:
            entry, record, facts = records[event["record_path"]]
            if event["record_sha256"] != entry["sha256"] or judgement.sha(event["comparison_path"]) != event["comparison_sha256"]:
                raise ValueError("independent settlement hash differs from exact registered record/comparison")
            manifest = Path(event["outcome_archive"]) / "manifest.json"
            if judgement.sha(manifest) != event["outcome_manifest_sha256"]:
                raise ValueError("independent outcome manifest hash changed")
            comparison, rows, _ = feedback._comparison(event["record_path"], record, facts, event["comparison_path"])
            if len(rows) != event["pool_size"] or comparison["metrics"] != event["metrics"]:
                raise ValueError("independent ledger full pool or original metrics differ")
            item.update(status="FULL_SOURCE_ROWS_METRICS_AND_BREAKDOWN_REVERIFIED", metrics=comparison["metrics"], pool_size=len(rows))
            latest[event["record_path"]] = comparison
        except Exception as exc:
            item.update(status="INVALID_NEW_RECONSTRUCTION", error=str(exc))
            errors.append({"scope": "INDEPENDENT_SETTLEMENT", **item})
        checks.append(item)
    calls = {arm: {} for arm in ("independent", "BASE")}; labels = {}
    for path, comparison in latest.items():
        for row in comparison["rows"]:
            key = path + ":" + row["code"]
            labels[key] = row["actual_promoted"]
            calls["independent"][key] = row["independent_call"]; calls["BASE"][key] = row["BASE_call"]
    return {"status": "SEPARATE_INDEPENDENT_LEDGER_REVERIFIED" if not errors else "ATTENTION_REQUIRED_AGGREGATE_WITHHELD",
            "registered_records": len(records), "settled_records": len(latest), "checks": checks, "errors": errors,
            "metrics": {arm: judgement.metrics(values, labels) for arm, values in calls.items()} if not errors else None,
            "pending_registered_signals": [{"signal_date": record["signal_date"], "target_date": record["target_date"], "pool_size": len(record["rows"]),
                                            "status": "NO_SETTLEMENT_IN_THIS_ORIGINAL_INDEPENDENT_LEDGER"}
                                           for path, (_, record, _) in records.items() if path not in latest],
            "original_formal_ledger_new_samples": 0,
            "unregistered_or_supplemental_old_comparisons_automatically_imported": False}


def pending_forum_status(path):
    value = judgement.read(path)
    judgement.check_sources(value["source_bindings"])
    if value.get("status") != "PENDING_FULL_POOL_ARGUMENT_FEEDBACK_NO_FORECAST" or value.get("outcome_pending") is not True:
        raise ValueError("forum pending context must retain pending/no-forecast scope")
    if len(value["rows"]) != value["pool_size"] or len(judgement.unique_rows(value["rows"])) != value["pool_size"]:
        raise ValueError("forum pending full pool changed")
    if value.get("forecast_registered") is not False or value.get("original_comparison_metrics") is not None:
        raise ValueError("pending discussion must not carry a scored forecast")
    for row in value["rows"]:
        if row.get("outcome_pending") is not True or row.get("error_evaluation_status") != "NOT_EVALUATED":
            raise ValueError("pending row cannot be scored as an abstention or error")
        for item in (row, *row.get("opinions", [])):
            if any(item.get(k) is not None for k in ("independent_call", "BASE_call", "actual_promoted", "independent_error_flags", "BASE_error_flags")):
                raise ValueError("pending report contains a forecast, label or error flags")
    return {"status": "FORUM_ARGUMENTS_PENDING_NOT_FORECAST_ABSTENTIONS", "signal_date": value["signal_date"],
            "target_date": value["target_date"], "pool_size": value["pool_size"], "opinion_count": value["opinion_count"],
            "new_formal_mature_samples": 0, "new_forecast_abstentions": 0, "included_in_formal_or_independent_metrics": False}


def protocol_reference(protocol_path):
    p = judgement.read(protocol_path)
    fields = ["forward_sessions", "min_market_pairs", "min_class_support", "min_relay_pairs", "min_coverage", "min_improvement", "historical_results_allow_upgrade", "calibrated_probability_allowed"]
    if p["forward_sessions"] != 120:
        raise ValueError("original fixed forward window changed")
    windows = {}
    for module in ("market", "relay"):
        path = Path(protocol_path).parent / "shadow" / module / "window.json"
        if path.exists():
            window = judgement.read(path)
            if window["protocol_sha256"] != judgement.sha(protocol_path):
                raise ValueError("existing fixed shadow window no longer binds original protocol")
            windows[module] = {"path": str(path), "sha256": judgement.sha(path), **window}
    return {"status": "ORIGINAL_PROTOCOL_REFERENCE_ONLY_NO_ACCEPTANCE_REVIEW", **{k: p[k] for k in fields},
            "path": str(Path(protocol_path).resolve()), "sha256": judgement.sha(protocol_path),
            "model_upgrade_shadow_source": str(ROOT / "model_upgrade_shadow.py"),
            "model_upgrade_shadow_sha256": judgement.sha(ROOT / "model_upgrade_shadow.py"), "unchanged_existing_windows": windows,
            "review_node_predict_rollback_or_shadow_registration_called": False, "new_shadow_opened_or_window_migrated": False}


def build(output):
    output = Path(output).resolve()
    protected = [ROOT / "data/v293/settlement_ledger.jsonl", ROOT / "data/independent_stock_judgement/registry.jsonl",
                 ROOT / "data/independent_stock_judgement/settlement_ledger.jsonl",
                 *sorted((ROOT / "data/v271/expectation_registry").glob("*.json")),
                 ROOT / "research/model_upgrade_validation/fixed_20261007T112017Z/protocol_BEFORE_EFFECTS.json",
                 *sorted((ROOT / "research/model_upgrade_validation/fixed_20261007T112017Z/shadow").rglob("*.json"))]
    before = {str(path): judgement.sha(path) for path in protected}
    report = {"version": VERSION, "formed_at": datetime.now(timezone.utc).isoformat(),
              "formal_promotion": formal_status(ROOT / "data/v271/expectation_registry", ROOT / "data/v293/settlement_ledger.jsonl"),
              "independent_comparison": independent_status(ROOT / "data/independent_stock_judgement/registry.jsonl", ROOT / "data/independent_stock_judgement/settlement_ledger.jsonl"),
              "forum_pending": pending_forum_status(ROOT / "research/model_upgrade_validation/project_audit_20261014/pending_full15/report.json"),
              "fixed_forward_gate": protocol_reference(ROOT / "research/model_upgrade_validation/fixed_20261007T112017Z/protocol_BEFORE_EFFECTS.json"),
              "score_interpretation": "UNCALIBRATED_MODEL_SCORE", "reliable_probability_claim_allowed": False,
              "fit_count": 0, "new_formal_mature_samples": 0, "deployment_or_production_pointer_changed": False}
    if any(judgement.sha(path) != digest for path, digest in before.items()):
        raise ValueError("read-only status mutated an original ledger/registration/window")
    report["protected_original_hashes_unchanged"] = before
    output.mkdir(parents=True, exist_ok=False)
    judgement.write_new(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps({k: result[k] for k in ("version", "formal_promotion", "independent_comparison", "forum_pending", "score_interpretation")}, ensure_ascii=True))
