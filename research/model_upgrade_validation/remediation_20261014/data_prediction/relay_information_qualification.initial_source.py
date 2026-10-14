"""Read-only local parent/board information qualification; never fit or fetch.

Reported event values, temporal availability and a closing order-book snapshot
are different qualifications. Full candidate pools survive missing sources.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import joblib
import pandas as pd

from relay_source_clock_context import clock_context

ROOT = Path(__file__).resolve().parent
VERSION = "relay-information-qualification-1"
FIELDS = ("first_seal", "last_seal", "bomb_count", "sealed_fund", "amount", "float_cap")
MAP = {"代码": "code", "名称": "name", "连板数": "boards", "首次封板时间": "first_seal",
       "最后封板时间": "last_seal", "炸板次数": "bomb_count", "封板资金": "sealed_fund",
       "成交额": "amount", "流通市值": "float_cap", "c": "code", "n": "name",
       "lbc": "boards", "fbt": "first_seal", "lbt": "last_seal", "zbc": "bomb_count",
       "fund": "sealed_fund", "ltsz": "float_cap"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def date_token(value):
    try:
        return pd.Timestamp(str(value)).date().isoformat()
    except (ValueError, TypeError):
        return None


def aware_stamp(value):
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return stamp if stamp.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


def inventory_paths():
    # Freeze a bounded roster of existing files, without network or future data.
    run = subprocess.run(["rg", "-l", '"fbt"|首次封板时间', "data", "results", "-g", "*.json",
                          "-g", "*.csv", "-g", "*.raw"], cwd=ROOT, capture_output=True,
                         encoding="utf-8", check=False)
    if run.returncode not in (0, 1):
        raise RuntimeError(run.stderr)
    paths = []
    for name in run.stdout.splitlines():
        path = ROOT / name
        if (path.name.startswith("zt_pool_") or path.name in {"limit_up.csv", "limit_up.json", "limit_up.raw"}
                or re.fullmatch(r"\d{4}-\d{2}-\d{2}_limit_up\.json", path.name)):
            paths.append(path.resolve())
    return sorted(set(paths))


def receipt_for(path):
    for directory in (path.parent, path.parent.parent):
        for name in ("manifest.json", "snapshot_receipt.json"):
            meta_path = directory / name
            if not meta_path.exists():
                continue
            meta = read(meta_path)
            for entry in meta.get("requests", []) + meta.get("sources", []):
                raw_path = entry.get("raw_path")
                if raw_path and Path(raw_path).resolve() == path.resolve():
                    return meta, meta_path, entry
            if path.suffix == ".csv" and "limit_up" in meta.get("datasets", {}):
                return meta, meta_path, meta["datasets"]["limit_up"]
    return {}, None, {}


def source_rows(path, sessions):
    path = Path(path).resolve()
    meta, receipt_path, entry = receipt_for(path)
    digest = sha(path)
    bound_hash = entry.get("raw_sha256", entry.get("sha256"))
    evidence = {"path": str(path), "sha256": digest,
                "receipt_path": str(receipt_path) if receipt_path else None,
                "receipt_sha256": sha(receipt_path) if receipt_path else None,
                "original_receipt_hash_present": bool(bound_hash),
                "original_receipt_hash_match": bound_hash == digest if bound_hash else None,
                "available_at": entry.get("available_at", entry.get("obtained_at", meta.get("collected_at"))),
                "market_snapshot_at": entry.get("market_snapshot_at"),
                "capture_scope": meta.get("market_session", "SNAPSHOT_TIME_UNKNOWN"),
                "event_semantics": "PROVIDER_REPORTED_NOT_TICK_OR_QUEUE_PROOF"}
    if bound_hash and bound_hash != digest:
        return [], evidence | {"status": "CAPTURE_HASH_MISMATCH"}
    expected = entry.get("date") or entry.get("parameters", {}).get("date")
    name_date = re.search(r"(\d{4}-\d{2}-\d{2}|\d{8})", path.stem)
    if not expected and name_date:
        expected = name_date.group(0)
    if path.suffix == ".csv":
        raw = pd.read_csv(path, dtype={"代码": str})
        if "data_source" in raw and raw.data_source.astype(str).str.contains("lianban", case=False).any():
            return [], evidence | {"status": "RECONSTRUCTED_PLACEHOLDER_NOT_REPORTED_EVENT",
                                  "requested_date": date_token(expected), "rows": len(raw)}
        if not receipt_path:
            return [], evidence | {"status": "CSV_NO_CAPTURE_MANIFEST", "rows": len(raw)}
        day = date_token(meta.get("date"))
        if entry.get("rows") != len(raw) or not len(raw):
            return [], evidence | {"status": "EMPTY_OR_ROW_COUNT_MISMATCH", "date": day}
    else:
        payload = read(path)
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            return [], evidence | {"status": "NOT_ORIGINAL_PROVIDER_POOL"}
        data = payload["data"]
        pool = data.get("pool")
        if payload.get("rc") != 0 or not isinstance(pool, list) or not pool or data.get("tc") != len(pool):
            return [], evidence | {"status": "EMPTY_OR_INCOMPLETE_PROVIDER_POOL"}
        day = date_token(data.get("qdate"))
        if expected and date_token(expected) != day:
            return [], evidence | {"status": "SOURCE_DATE_MISMATCH", "requested_date": date_token(expected),
                                  "response_date": day, "rows": len(pool)}
        raw = pd.DataFrame(pool)
    if day not in sessions:
        return [], evidence | {"status": "NON_SESSION_OR_UNBOUND_DATE", "date": day}
    raw = raw.rename(columns=MAP)
    if not {"code", "name", "boards"}.issubset(raw.columns):
        return [], evidence | {"status": "REQUIRED_IDENTITY_FIELDS_MISSING", "date": day}
    raw["code"] = raw.code.astype(str).str.zfill(6)
    if raw.code.duplicated().any():
        return [], evidence | {"status": "DUPLICATE_PROVIDER_CODE", "date": day}
    original_all_count = len(raw)
    raw = raw[raw.code.str.match(r"^(60|00)\d{4}$") & ~raw.name.astype(str).str.contains("ST|退", regex=True)]
    result = []
    for row in raw.to_dict("records"):
        first = clock_context(row.get("first_seal"))
        last = clock_context(row.get("last_seal"))
        clocks = (first["semantics"] == last["semantics"] == "ORIGINAL_CONTRACT_TIME_ACCEPTED_NO_QUEUE_OR_FILL_PROOF"
                  and last["structural_seconds"] >= first["structural_seconds"])
        numeric = {}
        for field in FIELDS:
            value = pd.to_numeric(row.get(field), errors="coerce")
            numeric[field] = None if pd.isna(value) else float(value)
        bomb_valid = numeric["bomb_count"] is not None and numeric["bomb_count"] >= 0
        result.append({"date": day, "code": row["code"], "name": row["name"], "boards": float(row["boards"]),
                       **numeric, "first_clock_context": first, "last_clock_context": last,
                       "reported_clock_contract_valid": bool(clocks and bomb_valid),
                       "bomb_count_is_reported_zero": numeric["bomb_count"] == 0,
                       "provider_source_path": str(path), "provider_source_sha256": digest,
                       "available_at": evidence["available_at"], "capture_scope": evidence["capture_scope"],
                       "market_snapshot_at": evidence["market_snapshot_at"],
                       "closing_order_book_verified": False,
                       "forward_input_time_qualified": False})
    return result, evidence | {"status": "SOURCE_BOUND_PROVIDER_VALUES_DIAGNOSTIC",
                              "date": day, "all_market_rows": original_all_count, "mainboard_rows": len(result)}


def complete_projection(pool, events):
    """Project without dropping uncovered candidates or preferring a successful source."""
    if pool.duplicated(["date", "code"]).any():
        raise ValueError("duplicate complete pool key")
    groups = defaultdict(list)
    for event in events:
        groups[(event["date"], event["code"])].append(event)
    rows = []
    for original in pool.to_dict("records"):
        key = (original["date"], str(original["code"]).zfill(6))
        candidates = groups[key]
        comparable = [x for x in candidates if x["boards"] == float(original["boards"]) and x["capture_scope"] != "INTRADAY"]
        # Preserve differing snapshots; a closing value is not selected by target outcome.
        distinct = {tuple(x[f] for f in FIELDS) for x in comparable}
        agreed = len(distinct) == 1
        values = comparable[0] if agreed else {}
        rows.append({"date": key[0], "code": key[1], "name": original["name"], "boards": float(original["boards"]),
                     "source_count": len(candidates), "board_matching_nonintraday_source_count": len(comparable),
                     "distinct_event_snapshots": len(distinct), "reported_event_state":
                     "SOURCE_BOUND_REPORTS_AGREE" if agreed else "CONFLICTING_SNAPSHOTS_UNKNOWN" if comparable else "UNKNOWN_NO_QUALIFIED_LOCAL_REPORT",
                     **{f: values.get(f) for f in FIELDS},
                     "reported_clock_contract_valid": bool(agreed and values.get("reported_clock_contract_valid")),
                     "closing_order_book_verified": False, "strict_original_time_qualified": False,
                     "parent_connection_observed": None, "child_continues_parent_break": None,
                     "child_restarts_parent_break": None, "parent_information_state": "UNKNOWN_NO_COMPLETE_HISTORICAL_RELATION_INPUT",
                     "unknown_is_false": False, "training_or_prediction_changed": False})
    return pd.DataFrame(rows)


def future_capture_contract(pool_path, captured_at=None):
    """Full-pool request/validation skeleton; this does not assert data was captured."""
    pool = read(pool_path)
    if len({r["code"] for r in pool["rows"]}) != len(pool["rows"]):
        raise ValueError("full dated candidate pool must be unique")
    return {"version": VERSION, "status": "PENDING_REAL_CAPTURE_NOT_FILLED_DATA",
            "signal_date": pool["signal_date"], "target_date": pool["target_date"],
            "pool_path": str(Path(pool_path).resolve()), "pool_sha256": sha(pool_path),
            "created_at": captured_at or datetime.now(timezone.utc).isoformat(),
            "sources": [], "required_snapshot_semantics": "Record actual provider market snapshot timestamp; captured_at is receive time and not automatically market time",
            "required_fields": list(FIELDS), "required_receipt_fields": ["url", "requested_date", "requested_at", "received_at", "raw_path", "raw_sha256", "http_status", "market_snapshot_at", "field_definitions"],
            "source_priority": "Source semantics and exact dated complete candidate coverage, never known target winners",
            "unknown_policy": "Unknown remains null; original reported bomb_count0 may be real, reconstructed/placeholder0 is not",
            "parent_policy": "Parent anchors may be outside current pool; bind every alternative and explicit break/start observations no later than signal close. Author/comment claims retain separate clocks.",
            "rows": [{"code": r["code"], "name": r["name"], "boards": r["boards"],
                      "reported_event_fields": {f: {"value": None, "status": "UNKNOWN", "source_id": None} for f in FIELDS},
                      "parent_relations": [], "relation_coverage": "UNKNOWN",
                      "promotion_outcome_used_for_extraction": False} for r in pool["rows"]],
            "fit_authorized_by_this_contract": False, "prediction_accuracy_improvement_proved": False}


def validate_capture_contract(document, pool_path):
    pool = read(pool_path)
    if document.get("pool_sha256") != sha(pool_path) or document.get("signal_date") != pool["signal_date"] or document.get("target_date") != pool["target_date"]:
        raise ValueError("complete pool/date binding changed")
    expected = {(r["code"], r["name"], r["boards"]) for r in pool["rows"]}
    actual = {(r["code"], r["name"], r["boards"]) for r in document["rows"]}
    if len(document["rows"]) != len(expected) or actual != expected:
        raise ValueError("full candidate pool must survive missing sources")
    sources = {s["source_id"]: s for s in document["sources"]}
    if len(sources) != len(document["sources"]):
        raise ValueError("duplicate capture source")
    for source in sources.values():
        required = {"url", "requested_date", "requested_at", "received_at", "raw_path", "raw_sha256", "http_status", "market_snapshot_at", "field_definitions"}
        if not required.issubset(source) or not source["field_definitions"]:
            raise ValueError("source semantics and all capture receipt fields must be explicit")
        if sha(source["raw_path"]) != source["raw_sha256"]:
            raise ValueError("raw capture bytes changed")
        if source["requested_date"] != document["signal_date"] or source["http_status"] != 200:
            raise ValueError("wrong source date or failed response")
        requested, received = aware_stamp(source["requested_at"]), aware_stamp(source["received_at"])
        if requested is None or received is None or requested > received:
            raise ValueError("real timezone-aware capture clocks required")
    for row in document["rows"]:
        if row.get("promotion_outcome_used_for_extraction") is not False:
            raise ValueError("target outcomes may not select observed inputs")
        if set(row["reported_event_fields"]) != set(FIELDS):
            raise ValueError("all required fields must be retained, including unknown")
        for field, item in row["reported_event_fields"].items():
            if item["status"] == "UNKNOWN":
                if item["value"] is not None:
                    raise ValueError("unknown values cannot carry placeholder data")
            else:
                if item["status"] not in {"PROVIDER_REPORTED", "PROVIDER_REPORTED_TIME_SEMANTICS_UNKNOWN"}:
                    raise ValueError("unsupported field qualification status")
                source = sources.get(item.get("source_id"))
                if source is None or item.get("original_provider_report") is not True or not item.get("raw_json_pointer"):
                    raise ValueError("nonunknown fields require exact original provider source pointer")
                value = read(source["raw_path"])
                for token in item["raw_json_pointer"].strip("/").split("/"):
                    value = value[int(token)] if isinstance(value, list) else value[token.replace("~1", "/").replace("~0", "~")]
                if value != item["value"]:
                    raise ValueError("field value differs from original provider response")
                if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                    raise ValueError("reported numeric field must be finite and nonnegative")
                if field == "bomb_count" and value != int(value):
                    raise ValueError("reported bomb count must be a nonnegative integer")
                if field in {"first_seal", "last_seal"} and clock_context(value)["structural_seconds"] is None:
                    raise ValueError("reported event cannot carry an invalid or placeholder timestamp")
    return {"status": "CAPTURE_STRUCTURE_VALIDATED_NOT_TRAINING_AUTHORIZATION", "pool_rows": len(expected),
            "source_count": len(sources), "nonunknown_fields": sum(i["status"] != "UNKNOWN" for r in document["rows"] for i in r["reported_event_fields"].values()),
            "closing_queue_or_fill_verified": False, "fit_authorized": False}


def audit(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt_path = ROOT / "research/model_upgrade_validation/relay_context_20261011T151204Z/source_reconstruction_BEFORE_FITS.json"
    receipt = read(receipt_path)
    prepared_path = Path(receipt["prepared_path"])
    if sha(prepared_path) != receipt["prepared_sha256"]:
        raise ValueError("original prepared source changed")
    full = joblib.load(prepared_path)["relay"]
    if len(full) != 9245:
        raise ValueError("original full cohort changed")
    paths = inventory_paths()
    roster = [{"path": str(p), "sha256": sha(p)} for p in paths]
    write(output / "inventory_BEFORE_AUDIT.json", {"version": VERSION, "formed_at": datetime.now(timezone.utc).isoformat(), "scope": "Existing data/results provider-pool files only; no network/fit", "files": roster})
    events, sources = [], []
    for path in paths:
        rows, evidence = source_rows(path, set(receipt["sessions"]))
        events.extend(rows); sources.append(evidence)
    projected = complete_projection(full, events)
    projected.to_csv(output / "complete9245_information_qualification.csv", index=False, encoding="utf-8-sig")
    write(output / "all_provider_event_rows.json", events)
    write(output / "all_source_qualification.json", sources)
    keys = set(zip(projected[projected.reported_event_state.eq("SOURCE_BOUND_REPORTS_AGREE")].date,
                   projected[projected.reported_event_state.eq("SOURCE_BOUND_REPORTS_AGREE")].code))
    overlap = []
    import model_upgrade_history as history
    for fold in history.FOLDS:
        training = history.training_rows(full, fold["start"], "relay")
        matched = training[["date", "code"]].apply(tuple, axis=1).isin(keys)
        overlap.append({"fold": fold["id"], "training_rows": len(training), "known_local_report_keys": int(matched.sum()),
                        "parent_chronology_known_keys": 0, "parent_challenge_input_qualified": False})
    relation_bank = ROOT / "research/model_upgrade_validation/supplement_recognition_20261014T174129Z/qualified_relation_bank_v2.json"
    import supplement_relation_review as relation
    relation_pool = ROOT / "research/model_upgrade_validation/supplement_recognition_20261014T174129Z/evidence_agent/full15_score_free_pool.json"
    current = relation.project(relation_pool, relation_bank)
    relation_proof = {"current_signal_date": current["signal_date"], "current_pool_rows": current["pool_size"],
                      "historical_signal_range": [full.date.min(), full.date.max()],
                      "current_relations_are_old_training_inputs": False,
                      "rows": [{"code": r["code"], "coverage": r["coverage"],
                                "relations": [{"relation_id": e["relation_id"], "recognition_state": e["recognition_state"],
                                               "first_required_current_streak_date": e["timing"].get("current_streak_start"),
                                               "target_window_training_eligible": False,
                                               "reason": "Current Oct14 relation scope requires October observations; Sept30 discussion is grouped background, child initial board outside original relay pool"}
                                              for e in r["relations"]]} for r in current["rows"]]}
    write(output / "parent_protocol_full15_and_historical_coverage.json", relation_proof)
    scaffold = future_capture_contract(relation_pool)
    write(output / "full15_future_information_capture_PENDING.json", scaffold)
    write(output / "future_contract_validation.json", validate_capture_contract(scaffold, relation_pool))
    original_archive = ROOT / "data/v275/reported_seal_archive_20261002_v2/reported_seal_rows.csv"
    old = pd.read_csv(original_archive, dtype={"code": str})
    oldkeys = set(zip(old[old.boards.ge(2)].date, old[old.boards.ge(2)].code))
    newkeys = sorted(keys - oldkeys)
    checks = [{"path": x["path"], "sha256_unchanged": sha(x["path"]) == x["sha256"]} for x in roster]
    if not all(r["sha256_unchanged"] for r in checks):
        raise ValueError("source mutated during audit")
    report = {"version": VERSION, "formed_at": datetime.now(timezone.utc).isoformat(), "fit_count": 0,
              "sources_scanned": len(sources), "source_status_counts": dict(Counter(r["status"] for r in sources)),
              "provider_source_rows_including_alternative_snapshots": len(events),
              "full_pool_rows": len(projected), "full_pool_unique_keys": len(projected[["date", "code"]].drop_duplicates()),
              "qualification_states": projected.reported_event_state.value_counts().to_dict(),
              "old_archive_relay_keys": len(oldkeys), "additional_known_historical_relay_keys": len(newkeys),
              "additional_known_historical_relay_key_list": [list(k) for k in newkeys], "training_overlap": overlap,
              "strict_original_time_complete_parent_chronology_training_keys": 0,
              "parent_numeric_challenge_ready": False, "prediction_accuracy_improvement_proved": False,
              "original_scores_labels_weights_registries_unchanged": True,
              "current_observed_relationships_not_downgraded_to_absence": True,
              "capture_contract_implemented_and_full_pool_scaffold_validated": True,
              "residual_dependencies": ["Historical complete dated discussion relationship inputs and all alternatives before any fit",
                  "Provider definition of09:25:01-04 timestamp semantics", "Closing sealed-fund snapshot time and field definitions",
                  "New future actual observations and complete mature outcomes; cannot be manufactured from tests"],
              "sources": {str(receipt_path): sha(receipt_path), str(prepared_path): sha(prepared_path),
                          str(relation_bank): sha(relation_bank), str(relation_pool): sha(relation_pool),
                          str(original_archive): sha(original_archive), str(Path(__file__).resolve()): sha(__file__)}}
    write(output / "source_immutability_checks.json", checks)
    write(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.output), ensure_ascii=True))
