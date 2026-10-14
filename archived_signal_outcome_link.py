"""Read-only joint index of genuine original-signal and current-target receipts.

Preserves incorrect reacquisitions in their immutable parent; never relabels a
returned date, invents a legacy snapshot, or changes acquisition timestamps.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from next_session_expectation_contract_v271 import load_registered
from public_outcome_sources_v293 import reconstruct, sha, VERSION, ENDPOINTS
from daily_research_contract_v292 import stamp


def link(prior_archive, target_archive, registered_path, output):
    prior = Path(prior_archive).resolve(); target = Path(target_archive).resolve()
    original, old_pools, old_daily, _ = reconstruct(prior)
    current, new_pools, daily, failures = reconstruct(target)
    registered = load_registered(registered_path); plan = registered["plan"]
    signal, day = plan["signal_date"], plan["target_date"]
    if original["target_date"] != signal or (current["signal_date"], current["target_date"]) != (signal, day):
        raise ValueError("parent dates differ from the exact original signal/target")
    if stamp(original["recorded_at"]) > stamp(plan["knowledge_cutoff"]):
        raise ValueError("original complete archive formed after the frozen knowledge cutoff")
    pool = old_pools.get((signal, "limit_up"))
    if not pool or stamp(pool["available_at"]).date().isoformat() != signal:
        raise ValueError("genuine original same-session signal pool required")
    roster = {r["code"]: r["boards"] for r in pool["rows"] if r["boards"] >= 2}
    if roster != {r["code"]: r["boards"] for r in plan["candidates"]}:
        raise ValueError("original complete relay roster disagrees with registration")
    if not {r["code"] for r in pool["rows"]}.issubset(current["requested_codes"]):
        raise ValueError("target archive omits original signal stocks")
    for code in roster:
        left = old_daily.get(code, {}).get("days", {}).get(signal)
        right = daily.get(code, {}).get("days", {}).get(signal)
        if left is not None and right is not None and any(left[k] != right[k] for k in ("open", "high", "low", "close")):
            raise ValueError("original and target unadjusted signal prices conflict")
    original_entries = {(r["kind"], r.get("date")): r for r in original["requests"] if r.get("date") == signal}
    if any((signal, kind) not in old_pools or (kind, signal) not in original_entries for kind in ENDPOINTS):
        raise ValueError("complete original returned-date pool receipts required")
    requests = [original_entries[(r["kind"], signal)] if r.get("date") == signal else r for r in current["requests"]]
    rejected = [r for r in current["requests"] if r.get("date") == signal]
    pools = {**{key: value for key, value in new_pools.items() if key[0] == day},
             **{key: value for key, value in old_pools.items() if key[0] == signal}}
    remaining = [r for r in failures if r["key"][1] != signal]
    normalized = {"pools": {date+"/"+kind: value for (date, kind), value in pools.items()},
                  "daily": daily, "unavailable": remaining}
    paths = [prior/"manifest.json", prior/"normalized.json", target/"manifest.json", target/"normalized.json",
             Path(registered_path).resolve(), Path(__file__).resolve()]
    paths += [Path(r["raw_path"]) for r in rejected if r.get("raw_path")]
    manifest = {**current, "version": VERSION, "recorded_at": datetime.now(timezone.utc).isoformat(),
        "archive_type": "READ_ONLY_JOINT_INDEX_OF_TWO_GENUINE_IMMUTABLE_RECEIPT_ARCHIVES",
        "requests": requests, "source_hashes": {**current["source_hashes"], **{str(p): sha(p) for p in paths}},
        "parent_archives": [{"path": str(p), "manifest_sha256": sha(p/"manifest.json")} for p in (prior, target)],
        "rejected_signal_reacquisition_requests": rejected,
        "rejected_signal_reacquisition_failures": [r for r in failures if r["key"][1] == signal],
        "unavailable_entries": len(remaining), "original_pool_fields_not_rewritten": True,
        "acquisition_times_changed": False, "new_raw_requests": 0}
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    for name, value in (("manifest.json", manifest), ("normalized.json", normalized)):
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    reconstruct(out)  # Existing raw parser independently re-verifies every receipt.
    return {"status": "JOINT_RAW_RECONSTRUCTION_PASSED", "original_candidates": len(roster),
            "preserved_rejected_reacquisitions": len(rejected), "new_raw_requests": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-archive", required=True); parser.add_argument("--target-archive", required=True)
    parser.add_argument("--registered", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(link(args.prior_archive, args.target_archive, args.registered, args.output)))
