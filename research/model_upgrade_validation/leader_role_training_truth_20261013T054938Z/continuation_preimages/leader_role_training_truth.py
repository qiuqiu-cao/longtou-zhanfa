"""Append original promotion training truth, never BASE scores or role facts."""
from __future__ import annotations

import argparse
import copy
import json
from functools import lru_cache
from pathlib import Path

import leader_role_engine as e

VERSION = "leader-role-original-promotion-training-truth-1"


def project(frame, dataset, captured, formed):
    """Pure projection; source adapter validation is performed by build_bank."""
    import pandas as pd
    days = e.calendar(dataset["calendar_path"], dataset["calendar_sha256"])
    indexed = {r["candidate_id"]: r for r in dataset["rows"]}
    records = frame.to_dict("records"); rows = {}
    for raw in records:
        signal = str(raw["date"]); code = str(raw["code"]).zfill(6); ident = signal+":"+code
        if ident not in indexed or ident in rows:
            raise ValueError("training truth changes or duplicates original candidate range")
        original = indexed[ident]
        if int(raw["boards"]) != original["boards"]:
            raise ValueError("training truth original board mismatch")
        price = original["groups"]["price"]["value"]
        if any(e.clean(raw.get(k)) != price.get(k) for k in e.RELAY_FIELDS):
            raise ValueError("training truth source differs from original feature allowlist")
        target = e.future_day(days, signal, 1)
        maturity = target+"T15:00:00+08:00"
        exact = all(pd.notna(raw.get(k)) and str(raw[k]) == target for k in ("target_date", "official_target_date"))
        valid = type(raw.get("label_valid")) is bool and raw["label_valid"]
        value = e.number(raw.get("target")); reason = None
        if not exact: reason = "ORIGINAL_TARGET_NOT_EXACT_OFFICIAL_SUCCESSOR"
        elif not valid: reason = "ORIGINAL_LABEL_UNAVAILABLE_OR_INVALID"
        elif value not in (0., 1.): reason = "ORIGINAL_TARGET_UNKNOWN"
        elif e.stamp(maturity) >= e.stamp(captured): reason = "ORIGINAL_LABEL_SOURCE_PRECEDES_MATURITY"
        rows[ident] = {"value": bool(value) if reason is None else None,
            "signal_date": signal, "code": code, "target_date": target,
            "mature_at": maturity, "source_label_available_at": captured,
            "projection_available_at": formed, "available_at": max((captured, formed), key=e.stamp),
            "original_label_valid": valid, "original_label_status": e.clean(raw.get("promotion_label_status")),
            "unknown_reason": reason, "origin": "ORIGINAL_PREPARED_RELAY_LABEL_NOT_BASE_PREDICTION"}
    if set(rows) != set(indexed):
        raise ValueError("training truth omits original candidates including unknowns")
    return rows


def build_bank(dataset, receipt_path, formed):
    import joblib
    receipt = e.read(receipt_path)
    sources = receipt["sources"] | {receipt["prepared_path"]: receipt["prepared_sha256"]} | e.bind(receipt_path)
    e.verify(sources)
    captured = receipt["formed_at"]
    if e.stamp(captured) > e.stamp(formed) or e.stamp(formed) > e.stamp(e.now()):
        raise ValueError("training truth real source/formation time inconsistent")
    frame = joblib.load(receipt["prepared_path"])["relay"]
    return {"version": VERSION, "formed_at": formed,
        "receipt_path": str(Path(receipt_path).resolve()), "source_bindings": sources,
        "rows": project(frame, dataset, captured, formed),
        "targets_in_role_groups": False, "historical_diagnostic_is_forward": False}


@lru_cache(maxsize=4)
def _bank(path, expected):
    e.verify({path: expected})
    bank = e.read(path)
    if bank.get("version") != VERSION: raise ValueError("training truth bank version mismatch")
    return bank


def check_bank(data):
    binding = data["training_truth_binding"]
    path, expected = binding["path"], binding["sha256"]
    if data["source_bindings"].get(path) != expected:
        raise ValueError("training truth bank missing from source bindings")
    e.verify({path: expected}); bank = _bank(path, expected); e.verify(bank["source_bindings"])
    if any(data["source_bindings"].get(p) != h for p, h in bank["source_bindings"].items()):
        raise ValueError("training truth original sources not bound")
    rebuilt = build_bank(data, bank["receipt_path"], bank["formed_at"])
    if bank != rebuilt or data["promotion_training_truth"] != bank["rows"]:
        raise ValueError("training truth differs from original source replay")
    return bank


def load_attachment(data):
    a = data["adapter"]; e.verify({a["parent_path"]: a["parent_sha256"]})
    parent = e.load_dataset(a["parent_path"])
    bank = check_bank(data)
    if e.stamp(data["formed_at"]) < e.stamp(bank["formed_at"]) or e.stamp(data["formed_at"]) > e.stamp(e.now()):
        raise ValueError("training attachment formation is not actual")
    expected = copy.deepcopy(parent)
    expected.update(adapter=a, formed_at=data["formed_at"], training_truth_binding=data["training_truth_binding"],
        promotion_training_truth=bank["rows"], source_bindings=parent["source_bindings"] | bank["source_bindings"] |
        e.bind(a["parent_path"]) | {data["training_truth_binding"]["path"]:data["training_truth_binding"]["sha256"]})
    if data != expected:
        raise ValueError("training truth attachment altered identity facts or BASE predictions")
    return data


@e.verified_session
def attach(dataset_path, receipt_path, output_dir):
    root = Path(output_dir)
    if root.exists(): raise ValueError("training truth output directory must be new")
    parent = e.load_dataset(dataset_path)
    if parent.get("training_truth_binding"): raise ValueError("original training truth already bound")
    formed = e.now(); bank = build_bank(parent, receipt_path, formed)
    root.mkdir(parents=True); path = root/"original_promotion_training_truth.json"; e.write_new(path, bank)
    data = copy.deepcopy(parent)
    data.update(formed_at=formed, adapter={"kind":"PROMOTION_TRAINING_TRUTH_ATTACHMENT",
        "parent_path":str(Path(dataset_path).resolve()), "parent_sha256":e.sha(dataset_path)},
        training_truth_binding={"path":str(path.resolve()), "sha256":e.sha(path)},
        promotion_training_truth=bank["rows"], source_bindings=parent["source_bindings"] | bank["source_bindings"] |
        e.bind(dataset_path) | e.bind(path))
    e.write_new(root/"history_with_original_training_truth.json", data)
    return data


def target(data, candidate_id, cutoff, track):
    cell = data.get("promotion_training_truth", {}).get(candidate_id)
    if not data.get("training_truth_binding") or not cell:
        return None
    binding = data["training_truth_binding"]
    e.verify({binding["path"]:binding["sha256"]})
    bank = _bank(binding["path"], binding["sha256"]); e.verify(bank["source_bindings"])
    if cell != bank["rows"].get(candidate_id):
        raise ValueError("training target altered after source binding")
    # The loader already replays the bound full bank. Admission still checks
    # exact maturity and actual projection availability at every train cutoff.
    if cell.get("value") is None or e.stamp(cell["mature_at"]) >= e.stamp(cutoff): return None
    if track == "STRICT" and (e.stamp(cell["available_at"]) >= e.stamp(cutoff) or
                               e.stamp(cell["projection_available_at"]) >= e.stamp(cutoff)): return None
    return cell["value"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "receipt", "output-dir"): p.add_argument("--"+name, required=True)
    args = p.parse_args(); data = attach(args.dataset, args.receipt, args.output_dir)
    rows = list(data["promotion_training_truth"].values())
    print(json.dumps({"candidate_count":len(rows), "known":sum(r["value"] is not None for r in rows), "fits":0,
        "BASE_predictions_changed":False}, ensure_ascii=False))


if __name__ == "__main__": main()
