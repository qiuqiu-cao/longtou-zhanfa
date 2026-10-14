"""One preregistered retraining ablation: remove the legacy low-turnover proxy.

Original BASE artifacts/predictions remain intact. One challenge, three fixed
segments, same original members/labels, no result-driven rerun or deployment.
"""
import argparse
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import relay_context_research as common
from relay_expert_research import subgroup

VERSION = "relay-proxy-ablation-fixed-1"
ARM = "P1_DROP_PROXY"
FIELDS = [f for f in common.RELAY_FIELDS if f != "signal_one_word"]


def freeze(receipt_path, reproduction_path, base_root, output, started):
    receipt = common.read(receipt_path)
    common.verify(receipt["sources"] | {receipt["prepared_path"]: receipt["prepared_sha256"]})
    reproduced = common.read(reproduction_path)
    if not all(v["passed"] for v in reproduced["modules"].values()):
        raise ValueError("saved original BASE reproduction missing")
    proof = receipt["sources"] | {receipt["prepared_path"]: receipt["prepared_sha256"]}
    for filename in [receipt_path, reproduction_path, __file__, "relay_context_research.py", "relay_expert_research.py",
                     "model_upgrade_history.py", "rolling_forecast_challenger_v274.py", "next_session_forecast_lab_v271.py",
                     Path(base_root)/"relay_history/predictions.csv", Path(base_root)/"models/relay/BASE/model.joblib"]:
        path = Path(filename).resolve(); proof[str(path)] = common.sha(path)
    p = {"version": VERSION, "formed_at": common.now(), "batch_started_at": started,
        "hypothesis": "Legacy low-turnover OHLC proxy dominates non-oneword calls; removing this single field may reduce missed genuine turnover promotions without harming complete-cohort Accuracy.",
        "arms": ["BASE", ARM], "base_fields": common.RELAY_FIELDS, "challenger_fields": FIELDS,
        "removed_field": "signal_one_word", "folds": common.history.FOLDS,
        "training_window": 252, "threshold": .5, "algorithm": "original logistic pipeline, original C=.25 and preprocessing",
        "member_policy": "same complete original mature training members and daily 1/group-size weights; no subgroup removal or target-day inputs",
        "historical_gate": "all three fixed segments Accuracy >=BASE+3pp and original Precision/Recall/F1/coverage/nonword-first guards pass",
        "future_gate": common.GATE, "reused_history_is_blind": False,
        "new_topic_fields_added": False, "forum_evidence_not_training_features": True,
        "expected_effect": "test the proxy dependence hypothesis, not guarantee improvement",
        "stop": "one fit per fixed segment, stop on failure or 30 minutes; no parameter/threshold change or repeat",
        "user_authorization": "重训模型试试，然后再去看看股吧的经验，这个九阳和时代可以得出很多经验的",
        "base_root": str(Path(base_root).resolve()), "receipt_path": str(Path(receipt_path).resolve()),
        "sources": proof, "runtime_versions": receipt["runtime_versions"],
        "production_changed": False, "calibrated_probability": False}
    common.write_new(output, p)
    return p


def load_protocol(path):
    import importlib.metadata
    p = common.read(path)
    if (p["version"] != VERSION or p["arms"] != ["BASE", ARM] or p["challenger_fields"] != FIELDS
            or p["base_fields"] != common.RELAY_FIELDS or p["folds"] != common.history.FOLDS
            or p["training_window"] != 252 or p["threshold"] != .5 or p["future_gate"] != common.GATE):
        raise ValueError("fixed ablation definition changed")
    common.verify(p["sources"])
    if any(importlib.metadata.version(k) != v for k, v in p["runtime_versions"].items()):
        raise ValueError("original dependency versions changed")
    return p


def compare(path):
    p = load_protocol(path); root = Path(path).parent
    if (root/"attempt_STARTED.json").exists():
        raise ValueError("fixed ablation already attempted; no rerun or retune")
    receipt = common.read(p["receipt_path"])
    full = joblib.load(receipt["prepared_path"])["relay"]
    saved = pd.read_csv(Path(p["base_root"])/"relay_history/predictions.csv", dtype={"code": str}, low_memory=False)
    checks = []
    for fold in p["folds"]:
        train = common.history.training_rows(full, fold["start"], "relay")
        expected = next(c for c in receipt["training_checks"] if c["fold"] == fold["id"])
        if (common.history._frame_digest(train, common.RELAY_FIELDS) != expected["training_digest"]
                or train.target_date.max() != expected["training_max_label_date"]):
            raise ValueError("original mature training members, inputs or cutoff differ")
        checks.append({**expected, "challenger_fields": FIELDS,
                       "challenger_training_digest": common.history._frame_digest(train, FIELDS)})
    common.write_new(root/"training_checks_BEFORE_FITS.json", {"formed_at": common.now(), "checks": checks})
    common.write_new(root/"attempt_STARTED.json", {"formed_at": common.now(), "protocol_sha256": common.sha(path)})
    comparisons, rows, logs = {}, [], []
    stopped, september_model = False, None
    for fold in p["folds"]:
        train = common.history.training_rows(full, fold["start"], "relay")
        test = full[full.date.between(fold["start"], fold["end"])].copy()
        base = saved[(saved.arm == "BASE") & (saved.fold == fold["id"])]
        if len(base) != len(test) or set(zip(base.date, base.code)) != set(zip(test.date, test.code)):
            raise ValueError("full same-date same-stock evaluation roster differs")
        br = common.project(base, "BASE", receipt["sessions"])
        bm = common.score(br, receipt["sessions"], fold)
        error, roundtrip = None, None
        try:
            if stopped or (datetime.now(timezone.utc)-datetime.fromisoformat(p["batch_started_at"])).total_seconds() >= 1800:
                raise ValueError("fixed budget or prior fit failure reached")
            # One actual fit. Original preprocessing/C/class threshold unchanged.
            model = common.history._fit(train, FIELDS, "relay")
            pred = common.history._predict(model, test, FIELDS, "relay")
            buffer = BytesIO(); joblib.dump(model, buffer); buffer.seek(0)
            loaded = common.history._predict(joblib.load(buffer), test, FIELDS, "relay")
            roundtrip = common.history.compare_reproduction(loaded, pred, "relay")
            if fold["id"] == "2026Sep":
                september_model = model
        except Exception as exc:
            stopped = True; error = type(exc).__name__+": "+str(exc)
            pred = test.copy(); pred["predicted"], pred["raw_score"], pred["status"] = None, np.nan, "FIT_FAILED_UNKNOWN"
        cr = common.project(pred, ARM, receipt["sessions"])
        label = {(r["signal_date"], r["code"]): r["truth"] for r in br}
        if any(label[(r["signal_date"], r["code"])] != r["truth"] for r in cr):
            raise ValueError("paired candidate truth changed")
        cm = common.score(cr, receipt["sessions"], fold)
        delta = cm["Accuracy"]-bm["Accuracy"] if cm["Accuracy"] is not None else None
        guards = common.guards(bm, cm)
        comparisons[fold["id"]] = {"BASE": bm, ARM: cm, "Accuracy_delta": delta, "guards": guards,
            "passed": not error and delta is not None and delta >= .03 and all(guards.values()),
            "subgroups": {"BASE": subgroup(br, test), ARM: subgroup(cr, test)}, "fit_failure": error}
        rows.extend([{**r, "fold": fold["id"]} for r in br+cr])
        logs.append({"fold": fold["id"], "arm": ARM, "fit_attempts": 0 if stopped and error and "budget" in error else 1,
            "failure": error, "train_members": len(train), "max_training_label_date": train.target_date.max(),
            "save_load_reproduction": roundtrip, "deployment_weights_saved": False})
        pd.DataFrame(rows).to_csv(root/"full_candidate_comparison.csv", index=False, encoding="utf-8-sig")
        common.write_new(root/(fold["id"]+"_checkpoint.json"), {"formed_at": common.now(), "comparison": comparisons[fold["id"]]})
    passed = all(c["passed"] for c in comparisons.values())
    artifact = None
    if passed and september_model is not None:
        artifact = root/"research_only_survivor.joblib"; joblib.dump(september_model, artifact)
    result = {"formed_at": common.now(), "status": "SURVIVOR_REQUIRES_ORIGINAL120_FORWARD_GATES" if passed else "REJECTED_FIXED_PROXY_ABLATION",
        "survivors": [ARM] if passed else [], "comparisons": comparisons, "fit_logs": logs,
        "protocol_sha256": common.sha(path), "predictions_sha256": common.sha(root/"full_candidate_comparison.csv"),
        "model_path": str(artifact.resolve()) if artifact else None, "no_deployable_failed_weights_saved": not passed,
        "BASE_refitted": False, "production_changed": False, "new_mature_forward_samples": 0,
        "historical_is_real_forward": False, "calibrated_Brier": None}
    common.write_new(root/"report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    for field in ("receipt", "reproduction", "base-root", "output", "started"):
        f.add_argument("--"+field, required=True)
    c = sub.add_parser("compare"); c.add_argument("--protocol", required=True)
    args = parser.parse_args()
    result = compare(args.protocol) if args.command == "compare" else freeze(args.receipt, args.reproduction, args.base_root, args.output, args.started)
    import json
    print(json.dumps({k: result[k] for k in ("status", "survivors") if k in result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
