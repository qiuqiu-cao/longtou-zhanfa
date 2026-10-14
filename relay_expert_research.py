"""One fixed H1/H2 architecture comparison, independently preregistered."""
import argparse
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn.ensemble._hist_gradient_boosting.gradient_boosting as hgb_source

import relay_context_research as common
from relay_expert_model import PARAMETERS, RelayExpertClassifier

VERSION = "relay-proxy-conditional-boosting-fixed-1"
ARMS = ["BASE", "H1", "H2"]


def freeze(receipt_path, base_root, previous_protocol, reproduction, output, started):
    receipt = common.read(receipt_path)
    common.verify(receipt["sources"] | {receipt["prepared_path"]: receipt["prepared_sha256"]})
    common.load_protocol(previous_protocol)
    proof = common.read(reproduction)
    if not all(v["passed"] for v in proof["modules"].values()):
        raise ValueError("original BASE reproduction required")
    sources = receipt["sources"] | {
        str(Path(receipt_path).resolve()): common.sha(receipt_path), receipt["prepared_path"]: receipt["prepared_sha256"],
        str(Path(previous_protocol).resolve()): common.sha(previous_protocol),
        str(Path(reproduction).resolve()): common.sha(reproduction)}
    for filename in [__file__, "relay_expert_model.py", "relay_context_research.py", "relay_class_balance_research.py",
                     "cycle_regime_diagnostics.py", "model_upgrade_history.py", "relay_expert_shadow.py",
                     "relay_context_shadow.py", hgb_source.__file__]:
        path = Path(filename).resolve()
        sources[str(path)] = common.sha(path)
    for module in ("relay", "market"):
        for rel in (f"models/{module}/BASE/manifest.json", f"models/{module}/BASE/model.joblib"):
            path = Path(base_root).resolve() / rel
            sources[str(path)] = common.sha(path)
    path = Path(base_root).resolve() / "relay_history/predictions.csv"
    sources[str(path)] = common.sha(path)
    protocol = {"version": VERSION, "formed_at": common.now(), "batch_started_at": started,
        "hypothesis": "Shared linear price-volume effects overdepend on legacy low-turnover proxy; fixed boosting and separate proxy branches may improve complete-cohort and non-oneword predictions.",
        "arms": ARMS, "fields": common.RELAY_FIELDS, "parameters": PARAMETERS, "folds": common.history.FOLDS,
        "scope": "original mainboard non-ST >=2-board candidates; first-board excluded",
        "training_window": 252, "threshold": .5, "daily_weight": "1/full original labelled signal-date candidate count, computed before branching",
        "preprocessing": "Original median + missing indicators + scaler fitted on whole original training cohort only",
        "routing": "H1 global; H2 global plus proxy=0 and proxy=1 heads, unknown proxy uses global. Proxy is not true one-word identity.",
        "member_policy": "H1 all original members. H2 branches partition original known-proxy members, global uses all; no change of dates/labels",
        "historical_gate": "Each segment Accuracy +3pp; Precision, full Recall, F1, coverage and known-identity true nonword first cannot decrease",
        "future_gate": common.GATE,
        "future_tie_policy": "Higher same-date complete-cohort Accuracy; exact tie H1 simpler global architecture",
        "expected_improvement": "Reduce non-oneword false negatives without sacrificing complete-cohort Accuracy; no promised result",
        "historical_outputs_are_blind": False, "new_forward_samples": 0,
        "stop": "One attempt per fold/arm; no threshold/weight/window search; stop at completion or 30 minutes",
        "base_root": str(Path(base_root).resolve()), "receipt_path": str(Path(receipt_path).resolve()),
        "prior_reproduction_receipt": str(Path(reproduction).resolve()), "sources": sources,
        "runtime_versions": {k: importlib.metadata.version(k) for k in ["scikit-learn", "numpy", "scipy", "pandas", "joblib", "threadpoolctl"]}}
    common.write_new(output, protocol)
    return {"status": "FROZEN_ARCHITECTURE_BEFORE_FITS", "protocol_sha256": common.sha(output)}


def load_protocol(filename):
    p = common.read(filename)
    if (p["version"] != VERSION or p["arms"] != ARMS or p["fields"] != common.RELAY_FIELDS
            or p["parameters"] != PARAMETERS or p["folds"] != common.history.FOLDS
            or p["training_window"] != 252 or p["threshold"] != .5 or p["future_gate"] != common.GATE):
        raise ValueError("fixed architecture definition changed")
    common.verify(p["sources"])
    if any(importlib.metadata.version(k) != v for k, v in p["runtime_versions"].items()):
        raise ValueError("model runtime dependency changed")
    return p


def subgroup(rows, test):
    index = test.set_index(["date", "code"])
    parts = {"true_nonword": [], "true_oneword": [], "unknown_true_identity": [], "proxy0": [], "proxy1": []}
    for row in rows:
        source = index.loc[(row["signal_date"], row["code"])]
        actual = row["actual_one_word_daily"]
        parts["true_nonword" if actual == 0. else "true_oneword" if actual == 1. else "unknown_true_identity"].append(row)
        proxy = source.signal_one_word
        if proxy in (0., 1.): parts["proxy" + str(int(proxy))].append(row)
    return {key: common.metrics(value) for key, value in parts.items()}


def compare(filename):
    p = load_protocol(filename); root = Path(filename).parent
    if (root / "attempt_STARTED.json").exists():
        raise ValueError("architecture comparison already attempted; no refit/retune")
    receipt = common.read(p["receipt_path"])
    full = joblib.load(receipt["prepared_path"])["relay"]
    saved = pd.read_csv(Path(p["base_root"]) / "relay_history/predictions.csv", dtype={"code": str}, low_memory=False)
    training_checks = []
    for fold in p["folds"]:
        train = common.history.training_rows(full, fold["start"], "relay")
        expected = next(c for c in receipt["training_checks"] if c["fold"] == fold["id"])
        if common.history._frame_digest(train, common.RELAY_FIELDS) != expected["training_digest"]:
            raise ValueError("original training input/members changed")
        if train.target_date.max() != expected["training_max_label_date"]:
            raise ValueError("training labels cross original cutoff")
        training_checks.append(expected)
    common.write_new(root / "training_checks_BEFORE_FITS.json", {"formed_at": common.now(), "checks": training_checks})
    common.write_new(root / "attempt_STARTED.json", {"formed_at": common.now(), "protocol_sha256": common.sha(filename)})
    rows, comparisons, logs = [], {}, []
    stopped = False
    for fold in p["folds"]:
        train = common.history.training_rows(full, fold["start"], "relay")
        test = full[full.date.between(fold["start"], fold["end"])].copy()
        base = saved[(saved.arm == "BASE") & (saved.fold == fold["id"])]
        if len(base) != len(test) or set(zip(base.date, base.code)) != set(zip(test.date, test.code)):
            raise ValueError("complete candidate roster differs from original BASE")
        br = common.project(base, "BASE", receipt["sessions"])
        bm = common.score(br, receipt["sessions"], fold)
        comparisons[fold["id"]] = {"BASE": {"metrics": bm, "subgroups": subgroup(br, test)}}
        rows.extend([{**r, "fold": fold["id"]} for r in br])
        for arm in ("H1", "H2"):
            error, branch_members = None, None
            try:
                if stopped or (datetime.now(timezone.utc) - datetime.fromisoformat(p["batch_started_at"])).total_seconds() >= 1800:
                    raise ValueError("prior failure or 30-minute budget reached")
                weights = 1 / train.groupby("date").date.transform("size")
                model = RelayExpertClassifier(arm).fit(train[common.RELAY_FIELDS], train.target, weights)
                branch_members = model.members_by_branch_
                predicted = common.history._predict(model, test, common.RELAY_FIELDS, "relay")
                artifact = root / "research_estimators" / fold["id"] / arm / "model.joblib"
                artifact.parent.mkdir(parents=True, exist_ok=False)
                joblib.dump(model, artifact)
                replay = common.history._predict(joblib.load(artifact), test, common.RELAY_FIELDS, "relay")
                common.history.compare_reproduction(replay, predicted, "relay")
                common.write_new(artifact.parent / "manifest.json", {"formed_at": common.now(), "arm": arm,
                    "fields": common.RELAY_FIELDS, "model_path": str(artifact.resolve()), "model_sha256": common.sha(artifact),
                    "training_digest": common.history._frame_digest(train, common.RELAY_FIELDS),
                    "training_members": len(train), "branches": branch_members,
                    "training_max_label_date": train.target_date.max(), "protocol_sha256": common.sha(filename),
                    "research_only": True, "deployment_authorized": False})
            except Exception as exc:
                stopped = True; error = type(exc).__name__ + ": " + str(exc)
                predicted = test.copy(); predicted["predicted"], predicted["raw_score"], predicted["status"] = None, np.nan, "FIT_FAILED_UNKNOWN"
            cr = common.project(predicted, arm, receipt["sessions"])
            labels = {(r["signal_date"], r["code"]): r["truth"] for r in br}
            if any(r["truth"] != labels[(r["signal_date"], r["code"])] for r in cr):
                raise ValueError("same-date candidate labels changed")
            cm = common.score(cr, receipt["sessions"], fold)
            delta = cm["Accuracy"] - bm["Accuracy"] if cm["Accuracy"] is not None else None
            guards = common.guards(bm, cm); guards.pop("positive_first_coverage")
            comparisons[fold["id"]][arm] = {"metrics": cm, "Accuracy_delta": delta, "guards": guards,
                "historical_screen_passed": not error and delta is not None and delta >= .03 and all(guards.values()),
                "subgroups": subgroup(cr, test), "fit_failure": error}
            logs.append({"fold": fold["id"], "arm": arm, "attempted_once": True, "failure": error,
                         "training_members": len(train), "branch_members": branch_members,
                         "booster_fits": 1 if arm == "H1" else 3, "training_max_label_date": train.target_date.max()})
            rows.extend([{**r, "fold": fold["id"]} for r in cr])
        pd.DataFrame(rows).to_csv(root / "full_candidate_comparison.csv", index=False, encoding="utf-8-sig")
        common.write_new(root / (fold["id"] + "_checkpoint.json"), {"formed_at": common.now(), "comparison": comparisons[fold["id"]]})
    survivors = [a for a in ("H1", "H2") if all(c[a]["historical_screen_passed"] for c in comparisons.values())]
    result = {"formed_at": common.now(), "status": "HISTORICAL_SURVIVORS_REQUIRE_FIXED120_FORWARD" if survivors else "REJECTED_BOTH_ARCHITECTURE_CHALLENGES",
        "survivors": survivors, "comparisons": comparisons, "fit_logs": logs, "production_changed": False,
        "BASE_refitted": False, "protocol_sha256": common.sha(filename),
        "predictions_sha256": common.sha(root / "full_candidate_comparison.csv"),
        "new_mature_forward_samples": 0, "historical_results_are_real_forward": False, "calibrated_Brier": None}
    common.write_new(root / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    f = commands.add_parser("freeze")
    for name in ("receipt", "base-root", "previous-protocol", "reproduction", "output", "started"):
        f.add_argument("--" + name, required=True)
    c = commands.add_parser("compare"); c.add_argument("--protocol", required=True)
    for command in ("predict", "settle", "review", "monitor"):
        sub = commands.add_parser(command)
        sub.add_argument("--protocol", required=True)
        sub.add_argument("--output", required=command != "settle")
        if command == "predict":
            for name in ("features", "calendar", "compatibility"):
                sub.add_argument("--" + name, required=True)
            sub.add_argument("--previous-features")
        elif command == "settle":
            for name in ("registration", "archive", "compatibility"):
                sub.add_argument("--" + name, required=True)
        else:
            sub.add_argument("--calendar", required=True)
            if command == "monitor":
                sub.add_argument("--activation", required=True)
                sub.add_argument("--first-signal", required=True)
                sub.add_argument("--node", type=int, default=60)
    args = parser.parse_args()
    if args.command == "compare": result = compare(args.protocol)
    elif args.command == "freeze": result = freeze(args.receipt, args.base_root,
        args.previous_protocol, args.reproduction, args.output, args.started)
    else:
        from relay_expert_shadow import runtime
        result = runtime(args.protocol).run_command(args)
    import json
    print(json.dumps({k: result[k] for k in ("status", "survivors", "new_mature_forward_samples") if k in result}, ensure_ascii=False))


if __name__ == "__main__": main()
