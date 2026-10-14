"""Once-only fixed archived phase/proxy response, using saved monthly baseline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import relay_context_research as common
from relay_conditional_context import PHASES, phase_context
from relay_recency_walkforward import fit_once, weights, target_signature, groups, paired_interval

ROOT = Path(__file__).resolve().parent / "research/model_upgrade_validation/phase_response_20261013T1649Z"
ARM = "PHASE_PROXY_RESPONSE"
BASE = "ROLL252"
TERMS = ["archived_phase_" + str(i) for i in range(7)]
INTERACTIONS = [term + "_x_legacy_proxy" for term in TERMS]


def features(frame, context):
    rows = context["rows"]
    mapping = {(r["date"], r["code"]): r["phase_feature"] for r in rows}
    if len(mapping) != len(rows) or set(mapping) != set(zip(frame.date, frame.code)):
        raise ValueError("complete original cohort required for archived phase features")
    state = pd.Series([mapping[(r.date, r.code)] for r in frame.itertuples()], index=frame.index)
    if not state.dropna().isin(PHASES).all():
        raise ValueError("unregistered signal phase")
    if set(TERMS + INTERACTIONS) & set(frame.columns):
        raise ValueError("phase feature destination already present")
    result = frame.copy()
    for value, name in zip(PHASES, TERMS):
        result[name] = state.eq(value).astype(float).where(state.notna())
        result[name + "_x_legacy_proxy"] = result[name] * result.signal_one_word
    pd.testing.assert_frame_equal(frame, result[frame.columns])
    return result


def baseline_rows(frame):
    """Restore CSV boolean/unknown values without altering saved original scores."""
    rows = frame.to_dict("records")
    for row in rows:
        for name in ("truth", "prediction"):
            v = row[name]
            if pd.isna(v):
                row[name] = None
            elif v is True or v is False or v in (0, 1):
                row[name] = bool(v)
            else:
                raise ValueError("unexpected saved boolean field")
        for name in ("actual_one_word_daily", "legacy_proxy", "raw_score"):
            row[name] = common.safe(row.get(name))
    return rows


def calendar_sessions(p):
    paths = [path for path in p["source_bindings"] if path.replace("\\", "/").endswith(
        "/data/v298/exchange_calendar_extension_20261003_v2/calendar.json")]
    if len(paths) != 1:
        raise ValueError("one exact source-bound official calendar required")
    calendar = common.read(paths[0])
    if calendar["status"] != "verified" or not calendar.get("source"):
        raise ValueError("verified official calendar required")
    return calendar["sessions"]


def load(protocol_path):
    p = common.read(protocol_path)
    if (p["version"] != "relay-phase-proxy-response-fixed-1" or p["hypothesis_id"] != "ARCHIVED_PHASE_PROXY_RESPONSE"
            or p["baseline_arm"] != BASE or p["challenger_arm"] != ARM
            or Path(p["experiment_root"]) != ROOT or p["phase_states"] != list(PHASES)
            or p["base_fields"] != common.RELAY_FIELDS or p["fields"] != common.RELAY_FIELDS + TERMS + INTERACTIONS
            or p["phase_terms"] != TERMS or p["interaction_terms"] != INTERACTIONS
            or p["monthly_grid"] != [str(m) for m in pd.period_range("2025-07", "2026-09", freq="M")]
            or p["parameters"] != {"kind": "original logistic", "C": .25, "max_iter": 2500,
                "random_state": 271, "class_weight": None, "solver": "lbfgs", "threshold": "strict raw score >0.5",
                "training_window": 252, "weights": "inverse full original training rows per date, total252 unchanged",
                "preprocess": "training-only median/add_indicator/keep_empty_features/scaler original",
                "update_schedule": "monthly labels and signals strictly before month start"}
            or p["execution"]["maximum_new_challenger_fits"] != 15 or p["production_deployment_allowed"] is not False):
        raise ValueError("fixed phase-response hypothesis changed")
    common.verify(p["source_bindings"])
    original = common.read(p["prepared_receipt"])
    common.verify(original["sources"] | {p["prepared_path"]: p["prepared_sha256"]})
    baseline_receipt = common.read(p["baseline_fit_receipt"])
    common.verify(baseline_receipt["sources"])
    for name, version in p["runtime_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("original runtime changed")
    prepared = joblib.load(p["prepared_path"])
    context = common.read(p["phase_context_path"])
    if phase_context(prepared["relay"], prepared["market"]) != context["rows"]:
        raise ValueError("archived signal-phase source projection differs")
    return p, original, prepared, context


def prepare(protocol_path):
    p, original, prepared, context = load(protocol_path)
    full = prepared["relay"]
    changed = features(full, context)
    sessions = calendar_sessions(p)
    # The calendar path is source-bound; no inferred dates enter the training grid.
    start = common.read(ROOT / "execution_START_BEFORE_IMPLEMENTATION.json")["started_at"]
    baseline = pd.read_csv(p["baseline_predictions"], dtype={"code": str})
    baseline = baseline.loc[baseline.arm.eq(BASE)]
    checks = []
    for expected in p["training_checks"]:
        train = common.history.training_rows(changed, expected["cutoff"], "relay")
        if (common.history._frame_digest(train, p["base_fields"]) != expected["training_digest"]
                or common.history._frame_digest(train, p["fields"]) != expected["extended_training_digest"]
                or len(train) != expected["training_n"] or train.date.nunique() != 252
                or train.target_date.max() != expected["training_max_target"]):
            raise ValueError("original monthly members/labels or new phase terms differ")
        test = changed[changed.date.str[:7].eq(expected["month"])]
        b = baseline_rows(baseline.loc[baseline.month.eq(expected["month"])])
        if target_signature(b) != target_signature(common.project(test, ARM, sessions)):
            raise ValueError("saved baseline cohort/labels differ before any fit")
        w = weights(train, expected["cutoff"], sessions, None)
        checks.append({**expected, "weight_digest": common.history.digest(w.tolist()), "weight_sum": float(w.sum())})
    sources = dict(p["source_bindings"])
    sources.update(original["sources"])
    for filename in [protocol_path, __file__, "tests/test_relay_phase_response_research.py", "cycle_regime_diagnostics.py",
                     "relay_recency_walkforward.py", "current_model_diagnostics.py", "relay_class_balance_research.py"]:
        path = Path(filename).resolve()
        sources[str(path)] = common.sha(path)
    receipt = {"formed_at": common.now(), "started_at": start, "protocol_path": str(Path(protocol_path).resolve()),
        "sources": sources, "checks": checks, "actual_challenger_fits": 0, "BASE_ROLL252_refits": 0,
        "new_mature_forward_samples": 0}
    common.write_new(ROOT / "implementation_BEFORE_FITS.json", receipt)
    return receipt


def compare(receipt_path):
    if (ROOT / "attempt_STARTED.json").exists():
        raise ValueError("phase-response comparison already attempted; no repeat fitting")
    e = common.read(receipt_path)
    common.verify(e["sources"])
    p, original, prepared, context = load(e["protocol_path"])
    full = features(prepared["relay"], context)
    sessions = calendar_sessions(p)
    baseline = pd.read_csv(p["baseline_predictions"], dtype={"code": str})
    baseline = baseline_rows(baseline.loc[baseline.arm.eq(BASE)])
    common.write_new(ROOT / "attempt_STARTED.json", {"formed_at": common.now(), "receipt_sha256": common.sha(receipt_path)})
    rows, logs = [], []
    phases = {(r["date"], r["code"]): r["phase_feature"] for r in context["rows"]}
    for row in baseline:
        row["phase_feature"] = phases[(row["signal_date"], row["code"])]
    for expected in e["checks"]:
        if (datetime.now(timezone.utc) - datetime.fromisoformat(e["started_at"])).total_seconds() >= 1800:
            raise ValueError("30-minute comparison limit; preserve completed predictions")
        common.verify(e["sources"])
        month, cutoff = expected["month"], expected["cutoff"]
        train = common.history.training_rows(full, cutoff, "relay")
        test = full[full.date.str[:7].eq(month)].copy()
        w = weights(train, cutoff, sessions, None)
        if common.history.digest(w.tolist()) != expected["weight_digest"]:
            raise ValueError("original day weight changed")
        common.write_new(ROOT / (month + "_fit_STARTED.json"), {"formed_at": common.now(), "fit_attempts": 1,
            "training_digest": expected["extended_training_digest"], "weight_digest": expected["weight_digest"],
            "training_n": len(train), "training_max_target": train.target_date.max(), "cutoff": cutoff})
        try:
            model = fit_once(train, p["fields"], w)
            pred = common.history._predict(model, test, p["fields"], "relay")
            path = ROOT / (month + "_predictions_BEFORE_VALIDATION.csv")
            columns = ["date", "code", "name", "boards", "target_date", "target", "label_valid", "status", "predicted",
                "raw_score", "actual_one_word_daily", "signal_one_word", "emotion_state"]
            pred[columns].to_csv(path, index=False, encoding="utf-8-sig")
            buffer = BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)
            reproduction = common.history.compare_reproduction(
                common.history._predict(joblib.load(buffer), test, p["fields"], "relay"), pred, "relay")
            forecast = common.project(pred, ARM, sessions)
            bm = [r for r in baseline if r["month"] == month]
            if target_signature(forecast) != target_signature(bm):
                raise ValueError("full monthly paired targets/labels changed")
            attrs = test.set_index(["date", "code"])
            for r in forecast:
                source = attrs.loc[(r["signal_date"], r["code"])]
                r.update(month=month, emotion_state="UNKNOWN" if pd.isna(source.emotion_state) else str(source.emotion_state),
                    legacy_proxy=common.safe(source.signal_one_word), model_cutoff=cutoff, train_n=len(train),
                    training_max_target=train.target_date.max(), phase_feature=phases[(r["signal_date"], r["code"])])
            rows.extend(forecast)
            common.write_new(ROOT / (month + "_checkpoint.json"), {"formed_at": common.now(), "n": len(forecast),
                "metrics": common.metrics(forecast), "prediction_sha256": common.sha(path)})
            logs.append({"month": month, "actual_fit_attempts": 1, "full_candidates": len(forecast),
                         "training_n": len(train), "save_load_reproduction": reproduction})
            pd.DataFrame(baseline + rows).to_csv(ROOT / "full_paired_predictions.csv", index=False, encoding="utf-8-sig")
        except Exception as error:
            common.write_new(ROOT / "FAILURE_STOP.json", {"formed_at": common.now(), "month": month,
                "error": type(error).__name__ + ": " + str(error), "refit_allowed": False})
            raise
        print(month, "challenger fits", len(logs), flush=True)
    comparisons = {}
    for fold in p["folds"]:
        b = [r for r in baseline if fold["start"] <= r["signal_date"] <= fold["end"]]
        c = [r for r in rows if fold["start"] <= r["signal_date"] <= fold["end"]]
        bm, cm = common.metrics(b), common.metrics(c)
        bg, cg = groups(b), groups(c)
        for phase in PHASES:
            bg["archived_source_phase:" + phase] = common.metrics([r for r in b if r["phase_feature"] == phase])
            cg["archived_source_phase:" + phase] = common.metrics([r for r in c if r["phase_feature"] == phase])
        guards = {key: bm[key] is not None and cm[key] is not None and cm[key] >= bm[key] - 1e-12
                  for key in ("Precision", "complete_positive_Recall", "F1")}
        guards["same_coverage"] = bm["coverage"] == cm["coverage"]
        guards["positive_call_coverage"] = sum(r["prediction"] is True for r in c) >= sum(r["prediction"] is True for r in b)
        for key in ("complete_positive_Recall", "F1"):
            guards["proxy0_" + key] = bg["proxy0"][key] is not None and cg["proxy0"][key] is not None and cg["proxy0"][key] >= bg["proxy0"][key] - 1e-12
            if fold["id"] == "2026Sep":
                guards["actual_nonword_" + key] = bg["actual_nonword"][key] is not None and cg["actual_nonword"][key] is not None and cg["actual_nonword"][key] >= bg["actual_nonword"][key] - 1e-12
        interval = paired_interval(b, c, sessions, fold, p["fixed_acceptance"]["bootstrap"])
        guards["paired_interval_lower_positive"] = interval["lower"] is not None and interval["lower"] > 0
        delta = cm["Accuracy"] - bm["Accuracy"]
        comparisons[fold["id"]] = {BASE: bm, ARM: cm, "Accuracy_delta": delta, "guards": guards,
            "paired_interval": interval, "passed": delta >= .03 and all(guards.values()), "subgroups": {BASE: bg, ARM: cg}}
    passed = all(r["passed"] for r in comparisons.values())
    survivor = None
    if passed:
        survivor = ROOT / "research_only_survivor.joblib"
        joblib.dump(model, survivor)
    common.verify(e["sources"])
    report = {"formed_at": common.now(), "status": "HISTORICAL_PHASE_SURVIVOR_FORWARD_GATES_REQUIRED" if passed else "REJECTED_ARCHIVED_PHASE_PROXY_RESPONSE",
        "comparisons": comparisons, "actual_challenger_fits": len(logs), "fit_logs": logs,
        "BASE_ROLL252_refits": 0, "deployment_changed": False, "new_mature_forward_samples": 0,
        "historical_is_blind_or_forward": False, "calibrated_Brier": None,
        "research_only_survivor": str(survivor) if survivor else None,
        "full_paired_predictions_sha256": common.sha(ROOT / "full_paired_predictions.csv")}
    common.write_new(ROOT / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--protocol", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--receipt", required=True)
    args = parser.parse_args()
    result = prepare(args.protocol) if args.command == "prepare" else compare(args.receipt)
    print(result.get("status", "PHASE_RESPONSE_SOURCE_READY"))
