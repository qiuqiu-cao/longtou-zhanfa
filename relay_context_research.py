"""Fixed E1/E2 research. Never amend production predictions or old experiments."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import model_upgrade_history as history
from current_model_diagnostics import RELAY_FIELDS
from cycle_regime_diagnostics import HISTORY_MAP, metrics
from relay_class_balance_research import project, statistics

ENV = ["yesterday_premium", "emotion_quality_score_delta_1", "promotion_mid_delta_1",
       "yesterday_premium_delta_1", "failed_rate_delta_1", "down_limit_count_delta_1", "height_change"]
LEVELS = ["emotion_quality_score", "promotion_mid", "yesterday_premium",
          "failed_rate", "down_limit_count", "max_board"]
RANKS = ["signal_amount_same_height_percentile", "turnover_same_height_percentile"]
INTERACTIONS = [f"context_{rank}_x_{env}" for rank in RANKS for env in ENV]
ARMS = {"E1": RELAY_FIELDS + ENV, "E2": RELAY_FIELDS + ENV + INTERACTIONS}
PARAMETERS = {"C": .25, "max_iter": 2500, "random_state": 271, "solver": "lbfgs",
              "class_weight": None, "window": 252, "threshold": .5}
GATE = {"forward_sessions": 120, "minimum_paired_days": 60, "minimum_coverage": .9,
        "accuracy_increment": .03, "block_sessions": 5, "draws": 2000, "seed": 20261007,
        "tails": [.00625, .99375], "segment_sessions": 40,
        "guards": ["Precision", "complete_positive_Recall", "F1", "coverage"],
        "rollback_sessions": 60, "rollback_accuracy_drop": .05, "rollback_recall_drop": .10}
VERSION = "relay-environment-fixed-1"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(safe(value), stream, ensure_ascii=False, indent=2, allow_nan=False)


def verify(sources):
    for filename, wanted in sources.items():
        if not Path(filename).is_file() or sha(filename) != wanted:
            raise ValueError("source/program hash changed: " + filename)


def load_protocol(path):
    p = read(path)
    if (p.get("version") != VERSION or p.get("arms") != list(ARMS)
            or p.get("fields") != ARMS or p.get("parameters") != PARAMETERS
            or p.get("folds") != history.FOLDS or p.get("gate") != GATE):
        raise ValueError("fixed context definitions or acceptance changed")
    verify(p["sources"])
    return p


def context_features(relay, market, sessions):
    """Join only current and exact preceding official session; never read outcomes."""
    if relay.duplicated(["date", "code"]).any() or market.duplicated("date").any():
        raise ValueError("duplicate signal rows")
    if list(sessions) != sorted(set(sessions)):
        raise ValueError("official sessions must be unique and sorted")
    prior = dict(zip(sessions[1:], sessions[:-1]))
    env = market[["date", "data_status", "yesterday_premium"] +
                 [k for k in LEVELS if k != "yesterday_premium"]].copy()
    for field in LEVELS:
        env[field] = pd.to_numeric(env[field], errors="coerce")
        env.loc[env.data_status.ne("verified"), field] = np.nan
    indexed = env.set_index("date")
    rows = []
    for row in env.to_dict("records"):
        date = row["date"]
        previous = indexed.loc[prior[date]] if prior.get(date) in indexed.index else None
        out = {"date": date, "yesterday_premium": row["yesterday_premium"],
               "context_previous_official_date": prior.get(date),
               "context_source_status": row["data_status"]}
        for level, feature in zip(LEVELS, ENV[1:]):
            out[feature] = row[level] - previous[level] if previous is not None else np.nan
        rows.append(out)
    result = relay.drop(columns=[f for f in ENV if f in relay], errors="ignore").merge(
        pd.DataFrame(rows), on="date", how="left", validate="many_to_one")
    for rank in RANKS:
        known = result[rank].dropna()
        if not known.between(0, 1).all():
            raise ValueError("original full-group percentile outside [0,1]")
        for env_field in ENV:
            result[f"context_{rank}_x_{env_field}"] = (result[rank] - .5) * result[env_field]
    return result


def reproduce(prepared, base_root):
    result = {}
    for module in ("market", "relay"):
        root = Path(base_root)
        manifest = read(root / "models" / module / "BASE" / "manifest.json")
        verify({manifest["model_path"]: manifest["sha256"], **manifest["sources"]})
        fields = prepared[module + "_fields"]
        if fields != manifest["fields"] or len(fields) != (46 if module == "market" else 29):
            raise ValueError("original BASE field definition changed")
        train = history.training_rows(prepared[module], "2026-09-01", module)
        if train.target_date.max() != manifest["training_max_label_date"]:
            raise ValueError("original training cutoff changed")
        saved = pd.read_csv(root / "reproduction" / (module + "_predictions.csv"), dtype={"code": str})
        keys = ["date"] + (["code"] if module == "relay" else [])
        test = prepared[module].merge(saved[keys], on=keys, validate="one_to_one")
        estimator = joblib.load(manifest["model_path"])
        predicted = history._predict(estimator, test, fields, module)
        check = history.compare_reproduction(predicted, saved, module)
        if module == "market":
            for field in [k for k in saved if k.startswith("raw_score_")]:
                drift = float(np.max(np.abs(predicted[field].to_numpy() - saved[field].to_numpy())))
                if drift > 1e-8:
                    raise ValueError("market class raw score mismatch")
        result[module] = {**check, "model_sha256": manifest["sha256"],
                          "fields": fields, "training_digest": history._frame_digest(train, fields),
                          "training_members": len(train), "training_max_label_date": train.target_date.max(),
                          "BASE_refitted": False,
                          "coverage_note": "Final August-31 saved estimators replay all saved reproduction keys. Earlier-fold BASE uses immutable saved predictions; unavailable earlier weights are not refitted."}
    return result


def freeze(receipt_path, base_root, output, batch_start):
    receipt_path, base_root = Path(receipt_path).resolve(), Path(base_root).resolve()
    receipt = read(receipt_path)
    verify(receipt["sources"] | {receipt["prepared_path"]: receipt["prepared_sha256"]})
    sources = receipt["sources"] | {str(receipt_path): sha(receipt_path),
        receipt["prepared_path"]: receipt["prepared_sha256"], str(Path(__file__).resolve()): sha(__file__)}
    for name in ["relay_context_shadow.py", "relay_class_balance_research.py", "cycle_regime_diagnostics.py",
                 "model_upgrade_history.py", "model_upgrade_shadow.py", "model_upgrade_compat.py"]:
        path = Path(name).resolve()
        sources[str(path)] = sha(path)
    for module in ("relay", "market"):
        for rel in [f"models/{module}/BASE/manifest.json", f"models/{module}/BASE/model.joblib",
                    f"reproduction/{module}_predictions.csv"]:
            path = base_root / rel
            sources[str(path)] = sha(path)
    path = base_root / "relay_history/predictions.csv"
    sources[str(path)] = sha(path)
    p = {"version": VERSION, "formed_at": now(), "batch_started_at": batch_start,
         "hypothesis": "Price-volume effects depend on same-day emotional environment and its change.",
         "expected_improvement": "Each segment complete-candidate Accuracy +3pp with all guards; not a promise.",
         "arms": list(ARMS), "fields": ARMS, "parameters": PARAMETERS, "folds": history.FOLDS,
         "gate": GATE, "receipt_path": str(receipt_path), "base_root": str(base_root), "sources": sources,
         "scope": "original mainboard non-ST >=2-board cohort; no first-board samples",
         "unknown_policy": "NaN for absent/nonverified exact previous session; original training imputer",
         "percentile_policy": "Original full signal-date board group; no recomputation on selected rows",
         "training_policy": "same BASE members and target maturity cutoff; no BASE refit, each challenge once per fold",
         "future_policy": "Only historical survivors; separate 120-session window frozen after this protocol and model formation; no migration from three-arm/G1",
         "stop_policy": "one attempt per fold/arm, 30 minutes, convergence failure stops fits; no retuning",
         "production_changed": False, "calibrated_probability": False}
    write_new(output, p)
    return {"status": "FROZEN_BEFORE_FITS", "protocol_sha256": sha(output)}


def fit_once(train, arm):
    from rolling_forecast_challenger_v274 import make
    from sklearn.exceptions import ConvergenceWarning
    if arm not in ARMS or set(train.target.unique()) != {0., 1.}:
        raise ValueError("registered arm and both original classes required")
    estimator = make("logistic")
    weights = 1 / train.groupby("date").date.transform("size")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(train[ARMS[arm]], train.target.astype(int), logisticregression__sample_weight=weights)
    return estimator


def positive_nonword(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["signal_date"], []).append(row)
    selected, unknown = [], 0
    for day, group in sorted(groups.items()):
        if any(r.get("raw_score") is None or r.get("actual_one_word_daily") not in (0., 1.) for r in group):
            unknown += 1
            continue
        eligible = [r for r in group if r["actual_one_word_daily"] == 0. and r["prediction"] is True]
        if eligible:
            selected.append(sorted(eligible, key=lambda r: (-r["raw_score"], r["code"]))[0])
    return {"selected_days": len(selected), "candidate_days": len(groups), "unknown_identity_or_score_days": unknown,
            "coverage": len(selected) / len(groups) if groups else None, "metrics": metrics(selected)}


def score(rows, sessions, fold):
    return {**statistics(rows, sessions, fold), "positive_nonword_first": positive_nonword(rows)}


def guards(baseline, challenge):
    checks = {k: challenge[k] is not None and baseline[k] is not None and challenge[k] >= baseline[k] - 1e-12
              for k in GATE["guards"]}
    bt, ct = baseline["true_nonword_rank_first"], challenge["true_nonword_rank_first"]
    checks["true_nonword_rank_first"] = bt["rate"] is None or (
        ct["rate"] is not None and ct["rate"] >= bt["rate"] - 1e-12 and ct["scoreable_days"] >= bt["scoreable_days"])
    bp, cp = baseline["positive_nonword_first"], challenge["positive_nonword_first"]
    checks["positive_first_coverage"] = cp["selected_days"] >= bp["selected_days"]
    return checks


def compare(protocol_path):
    p = load_protocol(protocol_path)
    root = Path(protocol_path).parent
    if (root / "attempt_STARTED.json").exists():
        raise ValueError("fixed comparison already attempted; do not refit or retune")
    receipt = read(p["receipt_path"])
    if any(importlib.metadata.version(k) != v for k, v in receipt["runtime_versions"].items()):
        raise ValueError("runtime dependency drift")
    prepared = joblib.load(receipt["prepared_path"])
    reproduction = reproduce(prepared, p["base_root"])
    write_new(root / "BASE_reproduction.json", {"formed_at": now(), "modules": reproduction})
    full = context_features(prepared["relay"], prepared["market"], receipt["sessions"])
    saved = pd.read_csv(Path(p["base_root"]) / "relay_history/predictions.csv", dtype={"code": str}, low_memory=False)
    checks = []
    for fold in p["folds"]:
        train = history.training_rows(full, fold["start"], "relay")
        original = next(c for c in receipt["training_checks"] if c["fold"] == fold["id"])
        if (history._frame_digest(train, RELAY_FIELDS) != original["training_digest"]
                or train.target_date.max() != original["training_max_label_date"]):
            raise ValueError("original BASE training member/input/cutoff mismatch")
        checks.append({**original, "expanded_training_digests": {
            arm: history._frame_digest(train, fields) for arm, fields in ARMS.items()}})
    write_new(root / "training_checks_BEFORE_FITS.json", {"formed_at": now(), "checks": checks})
    write_new(root / "attempt_STARTED.json", {"formed_at": now(), "protocol_sha256": sha(protocol_path)})
    comparisons, output, logs = {}, [], []
    stopped = False
    for fold in p["folds"]:
        train = history.training_rows(full, fold["start"], "relay")
        test = full[full.date.between(fold["start"], fold["end"])].copy()
        baseline = saved[(saved.arm == "BASE") & (saved.fold == fold["id"])].copy()
        if len(baseline) != len(test) or set(zip(test.date, test.code)) != set(zip(baseline.date, baseline.code)):
            raise ValueError("complete original candidate roster differs")
        br = project(baseline, "BASE", receipt["sessions"])
        bm = score(br, receipt["sessions"], fold)
        comparisons[fold["id"]] = {"BASE": bm}
        output.extend([{**r, "fold": fold["id"]} for r in br])
        for arm, fields in ARMS.items():
            error = None
            try:
                if stopped or (datetime.now(timezone.utc) - datetime.fromisoformat(p["batch_started_at"])).total_seconds() >= 1800:
                    raise ValueError("fixed budget or prior convergence failure reached")
                estimator = fit_once(train, arm)
                predicted = history._predict(estimator, test, fields, "relay")
                artifact = root / "research_estimators" / fold["id"] / arm / "model.joblib"
                artifact.parent.mkdir(parents=True, exist_ok=False)
                joblib.dump(estimator, artifact)
                loaded = history._predict(joblib.load(artifact), test, fields, "relay")
                history.compare_reproduction(loaded, predicted, "relay")
                write_new(artifact.parent / "manifest.json", {"formed_at": now(), "arm": arm, "fields": fields,
                    "model_path": str(artifact.resolve()), "model_sha256": sha(artifact),
                    "training_digest": history._frame_digest(train, fields), "training_max_label_date": train.target_date.max(),
                    "protocol_sha256": sha(protocol_path), "research_only": True, "deployment_authorized": False})
            except Exception as exc:
                stopped = True
                error = type(exc).__name__ + ": " + str(exc)
                predicted = test.copy()
                predicted["status"], predicted["predicted"], predicted["raw_score"] = "FIT_FAILED_UNKNOWN", None, np.nan
            cr = project(predicted, arm, receipt["sessions"])
            original_truth = {(r["signal_date"], r["code"]): r["truth"] for r in br}
            if any(original_truth[(r["signal_date"], r["code"])] != r["truth"] for r in cr):
                raise ValueError("same-date labels differ")
            cm = score(cr, receipt["sessions"], fold)
            delta = cm["Accuracy"] - bm["Accuracy"] if cm["Accuracy"] is not None else None
            checks_arm = guards(bm, cm)
            positive_coverage_diagnostic = checks_arm.pop("positive_first_coverage")
            comparisons[fold["id"]][arm] = {"metrics": cm, "Accuracy_delta": delta, "guards": checks_arm,
                "historical_screen_passed": error is None and delta is not None and delta >= .03 and all(checks_arm.values()),
                "fit_failure": error, "environment_known_test_rows": {field: int(test[field].notna().sum()) for field in ENV}}
            comparisons[fold["id"]][arm]["positive_first_coverage_diagnostic"] = positive_coverage_diagnostic
            logs.append({"fold": fold["id"], "arm": arm, "fitted_once": error is None, "failure": error,
                         "training_members": len(train), "training_max_label_date": train.target_date.max()})
            output.extend([{**r, "fold": fold["id"]} for r in cr])
        pd.DataFrame(output).to_csv(root / "full_candidate_comparison.csv", index=False, encoding="utf-8-sig")
        write_new(root / (fold["id"] + "_checkpoint.json"), {"formed_at": now(), "comparison": comparisons[fold["id"]]})
    survivors = [arm for arm in ARMS if all(c[arm]["historical_screen_passed"] for c in comparisons.values())]
    result = {"status": "HISTORICAL_SURVIVORS_REQUIRE_NEW_FORWARD" if survivors else "REJECTED_BOTH_FIXED_CHALLENGES",
              "formed_at": now(), "comparisons": comparisons, "fit_logs": logs, "survivors": survivors,
              "protocol_sha256": sha(protocol_path), "predictions_sha256": sha(root / "full_candidate_comparison.csv"),
              "production_changed": False, "new_mature_forward_samples": 0, "calibrated_probability_Brier": None,
              "history_is_real_forward": False, "BASE_refitted": False}
    write_new(root / "report.json", result)
    return result


def stratify(rows, attributes):
    output = {"overall": metrics(rows)}
    for name, key in attributes.items():
        values = sorted({str(r.get(key, "UNKNOWN")) for r in rows})
        output[name] = {v: metrics([r for r in rows if str(r.get(key, "UNKNOWN")) == v]) for v in values}
    return output


def diagnose(protocol_path, forward_path, explanations, output):
    p = load_protocol(protocol_path)
    receipt = read(p["receipt_path"])
    prepared = joblib.load(receipt["prepared_path"])
    saved = pd.read_csv(Path(p["base_root"]) / "relay_history/predictions.csv", dtype={"code": str}, low_memory=False)
    saved = saved[saved.arm == "BASE"]
    rows = project(saved, "BASE", receipt["sessions"])
    linked = saved.set_index(["date", "code"])
    for row in rows:
        source = linked.loc[(row["signal_date"], row["code"])]
        row["regime"] = HISTORY_MAP.get(source.get("emotion_state"), "UNKNOWN")
        actual = row.get("actual_one_word_daily")
        row["true_oneword"] = "UNKNOWN" if actual is None else str(int(actual))
        proxy = source.get("signal_one_word")
        row["old_low_turnover_proxy"] = "UNKNOWN" if pd.isna(proxy) else str(int(proxy))
        row["missing_group"] = "MISSING" if source[RELAY_FIELDS].isna().any() else "COMPLETE"
    source_bindings = {str(Path(forward_path).resolve()): sha(forward_path)}
    forward = read(forward_path)
    verify(forward["source_hashes"])
    frows = forward["forward_rows"]
    detail = []
    truth = {(r["signal_date"], r["code"]): r["truth"] for r in frows}
    for path in explanations:
        explanation = read(path)
        verify(explanation["source_bindings"])
        source_bindings[str(Path(path).resolve())] = sha(path)
        for row in explanation["rows"]:
            terms = row["terms"]
            detail.append({**row, "signal_date": explanation["signal_date"],
                "target_date": explanation["target_date"], "mature_truth": truth.get((explanation["signal_date"], row["code"])),
                "display_call": "模型偏晋级" if row["predicted_positive"] else "模型偏断板",
                "supports": [t for t in terms if t["logit_contribution"] > 0],
                "counterevidence": [t for t in terms if t["logit_contribution"] < 0],
                "confirmation": "目标日竞价、真实换手、同题材反馈待核；不回填原预测"})
    # Missing identities/features in frozen forward plans remain explicit UNKNOWN.
    for r in frows:
        case = next((d for d in detail if (d["signal_date"], d["code"]) == (r["signal_date"], r["code"])), None)
        r["true_oneword"] = "UNKNOWN" if not case or case.get("actual_one_word_daily") is None else str(int(case["actual_one_word_daily"]))
        r["old_low_turnover_proxy"] = "UNKNOWN" if not case else str(next((t["signal_value"] for t in case["terms"] if t["field"] == "signal_one_word"), "UNKNOWN"))
        r["missing_group"] = "UNKNOWN" if not case else "MISSING" if case["missing_fields"] else "COMPLETE"
    attributes = {"by_cycle": "regime", "by_boards": "boards", "by_true_oneword": "true_oneword",
                  "by_old_proxy": "old_low_turnover_proxy", "by_missing": "missing_group"}
    historical = {fold["id"]: stratify([r for r in rows if fold["start"] <= r["signal_date"] <= fold["end"]], attributes)
                  for fold in p["folds"]}
    report = {"formed_at": now(), "historical_diagnostic": historical, "true_forward": stratify(frows, attributes),
              "forward_20_60_all": forward["true_forward"], "full_candidate_explanations": detail,
              "source_bindings": source_bindings, "protocol_sha256": sha(protocol_path),
              "market_target": "original emotion-quality change >0.05 IMPROVE, <-0.05 WORSEN, otherwise CONTINUE; not accuracy for all market/causal forum cycle interpretation",
              "proxy_note": "Low-turnover proxy is not actual daily one-word identity; losing its coefficient does not prove weakness",
              "new_mature_forward_samples": 0, "calibrated_probability_Brier": None}
    write_new(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    f = commands.add_parser("freeze")
    for name in ["receipt", "base-root", "output", "batch-start"]:
        f.add_argument("--" + name, required=True)
    c = commands.add_parser("compare")
    c.add_argument("--protocol", required=True)
    d = commands.add_parser("diagnose")
    for name in ["protocol", "forward", "output"]:
        d.add_argument("--" + name, required=True)
    d.add_argument("--explanation", action="append", required=True)
    for command in ["predict", "settle", "review", "monitor"]:
        p = commands.add_parser(command)
        p.add_argument("--protocol", required=True)
        p.add_argument("--output", required=command != "settle")
        if command == "predict":
            for name in ["features", "calendar", "compatibility"]:
                p.add_argument("--" + name, required=True)
            p.add_argument("--previous-features")
        elif command == "settle":
            for name in ["registration", "archive", "compatibility"]:
                p.add_argument("--" + name, required=True)
        else:
            p.add_argument("--calendar", required=True)
            if command == "monitor":
                p.add_argument("--activation", required=True)
                p.add_argument("--first-signal", required=True)
                p.add_argument("--node", type=int, default=60)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.receipt, args.base_root, args.output, args.batch_start)
    elif args.command == "compare":
        result = compare(args.protocol)
    elif args.command == "diagnose":
        result = diagnose(args.protocol, args.forward, args.explanation, args.output)
    else:
        from relay_context_shadow import run_command
        result = run_command(args)
    print(json.dumps(safe({k: result[k] for k in ["status", "survivors", "new_mature_forward_samples"] if k in result}), ensure_ascii=False))


if __name__ == "__main__":
    main()
