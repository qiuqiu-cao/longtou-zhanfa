"""One preregistered retrospective Wyckoff-inspired proxy comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from next_session_forecast_lab_v271 import DATA, prepare_relay
from rolling_forecast_challenger_v274 import fit, known_before, trailing
from research.experience_denominator import binary, rank
from research.wyckoff_features import extract_features

BASE = ROOT / "results/v278/september_falsification_20261002_v6"
RISK = "wyckoff_preboard_effort_result_risk"
MISSING = "wyckoff_input_missing"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def interval(daily, slots, column):
    values = daily.set_index("date")[column].reindex(slots).to_numpy(dtype=float)
    rng = np.random.default_rng(20261007)
    starts = rng.integers(0, len(values), size=(5000, int(np.ceil(len(values) / 5))))
    indices = ((starts[:, :, None] + np.arange(5)) % len(values)).reshape(5000, -1)[:, :len(values)]
    draws = values[indices]
    n = np.isfinite(draws).sum(axis=1)
    means = np.nansum(draws, axis=1)[n > 0] / n[n > 0]
    return {"confidence": .95, "bounds": np.quantile(means, [.025, .975]).tolist() if len(means) else None,
            "official_slots": len(slots), "unknown_slots": int(np.isnan(values).sum()),
            "method": "exploratory circular five-session block bootstrap, 5000 draws, seed20261007"}


def load_history_bars(codes):
    parts = []
    columns = ["date", "code", "open", "high", "low", "close", "volume", "source"]
    for name in ["point_in_time_current_daily_bars.csv", "verified_recovered_daily_bars.csv"]:
        for part in pd.read_csv(DATA / name, usecols=columns, dtype={"code": str}, chunksize=150000):
            part["code"] = part.code.str.zfill(6)
            keep = part.code.isin(codes) & part.date.lt("2026-09-01")
            if keep.any():
                parts.append(part.loc[keep].copy())
    return pd.concat(parts, ignore_index=True)


def run(output):
    output = Path(output)
    if (output / "report.json").exists():
        raise FileExistsError("completed immutable comparison already exists")
    protocol_path = output / "protocol_BEFORE_SCORING.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for path, wanted in {**protocol["input_hashes"], **protocol["protected_hashes"]}.items():
        if sha(path) != wanted:
            raise ValueError("preregistered source changed: " + path)
    original_report = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    for path, wanted in original_report["source_hashes"].items():
        if sha(ROOT / path) != wanted:
            raise ValueError("baseline source binding changed: " + path)
    calendar = json.loads((ROOT / "data/v279/exchange_calendar_audit_20261002_v1/calendar.json").read_text(encoding="utf-8"))
    if calendar["status"] != "verified":
        raise ValueError("verified official calendar required")
    sessions = calendar["sessions"]
    successor = dict(zip(sessions[:-1], sessions[1:]))
    market_dates = pd.read_csv(DATA / "market_emotion.csv", usecols=["date"]).date.tolist()
    history, fields = prepare_relay(DATA / "leader_candidates.csv", market_dates)
    train = trailing(known_before(history, "2026-09-01"), 252)
    if len(train) != 3059 or len(fields) != 29:
        raise ValueError("fixed baseline training boundary changed")
    if not train.target_date.eq(train.date.map(successor)).all():
        raise ValueError("training target differs from official successor")
    meta = ["date", "code", "name", "boards", "actual_one_word_daily", "ohlc_status", "source_available_at"]
    test = pd.read_csv(BASE / "relay_predictions_LABEL_ONLY.csv", dtype={"code": str}, usecols=list(dict.fromkeys(meta + fields)))
    if len(test) != 182 or test.duplicated(["date", "code"]).any():
        raise ValueError("original evaluation roster changed")
    history_bars = load_history_bars(set(train.code))
    test_bars = pd.read_csv(ROOT / "data/v290/full_long_interval_20261002_v1/unadjusted_daily_reference.csv")
    test_bars["code"] = test_bars.symbol.str[2:]
    test_bars["source"] = "Tencent_unadjusted_day_same_provider_volume_unit"
    test_bars = test_bars[test_bars.code.isin(test.code)].copy()
    features = {}
    for name, frame, bars in [("training", train, history_bars), ("evaluation", test, test_bars)]:
        observed = extract_features(frame[["date", "code", "boards"]], bars, sessions)
        if len(observed) != len(frame) or observed.duplicated(["date", "code"]).any():
            raise ValueError("feature roster changed")
        observed.to_csv(output / (name + "_features_BEFORE_LABEL_JOIN.csv"), index=False, encoding="utf-8-sig")
        augmented = frame.merge(observed.drop(columns=["boards"]), on=["date", "code"], validate="one_to_one")
        pd.testing.assert_frame_equal(frame[["date", "code"] + fields].reset_index(drop=True),
                                      augmented[["date", "code"] + fields].reset_index(drop=True))
        augmented["wyckoff_risk_model_value"] = augmented[RISK].fillna(0)
        features[name] = augmented
    arms = {"W0": fields, "Wmissing": fields + [MISSING],
            "Wrisk": fields + [MISSING, "wyckoff_risk_model_value"]}
    scored_before = features["evaluation"].copy()
    model_audit = {}
    for arm, columns in arms.items():
        mdl = fit(features["training"], columns, "logistic", "relay")
        scored_before[arm + "_score"] = mdl.predict_proba(scored_before[columns])[:, list(mdl.classes_).index(1)]
        names = mdl.steps[0][1].get_feature_names_out(columns).tolist()
        coefficients = dict(zip(names, mdl.steps[-1][1].coef_[0].tolist()))
        model_audit[arm] = {"input_columns": len(columns), "post_imputer_columns": len(names),
                            "standardized_coefficients": coefficients,
                            "train_rows": len(train), "same_original_values": True}
    reference = pd.read_csv(BASE / "relay_predictions_LABEL_ONLY.csv", dtype={"code": str}, usecols=["date", "code", "probability"])
    match = scored_before.merge(reference, on=["date", "code"], validate="one_to_one")
    error = float((match.W0_score - match.probability).abs().max())
    if error > 1e-9:
        raise ValueError("baseline score reconstruction differs")
    scored_before.to_csv(output / "predictions_BEFORE_LABEL_JOIN.csv", index=False, encoding="utf-8-sig")
    dump(output / "formation_BEFORE_LABEL_JOIN.json", {"recorded_at": datetime.now(timezone.utc).isoformat(),
         "protocol_sha256": sha(protocol_path), "feature_code_sha256": sha(ROOT / "research/wyckoff_features.py"),
         "runner_sha256": sha(__file__), "prediction_sha256": sha(output / "predictions_BEFORE_LABEL_JOIN.csv"),
         "target_columns_absent": not bool({"target", "target_date", "label_valid"} & set(scored_before)),
         "baseline_max_error": error})
    # Only here are evaluation outcomes loaded and joined.
    labels = pd.read_csv(BASE / "relay_predictions_LABEL_ONLY.csv", dtype={"code": str}, usecols=["date", "code", "target_date", "label_valid", "target"])
    valid = labels.label_valid & labels.target.notna()
    if not labels.loc[valid, "target_date"].eq(labels.loc[valid, "date"].map(successor)).all():
        raise ValueError("evaluation target differs from official successor")
    scored = scored_before.merge(labels, on=["date", "code"], how="left", validate="one_to_one")
    daily = []
    for day, group in scored.groupby("date", sort=True):
        complete = bool((group.label_valid & group.target.notna()).all())
        domain = bool(group.actual_one_word_daily.notna().all())
        pool = group[group.actual_one_word_daily.eq(0)]
        row = {"date": day, "registered": len(group), "full_labels_known": complete, "domain_known": domain}
        for arm in arms:
            row.update({arm + "_code": None, arm + "_hit": None, arm + "_positive": False, arm + "_score": None})
            if domain and len(pool):
                pick = rank(pool, arm + "_score").iloc[0]
                row.update({arm + "_code": pick.code, arm + "_positive": bool(pick[arm + "_score"] >= .5),
                            arm + "_score": float(pick[arm + "_score"]), arm + "_hit": int(pick.target) if complete else None})
        for control in ["W0", "Wmissing"]:
            row["Wrisk_minus_" + control] = row["Wrisk_hit"] - row[control + "_hit"] if row["Wrisk_hit"] is not None else None
        daily.append(row)
    daily = pd.DataFrame(daily)
    slots = [d for d in sessions if "2026-09-01" <= d <= "2026-09-30"]
    latest20 = [d for d in sessions if d <= "2026-09-30"][-20:]
    metrics = {}
    for arm in arms:
        complete = daily[daily[arm + "_hit"].notna()]
        positive = complete[complete[arm + "_positive"]]
        metrics[arm] = {"nonword_rank_top1": {"hits": int(complete[arm + "_hit"].sum()), "scorable_dates": len(complete),
            "rate": float(complete[arm + "_hit"].mean()) if len(complete) else None},
            "positive_nonword_top1": {"registered_calls": int(daily[arm + "_positive"].sum()), "scorable_calls": len(positive),
                "hits": int(positive[arm + "_hit"].sum()), "rate": float(positive[arm + "_hit"].mean()) if len(positive) else None},
            "all_candidate_binary": binary(scored, arm + "_score"),
            "nonword_binary": binary(scored[scored.actual_one_word_daily.eq(0)], arm + "_score"),
            "by_height_binary": {str(h): binary(g, arm + "_score") for h, g in scored.groupby("boards")},
            "latest20_historical": binary(scored[scored.date.isin(latest20)], arm + "_score"),
            "latest60_historical": {"status": "UNAVAILABLE_FIXED_OOS_ONLY_SEPTEMBER"},
            "forward20_60": {"status": "UNAVAILABLE_NO_NEW_SETTLED_FORWARD_SAMPLES"}}
    comparisons = {}
    for control in ["W0", "Wmissing"]:
        paired = daily[daily.Wrisk_hit.notna() & daily[control + "_hit"].notna()]
        differences = paired.Wrisk_hit - paired[control + "_hit"]
        common_positive = paired[paired.Wrisk_positive & paired[control + "_positive"]]
        comparisons[control] = {"rank_difference_pp": float(differences.mean() * 100),
             "paired_interval": interval(daily, slots, "Wrisk_minus_" + control),
             "changed_dates": daily.loc[daily.Wrisk_code.ne(daily[control + "_code"]), "date"].tolist(),
             "gained_dates": paired.loc[differences.eq(1), "date"].tolist(),
             "lost_dates": paired.loc[differences.eq(-1), "date"].tolist(),
             "common_positive_call_dates": len(common_positive),
             "common_positive_hits": {"Wrisk": int(common_positive.Wrisk_hit.sum()), control: int(common_positive[control + "_hit"].sum())}}
    coverage = {name: {"total": len(f), "risk_true": int(f[RISK].eq(1).sum()), "risk_false": int(f[RISK].eq(0).sum()),
        "unknown": int(f[RISK].isna().sum()), "available_fraction": float(f[RISK].notna().mean()),
        "risk_true_dates": int(f.loc[f[RISK].eq(1), "date"].nunique()),
        "status_counts": f.feature_status.value_counts().to_dict()} for name, f in features.items()}
    scored.to_csv(output / "all_candidates_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output / "paired_daily_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
    checks = {"risk_rows_at_least10": coverage["evaluation"]["risk_true"] >= 10,
              "risk_dates_at_least5": coverage["evaluation"]["risk_true_dates"] >= 5,
              "feature_coverage_at_least95pct": coverage["evaluation"]["available_fraction"] >= .95,
              "rank_gain_at_least5pp_vs_both": all(c["rank_difference_pp"] >= 5 for c in comparisons.values()),
              "rank_interval_positive_vs_both": all(c["paired_interval"]["bounds"][0] > 0 for c in comparisons.values())}
    for control in ["W0", "Wmissing"]:
        a, c = metrics["Wrisk"]["all_candidate_binary"], metrics[control]["all_candidate_binary"]
        checks["metrics_guard_vs_" + control] = all(a[k] >= c[k] for k in ["accuracy", "precision", "f1_known_labels"]) and a["known_label_recall"] >= c["known_label_recall"] - .02
    for path, wanted in {**protocol["input_hashes"], **protocol["protected_hashes"]}.items():
        if sha(path) != wanted:
            raise ValueError("protected input changed during comparison: " + path)
    report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "status": "FIXED_PROXY_COMPARISON_COMPLETE_DO_NOT_ADD_TO_PRODUCTION",
        "metrics": metrics, "Wrisk_vs_controls": comparisons, "feature_coverage": coverage, "diagnostic_gates": checks,
        "diagnostic_pass": all(checks.values()), "model_audit": model_audit, "baseline_reconstruction_max_error": error,
        "original_candidates_retained": len(scored), "full_label_dates": int(daily.full_labels_known.sum()),
        "official_september_slots": len(slots), "new_forward_samples": 0, "production_model_changed": False,
        "production_admission": False, "entire_Wyckoff_method_tested": False,
        "limitations": ["One frozen effort-result proxy, not complete Wyckoff stages or identified institutional flow",
            "Previously viewed retrospective data, not pristine blind or new forward evidence",
            "Source receipt occurred after historical signal dates; source time certification incomplete",
            "Historical contiguous board labels retained as original proxies; preboard anchor can be stale",
            "Endpoint exclusions do not certify absence of intraday price-limit touches",
            "No calibrated probability, complete unknown-label recall or executable return claim"],
        "protocol_sha256": sha(protocol_path), "feature_code_sha256": sha(ROOT / "research/wyckoff_features.py"),
        "runner_sha256": sha(__file__), "artifact_hashes": {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    dump(output / "report.json", report)
    return {"status": report["status"], "nonword_top1": {k: v["nonword_rank_top1"] for k, v in metrics.items()},
            "all_accuracy": {k: v["all_candidate_binary"]["accuracy"] for k, v in metrics.items()},
            "coverage": coverage, "diagnostic_gates": checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False))
