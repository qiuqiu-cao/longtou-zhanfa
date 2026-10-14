"""Fixed retrospective denominator-state comparison, never a production update.

Only two original feature columns change. The original ratio is retained when
admissible; no replacement daily volume, new theme or future observation is used.
"""
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
from rolling_forecast_challenger_v274 import known_before, trailing, fit

BASE = ROOT / "results/v278/september_falsification_20261002_v6"
CHANGED = ["turnover_vs_prev", "turnover_vs_prev_same_height_percentile"]
CONTEXT = ["prior_date", "prior_turnover", "prior_one_word_proxy",
           "official_previous_date", "prior_is_previous_session"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def context_features(rows, observations, sessions):
    """Bind the last finite turnover strictly before each requested signal day.

    Missing turnover never updates the original dictionary, but zero does.
    No outcome, same-day or future observation is accessed as a denominator.
    """
    keys = ["date", "code"]
    if rows.duplicated(keys).any() or observations.duplicated(keys).any():
        raise ValueError("duplicate observation or requested key")
    if list(sessions) != sorted(set(sessions)):
        raise ValueError("official sessions must be unique and chronological")
    if not set(rows.date).issubset(sessions):
        raise ValueError("signal date absent from official calendar")
    previous = dict(zip(sessions[1:], sessions[:-1]))
    obs = observations[keys + ["turnover", "one_word_proxy"]].sort_values(keys)
    events = list(obs.itertuples(index=False))
    position, latest, records = 0, {}, []
    for day, requested in rows.groupby("date", sort=True):
        while position < len(events) and events[position].date < day:
            event = events[position]
            if pd.notna(event.turnover) and np.isfinite(event.turnover):
                latest[event.code] = (event.date, float(event.turnover), event.one_word_proxy)
            position += 1
        for code in requested.code:
            prior_day, turnover, proxy = latest.get(code, (None, np.nan, np.nan))
            records.append({"date": day, "code": code, "prior_date": prior_day,
                            "prior_turnover": turnover, "prior_one_word_proxy": proxy,
                            "official_previous_date": previous.get(day),
                            "prior_is_previous_session": bool(prior_day is not None
                                and prior_day == previous.get(day))})
    return rows[keys].merge(pd.DataFrame(records), on=keys, how="left", validate="one_to_one")


def sanitize_ratio(rows, context, exclude_word=False):
    """Preserve every row and every other feature; recompute this one rank."""
    frame = rows.merge(context, on=["date", "code"], how="left", validate="one_to_one")
    if len(frame) != len(rows) or frame["prior_is_previous_session"].isna().any():
        raise ValueError("denominator context must cover every candidate")
    frame.index = rows.index
    ratio = pd.to_numeric(frame.turnover_vs_prev, errors="coerce")
    good = (frame.prior_is_previous_session & frame.prior_turnover.gt(0)
            & np.isfinite(frame.prior_turnover) & np.isfinite(ratio) & ratio.ge(0))
    if exclude_word:
        good &= frame.prior_one_word_proxy.eq(0)
    frame["ratio_retained"] = good
    frame[CHANGED[0]] = ratio.where(good)
    frame[CHANGED[1]] = frame.groupby(["date", "boards"])[CHANGED[0]].rank(pct=True)
    return frame


def check_reference(frame, context, rtol=1e-6, atol=1e-9):
    merged = frame.merge(context, on=["date", "code"], validate="one_to_one")
    present = merged.turnover_vs_prev.notna()
    if (present & (~merged.prior_turnover.gt(0) | merged.prior_date.isna())).any():
        raise ValueError("original ratio has no positive observed denominator")
    reconstructed = merged.turnover / merged.prior_turnover
    if not np.isclose(merged.loc[present, "turnover_vs_prev"], reconstructed[present],
                      rtol=rtol, atol=atol).all():
        raise ValueError("original ratio disagrees with reconstructed source")
    if "turnover_reference_date" in merged:
        stored = merged.turnover_reference_date.fillna("")
        actual = merged.prior_date.fillna("")
        if not stored.eq(actual).all():
            raise ValueError("stored reference date disagrees with past observation")
    return {"original_ratio_rows": int(present.sum()),
            "max_reconstruction_absolute_error": float((merged.loc[present, "turnover_vs_prev"]
                - reconstructed[present]).abs().max()) if present.any() else None,
            "previous_session_reference_rows": int(merged.prior_is_previous_session.sum()),
            "original_ratio_values_replaced": False}


def binary(frame, score):
    known = frame.label_valid & frame.target.notna()
    selected = frame.loc[known]
    y, prediction = selected.target.astype(int), selected[score].ge(.5).astype(int)
    tp = int(((prediction == 1) & (y == 1)).sum())
    fp = int(((prediction == 1) & (y == 0)).sum())
    fn = int(((prediction == 0) & (y == 1)).sum())
    tn = int(((prediction == 0) & (y == 0)).sum())
    unknown = int((~known).sum())
    return {"registered": len(frame), "scorable": len(selected), "unknown": unknown,
            "coverage": len(selected)/len(frame) if len(frame) else None,
            "accuracy": (tp+tn)/len(selected) if len(selected) else None,
            "precision": tp/(tp+fp) if tp+fp else None,
            "known_label_recall": tp/(tp+fn) if tp+fn else None,
            "complete_registered_positive_recall": tp/(tp+fn) if not unknown and tp+fn else None,
            "complete_recall_status": "UNKNOWN_OUTCOMES_EXIST" if unknown else "ALL_REGISTERED_LABELS_KNOWN",
            "f1_known_labels": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "all_fail_accuracy": float(y.eq(0).mean()) if len(selected) else None,
            "brier": None, "brier_reason": "uncalibrated internal ranking scores"}


def rank(frame, score):
    return frame.sort_values([score, "boards", "code"], ascending=[False, False, True], kind="stable")


def block_interval(daily, slots, difference, confidence=.975):
    values = daily.set_index("date")[difference].reindex(slots).to_numpy(dtype=float)
    rng = np.random.default_rng(20261003)
    starts = rng.integers(0, len(values), size=(5000, int(np.ceil(len(values)/5))))
    indices = ((starts[:, :, None] + np.arange(5)) % len(values)).reshape(5000, -1)[:, :len(values)]
    draws = values[indices]
    count = np.isfinite(draws).sum(axis=1)
    averages = np.nansum(draws, axis=1)[count > 0] / count[count > 0]
    alpha = (1-confidence)/2
    return {"confidence": confidence,
            "bounds": np.quantile(averages, [alpha, 1-alpha]).tolist() if len(averages) else None,
            "official_slots": len(slots), "unknown_slots": int(np.isnan(values).sum()),
            "valid_draw_fraction": float((count > 0).mean()),
            "method": "exploratory circular 5-session block bootstrap; 5000 draws"}


def run(output):
    output = Path(output)
    protocol_path = output/"protocol_BEFORE_SCORING.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for path, expected in protocol["input_hashes"].items():
        if sha(path) != expected:
            raise ValueError("predeclared source changed: " + path)
    source_report = json.loads((BASE/"report.json").read_text(encoding="utf-8"))
    for path, expected in source_report["source_hashes"].items():
        if sha(ROOT/path) != expected:
            raise ValueError("original baseline source changed: " + path)
    calendar_path = ROOT/"data/v279/exchange_calendar_audit_20261002_v1/calendar.json"
    calendar = json.loads(calendar_path.read_text(encoding="utf-8"))
    if calendar["status"] != "verified":
        raise ValueError("verified calendar required")
    sessions = calendar["sessions"]
    successor = dict(zip(sessions[:-1], sessions[1:]))
    market = pd.read_csv(DATA/"market_emotion.csv", usecols=["date"])
    history, fields = prepare_relay(DATA/"leader_candidates.csv", market.date.tolist())
    known = history.label_valid & history.target.notna()
    if not history.loc[known, "target_date"].eq(history.loc[known, "date"].map(successor)).all():
        raise ValueError("historical target differs from official next session")
    train = trailing(known_before(history, "2026-09-01"), 252)
    if len(train) != 3059 or len(fields) != 29:
        raise ValueError("fixed training rows/feature boundary changed")
    metadata = ["date", "code", "actual_one_word_daily", "ohlc_status", "one_word_proxy",
                "turnover_reference_date", "source_available_at"]
    test = pd.read_csv(BASE/"relay_predictions_LABEL_ONLY.csv", dtype={"code": str},
                       usecols=list(dict.fromkeys(metadata + fields)))
    if len(test) != 182 or not test.date.between("2026-09-01", "2026-09-30").all():
        raise ValueError("original evaluation roster boundary changed")
    observations = pd.concat([history[["date", "code", "turnover", "one_word_proxy"]],
                              test[["date", "code", "turnover", "one_word_proxy"]]], ignore_index=True)
    history_context = context_features(history, observations, sessions)
    tr_context = train[["date", "code"]].merge(history_context, on=["date", "code"], validate="one_to_one")
    te_context = context_features(test, observations, sessions)
    audit = {"train": check_reference(train, tr_context), "evaluation": check_reference(test, te_context)}
    tr_context.to_csv(output/"training_denominator_context.csv", index=False, encoding="utf-8-sig")
    te_context.to_csv(output/"evaluation_denominator_context_NO_LABELS.csv", index=False, encoding="utf-8-sig")
    # Recompute ranks before selecting known-label training rows, so unknown
    # candidates do not disappear from the same-height observational universe.
    masked_history_a = sanitize_ratio(history, history_context)
    masked_history_b = sanitize_ratio(history, history_context, True)
    arms = {"baseline": (train, test),
            "A": (masked_history_a.loc[train.index], sanitize_ratio(test, te_context)),
            "B": (masked_history_b.loc[train.index], sanitize_ratio(test, te_context, True))}
    for arm, (tr, te) in arms.items():
        if arm != "baseline":
            for original, changed in [(train, tr), (test, te)]:
                pd.testing.assert_frame_equal(original[["date", "code"] + [f for f in fields if f not in CHANGED]],
                                              changed[["date", "code"] + [f for f in fields if f not in CHANGED]])
            te[["date", "code"] + fields + CONTEXT + ["ratio_retained"]].to_csv(
                output/(arm+"_features_BEFORE_LABEL_JOIN.csv"), index=False, encoding="utf-8-sig")
        fitted = fit(tr, fields, "logistic", "relay")
        test[arm+"_score"] = fitted.predict_proba(te[fields])[:, list(fitted.classes_).index(1)]
        audit[arm] = {"train_rows": len(tr), "evaluation_rows": len(te),
                      "train_ratio_known": int(tr.turnover_vs_prev.notna().sum()),
                      "evaluation_ratio_known": int(te.turnover_vs_prev.notna().sum()),
                      "train_ratio_feature_coverage": float(tr.turnover_vs_prev.notna().mean()),
                      "evaluation_ratio_feature_coverage": float(te.turnover_vs_prev.notna().mean())}
    legacy = pd.read_csv(BASE/"relay_predictions_LABEL_ONLY.csv", dtype={"code": str}, usecols=["date", "code", "probability"])
    check = test.merge(legacy, on=["date", "code"], validate="one_to_one")
    reconstruction_error = float((check.baseline_score-check.probability).abs().max())
    if reconstruction_error > 1e-9:
        raise ValueError("baseline score reconstruction differs")
    test.to_csv(output/"predictions_BEFORE_LABEL_JOIN.csv", index=False, encoding="utf-8-sig")
    labels = pd.read_csv(BASE/"relay_predictions_LABEL_ONLY.csv", dtype={"code": str},
                         usecols=["date", "code", "target_date", "label_valid", "target"])
    valid = labels.label_valid & labels.target.notna()
    if not labels.loc[valid, "target_date"].eq(labels.loc[valid, "date"].map(successor)).all():
        raise ValueError("evaluation target differs from official next session")
    scored = test.merge(labels, on=["date", "code"], how="left", validate="one_to_one")
    daily_rows = []
    for day, group in scored.groupby("date", sort=True):
        complete = bool((group.label_valid & group.target.notna()).all())
        domain = bool(group.actual_one_word_daily.notna().all())
        pool = group[group.actual_one_word_daily.eq(0)]
        row = {"date": day, "registered_candidates": len(group), "nonword_candidates": len(pool),
               "full_labels_known": complete, "domain_known": domain}
        for arm in arms:
            row.update({arm+"_code": None, arm+"_hit": None, arm+"_positive": None, arm+"_score": None})
            if domain and len(pool):
                pick = rank(pool, arm+"_score").iloc[0]
                row[arm+"_code"] = pick.code
                row[arm+"_positive"] = bool(pick[arm+"_score"] >= .5)
                row[arm+"_score"] = float(pick[arm+"_score"])
                if complete:
                    row[arm+"_hit"] = int(pick.target)
        for arm in ["A", "B"]:
            row[arm+"_difference"] = (row[arm+"_hit"]-row["baseline_hit"]
                if row[arm+"_hit"] is not None and row["baseline_hit"] is not None else None)
        daily_rows.append(row)
    daily = pd.DataFrame(daily_rows)
    slots = [s for s in sessions if "2026-09-01" <= s <= "2026-09-30"]
    latest20 = [s for s in sessions if s <= "2026-09-30"][-20:]
    metrics, comparisons = {}, {}
    for arm in arms:
        matched = daily[daily.baseline_hit.notna() & daily[arm+"_hit"].notna()]
        positive = matched[matched[arm+"_positive"].eq(True)]
        metrics[arm] = {"nonword_rank_top1": {"hits": int(matched[arm+"_hit"].sum()), "scorable_dates": len(matched),
            "rate": float(matched[arm+"_hit"].mean()) if len(matched) else None},
            "positive_nonword_top1": {"registered_calls": int(daily[arm+"_positive"].eq(True).sum()),
                "scorable_calls": len(positive), "hits": int(positive[arm+"_hit"].sum()),
                "rate": float(positive[arm+"_hit"].mean()) if len(positive) else None},
            "all_candidate_binary": binary(scored, arm+"_score"),
            "nonword_binary": binary(scored[scored.actual_one_word_daily.eq(0)], arm+"_score"),
            "by_height_binary": {str(b): binary(g, arm+"_score") for b,g in scored.groupby("boards")},
            "latest20_official_sessions_historical": binary(scored[scored.date.isin(latest20)], arm+"_score"),
            "latest60_historical": {"status": "UNAVAILABLE_FIXED_OOS_ONLY_COVERS_SEPTEMBER",
                "reason": "do not combine training/in-sample scores with this fixed evaluation"},
            "true_forward20_60": {"status": "UNAVAILABLE_NO_NEW_SETTLED_FORWARD_SAMPLES"}}
        if arm != "baseline":
            diff = matched[arm+"_hit"]-matched.baseline_hit
            comparisons[arm] = {"top1_difference_pp": float(diff.mean()*100) if len(diff) else None,
                "paired_interval": block_interval(daily, slots, arm+"_difference"),
                "gained_dates": matched.loc[diff.eq(1), "date"].tolist(),
                "lost_dates": matched.loc[diff.eq(-1), "date"].tolist(),
                "changed_selection_dates": daily.loc[daily[arm+"_code"].ne(daily.baseline_code), "date"].tolist(),
                "all_binary_accuracy_difference_pp": 100*(metrics[arm]["all_candidate_binary"]["accuracy"]
                    - metrics["baseline"]["all_candidate_binary"]["accuracy"])}
    scored.to_csv(output/"all_candidates_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output/"paired_daily_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "FIXED_COMPARISON_COMPLETE_NO_UPGRADE", "metrics": metrics,
              "comparisons": comparisons, "denominator_audit": audit,
              "baseline_reconstruction_max_error": reconstruction_error,
              "registered_calendar_slots": len(slots), "dates_with_candidates": int(scored.date.nunique()),
              "full_label_dates": int(daily.full_labels_known.sum()), "original_candidates_retained": len(scored),
              "no_stable60_claim": True, "new_forward_samples": 0, "production_model_changed": False,
              "limitations": ["viewed retrospective history, not blind or forward",
                "archives formed after historical signal dates; nominal 15:00 timestamps are not acquisition proof",
                "proxy definitions differ between historical and September sources",
                "missing true previous first-board denominator is not reconstructed",
                "feature coverage falls; median imputation is not verified market evidence",
                "original historical outcome/price conflicts retained for matched baseline only",
                "no executable trading-return or calibrated-probability claim"],
              "protocol_sha256": sha(protocol_path), "implementation_sha256": sha(__file__)}
    # Protected inputs must remain unchanged, including the unrelated C1/C2.
    for path, expected in protocol["input_hashes"].items():
        if sha(path) != expected:
            raise ValueError("protected input changed during comparison: " + path)
    write_new(output/"report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"status": result["status"], "metrics": result["metrics"],
                      "comparisons": result["comparisons"]}, ensure_ascii=False))
