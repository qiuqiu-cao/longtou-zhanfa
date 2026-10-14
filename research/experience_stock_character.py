"""One fixed, retrospective increment test of recorded relay history.

The forum/repository experience is a hypothesis, not a production rule.
Only the original >=2-board archive is observed; this is not full-market
stock character, an independent blind test, or executable trading evidence.
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
from next_session_forecast_lab_v271 import DATA, prepare_relay, model
from rolling_forecast_challenger_v274 import known_before, trailing, fit

BASE = ROOT / "results/v278/september_falsification_20261002_v6"
CALENDAR = ROOT / "data/v279/exchange_calendar_audit_20261002_v1/calendar.json"
REGISTRY = ROOT / "data/v271/expectation_registry/2026-09-30_2c74dc406eef7dcf.json"
CUTOFF = "2026-09-01"
EXTRA = ["prior_relay_wilson_lower", "log_prior_relay_known_n"]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)


def wilson(k, n):
    if n == 0:
        return None
    z = 1.959963984540054
    p = k / n
    d = 1 + z*z/n
    mid = (p + z*z/(2*n))/d
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [float(mid-half), float(mid+half)]


def historical_features(rows, history, sessions, frozen_at=None):
    """Past recorded events only; unknown outcomes never become failures.

    Frozen evaluation uses only outcomes realized strictly before Sep 1.
    Each training row also excludes outcomes realized on its signal day.
    The 250-session window is measured using the official calendar.
    """
    indices = {d: i for i, d in enumerate(sessions)}
    records = []
    for row in rows[["date", "code"]].itertuples(index=False):
        i = indices[row.date]
        lower = sessions[max(0, i-250)]
        stop = min(row.date, frozen_at) if frozen_at else row.date
        prior = history[(history.code == row.code) & history.date.ge(lower)
                        & history.date.lt(row.date) & history.target_date.lt(stop)]
        good = prior.label_valid & prior.target.notna()
        observed = prior.loc[good, "target"]
        if not observed.isin([0, 1]).all():
            raise ValueError("history target outside binary outcome domain")
        n, k = len(observed), int(observed.sum())
        interval = wilson(k, n)
        records.append({"date": row.date, "code": row.code,
                        EXTRA[0]: interval[0] if interval else np.nan,
                        EXTRA[1]: float(np.log1p(n)),
                        "prior_known_n": n, "prior_successes": k,
                        "prior_unknown_n": int((~good).sum()),
                        "prior_recorded_n": len(prior),
                        "prior_max_target_date": prior.loc[good, "target_date"].max() if n else None,
                        "prior_window_start": lower,
                        "prior_outcome_exclusive_cutoff": stop})
    return pd.DataFrame(records)


def rank(frame, score):
    return frame.sort_values([score, "boards", "code"],
                             ascending=[False, False, True], kind="stable")


def paired_interval(daily, sessions):
    values = daily.set_index("date")["difference"].reindex(sessions).to_numpy(dtype=float)
    rng = np.random.default_rng(20261003)
    starts = rng.integers(0, len(values), size=(5000, int(np.ceil(len(values)/5))))
    indices = ((starts[:, :, None]+np.arange(5)) % len(values)).reshape(5000, -1)[:, :len(values)]
    draws = values[indices]
    count = np.isfinite(draws).sum(axis=1)
    means = np.nansum(draws, axis=1)[count > 0] / count[count > 0]
    return {"interval95": np.quantile(means, [.025, .975]).tolist() if len(means) else None,
            "valid_draw_fraction": float((count > 0).mean()),
            "official_calendar_slots": len(values), "unknown_slots": int(np.isnan(values).sum()),
            "method": "exploratory 5-session circular block percentile bootstrap, 5000 draws"}


def binary(rows, score):
    known = rows.label_valid & rows.target.notna()
    g = rows.loc[known]
    y = g.target.astype(int)
    p = g[score].ge(.5).astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    return {"registered_rows": len(rows), "known_rows": len(g),
            "unknown_rows": int((~known).sum()), "coverage": len(g)/len(rows) if len(rows) else None,
            "accuracy": float((p == y).mean()) if len(g) else None,
            "precision": precision, "complete_positive_recall": recall,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "all_fail_accuracy": float((y == 0).mean()) if len(g) else None,
            "brier": None, "brier_reason": "internal class scores are not accepted calibrated probabilities"}


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    original_registry = sha(REGISTRY)
    source_report = json.loads((BASE/"report.json").read_text(encoding="utf-8"))
    for name, wanted in source_report["source_hashes"].items():
        if sha(ROOT/name) != wanted:
            raise ValueError("original source hash changed: "+name)
    calendar = json.loads(CALENDAR.read_text(encoding="utf-8"))
    sessions = calendar["sessions"]
    successor = dict(zip(sessions[:-1], sessions[1:]))
    if calendar["status"] != "verified":
        raise ValueError("verified calendar required")
    market = pd.read_csv(DATA/"market_emotion.csv")
    history, fields = prepare_relay(DATA/"leader_candidates.csv", market.date.tolist())
    good = history.label_valid & history.target.notna()
    if not history.loc[good, "target_date"].eq(history.loc[good, "date"].map(successor)).all():
        raise ValueError("historical known result skips exact official next session")
    metadata = ["date", "code", "actual_one_word_daily", "ohlc_status"]
    test = pd.read_csv(BASE/"relay_predictions_LABEL_ONLY.csv", dtype={"code": str},
                       usecols=list(dict.fromkeys(metadata+fields)))
    if test.duplicated(["date", "code"]).any() or history.duplicated(["date", "code"]).any():
        raise ValueError("duplicate candidate key")
    if not test.date.between("2026-09-01", "2026-09-30").all():
        raise ValueError("evaluation boundary changed")
    train = trailing(known_before(history, CUTOFF), 252)
    source_files = [DATA/"leader_candidates.csv", DATA/"market_emotion.csv", BASE/"report.json",
                    BASE/"relay_predictions_LABEL_ONLY.csv", CALENDAR, REGISTRY,
                    Path(__file__), ROOT/"next_session_forecast_lab_v271.py",
                    ROOT/"rolling_forecast_challenger_v274.py"]
    protocol = {"registered_at": stamp(), "hypothesis": "Prior recorded same-code relay advancement consistency adds information beyond the frozen 29 closing features",
                "source": "https://github.com/guoyaohua/limit-up-sniper", "source_scope": "first-board historical consistency motivates a narrower >=2-board recorded-event proxy",
                "study_kind": "VIEWED_HISTORY_RETROSPECTIVE_INCREMENT_NOT_BLIND_OR_FORWARD",
                "baseline": "Exactly V278 September frozen logistic C=.25, last 252 known-label dates before Sep 1; no new model selection",
                "challenger": "Same training rows/date weights/imputer/scaler/logistic settings, add only Wilson95 lower statistic and log1p known prior event count",
                "extra_fields": EXTRA, "history_window": "250 official signal sessions; target_date strictly earlier than row date and frozen Sep-1 cutoff for evaluation",
                "zero_history": "Wilson statistic remains NaN, count=0; original train-only median and missing indicator; not zero failures",
                "unknown_history": "Count separately, exclude from success/known denominators; historical unseen/missing full-market events remain unobservable",
                "primary": "true OHLC nonword Top1 exact next-session closing advancement, same full-label dates",
                "expected_minimum_improvement_pp": 5, "minimum_support": "60 complete matched dates and paired interval lower>0; current September cannot pass support gate",
                "ranking_ties": "score descending, boards descending, code ascending; rank is separate from predicted promotion flag",
                "binary_threshold": .5, "limits": "All original candidates retained; no same-day/September history outcomes in added features; no parameter scan, production changes or saved production weights",
                "ending": "one fixed comparison and source checks complete; do not change rules after results",
                "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in source_files}}
    write(output/"protocol_BEFORE_SCORING.json", protocol)
    tr_extra = historical_features(train, history, sessions)
    te_extra = historical_features(test, history, sessions, frozen_at=CUTOFF)
    train = train.merge(tr_extra, on=["date", "code"], validate="one_to_one")
    test = test.merge(te_extra, on=["date", "code"], validate="one_to_one")
    te_extra.to_csv(output/"historical_features_NO_EVALUATION_LABELS.csv", index=False, encoding="utf-8-sig")
    baseline = fit(train, fields, "logistic", "relay")
    challenger = fit(train, fields+EXTRA, "logistic", "relay")
    for name, fitted, columns in [("baseline_score", baseline, fields), ("challenger_score", challenger, fields+EXTRA)]:
        test[name] = fitted.predict_proba(test[columns])[:, list(fitted.classes_).index(1)]
    legacy = pd.read_csv(BASE/"relay_predictions_LABEL_ONLY.csv", dtype={"code": str}, usecols=["date", "code", "probability"])
    check = test.merge(legacy, on=["date", "code"], validate="one_to_one")
    error = float(np.abs(check.baseline_score-check.probability).max())
    if error > 1e-9:
        raise ValueError(f"baseline reconstruction differs: {error}")
    test.to_csv(output/"predictions_BEFORE_EVALUATION_LABEL_JOIN.csv", index=False, encoding="utf-8-sig")
    labels = pd.read_csv(BASE/"relay_predictions_LABEL_ONLY.csv", dtype={"code": str}, usecols=["date", "code", "target_date", "label_valid", "target"])
    if not labels.loc[labels.label_valid, "target_date"].eq(labels.loc[labels.label_valid, "date"].map(successor)).all():
        raise ValueError("evaluation labels skip official next session")
    scored = test.merge(labels, on=["date", "code"], validate="one_to_one")
    complete = scored.groupby("date").apply(lambda g: bool((g.label_valid & g.target.notna()).all()), include_groups=False)
    daily_records = []
    for date, all_rows in scored.groupby("date", sort=True):
        pool = all_rows[all_rows.actual_one_word_daily.eq(0)]
        row = {"date": date, "registered_candidates": len(all_rows), "nonword_candidates": len(pool),
               "full_labels_known": bool(complete[date]), "domain_known": bool(all_rows.actual_one_word_daily.notna().all()),
               "baseline_code": None, "challenger_code": None, "baseline_hit": None, "challenger_hit": None, "difference": None}
        if len(pool) and row["domain_known"]:
            for arm in ["baseline", "challenger"]:
                pick = rank(pool, arm+"_score").iloc[0]
                row[arm+"_code"] = pick.code
                row[arm+"_positive_call"] = bool(pick[arm+"_score"] >= .5)
                row[arm+"_internal_score"] = float(pick[arm+"_score"])
                if row["full_labels_known"]:
                    row[arm+"_hit"] = int(pick.target)
            if row["full_labels_known"]:
                row["difference"] = row["challenger_hit"]-row["baseline_hit"]
        daily_records.append(row)
    daily = pd.DataFrame(daily_records)
    scored.to_csv(output/"all_candidates_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output/"paired_daily_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
    matched = daily[daily.difference.notna()]
    metrics = {}
    for arm in ["baseline", "challenger"]:
        hits = int(matched[arm+"_hit"].sum())
        metrics[arm] = {"nonword_top1": {"hits": hits, "known_dates": len(matched), "rate": hits/len(matched) if len(matched) else None,
                                         "wilson95_descriptive": wilson(hits, len(matched))},
                        "all_candidate_binary": binary(scored, arm+"_score"),
                        "nonword_binary": binary(scored[scored.actual_one_word_daily.eq(0)], arm+"_score"),
                        "by_height_binary": {str(b): binary(g, arm+"_score") for b, g in scored.groupby("boards")}}
    official_test = [d for d in sessions if "2026-09-01" <= d <= "2026-09-30"]
    interval = paired_interval(daily, official_test)
    raw_delta = float(matched.difference.mean()) if len(matched) else None
    disposition = "INSUFFICIENT_EVIDENCE_NO_UPGRADE"
    if raw_delta is not None and raw_delta <= 0:
        disposition = "NO_POSITIVE_INCREMENT_NO_UPGRADE"
    if sha(REGISTRY) != original_registry:
        raise ValueError("original production registry changed")
    report = {"created_at": stamp(), "status": disposition, "metrics": metrics,
              "primary_difference_pp": 100*raw_delta if raw_delta is not None else None,
              "paired_uncertainty": interval, "gained_dates": matched.loc[matched.difference.eq(1), "date"].tolist(),
              "lost_dates": matched.loc[matched.difference.eq(-1), "date"].tolist(),
              "changed_selection_dates": daily.loc[daily.baseline_code.ne(daily.challenger_code), "date"].tolist(),
              "candidate_rows": len(scored), "candidate_dates": int(scored.date.nunique()), "official_test_dates": len(official_test),
              "missing_entire_dates": sorted(set(official_test)-set(scored.date)), "unsettled_candidate_dates": daily.loc[~daily.full_labels_known, "date"].tolist(),
              "known_history_feature_rows": int(te_extra.prior_known_n.gt(0).sum()), "zero_history_rows": int(te_extra.prior_known_n.eq(0).sum()),
              "history_observation_scope": "Only archived >=2-board events; repeated code and same run observations are correlated; not full-market history or intrinsic stock quality",
              "train_rows": len(train), "train_dates": int(train.date.nunique()), "train_max_target_date": train.target_date.max(),
              "baseline_max_absolute_score_error": error, "new_forward_samples": 0, "core_model_upgraded": False,
              "stable_60_verified": False, "original_registry_sha256": original_registry,
              "limitations": ["Previously viewed September labels; post-hoc experience hypothesis, no blind-test claim",
                              "Sources obtained after signal dates; feature reconstruction is not actual historical knowledge-time proof",
                              "Provider candidate scope/security status/historical closing-limit truth not fully exchange certified",
                              "No verified historical theme/competition, auctions or actual trades; no profit claim",
                              "Single short evaluation window cannot validate stable 60%; no retuning after this result"],
              "source_and_artifact_hashes": {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    write(output/"report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        r = run(args.output)
        print(json.dumps({k: r[k] for k in ["status", "metrics", "primary_difference_pp", "paired_uncertainty", "candidate_rows", "new_forward_samples"]}, ensure_ascii=False, allow_nan=False))
    except Exception as exc:
        p = Path(args.output)/"failure.json"
        if p.parent.exists() and not p.exists():
            write(p, {"failed_at": stamp(), "error": str(exc)})
        raise
