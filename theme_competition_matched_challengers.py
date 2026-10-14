"""Fixed matched research rankers; no production or order entry points.

Inputs must first be reconstructed from a hash-bound V151 archive and a
hash-bound baseline score/OHLC bundle. Boolean/status columns alone are not
provenance verification. Unknown rows remain registered and are never zero
strength. Synthetic tests do not constitute market/model validation.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CONTROL_FIELDS = ["baseline_logit"]
THEME_FIELDS = CONTROL_FIELDS + [
    "log_theme_peers", "log_theme_same_height_peers", "log_theme_higher_peers",
    "theme_top_height", "theme_ladder_present", "theme_capacity_core",
]
MODEL_PARAMETERS = {"C": 0.25, "solver": "lbfgs", "tol": 0.0001,
    "max_iter": 2500, "fit_intercept": True, "class_weight": None,
    "l1_ratio": 0.0, "random_state": 20261003}
MODEL_IDS = ["B0_CURRENT_FROZEN", "C1_SCORE_ONLY_CONTROL", "C2_SCORE_PLUS_THEME"]
MIN_TRAINING_DATES = 40
MIN_TRAINING_ROWS = 200
MIN_TRAINING_PER_CLASS = 40
MIN_TRAINING_COVERAGE = 0.80


def now():
    return datetime.now(timezone.utc).isoformat()


def stamp(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("exact timezone-aware timestamps required")
    return result.tz_convert("Asia/Shanghai")


def truth(values):
    return values.astype(str).str.lower().isin(["true", "1", "1.0"])


def basic(rows):
    required = {"date", "code", "boards", "baseline_raw_score", "actual_one_word_daily",
        "available_at", "eligible_for_leader_factor_training", "full_market_complete",
        "theme_factor_schema_status", "same_theme_peer_count",
        "same_theme_board_peer_count", "higher_theme_peer_count", "events"}
    if required - set(rows):
        raise ValueError(f"required input fields missing: {sorted(required-set(rows))}")
    if rows.empty or rows.duplicated(["date", "code"]).any():
        raise ValueError("nonempty unique declared relay roster required")
    if not rows.code.map(lambda c: isinstance(c, str) and len(c) == 6 and c.isdigit()).all():
        raise ValueError("six-digit string security codes required")
    boards = pd.to_numeric(rows.boards, errors="coerce")
    if not (boards.ge(2) & boards.mod(1).eq(0)).all():
        raise ValueError("this research ranker takes the entire declared 2+ board roster")
    for row in rows.itertuples():
        cutoff = stamp(str(row.date)+"T16:30:00+08:00")
        available = stamp(row.available_at)
        if not stamp(str(row.date)+"T15:00:00+08:00") <= available <= cutoff:
            raise ValueError("derived candidate inputs were not available by the signal cutoff")
    p = pd.to_numeric(rows.baseline_raw_score, errors="coerce")
    if (p.notna() & ~p.between(0, 1)).any():
        raise ValueError("baseline class-1 score must be in [0,1] or unknown")


def features(rows):
    basic(rows)
    p = pd.to_numeric(rows.baseline_raw_score, errors="coerce")
    result = pd.DataFrame(index=rows.index)
    result["baseline_logit"] = np.log(p.clip(1e-6, 1-1e-6)/(1-p.clip(1e-6, 1-1e-6)))
    admitted = truth(rows.eligible_for_leader_factor_training) & truth(rows.full_market_complete)
    admitted &= rows.theme_factor_schema_status.eq("V150_COMPLETE")
    counts = rows[["same_theme_peer_count", "same_theme_board_peer_count", "higher_theme_peer_count"]].apply(pd.to_numeric, errors="coerce")
    valid_counts = counts.notna().all(axis=1) & counts.ge(0).all(axis=1) & counts.mod(1).eq(0).all(axis=1)
    valid_counts &= (counts.same_theme_board_peer_count+counts.higher_theme_peer_count).le(counts.same_theme_peer_count)
    if (admitted & ~valid_counts).any():
        raise ValueError("admitted theme records have invalid rival counts")
    for raw, output in [("same_theme_peer_count", "log_theme_peers"),
                        ("same_theme_board_peer_count", "log_theme_same_height_peers"),
                        ("higher_theme_peer_count", "log_theme_higher_peers")]:
        result[output] = np.log1p(counts[raw]).where(admitted)
    result["theme_top_height"] = counts.higher_theme_peer_count.eq(0).astype(float).where(admitted)
    result["theme_ladder_present"] = np.nan
    result["theme_capacity_core"] = np.nan
    for i in rows.index[admitted]:
        events = json.loads(rows.at[i, "events"])
        if not isinstance(events, list) or not all(isinstance(event, dict) and isinstance(event.get("type"), str) for event in events):
            raise ValueError("admitted rows require a complete V114 event list")
        result.at[i, "theme_ladder_present"] = float(any(e["type"] == "THEME_LADDER_PRESENT_AT_CLOSE" for e in events))
        result.at[i, "theme_capacity_core"] = float(any(e["type"] == "CAPACITY_LEADER_BY_CLOSE_AMOUNT" and e.get("rank") == 1 for e in events))
    return result


class MatchedThemeRankers:
    def __init__(self):
        self.models = None
        self.training_audit = None

    def fit(self, rows, validation_start):
        rows = rows.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
        if self.models is not None:
            raise ValueError("models are fixed after the single registered training fit")
        required = {"signal_index", "target_date", "label", "label_valid", "label_available_at"}
        if required-set(rows):
            raise ValueError("training phase/label provenance fields missing")
        if not rows.signal_index.between(1, 60).all():
            raise ValueError("validation/test candidates may not enter training")
        clock = stamp(now())
        if clock >= stamp(validation_start+"T09:15:00+08:00"):
            raise ValueError("model must be frozen before validation begins")
        x = features(rows)
        if any(stamp(value)>clock for value in rows.available_at):
            raise ValueError("training features were formed after model fitting")
        purged = rows.target_date.ge(validation_start)
        known = truth(rows.label_valid) & rows.label.isin([0, 1])
        for i in rows.index[known]:
            available = stamp(rows.at[i, "label_available_at"])
            target_close = stamp(rows.at[i, "target_date"]+"T15:00:00+08:00")
            if not target_close <= available <= clock:
                raise ValueError("training outcome was not actually available after target close")
            if rows.at[i, "target_date"] <= rows.at[i, "date"]:
                raise ValueError("target must follow the signal day under the official calendar")
        usable = ~purged & known & x[THEME_FIELDS].notna().all(axis=1)
        denominator = int((~purged).sum())
        counts = rows.loc[usable, "label"].value_counts()
        dates = int(rows.loc[usable, "date"].nunique())
        coverage = int(usable.sum())/denominator if denominator else 0
        if (dates < MIN_TRAINING_DATES or usable.sum() < MIN_TRAINING_ROWS
                or min(counts.get(0, 0), counts.get(1, 0)) < MIN_TRAINING_PER_CLASS
                or coverage < MIN_TRAINING_COVERAGE):
            raise ValueError("fixed warmup window has insufficient verified training evidence; do not roll it forward")
        # Missing members do not increase the contribution of remaining members.
        weights = 1/rows.groupby("date").date.transform("size")
        models = []
        for fields in [CONTROL_FIELDS, THEME_FIELDS]:
            model = make_pipeline(StandardScaler(), LogisticRegression(**MODEL_PARAMETERS))
            model.fit(x.loc[usable, fields], rows.loc[usable, "label"].astype(int),
                      logisticregression__sample_weight=weights.loc[usable])
            models.append(model)
        completed = stamp(now())
        if completed >= stamp(validation_start+"T09:15:00+08:00"):
            raise ValueError("complete model formation crossed the registered training boundary")
        self.models = models
        self.training_audit = {
            "recorded_at": completed.isoformat(), "declared_rows": len(rows),
            "boundary_purged_rows": int(purged.sum()), "eligible_rows": int(usable.sum()),
            "training_dates": dates, "training_coverage": coverage,
            "max_label_date": rows.loc[usable, "target_date"].max(),
            "validation_start": validation_start, "same_training_indices_for_both_models": True,
            "excluded_candidate_ids": (rows.loc[~usable, "date"]+":"+rows.loc[~usable, "code"]).tolist(),
            "score_semantics": "UNCALIBRATED_RANK_SCORE", "probability_calibrated": False,
        }
        return self

    def predict(self, rows):
        rows = rows.reset_index(drop=True).copy()
        if any(name in rows and rows[name].notna().any() for name in ["label", "promoted_next_day", "target"]):
            raise ValueError("future outcomes must not be supplied to the forecasting function")
        clock = stamp(now())
        x = features(rows)
        result = rows.copy()
        result["baseline_raw_score"] = pd.to_numeric(result.baseline_raw_score, errors="coerce")
        result["boards"] = pd.to_numeric(result.boards, errors="raise")
        for model_id in MODEL_IDS:
            result[model_id+"_selected"] = False
        result["research_route"] = "NO_SELECTION"
        for date, group in result.groupby("date", sort=True):
            cutoff = stamp(date+"T16:30:00+08:00")
            if not stamp(date+"T15:00:00+08:00") <= clock <= cutoff:
                raise ValueError("prospective shadow forecasts require the actual signal-day clock before cutoff")
            if any(stamp(value)>clock for value in group.available_at):
                raise ValueError("input record was formed after forecast generation")
            if not group.actual_one_word_daily.isin([0, 1]).all():
                result.loc[group.index, "research_route"] = "DOMAIN_UNVERIFIED_NO_CALL"
                continue
            eligible = group[group.actual_one_word_daily.eq(0)]
            if eligible.empty:
                result.loc[group.index, "research_route"] = "NO_NONWORD_CANDIDATE"
                continue
            if eligible.baseline_raw_score.isna().any():
                result.loc[group.index, "research_route"] = "BASELINE_SCORE_UNAVAILABLE_NO_CALL"
                continue
            original = eligible.sort_values(["baseline_raw_score", "boards", "code"], ascending=[False, False, True], kind="stable").index[0]
            result.loc[original, MODEL_IDS[0]+"_selected"] = True
            active = self.models is not None and x.loc[eligible.index, THEME_FIELDS].notna().all().all()
            if self.models is not None and stamp(self.training_audit["recorded_at"]) >= stamp(date+"T09:15:00+08:00"):
                raise ValueError("models were not formed before the prediction session")
            if active:
                for model, fields, model_id in zip(self.models, [CONTROL_FIELDS, THEME_FIELDS], MODEL_IDS[1:]):
                    ranked = eligible.copy()
                    ranked["research_score"] = model.decision_function(x.loc[eligible.index, fields])
                    index = ranked.sort_values(["research_score", "boards", "code"], ascending=[False, False, True], kind="stable").index[0]
                    result.loc[index, model_id+"_selected"] = True
                route = "MATCHED_RESEARCH_RANKERS"
            else:
                for model_id in MODEL_IDS[1:]:
                    result.loc[original, model_id+"_selected"] = True
                route = "WARMUP_BASELINE_ONLY" if self.models is None else "DAY_FALLBACK_BASELINE_THEME_UNAVAILABLE"
            result.loc[group.index, "research_route"] = route
        if any(stamp(now())>stamp(date+"T16:30:00+08:00") for date in result.date.unique()):
            raise ValueError("complete shadow prediction crossed the fixed cutoff")
        result["score_semantics"] = "RESEARCH_SELECTION_NOT_PROMOTION_CALL_OR_CALIBRATED_PROBABILITY"
        result["production_model_modified"] = False
        return result


def circular_block_interval(values, confidence=0.95):
    """Resample registered official-session slots; keep missing slots in place."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(values).any():
        return None
    if np.isinf(values).any():
        raise ValueError("infinite observations forbidden")
    rng = np.random.default_rng(20261003)
    length = min(5, len(values))
    starts = rng.integers(0, len(values), size=(10000, int(np.ceil(len(values)/length))))
    indices = ((starts[:, :, None]+np.arange(length)) % len(values)).reshape(10000, -1)[:, :len(values)]
    draws = values[indices]
    valid = np.isfinite(draws).sum(axis=1)
    means = np.nansum(draws, axis=1)[valid>0]/valid[valid>0]
    alpha = 1-confidence
    return np.quantile(means, [alpha/2, 1-alpha/2]).tolist()
