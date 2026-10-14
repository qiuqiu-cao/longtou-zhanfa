"""Early leader Top1/Top3/Top5 walk-forward diagnostic.

The experiment uses only signal-date EOD fields to rank first-, second-, and
third-board candidates. Labels are constructed from later rows of the same
known episode over the next five observed sessions. They are retrospective
daily-bar labels, not live probabilities or executable fills. The source
manifest explicitly says the universe/features were reconstructed after the
fact and are not training-authorized; this script therefore remains
DIAGNOSTIC_ONLY.

The fixed rolling protocol is TRAIN -> VALIDATION -> BLIND. The blind split is
temporally held out and never used to fit that fold, but it is not a pristine
blind test because the underlying historical reconstruction was available for
research after the fact.

Three requested roles are kept separate:
 A: next-board promotion from the current stage;
 B: topic leader (UNAVAILABLE because verified theme membership is absent);
 C: early market-space leader among current 1-3 board candidates, defined as
    reaching the highest future board within the five-session label horizon
    and reaching at least five boards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


VERSION = "v263-early-leader-topk-forward-proxy-1"
HORIZON_SESSIONS = 5

FOLDS = (
    {"name": "2024H2", "validation_start": "2024-04-01",
     "validation_end": "2024-06-28", "blind_start": "2024-07-01",
     "blind_end": "2024-12-31"},
    {"name": "2025", "validation_start": "2024-10-09",
     "validation_end": "2024-12-31", "blind_start": "2025-01-02",
     "blind_end": "2025-12-31"},
    {"name": "2026YTD", "validation_start": "2025-10-09",
     "validation_end": "2025-12-31", "blind_start": "2026-01-05",
     "blind_end": "2026-08-24"},
)

IDENTITY = [
    "boards", "height_gap_to_known_top", "same_height_known_count",
    "sole_known_top_proxy", "candidate_pool_count", "unknown_height_pool_count",
    "known_pool_max_boards",
]
EMOTION = [
    "market_limit_count", "market_consecutive_count", "market_failed_count",
    "market_down_limit_count", "market_failed_rate", "market_seal_rate",
    "market_height_change", "market_limit_change", "market_middle_density",
    "market_ladder_gap_count", "market_breadth_height_divergence",
    "market_yesterday_premium", "market_previous_limit_tail_loss",
    "market_failed_tail_loss", "market_promotion_overall", "market_promotion_1to2",
    "market_promotion_2to3", "market_promotion_3to4", "market_promotion_4to5",
    "market_promotion_mid", "market_space_feedback", "market_space_feedback_3d",
    "market_breadth_health", "market_relay_appetite", "market_ladder_health",
    "market_loss_effect_health", "market_crowding_risk", "market_regime_fragility",
    "market_false_acceleration", "market_emotion_quality_score", "market_emotion_score",
    "market_state_start", "market_state_warming", "market_state_acceleration",
    "market_state_consensus", "market_state_divergence", "market_state_retreat",
    "market_state_repair",
]
PRICE_VOLUME = ["open_gap_pct", "amplitude_pct", "amount_to_float_cap", "turnover"]
FEATURES = list(dict.fromkeys(IDENTITY + EMOTION + PRICE_VOLUME))
FORBIDDEN_FEATURES = {
    "promotion_LABEL_ONLY", "promotion_state", "entry_date_LABEL_ONLY", "exit_date_LABEL_ONLY",
    "b1_date_LABEL_ONLY", "b2_date_LABEL_ONLY", "b3_date_LABEL_ONLY",
    "future_max_board", "future_gain", "future_promotion", "future_stage_leader",
    "future_market_leader", "b1_net_close_mark_LABEL_ONLY", "b2_net_close_mark_LABEL_ONLY",
    "b3_net_close_mark_LABEL_ONLY",
}

STAGES = {
    "A_first_board": {"boards": 1, "score_target": "future_promotion",
                      "leader_target": "future_stage_leader"},
    "B_second_board": {"boards": 2, "score_target": "future_stage_leader",
                       "leader_target": "future_stage_leader"},
    "C_third_board": {"boards": 3, "score_target": "future_market_leader",
                      "leader_target": "future_market_leader"},
}


def _bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, (pd.Series, pd.DataFrame, np.ndarray)):
        raise TypeError("report contains a non-scalar pandas/numpy object")
    if not isinstance(value, (str, bytes, bool)):
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            return value
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    return value


def _wilson(successes: int, total: int) -> list[float | None]:
    if not total:
        return [None, None]
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [float(max(0, center - half)), float(min(1, center + half))]


def _stable_rank(date: pd.Series, code: pd.Series) -> pd.Series:
    return (date.astype(str) + "|" + code.astype(str)).map(
        lambda value: int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)
    )


def _attach_forward_bar_proxies(rows: pd.DataFrame, bars: pd.DataFrame | None) -> dict:
    """Attach non-executable close-to-close path proxies from daily bars.

    These fields are deliberately not used by the model. They are calculated
    only after the signal-date features have been formed, and are reported as
    reconstructed OHLC diagnostics rather than fills.
    """
    fields = ["forward_return_1d_proxy", "forward_return_3d_proxy",
              "forward_return_5d_proxy", "forward_max_gain_5d_proxy",
              "forward_max_drawdown_5d_proxy"]
    for field in fields:
        rows[field] = np.nan
    if bars is None:
        return {"status": "UNAVAILABLE_BARS_NOT_SUPPLIED", "known_rows_5d": 0}
    required = {"code", "date", "close", "high", "low"}
    missing = required - set(bars.columns)
    if missing:
        return {"status": "UNAVAILABLE_BARS_MISSING_FIELDS", "missing": sorted(missing),
                "known_rows_5d": 0}
    daily = bars.copy()
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for field in ("close", "high", "low"):
        daily[field] = pd.to_numeric(daily[field], errors="coerce")
    verification_status = "NOT_PROVIDED"
    if "status_verified" in daily:
        verified = _bool(daily["status_verified"])
        # Some reconstructed archives encode every row as status_verified=False.
        # Do not silently erase the sample; retain it and surface the quality
        # downgrade in the audit instead.
        if verified.eq(True).any():
            daily = daily[verified.eq(True)]
            verification_status = "MIXED_FILTERED_TO_TRUE"
        else:
            verification_status = "ALL_FALSE_UNVERIFIED_RECONSTRUCTION"
    if "st" in daily:
        daily = daily[_bool(daily["st"]).eq(False)]
    if "new_listing" in daily:
        daily = daily[_bool(daily["new_listing"]).eq(False)]
    daily = daily.dropna(subset=["code", "date", "close", "high", "low"])
    daily = daily.drop_duplicates(["code", "date"], keep="last")
    calendar = sorted(daily["date"].unique().tolist())
    date_pos = {date: index for index, date in enumerate(calendar)}
    lookup = daily.set_index(["code", "date"])[["close", "high", "low"]].to_dict("index")
    known_5d = 0
    for index, row in rows.iterrows():
        code = row["code"]
        date = row["signal_date"]
        pos = date_pos.get(date)
        base = lookup.get((code, date)) if pos is not None else None
        if base is None or not np.isfinite(float(base["close"])):
            continue
        future = []
        for horizon in range(1, 6):
            if pos + horizon >= len(calendar):
                break
            bar = lookup.get((code, calendar[pos + horizon]))
            if bar is None:
                break
            future.append(bar)
        if len(future) >= 1:
            rows.at[index, "forward_return_1d_proxy"] = future[0]["close"] / base["close"] - 1
        if len(future) >= 3:
            rows.at[index, "forward_return_3d_proxy"] = future[2]["close"] / base["close"] - 1
        if len(future) >= 5:
            rows.at[index, "forward_return_5d_proxy"] = future[4]["close"] / base["close"] - 1
            rows.at[index, "forward_max_gain_5d_proxy"] = max(
                bar["high"] for bar in future[:5]) / base["close"] - 1
            rows.at[index, "forward_max_drawdown_5d_proxy"] = min(
                bar["low"] for bar in future[:5]) / base["close"] - 1
            known_5d += 1
    return {
        "status": "INFERRED_CLOSE_TO_CLOSE_OHLC_PROXY",
        "bars_rows_after_quality_filters": int(len(daily)),
        "bars_calendar_dates": int(len(calendar)),
        "bars_calendar_start": calendar[0] if calendar else None,
        "bars_calendar_end": calendar[-1] if calendar else None,
        "status_verified_filter": verification_status,
        "known_rows_5d": int(known_5d),
        "actual_fill_count": 0,
    }


def _prepare(features: pd.DataFrame, outcomes: pd.DataFrame | None = None,
             bars: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    rows = features.copy()
    rows["code"] = rows["code"].astype(str).str.zfill(6)
    rows["signal_date"] = pd.to_datetime(rows["signal_date"], errors="raise").dt.strftime("%Y-%m-%d")
    if rows.duplicated(["code", "signal_date"]).any():
        raise ValueError("duplicate code/signal_date feature key")
    if set(FEATURES) & FORBIDDEN_FEATURES:
        raise AssertionError("future field included in feature list")
    needed = {"code", "signal_date", "episode_id", "boards", "start_known", "feature_status",
              "one_price_day_proxy", *FEATURES}
    missing = needed - set(rows)
    if missing:
        raise ValueError(f"missing feature fields: {sorted(missing)}")
    rows["boards"] = pd.to_numeric(rows["boards"], errors="coerce")
    rows["one_price_day_proxy"] = pd.to_numeric(rows["one_price_day_proxy"], errors="coerce")
    for field in FEATURES:
        rows[field] = pd.to_numeric(rows[field], errors="coerce").replace([np.inf, -np.inf], np.nan)
    known = (
        rows.feature_status.eq("historical_reconstruction_not_forward_archive")
        & _bool(rows.start_known).eq(True)
        & rows.boards.notna()
        & rows.boards.mod(1).eq(0)
        & rows.episode_id.notna()
    )
    rows = rows.loc[known].copy()
    rows["board_stage"] = rows.boards.astype(int)
    calendar = sorted(rows.signal_date.unique().tolist())
    date_pos = {date: index for index, date in enumerate(calendar)}
    rows["signal_pos"] = rows.signal_date.map(date_pos).astype(int)
    rows["future_label_status"] = "COMPLETE_HORIZON"
    rows.loc[rows.signal_pos + HORIZON_SESSIONS >= len(calendar), "future_label_status"] = "RIGHT_CENSORED"
    for col in ["future_max_board", "future_gain", "future_promotion",
                "future_reaches_3", "future_reaches_4", "future_reaches_5",
                "future_stage_leader", "future_market_leader"]:
        rows[col] = np.nan

    episode_groups = {key: group for key, group in rows.groupby("episode_id", sort=False)}
    for index, row in rows.iterrows():
        if row.future_label_status != "COMPLETE_HORIZON":
            continue
        end_pos = row.signal_pos + HORIZON_SESSIONS
        future = episode_groups[row.episode_id]
        future = future[(future.signal_pos > row.signal_pos) & (future.signal_pos <= end_pos)]
        max_board = max([int(row.boards)] + future.boards.dropna().astype(int).tolist())
        rows.at[index, "future_max_board"] = max_board
        rows.at[index, "future_gain"] = max_board - int(row.boards)
        rows.at[index, "future_promotion"] = float(max_board > int(row.boards))
        for threshold in (3, 4, 5):
            rows.at[index, f"future_reaches_{threshold}"] = float(max_board >= threshold)

    complete = rows.future_label_status.eq("COMPLETE_HORIZON")
    for date, group in rows[complete].groupby("signal_date", sort=False):
        for stage, stage_rows in group.groupby("board_stage"):
            if stage not in (1, 2, 3):
                continue
            best_gain = stage_rows.future_gain.max()
            if pd.notna(best_gain) and best_gain >= 2:
                indices = stage_rows.index[stage_rows.future_gain.eq(best_gain)]
                rows.loc[indices, "future_stage_leader"] = 1.0
                rows.loc[stage_rows.index.difference(indices), "future_stage_leader"] = 0.0
            else:
                rows.loc[stage_rows.index, "future_stage_leader"] = 0.0
        early = group[group.board_stage.le(3)]
        max_future = early.future_max_board.max()
        if pd.notna(max_future) and max_future >= 5:
            rows.loc[early.index, "future_market_leader"] = early.future_max_board.eq(max_future).astype(float)
        else:
            rows.loc[early.index, "future_market_leader"] = 0.0

    if outcomes is not None:
        outcome = outcomes.copy()
        outcome["code"] = outcome["code"].astype(str).str.zfill(6)
        outcome["signal_date"] = pd.to_datetime(outcome["signal_date"], errors="raise").dt.strftime("%Y-%m-%d")
        if outcome.duplicated(["code", "signal_date"]).any():
            raise ValueError("duplicate outcome key")
        outcome_fields = ["code", "signal_date", "b1_net_close_mark_LABEL_ONLY",
                          "b2_net_close_mark_LABEL_ONLY", "b3_net_close_mark_LABEL_ONLY"]
        available = [field for field in outcome_fields if field in outcome]
        rows = rows.merge(outcome[available], on=["code", "signal_date"], how="left", validate="one_to_one")
        for field in available[2:]:
            rows[field] = pd.to_numeric(rows[field], errors="coerce")

    bar_audit = _attach_forward_bar_proxies(rows, bars)
    rows["environment"] = rows.apply(_environment, axis=1)
    audit = {
        "source_rows": int(len(features)),
        "known_rows": int(len(rows)),
        "calendar_dates": int(len(calendar)),
        "calendar_start": calendar[0] if calendar else None,
        "calendar_end": calendar[-1] if calendar else None,
        "future_horizon_sessions": HORIZON_SESSIONS,
        "future_complete_rows": int(rows.future_label_status.eq("COMPLETE_HORIZON").sum()),
        "right_censored_rows": int(rows.future_label_status.eq("RIGHT_CENSORED").sum()),
        "theme_status": "UNAVAILABLE_IN_FEATURE_SCHEMA",
        "source_status": "HISTORICAL_RECONSTRUCTION_NOT_FORWARD_ARCHIVE",
        "forward_bar_proxy": bar_audit,
    }
    return rows, audit


def _environment(row: pd.Series) -> str:
    for name in ("divergence", "retreat", "consensus", "acceleration", "warming", "repair", "start"):
        if row.get("market_state_" + name) == 1:
            return name.upper()
    value = row.get("market_emotion_state")
    return "UNKNOWN" if pd.isna(value) else str(value)


def _fit(train: pd.DataFrame, test: pd.DataFrame, target: str) -> tuple[np.ndarray, str, float | None]:
    eligible = train[target].isin([0, 1])
    train = train.loc[eligible].copy()
    if train.empty:
        return np.full(len(test), np.nan), "NO_TRAIN_LABELS", None
    prior = float(train[target].mean())
    if train[target].nunique() < 2:
        return np.full(len(test), prior), "PRIOR_ONLY_ONE_CLASS", prior
    weights = 1 / train.groupby("signal_date").signal_date.transform("size")
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(), LogisticRegression(C=1, max_iter=3000, random_state=261),
    )
    model.fit(train[FEATURES], train[target], logisticregression__sample_weight=weights)
    return model.predict_proba(test[FEATURES])[:, 1], "LOGISTIC_WALK_FORWARD", prior


def _rank_scores(frame: pd.DataFrame, score: str, baseline: str | None = None) -> pd.DataFrame:
    scored = frame.copy()
    if baseline == "identity_rule":
        scored[score] = (
            -scored.height_gap_to_known_top.fillna(999)
            - 0.01 * scored.same_height_known_count.fillna(999)
            + 0.05 * scored.sole_known_top_proxy.fillna(0)
        )
    elif baseline == "random":
        scored[score] = -_stable_rank(scored.signal_date, scored.code).astype(float)
    return scored


def _select(frame: pd.DataFrame, score: str, k: int) -> pd.DataFrame:
    selected = frame.sort_values(["signal_date", score, "boards", "code"],
                                 ascending=[True, False, False, True], kind="mergesort")
    selected = selected.groupby("signal_date", sort=False, group_keys=False).head(k).copy()
    selected["selection_rank"] = selected.groupby("signal_date").cumcount() + 1
    return selected


def _proxy_metrics(selected: pd.DataFrame) -> dict:
    field = next((field for field in ("b1_net_close_mark_LABEL_ONLY",
                                      "b2_net_close_mark_LABEL_ONLY",
                                      "b3_net_close_mark_LABEL_ONLY") if field in selected), None)
    if field is None:
        return {"status": "UNAVAILABLE_NO_PROXY_FIELD"}
    values = pd.to_numeric(selected[field], errors="coerce").dropna()
    if values.empty:
        return {"status": "UNAVAILABLE_NO_KNOWN_PROXY_ROWS", "field": field}
    ordered = selected[["signal_date", field]].copy()
    ordered[field] = pd.to_numeric(ordered[field], errors="coerce")
    daily = ordered.dropna().groupby("signal_date")[field].mean().sort_index()
    cumulative = daily.cumsum()
    drawdown = cumulative - cumulative.cummax()
    wins = values[values > 0]
    losses = values[values < 0]
    signs = values.lt(0).astype(int).tolist()
    longest = current = 0
    for sign in signs:
        current = current + 1 if sign else 0
        longest = max(longest, current)
    return {
        "status": "INFERRED_DAILY_CLOSE_PROXY",
        "field": field,
        "known_rows": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "positive_rate": float(values.gt(0).mean()),
        "average_win": float(wins.mean()) if not wins.empty else None,
        "average_loss": float(losses.mean()) if not losses.empty else None,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if not losses.empty else None,
        "max_single_loss": float(values.min()),
        "max_drawdown_daily_mean": float(drawdown.min()) if not drawdown.empty else None,
        "longest_loss_streak": int(longest),
        "actual_fill_count": 0,
    }


def _forward_bar_metrics(selected: pd.DataFrame) -> dict:
    """Summarize close-to-close and adverse/favorable excursion proxies."""
    result = {"status": "UNAVAILABLE_NO_BAR_PROXY"}
    for horizon in (1, 3, 5):
        field = f"forward_return_{horizon}d_proxy"
        if field not in selected:
            continue
        values = pd.to_numeric(selected[field], errors="coerce").dropna()
        if values.empty:
            result[f"{horizon}d"] = {"known_rows": 0}
            continue
        wins = values[values > 0]
        losses = values[values < 0]
        ordered = selected[["signal_date", field]].copy()
        ordered[field] = pd.to_numeric(ordered[field], errors="coerce")
        daily = ordered.dropna().groupby("signal_date")[field].mean().sort_index()
        cumulative = daily.cumsum()
        drawdown = cumulative - cumulative.cummax()
        longest = current = 0
        for is_loss in values.lt(0).tolist():
            current = current + 1 if is_loss else 0
            longest = max(longest, current)
        result["status"] = "INFERRED_CLOSE_TO_CLOSE_OHLC_PROXY"
        result[f"{horizon}d"] = {
            "known_rows": int(len(values)),
            "positive_rows": int(values.gt(0).sum()),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "positive_rate": float(values.gt(0).mean()),
            "average_win": float(wins.mean()) if not wins.empty else None,
            "average_loss": float(losses.mean()) if not losses.empty else None,
            "profit_factor": float(wins.sum() / abs(losses.sum())) if not losses.empty else None,
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "sharpe_annualized_proxy": float(values.mean() / values.std(ddof=1) * math.sqrt(252))
            if len(values) > 1 and values.std(ddof=1) > 0 else None,
            "max_drawdown_daily_mean": float(drawdown.min()) if not drawdown.empty else None,
            "longest_loss_streak": int(longest),
            "max_single_loss": float(values.min()),
        }
    if "forward_max_gain_5d_proxy" in selected:
        gain = pd.to_numeric(selected["forward_max_gain_5d_proxy"], errors="coerce").dropna()
        result["five_day_excursion"] = {
            "known_rows": int(len(gain)),
            "mean_max_gain": float(gain.mean()) if not gain.empty else None,
            "median_max_gain": float(gain.median()) if not gain.empty else None,
            "max_gain": float(gain.max()) if not gain.empty else None,
        }
    if "forward_max_drawdown_5d_proxy" in selected:
        draw = pd.to_numeric(selected["forward_max_drawdown_5d_proxy"], errors="coerce").dropna()
        result.setdefault("five_day_excursion", {})["mean_max_drawdown"] = (
            float(draw.mean()) if not draw.empty else None)
        result.setdefault("five_day_excursion", {})["worst_max_drawdown"] = (
            float(draw.min()) if not draw.empty else None)
    result["actual_fill_count"] = 0
    return result


def _metrics(selected: pd.DataFrame, pool: pd.DataFrame, target: str) -> dict:
    known = selected[target].isin([0, 1])
    y = selected.loc[known, target].astype(int)
    daily = selected.loc[known].groupby("signal_date")[target].max()
    total_positive = int(pool[target].eq(1).sum())
    selected_positive = int(y.sum())
    return {
        "selected_slots": int(len(selected)),
        "selected_dates": int(selected.signal_date.nunique()),
        "known_label_slots": int(len(y)),
        "positive_slots": selected_positive,
        "slot_precision": float(y.mean()) if len(y) else None,
        "slot_precision_wilson95": _wilson(selected_positive, len(y)),
        "daily_hit_rate": float(daily.mean()) if len(daily) else None,
        "daily_hit_dates": int(daily.sum()),
        "total_positive_pool_slots": total_positive,
        "positive_slot_recall": float(selected_positive / total_positive) if total_positive else None,
        "future_max_board_mean": float(selected.loc[known, "future_max_board"].mean()) if known.any() else None,
        "future_gain_mean": float(selected.loc[known, "future_gain"].mean()) if known.any() else None,
        "forward_bar_proxy": _forward_bar_metrics(selected),
        "proxy": _proxy_metrics(selected),
    }


def _failure_reasons(selected: pd.DataFrame, target: str) -> pd.DataFrame:
    failed = selected[selected[target].eq(0)].copy()
    reasons = []
    for _, row in failed.iterrows():
        if row.get("one_price_day_proxy") == 1:
            reason = "ONE_PRICE_UNBUYABLE"
        elif row.get("market_state_divergence") == 1:
            reason = "DIVERGENCE_CONTEXT"
        elif pd.notna(row.get("market_ladder_health")) and row.market_ladder_health < 0.5:
            reason = "LADDER_WEAK"
        elif pd.notna(row.get("market_crowding_risk")) and row.market_crowding_risk >= 0.7:
            reason = "CROWDING_RISK"
        elif pd.notna(row.get("market_promotion_mid")) and row.market_promotion_mid < 0.25:
            reason = "LOW_MID_PROMOTION"
        else:
            reason = "UNCLASSIFIED_PRE_SIGNAL_FAILURE"
        reasons.append(reason)
    if failed.empty:
        return pd.DataFrame(columns=["signal_date", "code", "boards", "failure_reason"])
    failed["failure_reason"] = reasons
    category = {
        "ONE_PRICE_UNBUYABLE": "D_ENTRY_OR_EXECUTION",
        "DIVERGENCE_CONTEXT": "C_EMOTION_CYCLE",
        "LADDER_WEAK": "B_THEME_OR_LADDER",
        "CROWDING_RISK": "C_EMOTION_CYCLE",
        "LOW_MID_PROMOTION": "B_THEME_OR_LADDER",
        "UNCLASSIFIED_PRE_SIGNAL_FAILURE": "A_LEADER_IDENTIFICATION",
    }
    failed["failure_category"] = failed.failure_reason.map(category).fillna(
        "A_LEADER_IDENTIFICATION")
    return failed[["signal_date", "code", "boards", "environment", "failure_category",
                   "failure_reason", target, "future_max_board", "future_gain"]]


def _environment_metrics(selected: pd.DataFrame, target: str) -> dict:
    result = {}
    for env, group in selected.groupby("environment", dropna=False, sort=True):
        result[str(env)] = _metrics(group, group, target)
    return result


def _evaluate_stage(train: pd.DataFrame, validation: pd.DataFrame, stage_name: str, spec: dict) -> tuple[dict, list[pd.DataFrame]]:
    stage = int(spec["boards"])
    validation = validation[validation.board_stage.eq(stage)].copy()
    train = train[train.board_stage.eq(stage)].copy()
    if validation.empty:
        return {"status": "NO_VALIDATION_ROWS", "stage": stage}, []
    score_target = spec["score_target"]
    score, fit_status, prior = _fit(train, validation, score_target)
    validation["model_score"] = score
    validation = validation[validation[score_target].isin([0, 1])].copy()
    if validation.empty:
        return {"status": "NO_VALID_LABELS", "stage": stage, "fit_status": fit_status}, []
    identity = _rank_scores(validation, "identity_score", "identity_rule")
    random = _rank_scores(validation, "random_score", "random")
    outputs = {}
    chosen = {}
    for name, frame, score_field in (("model", validation, "model_score"),
                                     ("identity_rule", identity, "identity_score"),
                                     ("random_control", random, "random_score")):
        chosen[name] = frame
        outputs[name] = {f"top{k}": _metrics(_select(frame, score_field, k), validation, spec["leader_target"])
                         for k in (1, 3, 5)}
        outputs[name]["top1_by_environment"] = _environment_metrics(
            _select(frame, score_field, 1), spec["leader_target"])
        failure_frame = _failure_reasons(
            _select(frame, score_field, 1), spec["leader_target"])
        outputs[name]["top1_failures"] = failure_frame.to_dict("records")
    report = {
        "status": "OK",
        "stage": stage,
        "stage_name": stage_name,
        "score_target": score_target,
        "leader_target": spec["leader_target"],
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "fit_status": fit_status,
        "training_prior": prior,
        "models": outputs,
        "theme_leader_status": "UNAVAILABLE_THEME_DATA",
    }
    failures = []
    for name, frame in chosen.items():
        failure = _failure_reasons(
            _select(frame, "model_score" if name == "model" else
                    ("identity_score" if name == "identity_rule" else "random_score"), 1),
            spec["leader_target"])
        if not failure.empty:
            failures.append(failure.assign(stage_name=stage_name, selection_model=name))
    return report, failures


def _combine_stage_reports(reports: list[dict], failures: list[pd.DataFrame], stage_name: str) -> dict:
    aggregate = {"folds": len(reports), "stage_name": stage_name, "models": {}}
    for model in ("model", "identity_rule", "random_control"):
        model_report = {}
        for k in (1, 3, 5):
            rows = [r["models"][model][f"top{k}"] for r in reports if r.get("status") == "OK"]
            if not rows:
                model_report[f"top{k}"] = {"status": "NO_FOLDS"}
                continue
            slots = sum(r.get("selected_slots", 0) for r in rows)
            positives = sum(r.get("positive_slots", 0) for r in rows)
            dates = sum(r.get("selected_dates", 0) for r in rows)
            hit_dates = sum(r.get("daily_hit_dates", 0) for r in rows)
            total_pool = sum(r.get("total_positive_pool_slots", 0) for r in rows)
            model_report[f"top{k}"] = {
                "selected_slots": slots,
                "selected_dates": dates,
                "positive_slots": positives,
                "slot_precision": positives / slots if slots else None,
                "slot_precision_wilson95": _wilson(positives, slots),
                "daily_hit_rate": hit_dates / dates if dates else None,
                "positive_slot_recall": positives / total_pool if total_pool else None,
            }
        # Aggregate forward path proxies from the same selected TopK rows.
        proxy_reports = [r["models"][model]["top1"].get("forward_bar_proxy", {})
                         for r in reports if r.get("status") == "OK"]
        aggregate_proxy = {}
        for horizon in (1, 3, 5):
            items = [p.get(f"{horizon}d", {}) for p in proxy_reports
                     if p.get(f"{horizon}d", {}).get("known_rows", 0)]
            known = sum(int(item.get("known_rows", 0)) for item in items)
            if not known:
                aggregate_proxy[f"{horizon}d"] = {"known_rows": 0}
                continue
            aggregate_proxy[f"{horizon}d"] = {
                "known_rows": known,
                "mean": sum(item["known_rows"] * item["mean"] for item in items) / known,
                "positive_rate": sum(item["known_rows"] * item["positive_rate"]
                                      for item in items) / known,
                "positive_rows": sum(int(item.get("positive_rows", 0)) for item in items),
                "mean_drawdown_daily_mean": sum(
                    item["known_rows"] * item["max_drawdown_daily_mean"] for item in items
                    if item.get("max_drawdown_daily_mean") is not None) / known,
                "worst_single_loss": min(item["max_single_loss"] for item in items),
                "longest_loss_streak": max(item.get("longest_loss_streak", 0) for item in items),
            }
            global_mean = aggregate_proxy[f"{horizon}d"]["mean"]
            sum_sq = sum(
                item["known_rows"] * ((item.get("std") or 0) ** 2 +
                                       (item["mean"] - global_mean) ** 2)
                for item in items)
            pooled_std = math.sqrt(sum_sq / (known - 1)) if known > 1 else None
            aggregate_proxy[f"{horizon}d"]["std"] = pooled_std
            aggregate_proxy[f"{horizon}d"]["sharpe_annualized_proxy"] = (
                global_mean / pooled_std * math.sqrt(252)
                if pooled_std and pooled_std > 0 else None)
        excursion = [p.get("five_day_excursion", {}) for p in proxy_reports
                     if p.get("five_day_excursion", {}).get("known_rows", 0)]
        if excursion:
            known = sum(int(item["known_rows"]) for item in excursion)
            aggregate_proxy["five_day_excursion"] = {
                "known_rows": known,
                "mean_max_gain": sum(item["known_rows"] * item["mean_max_gain"]
                                      for item in excursion) / known,
                "mean_max_drawdown": sum(item["known_rows"] * item["mean_max_drawdown"]
                                          for item in excursion) / known,
                "worst_max_drawdown": min(item["worst_max_drawdown"] for item in excursion),
            }
        aggregate_proxy["status"] = (
            "INFERRED_CLOSE_TO_CLOSE_OHLC_PROXY"
            if any(p.get("status") == "INFERRED_CLOSE_TO_CLOSE_OHLC_PROXY"
                   for p in proxy_reports)
            else "UNAVAILABLE_NO_BAR_PROXY")
        model_report["top1_forward_bar_proxy"] = aggregate_proxy
        environment_pool = {}
        for report in reports:
            if report.get("status") != "OK":
                continue
            for env, item in report["models"][model].get("top1_by_environment", {}).items():
                bucket = environment_pool.setdefault(env, {"selected_slots": 0, "positive_slots": 0,
                                                            "selected_dates": 0, "hit_dates": 0})
                bucket["selected_slots"] += item.get("known_label_slots", 0)
                bucket["positive_slots"] += item.get("positive_slots", 0)
                bucket["selected_dates"] += item.get("selected_dates", 0)
                bucket["hit_dates"] += item.get("daily_hit_dates", 0)
        model_report["top1_by_environment"] = {
            env: {
                **item,
                "slot_precision": item["positive_slots"] / item["selected_slots"]
                if item["selected_slots"] else None,
                "daily_hit_rate": item["hit_dates"] / item["selected_dates"]
                if item["selected_dates"] else None,
            }
            for env, item in sorted(environment_pool.items())
        }
        known_future = sum(r["models"][model]["top1"].get("known_label_slots", 0)
                           for r in reports if r.get("status") == "OK")
        if known_future:
            model_report["top1_future_max_board_mean"] = sum(
                r["models"][model]["top1"].get("known_label_slots", 0) *
                (r["models"][model]["top1"].get("future_max_board_mean") or 0)
                for r in reports if r.get("status") == "OK") / known_future
            model_report["top1_future_gain_mean"] = sum(
                r["models"][model]["top1"].get("known_label_slots", 0) *
                (r["models"][model]["top1"].get("future_gain_mean") or 0)
                for r in reports if r.get("status") == "OK") / known_future
        aggregate["models"][model] = model_report
    aggregate["failure_counts_top1"] = (
        pd.concat(failures).failure_reason.value_counts().to_dict() if failures else {}
    )
    aggregate["failure_category_counts_top1"] = (
        pd.concat(failures).failure_category.value_counts().to_dict() if failures else {}
    )
    return aggregate


def run(features: pd.DataFrame, outcomes: pd.DataFrame | None = None,
        bars: pd.DataFrame | None = None) -> tuple[dict, list[pd.DataFrame]]:
    rows, audit = _prepare(features, outcomes, bars)
    fold_reports = []
    all_failures = []
    for fold in FOLDS:
        eligible = rows[rows.future_label_status.eq("COMPLETE_HORIZON") &
                        rows.one_price_day_proxy.eq(0)].copy()
        calibration = eligible[(eligible.signal_date >= fold["validation_start"]) &
                               (eligible.signal_date <= fold["validation_end"])].copy()
        blind = eligible[(eligible.signal_date >= fold["blind_start"]) &
                         (eligible.signal_date <= fold["blind_end"])].copy()
        train_base = eligible[eligible.signal_date < fold["validation_start"]].copy()
        # Remove every episode visible in calibration or blind from the fitting
        # pool. This prevents a continuous relay from leaking future labels into
        # an earlier split even when its earlier rows predate the split boundary.
        embargo_episodes = set(calibration.episode_id.dropna()) | set(blind.episode_id.dropna())
        train = train_base[~train_base.episode_id.isin(embargo_episodes)].copy()
        fit_after_validation = pd.concat([train, calibration], ignore_index=True)
        fit_after_validation = fit_after_validation[
            ~fit_after_validation.episode_id.isin(set(blind.episode_id.dropna()))
        ].copy()
        item = {"fold": fold, "training_rows": int(len(train)),
                "validation_rows": int(len(calibration)), "blind_rows": int(len(blind)),
                "stages": {}}
        fold_failures = []
        for name, spec in STAGES.items():
            validation_report, validation_failures = _evaluate_stage(
                train, calibration, name, spec)
            blind_report, blind_failures = _evaluate_stage(
                fit_after_validation, blind, name, spec)
            item["stages"][name] = {
                "validation": validation_report,
                "blind": blind_report,
            }
            # Only the blind partition contributes to the aggregate failure
            # taxonomy. Validation remains a model-selection/audit partition.
            fold_failures.extend(blind_failures)
        all_failures.extend(fold_failures)
        fold_reports.append(item)
    aggregate = {}
    for stage_name in STAGES:
        reports = [fold["stages"][stage_name]["blind"] for fold in fold_reports]
        failures = [f for f in all_failures if f.stage_name.iloc[0] == stage_name] if all_failures else []
        aggregate[stage_name] = _combine_stage_reports(reports, failures, stage_name)
    result = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "target_definitions": {
            "A_first_board": "future_promotion: same episode reaches at least one higher board within next five observed sessions",
            "B_second_board": "future_stage_leader: largest future board gain among same-stage candidates, requiring gain >=2; verified theme leader unavailable",
            "C_third_board": "future_market_leader: tied highest future board among current 1-3 boards within horizon, requiring future max >=5",
        },
        "topk": [1, 3, 5],
        "folds": fold_reports,
        "aggregate": aggregate,
        "data_audit": audit,
        "feature_fields": FEATURES,
        "interpretation": {
            "is_pristine_blind_test": False,
            "split_protocol": "TRAIN -> VALIDATION -> BLIND, fixed dates; blind is temporally held out but source is reconstructed",
            "historical_reconstruction": True,
            "theme_leader_supported": False,
            "actual_fill_count": 0,
            "entry_authorized": False,
            "probabilities_calibrated": False,
            "future_labels_in_features": False,
            "source_manifest_training_authorized": False,
            "survivorship_risk": True,
            "calendar_status": "PROVISIONAL_FEATURE_DATE_SEQUENCE_NOT_INDEPENDENTLY_VERIFIED",
            "execution_status": "CASH_DIAGNOSTIC_ONLY",
        },
        "limitations": [
            "features and labels are reconstructed after the fact, not a contemporaneous archive",
            "candidate universe is not complete and has survivorship risk",
            "blind partition is not pristine because the source was reconstructed after the fact and was available for research",
            "topic membership is unavailable, so B is not a verified topic-leader label",
            "future five-session labels use the observed feature-date sequence and are not independently calendar-verified",
            "proxy path fields are inferred daily OHLC marks and have zero actual fills",
            "1/3/5-day and five-session excursion fields are close-to-close OHLC proxies; they do not model auction fills, queue priority, slippage, or intraday stops",
            "TopK hit/precision is identity evidence, not executable trade win rate",
        ],
    }
    return _clean(result), all_failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--bars", type=Path,
                        help="optional reconstructed daily OHLC bars for non-executable return proxies")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable output directory already exists")
    features = pd.read_csv(args.features, dtype={"code": str}, low_memory=False)
    outcomes = (pd.read_csv(args.outcomes, dtype={"code": str}, low_memory=False)
                if args.outcomes else None)
    bars = (pd.read_csv(args.bars, dtype={"code": str}, low_memory=False)
            if args.bars else None)
    result, failures = run(features, outcomes, bars)
    args.output.mkdir(parents=True)
    source_paths = [args.features, Path(__file__)]
    if args.outcomes:
        source_paths.append(args.outcomes)
    if args.bars:
        source_paths.append(args.bars)
    result["source_hashes"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in source_paths}
    (args.output / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    if failures:
        pd.concat(failures, ignore_index=True).to_csv(
            args.output / "top1_failures.csv", index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame().to_csv(args.output / "top1_failures.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"version": result["version"], "status": result["status"],
                      "aggregate": result["aggregate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
