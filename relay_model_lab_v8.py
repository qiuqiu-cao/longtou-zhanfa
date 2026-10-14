#!/usr/bin/env python3
"""Causal V8 laboratory for 2--5 board relay and T+1 execution risk.

The lab predicts four separate observables: next-day promotion, next-day
buyability, at least two additional boards, and large loss conditional on a
daily-limit-touch proxy fill.  All model training respects label maturity.
Historical dates are development evidence only and cannot promote deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from emotion_challenger_v4 import ACTIVE_STATES


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "v7" / "point_in_time_rebuild" / "leader_candidates.csv"
EMOTION_PREDICTIONS = ROOT / "results" / "v7" / "emotion_model_lab" / "prequential_predictions.csv"
OUTPUT = ROOT / "results" / "v8" / "relay_model_lab"
VERSION = "v8-relay-model-lab-1"
INITIAL_DATES = 220
RETRAIN_DATES = 20
FILL_STATUSES = {"inferred_daily_limit_touch_fill", "sell_blocked_one_word_down"}

STATE_FEATURES = [
    "state_start", "state_warming", "state_acceleration", "state_consensus",
    "state_divergence", "state_retreat", "state_repair",
]
BASE_FEATURES = [
    "boards", "height_gap", "is_space_leader", "unique_space", "height_share",
    "valid_identity", "buyable_eod", "one_word_proxy", "one_word_streak",
    "same_height_rivals", "breadth_health", "relay_appetite", "ladder_health",
    "loss_effect_health", "crowding_risk", "regime_fragility", "false_acceleration",
    "promotion_mid", "failed_rate", "down_limit_rate", "middle_density",
    "ladder_gap_count", "breadth_height_divergence", "yesterday_premium",
    "previous_limit_tail_loss", "emotion_adverse_markov_probability",
    "emotion_v7_adverse_probability", *STATE_FEATURES,
]
INDIVIDUAL_FEATURES = [
    "turnover", "turnover_vs_prev", "amount_float_cap", "amplitude",
    "relative_strength", "price", "float_cap_log",
]
RANK_FEATURES = [
    "turnover_rank_pct", "turnover_change_rank_pct", "amount_float_rank_pct",
    "amplitude_rank_pct", "relative_strength_rank_pct", "price_rank_pct",
    "float_cap_rank_pct",
]
MEMORY_FEATURES = [
    "prior_signal_count_log", "prior_promotion_count", "prior_promotion_rate",
    "prior_two_more_count", "prior_two_more_rate", "prior_large_loss_count",
    "prior_large_loss_rate", "prior_terminal_max", "prior_campaign_count_log",
    "days_since_prior_campaign",
]
FEATURE_STAGES = {
    "structure": BASE_FEATURES,
    "plus_individual": BASE_FEATURES + INDIVIDUAL_FEATURES,
    "plus_cross_section": BASE_FEATURES + INDIVIDUAL_FEATURES + RANK_FEATURES,
    "plus_stock_memory": BASE_FEATURES + INDIVIDUAL_FEATURES + RANK_FEATURES + MEMORY_FEATURES,
}
TARGETS = {
    "promotion": {"label": "promotion_target", "maturity": "promotion_maturity"},
    "fill": {"label": "fill_target", "maturity": "fill_maturity"},
    "two_more": {"label": "two_more_target", "maturity": "two_more_maturity"},
    "large_loss": {"label": "large_loss_target", "maturity": "large_loss_maturity"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nullable_bool(series: pd.Series) -> pd.Series:
    """Parse CSV boolean-like values without turning missing values into False."""
    mapping = {
        "true": True, "1": True, "1.0": True, "yes": True,
        "false": False, "0": False, "0.0": False, "no": False,
    }
    text = series.astype("string").str.strip().str.lower()
    return text.map(mapping).astype("boolean")


def derive_fill_target(frame: pd.DataFrame) -> pd.Series:
    """Label a daily-bar proxy fill from limit touch and one-word status.

    The exit bar is irrelevant to whether the trade-date order could fill.  In
    particular, ``missing_exit_bar`` remains a positive fill label when the
    next day touched limit and was not a one-word up board.
    """
    touched = nullable_bool(frame["trade_touched_limit"])
    one_word_up = nullable_bool(frame["trade_one_word_up"])
    known = touched.notna() & one_word_up.notna()
    values = (touched & ~one_word_up).astype("Float64")
    return values.where(known, np.nan).astype(float)


def prepare_frame(input_path: Path = INPUT, emotion_path: Path = EMOTION_PREDICTIONS) -> pd.DataFrame:
    frame = pd.read_csv(input_path, dtype={"code": "string"}, low_memory=False)
    frame["code"] = frame["code"].astype("string").str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame = frame[frame["boards"].between(2, 5)].copy()
    frame["promotion_target"] = pd.to_numeric(frame["promoted_next_day"], errors="coerce")
    frame["fill_target"] = derive_fill_target(frame)
    terminal_known = frame["terminal_height_label_status"].eq("inferred_from_future_daily_limit_sequence")
    frame["two_more_target"] = np.where(
        terminal_known, (pd.to_numeric(frame["additional_boards"], errors="coerce") >= 2).astype(float), np.nan,
    )
    frame["large_loss_target"] = pd.to_numeric(frame["large_loss"], errors="coerce")
    frame["promotion_maturity"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["fill_maturity"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["two_more_maturity"] = pd.to_datetime(frame["terminal_date"], errors="coerce")
    frame["large_loss_maturity"] = pd.to_datetime(frame["exit_date"], errors="coerce")

    emotion = pd.read_csv(emotion_path, usecols=[
        "date", "adverse_state_next__plus_dimensions__logistic",
    ])
    emotion["date"] = pd.to_datetime(emotion["date"], errors="raise").dt.normalize()
    emotion = emotion.rename(columns={
        "adverse_state_next__plus_dimensions__logistic": "emotion_v7_adverse_probability",
    })
    frame = frame.merge(emotion, on="date", how="left", validate="many_to_one")
    frame = add_cross_sectional_ranks(frame)
    frame = add_causal_stock_memory(frame)
    return frame.sort_values(["date", "boards", "code"]).reset_index(drop=True)


def add_cross_sectional_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    mapping = {
        "turnover": "turnover_rank_pct", "turnover_vs_prev": "turnover_change_rank_pct",
        "amount_float_cap": "amount_float_rank_pct", "amplitude": "amplitude_rank_pct",
        "relative_strength": "relative_strength_rank_pct", "price": "price_rank_pct",
        "float_cap_log": "float_cap_rank_pct",
    }
    for source, target in mapping.items():
        values = pd.to_numeric(result[source], errors="coerce")
        result[target] = values.groupby(result["date"]).rank(pct=True, method="average")
    return result


def add_causal_stock_memory(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for name in MEMORY_FEATURES:
        result[name] = np.nan
    for _, indices in result.groupby("code", sort=False).groups.items():
        ordered_indices = list(result.loc[indices].sort_values("date").index)
        for position, index in enumerate(ordered_indices):
            current_date = result.at[index, "date"]
            prior_indices = ordered_indices[:position]
            result.at[index, "prior_signal_count_log"] = math.log1p(len(prior_indices))
            if not prior_indices:
                result.at[index, "prior_campaign_count_log"] = 0.0
                continue
            prior = result.loc[prior_indices]
            promotion = prior[
                prior["promotion_target"].notna() & prior["promotion_maturity"].notna()
                & (prior["promotion_maturity"] <= current_date)
            ]
            two_more = prior[
                prior["two_more_target"].notna() & prior["two_more_maturity"].notna()
                & (prior["two_more_maturity"] < current_date)
            ]
            loss = prior[
                prior["large_loss_target"].notna() & prior["large_loss_maturity"].notna()
                & (prior["large_loss_maturity"] <= current_date)
            ]
            completed = prior[prior["two_more_maturity"].notna() & (prior["two_more_maturity"] < current_date)]
            result.at[index, "prior_promotion_count"] = len(promotion)
            result.at[index, "prior_promotion_rate"] = promotion["promotion_target"].mean() if len(promotion) else np.nan
            result.at[index, "prior_two_more_count"] = len(two_more)
            result.at[index, "prior_two_more_rate"] = two_more["two_more_target"].mean() if len(two_more) else np.nan
            result.at[index, "prior_large_loss_count"] = len(loss)
            result.at[index, "prior_large_loss_rate"] = loss["large_loss_target"].mean() if len(loss) else np.nan
            if len(completed):
                result.at[index, "prior_terminal_max"] = pd.to_numeric(completed["terminal_board"], errors="coerce").max()
                terminal_dates = completed["two_more_maturity"].dropna()
                result.at[index, "prior_campaign_count_log"] = math.log1p(terminal_dates.nunique())
                result.at[index, "days_since_prior_campaign"] = (current_date - terminal_dates.max()).days
            else:
                result.at[index, "prior_campaign_count_log"] = 0.0
    return result


def make_model(kind: str) -> Pipeline:
    if kind == "logistic":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.20, max_iter=2500, random_state=17)),
        ])
    if kind == "hist_gb":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.035, max_iter=140, max_leaf_nodes=7,
                min_samples_leaf=35, l2_regularization=2.5,
                early_stopping=False, random_state=17,
            )),
        ])
    raise ValueError(kind)


def platt(raw_cal: np.ndarray, y_cal: np.ndarray, raw_test: np.ndarray) -> np.ndarray:
    if len(y_cal) < 60 or np.unique(y_cal).size < 2:
        return raw_test
    eps = 1e-6
    transform = lambda p: np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps)).reshape(-1, 1)
    model = LogisticRegression(C=0.5, max_iter=1000, random_state=19)
    model.fit(transform(raw_cal), y_cal)
    return model.predict_proba(transform(raw_test))[:, 1]


def prequential(
    frame: pd.DataFrame, target_name: str, features: list[str], kind: str,
    initial_dates: int = INITIAL_DATES, retrain_dates: int = RETRAIN_DATES,
) -> tuple[np.ndarray, np.ndarray]:
    spec = TARGETS[target_name]
    label, maturity = spec["label"], spec["maturity"]
    dates = np.array(sorted(frame["date"].unique()))
    probabilities = np.full(len(frame), np.nan)
    baselines = np.full(len(frame), np.nan)
    for start in range(initial_dates, len(dates), retrain_dates):
        block_dates = dates[start:min(len(dates), start + retrain_dates)]
        current_date = pd.Timestamp(block_dates[0])
        train = frame[
            (frame["date"] < current_date) & frame[label].notna()
            & frame[maturity].notna() & (frame[maturity] <= current_date)
        ].copy()
        train_dates = np.array(sorted(train["date"].unique()))
        if len(train) < 500 or len(train_dates) < 160 or train[label].nunique() < 2:
            continue
        split = max(1, int(len(train_dates) * 0.80))
        core_dates = set(train_dates[:split])
        calibration_dates = set(train_dates[min(len(train_dates), split + 1):])
        core = train[train["date"].isin(core_dates)]
        calibration = train[train["date"].isin(calibration_dates)]
        if len(core) < 300 or len(calibration) < 60 or core[label].nunique() < 2:
            continue
        model = make_model(kind)
        model.fit(core[features], core[label].astype(int))
        raw_test = model.predict_proba(frame[frame["date"].isin(block_dates)][features])[:, 1]
        if calibration[label].nunique() == 2:
            raw_cal = model.predict_proba(calibration[features])[:, 1]
            raw_test = platt(raw_cal, calibration[label].astype(int).to_numpy(), raw_test)
        indices = frame.index[frame["date"].isin(block_dates)]
        probabilities[indices] = raw_test
        baselines[indices] = (float(train[label].sum()) + 1.0) / (len(train) + 2.0)
    return probabilities, baselines


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if np.unique(y).size == 2 else math.nan


def metric_block(y: np.ndarray, p: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(y) & np.isfinite(p) & np.isfinite(baseline)
    y, p, baseline = y[mask], np.clip(p[mask], 1e-6, 1 - 1e-6), np.clip(baseline[mask], 1e-6, 1 - 1e-6)
    if not len(y):
        return {"n": 0}
    brier, base_brier = float(np.mean((p - y) ** 2)), float(np.mean((baseline - y) ** 2))
    return {
        "n": int(len(y)), "positive_rate": float(y.mean()), "auc": safe_auc(y, p),
        "brier": brier, "baseline_brier": base_brier,
        "brier_skill_vs_baseline": 1 - brier / base_brier if base_brier else math.nan,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "baseline_log_loss": float(log_loss(y, baseline, labels=[0, 1])),
        "log_loss_improvement_vs_baseline": float(log_loss(y, baseline, labels=[0, 1]) - log_loss(y, p, labels=[0, 1])),
    }


def evaluate(frame: pd.DataFrame, target_name: str, p: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    label = TARGETS[target_name]["label"]
    y = frame[label].to_numpy(float)
    valid = np.flatnonzero(np.isfinite(y) & np.isfinite(p) & np.isfinite(baseline))
    date_values = np.array(sorted(frame.loc[valid, "date"].unique()))
    folds = []
    for number, fold_dates in enumerate(np.array_split(date_values, 3), 1):
        indices = valid[frame.loc[valid, "date"].isin(fold_dates).to_numpy()]
        folds.append({"fold": number, **metric_block(y[indices], p[indices], baseline[indices])})

    def gate(items: list[dict[str, Any]]) -> bool:
        return len(items) == 3 and all(
            item.get("auc", math.nan) > 0.5
            and item.get("brier_skill_vs_baseline", math.nan) > 0
            and item.get("log_loss_improvement_vs_baseline", math.nan) > 0
            for item in items
        )

    board_rows = {}
    for board in range(2, 6):
        indices = valid[frame.loc[valid, "boards"].eq(board).to_numpy()]
        board_rows[str(board)] = metric_block(y[indices], p[indices], baseline[indices])
    active = frame["emotion_state"].isin(ACTIVE_STATES).to_numpy(bool)
    active_indices = valid[active[valid]]
    return {
        "overall": metric_block(y[valid], p[valid], baseline[valid]),
        "folds": folds, "stability_gate": gate(folds),
        "active": metric_block(y[active_indices], p[active_indices], baseline[active_indices]),
        "by_board": board_rows,
    }


def run(input_path: Path = INPUT, emotion_path: Path = EMOTION_PREDICTIONS, output: Path = OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    frame = prepare_frame(input_path, emotion_path)
    missing = sorted({x for values in FEATURE_STAGES.values() for x in values}.difference(frame.columns))
    if missing:
        raise ValueError(f"V8 missing features: {missing}")
    predictions = frame[[
        "date", "trade_date", "exit_date", "terminal_date", "code", "name", "boards",
        "emotion_state", "valid_identity", "is_space_leader", "unique_space",
        "buyable_eod", "one_word_proxy", "execution_status", "promoted_next_day",
        "large_loss", "net_return", "promotion_target", "fill_target", "two_more_target",
        "large_loss_target",
    ]].copy()
    experiments = []
    for target in TARGETS:
        for stage, features in FEATURE_STAGES.items():
            kinds = ("logistic", "hist_gb") if stage == "plus_stock_memory" else ("logistic",)
            for kind in kinds:
                name = f"{target}__{stage}__{kind}"
                p, baseline = prequential(frame, target, features, kind)
                predictions[f"p__{name}"] = p
                predictions[f"baseline__{name}"] = baseline
                experiments.append({
                    "name": name, "target": target, "feature_stage": stage,
                    "model": kind, "feature_count": len(features),
                    **evaluate(frame, target, p, baseline),
                })
    predictions_path = output / "prequential_predictions.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    result = {
        "version": VERSION, "status": "RESEARCH_ONLY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_start": frame["date"].min().strftime("%Y-%m-%d"),
        "data_end": frame["date"].max().strftime("%Y-%m-%d"),
        "rows": int(len(frame)), "dates": int(frame["date"].nunique()),
        "initial_dates": INITIAL_DATES, "retrain_dates": RETRAIN_DATES,
        "label_maturity_enforced": True, "calibration_embargo_dates": 1,
        "feature_stages": FEATURE_STAGES, "targets": TARGETS,
        "experiments": experiments,
        "stable_experiments": [item["name"] for item in experiments if item["stability_gate"]],
        "deployment_allowed": False,
        "deployment_reason": "historical dates are reused development evidence and execution labels are inferred daily OHLC proxies",
        "input": {"path": str(input_path.resolve()), "sha256": sha256_file(input_path)},
        "emotion_input": {"path": str(emotion_path.resolve()), "sha256": sha256_file(emotion_path)},
        "predictions": {"path": str(predictions_path.resolve()), "sha256": sha256_file(predictions_path)},
    }
    card_path = output / "relay_model_lab_card.json"
    card_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["card"] = str(card_path.resolve())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run causal V8 relay model lab")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--emotion", type=Path, default=EMOTION_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args.input, args.emotion, args.output)
    print(json.dumps({
        "status": result["status"], "rows": result["rows"], "dates": result["dates"],
        "stable_experiments": result["stable_experiments"], "card": result["card"],
    }, ensure_ascii=False, indent=2))
