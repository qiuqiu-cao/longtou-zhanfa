"""Fail-closed selective policy experiment for the reconstructed full pool.

This is deliberately a proxy feasibility audit.  A threshold can only be
chosen from the calibration window.  If calibration cannot meet the fixed
precision and sample requirements, the operational threshold remains null and
later windows are shown only as fixed-grid diagnostics, never as deployment
evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from full_pool_proxy_backtest import FULL, TARGET, prepare, select_top1


THRESHOLDS = tuple(round(value, 2) for value in np.arange(.10, .451, .05))
WINDOWS = {
    "calibration": ("2025-01-01", "2025-04-01"),
    "validation": ("2025-04-01", "2025-07-28"),
    "reused_research_evaluation": ("2025-07-28", "2025-12-26"),
}
TRAIN_END = "2025-01-01"
TARGET_PRECISION = .70
MINIMUM_SELECTIONS = 30
MINIMUM_SIGNALS_PER_ACTIVE_WEEK = 3.
MAXIMUM_SIGNALS_PER_ACTIVE_WEEK = 5.
ACTIVE_STATE_COLUMNS = [
    "market_state_start", "market_state_warming", "market_state_acceleration", "market_state_repair",
]


def metrics(selected, active_weeks=None):
    count = len(selected)
    successes = int(selected.opportunity_proxy_label.sum()) if count else 0
    weekly = count / active_weeks if active_weeks else None
    return {
        "selections": int(count),
        "proxy_successes": successes,
        "proxy_precision": float(successes / count) if count else None,
        "first_board_selections": int(selected.board_group.eq("FIRST_BOARD").sum()) if count else 0,
        "consecutive_board_selections": int(selected.board_group.eq("CONSECUTIVE_BOARD").sum()) if count else 0,
        "signals_per_active_week": float(weekly) if weekly is not None else None,
    }


def threshold_frontier(top1):
    if "emotion_active" not in top1 or top1.emotion_active.dtype != bool:
        raise ValueError("emotion-active flag missing or invalid")
    active = top1[top1.emotion_active]
    active_weeks = active.signal_date.map(
        lambda value: tuple(pd.Timestamp(value).isocalendar()[:2])).nunique()
    rows = []
    for threshold in THRESHOLDS:
        result = metrics(active[active.model_score_NOT_LIVE >= threshold], active_weeks)
        result.update(
            threshold=threshold,
            minimum_sample_pass=result["selections"] >= MINIMUM_SELECTIONS,
            proxy_precision_pass=(result["proxy_precision"] is not None
                                  and result["proxy_precision"] >= TARGET_PRECISION),
            active_week_frequency_pass=(result["signals_per_active_week"] is not None
                                        and MINIMUM_SIGNALS_PER_ACTIVE_WEEK
                                        <= result["signals_per_active_week"]
                                        <= MAXIMUM_SIGNALS_PER_ACTIVE_WEEK),
        )
        result["proxy_gate_pass"] = (result["minimum_sample_pass"]
                                     and result["proxy_precision_pass"]
                                     and result["active_week_frequency_pass"])
        rows.append(result)
    return rows


def best_with_minimum(frontier):
    eligible = [row for row in frontier if row["minimum_sample_pass"]]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (row["proxy_precision"], row["selections"], row["threshold"]))


def run(features, outcomes):
    rows = prepare(features, outcomes)
    future = rows[rows.signal_date >= TRAIN_END]
    train = rows[(rows.signal_date < TRAIN_END)
                 & rows.exit_date_LABEL_ONLY.notna()
                 & (rows.exit_date_LABEL_ONLY < TRAIN_END)].copy()
    train = train[~train.episode_id.isin(future.episode_id)].copy()
    if train.empty or train.opportunity_proxy_label.nunique() != 2:
        raise ValueError("training partition lacks data or both classes")
    weights = 1 / train.groupby("signal_date").signal_date.transform("size")
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(), LogisticRegression(C=1, max_iter=3000, random_state=67),
    )
    model.fit(train[FULL], train.opportunity_proxy_label,
              logisticregression__sample_weight=weights)
    output, scored_top1 = {}, {}
    for name, (start, end) in WINDOWS.items():
        window = rows[(rows.signal_date >= start) & (rows.signal_date < end)].copy()
        if window.empty:
            raise ValueError("empty policy window: " + name)
        window["model_score_NOT_LIVE"] = model.predict_proba(window[FULL])[:, 1]
        state_values = window[ACTIVE_STATE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        window["emotion_active"] = state_values.eq(1).any(axis=1)
        top1 = select_top1(window, "model_score_NOT_LIVE")
        active_weeks = top1[top1.emotion_active].signal_date.map(
            lambda value: tuple(pd.Timestamp(value).isocalendar()[:2])).nunique()
        frontier = threshold_frontier(top1)
        scored_top1[name] = top1
        output[name] = {
            "start": start, "end_exclusive": end,
            "candidate_rows": int(len(window)), "sessions": int(window.signal_date.nunique()),
            "emotion_active_sessions": int(top1.emotion_active.sum()),
            "unconditional_top1": metrics(top1, active_weeks),
            "emotion_active_top1": metrics(top1[top1.emotion_active], active_weeks),
            "fixed_threshold_frontier": frontier,
            "best_fixed_threshold_with_minimum_sample": best_with_minimum(frontier),
        }
    feasible = [row for row in output["calibration"]["fixed_threshold_frontier"] if row["proxy_gate_pass"]]
    chosen = max(feasible, key=lambda row: (row["selections"], row["proxy_precision"], row["threshold"])) if feasible else None
    # Fail closed: do not invent or relax a threshold after calibration failure.
    operational = chosen["threshold"] if chosen else None
    policy_validation = None
    if operational is not None:
        validation_top1 = scored_top1["validation"]
        active = validation_top1[validation_top1.emotion_active]
        active_weeks = active.signal_date.map(
            lambda value: tuple(pd.Timestamp(value).isocalendar()[:2])).nunique()
        policy_validation = metrics(active[active.model_score_NOT_LIVE >= operational], active_weeks)
    report = {
        "status": "SELECTIVE_PROXY_THRESHOLD_FROZEN" if chosen else "NO_SELECTIVE_PROXY_THRESHOLD",
        "target": TARGET,
        "model": "same_day_weighted_logistic_identity_emotion_price_volume",
        "train_end_exclusive": TRAIN_END,
        "training_rows": int(len(train)), "training_sessions": int(train.signal_date.nunique()),
        "outcome_availability_purge": True, "episode_overlap_purged": True,
        "threshold_grid_frozen": list(THRESHOLDS),
        "calibration_requirements": {
            "minimum_proxy_precision": TARGET_PRECISION,
            "minimum_selections": MINIMUM_SELECTIONS,
            "signals_per_active_week": [MINIMUM_SIGNALS_PER_ACTIVE_WEEK,
                                         MAXIMUM_SIGNALS_PER_ACTIVE_WEEK],
        },
        "operational_threshold": operational,
        "default_decision": "CASH" if operational is None else "CONDITIONAL_PAPER_ONLY",
        "validation_at_operational_threshold": policy_validation,
        "windows": output,
        "deployment_authorized": False,
        "complete_trade_evidence": False,
        "limitations": [
            "the target is a reconstructed daily price-opportunity proxy, not a completed trade",
            "all reported windows are historical research windows and none is a sealed future blind period",
            "emotion-active weeks use reconstructed same-day state flags and are not original forward archives",
            "no actual order, queue, fill, exit, announcement, or point-in-time theme evidence is present",
            "a null threshold means cash; validation or later-window results may not be used to rescue calibration",
        ],
    }
    predictions = pd.concat(
        [frame.assign(window=name) for name, frame in scored_top1.items()], ignore_index=True)
    return predictions, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite immutable policy experiment")
    inputs = [args.features, args.outcomes, Path(__file__)]
    predictions, report = run(
        pd.read_csv(args.features, dtype={"code": str}),
        pd.read_csv(args.outcomes, dtype={"code": str}),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(args.output / "daily_top1_scores_RESEARCH_ONLY.csv", index=False, encoding="utf-8-sig")
    report["source_hashes"] = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs
    }
    with (args.output / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
