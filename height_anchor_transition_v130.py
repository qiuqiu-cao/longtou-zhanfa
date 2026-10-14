"""Walk-forward audit for becoming or retaining the known market height anchor.

Labels are derived only from the next one to three trading sessions.  Training
for a validation date excludes the preceding three signal sessions so no open
future label window crosses the split.  This is a reconstructed-pool proxy, not
proof of full-market historical completeness or actual trade profitability.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "boards_for_comparison", "height_gap_to_known_top", "same_height_known_count",
    "sole_known_top_proxy", "turnover", "amount_to_float_cap", "one_price_day_proxy",
    "open_gap_pct", "amplitude_pct", "market_consecutive_count", "market_max_board",
    "market_failed_rate", "market_seal_rate", "market_ladder_gap_count",
    "market_yesterday_premium", "market_previous_limit_tail_loss",
    "market_promotion_overall", "market_relay_appetite", "market_crowding_risk",
    "market_regime_fragility",
]


def build_labels(frame: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    data = frame.copy()
    data["code"] = data.code.astype(str).str.zfill(6)
    data["signal_date"] = pd.to_datetime(data.signal_date).dt.strftime("%Y-%m-%d")
    dates = sorted(data.signal_date.unique())
    position = {date: index for index, date in enumerate(dates)}
    by_date = {date: group.set_index("code", drop=False) for date, group in data.groupby("signal_date")}
    labels = []
    for row in data.itertuples(index=False):
        index = position[row.signal_date]
        next_dates = dates[index + 1:index + horizon + 1]
        top_each_day = []
        for future_date in next_dates:
            future = by_date[future_date]
            if row.code not in future.index:
                top_each_day.append(0)
                continue
            candidate = future.loc[row.code]
            if isinstance(candidate, pd.DataFrame):
                candidate_board = pd.to_numeric(candidate.boards_for_comparison, errors="coerce").max()
            else:
                candidate_board = pd.to_numeric(
                    pd.Series([candidate.boards_for_comparison]), errors="coerce").iloc[0]
            daily_max = pd.to_numeric(future.boards_for_comparison, errors="coerce").max()
            top_each_day.append(int(pd.notna(candidate_board) and candidate_board == daily_max))
        labels.append({
            "code": row.code, "signal_date": row.signal_date,
            "future_sessions_available": len(next_dates),
            "top_next_session": top_each_day[0] if top_each_day else np.nan,
            "top_within_3_sessions": max(top_each_day) if len(next_dates) == horizon else np.nan,
            "new_challenger_top_within_3": (
                max(top_each_day) if len(next_dates) == horizon
                and float(row.height_gap_to_known_top) > 0 else np.nan),
        })
    return data.merge(pd.DataFrame(labels), on=["signal_date", "code"], how="left",
                      validate="one_to_one")


def _fit(train: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    weights = 1 / train.groupby("signal_date").signal_date.transform("size")
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(), LogisticRegression(C=.25, max_iter=3000, random_state=130),
    )
    model.fit(train[FEATURES], train.top_within_3_sessions,
              logisticregression__sample_weight=weights)
    return model.predict_proba(validation[FEATURES])[:, 1]


def _daily_pick(frame: pd.DataFrame, score: str, ascending: bool = False) -> pd.DataFrame:
    return (frame.sort_values(["signal_date", score, "code"],
                              ascending=[True, ascending, True], na_position="last")
            .drop_duplicates("signal_date", keep="first").reset_index(drop=True))


def _metrics(selected: pd.DataFrame, label: str) -> dict:
    values = pd.to_numeric(selected[label], errors="coerce").dropna()
    n, successes = len(values), int(values.sum())
    z = 1.959963984540054
    if n:
        p = successes / n
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        interval = [center - radius, center + radius]
    else:
        interval = None
    return {"sessions": int(n), "successes": successes,
            "precision": float(values.mean()) if n else None, "wilson95": interval}


def _paired_delta(model_pick: pd.DataFrame, baseline_pick: pd.DataFrame,
                  label: str, bootstrap_samples: int = 5000) -> dict:
    left = model_pick[["signal_date", label]].rename(columns={label: "model"})
    right = baseline_pick[["signal_date", label]].rename(columns={label: "baseline"})
    paired = left.merge(right, on="signal_date", how="inner").dropna()
    differences = paired.model.astype(float).to_numpy() - paired.baseline.astype(float).to_numpy()
    if not len(differences):
        return {"sessions": 0, "precision_delta": None, "bootstrap95": None}
    rng = np.random.default_rng(130)
    samples = rng.choice(differences, size=(bootstrap_samples, len(differences)), replace=True).mean(axis=1)
    return {"sessions": int(len(differences)), "precision_delta": float(differences.mean()),
            "bootstrap95": [float(np.quantile(samples, .025)),
                            float(np.quantile(samples, .975))]}


def walk_forward(labeled: pd.DataFrame, min_train_sessions: int = 100,
                 embargo_sessions: int = 3) -> tuple[pd.DataFrame, dict]:
    eligible = labeled[labeled.top_within_3_sessions.notna()].copy()
    dates = sorted(eligible.signal_date.unique())
    predictions = []
    for index in range(min_train_sessions + embargo_sessions, len(dates)):
        train_dates = dates[:index - embargo_sessions]
        validation_date = dates[index]
        train = eligible[eligible.signal_date.isin(train_dates)]
        validation = eligible[eligible.signal_date.eq(validation_date)].copy()
        if train.top_within_3_sessions.nunique() < 2 or validation.empty:
            continue
        validation["model_score_not_calibrated"] = _fit(train, validation)
        predictions.append(validation)
    if not predictions:
        return pd.DataFrame(), {"status": "INSUFFICIENT_WALK_FORWARD_WINDOWS"}
    scored = pd.concat(predictions, ignore_index=True)
    scored["current_height_score"] = -scored.height_gap_to_known_top
    scored["second_height_score"] = -scored.height_gap_to_known_top
    model_pick = _daily_pick(scored, "model_score_not_calibrated")
    height_pick = _daily_pick(scored, "current_height_score")
    challengers = scored[scored.height_gap_to_known_top.gt(0)].copy()
    model_challenger = _daily_pick(challengers, "model_score_not_calibrated")
    second_height = _daily_pick(challengers, "second_height_score")
    report = {
        "status": "DIAGNOSTIC_ONLY",
        "scored_rows": int(len(scored)), "validation_sessions": int(scored.signal_date.nunique()),
        "model_all_top1": _metrics(model_pick, "top_within_3_sessions"),
        "current_height_baseline": _metrics(height_pick, "top_within_3_sessions"),
        "model_challenger_top1": _metrics(model_challenger, "new_challenger_top_within_3"),
        "second_height_baseline": _metrics(second_height, "new_challenger_top_within_3"),
        "model_vs_current_height_paired": _paired_delta(
            model_pick, height_pick, "top_within_3_sessions"),
        "model_challenger_vs_second_height_paired": _paired_delta(
            model_challenger, second_height, "new_challenger_top_within_3"),
        "embargo_sessions": embargo_sessions,
    }
    selections = pd.concat([
        model_pick.assign(selection_method="MODEL_ALL_TOP1"),
        height_pick.assign(selection_method="CURRENT_HEIGHT_BASELINE"),
        model_challenger.assign(selection_method="MODEL_CHALLENGER_TOP1"),
        second_height.assign(selection_method="SECOND_HEIGHT_BASELINE"),
    ], ignore_index=True)
    return selections, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-train-sessions", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite immutable v130 output")
    frame = pd.read_csv(args.features, dtype={"code": str})
    frame = frame[pd.to_numeric(frame.boards_for_comparison, errors="coerce").ge(1)].copy()
    labeled = build_labels(frame)
    selections, evaluation = walk_forward(labeled, args.min_train_sessions)
    args.output.mkdir(parents=True, exist_ok=False)
    labeled.to_csv(args.output / "height_anchor_labels_LABEL_ONLY.csv", index=False,
                   encoding="utf-8-sig")
    if not selections.empty:
        selections.to_csv(args.output / "walk_forward_selections.csv", index=False,
                          encoding="utf-8-sig")
    report = {
        "version": "v130-height-anchor-transition-1",
        "evaluation": evaluation,
        "target": "candidate becomes or remains known maximum board in next 1-3 sessions",
        "governance": {
            "future_window_embargoed": True, "score_is_probability": False,
            "actual_trade_profitability_target": False, "deployment_authorized": False,
        },
        "limitations": [
            "historical reconstructed pool is not independently proven full-market complete",
            "being a height anchor is a role label, not evidence of buyability or profit",
            "ties at maximum board count as height anchors and are not forced into a single winner",
        ],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
