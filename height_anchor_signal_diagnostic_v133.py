"""Candidate-level OOS diagnostic for the v132 joint buy components.

This version does not tune weights or create a new trading rule.  It records
every prequential candidate score and tests whether each component predicts its
own declared target before any further model iteration is permitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from height_anchor_joint_buy_model_v132 import (
    _fit_probability, joint_score, prepare,
)


TARGETS = {
    "anchor_score_not_calibrated": "top_within_3_sessions",
    "fill_score_not_calibrated": "fill_proxy_target",
    "profit_score_not_calibrated": "b2_positive_target",
    "large_loss_score_not_calibrated": "b2_large_loss_target",
}


def score_walk_forward(rows: pd.DataFrame, min_train_sessions: int = 100,
                       embargo_sessions: int = 3) -> pd.DataFrame:
    eligible = rows[rows.top_within_3_sessions.notna()].copy()
    dates = sorted(eligible.signal_date.unique())
    output = []
    for index in range(min_train_sessions + embargo_sessions, len(dates)):
        train_dates = dates[:index - embargo_sessions]
        train = eligible[eligible.signal_date.isin(train_dates)]
        validation = eligible[eligible.signal_date.eq(dates[index])].copy()
        if validation.empty:
            continue
        validation["anchor_score_not_calibrated"] = _fit_probability(
            train, validation, "top_within_3_sessions", 132)
        validation["fill_score_not_calibrated"] = _fit_probability(
            train, validation, "fill_proxy_target", 133)
        validation["profit_score_not_calibrated"] = _fit_probability(
            train, validation, "b2_positive_target", 134)
        validation["large_loss_score_not_calibrated"] = _fit_probability(
            train, validation, "b2_large_loss_target", 135)
        validation["joint_score_not_calibrated"] = joint_score(
            validation.anchor_score_not_calibrated,
            validation.fill_score_not_calibrated,
            validation.profit_score_not_calibrated,
            validation.large_loss_score_not_calibrated,
        )
        output.append(validation)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def _auc(frame: pd.DataFrame, score: str, target: str) -> dict:
    values = frame[["signal_date", score, target]].dropna().copy()
    y = values[target].astype(int)
    return {
        "rows": int(len(values)),
        "sessions": int(values.signal_date.nunique()),
        "positive_rate": float(y.mean()) if len(y) else None,
        "auc": float(roc_auc_score(y, values[score])) if y.nunique() == 2 else None,
    }


def _quintiles(frame: pd.DataFrame, score: str, target: str) -> list[dict]:
    values = frame[[score, target]].dropna().copy()
    if len(values) < 10:
        return []
    values["bucket"] = pd.qcut(values[score], 5, labels=False, duplicates="drop")
    return [
        {"quintile_low_to_high": int(bucket) + 1, "rows": int(len(part)),
         "mean_score": float(part[score].mean()), "target_rate": float(part[target].mean())}
        for bucket, part in values.groupby("bucket", observed=True)
    ]


def diagnose(scored: pd.DataFrame) -> dict:
    components = {}
    for score, target in TARGETS.items():
        components[score] = {
            "target": target,
            "overall": _auc(scored, score, target),
            "by_year": {
                str(year): _auc(part, score, target)
                for year, part in scored.groupby(pd.to_datetime(scored.signal_date).dt.year)
            },
            "quintiles": _quintiles(scored, score, target),
        }
    filled = scored[scored.execution_class.eq("LIMIT_PRICE_PATH_PROXY")].copy()
    top_methods = {}
    for score in ["anchor_score_not_calibrated", "profit_score_not_calibrated",
                  "joint_score_not_calibrated"]:
        top = (scored.sort_values(["signal_date", score, "code"],
                                  ascending=[True, False, True])
               .drop_duplicates("signal_date", keep="first").reset_index(drop=True))
        top_fill = top[top.execution_class.eq("LIMIT_PRICE_PATH_PROXY")]
        b2 = pd.to_numeric(top_fill.b2_net_close_mark_LABEL_ONLY, errors="coerce").dropna()
        top_methods[score] = {
            "sessions": int(len(top)), "fill_proxies": int(len(top_fill)),
            "b2_known": int(len(b2)), "b2_mean": float(b2.mean()) if len(b2) else None,
            "b2_positive_rate": float(b2.gt(0).mean()) if len(b2) else None,
            "b2_large_loss_rate": float(b2.le(-.07).mean()) if len(b2) else None,
        }
    return {
        "status": "DIAGNOSTIC_ONLY",
        "scored_rows": int(len(scored)),
        "sessions": int(scored.signal_date.nunique()),
        "component_validity": components,
        "daily_top1_comparison": top_methods,
        "filled_candidate_rows": int(len(filled)),
        "rule_change_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-train-sessions", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite immutable v133 output")
    features = pd.read_csv(args.features, dtype={"code": str})
    features = features[pd.to_numeric(features.boards_for_comparison, errors="coerce").ge(1)].copy()
    outcomes = pd.read_csv(args.outcomes, dtype={"code": str})
    rows = prepare(features, outcomes)
    scored = score_walk_forward(rows, args.min_train_sessions)
    evaluation = diagnose(scored)
    artifact = {
        "version": "v133-height-anchor-signal-diagnostic-1",
        "evaluation": evaluation,
        "governance": {
            "weights_reoptimized_on_results": False,
            "time_embargo_sessions": 3,
            "future_columns_in_features": False,
            "deployment_authorized": False,
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    keep = list(dict.fromkeys(
        ["signal_date", "code", "name", "boards", "board_group", "market_emotion_state",
         "execution_class", "top_within_3_sessions", "fill_proxy_target",
         "b2_positive_target", "b2_large_loss_target", "b1_net_close_mark_LABEL_ONLY",
         "b2_net_close_mark_LABEL_ONLY", "b3_net_close_mark_LABEL_ONLY",
         *TARGETS.keys(), "joint_score_not_calibrated"]
    ))
    scored[[column for column in keep if column in scored]].to_csv(
        args.output / "candidate_oos_scores_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
    (args.output / "report.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
