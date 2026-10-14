"""Prequential joint research model for height role, buyability and B+2 value.

This model is deliberately not a production signal.  It uses only D-close
features to rank the full reconstructed main-board limit-up pool, then waits
for a next-session non-one-price limit touch before a price-path entry exists.
No runner-up is substituted when the daily top pick is unfilled.

The online threshold is chosen only from matured prior out-of-sample top picks.
If the prior history cannot simultaneously show >=70% B+2 positive marks and
<=10% B+2 large losses with enough fills, the gate remains closed.
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

from height_anchor_transition_v130 import FEATURES, build_labels


def _truthy(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False})


def prepare(features: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    labeled = build_labels(features)
    labels = outcomes.copy()
    for frame in (labeled, labels):
        frame["code"] = frame.code.astype(str).str.zfill(6)
        frame["signal_date"] = pd.to_datetime(frame.signal_date).dt.strftime("%Y-%m-%d")
    required = {
        "code", "signal_date", "price_path_state", "entry_one_price_LABEL_ONLY",
        "b1_net_close_mark_LABEL_ONLY", "b2_net_close_mark_LABEL_ONLY",
        "b3_net_close_mark_LABEL_ONLY",
    }
    if required - set(labels):
        raise ValueError("missing outcome columns: " + str(sorted(required - set(labels))))
    if labels.duplicated(["code", "signal_date"]).any():
        raise ValueError("duplicate outcome key")
    rows = labeled.merge(labels, on=["code", "signal_date"], how="left",
                         validate="one_to_one", indicator="outcome_join")
    one_price = _truthy(rows.entry_one_price_LABEL_ONLY)
    valid_path = rows.price_path_state.eq("PRICE_OPPORTUNITY_PROXY_ONLY")
    b1 = pd.to_numeric(rows.b1_net_close_mark_LABEL_ONLY, errors="coerce")
    b2 = pd.to_numeric(rows.b2_net_close_mark_LABEL_ONLY, errors="coerce")
    rows["execution_class"] = "OUTCOME_UNAVAILABLE"
    rows.loc[valid_path & one_price.eq(True), "execution_class"] = "QUEUE_UNVERIFIABLE"
    rows.loc[valid_path & one_price.eq(False) & b1.isna(),
             "execution_class"] = "NO_NEXT_DAY_BOARD_TOUCH"
    rows.loc[valid_path & one_price.eq(False) & b1.notna(),
             "execution_class"] = "LIMIT_PRICE_PATH_PROXY"
    rows["fill_proxy_target"] = np.where(valid_path & one_price.notna(),
                                         (one_price.eq(False) & b1.notna()).astype(float), np.nan)
    filled_b2 = rows.execution_class.eq("LIMIT_PRICE_PATH_PROXY") & b2.notna()
    rows["b2_positive_target"] = np.where(filled_b2, b2.gt(0).astype(float), np.nan)
    rows["b2_large_loss_target"] = np.where(filled_b2, b2.le(-.07).astype(float), np.nan)
    for column in FEATURES:
        rows[column] = pd.to_numeric(rows[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan)
    return rows


def _fit_probability(train: pd.DataFrame, validation: pd.DataFrame, target: str,
                     seed: int) -> np.ndarray:
    known = train[target].notna()
    fitting = train.loc[known].copy()
    if fitting.empty or fitting[target].nunique() < 2:
        return np.full(len(validation), np.nan)
    weights = 1 / fitting.groupby("signal_date").signal_date.transform("size")
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(), LogisticRegression(C=.25, max_iter=3000, random_state=seed),
    )
    model.fit(fitting[FEATURES], fitting[target],
              logisticregression__sample_weight=weights)
    return model.predict_proba(validation[FEATURES])[:, 1]


def joint_score(anchor: pd.Series, fill: pd.Series, profit: pd.Series,
                risk: pd.Series) -> pd.Series:
    """Fixed conjunctive score; zero/unknown in any gate prevents optimism."""
    values = pd.concat([anchor, fill, profit, 1 - risk], axis=1)
    values = values.clip(lower=1e-9, upper=1).fillna(1e-9)
    return np.exp(np.log(values).mean(axis=1))


def daily_pick(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return (frame.sort_values(["signal_date", "joint_score_not_calibrated", "code"],
                              ascending=[True, False, True])
            .drop_duplicates("signal_date", keep="first").reset_index(drop=True))


def choose_threshold(history: pd.DataFrame, min_fills: int = 30,
                     target_positive: float = .70, max_large_loss: float = .10) -> dict:
    """Choose the lowest qualifying score using matured prior OOS picks only."""
    filled = history[
        history.execution_class.eq("LIMIT_PRICE_PATH_PROXY")
        & history.b2_positive_target.notna()
        & history.b2_large_loss_target.notna()
        & history.joint_score_not_calibrated.notna()
    ].copy()
    if len(filled) < min_fills:
        return {"status": "INSUFFICIENT_PRIOR_FILLS", "threshold": None,
                "prior_fills": int(len(filled))}
    scores = np.unique(filled.joint_score_not_calibrated.to_numpy())
    qualifiers = []
    for threshold in np.sort(scores):
        accepted_signals = history.joint_score_not_calibrated.ge(threshold)
        sample = filled[filled.joint_score_not_calibrated.ge(threshold)]
        if len(sample) < min_fills:
            continue
        positive = float(sample.b2_positive_target.mean())
        large_loss = float(sample.b2_large_loss_target.mean())
        if positive >= target_positive and large_loss <= max_large_loss:
            qualifiers.append({
                "threshold": float(threshold), "prior_fills": int(len(sample)),
                "prior_signals": int(accepted_signals.sum()),
                "prior_positive_rate": positive, "prior_large_loss_rate": large_loss,
            })
    if not qualifiers:
        return {"status": "NO_THRESHOLD_MEETS_CHALLENGE", "threshold": None,
                "prior_fills": int(len(filled))}
    chosen = sorted(qualifiers, key=lambda item: (item["threshold"], -item["prior_fills"]))[0]
    return {"status": "PRIOR_OOS_THRESHOLD_AVAILABLE", **chosen}


def walk_forward(rows: pd.DataFrame, min_train_sessions: int = 100,
                 embargo_sessions: int = 3, min_threshold_fills: int = 30) -> pd.DataFrame:
    eligible = rows[rows.top_within_3_sessions.notna()].copy()
    dates = sorted(eligible.signal_date.unique())
    date_position = {date: index for index, date in enumerate(dates)}
    top_history: list[pd.DataFrame] = []
    output: list[pd.DataFrame] = []
    for index in range(min_train_sessions + embargo_sessions, len(dates)):
        validation_date = dates[index]
        train_dates = dates[:index - embargo_sessions]
        train = eligible[eligible.signal_date.isin(train_dates)]
        validation = eligible[eligible.signal_date.eq(validation_date)].copy()
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
        picked = daily_pick(validation)
        matured = [part for part in top_history
                   if date_position[str(part.iloc[0].signal_date)] <= index - embargo_sessions - 1]
        threshold_history = (pd.concat(matured, ignore_index=True)
                             if matured else pd.DataFrame(columns=picked.columns))
        gate = choose_threshold(threshold_history, min_fills=min_threshold_fills)
        picked["threshold_status"] = gate["status"]
        picked["online_threshold"] = gate.get("threshold")
        picked["prior_threshold_fills"] = gate.get("prior_fills", 0)
        picked["prior_threshold_positive_rate"] = gate.get("prior_positive_rate")
        picked["prior_threshold_large_loss_rate"] = gate.get("prior_large_loss_rate")
        picked["gate_pass"] = bool(
            gate.get("threshold") is not None
            and float(picked.iloc[0].joint_score_not_calibrated) >= float(gate["threshold"]))
        output.append(picked)
        top_history.append(picked.copy())
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def _wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(center - radius), float(center + radius)]


def path_metrics(frame: pd.DataFrame) -> dict:
    filled = frame.execution_class.eq("LIMIT_PRICE_PATH_PROXY")
    result: dict[str, object] = {
        "signals": int(len(frame)),
        "weeks": int(pd.to_datetime(frame.signal_date).dt.to_period("W").nunique()) if len(frame) else 0,
        "signals_per_week": (float(len(frame) / pd.to_datetime(frame.signal_date).dt.to_period("W").nunique())
                             if len(frame) else None),
        "fill_proxies": int(filled.sum()),
        "queue_unverifiable": int(frame.execution_class.eq("QUEUE_UNVERIFIABLE").sum()),
        "no_next_day_board_touch": int(frame.execution_class.eq("NO_NEXT_DAY_BOARD_TOUCH").sum()),
        "outcome_unavailable": int(frame.execution_class.eq("OUTCOME_UNAVAILABLE").sum()),
    }
    for horizon in (1, 2, 3):
        values = pd.to_numeric(
            frame.loc[filled, f"b{horizon}_net_close_mark_LABEL_ONLY"], errors="coerce").dropna()
        positives = int(values.gt(0).sum())
        losses = int(values.le(-.07).sum())
        result[f"b{horizon}"] = {
            "known": int(len(values)),
            "mean": float(values.mean()) if len(values) else None,
            "median": float(values.median()) if len(values) else None,
            "positive_rate": float(values.gt(0).mean()) if len(values) else None,
            "positive_wilson95": _wilson(positives, len(values)),
            "large_loss_rate": float(values.le(-.07).mean()) if len(values) else None,
            "large_loss_wilson95": _wilson(losses, len(values)),
        }
    return result


def report(selections: pd.DataFrame) -> dict:
    gated = selections[selections.gate_pass.eq(True)].copy()
    return {
        "status": "DIAGNOSTIC_ONLY",
        "rank_top1_every_session": path_metrics(selections),
        "online_threshold_gate": path_metrics(gated),
        "threshold_status_counts": {
            str(key): int(value) for key, value in selections.threshold_status.value_counts().items()
        },
        "gated_role_success": {
            "known": int(gated.top_within_3_sessions.notna().sum()),
            "rate": (float(pd.to_numeric(gated.top_within_3_sessions, errors="coerce").mean())
                     if len(gated) else None),
        },
        "governance": {
            "future_label_in_features": False,
            "time_embargo_applied": True,
            "threshold_uses_current_or_future_result": False,
            "unfilled_runner_up_substitution": False,
            "actual_fill_verified": False,
            "complete_exit_rule_tested": False,
            "deployment_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-train-sessions", type=int, default=100)
    parser.add_argument("--min-threshold-fills", type=int, default=30)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite immutable v132 output")
    features = pd.read_csv(args.features, dtype={"code": str})
    features = features[pd.to_numeric(features.boards_for_comparison, errors="coerce").ge(1)].copy()
    outcomes = pd.read_csv(args.outcomes, dtype={"code": str})
    rows = prepare(features, outcomes)
    selections = walk_forward(rows, min_train_sessions=args.min_train_sessions,
                              min_threshold_fills=args.min_threshold_fills)
    evaluation = report(selections)
    artifact = {
        "version": "v132-height-anchor-joint-buy-1",
        "evaluation": evaluation,
        "target_definition": {
            "role": "becomes or remains reconstructed maximum board within next 1-3 sessions",
            "execution": "next session touches inferred upper limit and is not one-price",
            "value": "B+2 close net mark relative to hypothetical upper-limit entry is positive",
            "risk": "same B+2 net close mark is <= -7%",
        },
        "limitations": [
            "reconstructed historical pool is not independently proven full-market complete",
            "daily high touch cannot prove an accepted order or fill",
            "online threshold is prequential research, not an untouched final blind test",
            "B+2 close mark is not a frozen complete-trade exit result",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    selections.to_csv(args.output / "prequential_daily_top1.csv", index=False,
                      encoding="utf-8-sig")
    (args.output / "report.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
