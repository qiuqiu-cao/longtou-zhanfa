"""Walk-forward audit of next-session seven-state emotion probabilities.

Every prediction uses transitions observed strictly before the predicted target
session.  The report is diagnostic and does not promote a trading model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


STATES = ["启动", "升温", "加速", "一致", "大分歧", "退潮", "修复"]
ADVERSE = {"大分歧", "退潮"}
SUPPORTIVE = {"升温", "加速", "一致", "修复"}
REGIME_GROUPS = {
    "RISK_ON": ("启动", "升温", "加速", "一致"),
    "RISK_OFF": ("大分歧", "退潮"),
    "REPAIR": ("修复",),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {"n": 0}
    y = rows.target_state.map(STATES.index).to_numpy()
    p = rows[[f"p_{state}" for state in STATES]].to_numpy(float)
    onehot = np.eye(len(STATES))[y]
    predicted = p.argmax(axis=1)
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    log_loss = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))
    confusion = pd.crosstab(
        pd.Categorical(rows.target_state, categories=STATES),
        pd.Categorical([STATES[i] for i in predicted], categories=STATES),
        dropna=False,
    )
    per_state = {}
    for state in STATES:
        mask = rows.target_state.eq(state)
        per_state[state] = {
            "support": int(mask.sum()),
            "recall": float((predicted[mask.to_numpy()] == STATES.index(state)).mean()) if mask.any() else None,
            "mean_assigned_probability": float(rows.loc[mask, f"p_{state}"].mean()) if mask.any() else None,
        }
    adverse_y = rows.target_state.isin(ADVERSE).astype(float).to_numpy()
    adverse_p = rows[[f"p_{s}" for s in ADVERSE]].sum(axis=1).to_numpy(float)
    return {
        "n": int(len(rows)),
        "accuracy": float((predicted == y).mean()),
        "multiclass_brier": brier,
        "log_loss": log_loss,
        "adverse_binary_brier": float(np.mean((adverse_p - adverse_y) ** 2)),
        "adverse_base_rate": float(adverse_y.mean()),
        "per_state": per_state,
        "confusion": {state: {col: int(confusion.loc[state, col]) for col in STATES} for state in STATES},
    }


def _group_metrics(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {"n": 0}
    names = list(REGIME_GROUPS)
    target_map = {state: group for group, states in REGIME_GROUPS.items() for state in states}
    target = rows.target_state.map(target_map)
    probabilities = pd.DataFrame({
        group: rows[[f"p_{state}" for state in states]].sum(axis=1)
        for group, states in REGIME_GROUPS.items()
    })
    predicted = probabilities.idxmax(axis=1)
    onehot = pd.get_dummies(pd.Categorical(target, categories=names)).reindex(columns=names, fill_value=0)
    recalls, f1s, per_group = [], [], {}
    for group in names:
        truth = target.eq(group)
        called = predicted.eq(group)
        tp = int((truth & called).sum())
        recall = tp / int(truth.sum()) if truth.any() else None
        precision = tp / int(called.sum()) if called.any() else (0.0 if truth.any() else None)
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else
              0.0 if truth.any() else None)
        if recall is not None:
            recalls.append(recall)
        if f1 is not None:
            f1s.append(f1)
        per_group[group] = {
            "support": int(truth.sum()), "predicted": int(called.sum()),
            "precision": precision, "recall": recall, "f1": f1,
            "mean_assigned_probability": float(probabilities.loc[truth, group].mean()) if truth.any() else None,
        }
    p = probabilities[names].to_numpy(float)
    y = onehot[names].to_numpy(float)
    actual_probability = np.array([
        probabilities.loc[index, group] for index, group in target.items()
    ])
    return {
        "n": int(len(rows)),
        "accuracy": float(predicted.eq(target).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "multiclass_brier": float(np.mean(np.sum((p - y) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(np.clip(actual_probability, 1e-12, 1)))),
        "per_group": per_group,
    }


def walk_forward(frame: pd.DataFrame, initial_transitions: int = 120, alpha: float = 1.0) -> pd.DataFrame:
    data = frame[["date", "emotion_state"]].copy()
    data["date"] = pd.to_datetime(data.date, errors="coerce")
    data = data.dropna().sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if not set(data.emotion_state).issubset(STATES):
        unknown = sorted(set(data.emotion_state) - set(STATES))
        raise ValueError(f"unknown emotion states: {unknown}")
    counts = pd.DataFrame(0.0, index=STATES, columns=STATES)
    marginal = pd.Series(0.0, index=STATES)
    rows = []
    for i in range(len(data) - 1):
        current = data.loc[i, "emotion_state"]
        target = data.loc[i + 1, "emotion_state"]
        # The transition i-1 -> i is known at the close of session i.
        if i > 0:
            prior = data.loc[i - 1, "emotion_state"]
            counts.loc[prior, current] += 1
            marginal.loc[current] += 1
        if i >= initial_transitions:
            probs = (counts.loc[current] + alpha) / (counts.loc[current].sum() + alpha * len(STATES))
            marginal_probs = (marginal + alpha) / (marginal.sum() + alpha * len(STATES))
            row = {
                "signal_date": data.loc[i, "date"].strftime("%Y-%m-%d"),
                "target_date": data.loc[i + 1, "date"].strftime("%Y-%m-%d"),
                "current_state": current,
                "target_state": target,
            }
            for state in STATES:
                row[f"p_{state}"] = float(probs[state])
                row[f"marginal_p_{state}"] = float(marginal_probs[state])
            rows.append(row)
    return pd.DataFrame(rows)


def latest_prediction(frame: pd.DataFrame, alpha: float = 1.0) -> dict:
    data = frame[["date", "emotion_state"]].copy()
    data["date"] = pd.to_datetime(data.date, errors="coerce")
    data = data.dropna().sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    counts = pd.DataFrame(0.0, index=STATES, columns=STATES)
    for i in range(len(data) - 1):
        counts.loc[data.loc[i, "emotion_state"], data.loc[i + 1, "emotion_state"]] += 1
    current = data.iloc[-1].emotion_state
    probs = (counts.loc[current] + alpha) / (counts.loc[current].sum() + alpha * len(STATES))
    p = {state: float(probs[state]) for state in STATES}
    adverse = sum(p[s] for s in ADVERSE)
    supportive = sum(p[s] for s in SUPPORTIVE)
    if adverse >= .50:
        call = "偏分歧或退潮"
    elif p["一致"] + p["加速"] >= .35:
        call = "偏加速或一致"
    elif p["修复"] >= max(p["大分歧"], p["退潮"]):
        call = "偏修复"
    else:
        call = "方向不清，按分歧预案"
    return {
        "signal_date": data.iloc[-1].date.strftime("%Y-%m-%d"),
        "current_state": current,
        "probabilities": p,
        "group_probabilities": {
            group: sum(p[state] for state in states)
            for group, states in REGIME_GROUPS.items()
        },
        "adverse_probability": adverse,
        "supportive_probability": supportive,
        "direction_call": call,
        "status": "DIAGNOSTIC_ONLY",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/v3/market_emotion.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-transitions", type=int, default=120)
    args = parser.parse_args()
    source, output = Path(args.input), Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable audit directory")
    frame = pd.read_csv(source)
    predictions = walk_forward(frame, args.initial_transitions)
    probability_columns = [f"p_{s}" for s in STATES]
    marginal = predictions.copy()
    marginal[probability_columns] = predictions[[f"marginal_p_{s}" for s in STATES]].to_numpy()
    report = {
        "version": "v82-emotion-multiclass-audit-2",
        "status": "DIAGNOSTIC_ONLY",
        "target": "next_trading_session_emotion_state",
        "states": STATES,
        "method": "expanding_transition_matrix_laplace_alpha_1",
        "time_integrity": "prediction uses only transitions known by signal-date close",
        "initial_transitions": args.initial_transitions,
        "evaluation": {
            "expanding_markov": _metrics(predictions),
            "expanding_marginal_baseline": _metrics(marginal),
            "recent_120_markov": _metrics(predictions.tail(120)),
            "recent_120_marginal_baseline": _metrics(marginal.tail(120)),
            "three_regime_markov": _group_metrics(predictions),
            "three_regime_marginal_baseline": _group_metrics(marginal),
            "recent_120_three_regime_markov": _group_metrics(predictions.tail(120)),
            "recent_120_three_regime_marginal_baseline": _group_metrics(marginal.tail(120)),
        },
        "latest_prediction": latest_prediction(frame),
        "limitations": [
            "emotion-state labels are deterministic summaries of daily market data, not ground truth psychology",
            "the current session has no realised next-state label yet",
            "probabilities guide exposure; they do not identify a buyable stock or guarantee a premium",
        ],
        "input": {"path": str(source.resolve()), "sha256": _sha256(source)},
    }
    output.mkdir(parents=True)
    csv_path = output / "walk_forward_predictions.csv"
    predictions.to_csv(csv_path, index=False, encoding="utf-8-sig")
    report["predictions"] = {"path": str(csv_path.resolve()), "sha256": _sha256(csv_path)}
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
