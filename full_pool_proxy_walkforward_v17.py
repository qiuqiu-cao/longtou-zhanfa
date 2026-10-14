"""Fixed-window walk-forward audit for the full-pool opportunity proxy.

This is deliberately separate from the single exploratory validation script.
Each validation window is scored only from observations whose signal and target
information precede that window.  It still cannot establish fills or complete
trade profitability; the output remains ``EXPLORATORY_ONLY``.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from full_pool_proxy_backtest import (
    FULL, GROUPS, ROLE_NORMALIZED, _summary_core, candidate_summary,
    prepare, role_normalized_scores, select_top1, summarize,
)


DEFAULT_WINDOWS = [
    ("2025-01-02", "2025-04-01"),
    ("2025-04-01", "2025-07-01"),
    ("2025-07-01", "2025-10-01"),
    ("2025-10-01", "2026-01-01"),
    ("2026-01-01", "2026-04-01"),
    ("2026-04-01", "2026-07-01"),
    ("2026-07-01", "2026-09-01"),
]


def _detail_columns(validation):
    columns = [
        "code", "signal_date", "opportunity_proxy_label",
        "next_session_one_price_LABEL_ONLY",
        "next_session_touched_limit_LABEL_ONLY",
        "b2_close_mark_net_return_proxy",
    ]
    path_columns = [
        "b1_net_close_mark_LABEL_ONLY", "b2_net_close_mark_LABEL_ONLY",
        "b3_net_close_mark_LABEL_ONLY", "b1_big_loss_mark_LABEL_ONLY",
        "b2_big_loss_mark_LABEL_ONLY", "b3_big_loss_mark_LABEL_ONLY",
    ]
    return columns + [column for column in path_columns if column in validation]


def _fit_probability(train, validation, columns):
    weights = 1 / train.groupby("signal_date").signal_date.transform("size")
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(), LogisticRegression(C=1, max_iter=3000, random_state=161),
    )
    model.fit(train[columns], train.opportunity_proxy_label,
              logisticregression__sample_weight=weights)
    return model.predict_proba(validation[columns])[:, 1]


def _window(rows, start, end):
    validation = rows[(rows.signal_date >= start) & (rows.signal_date < end)].copy()
    train = rows[(rows.signal_date < start)
                 & rows.exit_date_LABEL_ONLY.notna()
                 & (rows.exit_date_LABEL_ONLY < start)].copy()
    train = train[~train.episode_id.isin(validation.episode_id)].copy()
    if train.empty or validation.empty or train.opportunity_proxy_label.nunique() != 2:
        return None, None, "INSUFFICIENT_TRAIN_OR_VALIDATION_DATA"
    if set(validation.board_group) != {"FIRST_BOARD", "CONSECUTIVE_BOARD"}:
        return None, None, "VALIDATION_MISSING_BOARD_GROUP"
    return train, validation, None


def _evaluate_model(predictions, validation, score, probability_score):
    detail = _detail_columns(validation)
    top = select_top1(predictions, score)
    evaluated = top.merge(validation[detail], on=["code", "signal_date"], validate="one_to_one")
    by_group = {}
    group_selected_rows = {}
    for group, part in predictions.groupby("board_group", sort=True):
        group_top = select_top1(part, score)
        group_eval = group_top.merge(validation[detail], on=["code", "signal_date"], validate="one_to_one")
        by_group[str(group)] = summarize(group_eval, score, probability_score=probability_score)
        group_selected_rows[str(group)] = group_eval
    all_eval = predictions[["code", "signal_date", "boards", "board_group", score]].merge(
        validation[["code", "signal_date", "opportunity_proxy_label"]],
        on=["code", "signal_date"], validate="one_to_one")
    return {
        "all_candidates": candidate_summary(all_eval, score, probability_score=probability_score),
        "daily_top1": summarize(evaluated, score, probability_score=probability_score),
        "daily_top1_by_board_group": by_group,
        "selected_rows": evaluated,
        "group_selected_rows": group_selected_rows,
    }


def _paired_proxy_delta(model_rows, baseline_rows, repetitions=5000, seed=152):
    paired = model_rows[["signal_date", "opportunity_proxy_label"]].merge(
        baseline_rows[["signal_date", "opportunity_proxy_label"]],
        on="signal_date", suffixes=("_model", "_baseline"), validate="one_to_one",
    )
    if paired.empty:
        return {"paired_sessions": 0, "difference": None, "bootstrap95": [None, None]}
    difference = (
        paired.opportunity_proxy_label_model.astype(float)
        - paired.opportunity_proxy_label_baseline.astype(float)
    ).to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.asarray([
        rng.choice(difference, len(difference), replace=True).mean()
        for _ in range(repetitions)
    ])
    return {
        "paired_sessions": int(len(paired)),
        "model_successes": int(paired.opportunity_proxy_label_model.sum()),
        "baseline_successes": int(paired.opportunity_proxy_label_baseline.sum()),
        "difference": float(difference.mean()),
        "bootstrap95": [float(value) for value in np.quantile(boot, [.025, .975])],
    }


def walk_forward(features, outcomes, windows=None):
    rows = prepare(features, outcomes)
    windows = DEFAULT_WINDOWS if windows is None else windows
    specs = list(GROUPS.items()) + [(ROLE_NORMALIZED, FULL)]
    fold_reports = []
    selected_by_model = {name: [] for name, _ in specs}
    all_model_names = [name for name, _ in specs] + ["highest_board_baseline"]
    selected_by_model_group = {
        name: {group: [] for group in ("FIRST_BOARD", "CONSECUTIVE_BOARD")}
        for name in all_model_names
    }
    for start, end in windows:
        train, validation, error = _window(rows, start, end)
        fold = {"requested_start": start, "requested_end_exclusive": end}
        if error:
            fold.update({"status": error, "training_rows": 0, "validation_rows": 0})
            fold_reports.append(fold)
            continue
        fold.update({"status": "OK", "training_rows": int(len(train)),
                     "training_sessions": int(train.signal_date.nunique()),
                     "validation_rows": int(len(validation)),
                     "validation_sessions": int(validation.signal_date.nunique())})
        predictions = validation[["code", "signal_date", "episode_id", "boards", "board_group"]].copy()
        for name, columns in specs:
            probability_score = name != ROLE_NORMALIZED
            score = "score_" + name + "_NOT_LIVE"
            if name == ROLE_NORMALIZED:
                predictions[score] = role_normalized_scores(train, validation, columns)
            else:
                predictions[score] = _fit_probability(train, validation, columns)
            evaluated = _evaluate_model(predictions, validation, score, probability_score)
            selected_by_model[name].append(evaluated.pop("selected_rows"))
            group_rows = evaluated.pop("group_selected_rows")
            for group, selected_group_rows in group_rows.items():
                selected_by_model_group[name][group].append(selected_group_rows)
            fold.setdefault("models", {})[name] = evaluated
        baseline = validation[["code", "signal_date", "episode_id", "boards", "board_group"]].copy()
        baseline["highest_board_score_NOT_LIVE"] = baseline["boards"].astype(float)
        baseline_eval = _evaluate_model(baseline, validation, "highest_board_score_NOT_LIVE", False)
        baseline_eval.pop("selected_rows")
        baseline_group_rows = baseline_eval.pop("group_selected_rows")
        for group, selected_group_rows in baseline_group_rows.items():
            selected_by_model_group["highest_board_baseline"][group].append(selected_group_rows)
        fold["models"]["highest_board_baseline"] = baseline_eval
        fold_reports.append(fold)

    aggregate = {}
    for name, selected in selected_by_model.items():
        if not selected:
            continue
        frame = pd.concat(selected, ignore_index=True)
        score = "score_" + name + "_NOT_LIVE"
        aggregate[name] = {
            "folds_with_selection": len(selected),
            "selected_sessions": int(len(frame)),
            "summary": summarize(frame, score, probability_score=name != ROLE_NORMALIZED),
        }
    aggregate_groups = {}
    baseline_groups = {
        group: pd.concat(frames, ignore_index=True)
        for group, frames in selected_by_model_group["highest_board_baseline"].items()
        if frames
    }
    for name, groups in selected_by_model_group.items():
        aggregate_groups[name] = {}
        for group, frames in groups.items():
            if not frames:
                continue
            frame = pd.concat(frames, ignore_index=True)
            score = ("highest_board_score_NOT_LIVE" if name == "highest_board_baseline"
                     else "score_" + name + "_NOT_LIVE")
            item = summarize(frame, score, probability_score=(
                name not in {ROLE_NORMALIZED, "highest_board_baseline"}
            ))
            if name != "highest_board_baseline" and group in baseline_groups:
                item["paired_vs_group_baseline"] = _paired_proxy_delta(
                    frame, baseline_groups[group], seed=152 + len(name) + len(group)
                )
            aggregate_groups[name][group] = item
    return {
        "status": "EXPLORATORY_ONLY",
        "validation_protocol": "fixed chronological walk-forward; no validation retuning",
        "target_coverage": dict(rows.attrs),
        "windows": fold_reports,
        "aggregate_selected": aggregate,
        "aggregate_selected_by_board_group": aggregate_groups,
        "group_baseline_definition": {
            "FIRST_BOARD": "all boards equal; deterministic boards-then-code tie break",
            "CONSECUTIVE_BOARD": "highest board then code",
        },
        "deployment_authorized": False,
        "blind_complete_trade_count": 0,
        "limitations": [
            "daily OHLC proxy cannot prove queue position, fills, sellability or complete-trade profit",
            "historical candidate universe remains incomplete for survivorship purposes",
            "theme, auction, Level-2 and broker execution evidence are absent",
            "windows are exploratory and not the final frozen blind test",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite immutable walk-forward output")
    features = pd.read_csv(args.features, dtype={"code": str})
    outcomes = pd.read_csv(args.outcomes, dtype={"code": str})
    report = walk_forward(features, outcomes)
    report["source_hashes"] = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.features, args.outcomes, Path(__file__))
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "folds": len(report["windows"]),
                      "aggregate_models": list(report["aggregate_selected"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
