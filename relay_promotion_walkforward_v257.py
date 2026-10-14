"""Multi-window walk-forward stability audit for the relay-promotion model.

This audit is deliberately separate from the V256 2026 diagnostic.  It fixes
three chronological folds in advance, purges labels whose next session is
after the training freeze, evaluates only complete-label dates, and reports
the same promotion target for two-board and three-plus-board pools.  It is an
identity/promotion audit, not a fill or live-trading backtest: the source has
no verified theme, auction, intraday, Level-2, or actual-fill evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from relay_promotion_holdout_v256 import (
    FORBIDDEN_MODEL_FIELDS,
    SPECS,
    _fit,
    _paired,
    _proxy_delta,
    _same_height_score,
    _summarize,
    _top,
    load_rows,
)


VERSION = "v257-multi-window-walkforward-1"
FOLDS = (
    {"name": "2024H2", "train_end_exclusive": "2024-07-01",
     "test_start": "2024-07-01", "test_end_inclusive": "2024-12-31"},
    {"name": "2025", "train_end_exclusive": "2025-01-01",
     "test_start": "2025-01-01", "test_end_inclusive": "2025-12-31"},
    {"name": "2026YTD", "train_end_exclusive": "2026-01-01",
     "test_start": "2026-01-05", "test_end_inclusive": "2026-08-31"},
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_fold(rows: pd.DataFrame, fold: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return purged training rows, all test rows, and complete test dates."""
    cutoff = pd.Timestamp(fold["train_end_exclusive"])
    start = pd.Timestamp(fold["test_start"])
    end = pd.Timestamp(fold["test_end_inclusive"])
    dates = pd.to_datetime(rows.date, errors="coerce")
    next_date = pd.to_datetime(rows.trade_date, errors="coerce")
    valid = rows.promoted_next_day.isin([0, 1]) & rows.promotion_label_status.eq(
        "verified_from_next_daily_limit_close"
    )
    train = rows[(dates < cutoff) & valid & (next_date < cutoff) & (next_date > dates)].copy()
    test_all = rows[(dates >= start) & (dates <= end)].copy()
    complete = valid.loc[test_all.index].groupby(test_all.date).all()
    return train, test_all, complete


def _eval_pool(pool: pd.DataFrame) -> dict:
    """Evaluate all frozen model specifications on one fold/pool."""
    if pool.empty:
        return {"status": "NO_CANDIDATES"}
    baseline = _top(pool)
    random_baseline = _random_top(pool)
    pool = pool.copy()
    pool["score_same_height_identity"] = _same_height_score(pool)
    identity_baseline = _top(pool, "score_same_height_identity")
    result = {
        "pool_rows": int(len(pool)),
        "pool_dates": int(pool.date.nunique()),
        "baseline": _summarize(baseline),
        "random_same_pool_baseline": _summarize(random_baseline),
        "same_height_identity_baseline": _summarize(identity_baseline),
        "same_height_warning": "shared fields within height; not an independent strong comparator",
        "models": {},
    }
    for name in SPECS:
        chosen = _top(pool, "score_" + name)
        result["models"][name] = {
            "summary": _summarize(chosen, "score_" + name),
            "paired_vs_same_pool_height": _paired(chosen, baseline),
            "paired_vs_random_same_pool": _paired(chosen, random_baseline),
            "paired_vs_same_height_identity": _paired(chosen, identity_baseline),
            "proxy_trade_outcome_vs_same_pool_height": _proxy_delta(chosen, baseline),
        }
    return result


def _random_top(rows: pd.DataFrame) -> pd.DataFrame:
    """Stable null control: one deterministic pseudo-random candidate per date."""
    keys = rows.date.astype(str) + "|" + rows.code.astype(str)
    rank = keys.map(lambda x: int(hashlib.sha256(x.encode("utf-8")).hexdigest()[:16], 16))
    return rows.assign(_random_rank=rank).sort_values(
        ["date", "_random_rank"], ascending=[True, True], kind="mergesort"
    ).drop_duplicates("date", keep="first").drop(columns="_random_rank").copy()


def run(data_path: str, output_dir: str) -> dict:
    source = Path(data_path)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("immutable output directory already exists")
    if set(sum(SPECS.values(), [])) & FORBIDDEN_MODEL_FIELDS:
        raise AssertionError("future or execution field included as model input")

    rows, audit = load_rows(source)
    fold_reports: dict[str, dict] = {}
    combined: dict[str, dict[str, list[pd.DataFrame]]] = {}
    all_predictions: list[pd.DataFrame] = []

    for fold in FOLDS:
        train, test_all, complete = split_fold(rows, fold)
        complete_dates = set(complete[complete].index)
        test = test_all[test_all.date.isin(complete_dates)].copy()
        if train.empty or test.empty:
            raise ValueError(f"empty training or complete test rows in {fold['name']}")
        scored = test_all.copy()
        for name, fields in SPECS.items():
            scored["score_" + name] = _fit(train, scored, fields)
        scored["evaluation_date_complete"] = scored.date.isin(complete_dates)
        scored["fold"] = fold["name"]
        scored["identity_channel"] = "LEADER_IDENTITY_DIAGNOSTIC"
        scored["trade_authorization"] = "CASH"
        scored["trade_authorization_reason"] = "theme_intraday_actual_fill_evidence_missing"
        all_predictions.append(scored)
        eval_scored = scored[scored.evaluation_date_complete].copy()

        fold_item = {
            "fold": fold,
            "train_rows": int(len(train)),
            "train_dates": int(train.date.nunique()),
            "max_training_label_date": str(train.trade_date.max()),
            "test_rows_all_candidates": int(len(test_all)),
            "test_rows_complete_dates": int(len(test)),
            "test_dates_complete": int(test.date.nunique()),
            "test_days_dropped_for_incomplete_labels": int((~complete).sum()),
            "pools": {},
        }
        for scope_name, predicate in {
            "two_board": lambda x: x.boards.eq(2),
            "three_plus": lambda x: x.boards.ge(3),
            "all_boards": lambda x: x.boards.ge(2),
        }.items():
            pool = eval_scored[predicate(eval_scored) & eval_scored.one_word_proxy.eq(0)].copy()
            key = scope_name + ":non_one_price"
            fold_item["pools"][key] = _eval_pool(pool)
            if not pool.empty:
                combined.setdefault(key, {})
                for model_name in ["baseline", "same_height_identity_baseline", *SPECS.keys()]:
                    if model_name in ("baseline", "same_height_identity_baseline"):
                        score = None if model_name == "baseline" else "score_same_height_identity"
                        selected = _top(pool if score is None else pool.assign(
                            score_same_height_identity=_same_height_score(pool)), score)
                    else:
                        selected = _top(pool, "score_" + model_name)
                    combined[key].setdefault(model_name, []).append(selected)
                combined[key].setdefault("random_same_pool_baseline", []).append(_random_top(pool))
        fold_reports[fold["name"]] = fold_item

    combined_reports: dict[str, dict] = {}
    for key, models in combined.items():
        baseline = pd.concat(models["baseline"], ignore_index=True)
        item = {"pool_dates": int(baseline.date.nunique()), "models": {}}
        for name in SPECS:
            chosen = pd.concat(models[name], ignore_index=True)
            random_baseline = pd.concat(models["random_same_pool_baseline"], ignore_index=True)
            item["models"][name] = {
                "summary": _summarize(chosen, "score_" + name),
                "paired_vs_same_pool_height": _paired(chosen, baseline),
                "paired_vs_random_same_pool": _paired(chosen, random_baseline),
                "proxy_trade_outcome_vs_same_pool_height": _proxy_delta(chosen, baseline),
            }
        combined_reports[key] = item

    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "target": "next_session_closing_next_board_promotion",
        "folds": list(FOLDS),
        "fold_reports": fold_reports,
        "combined_reports": combined_reports,
        "source_audit": audit,
        "model_features": SPECS,
        "interpretation": {
            "is_pristine_blind_test": False,
            "reason": "historical archive was already used by earlier project iterations",
            "theme_and_intraday_missing": True,
            "actual_fill_count": 0,
            "promotion_is_eventual_leader": False,
            "entry_authorized": False,
            "decision_contract": {
                "identity_channel": "LEADER_IDENTITY_DIAGNOSTIC",
                "trade_authorization": "CASH",
                "reason": "theme_intraday_actual_fill_evidence_missing",
            },
        },
        "source_hashes": {
            str(source.resolve()): sha(source),
            str(Path(__file__).resolve()): sha(Path(__file__)),
        },
    }
    target.mkdir(parents=True)
    pred = pd.concat(all_predictions, ignore_index=True)
    pred.to_csv(target / "all_fold_predictions_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
    report["artifact_hashes"] = {p.name: sha(p) for p in target.iterdir() if p.is_file()}
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.data, args.output)
    print(json.dumps({
        "version": result["version"],
        "status": result["status"],
        "folds": list(result["fold_reports"]),
        "combined": result["combined_reports"],
    }, ensure_ascii=False))
