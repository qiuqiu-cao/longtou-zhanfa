"""Emotion overlay for V257 ranking without arbitrary code-order substitution.

V258's historical gate replaced divergence-day selections with the first code
in a pool. That can improve a backtest number but is not a leader rule. V260
keeps the diagnostic model's top identity candidate on every date, and changes
only its status: non-divergence is a primary relay-identity candidate;
divergence is an emotion-observation candidate with no trading permission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from relay_promotion_holdout_v256 import _paired, _proxy_delta, _summarize, _top


VERSION = "v260-emotion-overlay-no-substitution-1"
MODEL_SCORE = "score_identity_emotion_price_volume"
REQUIRED = {
    "date", "fold", "boards", "code", "one_word_proxy", "promoted_next_day",
    "promotion_label_status", "evaluation_date_complete", "state_divergence",
    MODEL_SCORE, "execution_status", "net_return", "large_loss",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_date_state(rows: pd.DataFrame) -> None:
    counts = rows.groupby("date").state_divergence.nunique(dropna=False)
    if (counts > 1).any():
        raise ValueError("state_divergence is not constant within a signal date")


def _selection(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = _top(pool)
    model = _top(pool, MODEL_SCORE)
    if model.date.duplicated().any():
        raise AssertionError("model selected more than one row per date")
    status = np.where(
        model.state_divergence.eq(1),
        "EMOTION_OBSERVATION_ONLY",
        np.where(model.state_divergence.eq(0), "PRIMARY_RELAY_IDENTITY", "UNKNOWN_STATE"),
    )
    model = model.assign(identity_status=status,
                         trade_authorization="CASH",
                         trade_authorization_reason="theme_intraday_actual_fill_evidence_missing")
    return baseline, model


def _conditional(model: pd.DataFrame, baseline: pd.DataFrame, mask: pd.Series) -> dict:
    m = model[mask.loc[model.index]]
    b = baseline[mask.loc[baseline.index]]
    if m.empty or b.empty:
        return {"status": "NO_DATES"}
    return {
        "model": _summarize(m, MODEL_SCORE),
        "baseline_comparator": _summarize(b),
        "paired": _paired(m, b),
        "proxy_trade_outcome": _proxy_delta(m, b),
    }


def _evaluate(pool: pd.DataFrame) -> dict:
    if pool.empty:
        return {"status": "NO_CANDIDATES"}
    baseline, model = _selection(pool)
    non_div = pool.state_divergence.fillna(0).eq(0)
    div = pool.state_divergence.fillna(0).eq(1)
    # The top row has the date-level state; use date membership, not row index.
    non_div_dates = set(pool.loc[non_div, "date"])
    div_dates = set(pool.loc[div, "date"])
    model_non = model.date.isin(non_div_dates)
    model_div = model.date.isin(div_dates)
    base_non = baseline.date.isin(non_div_dates)
    base_div = baseline.date.isin(div_dates)
    return {
        "pool_rows": int(len(pool)),
        "pool_dates": int(pool.date.nunique()),
        "overall_model": _summarize(model, MODEL_SCORE),
        "conditional_non_divergence": {
            "dates": int(model_non.sum()),
            "model": _summarize(model[model_non], MODEL_SCORE),
            "baseline_comparator": _summarize(baseline[base_non]),
            "paired": _paired(model[model_non], baseline[base_non]),
            "proxy_trade_outcome": _proxy_delta(model[model_non], baseline[base_non]),
        },
        "conditional_divergence": {
            "dates": int(model_div.sum()),
            "model": _summarize(model[model_div], MODEL_SCORE),
            "baseline_comparator": _summarize(baseline[base_div]),
            "paired": _paired(model[model_div], baseline[base_div]),
            "proxy_trade_outcome": _proxy_delta(model[model_div], baseline[base_div]),
        },
        "selection_status_counts": model.identity_status.value_counts(dropna=False).to_dict(),
        "substitution_performed": False,
    }


def run(predictions_path: str, output_dir: str) -> dict:
    source = Path(predictions_path)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("immutable output directory already exists")
    rows = pd.read_csv(source, dtype={"code": str}, low_memory=False)
    missing = REQUIRED - set(rows)
    if missing:
        raise ValueError(f"missing prediction fields: {sorted(missing)}")
    _check_date_state(rows)
    rows = rows[rows.evaluation_date_complete.eq(True) & rows.one_word_proxy.eq(0)].copy()

    fold_reports: dict[str, dict] = {}
    combined: dict[str, list[pd.DataFrame]] = {}
    selections: dict[str, pd.DataFrame] = {}
    for fold, frame in rows.groupby("fold", sort=True):
        item = {"fold": fold, "pools": {}}
        for scope, predicate in {
            "two_board": frame.boards.eq(2),
            "three_plus": frame.boards.ge(3),
            "all_boards": frame.boards.ge(2),
        }.items():
            key = scope + ":non_one_price"
            pool = frame[predicate].copy()
            item["pools"][key] = _evaluate(pool)
            if not pool.empty:
                combined.setdefault(key, []).append(pool)
                _, selected = _selection(pool)
                selections[f"{fold}__{key}"] = selected
        fold_reports[str(fold)] = item

    combined_reports = {key: _evaluate(pd.concat(parts, ignore_index=True))
                        for key, parts in combined.items()}
    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "supersedes": "v258-regime-gated-relay-ranking-1",
        "rule": "keep model top; non-divergence=PRIMARY_RELAY_IDENTITY; divergence=EMOTION_OBSERVATION_ONLY",
        "fold_reports": fold_reports,
        "combined_reports": combined_reports,
        "interpretation": {
            "is_pristine_blind_test": False,
            "reason": "historical V257 windows were already inspected",
            "post_hoc_candidate": True,
            "theme_and_intraday_missing": True,
            "actual_fill_count": 0,
            "promotion_is_eventual_leader": False,
            "entry_authorized": False,
            "code_order_substitution": False,
            "decision_contract": {
                "identity_channel": "LEADER_IDENTITY_DIAGNOSTIC",
                "divergence_channel": "EMOTION_OBSERVATION_ONLY",
                "trade_authorization": "CASH",
                "reason": "theme_intraday_actual_fill_evidence_missing",
            },
        },
        "source_hashes": {str(source.resolve()): sha(source), str(Path(__file__).resolve()): sha(Path(__file__))},
    }
    target.mkdir(parents=True)
    for name, selected in selections.items():
        selected.to_csv(target / (name + "__identity_candidates.csv"), index=False, encoding="utf-8-sig")
    report["artifact_hashes"] = {p.name: sha(p) for p in target.iterdir() if p.is_file()}
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.predictions, args.output)
    print(json.dumps({"version": result["version"], "status": result["status"],
                      "combined": result["combined_reports"]}, ensure_ascii=False))
