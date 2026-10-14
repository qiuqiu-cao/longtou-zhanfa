"""Regime-gated relay ranking candidate built from the V257 walk-forward output.

The gate is intentionally simple and domain-preregisterable: when the
signal-date market is in ``state_divergence``, do not trust the price/volume
model's top candidate; keep the neutral height/code baseline for identity
observation.  Outside divergence, use the V257 full model.  This is a
post-hoc diagnostic candidate and must be tested on a future untouched period
before production use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from relay_promotion_holdout_v256 import _paired, _proxy_delta, _summarize, _top


VERSION = "v258-regime-gated-relay-ranking-1"
MODEL_SCORE = "score_identity_emotion_price_volume"
REQUIRED = {
    "date", "fold", "boards", "code", "one_word_proxy", "promoted_next_day",
    "promotion_label_status", "evaluation_date_complete", "state_divergence",
    MODEL_SCORE, "execution_status", "net_return", "large_loss",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_date_state(rows: pd.DataFrame) -> None:
    states = rows.groupby("date").state_divergence.nunique(dropna=False)
    if (states > 1).any():
        raise ValueError("state_divergence is not constant within a signal date")


def _select(pool: pd.DataFrame) -> dict[str, pd.DataFrame]:
    baseline = _top(pool)
    model = _top(pool, MODEL_SCORE)
    # State is a signal-date field, not a next-session label.  In divergence,
    # retain the neutral baseline for identity observation; otherwise use model.
    model_dates = set(model.loc[model.state_divergence.fillna(0).eq(0), "date"])
    gated = pd.concat([
        model[model.date.isin(model_dates)],
        baseline[~baseline.date.isin(model_dates)],
    ], ignore_index=True)
    if gated.date.duplicated().any():
        raise AssertionError("regime gate selected more than one row per date")
    return {"baseline": baseline, "model": model, "regime_gated": gated}


def _evaluate(pool: pd.DataFrame) -> dict:
    selected = _select(pool)
    baseline = selected["baseline"]
    return {
        "pool_rows": int(len(pool)),
        "pool_dates": int(pool.date.nunique()),
        "divergence_dates": int(pool.loc[pool.state_divergence.fillna(0).eq(1), "date"].nunique()),
        "selections": {name: _summarize(frame) for name, frame in selected.items()},
        "paired_vs_baseline": {
            name: _paired(frame, baseline)
            for name, frame in selected.items() if name != "baseline"
        },
        "proxy_trade_outcome_vs_baseline": {
            name: _proxy_delta(frame, baseline)
            for name, frame in selected.items() if name != "baseline"
        },
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

    evaluated = rows[rows.evaluation_date_complete.eq(True)].copy()
    evaluated = evaluated[evaluated.one_word_proxy.eq(0)].copy()
    fold_reports: dict[str, dict] = {}
    combined: dict[str, list[pd.DataFrame]] = {}
    for fold, frame in evaluated.groupby("fold", sort=True):
        item = {"fold": fold, "pools": {}}
        for scope, predicate in {
            "two_board": frame.boards.eq(2),
            "three_plus": frame.boards.ge(3),
            "all_boards": frame.boards.ge(2),
        }.items():
            key = scope + ":non_one_price"
            pool = frame[predicate].copy()
            item["pools"][key] = _evaluate(pool) if not pool.empty else {"status": "NO_CANDIDATES"}
            if not pool.empty:
                combined.setdefault(key, []).append(pool)
        fold_reports[str(fold)] = item

    combined_reports = {}
    for key, parts in combined.items():
        combined_reports[key] = _evaluate(pd.concat(parts, ignore_index=True))

    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "gate": "state_divergence=1 -> neutral baseline; otherwise V257 full model",
        "fold_reports": fold_reports,
        "combined_reports": combined_reports,
        "interpretation": {
            "is_pristine_blind_test": False,
            "reason": "gate was selected after inspecting V257 historical windows",
            "post_hoc_candidate": True,
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
        "source_hashes": {str(source.resolve()): sha(source)},
    }
    target.mkdir(parents=True)
    for key, parts in combined.items():
        selected = _select(pd.concat(parts, ignore_index=True))
        for name, frame in selected.items():
            frame.to_csv(target / (key.replace(":", "__") + "__" + name + ".csv"),
                         index=False, encoding="utf-8-sig")
    report["artifact_hashes"] = {p.name: sha(p) for p in target.iterdir() if p.is_file()}
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args.predictions, args.output)
    print(json.dumps({
        "version": report["version"],
        "status": report["status"],
        "combined": report["combined_reports"],
    }, ensure_ascii=False))
