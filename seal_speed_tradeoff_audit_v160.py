"""Audit the first-board seal-speed trade-off on prequential dates.

The test distinguishes identity recognition (next-session promotion) from an
executable B+2 profit path.  Fast sealing is not promoted unless it improves
both while not worsening large-loss risk on the same later date block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_rate(values: pd.Series) -> float | None:
    values = values.dropna()
    return float(values.mean()) if len(values) else None


def _group_metrics(part: pd.DataFrame) -> dict:
    missing = part.fill_status.isin(["MISSING_B_BAR", "INVALID_B_PRICE"])
    execution_observed = part[~missing]
    filled = part[part.fill_status.eq("FILLED_OPEN_PROXY")]
    b2 = pd.to_numeric(filled.net_return_B_plus_2, errors="coerce").dropna()
    big = pd.to_numeric(filled.big_loss_B_plus_1, errors="coerce").dropna()
    return {
        "candidate_n": int(len(part)),
        "session_n": int(part.signal_date.nunique()),
        "promotion_rate": _safe_rate(pd.to_numeric(part.promotion_label, errors="coerce")),
        "execution_observed_n": int(len(execution_observed)),
        "missing_execution_evidence_n": int(missing.sum()),
        "unverifiable_limit_queue_n": int(
            execution_observed.fill_status.eq("UNVERIFIABLE_LIMIT_QUEUE").sum()),
        "filled_n": int(len(filled)),
        "fill_rate": (
            float(len(filled) / len(execution_observed))
            if len(execution_observed) else None),
        "b2_known_n": int(len(b2)),
        "b2_mean_net_return": float(b2.mean()) if len(b2) else None,
        "b2_positive_rate": float(b2.gt(0).mean()) if len(b2) else None,
        "b1_big_loss_known_n": int(len(big)),
        "b1_big_loss_rate": float(big.mean()) if len(big) else None,
    }


def _difference(frame: pd.DataFrame, metric: str) -> float | None:
    fast = _group_metrics(frame[frame.fast_seal.eq(1)]).get(metric)
    slow = _group_metrics(frame[frame.fast_seal.eq(0)]).get(metric)
    return float(fast - slow) if fast is not None and slow is not None else None


def _cluster_bootstrap(frame: pd.DataFrame, repetitions=2000, seed=160) -> dict:
    dates = np.array(sorted(frame.signal_date.unique()))
    rng = np.random.default_rng(seed)
    metrics = [
        "promotion_rate", "fill_rate", "b2_mean_net_return",
        "b2_positive_rate", "b1_big_loss_rate",
    ]
    draws = {name: [] for name in metrics}
    groups = {date: frame[frame.signal_date.eq(date)] for date in dates}
    for _ in range(repetitions):
        sampled_dates = rng.choice(dates, size=len(dates), replace=True)
        sampled = pd.concat([groups[date] for date in sampled_dates], ignore_index=True)
        for metric in metrics:
            value = _difference(sampled, metric)
            if value is not None and np.isfinite(value):
                draws[metric].append(value)
    return {
        name: np.quantile(values, [.025, .975]).tolist() if values else None
        for name, values in draws.items()
    }


def audit(features: pd.DataFrame, outcomes: pd.DataFrame,
          prequential: pd.DataFrame, bootstrap_repetitions: int = 2000) -> tuple[pd.DataFrame, dict]:
    required_features = {"signal_date", "code", "fast_seal", "first_seal_minute"}
    required_outcomes = {
        "signal_date", "code", "promotion_label", "fill_status",
        "net_return_B_plus_2", "big_loss_B_plus_1",
    }
    if required_features - set(features) or required_outcomes - set(outcomes):
        raise ValueError("seal-speed feature or execution outcome fields missing")
    if not {"signal_date", "code"} <= set(prequential):
        raise ValueError("prequential keys missing")
    keys = prequential[["signal_date", "code"]].drop_duplicates()
    if len(keys) != len(prequential):
        raise ValueError("prequential candidate keys must be unique")
    feature_view = features[["signal_date", "code", "fast_seal", "first_seal_minute"]]
    outcome_view = outcomes[[
        "signal_date", "code", "promotion_label", "fill_status",
        "net_return_B_plus_2", "big_loss_B_plus_1",
    ]]
    joined = (keys.merge(feature_view, on=["signal_date", "code"], how="left", validate="one_to_one")
              .merge(outcome_view, on=["signal_date", "code"], how="left", validate="one_to_one"))
    if joined.fast_seal.isna().any() or joined.promotion_label.isna().any():
        raise ValueError("prequential rows are not fully covered by feature/outcome evidence")
    joined["fast_seal"] = pd.to_numeric(joined.fast_seal, errors="raise").astype(int)
    if not set(joined.fast_seal.unique()) <= {0, 1}:
        raise ValueError("fast_seal must be binary")
    fast = _group_metrics(joined[joined.fast_seal.eq(1)])
    slow = _group_metrics(joined[joined.fast_seal.eq(0)])
    differences = {
        metric: _difference(joined, metric)
        for metric in (
            "promotion_rate", "fill_rate", "b2_mean_net_return",
            "b2_positive_rate", "b1_big_loss_rate",
        )
    }
    intervals = _cluster_bootstrap(joined, repetitions=bootstrap_repetitions)
    missing_execution = joined.fill_status.isin(["MISSING_B_BAR", "INVALID_B_PRICE"])
    execution_observed = joined[~missing_execution]
    complete_execution_sessions = int(sum(
        not group.fill_status.isin(["MISSING_B_BAR", "INVALID_B_PRICE"]).any()
        for _, group in joined.groupby("signal_date")
    ))
    promotion_ci = intervals["promotion_rate"]
    b2_mean_ci = intervals["b2_mean_net_return"]
    b2_positive_ci = intervals["b2_positive_rate"]
    big_loss_ci = intervals["b1_big_loss_rate"]
    fast_positive_factor = bool(
        complete_execution_sessions >= 20
        and promotion_ci and promotion_ci[0] > 0
        and b2_mean_ci and b2_mean_ci[0] > 0
        and b2_positive_ci and b2_positive_ci[0] > 0
        and big_loss_ci and big_loss_ci[1] <= 0
    )
    fast_hard_reject = bool(
        promotion_ci and promotion_ci[1] < 0
        and b2_mean_ci and b2_mean_ci[1] < 0
        and big_loss_ci and big_loss_ci[0] > 0
    )
    earliest = (joined.sort_values(
        ["signal_date", "first_seal_minute", "code"], ascending=[True, True, True],
        na_position="last").groupby("signal_date", as_index=False).first())
    report = {
        "version": "v160-seal-speed-tradeoff-2",
        "status": "DIAGNOSTIC_ONLY",
        "scope": "first-board candidates on frozen prequential evaluation dates",
        "fast_definition": "first seal no later than 09:35 (20 minutes after 09:15)",
        "evaluation_sessions": int(joined.signal_date.nunique()),
        "execution_observed_sessions": int(execution_observed.signal_date.nunique()),
        "complete_execution_sessions": complete_execution_sessions,
        "candidate_rows": int(len(joined)),
        "execution_observed_rows": int(len(execution_observed)),
        "missing_execution_evidence_rows": int(missing_execution.sum()),
        "groups": {"fast": fast, "not_fast": slow},
        "fast_minus_not_fast": differences,
        "date_cluster_bootstrap95": intervals,
        "earliest_daily_top1": {
            "sessions": int(len(earliest)),
            "promotion_rate": _safe_rate(earliest.promotion_label),
            "filled_n": int(earliest.fill_status.eq("FILLED_OPEN_PROXY").sum()),
        },
        "governance": {
            "fast_seal_positive_ranking_factor_allowed": fast_positive_factor,
            "minimum_complete_execution_sessions_for_factor": 20,
            "missing_bar_is_unfilled_trade": False,
            "fast_seal_hard_reject_allowed": fast_hard_reject,
            "allowed_role": "ATTENTION_OR_IDENTITY_CLUE_ONLY",
            "stock_entry_authorized": False,
        },
        "interpretation": (
            "Seal speed may correlate with next-session promotion while failing to improve "
            "B+2 net return or large-loss risk. It must interact with specific-theme status, "
            "cross-theme competition, buyability and opening absorption."
        ),
        "limitations": [
            "identity and execution coverage are reported separately",
            "a missing B-session bar is unavailable evidence, never an unfilled trade",
            "the evaluation period has already been inspected by prior research",
            "daily open fill is a proxy and cannot verify auction queue or partial execution",
            "specific-theme and cross-theme V150/V157 factors are not present in this sample",
        ],
    }
    return joined, report


def run(features_path, outcomes_path, prequential_path, output_dir):
    paths = list(map(Path, (features_path, outcomes_path, prequential_path)))
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable seal-speed audit")
    joined, report = audit(
        pd.read_csv(paths[0], dtype={"code": str}),
        pd.read_csv(paths[1], dtype={"code": str}),
        pd.read_csv(paths[2], dtype={"code": str}),
    )
    target.mkdir(parents=True)
    joined_path = target / "prequential_speed_tradeoff_rows.csv"
    joined.to_csv(joined_path, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(path.resolve()): _sha256(path) for path in paths}
    report["rows_sha256"] = _sha256(joined_path)
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--prequential", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.outcomes, args.prequential, args.output),
                     ensure_ascii=False, indent=2))
