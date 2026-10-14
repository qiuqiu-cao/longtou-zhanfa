"""Test whether reopening path is more informative than raw failed-seal count.

The audit is deliberately diagnostic.  It compares a clean seal, a reopened
board that repairs within 30 minutes, and a prolonged contest.  A category is
not promoted into ranking unless it improves promotion, executable B+2 return,
and B+1 tail risk on later prequential sessions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUPS = ("LOCKED_NO_REOPEN", "QUICK_REPAIR", "PROLONGED_CONTEST")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_path(failed_seal_count, last_minus_first_minute) -> str:
    failed = pd.to_numeric(pd.Series([failed_seal_count]), errors="coerce").iloc[0]
    span = pd.to_numeric(pd.Series([last_minus_first_minute]), errors="coerce").iloc[0]
    if pd.isna(failed) or failed < 0:
        return "UNAVAILABLE"
    if failed == 0:
        return "LOCKED_NO_REOPEN"
    if pd.isna(span) or span < 0:
        return "UNAVAILABLE"
    return "QUICK_REPAIR" if span <= 30 else "PROLONGED_CONTEST"


def _rate(values: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _metrics(part: pd.DataFrame) -> dict:
    missing = part.fill_status.isin(["MISSING_B_BAR", "INVALID_B_PRICE"])
    execution_observed = part[~missing]
    filled = part[part.fill_status.eq("FILLED_OPEN_PROXY")]
    b2 = pd.to_numeric(filled.net_return_B_plus_2, errors="coerce").dropna()
    loss = pd.to_numeric(filled.big_loss_B_plus_1, errors="coerce").dropna()
    return {
        "candidate_n": int(len(part)),
        "session_n": int(part.signal_date.nunique()),
        "promotion_rate": _rate(part.promotion_label),
        "execution_observed_n": int(len(execution_observed)),
        "missing_execution_evidence_n": int(missing.sum()),
        "unverifiable_limit_queue_n": int(
            execution_observed.fill_status.eq("UNVERIFIABLE_LIMIT_QUEUE").sum()),
        "filled_n": int(len(filled)),
        "b2_known_n": int(len(b2)),
        "b2_mean_net_return": float(b2.mean()) if len(b2) else None,
        "b2_positive_rate": float(b2.gt(0).mean()) if len(b2) else None,
        "b1_big_loss_known_n": int(len(loss)),
        "b1_big_loss_rate": float(loss.mean()) if len(loss) else None,
    }


def _difference(frame: pd.DataFrame, left: str, right: str, metric: str) -> float | None:
    a = _metrics(frame[frame.path_group.eq(left)]).get(metric)
    b = _metrics(frame[frame.path_group.eq(right)]).get(metric)
    if a is None or b is None:
        return None
    return float(a - b)


def _bootstrap(frame: pd.DataFrame, left: str, right: str,
               repetitions: int, seed: int = 167) -> dict:
    dates = np.array(sorted(frame.signal_date.unique()))
    groups = {date: frame[frame.signal_date.eq(date)] for date in dates}
    metrics = ("promotion_rate", "b2_mean_net_return", "b2_positive_rate", "b1_big_loss_rate")
    draws = {metric: [] for metric in metrics}
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        sample_dates = rng.choice(dates, size=len(dates), replace=True)
        sample = pd.concat([groups[date] for date in sample_dates], ignore_index=True)
        for metric in metrics:
            value = _difference(sample, left, right, metric)
            if value is not None and np.isfinite(value):
                draws[metric].append(value)
    return {
        metric: np.quantile(values, [.025, .975]).tolist() if values else None
        for metric, values in draws.items()
    }


def audit(features: pd.DataFrame, outcomes: pd.DataFrame, prequential: pd.DataFrame,
          bootstrap_repetitions: int = 2000) -> tuple[pd.DataFrame, dict]:
    feature_fields = {
        "signal_date", "code", "failed_seal_count", "last_minus_first_minute",
    }
    outcome_fields = {
        "signal_date", "code", "promotion_label", "fill_status",
        "net_return_B_plus_2", "big_loss_B_plus_1",
    }
    if feature_fields - set(features) or outcome_fields - set(outcomes):
        raise ValueError("failed-seal path feature or outcome fields missing")
    if not {"signal_date", "code"} <= set(prequential):
        raise ValueError("prequential keys missing")
    keys = prequential[["signal_date", "code"]].drop_duplicates()
    if len(keys) != len(prequential):
        raise ValueError("prequential candidate keys must be unique")
    joined = (keys.merge(features[list(feature_fields)], on=["signal_date", "code"],
                         how="left", validate="one_to_one")
              .merge(outcomes[list(outcome_fields)], on=["signal_date", "code"],
                     how="left", validate="one_to_one"))
    if joined.promotion_label.isna().any():
        raise ValueError("prequential rows are not fully covered by outcomes")
    joined["path_group"] = [
        classify_path(failed, span)
        for failed, span in zip(joined.failed_seal_count, joined.last_minus_first_minute)
    ]
    unavailable = joined.path_group.eq("UNAVAILABLE")
    evaluable = joined[~unavailable].copy()
    missing_execution = evaluable.fill_status.isin(["MISSING_B_BAR", "INVALID_B_PRICE"])
    execution_observed = evaluable[~missing_execution]
    complete_execution_sessions = int(sum(
        not group.fill_status.isin(["MISSING_B_BAR", "INVALID_B_PRICE"]).any()
        for _, group in evaluable.groupby("signal_date")
    ))
    group_metrics = {group: _metrics(evaluable[evaluable.path_group.eq(group)]) for group in GROUPS}
    comparison = ("QUICK_REPAIR", "PROLONGED_CONTEST")
    metrics = ("promotion_rate", "b2_mean_net_return", "b2_positive_rate", "b1_big_loss_rate")
    differences = {
        metric: _difference(evaluable, *comparison, metric) for metric in metrics
    }
    intervals = _bootstrap(evaluable, *comparison, bootstrap_repetitions)
    quick = group_metrics["QUICK_REPAIR"]
    evidence_gate = bool(
        complete_execution_sessions >= 20
        and quick["candidate_n"] >= 100 and quick["b2_known_n"] >= 50
        and intervals["promotion_rate"] and intervals["promotion_rate"][0] > 0
        and intervals["b2_mean_net_return"] and intervals["b2_mean_net_return"][0] > 0
        and intervals["b1_big_loss_rate"] and intervals["b1_big_loss_rate"][1] <= 0
    )
    report = {
        "version": "v167-failed-seal-path-tradeoff-2",
        "status": "DIAGNOSTIC_ONLY",
        "scope": "first-board candidates on frozen prequential evaluation dates",
        "definitions": {
            "LOCKED_NO_REOPEN": "failed_seal_count = 0",
            "QUICK_REPAIR": "failed_seal_count > 0 and last seal minus first seal <= 30 minutes",
            "PROLONGED_CONTEST": "failed_seal_count > 0 and last seal minus first seal > 30 minutes",
        },
        "evaluation_sessions": int(evaluable.signal_date.nunique()),
        "execution_observed_sessions": int(execution_observed.signal_date.nunique()),
        "complete_execution_sessions": complete_execution_sessions,
        "candidate_rows": int(len(joined)),
        "evaluable_rows": int(len(evaluable)),
        "unavailable_rows": int(unavailable.sum()),
        "execution_observed_rows": int(len(execution_observed)),
        "missing_execution_evidence_rows": int(missing_execution.sum()),
        "groups": group_metrics,
        "quick_repair_minus_prolonged_contest": differences,
        "date_cluster_bootstrap95": intervals,
        "governance": {
            "failed_seal_path_positive_ranking_factor_allowed": evidence_gate,
            "minimum_complete_execution_sessions_for_factor": 20,
            "missing_bar_is_unfilled_trade": False,
            "raw_failed_seal_count_monotonic_rule_allowed": False,
            "stock_entry_authorized": False,
        },
        "interpretation": (
            "Raw reopening count is not a sufficient strength score.  The duration of the "
            "contest, later promotion, executable holding return and tail loss must agree."
        ),
        "limitations": [
            "only later frozen first-board dates are evaluated",
            "identity and execution coverage are reported separately",
            "a missing B-session bar is unavailable evidence, never an unfilled trade",
            "last-minus-first seal time does not reveal every intraday reopen duration",
            "daily open fill is a proxy rather than queue-level execution",
            "specific-theme and auction absorption are absent from this isolated test",
        ],
    }
    return joined, report


def run(features_path, outcomes_path, prequential_path, output_dir) -> dict:
    paths = list(map(Path, (features_path, outcomes_path, prequential_path)))
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable failed-seal path audit")
    joined, report = audit(
        pd.read_csv(paths[0], dtype={"code": str}),
        pd.read_csv(paths[1], dtype={"code": str}),
        pd.read_csv(paths[2], dtype={"code": str}),
    )
    target.mkdir(parents=True)
    rows_path = target / "prequential_failed_seal_path_rows.csv"
    joined.to_csv(rows_path, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(path.resolve()): _sha256(path) for path in paths}
    report["rows_sha256"] = _sha256(rows_path)
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
