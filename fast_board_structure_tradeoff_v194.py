"""Audit fast-board structure without inferring an unobservable controlling trader.

The study separates a thin/one-price proxy, a turnover-validated fast proxy,
other fast boards, and non-fast boards.  Promotion identity, observable
execution, B+2 payoff, and B+1 large-loss risk remain separate targets.
Missing B-session bars are unavailable evidence rather than failed fills.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUPS = (
    "THIN_LIQUIDITY_OR_ONE_PRICE_PROXY",
    "TURNOVER_VALIDATED_FAST_PROXY",
    "OTHER_FAST",
    "NOT_FAST",
)
MISSING_EXECUTION = {"MISSING_B_BAR", "INVALID_B_PRICE"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_fast_structure(fast, thin_proxy, validated_proxy) -> str:
    values = [pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
              for value in (fast, thin_proxy, validated_proxy)]
    if any(pd.isna(value) or value not in (0, 1) for value in values):
        return "UNAVAILABLE"
    fast, thin_proxy, validated_proxy = map(int, values)
    if thin_proxy and validated_proxy:
        return "INVALID_OVERLAP"
    if not fast:
        return "NOT_FAST"
    if thin_proxy:
        return "THIN_LIQUIDITY_OR_ONE_PRICE_PROXY"
    if validated_proxy:
        return "TURNOVER_VALIDATED_FAST_PROXY"
    return "OTHER_FAST"


def _rate(values: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _metrics(part: pd.DataFrame) -> dict:
    missing = part.fill_status.isin(MISSING_EXECUTION)
    observed = part[~missing]
    filled = observed[observed.fill_status.eq("FILLED_OPEN_PROXY")]
    b2 = pd.to_numeric(filled.net_return_B_plus_2, errors="coerce").dropna()
    loss = pd.to_numeric(filled.big_loss_B_plus_1, errors="coerce").dropna()
    return {
        "candidate_n": int(len(part)),
        "session_n": int(part.signal_date.nunique()),
        "promotion_rate": _rate(part.promotion_label),
        "execution_observed_n": int(len(observed)),
        "missing_execution_evidence_n": int(missing.sum()),
        "filled_n": int(len(filled)),
        "observable_fill_rate": float(len(filled) / len(observed)) if len(observed) else None,
        "unverifiable_limit_queue_rate": (
            float(observed.fill_status.eq("UNVERIFIABLE_LIMIT_QUEUE").mean())
            if len(observed) else None),
        "b2_known_n": int(len(b2)),
        "b2_mean_net_return": float(b2.mean()) if len(b2) else None,
        "b2_positive_rate": float(b2.gt(0).mean()) if len(b2) else None,
        "b1_big_loss_known_n": int(len(loss)),
        "b1_big_loss_rate": float(loss.mean()) if len(loss) else None,
    }


def _difference(frame: pd.DataFrame, left: str, right: str, metric: str) -> float | None:
    a = _metrics(frame[frame.structure_group.eq(left)]).get(metric)
    b = _metrics(frame[frame.structure_group.eq(right)]).get(metric)
    return float(a - b) if a is not None and b is not None else None


def _bootstrap(frame: pd.DataFrame, left: str, right: str,
               repetitions: int, seed: int = 194) -> dict:
    dates = np.array(sorted(frame.signal_date.unique()))
    by_date = {date: frame[frame.signal_date.eq(date)] for date in dates}
    metrics = (
        "promotion_rate", "observable_fill_rate", "unverifiable_limit_queue_rate",
        "b2_mean_net_return", "b2_positive_rate", "b1_big_loss_rate",
    )
    draws = {metric: [] for metric in metrics}
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        sampled_dates = rng.choice(dates, size=len(dates), replace=True)
        sample = pd.concat([by_date[date] for date in sampled_dates], ignore_index=True)
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
        "signal_date", "code", "fast_seal", "thin_control_risk",
        "validated_fast_seal", "one_price_proxy", "turnover",
        "amount_to_float_cap", "failed_seal_count",
    }
    outcome_fields = {
        "signal_date", "code", "promotion_label", "fill_status",
        "net_return_B_plus_2", "big_loss_B_plus_1",
    }
    if feature_fields - set(features) or outcome_fields - set(outcomes):
        raise ValueError("fast-board structure feature or outcome fields missing")
    if not {"signal_date", "code"} <= set(prequential):
        raise ValueError("prequential keys missing")
    keys = prequential[["signal_date", "code"]].drop_duplicates()
    if len(keys) != len(prequential):
        raise ValueError("prequential candidate keys must be unique")
    joined = (keys.merge(features[list(feature_fields)], on=["signal_date", "code"],
                         how="left", validate="one_to_one")
              .merge(outcomes[list(outcome_fields)], on=["signal_date", "code"],
                     how="left", validate="one_to_one"))
    if joined[["fast_seal", "thin_control_risk", "validated_fast_seal",
               "promotion_label", "fill_status"]].isna().any().any():
        raise ValueError("prequential rows are not fully covered by feature/outcome evidence")
    joined["structure_group"] = [
        classify_fast_structure(fast, thin, validated)
        for fast, thin, validated in zip(
            joined.fast_seal, joined.thin_control_risk, joined.validated_fast_seal)
    ]
    if joined.structure_group.isin(["UNAVAILABLE", "INVALID_OVERLAP"]).any():
        raise ValueError("fast-board structure flags are invalid or overlapping")

    missing = joined.fill_status.isin(MISSING_EXECUTION)
    observed = joined[~missing]
    complete_sessions = int(sum(
        not group.fill_status.isin(MISSING_EXECUTION).any()
        for _, group in joined.groupby("signal_date")
    ))
    group_metrics = {group: _metrics(joined[joined.structure_group.eq(group)])
                     for group in GROUPS}
    comparisons = {}
    for left, right, label in (
        ("THIN_LIQUIDITY_OR_ONE_PRICE_PROXY", "TURNOVER_VALIDATED_FAST_PROXY",
         "thin_minus_turnover_validated_fast"),
        ("THIN_LIQUIDITY_OR_ONE_PRICE_PROXY", "NOT_FAST", "thin_minus_not_fast"),
        ("TURNOVER_VALIDATED_FAST_PROXY", "NOT_FAST", "validated_fast_minus_not_fast"),
    ):
        point = {
            metric: _difference(joined, left, right, metric)
            for metric in (
                "promotion_rate", "observable_fill_rate", "unverifiable_limit_queue_rate",
                "b2_mean_net_return", "b2_positive_rate", "b1_big_loss_rate",
            )
        }
        comparisons[label] = {
            "left": left,
            "right": right,
            "difference": point,
            "date_cluster_bootstrap95": _bootstrap(
                joined, left, right, bootstrap_repetitions),
        }

    report = {
        "version": "v194-fast-board-structure-tradeoff-1",
        "status": "DIAGNOSTIC_ONLY",
        "scope": "first-board candidates on frozen prequential evaluation dates",
        "candidate_rows": int(len(joined)),
        "identity_sessions": int(joined.signal_date.nunique()),
        "execution_observed_rows": int(len(observed)),
        "execution_observed_sessions": int(observed.signal_date.nunique()),
        "complete_execution_sessions": complete_sessions,
        "missing_execution_evidence_rows": int(missing.sum()),
        "groups": group_metrics,
        "comparisons": comparisons,
        "governance": {
            "control_intent_inference_allowed": False,
            "thin_proxy_is_actual_control_or_manipulation": False,
            "missing_bar_is_unfilled_trade": False,
            "minimum_complete_execution_sessions_for_factor": 20,
            "fast_structure_positive_ranking_factor_allowed": False,
            "allowed_role": "EXECUTION_RISK_AND_IDENTITY_CONTEXT_ONLY",
            "stock_entry_authorized": False,
        },
        "interpretation": (
            "A fast seal can describe attention or thin price discovery, but daily pool data "
            "cannot identify who controlled the queue or why.  Promotion, observable fill, "
            "post-fill B+2 payoff and B+1 tail loss must agree before any structure becomes a "
            "ranking factor."
        ),
        "limitations": [
            "thin_control_risk is renamed conceptually as a thin-liquidity/one-price proxy",
            "no Level-2 owner, cancellation or order-event evidence is available",
            "specific-theme membership and within-theme propagation order are absent",
            "the evaluation block has been inspected and is not a fresh blind test",
            "missing B-session bars are unavailable evidence rather than failed fills",
        ],
    }
    return joined, report


def run(features_path, outcomes_path, prequential_path, output_dir) -> dict:
    paths = list(map(Path, (features_path, outcomes_path, prequential_path)))
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable fast-board structure audit")
    rows, report = audit(
        pd.read_csv(paths[0], dtype={"code": str}),
        pd.read_csv(paths[1], dtype={"code": str}),
        pd.read_csv(paths[2], dtype={"code": str}),
    )
    target.mkdir(parents=True)
    rows_path = target / "prequential_fast_board_structure_rows.csv"
    rows.to_csv(rows_path, index=False, encoding="utf-8-sig")
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
