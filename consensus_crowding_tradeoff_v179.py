"""Audit the common claim that more one-price boards imply consensus blow-off.

All signal fields are known at the signal-day close and all outcomes are shifted
to the immediately following verified session.  Thresholds are frozen on the
first 70 percent of eligible dates.  The study reports both the broad sample and
the much smaller formal-consensus subset, and cannot authorize stock ranking or
entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v179-consensus-crowding-tradeoff-1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _mean(series):
    values = _numeric(series).dropna()
    return float(values.mean()) if len(values) else None


def _summary(frame: pd.DataFrame) -> dict:
    return {
        "sessions": int(len(frame)),
        "signal_one_word_share_mean": _mean(frame.signal_one_word_share),
        "signal_consensus_rate": _mean(frame.signal_consensus),
        "next_adverse_state_rate": _mean(frame.next_adverse_state),
        "next_failed_rate_mean": _mean(frame.next_failed_rate),
        "next_promotion_mid_mean": _mean(frame.next_promotion_mid),
        "next_previous_limit_median_return_mean": _mean(
            frame.next_previous_limit_median_return),
    }


def _difference(frame: pd.DataFrame, column: str, seed: int,
                draws: int = 3000) -> dict:
    high = _numeric(frame.loc[frame.crowding_bucket.eq("HIGH"), column]).dropna()
    low = _numeric(frame.loc[frame.crowding_bucket.eq("LOW"), column]).dropna()
    if high.empty or low.empty:
        return {"metric": column, "estimate_high_minus_low": None,
                "ci95": [None, None], "high_sessions": int(len(high)),
                "low_sessions": int(len(low))}
    estimate = float(high.mean() - low.mean())
    rng = np.random.default_rng(seed)
    simulations = []
    for _ in range(draws):
        simulations.append(
            rng.choice(high.to_numpy(), len(high), replace=True).mean()
            - rng.choice(low.to_numpy(), len(low), replace=True).mean())
    return {"metric": column, "estimate_high_minus_low": estimate,
            "ci95": [float(np.quantile(simulations, .025)),
                     float(np.quantile(simulations, .975))],
            "high_sessions": int(len(high)), "low_sessions": int(len(low))}


def build(market: pd.DataFrame,
          cutoff_exclusive: str = "2025-12-26") -> tuple[pd.DataFrame, dict]:
    required = {
        "date", "data_status", "one_word_share", "state_consensus",
        "state_divergence", "state_retreat", "failed_rate", "promotion_mid",
        "previous_limit_median_return",
    }
    if required - set(market):
        raise ValueError("market fields missing")
    frame = market.copy()
    frame["session"] = pd.to_datetime(frame.date, errors="coerce")
    if frame.session.isna().any() or frame.session.duplicated().any():
        raise ValueError("market dates must be valid and unique")
    frame = frame.sort_values("session").reset_index(drop=True)
    for column in (
        "one_word_share", "state_consensus", "state_divergence", "state_retreat",
        "failed_rate", "promotion_mid", "previous_limit_median_return",
    ):
        frame[column] = _numeric(frame[column])
    frame["previous_status"] = frame.data_status.shift(1)
    frame["next_status"] = frame.data_status.shift(-1)
    frame["next_session"] = frame.session.shift(-1).dt.strftime("%Y-%m-%d")
    frame["next_adverse_state"] = (
        frame.state_divergence.shift(-1).eq(1)
        | frame.state_retreat.shift(-1).eq(1)
    ).astype(float)
    frame["next_failed_rate"] = frame.failed_rate.shift(-1)
    frame["next_promotion_mid"] = frame.promotion_mid.shift(-1)
    frame["next_previous_limit_median_return"] = (
        frame.previous_limit_median_return.shift(-1))

    eligible = (
        frame.session.lt(pd.Timestamp(cutoff_exclusive))
        & frame.data_status.eq("verified")
        & frame.previous_status.eq("verified")
        & frame.next_status.eq("verified")
        & frame.one_word_share.notna()
    )
    rows = frame.loc[eligible, [
        "date", "next_session", "one_word_share", "state_consensus",
        "next_adverse_state", "next_failed_rate", "next_promotion_mid",
        "next_previous_limit_median_return",
    ]].rename(columns={
        "date": "signal_date", "one_word_share": "signal_one_word_share",
        "state_consensus": "signal_consensus",
    }).reset_index(drop=True)
    if len(rows) < 200:
        raise ValueError("insufficient verified one-word-share sessions")

    dates = rows.signal_date.drop_duplicates().sort_values().tolist()
    split_index = max(1, min(len(dates) - 1, int(len(dates) * .70)))
    split_date = dates[split_index]
    development_mask = rows.signal_date.lt(split_date)
    development = rows.loc[development_mask]
    low_threshold = float(development.signal_one_word_share.quantile(.30))
    high_threshold = float(development.signal_one_word_share.quantile(.70))
    if low_threshold >= high_threshold:
        raise ValueError("development crowding thresholds do not separate")
    rows["crowding_bucket"] = np.select(
        [rows.signal_one_word_share.le(low_threshold),
         rows.signal_one_word_share.ge(high_threshold)],
        ["LOW", "HIGH"], default="MIDDLE")
    development = rows.loc[development_mask]
    validation = rows.loc[~development_mask]

    metrics = [
        "next_adverse_state", "next_failed_rate", "next_promotion_mid",
        "next_previous_limit_median_return",
    ]
    development_comparisons = {
        metric: _difference(development, metric, 17900 + index)
        for index, metric in enumerate(metrics)
    }
    validation_comparisons = {
        metric: _difference(validation, metric, 17910 + index)
        for index, metric in enumerate(metrics)
    }
    adverse_development = development_comparisons["next_adverse_state"]
    adverse_validation = validation_comparisons["next_adverse_state"]
    adverse_stable = bool(
        adverse_development["ci95"][0] is not None
        and adverse_development["ci95"][0] > 0
        and adverse_validation["ci95"][0] is not None
        and adverse_validation["ci95"][0] > 0)

    quartiles = pd.qcut(rows.signal_one_word_share, 4, duplicates="drop")
    quartile_table = []
    for bucket, group in rows.groupby(quartiles, observed=True):
        item = _summary(group)
        item["interval"] = str(bucket)
        quartile_table.append(item)
    adverse_rates = [item["next_adverse_state_rate"] for item in quartile_table]
    monotonic_risk = bool(
        len(adverse_rates) == 4
        and all(value is not None for value in adverse_rates)
        and all(left <= right for left, right in zip(adverse_rates, adverse_rates[1:])))

    consensus = rows[rows.signal_consensus.eq(1)]
    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "cutoff_exclusive": pd.Timestamp(cutoff_exclusive).strftime("%Y-%m-%d"),
        "session_rows": int(len(rows)),
        "split_date": split_date,
        "development_thresholds": {
            "low_one_word_share_q30": low_threshold,
            "high_one_word_share_q70": high_threshold,
            "thresholds_frozen_before_validation": True,
        },
        "bucket_summaries": {
            bucket: _summary(rows[rows.crowding_bucket.eq(bucket)])
            for bucket in ("LOW", "MIDDLE", "HIGH")
        },
        "development_comparisons": development_comparisons,
        "validation_comparisons": validation_comparisons,
        "quartile_summaries": quartile_table,
        "formal_consensus_subset": {
            **_summary(consensus),
            "minimum_sessions_for_factor_claim": 50,
            "factor_claim_sample_sufficient": bool(len(consensus) >= 50),
        },
        "high_one_word_share_next_adverse_association_stable": adverse_stable,
        "one_word_share_monotonic_adverse_risk_supported": monotonic_risk,
        "governance": {
            "one_word_share_alone_proves_consensus_blowoff": False,
            "formal_consensus_sample_supports_factor_claim": bool(len(consensus) >= 50),
            "direct_emotion_gate_allowed": bool(adverse_stable and len(consensus) >= 50),
            "direct_stock_reranking_allowed": False,
            "entry_authorized": False,
        },
        "interpretation": (
            "One-price-board share is a crowding and buyability descriptor, not a "
            "monotonic forecast of next-session deterioration. Formal consensus days "
            "must be reported separately because their sample is sparse."),
        "limitations": [
            "one_word_share is an aggregate close field and does not prove queue ownership or fill",
            "formal consensus days are rare in the archived sample",
            "no theme structure, candidate identity, entry fill, or B+1/B+2/B+3 trade path",
            "validation dates have been inspected by earlier project research",
        ],
    }
    return rows, report


def run(input_path: str, output_dir: str,
        cutoff_exclusive: str = "2025-12-26") -> dict:
    source, target = Path(input_path), Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V179 output")
    rows, report = build(pd.read_csv(source), cutoff_exclusive)
    report["source_hashes"] = {str(source.resolve()): _sha256(source)}
    target.mkdir(parents=True, exist_ok=False)
    rows.to_csv(target / "crowding_rows.csv", index=False, encoding="utf-8-sig")
    report["rows_sha256"] = _sha256(target / "crowding_rows.csv")
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/v3/market_emotion.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cutoff-exclusive", default="2025-12-26")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.cutoff_exclusive),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
