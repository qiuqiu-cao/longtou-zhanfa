"""Cluster-bootstrap uncertainty for the fixed 10:00 cumulative-VWAP filter."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def _metrics(frame):
    value = pd.to_numeric(frame.b2_net_close_mark_LABEL_ONLY, errors="coerce")
    known = value.notna()
    observed = value[known]
    return {"feature_rows": int(len(frame)), "b2_known": int(known.sum()),
            "b2_opportunity_proxy_rate": float(known.mean()) if len(frame) else None,
            "b2_mean": float(observed.mean()) if len(observed) else None,
            "b2_positive_rate": float(observed.gt(0).mean()) if len(observed) else None,
            "b2_large_loss_rate": float(observed.le(-.07).mean()) if len(observed) else None}


def _differences(frame):
    high = _metrics(frame[frame.close_1000_above_cum_vwap.eq(True)])
    low = _metrics(frame[frame.close_1000_above_cum_vwap.eq(False)])
    fields = ["b2_opportunity_proxy_rate", "b2_mean", "b2_positive_rate", "b2_large_loss_rate"]
    diff = {field: (high[field]-low[field] if high[field] is not None and low[field] is not None else None)
            for field in fields}
    return high, low, diff


def audit(cohort, features, outcomes, repetitions=3000):
    for frame in (cohort, features, outcomes):
        frame["code"] = frame.code.astype(str).str.zfill(6)
    joined = cohort.merge(features, left_on=["code", "entry_date"], right_on=["code", "date"],
                          how="left", validate="one_to_one")
    joined = joined.merge(outcomes, on=["code", "signal_date"], how="left", validate="one_to_one")
    joined = joined[joined.feature_usable.fillna(False)].copy()
    rng = np.random.default_rng(139)
    result = {}
    for board_group, part in joined.groupby("board_group"):
        high, low, difference = _differences(part)
        dates = part.signal_date.unique()
        samples = {field: [] for field in difference}
        for _ in range(repetitions):
            chosen = rng.choice(dates, size=len(dates), replace=True)
            sampled = pd.concat([part[part.signal_date.eq(day)] for day in chosen], ignore_index=True)
            _, _, delta = _differences(sampled)
            for field, value in delta.items():
                if value is not None and np.isfinite(value):
                    samples[field].append(value)
        intervals = {field: (np.quantile(values, [.025, .975]).tolist() if values else None)
                     for field, values in samples.items()}
        result[board_group] = {"above_vwap": high, "not_above_vwap": low,
                               "above_minus_not_above": difference,
                               "cluster_bootstrap_95": intervals,
                               "cluster_unit": "signal_date", "sessions": int(len(dates))}
    return {"version": "v139-absorption-filter-uncertainty-1", "status": "DIAGNOSTIC_ONLY",
            "fixed_filter": "10:00 completed five-minute close >= cumulative VWAP",
            "groups": result, "bootstrap_repetitions": repetitions,
            "actual_verified_fills": 0, "complete_trade_exit_tested": False,
            "positive_buy_bonus_authorized": False, "research_veto_authorized": False,
            "deployment_authorized": False,
            "limitations": ["B+2 is a close-mark proxy conditional on hypothetical price-path opportunity",
                            "first-board minute coverage is incomplete and selected",
                            "historical data were already inspected; this is not a new blind test"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(pd.read_csv(args.cohort, dtype={"code": str}),
                   pd.read_csv(args.features, dtype={"code": str}),
                   pd.read_csv(args.outcomes, dtype={"code": str}))
    report["source_hashes"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (args.cohort, args.features, args.outcomes)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report["groups"], ensure_ascii=False, indent=2))
