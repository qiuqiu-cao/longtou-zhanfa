"""Conditional arithmetic bound at signal close, not regulatory certification.

If eligible reset windows are suffixes of these exact aligned return series,
their deviations cannot exceed the maximum over all suffixes. This does NOT
prove the stock never triggered earlier, and does not validate corporate actions.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from abnormal_distance_v18 import window_summary


def bounds_at(stocks, benchmark, code, signal_date):
    base = dict(code=code, signal_date=signal_date, status="DATA_LIMITED",
                indicator_status="inferred_NOT_regulatory_verified",
                current_below_both_conditional=None, prior_events_ruled_out=False,
                trading_permission=False)
    index = benchmark[benchmark.date <= signal_date].sort_values("date")
    bars = stocks[(stocks.code == code) & (stocks.date <= signal_date)].sort_values("date")
    if index.date.duplicated().any() or bars.date.duplicated().any():
        raise ValueError("duplicate sessions")
    if len(index) < 31 or index.iloc[-1].date != signal_date:
        return {**base, "reason": "incomplete_benchmark_window"}
    index = index.tail(31)
    days = index.date.tolist()[1:]
    bars = bars[bars.date.isin(days)]
    if bars.date.tolist() != days:
        return {**base, "reason": "missing_stock_session_or_suspension"}
    for row in bars.itertuples(index=False):
        flags = (row.status_verified, row.st, row.new_listing)
        if (not all(str(x).lower() in {"true", "false", "0", "1"} for x in flags)
                or str(flags[0]).lower() not in {"true", "1"}
                or any(str(x).lower() in {"true", "1"} for x in flags[1:])):
            return {**base, "reason": "unsupported_historical_security_status"}
    prices = pd.to_numeric(index.close, errors="coerce")
    close = pd.to_numeric(bars.close, errors="coerce")
    pre = pd.to_numeric(bars.pre_close, errors="coerce")
    if not (prices.gt(0).all() and close.gt(0).all() and pre.gt(0).all()):
        return {**base, "reason": "invalid_price"}
    stock_returns = ((close/pre-1)*100).tolist()
    index_returns = ((prices/prices.shift()-1)*100).iloc[1:].tolist()
    try:
        ten = window_summary(stock_returns, index_returns, 10)
        thirty = window_summary(stock_returns, index_returns, 30)
    except (ValueError, TypeError):
        return {**base, "reason": "invalid_returns"}
    below = ten["maximum_suffix_deviation_pct"] < 100 and thirty["maximum_suffix_deviation_pct"] < 200
    return {**base, "status": "CONDITIONAL_ARITHMETIC_ONLY", "reason": "all_suffix_upper_bound",
            "current_below_both_conditional": below,
            "max_10d_pct": ten["maximum_suffix_deviation_pct"],
            "max_30d_pct": thirty["maximum_suffix_deviation_pct"],
            "interpretation": "below_given_input_returns_and_suffix_window_assumption" if below
            else "cannot_decide_actual_trigger_without_reset_history"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scope", "bars", "benchmarks", "output"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    scope = json.loads(Path(args.scope).read_text(encoding="utf-8"))
    bars = pd.read_csv(args.bars, dtype={"code": str}, usecols=["code", "date", "close", "pre_close", "status_verified", "st", "new_listing"])
    results, sources = [], [args.scope, args.bars]
    benchmarks = {}
    for key in ("sh000002", "sz399107"):
        path = Path(args.benchmarks)/f"{key}.csv"
        benchmarks[key] = pd.read_csv(path)
        if not benchmarks[key].symbol.eq(key).all():
            raise ValueError("wrong benchmark symbol")
        sources.append(str(path))
    for row in scope["rows"]:
        if row["scope_status"].startswith("SAME_RELAY_POTENTIAL"):
            key = "sh000002" if row["code"].startswith("60") else "sz399107"
            results.append(bounds_at(bars, benchmarks[key], row["code"], row["signal_date"]))
    report = dict(status="DIAGNOSTIC_ONLY", rows=results, verified_trade_labels=0,
                  limitations=["returns use provider pre_close; ex-rights adjustment is not independently verified",
                               "security-status input provenance not reaudited here",
                               "special session exclusions could invalidate the suffix assumption",
                               "below at signal close does not prove no prior trigger, nor below at next-day entry",
                               "negative bound result is inconclusive, not a confirmed trigger"],
                  source_hashes={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
