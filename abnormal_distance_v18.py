"""Research arithmetic for severe-deviation distances, never hit probabilities.

Inputs are aligned exchange trading sessions with corporate-action-adjusted
stock returns and the correct classification-index returns (percentage points).
Official reset/session eligibility must be provided, not guessed from titles.
Output remains inferred until reconciled against official disclosure examples.
"""
from __future__ import annotations

import math
from decimal import Decimal

import pandas as pd


def interval_deviation(stock_returns, benchmark_returns):
    if not stock_returns or len(stock_returns) != len(benchmark_returns):
        raise ValueError("nonempty aligned return series required")
    stock, benchmark = Decimal(1), Decimal(1)
    for s, b in zip(stock_returns, benchmark_returns):
        for value in (s, b):
            if not math.isfinite(float(value)) or float(value) <= -100:
                raise ValueError("returns must be finite and greater than -100 percent")
        stock *= 1 + Decimal(str(s)) / 100
        benchmark *= 1 + Decimal(str(b)) / 100
    return float((stock - benchmark) * 100)


def window_summary(stock, benchmark, window):
    if window < 1 or not stock or len(stock) != len(benchmark):
        raise ValueError("invalid window or alignment")
    length = min(window, len(stock))
    candidates = [(n, interval_deviation(stock[-n:], benchmark[-n:])) for n in range(1, length + 1)]
    best = max(candidates, key=lambda item: item[1])
    return {"trailing_deviation_pct": candidates[-1][1],
            "maximum_suffix_deviation_pct": best[1], "maximum_suffix_sessions": best[0],
            "available_sessions": length}


def reconstruct(frame: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    """One stock and one benchmark; preserve missing sessions as unknown.

    reset_before=1 starts a known new measurement period on that row. A reset
    must be externally evidenced. Otherwise 30 preceding usable sessions are
    required before all-history coverage is called complete. No-limit days are
    left unsupported, rather than silently skipped or treated as zero return.
    """
    required = {"date", "stock_return_pct", "benchmark_return_pct", "eligible_session",
                "reset_before", "data_status"}
    if not required.issubset(frame):
        raise ValueError(f"required columns: {sorted(required)}")
    sessions = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in calendar]
    if not sessions or sessions != sorted(set(sessions)):
        raise ValueError("calendar must be sorted and unique")
    bars = frame.copy()
    bars["date"] = pd.to_datetime(bars["date"]).dt.strftime("%Y-%m-%d")
    if bars["date"].duplicated().any() or not set(bars.date).issubset(sessions):
        raise ValueError("duplicate or non-calendar dates")
    bars = bars.set_index("date")
    stock, benchmark, known_start = [], [], False
    output = []
    for date in sessions:
        row = bars.loc[date] if date in bars.index else None
        result = {"date": date, "indicator_status": "unavailable",
                  "reason": "missing_or_unverified_session"}
        valid = row is not None and row.data_status == "verified"
        if valid:
            valid = row.eligible_session == 1 and row.reset_before in (0, 1)
        if valid:
            try:
                s, b = float(row.stock_return_pct), float(row.benchmark_return_pct)
                interval_deviation([s], [b])
            except (TypeError, ValueError):
                valid = False
        if not valid:
            stock, benchmark, known_start = [], [], False
            output.append(result)
            continue
        if row.reset_before == 1:
            stock, benchmark, known_start = [], [], True
        stock.append(s)
        benchmark.append(b)
        stock, benchmark = stock[-30:], benchmark[-30:]
        result["coverage_complete"] = known_start or len(stock) >= 30
        result["reason"] = "requires_official_reconciliation" if result["coverage_complete"] else "left_censored_history"
        result["indicator_status"] = "inferred"
        for window in (10, 30):
            summary = window_summary(stock, benchmark, window)
            result.update({f"{window}d_{key}": value for key, value in summary.items()})
        output.append(result)
    return pd.DataFrame(output)


def project_limit_path(stock_returns, benchmark_returns, future_stock_returns,
                       future_benchmark_returns):
    """Explicit scenario, not a forecast. Caller supplies rounded limit returns.

    No future exchange reset assumed; paths beyond a first breach are only
    counterfactual arithmetic and must not be reported as regulatory headroom.
    """
    if len(stock_returns) < 30 or len(stock_returns) != len(benchmark_returns):
        raise ValueError("at least 30 aligned historical returns required")
    if not future_stock_returns or len(future_stock_returns) != len(future_benchmark_returns):
        raise ValueError("explicit aligned future scenarios required")
    stock, benchmark = list(stock_returns), list(benchmark_returns)
    rows = []
    prior_breach = (window_summary(stock, benchmark, 10)["maximum_suffix_deviation_pct"] >= 100
                    or window_summary(stock, benchmark, 30)["maximum_suffix_deviation_pct"] >= 200)
    for step, (s, b) in enumerate(zip(future_stock_returns, future_benchmark_returns), 1):
        stock.append(s)
        benchmark.append(b)
        ten = window_summary(stock, benchmark, 10)["maximum_suffix_deviation_pct"]
        thirty = window_summary(stock, benchmark, 30)["maximum_suffix_deviation_pct"]
        breach = ten >= 100 or thirty >= 200
        rows.append({"future_session": step, "scenario_10d_max_pct": ten,
                     "scenario_30d_max_pct": thirty, "distance_100_pp": 100-ten,
                     "distance_200_pp": 200-thirty, "scenario_threshold_hit": breach,
                     "after_prior_breach": prior_breach, "indicator_status": "inferred",
                     "interpretation": "counterfactual_no_reset_scenario"})
        prior_breach = prior_breach or breach
    return pd.DataFrame(rows)
