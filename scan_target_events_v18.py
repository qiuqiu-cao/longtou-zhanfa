"""Find unverified threshold-crossing cases for official reconciliation.

Never a live feature source or training-label archive: historical rule versions,
corporate actions, security status and severe-event resets remain unverified.
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def scan_one(bars, benchmark):
    if bars.date.duplicated().any() or benchmark.date.duplicated().any():
        raise ValueError("duplicate sessions")
    aligned = benchmark[["date", "close"]].rename(columns={"close": "index_close"}).merge(
        bars, on="date", how="left", validate="one_to_one").sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(aligned.close, errors="coerce")
    pre = pd.to_numeric(aligned.pre_close, errors="coerce")
    # The raw pre_close is not asserted to be ex-rights adjusted.
    factor = (close / pre).where((close > 0) & (pre > 0))
    logs = np.log(factor.where(np.isfinite(factor)))
    index_logs = np.log(aligned.index_close / aligned.index_close.shift())
    maximum = pd.Series(np.nan, index=aligned.index)
    for length in range(1, 31):
        deviation = (np.exp(logs.rolling(length, min_periods=length).sum()) -
                     np.exp(index_logs.rolling(length, min_periods=length).sum())) * 100
        maximum = pd.concat([maximum, deviation], axis=1).max(axis=1, skipna=True)
        if length in (10, 30):
            aligned[f"trailing_{length}d_pct"] = deviation
            # Require complete calendar coverage, even for shorter suffix maxima.
            aligned[f"max_suffix_{length}d_pct"] = maximum.where(logs.rolling(length).count().eq(length))
    limit_price = np.floor(pre * 110 + 0.5 + 1e-8) / 100
    aligned["limit_up_proxy"] = (close - limit_price).abs().lt(0.000001).where(close.notna())
    for window, threshold in ((10, 100), (30, 200)):
        hit = aligned[f"max_suffix_{window}d_pct"].ge(threshold)
        aligned[f"cross_{threshold}_proxy"] = hit & ~hit.shift(fill_value=False)
    return aligned


def run(bars_path, benchmark_dir, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    columns = ["date", "code", "name", "close", "pre_close", "st", "status_verified"]
    bars = pd.read_csv(bars_path, usecols=columns, dtype={"code": str})
    bars = bars[bars.code.str.match(r"^(00|60)")].copy()
    benchmarks = {key: pd.read_csv(Path(benchmark_dir) / f"{key}.csv")
                  for key in ("sh000002", "sz399107")}
    events, failures, scanned = [], [], 0
    for code, group in bars.groupby("code", sort=True):
        try:
            symbol = "sh000002" if code.startswith("60") else "sz399107"
            aligned = scan_one(group, benchmarks[symbol])
            selected = aligned[aligned.cross_100_proxy | aligned.cross_200_proxy].copy()
            selected["benchmark_symbol"] = symbol
            selected["indicator_status"] = "inferred_unreconciled"
            events.append(selected)
            scanned += 1
        except Exception as exc:
            failures.append({"code": code, "error": str(exc)})
    event_frame = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    path = output / "threshold_cases_unverified.csv"
    event_frame.to_csv(path, index=False, encoding="utf-8-sig")
    report = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "status": "DIAGNOSTIC_ONLY", "purpose": "official_reconciliation_case_discovery",
              "input_rows": len(bars), "scanned_codes": scanned, "failures": failures,
              "event_rows": len(event_frame),
              "cross_100_rows": int(event_frame.cross_100_proxy.sum()) if len(event_frame) else 0,
              "cross_200_rows": int(event_frame.cross_200_proxy.sum()) if len(event_frame) else 0,
              "usable_verified_labels": 0,
              "limitations": ["no severe-event reset reconstruction", "unverified corporate-action adjustments",
                              "historical rule versions not applied", "historical ST/IPO status not filtered",
                              "survivorship and universe coverage incomplete", "no execution simulation"],
              "input_sha256": hashlib.sha256(Path(bars_path).read_bytes()).hexdigest(),
              "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "benchmark_sha256": {key: hashlib.sha256((Path(benchmark_dir)/f"{key}.csv").read_bytes()).hexdigest()
                                   for key in benchmarks}}
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, required=True)
    parser.add_argument("--benchmarks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.bars, args.benchmarks, args.output), ensure_ascii=False, indent=2))
