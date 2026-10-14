"""Reconcile manually reviewed official intervals; not a training-set builder."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from abnormal_distance_v18 import interval_deviation


def reconcile(case, stocks, benchmark):
    stock = stocks[stocks.code.eq(case["code"]) & stocks.date.between(case["start"], case["end"])].sort_values("date")
    index = benchmark[benchmark.date.between(case["start"], case["end"])].sort_values("date")
    result = {**case, "result": "DATA_LIMITED", "calculated_deviation_pct": None,
              "training_label_ready": False}
    if (stock.date.duplicated().any() or index.date.duplicated().any()
            or len(stock) != case["sessions"] or stock.date.tolist() != index.date.tolist()):
        result["reason"] = "session_alignment_or_count_mismatch"
        return result
    if len(stock) == 0 or stock.date.iloc[0] != case["start"] or stock.date.iloc[-1] != case["end"]:
        result["reason"] = "interval_boundary_mismatch"
        return result
    try:
        returns = (stock.close / stock.pre_close - 1) * 100
        deviation = interval_deviation(returns.tolist(), index.benchmark_return_pct.tolist())
    except (TypeError, ValueError, ZeroDivisionError):
        result["reason"] = "invalid_returns"
        return result
    if case["comparison"] != "greater_than":
        raise ValueError("unsupported official comparison")
    result.update(calculated_deviation_pct=deviation,
                  result="THRESHOLD_DIRECTION_MATCH" if deviation > case["threshold_pct"] else "MISMATCH",
                  reason="matches_threshold_direction_not_an_exact_official_numeric_value")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--bars", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite reconciliation")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    stocks = pd.read_csv(args.bars, usecols=["date", "code", "close", "pre_close"], dtype={"code": str})
    benchmark = pd.read_csv(args.benchmark)
    if not benchmark.symbol.eq("sz399107").all() or any(not c["code"].startswith("00") for c in cases):
        raise ValueError("this case run requires Shenzhen mainboard and sz399107")
    report = {"status": "DIAGNOSTIC_ONLY", "cases": [reconcile(c, stocks, benchmark) for c in cases],
              "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [args.cases, args.bars, args.benchmark]},
              "unresolved": ["full event/reset chronology", "negative labels and failed leaders",
                             "entry executability", "historical point-in-time feature data"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["cases"], ensure_ascii=False, indent=2))
