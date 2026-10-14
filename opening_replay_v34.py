"""Historical bar-end observations, separated from future execution diagnostics.

Not a buy signal: five-minute timestamps are assumed to denote bar ends.
No intrabar order, queue priority or actual receipt latency is available.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def observe(bars, asof, limit_price):
    """Only completed bars enter the observation; future rows are not features."""
    if not math.isfinite(limit_price) or limit_price <= 0:
        raise ValueError("positive finite limit price required")
    prefix = [b for b in bars if b["time"] <= asof]
    times = [b["time"] for b in prefix]
    if times != sorted(set(times)):
        raise ValueError("observed timestamps must be unique and ordered")
    if not prefix:
        return dict(state="NO_COMPLETED_BAR", observed_bars=0, decision="WATCH")
    for bar in prefix:
        values = [float(bar[k]) for k in ("open", "high", "low", "close", "volume")]
        if (not all(math.isfinite(x) for x in values) or min(values[:4]) <= 0
                or values[4] < 0 or values[2] > min(values[0], values[3])
                or values[1] < max(values[0], values[3]) or values[2] > values[1]):
            raise ValueError("invalid observed OHLCV")
    last = prefix[-1]
    at_limit = abs(float(last["close"])-limit_price) < 1e-6
    touched = any(float(b["high"]) >= limit_price-1e-6 for b in prefix)
    return dict(state="CLOSE_AT_LIMIT_PROXY" if at_limit else
                "TOUCHED_THEN_BELOW_PROXY" if touched else "BELOW_LIMIT",
                observed_bars=len(prefix), latest_bar=last["time"],
                close=float(last["close"]), decision="WATCH",
                probability=None, exact_seal_time=None)


def next_bar_diagnostic(bars, asof, limit_price):
    """LABEL ONLY: hypothetical buy limit at limit_price after the decision.

    A trade below the order price is only an opportunity proxy, not a fill.
    Submission/receipt latency and intrabar sequencing remain unknown.
    """
    future = sorted((b for b in bars if b["time"] > asof), key=lambda b: b["time"])
    result = dict(label_only=True, fill_verified=False, simulated_entry_price=None)
    if not future:
        return dict(result, state="NO_NEXT_BAR")
    bar = future[0]
    low, high, volume = (float(bar[k]) for k in ("low", "high", "volume"))
    if not all(math.isfinite(x) for x in (low, high, volume)) or low <= 0 or high < low or volume < 0:
        return dict(result, state="INVALID_NEXT_BAR")
    if volume == 0:
        state = "NO_RECORDED_VOLUME"
    elif low > limit_price + 1e-6:
        state = "NO_PRICE_CROSS"
    elif abs(low-limit_price) < 1e-6 and abs(high-limit_price) < 1e-6:
        state = "LIMIT_QUEUE_UNRESOLVED"
    elif low < limit_price-1e-6:
        state = "BELOW_LIMIT_TRADES_OPPORTUNITY_ONLY"
    else:
        state = "TOUCH_ONLY_QUEUE_UNRESOLVED"
    return dict(result, state=state, next_bar=bar["time"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.archive/"manifest.json").read_text(encoding="utf-8"))
    # Case-study constants, NOT inferred from future intraday highs.
    # Historical daily limit-price proxies; not exchange-certified inputs.
    limits = {("sz.000017", "2024-01-15"): 7.21, ("sz.000017", "2024-01-17"): 8.72,
              ("sz.000759", "2024-12-27"): 10.81, ("sz.000759", "2024-12-30"): 11.89}
    rows = []
    for case in manifest["cases"]:
        path = args.archive/(case["code"]+"_"+case["date"]+".json")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != case["raw_sha256"]:
            raise ValueError("raw archive hash mismatch")
        raw = json.loads(payload)
        if raw["code"] != case["code"] or raw["requested_date"] != case["date"] or raw["error_code"] != "0":
            raise ValueError("archive identity or provider error")
        bars = [dict(zip(raw["fields"], row)) for row in raw["rows"]]
        limit = limits[(case["code"], case["date"])]
        for clock in ("093500000", "095000000"):
            asof = case["date"].replace("-", "")+clock
            rows.append(dict(code=case["code"], date=case["date"], asof=asof,
                             observation=observe(bars, asof, limit),
                             execution_LABEL_ONLY=next_bar_diagnostic(bars, asof, limit)))
    report = dict(status="DIAGNOSTIC_ONLY", retrospective_case_study=True,
                  timestamp_semantics="assumed bar-end; receipt latency unknown",
                  limit_price_status="inferred historical proxy",
                  note="Read V33 OHLC mismatches separately; no live entry authorization or win rate.",
                  verified_fills=0, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
