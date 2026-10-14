"""Separate historical promotion and hypothetical price-path outcome labels.

No fills are established. Exclusive outcome cutoff must be explicitly supplied
so reserved later periods are not silently consumed by exploratory research.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from premium_opportunity_research import label_case, rounded_limit, COSTS
from t1_scenario_v44 import net_return


def usable(row):
    if row is None:
        return False
    if not (str(row.get("status_verified")).lower() in {"true", "1"}
            and str(row.get("st")).lower() in {"false", "0"}
            and str(row.get("new_listing")).lower() in {"false", "0"}):
        return False
    try:
        o, h, l, c, p, v = [float(row[k]) for k in ("open", "high", "low", "close", "pre_close", "volume")]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(all(np.isfinite([o,h,l,c,p,v])) and min(o,h,l,c,p,v) > 0
                and l <= min(o,c) <= max(o,c) <= h)


def case(signal, entry, exit_bar):
    result = path_case(signal, entry, [exit_bar])
    # Preserve the original single-window field names for existing consumers.
    result["price_path_LABEL_ONLY"] = result.get("b1_price_path_LABEL_ONLY")
    result["net_close_mark_LABEL_ONLY"] = result.get("b1_net_close_mark_LABEL_ONLY")
    result["big_loss_mark_LABEL_ONLY"] = result.get("b1_big_loss_mark_LABEL_ONLY")
    result["exit_down_limit_LABEL_ONLY"] = result.get("b1_exit_down_limit_LABEL_ONLY")
    result["actual_T1_net_premium_LABEL_ONLY"] = None
    result["complete_trade_profit_LABEL_ONLY"] = None
    return result


def path_case(signal, entry, future_bars):
    """Create B+1/B+2/B+3 close-mark proxies from a common hypothetical entry.

    The entry is only a price-path assumption: a daily high touching the
    inferred limit does not prove queue position, order acceptance or fill.
    """
    result = dict(promotion_LABEL_ONLY=None, promotion_state="MISSING_OR_INVALID_ENTRY_EVIDENCE",
                  entry_one_price_LABEL_ONLY=None, price_path_state="MISSING_OR_INVALID_EXIT_EVIDENCE",
                  next_session_touched_limit_LABEL_ONLY=None,
                  next_session_one_price_limit_LABEL_ONLY=None,
                  actual_entry_verified=False)
    for horizon in range(1, 4):
        result.update({
            f"b{horizon}_price_path_LABEL_ONLY": None,
            f"b{horizon}_net_close_mark_LABEL_ONLY": None,
            f"b{horizon}_big_loss_mark_LABEL_ONLY": None,
            f"b{horizon}_exit_down_limit_LABEL_ONLY": None,
        })
    if not usable(signal) or not usable(entry):
        return result
    if abs(float(entry["pre_close"])-float(signal["close"])) > .011:
        result["promotion_state"] = "PRICE_BASIS_DISCONTINUITY"
        return result
    ceiling = rounded_limit(signal["close"], "1.10")
    if float(entry["high"]) > ceiling + .011:
        result["promotion_state"] = "LIMIT_RULE_UNCONFIRMED"
        return result
    touched = float(entry["high"]) >= ceiling - .00001
    one_price = float(entry["low"]) >= ceiling - .00001
    result.update(promotion_LABEL_ONLY=int(abs(float(entry["close"])-ceiling) < .00001),
                  promotion_state="ROUNDED_DAILY_PROXY", entry_one_price_LABEL_ONLY=one_price,
                  next_session_touched_limit_LABEL_ONLY=touched,
                  next_session_one_price_limit_LABEL_ONLY=one_price)
    for horizon, future in enumerate(list(future_bars)[:3], start=1):
        if not usable(future):
            continue
        if abs(float(future["pre_close"]) - float(entry["close"])) > .011 and horizon == 1:
            result["price_path_state"] = "PRICE_BASIS_DISCONTINUITY"
            continue
        if horizon == 1:
            result["price_path_state"] = "PRICE_OPPORTUNITY_PROXY_ONLY"
        if not touched:
            result[f"b{horizon}_price_path_LABEL_ONLY"] = 0
            continue
        mark = net_return(ceiling, float(future["close"]), **COSTS)
        result[f"b{horizon}_price_path_LABEL_ONLY"] = int(mark > 0)
        result[f"b{horizon}_net_close_mark_LABEL_ONLY"] = mark
        result[f"b{horizon}_big_loss_mark_LABEL_ONLY"] = int(mark <= -.07)
        result[f"b{horizon}_exit_down_limit_LABEL_ONLY"] = (
            float(future["close"]) <= rounded_limit(future["pre_close"], ".90") + .00001)
    if result["price_path_state"] == "MISSING_OR_INVALID_EXIT_EVIDENCE" and not future_bars:
        result["price_path_state"] = "MISSING_OR_INVALID_EXIT_EVIDENCE"
    return result


def build(pool, daily, calendar, end):
    pool = pool.copy()
    if "signal_date" not in pool.columns:
        if "date" not in pool.columns:
            raise ValueError("pool missing signal_date/date")
        pool = pool.rename(columns={"date": "signal_date"})
    if calendar != sorted(set(calendar)) or not daily.index.is_unique:
        raise ValueError("invalid calendar or duplicate daily keys")
    if pool.duplicated(["code", "signal_date"]).any():
        raise ValueError("duplicate candidate keys")
    successor = dict(zip(calendar[:-1], calendar[1:]))
    output = []
    for item in pool.to_dict("records"):
        d0 = item["signal_date"]
        if d0 >= end:
            continue
        if d0 not in calendar:
            raise ValueError("signal missing from calendar")
        d1 = successor.get(d0)
        d2 = successor.get(d1)
        d3 = successor.get(d2)
        d4 = successor.get(d3)
        bars = []
        for day in (d0, d1, d2, d3, d4):
            key = (item["code"], day)
            bars.append(daily.loc[key].to_dict() if day and day < end and key in daily.index else None)
        outcome = path_case(bars[0], bars[1], bars[2:])
        outcome.update(
            price_path_LABEL_ONLY=outcome.get("b1_price_path_LABEL_ONLY"),
            net_close_mark_LABEL_ONLY=outcome.get("b1_net_close_mark_LABEL_ONLY"),
            big_loss_mark_LABEL_ONLY=outcome.get("b1_big_loss_mark_LABEL_ONLY"),
            exit_down_limit_LABEL_ONLY=outcome.get("b1_exit_down_limit_LABEL_ONLY"),
            actual_T1_net_premium_LABEL_ONLY=None,
            complete_trade_profit_LABEL_ONLY=None,
        )
        output.append(dict(code=item["code"], signal_date=d0,
                           entry_date_LABEL_ONLY=d1, exit_date_LABEL_ONLY=d2,
                           b1_date_LABEL_ONLY=d2, b2_date_LABEL_ONLY=d3, b3_date_LABEL_ONLY=d4,
                           **outcome))
    return pd.DataFrame(output)


def path_summary(labels):
    """Summarize path marks without implying an executable trade result."""
    result = {}
    for horizon in (1, 2, 3):
        column = f"b{horizon}_net_close_mark_LABEL_ONLY"
        if column not in labels:
            continue
        values = pd.to_numeric(labels[column], errors="coerce")
        known = values.notna()
        result[f"b{horizon}"] = {
            "known": int(known.sum()),
            "mean_net_close_mark": float(values[known].mean()) if known.any() else None,
            "positive_rate": float(values[known].gt(0).mean()) if known.any() else None,
            "large_loss_rate": float(values[known].le(-.07).mean()) if known.any() else None,
        }
    if all(column in labels for column in ("b2_net_close_mark_LABEL_ONLY", "b3_net_close_mark_LABEL_ONLY")):
        b2 = pd.to_numeric(labels["b2_net_close_mark_LABEL_ONLY"], errors="coerce")
        b3 = pd.to_numeric(labels["b3_net_close_mark_LABEL_ONLY"], errors="coerce")
        both = b2.notna() & b3.notna()
        result["b2_b3_both_positive"] = {
            "known": int(both.sum()),
            "rate": float((b2[both].gt(0) & b3[both].gt(0)).mean()) if both.any() else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observations", "daily", "calendar", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--outcome-end-exclusive", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite labels")
    daily = pd.read_csv(args.daily, dtype={"code": str})
    daily = daily[daily.date < args.outcome_end_exclusive].set_index(["code", "date"])
    labels = build(pd.read_csv(args.observations, dtype={"code": str}), daily,
                   pd.read_csv(args.calendar).date.tolist(), args.outcome_end_exclusive)
    args.output.mkdir(parents=True, exist_ok=False)
    labels.to_csv(args.output/"outcomes_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
    report = dict(status="DIAGNOSTIC_ONLY", rows=len(labels),
                  outcome_end_exclusive=args.outcome_end_exclusive,
                  promotion_states=labels.promotion_state.value_counts().to_dict(),
                  price_path_states=labels.price_path_state.value_counts().to_dict(),
                  verified_entries=0, actual_T1_net_premium_rate=None, complete_trade_win_rate=None,
                  path_proxy_summary=path_summary(labels),
                  costs=COSTS, reserved_later_prices_used=False,
                  limitations=["daily proxy, no queue/timing/fills", "current-universe survivorship bias",
                               "historical status and calendar not independently reverified",
                               "mark returns are not executable realized profit"],
                  source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (args.observations,args.daily,args.calendar,Path(__file__))})
    (args.output/"manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
