"""Conditional T+1 stress scenario, not a verified execution backtest."""
import argparse
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path
import pandas as pd


def net_return(entry, exit_price, shares, commission, minimum, sell_tax, slippage):
    values = [entry,exit_price,shares,commission,minimum,sell_tax,slippage]
    if not all(math.isfinite(float(v)) for v in values) or min(entry,exit_price,shares) <= 0:
        raise ValueError("finite positive prices and shares required")
    if shares % 100 or min(commission,minimum,sell_tax,slippage) < 0 or max(commission,sell_tax,slippage) >= 1:
        raise ValueError("invalid fee assumptions or lot size")
    # Entry at submitted limit, no invented favorable price improvement.
    cost = entry*shares + max(minimum,entry*shares*commission)
    sell = exit_price*(1-slippage)*shares
    proceeds = sell-max(minimum,sell*commission)-sell*sell_tax
    return proceeds/cost-1


def evaluate(row, next_day, daily, costs):
    base = dict(code=row["code"],entry_date=row["entry_date"],next_session=next_day,
                data_status=row["data_status"],verified_fill=False,
                net_t1_mark_return=None,large_loss_proxy=None,realized_return=None)
    if "observation" not in row:
        return dict(base,state="SIGNAL_DATA_UNAVAILABLE")
    if row["observation"].get("state") != "CLOSE_AT_LIMIT_PROXY":
        return dict(base,state="NO_FIXED_OPENING_SIGNAL")
    execution = row["execution_LABEL_ONLY"]["state"]
    if execution != "BELOW_LIMIT_TRADES_OPPORTUNITY_ONLY":
        return dict(base,state="ENTRY_UNRESOLVED",execution_evidence=execution)
    if next_day is None or daily is None:
        return dict(base,state="EXIT_DATA_MISSING")
    if next_day <= row["entry_date"]:
        raise ValueError("T+1 requires a later exchange session")
    o,h,l,c,p,v = [float(daily[k]) for k in ("open","high","low","close","pre_close","volume")]
    if not all(math.isfinite(x) for x in (o,h,l,c,p,v)) or min(o,h,l,c,p) <= 0 or v < 0 or l > min(o,c) or h < max(o,c) or l > h:
        return dict(base,state="EXIT_DATA_INVALID")
    entry = row["limit_price_proxy"]
    mark = net_return(entry,c,**costs)
    down = float((Decimal(str(p))*Decimal("0.90")).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP))
    # Closing at down-limit does not prove an exit, even when earlier trades exist.
    blocked = v == 0 or c <= down+1e-6
    return dict(base,state="CONDITIONAL_ENTRY_EXIT_UNRESOLVED" if blocked else "CONDITIONAL_T1_MARK_SCENARIO",
                assumed_entry_price=entry,t1_close=c,net_t1_mark_return=mark,
                large_loss_proxy=mark <= -0.07,exit_execution_confirmed=False,
                assumption="hypothetical entry at limit after below-limit trades; no verified latency or queue")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("replay","daily","calendar","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    # Explicit scenario inputs, not a claim about a user's brokerage fees.
    costs = dict(shares=100,commission=0.0003,minimum=5.0,sell_tax=0.0005,slippage=0.001)
    protocol = dict(status="DIAGNOSTIC_ONLY",cost_assumptions=costs,
                    order_rule="09:35 close-at-limit proxy -> hypothetical limit order, valid next 5-minute bar only",
                    exit_rule="T+1 close mark with hypothetical costs; not proof of closing auction fill",
                    portfolio=False,live_signal=False,threshold_tuning=False)
    (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    calendar = pd.read_csv(args.calendar).date.tolist()
    if calendar != sorted(set(calendar)):
        raise ValueError("invalid calendar")
    successors = dict(zip(calendar[:-1],calendar[1:]))
    daily = pd.read_csv(args.daily,dtype={"code":str}).set_index(["code","date"])
    if not daily.index.is_unique:
        raise ValueError("duplicate daily bars")
    rows = []
    for row in replay["rows"]:
        day = successors.get(row["entry_date"])
        key = (row["code"],day)
        rows.append(evaluate(row,day,daily.loc[key,:] if key in daily.index else None,costs))
    marked = [r for r in rows if r["net_t1_mark_return"] is not None]
    report = dict(protocol,rows=rows,states=dict(Counter(r["state"] for r in rows)),
                  hypothetical_marked_cases=len(marked),
                  hypothetical_positive_marks=sum(r["net_t1_mark_return"]>0 for r in marked),
                  hypothetical_large_loss_marks=sum(r["large_loss_proxy"] for r in marked),
                  verified_trades=0,validated_strategy_win_rate=None,portfolio_max_drawdown=None,
                  source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.replay,args.daily,args.calendar,Path(__file__))},
                  limitations=["OHLC discrepancies and hypothetical entries remain unresolved",
                               "post-decision opportunity is outcome, never a pre-entry selection filter",
                               "independent 100-share scenarios, not a capital-constrained strategy",
                               "slippage/fees are illustrative; no broker fee certification or complete transaction-cost model"])
    (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows","source_hashes")},indent=2))


if __name__ == "__main__":
    main()
