"""Accounting for caller-confirmed fills, with open-position daily valuation.

Fees are incremental charged cash supplied per fill (not a fee estimator).
The caller must verify fills, dated fees, plans and session calendar separately.
"""
import math
import argparse
import hashlib
import json
from pathlib import Path
from leader_target_screen_v17 import timestamp
from entry_plan_audit import audit as audit_entry_plans
from risk_budget_audit import audit as audit_risk_budget
from exit_plan_audit import audit as audit_exit_plans


def number(value, positive=False):
    value=float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("invalid nonnegative amount")
    return value


def completed_statistics(closed):
    ordered=sorted(closed,key=lambda t:(t["closed_at"],t["trade_id"]))
    winners=[t["net_profit"] for t in ordered if t["net_profit"]>0]
    losers=[t["net_profit"] for t in ordered if t["net_profit"]<0]
    streak=maximum=0
    for t in ordered:
        streak=streak+1 if t["net_profit"]<0 else 0
        maximum=max(maximum,streak)
    win=sum(winners)/len(winners) if winners else None
    loss=sum(losers)/len(losers) if losers else None
    return dict(completed_trades=len(ordered),winning_trades=len(winners),losing_trades=len(losers),
                breakeven_trades=len(ordered)-len(winners)-len(losers),
                average_winning_cash=win,average_losing_cash=loss,
                cash_payoff_ratio=win/abs(loss) if win is not None and loss is not None else None,
                profit_factor=sum(winners)/abs(sum(losers)) if losers else None,
                profit_factor_status="DEFINED" if losers else "NO_REALIZED_LOSSES_DENOMINATOR",
                maximum_consecutive_losses=maximum,
                worst_completed_net_return=min((t["net_return"] for t in ordered),default=None),
                ordering="exit_timestamp_then_trade_id",open_trades_excluded_from_completed_statistics=True)


def account(initial_cash, fills, marks, sessions, kind, asof, plans=None, confirmations=None,
            risk_policy=None, exit_confirmations=None):
    if kind not in {"paper", "user_confirmed"}:
        raise ValueError("account kind required")
    cutoff=timestamp(asof)
    if sessions != sorted(set(sessions)) or not sessions:
        raise ValueError("sorted unique sessions required")
    cash=number(initial_cash,True)
    initial=cash
    ids=set()
    events=[]
    for f in fills:
        if f["kind"] != kind or not f.get("source") or not f.get("evidence_id") or not f.get("fee_evidence_id"):
            raise ValueError("account/source/fee evidence mismatch")
        if f["evidence_id"] in ids:
            raise ValueError("duplicate fill")
        ids.add(f["evidence_id"])
        t=timestamp(f["filled_at"])
        known=timestamp(f["available_at"])
        if known<t or known>cutoff or t>cutoff:
            raise ValueError("unavailable fill")
        day=t.strftime("%Y-%m-%d")
        if day not in sessions:
            raise ValueError("fill session absent")
        if f["side"] not in {"BUY","SELL"}:
            raise ValueError("invalid side")
        quantity=number(f["shares"],True)
        if not quantity.is_integer() or (f["side"]=="BUY" and quantity%100):
            raise ValueError("invalid share count")
        events.append((t,f,day,int(quantity),number(f["price"],True),number(f["fee_cash"])))
    events.sort(key=lambda r:(r[0],r[1]["evidence_id"]))
    if len({r[0] for r in events}) != len(events):
        raise ValueError("same-time fills require finer execution timestamps")
    quotes={}
    for m in marks:
        key=(m["date"],m["code"])
        if key in quotes:
            raise ValueError("duplicate mark")
        quotes[key]=m
    trades={}
    curve=[]
    peak=initial
    worst=0.
    complete_curve=True
    index=0
    for day in sessions:
        close_time=timestamp(day+"T15:00:00+08:00")
        if close_time>cutoff:
            break
        while index<len(events) and events[index][0]<=close_time:
            t,f,buyday,qty,price,fee=events[index]
            index+=1
            tid=f["trade_id"]
            if tid not in trades:
                if f["side"] != "BUY":
                    raise ValueError("sale without trade")
                if not f.get("plan_id"):
                    raise ValueError("plan reference required")
                trades[tid]=dict(trade_id=tid,code=f["code"],plan_id=f["plan_id"],opened_at=t.isoformat(),lots=[],
                                 cost=0.,proceeds=0.,closed_at=None)
            trade=trades[tid]
            if trade["code"]!=f["code"] or trade["plan_id"]!=f["plan_id"] or trade["closed_at"]:
                raise ValueError("trade identity changed or closed trade reused")
            if f["side"]=="BUY":
                debit=qty*price+fee
                if debit>cash+1e-8:
                    raise ValueError("insufficient cash")
                cash-=debit
                trade["cost"]+=debit
                trade["lots"].append([buyday,qty])
            else:
                sellable=sum(q for d,q in trade["lots"] if d<buyday)
                if qty>sellable:
                    raise ValueError("T+1 or oversell violation")
                remaining=qty
                for lot in trade["lots"]:
                    if lot[0]<buyday:
                        take=min(lot[1],remaining)
                        lot[1]-=take
                        remaining-=take
                credit=qty*price-fee
                cash+=credit
                trade["proceeds"]+=credit
                if sum(q for _,q in trade["lots"])==0:
                    trade["closed_at"]=t.isoformat()
        equity=cash
        missing=[]
        for trade in trades.values():
            qty=sum(q for _,q in trade["lots"])
            if not qty:
                continue
            m=quotes.get((day,trade["code"]))
            valid=False
            if m:
                try:
                    valid=(m.get("data_status")=="verified" and bool(m.get("source"))
                           and timestamp(m["observed_at"])==close_time
                           and close_time<=timestamp(m["available_at"])<=cutoff
                           and m.get("price_basis")=="unadjusted_no_corporate_action")
                    price=number(m["close"],True)
                except (ValueError,TypeError,KeyError):
                    valid=False
            if valid:
                equity+=qty*price
            else:
                missing.append(trade["code"])
        if missing:
            complete_curve=False
            curve.append(dict(date=day,cash=cash,equity=None,missing_codes=sorted(set(missing))))
        else:
            peak=max(peak,equity)
            worst=min(worst,equity/peak-1)
            curve.append(dict(date=day,cash=cash,equity=equity,missing_codes=[]))
    if index<len(events):
        raise ValueError("close-only ledger requires cutoff after all included fill sessions close")
    closed=[]
    opened=[]
    for t in trades.values():
        entry=dict(trade_id=t["trade_id"],code=t["code"],plan_id=t["plan_id"],
                   opened_at=t["opened_at"],
                   remaining_shares=sum(q for _,q in t["lots"]),entry_cost=t["cost"],sale_proceeds=t["proceeds"])
        if t["closed_at"]:
            entry.update(closed_at=t["closed_at"],net_profit=t["proceeds"]-t["cost"],
                         net_return=t["proceeds"]/t["cost"]-1)
            closed.append(entry)
        else:
            opened.append(entry)
    return dict(status="ACCOUNTING_ONLY_NOT_STRATEGY_VALIDATED",kind=kind,closed_trades=closed,
                entry_plan_audit=audit_entry_plans(fills,plans or [],confirmations or []),
                risk_budget_audit=audit_risk_budget(plans or [],risk_policy,asof),
                exit_plan_audit=audit_exit_plans(fills,plans or [],exit_confirmations or [],sessions,asof),
                completed_statistics=completed_statistics(closed),open_trade_count=len(opened),
                open_trades=opened,equity_curve=curve,
                completed_trade_win_rate=sum(t["net_profit"]>0 for t in closed)/len(closed) if closed else None,
                mean_completed_net_return=sum(t["net_return"] for t in closed)/len(closed) if closed else None,
                account_max_drawdown=worst if complete_curve else None,
                drawdown_status="COMPLETE_SUPPLIED_SESSIONS" if complete_curve else "MISSING_OPEN_POSITION_MARKS",
                execution_verified=False, frozen_exit_rule_compliance_verified=False,
                limitation="supplied sessions/fills/fees; no deposits, dividends or corporate actions; marks exclude hypothetical liquidation fees")


def run(input_path, output_path):
    input_path,output_path=Path(input_path),Path(output_path)
    if output_path.exists():
        raise FileExistsError("choose a new immutable accounting report path")
    payload=json.loads(input_path.read_text(encoding="utf-8"))
    report=account(**payload)
    report["asof"]=timestamp(payload["asof"]).isoformat()
    report["source_hashes"]={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (input_path,Path(__file__),Path(__file__).with_name("entry_plan_audit.py"),
                                       Path(__file__).with_name("opening_minute_measurements.py"))}
    risk_path=Path(__file__).with_name("risk_budget_audit.py")
    report["source_hashes"][str(risk_path.resolve())]=hashlib.sha256(risk_path.read_bytes()).hexdigest()
    exit_path=Path(__file__).with_name("exit_plan_audit.py")
    report["source_hashes"][str(exit_path.resolve())]=hashlib.sha256(exit_path.read_bytes()).hexdigest()
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open("x",encoding="utf-8") as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    report=run(args.input,args.output)
    print(json.dumps(report["completed_statistics"],ensure_ascii=False,indent=2))
