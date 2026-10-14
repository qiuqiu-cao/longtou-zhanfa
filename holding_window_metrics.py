"""B+1/2/3 close marks from a supplied fill and dated cost assumptions.

The caller must establish execution separately. Marks are never realized exits.
Only closes observed by evaluation time are consumed; calendar is supplied.
"""
import math
import argparse
import hashlib
import json
from pathlib import Path

from leader_target_screen_v17 import timestamp
from t1_scenario_v44 import net_return


def evaluate(fill, bars, calendar, asof, costs):
    cutoff = timestamp(asof)
    entered = timestamp(fill["filled_at"])
    if entered > cutoff or timestamp(fill["available_at"]) > cutoff:
        raise ValueError("fill unavailable at evaluation time")
    if timestamp(fill["available_at"]) < entered:
        raise ValueError("fill availability precedes execution")
    if fill.get("kind") not in {"paper_confirmed_fill", "user_confirmed_fill"}:
        raise ValueError("explicit paper or user-confirmed fill required")
    if not fill.get("evidence_id") or not fill.get("source"):
        raise ValueError("fill evidence reference and source required")
    if calendar != sorted(set(calendar)):
        raise ValueError("calendar must be sorted and unique")
    bought = entered.strftime("%Y-%m-%d")
    if bought not in calendar:
        raise ValueError("fill session missing from calendar")
    if not costs.get("source") or not costs.get("assumption_id"):
        raise ValueError("dated cost assumption required")
    if not costs["effective_from"] <= bought <= costs["effective_to"]:
        raise ValueError("entry outside cost assumption period")
    entry, shares = float(fill["price"]), int(fill["shares"])
    if shares != float(fill["shares"]):
        raise ValueError("fractional share count")
    fee = {key: costs[key] for key in ("commission", "minimum", "sell_tax", "slippage")}
    # Validate even when no future session has closed.
    net_return(entry, entry, shares, **fee)
    by_day = {}
    for row in bars:
        if row["code"] != fill["code"]:
            raise ValueError("price row belongs to another stock")
        if row["date"] in by_day:
            raise ValueError("duplicate closing date")
        by_day[row["date"]] = row
    results = []
    start = calendar.index(bought)
    for offset in (1, 2, 3):
        day = calendar[start+offset] if start+offset < len(calendar) else None
        result = dict(horizon=offset, session=day, state="SESSION_UNAVAILABLE",
                      net_close_mark=None, realized_profit=None)
        if day is None:
            results.append(result)
            continue
        close_time = timestamp(day+"T15:00:00+08:00")
        if close_time > cutoff:
            result["state"] = "NOT_YET_OBSERVABLE"
        elif not costs["effective_from"] <= day <= costs["effective_to"]:
            result["state"] = "COST_SCHEDULE_UNAVAILABLE"
        elif day not in by_day:
            result["state"] = "MISSING_CLOSE"
        else:
            row = by_day[day]
            try:
                known, observed = timestamp(row["available_at"]), timestamp(row["observed_at"])
                price = float(row["close"])
                valid = (observed == close_time and observed <= known <= cutoff
                         and row.get("data_status") == "verified" and bool(row.get("source"))
                         and row.get("price_basis") == fill.get("price_basis") == "unadjusted_no_corporate_action"
                         and math.isfinite(price) and price > 0)
            except (ValueError, TypeError, KeyError):
                valid = False
            if valid:
                result.update(state="CLOSE_MARK_ONLY", net_close_mark=net_return(entry, price, shares, **fee))
            else:
                result["state"] = "INVALID_OR_UNAVAILABLE_CLOSE"
        results.append(result)
    later = [r["net_close_mark"] for r in results if r["horizon"] in (2,3)]
    return dict(code=fill["code"], buy_session=bought, fill_kind=fill["kind"],
                fill_evidence_id=fill["evidence_id"], execution_independently_verified=False,
                status="VALUATION_ONLY", cost_assumption_id=costs["assumption_id"], horizons=results,
                both_B2_B3_positive=None if any(x is None for x in later) else all(x > 0 for x in later),
                full_trade_profit=None, mfe=None, mae=None,
                excursion_status="REQUIRES_POST_FILL_PATH_DATA",
                limitation="caller-confirmed fill; calendar and source declarations not independently verified")


def batch(records, calendar, asof):
    """Evaluate fill lots, keeping unavailable lots in coverage denominators."""
    cutoff = timestamp(asof)
    if (not calendar.get("source") or calendar.get("data_status") != "verified"
            or calendar.get("complete") is not True
            or timestamp(calendar.get("available_at")) > cutoff):
        raise ValueError("sourced calendar must be available and declared complete")
    ids = [r["fill"]["evidence_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate fill evidence IDs would inflate sample count")
    evaluated = [evaluate(r["fill"], r["bars"], calendar["sessions"], asof, r["costs"])
                 for r in records]
    summaries = {}
    for kind in ("paper_confirmed_fill", "user_confirmed_fill"):
        group = [r for r in evaluated if r["fill_kind"] == kind]
        summary = dict(fill_lots=len(group), horizons={})
        for horizon in (1,2,3):
            marks = [r["horizons"][horizon-1]["net_close_mark"] for r in group]
            available = [v for v in marks if v is not None]
            summary["horizons"][str(horizon)] = dict(
                observed=len(available), unresolved=len(group)-len(available),
                mean_net_close_mark=sum(available)/len(available) if available else None,
                positive_fraction=sum(v > 0 for v in available)/len(available) if available else None)
        both = [r["both_B2_B3_positive"] for r in group if r["both_B2_B3_positive"] is not None]
        summary["B2_B3_joint"] = dict(observed=len(both), unresolved=len(group)-len(both),
                                     positive_fraction=sum(both)/len(both) if both else None)
        summaries[kind] = summary
    return dict(asof=cutoff.isoformat(), status="VALUATION_ONLY", records=evaluated,
                summaries=summaries, unit_of_analysis="fill_lot_not_completed_trade",
                full_trade_win_rate=None, account_drawdown=None,
                limitation="fill lots may belong to the same trade; no independent-trade inference or realized P&L")


def run(input_path, calendar_path, asof, output_path):
    input_path, calendar_path, output_path = map(Path,(input_path,calendar_path,output_path))
    if output_path.exists():
        raise FileExistsError("valuation reports are immutable; choose a new output")
    records=json.loads(input_path.read_text(encoding="utf-8"))
    calendar=json.loads(calendar_path.read_text(encoding="utf-8"))
    result=batch(records,calendar,asof)
    result["source_hashes"]={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (input_path,calendar_path,Path(__file__))}
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open("x",encoding="utf-8") as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,help="JSON list of fill/bars/costs records")
    parser.add_argument("--calendar",required=True,help="timestamped trading calendar JSON")
    parser.add_argument("--asof",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    report=run(args.input,args.calendar,args.asof,args.output)
    print(json.dumps(report["summaries"],ensure_ascii=False,indent=2,allow_nan=False))
