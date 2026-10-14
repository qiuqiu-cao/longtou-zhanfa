"""Audit the candidate-to-fill funnel without equating triggers or touches to fills."""
import argparse
import hashlib
import json
from pathlib import Path

from leader_target_screen_v17 import timestamp, finite


CONFIRMATION_STATES = {"TRIGGERED", "CANCELLED", "EXPIRED", "DATA_LIMITED", "NO_TRIGGER"}
ORDER_STATES = {"SUBMITTED", "CANCELLED", "EXPIRED"}
DECISION_STATES = {"FOCUS", "WATCH", "REJECT"}


def _index(rows, key):
    result={}
    for row in rows:
        value=row.get(key)
        if not value or value in result:
            raise ValueError("missing or duplicate "+key)
        result[value]=row
    return result


def audit(manifests, decisions, confirmations, orders, fills, sessions, asof):
    cutoff=timestamp(asof)
    if sessions != sorted(set(sessions)) or not sessions:
        raise ValueError("sorted unique evaluation sessions required")
    manifest_map=_index(manifests,"trade_date")
    decision_map=_index(decisions,"decision_id")
    confirmation_map=_index(confirmations,"confirmation_id")
    order_map=_index(orders,"order_id")
    fill_map=_index(fills,"evidence_id") if fills else {}
    issues=[]
    kinds={row.get("kind") for row in manifests}
    account_kind=next(iter(kinds)) if len(kinds)==1 and next(iter(kinds)) in {"paper","user_confirmed"} else None
    if account_kind is None:
        issues.append("MISSING_OR_MIXED_ACCOUNT_KIND")
    complete_sessions=[]
    session_rows={day:[] for day in sessions}
    for decision in decisions:
        day=decision.get("trade_date")
        if day in session_rows:
            session_rows[day].append(decision)
        else:
            issues.append("DECISION_OUTSIDE_EVALUATION_SESSIONS_"+str(decision.get("decision_id")))
        try:
            frozen,available=timestamp(decision["frozen_at"]),timestamp(decision["available_at"])
            if (frozen>available or available>cutoff or decision.get("decision_state") not in DECISION_STATES
                    or decision.get("kind")!=account_kind):
                issues.append("INVALID_DECISION_"+str(decision.get("decision_id")))
            if decision.get("board_group") not in {"FIRST_BOARD","CONSECUTIVE_BOARD"}:
                issues.append("INVALID_DECISION_GROUP_"+str(decision.get("decision_id")))
            if not decision.get("source") or decision.get("data_status")!="verified":
                issues.append("UNVERIFIED_DECISION_"+str(decision.get("decision_id")))
            if decision.get("decision_state")!="FOCUS" and decision.get("plan_id"):
                issues.append("NONFOCUS_PLAN_"+str(decision.get("decision_id")))
        except (KeyError,TypeError,ValueError):
            issues.append("INVALID_DECISION_"+str(decision.get("decision_id")))
    for day in sessions:
        manifest=manifest_map.get(day)
        rows=session_rows[day]
        if manifest is None:
            issues.append("MISSING_SESSION_MANIFEST_"+day)
            continue
        try:
            expected=int(manifest["candidate_count"])
            codes=[str(row["code"]).zfill(6) for row in rows]
            observed=timestamp(manifest["observed_at"])
            available=timestamp(manifest["available_at"])
            valid=(expected>=0 and expected==len(rows) and len(codes)==len(set(codes))
                   and manifest.get("universe_complete") is True and manifest.get("data_status")=="verified"
                   and manifest.get("kind")==account_kind
                   and bool(manifest.get("source")) and observed<=available<=cutoff
                   and manifest.get("signal_date")==observed.strftime("%Y-%m-%d"))
            if not valid:
                issues.append("INCOMPLETE_OR_INVALID_SESSION_MANIFEST_"+day)
            else:
                complete_sessions.append(day)
        except (KeyError,TypeError,ValueError):
            issues.append("INCOMPLETE_OR_INVALID_SESSION_MANIFEST_"+day)
    plan_to_decision={}
    for row in decisions:
        plan=row.get("plan_id")
        if plan:
            if plan in plan_to_decision:
                issues.append("DUPLICATE_PLAN_DECISION_"+str(plan))
            plan_to_decision[plan]=row
    plan_confirmations={}
    confirmation_valid={}
    for confirmation in confirmations:
        cid=confirmation["confirmation_id"]
        plan=confirmation.get("plan_id")
        plan_confirmations.setdefault(plan,[]).append(confirmation)
        try:
            observed,available,expires=[timestamp(confirmation[key]) for key in ("observed_at","available_at","expires_at")]
            decision=plan_to_decision.get(plan)
            valid=(decision is not None and confirmation.get("status") in CONFIRMATION_STATES
                   and observed<=available<=cutoff and available<=expires
                   and confirmation.get("code")==decision.get("code") and confirmation.get("kind")==account_kind
                   and confirmation.get("data_status")=="verified" and bool(confirmation.get("source"))
                   and bool(confirmation.get("evidence_id")))
            confirmation_valid[cid]=valid
            if not valid:
                issues.append("INVALID_CONFIRMATION_"+str(cid))
        except (KeyError,TypeError,ValueError):
            confirmation_valid[cid]=False
            issues.append("INVALID_CONFIRMATION_"+str(cid))
    order_fills={order_id:[] for order_id in order_map}
    for fill in fill_map.values():
        order_id=fill.get("order_id")
        if order_id not in order_map:
            issues.append("FILL_WITHOUT_ORDER_"+str(fill.get("evidence_id")))
            continue
        order_fills[order_id].append(fill)
    order_results=[]
    fill_sessions=set()
    for order_id,order in order_map.items():
        local=[]
        confirmation=confirmation_map.get(order.get("confirmation_id"))
        decision=plan_to_decision.get(order.get("plan_id"))
        try:
            submitted,available,expires=[timestamp(order[key]) for key in ("submitted_at","available_at","expires_at")]
            shares=finite(order["shares"])
            if (confirmation is None or not confirmation_valid.get(order.get("confirmation_id"),False)
                    or confirmation.get("status")!="TRIGGERED"
                    or decision is None or order.get("status") not in ORDER_STATES
                    or submitted>available or available>cutoff or submitted>expires
                    or timestamp(confirmation["available_at"])>=submitted
                    or order.get("code")!=decision.get("code") or order.get("code")!=confirmation.get("code")
                    or order.get("kind")!=account_kind
                    or shares is None or shares<=0 or not shares.is_integer() or shares%100
                    or not order.get("source") or order.get("data_status")!="verified"
                    or not order.get("evidence_id")):
                local.append("INVALID_ORDER_OR_TRIGGER_SEQUENCE")
            ordered_shares=int(shares) if shares is not None and shares.is_integer() else None
        except (KeyError,TypeError,ValueError):
            ordered_shares=None
            expires=cutoff
            local.append("INVALID_ORDER_OR_TRIGGER_SEQUENCE")
        filled_shares=0
        for fill in sorted(order_fills.get(order_id,[]),key=lambda row:timestamp(row["filled_at"])):
            try:
                filled=timestamp(fill["filled_at"])
                known=timestamp(fill["available_at"])
                quantity=finite(fill["shares"])
                price=finite(fill["price"])
                if (fill.get("side")!="BUY" or fill.get("plan_id")!=order.get("plan_id")
                        or fill.get("code")!=order.get("code") or not submitted<filled<=expires
                        or filled.strftime("%Y-%m-%d") not in sessions
                        or (decision is not None and filled.strftime("%Y-%m-%d")!=decision.get("trade_date"))
                        or filled>known or known>cutoff or quantity is None or quantity<=0 or not quantity.is_integer()
                        or fill.get("kind")!=account_kind or fill.get("data_status")!="verified"
                        or price is None or price<=0 or not fill.get("source") or not fill.get("evidence_id")):
                    local.append("INVALID_FILL_"+str(fill.get("evidence_id")))
                else:
                    filled_shares+=int(quantity)
                    fill_sessions.add(filled.strftime("%Y-%m-%d"))
            except (KeyError,TypeError,ValueError):
                local.append("INVALID_FILL_"+str(fill.get("evidence_id")))
        if ordered_shares is not None and filled_shares>ordered_shares:
            local.append("OVERFILLED_ORDER")
        pending=ordered_shares is not None and filled_shares<ordered_shares and cutoff<=expires and order.get("status")=="SUBMITTED"
        if pending:
            fill_state="PENDING"
        elif filled_shares==0:
            fill_state="UNFILLED"
        elif ordered_shares is not None and filled_shares<ordered_shares:
            fill_state="PARTIAL"
        elif ordered_shares is not None and filled_shares==ordered_shares:
            fill_state="FULL"
        else:
            fill_state="INVALID"
        if local:
            fill_state="INVALID"
        issues.extend("ORDER_"+order_id+"_"+item for item in local)
        order_results.append(dict(order_id=order_id,plan_id=order.get("plan_id"),ordered_shares=ordered_shares,
                                  filled_shares=filled_shares,fill_state=fill_state,issues=local))
    triggered={row.get("plan_id") for row in confirmations
               if row.get("status")=="TRIGGERED" and confirmation_valid.get(row.get("confirmation_id"),False)}
    ordered_plans={row.get("plan_id") for row in orders}
    known_cash_days=[day for day in complete_sessions if day not in fill_sessions]
    submitted=len(order_results)
    completed=[row for row in order_results if row["fill_state"] in {"UNFILLED","PARTIAL","FULL"}]
    filled=[row for row in completed if row["fill_state"] in {"PARTIAL","FULL"}]
    ordered_total=sum(row["ordered_shares"] or 0 for row in completed)
    filled_total=sum(row["filled_shares"] for row in completed)
    manifest_candidate_count = sum(
        int(manifest["candidate_count"])
        for manifest in manifest_map.values()
        if isinstance(manifest.get("candidate_count"), (int, float))
        and int(manifest["candidate_count"]) >= 0
    )
    known_candidate_count = sum(
        int(manifest_map[day]["candidate_count"])
        for day in complete_sessions
        if isinstance(manifest_map[day].get("candidate_count"), (int, float))
    )
    return dict(status="PASS" if not issues else "ISSUES_FOUND",account_kind=account_kind,issues=sorted(set(issues)),
                # Keep the session-manifest denominator separate from the
                # decision rows.  A decision list can be filtered or absent;
                # it must not silently redefine the candidate universe.
                candidate_count=manifest_candidate_count,
                known_candidate_count=known_candidate_count,
                unknown_candidate_count=manifest_candidate_count-known_candidate_count,
                decision_count=len(decisions),
                focus_count=sum(row.get("decision_state")=="FOCUS" for row in decisions),
                planned_count=len(plan_to_decision),triggered_count=len(triggered),
                no_order_after_trigger_count=len(triggered-ordered_plans),submitted_order_count=submitted,
                pending_order_count=sum(row["fill_state"]=="PENDING" for row in order_results),
                unfilled_order_count=sum(row["fill_state"]=="UNFILLED" for row in order_results),
                partial_order_count=sum(row["fill_state"]=="PARTIAL" for row in order_results),
                full_order_count=sum(row["fill_state"]=="FULL" for row in order_results),
                order_fill_rate=len(filled)/len(completed) if completed else None,
                share_fill_rate=filled_total/ordered_total if ordered_total else None,
                buy_fill_event_count=sum(len(rows) for rows in order_fills.values()),
                executed_plan_count=len({row["plan_id"] for row in filled}),order_results=order_results,
                evaluation_session_count=len(sessions),complete_session_count=len(complete_sessions),
                unknown_session_count=len(sessions)-len(complete_sessions),known_cash_day_count=len(known_cash_days),
                cash_day_rate=len(known_cash_days)/len(complete_sessions) if complete_sessions else None,
                open_position_count=None,
                unresolved_position_count=None,
                exit_evidence_status="UNAVAILABLE_ENTRY_FUNNEL_ONLY",
                execution_independently_verified=False,touch_is_fill=False,
                limitation="sources and fills are caller declarations; pending orders are excluded from completed fill-rate denominators")


def run(input_path,output_path):
    source,target=Path(input_path),Path(output_path)
    if target.exists():
        raise FileExistsError("choose a new immutable funnel report")
    payload=json.loads(source.read_text(encoding="utf-8"))
    result=audit(**payload)
    result["source_hashes"]={str(source.resolve()):hashlib.sha256(source.read_bytes()).hexdigest(),
                             str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x",encoding="utf-8") as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.input,args.output),ensure_ascii=False,indent=2))
