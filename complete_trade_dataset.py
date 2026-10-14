"""Build policy-conditional complete-trade labels without labeling nontrades."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from leader_target_screen_v17 import timestamp, finite


TARGET_DEFINITION = "frozen_exit_complete_trade_net_profit_after_all_supplied_fees"
SELECTION_SCOPE = "all_mainboard_nonst_closing_limitups_first_and_consecutive"


def _index(rows,key,allow_none=False):
    result={}
    for row in rows:
        value=row.get(key)
        if value is None and allow_none:
            continue
        if not value or value in result:
            raise ValueError("missing or duplicate "+key)
        result[value]=row
    return result


def _feature_bundle(observation,allowlist,decision_at):
    values=observation.get("features",{})
    evidence=observation.get("feature_evidence",{})
    issues=[]
    selected={}
    for feature in allowlist:
        value=finite(values.get(feature))
        meta=evidence.get(feature,{})
        try:
            observed,available=timestamp(meta["observed_at"]),timestamp(meta["available_at"])
            valid=(value is not None and observed<=available<=decision_at and bool(meta.get("source"))
                   and bool(meta.get("definition_id")) and meta.get("data_status") in {"verified","inferred"})
        except (KeyError,TypeError,ValueError):
            valid=False
        if not valid:
            issues.append("FEATURE_UNAVAILABLE_OR_FUTURE_"+feature)
        else:
            selected[feature]=value
    extras=sorted(set(values)-set(allowlist))
    return selected,issues,extras


def build(observations,fills,account_report,funnel_report,policy,asof):
    cutoff=timestamp(asof)
    if account_report.get("kind") not in {"paper","user_confirmed"}:
        raise ValueError("explicit account kind required")
    kind=account_report["kind"]
    if funnel_report.get("account_kind")!=kind:
        raise ValueError("account and funnel kinds differ")
    try:
        frozen,available=timestamp(policy["frozen_at"]),timestamp(policy["available_at"])
        threshold=finite(policy["large_loss_threshold"])
        allowlist=policy["feature_allowlist"]
        valid_policy=(frozen<=available<=cutoff and bool(policy.get("policy_id"))
                      and policy.get("target_definition")==TARGET_DEFINITION
                      and policy.get("selection_scope")==SELECTION_SCOPE and threshold is not None
                      and abs(threshold-(-.07))<1e-12
                      and isinstance(allowlist,list) and allowlist and len(allowlist)==len(set(allowlist))
                      and all(isinstance(name,str) and name for name in allowlist))
    except (KeyError,TypeError,ValueError):
        valid_policy=False
        allowlist=[]
        threshold=None
    if not valid_policy:
        raise ValueError("invalid or unavailable frozen dataset policy")
    observation_map=_index(observations,"decision_id")
    closed=_index(account_report.get("closed_trades",[]),"plan_id")
    opened=_index(account_report.get("open_trades",[]),"plan_id")
    if set(closed)&set(opened):
        raise ValueError("plan cannot be open and closed")
    entry_by_plan={}
    for row in account_report.get("entry_plan_audit",{}).get("entries",[]):
        entry_by_plan.setdefault(row.get("plan_id"),[]).append(row)
    risk_by_plan={row.get("plan_id"):row for row in account_report.get("risk_budget_audit",{}).get("plans",[])}
    exit_by_plan={row.get("plan_id"):row for row in account_report.get("exit_plan_audit",{}).get("trades",[])}
    filled_plans={row.get("plan_id") for row in fills if row.get("side")=="BUY"}
    selected_plan_ids=[row.get("plan_id") for row in observations if row.get("plan_id")]
    if len(selected_plan_ids)!=len(set(selected_plan_ids)):
        raise ValueError("one plan cannot identify multiple candidate decisions")
    global_issues=[]
    selected_signal_dates=[row.get("signal_date") for row in observations if row.get("plan_id")]
    duplicate_signal_dates=sorted({day for day in selected_signal_dates if selected_signal_dates.count(day)>1})
    if duplicate_signal_dates:
        global_issues.append("MULTIPLE_SELECTED_CANDIDATES_SAME_SIGNAL_DATE:"+",".join(str(day) for day in duplicate_signal_dates))
    unmatched=(set(closed)|set(opened)|filled_plans)-set(selected_plan_ids)
    if unmatched:
        global_issues.append("ACCOUNT_OR_FILL_PLANS_WITHOUT_CANDIDATE:"+",".join(sorted(str(v) for v in unmatched)))
    if funnel_report.get("status")!="PASS":
        global_issues.append("EXECUTION_FUNNEL_NOT_PASSED")
    if account_report.get("status")!="ACCOUNTING_ONLY_NOT_STRATEGY_VALIDATED":
        global_issues.append("ACCOUNT_REPORT_STATUS_INVALID")
    rows=[]
    labels=[]
    for observation in observations:
        issues=[]
        decision_at=timestamp(observation["decision_at"])
        known=timestamp(observation["available_at"])
        if not (available<=decision_at<=cutoff and known<=decision_at):
            issues.append("POLICY_DECISION_AVAILABILITY_SEQUENCE_INVALID")
        boards=finite(observation.get("boards"))
        expected="FIRST_BOARD" if boards==1 else "CONSECUTIVE_BOARD" if boards is not None and boards>=2 and boards.is_integer() else None
        if (expected is None or observation.get("board_group")!=expected or observation.get("kind")!=kind
                or observation.get("data_status")!="verified" or not observation.get("source")):
            issues.append("INVALID_OBSERVATION_SCOPE_OR_SOURCE")
        if not isinstance(observation.get("emotion_active"),bool):
            issues.append("MISSING_FROZEN_EMOTION_ACTIVITY_FLAG")
        features,feature_issues,extras=_feature_bundle(observation,allowlist,decision_at)
        issues.extend(feature_issues)
        state=observation.get("selection_state")
        plan_id=observation.get("plan_id")
        if state not in {"FOCUS","WATCH","REJECT"} or (state=="FOCUS")!=(plan_id is not None):
            issues.append("SELECTION_PLAN_MISMATCH")
        if plan_id in closed:
            outcome_state="CLOSED"
            trade=closed[plan_id]
        elif plan_id in opened:
            outcome_state="OPEN"
            trade=opened[plan_id]
        elif plan_id in filled_plans:
            outcome_state="ACCOUNT_OUTCOME_MISSING"
            trade=None
            issues.append("FILLED_PLAN_ABSENT_FROM_ACCOUNT")
        elif plan_id:
            outcome_state="NOT_FILLED"
            trade=None
        else:
            outcome_state="NOT_SELECTED"
            trade=None
        entry_ok=bool(plan_id and entry_by_plan.get(plan_id)) and all(not row.get("issues") for row in entry_by_plan.get(plan_id,[]))
        risk_ok=bool(plan_id and risk_by_plan.get(plan_id)) and not risk_by_plan[plan_id].get("issues")
        exit_ok=bool(plan_id and exit_by_plan.get(plan_id)) and not exit_by_plan[plan_id].get("issues")
        funnel_ok=funnel_report.get("status")=="PASS"
        account_ok=account_report.get("status")=="ACCOUNTING_ONLY_NOT_STRATEGY_VALIDATED"
        if outcome_state=="CLOSED" and not entry_ok:
            issues.append("ENTRY_AUDIT_NOT_PASSED")
        if outcome_state=="CLOSED" and not risk_ok:
            issues.append("RISK_BUDGET_AUDIT_NOT_PASSED")
        if outcome_state=="CLOSED" and not exit_ok:
            issues.append("EXIT_AUDIT_NOT_PASSED")
        eligible=outcome_state=="CLOSED" and not issues and entry_ok and risk_ok and exit_ok and funnel_ok and account_ok
        net_return=finite(trade.get("net_return")) if trade else None
        if outcome_state=="CLOSED":
            try:
                if timestamp(trade["opened_at"])<=decision_at or timestamp(trade["closed_at"])>cutoff or net_return is None:
                    issues.append("INVALID_COMPLETE_TRADE_TIME_OR_RETURN")
                    eligible=False
            except (KeyError,TypeError,ValueError):
                issues.append("INVALID_COMPLETE_TRADE_TIME_OR_RETURN")
                eligible=False
        profit_label=bool(net_return>0) if eligible else None
        loss_label=bool(net_return<=threshold) if eligible else None
        if eligible:
            labels.append(dict(group=expected,profit=profit_label,large_loss=loss_label,
                               net_return=net_return,closed_at=trade["closed_at"]))
        rows.append(dict(decision_id=observation["decision_id"],plan_id=plan_id,code=str(observation.get("code","")).zfill(6),
                         signal_date=observation.get("signal_date"),trade_date=observation.get("trade_date"),
                         boards=int(boards) if boards is not None and boards.is_integer() else None,board_group=expected,
                         emotion_active=observation.get("emotion_active"),
                         selection_state=state,outcome_state=outcome_state,training_eligible=eligible,
                         closed_at=trade.get("closed_at") if trade and outcome_state=="CLOSED" else None,
                         complete_trade_profit_label=profit_label,large_loss_label=loss_label,
                         complete_trade_net_return=net_return if outcome_state=="CLOSED" else None,
                         selected_features=features,ignored_unregistered_features=extras,issues=sorted(set(issues))))
    def metrics(group=None):
        data=[row for row in labels if group is None or row["group"]==group]
        winners=[row["net_return"] for row in data if row["net_return"]>0]
        losers=[row["net_return"] for row in data if row["net_return"]<0]
        ordered=sorted(data,key=lambda row:timestamp(row["closed_at"]))
        streak=maximum=0
        for row in ordered:
            streak=streak+1 if row["net_return"]<0 else 0
            maximum=max(maximum,streak)
        average_win=sum(winners)/len(winners) if winners else None
        average_loss=sum(losers)/len(losers) if losers else None
        return dict(completed_trades=len(data),win_rate=sum(row["profit"] for row in data)/len(data) if data else None,
                    big_loss_rate=sum(row["large_loss"] for row in data)/len(data) if data else None,
                    mean_net_return=sum(row["net_return"] for row in data)/len(data) if data else None,
                    average_winning_return=average_win,average_losing_return=average_loss,
                    payoff_ratio=average_win/abs(average_loss) if average_win is not None and average_loss is not None else None,
                    profit_factor=sum(winners)/abs(sum(losers)) if losers else None,
                    profit_factor_status="DEFINED" if losers else "NO_LOSS_DENOMINATOR",
                    worst_net_return=min((row["net_return"] for row in data),default=None),
                    maximum_consecutive_losses=maximum)
    complete_sessions=(funnel_report.get("complete_session_count")==funnel_report.get("evaluation_session_count")
                       and funnel_report.get("evaluation_session_count",0)>0)
    total=len(labels)
    group_metrics={group:metrics(group) for group in ("FIRST_BOARD","CONSECUTIVE_BOARD")}
    observation_issue_count=sum(bool(row["issues"]) for row in rows)
    training_ready=(not global_issues and observation_issue_count==0 and total>=100
                    and all(group_metrics[group]["completed_trades"]>0 for group in group_metrics)
                    and complete_sessions and account_report.get("drawdown_status")=="COMPLETE_SUPPLIED_SESSIONS")
    return dict(status="POLICY_CONDITIONAL_DATASET_READY" if training_ready else "ISSUES_FOUND" if global_issues else "DIAGNOSTIC_ONLY",
                policy_id=policy["policy_id"],kind=kind,feature_allowlist=allowlist,
                target_definition=TARGET_DEFINITION,selection_scope=SELECTION_SCOPE,
                rows=rows,all_observation_count=len(rows),
                selected_count=sum(row["selection_state"]=="FOCUS" for row in rows),
                completed_eligible_count=total,open_count=sum(row["outcome_state"]=="OPEN" for row in rows),
                not_filled_count=sum(row["outcome_state"]=="NOT_FILLED" for row in rows),
                overall_metrics=metrics(),group_metrics=group_metrics,training_ready=training_ready,
                account_max_drawdown=account_report.get("account_max_drawdown"),
                drawdown_status=account_report.get("drawdown_status"),global_issues=global_issues,
                observation_issue_count=observation_issue_count,
                generalization_scope="FROZEN_SELECTION_POLICY_CONDITIONAL_ONLY",
                nonselected_rows_are_negative_labels=False,unfilled_rows_are_negative_labels=False,
                can_rank_unselected_full_pool=False,blind_test_status="NOT_ESTABLISHED_BY_DATASET_BUILDER",
                execution_independently_verified=False,
                limitations=["selected/fill outcomes identify policy-conditional performance, not counterfactual outcomes for unselected stocks",
                             "the one-stock policy permits at most one selected plan per signal date",
                             "paper and user-confirmed datasets remain separate",
                             "model training still requires chronological split, purge, calibration and an untouched blind period"])


def run(input_path,output_path):
    source,target=Path(input_path),Path(output_path)
    if target.exists():
        raise FileExistsError("choose a new immutable complete-trade dataset report")
    payload=json.loads(source.read_text(encoding="utf-8"))
    result=build(**payload)
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
