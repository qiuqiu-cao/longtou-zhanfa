"""Audit submitted plan/confirmation evidence; never infer a fill or remove one."""
from leader_target_screen_v17 import timestamp, finite
import operator
from opening_minute_measurements import derive as derive_minutes, derive_absorption


def numeric_condition(rule, evidence, confirmed_at):
    """Compare a dated measurement to a frozen scalar rule; no expression eval."""
    try:
        value,threshold=finite(evidence["value"]),finite(rule["threshold"])
        maximum_age=finite(rule["max_age_seconds"])
        if value is None or threshold is None or maximum_age is None or maximum_age<0:
            raise ValueError("invalid numeric condition")
        if isinstance(evidence["value"],bool) or isinstance(rule["threshold"],bool):
            raise ValueError("boolean is not a measurement")
        for field in ("metric","unit","window_seconds","definition_id"):
            if evidence[field]!=rule[field] or rule[field] is None:
                raise ValueError("measurement definition mismatch")
        window=finite(rule["window_seconds"])
        if window is None or window<0:
            raise ValueError("invalid window")
        observed,known=timestamp(evidence["observed_at"]),timestamp(evidence["available_at"])
        end=timestamp(confirmed_at)
        if observed>known or known>end or (end-observed).total_seconds()>maximum_age:
            raise ValueError("measurement unavailable or stale")
        if evidence.get("data_status")!="verified" or not evidence.get("source") or not evidence.get("evidence_id"):
            raise ValueError("measurement evidence required")
        op={">":operator.gt,">=":operator.ge,"<":operator.lt,"<=":operator.le,"==":operator.eq}[rule["operator"]]
        return dict(status="PASS" if op(value,threshold) else "FAIL",value=value,threshold=threshold,
                    evidence_id=evidence["evidence_id"])
    except (KeyError,TypeError,ValueError):
        return dict(status="UNAVAILABLE",value=None,threshold=None)


def audit(fills, plans, confirmations):
    def index(rows, key):
        result={}
        for row in rows:
            if row[key] in result:
                raise ValueError("duplicate evidence key")
            result[row[key]]=row
        return result
    plan_map=index(plans,"plan_id")
    confirmation_map=index(confirmations,"confirmation_id")
    cumulative={}
    results=[]
    for fill in sorted(fills,key=lambda f:timestamp(f["filled_at"])):
        if fill["side"]!="BUY":
            continue
        issues=[]
        measured={}
        plan=plan_map.get(fill.get("plan_id"))
        confirm=confirmation_map.get(fill.get("confirmation_id"))
        key=fill.get("plan_id")
        cumulative[key]=cumulative.get(key,0)+float(fill["shares"])
        if plan is None:
            issues.append("MISSING_PLAN")
        if confirm is None:
            issues.append("MISSING_CONFIRMATION")
        if plan is not None and confirm is not None:
            try:
                frozen=timestamp(plan["frozen_at"])
                known=timestamp(plan["available_at"])
                observed=timestamp(confirm["observed_at"])
                confirmed=timestamp(confirm["available_at"])
                executed=timestamp(fill["filled_at"])
                if not frozen<=known<=observed<=confirmed<executed:
                    issues.append("INVALID_PLAN_CONFIRM_FILL_SEQUENCE")
                start,end=timestamp(plan["entry_start"]),timestamp(plan["entry_end"])
                if start>end or not start<=observed<=confirmed<executed<=end:
                    issues.append("OUTSIDE_ENTRY_WINDOW")
                if executed>timestamp(confirm["expires_at"]):
                    issues.append("CONFIRMATION_EXPIRED")
                if plan["code"]!=fill["code"] or confirm["code"]!=fill["code"] or confirm["plan_id"]!=key:
                    issues.append("IDENTITY_MISMATCH")
                if plan.get("kind")!=fill["kind"] or confirm.get("kind")!=fill["kind"]:
                    issues.append("ACCOUNT_KIND_MISMATCH")
                if not plan.get("source") or not confirm.get("source") or confirm.get("data_status")!="verified":
                    issues.append("UNVERIFIED_SOURCE")
                required=plan["required_condition_ids"]
                conditions=confirm["conditions"]
                if not required or len(required)!=len(set(required)) or any(conditions.get(c) is not True for c in required):
                    issues.append("REQUIRED_CONDITIONS_NOT_CONFIRMED")
                for condition_id in required:
                    rule=plan.get("condition_rules",{}).get(condition_id,{})
                    evidence=confirm.get("measurements",{}).get(condition_id,{})
                    if rule.get("evidence_builder")=="opening_minutes_v1":
                        try:
                            evidence=derive_minutes(confirm.get("minute_rows",[]),fill["code"],
                                observed.strftime("%Y-%m-%d"),confirm["available_at"]).get(rule.get("metric"),{})
                        except (KeyError,TypeError,ValueError):
                            evidence={}
                    elif rule.get("evidence_builder")=="opening_absorption_15m_v1":
                        try:
                            evidence=derive_absorption(confirm.get("minute_rows",[]),fill["code"],
                                observed.strftime("%Y-%m-%d"),confirm["available_at"]).get(rule.get("metric"),{})
                        except (KeyError,TypeError,ValueError):
                            evidence={}
                    measured[condition_id]=numeric_condition(
                        rule,evidence,confirm["available_at"])
                if any(r["status"]!="PASS" for r in measured.values()):
                    issues.append("NUMERIC_CONDITION_EVIDENCE_NOT_PASSED")
                lower,upper,maximum=[finite(plan[k]) for k in ("min_price","max_price","max_shares")]
                if any(v is None or v<=0 for v in (lower,upper,maximum)) or lower>upper or maximum%100:
                    issues.append("INVALID_PLAN_BOUNDS")
                else:
                    if not lower<=float(fill["price"])<=upper:
                        issues.append("FILL_OUTSIDE_PRICE_BOUNDS")
                    if cumulative[key]>maximum:
                        issues.append("CUMULATIVE_PLAN_QUANTITY_EXCEEDED")
                if confirm.get("cancelled") is not False:
                    issues.append("CANCELLATION_NOT_CLEARED")
            except (KeyError,TypeError,ValueError):
                issues.append("INVALID_OR_MISSING_AUDIT_FIELDS")
        results.append(dict(evidence_id=fill["evidence_id"],plan_id=key,issues=issues,condition_audit=measured,
                            status="ISSUES_FOUND" if issues else "DECLARED_ENTRY_RULES_MATCH",
                            execution_verified=False))
    return dict(entries=results,issue_count=sum(bool(r["issues"]) for r in results),
                scope="declared entry conditions, time, price and cumulative quantity only",
                exit_rule_compliance="NOT_AUDITED",strategy_win_rate=None,
                limitation="condition truth and original freeze time supplied by caller; no independent archive or order-book verification")
