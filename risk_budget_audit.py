"""Audit frozen plan sizing against explicitly supplied pre-trade risk budgets."""
import math
from leader_target_screen_v17 import finite, timestamp


def audit(plans, policy, asof):
    if policy is None:
        return dict(status="UNAVAILABLE",plans=[],limitation="no frozen risk policy supplied")
    cutoff=timestamp(asof)
    try:
        if (policy["data_status"]!="verified" or not policy["source"] or not policy["policy_id"]
                or timestamp(policy["available_at"])>cutoff):
            raise ValueError("invalid policy evidence")
        risk_cap,gross_cap,theme_cap=[finite(policy[k]) for k in
                                     ("single_trade_risk_fraction","gross_exposure_fraction","theme_exposure_fraction")]
        if any(v is None or not 0<v<=1 for v in (risk_cap,gross_cap,theme_cap)):
            raise ValueError("invalid policy limits")
    except (KeyError,TypeError,ValueError):
        return dict(status="UNAVAILABLE",plans=[],limitation="invalid or unavailable risk policy")
    results=[]
    snapshot_keys=[]
    for plan in plans:
        issues=[]
        try:
            frozen,known=timestamp(plan["frozen_at"]),timestamp(plan["available_at"])
            if not frozen<=known<=cutoff or plan.get("risk_policy_id")!=policy["policy_id"]:
                issues.append("POLICY_OR_TIME_MISMATCH")
            equity,existing_gross,existing_theme,stress,max_price,max_shares=[finite(plan[k]) for k in
                ("account_equity_at_freeze","existing_gross_exposure_cash","existing_theme_exposure_cash",
                 "stress_loss_fraction","max_price","max_shares")]
            if (any(v is None for v in (equity,existing_gross,existing_theme,stress,max_price,max_shares))
                    or equity<=0 or min(existing_gross,existing_theme)<0 or max(existing_gross,existing_theme)>=equity or not 0<stress<=1
                    or max_price<=0 or max_shares<=0 or not max_shares.is_integer() or max_shares%100):
                issues.append("INVALID_RISK_INPUTS")
                allowed=planned=stress_cash=None
            else:
                allowed=math.floor((equity*risk_cap/(max_price*stress))/100)*100
                planned=max_price*max_shares
                stress_cash=planned*stress
                if max_shares>allowed:
                    issues.append("SINGLE_TRADE_RISK_EXCEEDED")
                if existing_gross+planned>equity*gross_cap+1e-8:
                    issues.append("GROSS_EXPOSURE_EXCEEDED")
                if existing_theme+planned>equity*theme_cap+1e-8:
                    issues.append("THEME_EXPOSURE_EXCEEDED")
            if not plan.get("theme_id") or not plan.get("equity_snapshot_id") or not plan.get("exposure_snapshot_id"):
                issues.append("MISSING_THEME_OR_SNAPSHOT_EVIDENCE")
            snapshot_keys.append((plan.get("equity_snapshot_id"),plan.get("exposure_snapshot_id")))
        except (KeyError,TypeError,ValueError):
            allowed=planned=stress_cash=None
            issues.append("INVALID_OR_MISSING_RISK_FIELDS")
        results.append(dict(plan_id=plan.get("plan_id"),status="PASS" if not issues else "ISSUES_FOUND",
                            issues=issues,max_shares_by_single_risk=allowed,
                            planned_gross_cash=planned,planned_stress_loss_cash=stress_cash))
    duplicates={key for key in snapshot_keys if key!=(None,None) and snapshot_keys.count(key)>1}
    if duplicates:
        for result, key in zip(results,snapshot_keys):
            if key in duplicates:
                result["issues"].append("SHARED_EXPOSURE_SNAPSHOT_CANNOT_AGGREGATE_MULTIPLE_PLANS")
                result["status"]="ISSUES_FOUND"
    return dict(status="PASS" if results and all(r["status"]=="PASS" for r in results)
                else "NO_PLANS" if not results else "ISSUES_FOUND",
                policy_id=policy["policy_id"],plans=results,
                scope="frozen maximum planned exposure under supplied snapshot",
                sizing_is_recommendation=False,
                limitation="stress fraction and exposure snapshots supplied by caller; no market gap guarantee")
