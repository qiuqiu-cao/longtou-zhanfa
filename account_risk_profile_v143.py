"""Freezeable account-risk policy and fee-aware one-stock position sizing.

The module does not choose a stock and never invents user capital.  A missing
or malformed profile fails closed.  Paper and user-confirmed accounts remain
separate, while both use the same auditable sizing arithmetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from leader_target_screen_v17 import finite, timestamp


REGIMES = {"启动", "升温", "加速", "一致", "大分歧", "退潮", "修复"}
BOARD_GROUPS = {"FIRST_BOARD", "CONSECUTIVE_BOARD"}
ACCOUNT_KINDS = {"paper", "user_confirmed"}


def _integer(value):
    number = finite(value)
    return (int(number) if number is not None and number.is_integer()
            and not isinstance(value, bool) else None)


def audit_profile(profile, asof, expected_initial_cash=None):
    if not isinstance(profile, dict):
        return {
            "status": "INCOMPLETE", "issues": ["ACCOUNT_RISK_PROFILE_NOT_SUPPLIED"],
            "can_size_positions": False, "can_compute_account_drawdown_budget": False,
        }
    issues = []
    try:
        cutoff = timestamp(asof)
        frozen, available = timestamp(profile["frozen_at"]), timestamp(profile["available_at"])
        if frozen > available or available > cutoff:
            issues.append("PROFILE_TIME_NOT_POINT_IN_TIME")
    except (KeyError, TypeError, ValueError):
        issues.append("PROFILE_TIME_INVALID")

    if not profile.get("policy_id") or not profile.get("source"):
        issues.append("PROFILE_ID_OR_SOURCE_MISSING")
    if profile.get("data_status") != "verified":
        issues.append("PROFILE_NOT_VERIFIED")
    if profile.get("account_kind") not in ACCOUNT_KINDS:
        issues.append("ACCOUNT_KIND_INVALID")
    if profile.get("account_kind") == "user_confirmed" and not profile.get("equity_evidence_id"):
        issues.append("USER_ACCOUNT_EQUITY_EVIDENCE_MISSING")

    fields = {
        "initial_cash": (0, None),
        "single_trade_risk_fraction": (0, .02),
        "gross_exposure_fraction": (0, 1),
        "theme_exposure_fraction": (0, 1),
        "maximum_drawdown_fraction": (0, .50),
        "minimum_cash_buffer_fraction": (0, 1),
        "commission_rate": (0, .02),
        "minimum_commission_cash": (0, None),
        "stamp_tax_rate_sell": (0, .02),
        "transfer_fee_rate_both": (0, .02),
        "other_fee_rate_both": (0, .02),
    }
    numbers = {}
    for name, (lower, upper) in fields.items():
        value = finite(profile.get(name))
        numbers[name] = value
        invalid = value is None or value < lower or (name in {
            "initial_cash", "single_trade_risk_fraction", "gross_exposure_fraction",
            "theme_exposure_fraction", "maximum_drawdown_fraction"} and value == lower)
        if upper is not None and value is not None and value > upper:
            invalid = True
        if invalid:
            issues.append("INVALID_" + name.upper())
    if (numbers.get("theme_exposure_fraction") is not None
            and numbers.get("gross_exposure_fraction") is not None
            and numbers["theme_exposure_fraction"] > numbers["gross_exposure_fraction"]):
        issues.append("THEME_EXPOSURE_EXCEEDS_GROSS_CAP")
    if expected_initial_cash is not None:
        expected = finite(expected_initial_cash)
        if expected is None or numbers.get("initial_cash") != expected:
            issues.append("INITIAL_CASH_DOES_NOT_MATCH_LEDGER")

    maximum_positions = _integer(profile.get("maximum_positions"))
    lot_size = _integer(profile.get("lot_size"))
    if maximum_positions != 1:
        issues.append("MAXIMUM_POSITIONS_MUST_EQUAL_ONE")
    if lot_size != 100:
        issues.append("A_SHARE_BUY_LOT_MUST_EQUAL_100")

    caps = profile.get("regime_position_caps")
    if not isinstance(caps, dict) or set(caps) != REGIMES:
        issues.append("REGIME_POSITION_CAPS_INCOMPLETE")
    else:
        for regime, raw in caps.items():
            value = finite(raw)
            if (value is None or value < 0 or value > 1
                    or (numbers.get("gross_exposure_fraction") is not None
                        and value > numbers["gross_exposure_fraction"])):
                issues.append("INVALID_REGIME_POSITION_CAP_" + regime)
    stress = profile.get("stress_loss_fraction_by_group")
    if not isinstance(stress, dict) or set(stress) != BOARD_GROUPS:
        issues.append("STRESS_LOSS_BY_GROUP_INCOMPLETE")
    else:
        for group, raw in stress.items():
            value = finite(raw)
            if value is None or not 0 < value <= .50:
                issues.append("INVALID_STRESS_LOSS_" + group)

    issues = sorted(set(issues))
    return {
        "status": "PASS" if not issues else "INCOMPLETE",
        "issues": issues,
        "policy_id": profile.get("policy_id"),
        "account_kind": profile.get("account_kind"),
        "can_size_positions": not issues,
        "can_compute_account_drawdown_budget": not issues,
        "maximum_drawdown_fraction": (numbers.get("maximum_drawdown_fraction")
                                      if not issues else None),
        "sizing_is_recommendation": False,
        "guarantee_against_gap_or_limit_down": False,
    }


def _fees(profile, gross, side):
    commission = max(profile["minimum_commission_cash"], gross * profile["commission_rate"])
    common = gross * (profile["transfer_fee_rate_both"] + profile["other_fee_rate_both"])
    stamp = gross * profile["stamp_tax_rate_sell"] if side == "SELL" else 0.0
    return commission + common + stamp


def size_position(profile, plan, asof):
    profile_audit = audit_profile(profile, asof)
    if profile_audit["status"] != "PASS":
        return {"status": "NO_SIZE", "issues": profile_audit["issues"],
                "max_shares": 0, "max_position_cash": 0.0}
    issues = []
    try:
        if plan.get("risk_policy_id") != profile["policy_id"]:
            issues.append("RISK_POLICY_ID_MISMATCH")
        if plan.get("regime") not in REGIMES:
            issues.append("REGIME_INVALID")
        if plan.get("board_group") not in BOARD_GROUPS:
            issues.append("BOARD_GROUP_INVALID")
        equity = finite(plan["account_equity_at_freeze"])
        gross = finite(plan["existing_gross_exposure_cash"])
        theme = finite(plan["existing_theme_exposure_cash"])
        cash = finite(plan["available_cash_at_freeze"])
        price = finite(plan["maximum_acceptable_price"])
        drawdown = finite(plan["current_drawdown_fraction"])
        if (any(value is None for value in (equity, gross, theme, cash, price, drawdown))
                or equity <= 0 or min(gross, theme, cash) < 0 or price <= 0
                or gross > equity or theme > equity or cash > equity or drawdown > 0):
            issues.append("INVALID_ACCOUNT_OR_PRICE_SNAPSHOT")
        if not plan.get("equity_snapshot_id") or not plan.get("exposure_snapshot_id"):
            issues.append("ACCOUNT_SNAPSHOT_EVIDENCE_MISSING")
        if not plan.get("theme_id"):
            issues.append("SPECIFIC_THEME_ID_MISSING")
        frozen, available = timestamp(plan["frozen_at"]), timestamp(plan["available_at"])
        if frozen > available or available > timestamp(asof):
            issues.append("PLAN_TIME_NOT_POINT_IN_TIME")
    except (KeyError, TypeError, ValueError):
        issues.append("PLAN_FIELDS_MISSING_OR_INVALID")
        equity = gross = theme = cash = price = drawdown = None

    if issues:
        return {"status": "NO_SIZE", "issues": sorted(set(issues)),
                "max_shares": 0, "max_position_cash": 0.0,
                "profile_audit": profile_audit}
    if drawdown <= -profile["maximum_drawdown_fraction"]:
        return {"status": "DRAWDOWN_HALT", "issues": ["MAXIMUM_DRAWDOWN_REACHED"],
                "max_shares": 0, "max_position_cash": 0.0,
                "profile_audit": profile_audit}

    regime_cap = equity * profile["regime_position_caps"][plan["regime"]]
    gross_remaining = equity * profile["gross_exposure_fraction"] - gross
    theme_remaining = equity * profile["theme_exposure_fraction"] - theme
    cash_remaining = cash - equity * profile["minimum_cash_buffer_fraction"]
    stress_fraction = profile["stress_loss_fraction_by_group"][plan["board_group"]]
    risk_cash_cap = equity * profile["single_trade_risk_fraction"]
    notional_risk_cap = risk_cash_cap / stress_fraction
    raw_cap = max(0.0, min(regime_cap, gross_remaining, theme_remaining,
                           cash_remaining, notional_risk_cap))
    lot = profile["lot_size"]
    shares = math.floor(raw_cap / price / lot) * lot
    # Fee-aware cash and stress checks.  Decrement one lot until every hard cap passes.
    while shares > 0:
        entry_gross = shares * price
        entry_fee = _fees(profile, entry_gross, "BUY")
        stress_exit_gross = entry_gross * (1 - stress_fraction)
        exit_fee = _fees(profile, stress_exit_gross, "SELL")
        stress_loss_cash = entry_gross - stress_exit_gross + entry_fee + exit_fee
        if (entry_gross + entry_fee <= cash_remaining + 1e-8
                and entry_gross <= min(regime_cap, gross_remaining, theme_remaining) + 1e-8
                and stress_loss_cash <= risk_cash_cap + 1e-8):
            break
        shares -= lot
    entry_gross = shares * price
    entry_fee = _fees(profile, entry_gross, "BUY") if shares else 0.0
    stress_exit_gross = entry_gross * (1 - stress_fraction)
    exit_fee = _fees(profile, stress_exit_gross, "SELL") if shares else 0.0
    stress_loss_cash = entry_gross - stress_exit_gross + entry_fee + exit_fee
    return {
        "status": "SIZE_READY" if shares else "BELOW_ONE_LOT_OR_BLOCKED",
        "issues": [] if shares else ["RISK_BUDGET_BELOW_ONE_LOT"],
        "max_shares": int(shares), "max_position_cash": entry_gross,
        "estimated_entry_fee_cash": entry_fee,
        "estimated_stress_exit_fee_cash": exit_fee,
        "estimated_stress_loss_cash": stress_loss_cash,
        "stress_loss_fraction_before_fees": stress_fraction,
        "binding_cash_caps": {
            "regime": regime_cap, "gross_remaining": gross_remaining,
            "theme_remaining": theme_remaining, "cash_after_buffer": cash_remaining,
            "single_trade_risk_notional_before_fees": notional_risk_cap,
        },
        "profile_audit": profile_audit,
        "sizing_is_maximum_not_target": True,
        "buy_signal": False,
    }


def run(input_path, output_path, asof):
    source, target = Path(input_path), Path(output_path)
    if target.exists():
        raise FileExistsError("choose a new immutable account-risk output path")
    payload = json.loads(source.read_text(encoding="utf-8"))
    profile = payload.get("profile")
    result = {"profile_audit": audit_profile(profile, asof)}
    if "plan" in payload:
        result["position_size"] = size_position(profile, payload["plan"], asof)
    result["asof"] = timestamp(asof).isoformat()
    result["source_hashes"] = {
        str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest(),
        str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asof", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.asof), ensure_ascii=False, indent=2))
