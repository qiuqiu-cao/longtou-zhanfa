"""Unify daily review, leader diagnostics, and the fail-closed input contract.

This module is deliberately an audit/decision-state bridge.  It does not rank a
new stock, manufacture missing auction/theme fields, or authorize an order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


VERSION = "v266-strategy-review-bridge-1"


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: Any) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        return stamp.tz_localize("Asia/Shanghai")
    return stamp.tz_convert("Asia/Shanghai")


def _review_age(review: dict[str, Any], reference_asof: str | None) -> dict[str, Any]:
    source = _timestamp(review.get("asof") or review.get("signal_date"))
    reference = _timestamp(reference_asof) if reference_asof else pd.Timestamp.now(tz="Asia/Shanghai")
    if source is None or reference is None:
        return {"status": "UNAVAILABLE", "age_days": None, "source_asof": review.get("asof")}
    age = max(0.0, (reference - source).total_seconds() / 86400.0)
    if age <= 1.5:
        status = "CURRENT"
    elif age <= 5:
        status = "DELAYED"
    else:
        status = "STALE"
    return {
        "status": status,
        "age_days": round(age, 3),
        "source_asof": source.isoformat(),
        "reference_asof": reference.isoformat(),
    }


def _candidate_snapshot(review: dict[str, Any]) -> dict[str, Any]:
    roles = review.get("roles") or {}
    anchors = []
    for item in roles.get("emotion_anchors") or []:
        if isinstance(item, dict):
            anchors.append({k: item.get(k) for k in ("code", "name", "boards", "role")})
    def compact(key: str) -> dict[str, Any] | None:
        item = roles.get(key)
        if not isinstance(item, dict):
            return None
        return {k: item.get(k) for k in ("code", "name", "boards", "research_score_not_probability", "evidence_status")}
    return {
        "emotion_anchors": anchors,
        "highest_leader_potential": compact("highest_leader_potential"),
        "next_height_challenger": compact("next_height_challenger"),
        "current_market_core": compact("current_market_core"),
        "leader_identity_status": roles.get("highest_leader_potential_status"),
        "challenger_identity_status": roles.get("next_height_challenger_status"),
        "buy_opportunity_status": roles.get("current_best_buy_opportunity_status"),
    }


def build_bridge(
    review: dict[str, Any],
    strategy_audit: dict[str, Any],
    input_contract: dict[str, Any],
    *,
    review_path: str | Path | None = None,
    strategy_path: str | Path | None = None,
    contract_path: str | Path | None = None,
    reference_asof: str | None = None,
) -> dict[str, Any]:
    identity_contract = input_contract.get("identity_review") or {}
    relay_contract = input_contract.get("conditional_relay_review") or {}
    contract_gate = {
        "identity_review_allowed": identity_contract.get("allowed") is True,
        "conditional_relay_review_allowed": relay_contract.get("allowed") is True,
        "trade_authorization": False,
    }
    readiness = review.get("decision_readiness") or {}
    live_ready = bool(strategy_audit.get("live_ready"))
    # Trade authorization is intentionally the conjunction of independent gates;
    # the bridge also hard-codes the final false default to prevent accidental buys.
    trade_authorized = bool(
        live_ready
        and contract_gate.get("trade_authorization") is True
        and readiness.get("actual_fill_status") not in {None, "UNAVAILABLE"}
    )
    freshness = _review_age(review, reference_asof)
    missing = {"conditional_relay": list(relay_contract.get("missing_fields") or [])}
    missing_groups = [group for group, fields in missing.items() if fields]
    blockers = list(review.get("blocking_reasons") or [])
    if not live_ready:
        blockers.append("V264_STRATEGY_AUDIT_NOT_LIVE_READY")
    if missing_groups:
        blockers.append("V265_MISSING_FIELDS_" + "_".join(sorted(missing_groups)))
    if freshness.get("status") in {"DELAYED", "STALE"}:
        blockers.append("DAILY_REVIEW_" + freshness["status"])
    return {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "decision": "TRADE_AUTHORIZED" if trade_authorized else "CASH",
        "freshness": freshness,
        "market_snapshot": {
            "signal_date": review.get("signal_date"),
            "emotion_regime": (review.get("market") or {}).get("emotion_regime"),
            "limit_count_mainboard": (review.get("market") or {}).get("limit_count_mainboard"),
            "promotion_mid": (review.get("market") or {}).get("previous_limit_feedback", {}).get("promotion_rate"),
            "failed_rate": (review.get("market") or {}).get("failed_rate"),
        },
        "candidate_snapshot": _candidate_snapshot(review),
        "gates": {
            "identity_review_allowed": contract_gate["identity_review_allowed"],
            "conditional_relay_review_allowed": contract_gate["conditional_relay_review_allowed"],
            "trade_authorization": trade_authorized,
            "strategy_upgrade_allowed": bool((strategy_audit.get("governance") or {}).get("strategy_upgrade_allowed")),
        },
        "input_contract": {
            "status": input_contract.get("status"),
            "latest_signal_date": input_contract.get("latest_signal_date"),
            "missing_groups": missing_groups,
            "missing_by_group": missing,
        },
        "blocking_reasons": sorted(set(blockers)),
        "governance": {
            "identity_may_be_reported_as_diagnostic": contract_gate["identity_review_allowed"],
            "relay_requires_complete_auction_intraday_theme_emotion": True,
            "no_order_submission": True,
            "trade_authorization_requires_live_verified_fill_and_exit": True,
        },
        "source_hashes": {
            str(path): _sha256(path)
            for path in (review_path, strategy_path, contract_path)
            if path is not None and Path(path).exists()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-review", required=True)
    parser.add_argument("--strategy-audit", required=True)
    parser.add_argument("--input-contract", required=True)
    parser.add_argument("--reference-asof")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_bridge(
        _load(args.daily_review), _load(args.strategy_audit), _load(args.input_contract),
        review_path=args.daily_review, strategy_path=args.strategy_audit,
        contract_path=args.input_contract, reference_asof=args.reference_asof,
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# V266 策略复盘桥接\n\n"
        f"版本：`{VERSION}`。该报告只统一身份诊断、字段契约和日复盘状态，不授权下单。\n\n"
        f"最终状态：`{report['decision']}`；数据新鲜度：`{report['freshness']['status']}`。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
