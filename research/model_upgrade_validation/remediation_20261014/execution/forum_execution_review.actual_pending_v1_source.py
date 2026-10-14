"""Bind forum forecasts to declared execution evidence without inventing trades.

This read-only adapter reuses the existing entry funnel, entry/exit rule audits
and cash ledger. It does not submit orders, register a forecast, infer a fill
from a daily bar, or treat an unknown execution day as a cash day.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, time
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import entry_plan_audit
import execution_funnel_audit
import exit_plan_audit
import forum_reasoning_review as forum
import trade_account_ledger


VERSION = "forum-execution-review-1"
SHANGHAI = ZoneInfo("Asia/Shanghai")
LISTS = ("plans", "confirmations", "orders", "fills", "exit_confirmations", "marks", "benchmarks")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp(value):
    if not isinstance(value, str):
        raise ValueError("explicit timezone-aware evidence timestamp required")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("explicit timezone-aware evidence timestamp required")
    return result.astimezone(SHANGHAI)


def write_new(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _rows(rows):
    result = {}
    for row in rows:
        code = row.get("code")
        if not isinstance(code, str) or len(code) != 6 or not code.isdigit() or code in result:
            raise ValueError("unique six-digit full-pool codes required")
        result[code] = row
    return result


def _binding(bindings):
    for path, expected in bindings.items():
        if not Path(path).is_absolute() or not Path(path).is_file() or sha(path) != expected:
            raise ValueError("execution source hash mismatch: " + str(path))


def _evidence(item, bindings):
    """Require a saved original declaration, not an unbound status flag."""
    ref = item.get("evidence_ref", {})
    path = ref.get("path")
    if not path or bindings.get(path) != ref.get("sha256"):
        raise ValueError("execution record needs its bound saved source")
    document = read(path)
    pointer = ref.get("json_pointer")
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("execution source requires a JSON pointer")
    value = document
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        try:
            value = value[int(token)] if isinstance(value, list) else value[token]
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            raise ValueError("execution source pointer is unavailable") from exc
    projected = {k: v for k, v in item.items() if k != "evidence_ref"}
    if value != projected:
        raise ValueError("execution source row differs from its saved declaration")
    if item.get("evidence_origin") not in {"USER_CONFIRMED_BROKER_EXPORT", "USER_CONFIRMED_FILL_RECORD", "PAPER_RECORDED_EVIDENCE"}:
        raise ValueError("daily bars, inferred fills and anonymous execution records are not actual execution")


def _market_clock(value, sessions):
    point = stamp(value)
    if point.date().isoformat() not in sessions:
        raise ValueError("execution timestamp outside the supplied exchange calendar")
    clock = point.timetz().replace(tzinfo=None)
    if not (time(9, 30) <= clock <= time(11, 30) or time(13) <= clock <= time(15)):
        raise ValueError("execution timestamp outside continuous/closing trading hours")
    return point


def project(facts, review, predictions, sessions, asof, evidence=None):
    """Project all candidates; supplied executions must pass the existing audits.

    source_bindings and evidence_ref are required on every execution record.
    CSV broker exports can first be normalized with broker_execution_import_v16;
    their confirmed normalized records are then saved in the evidence bank.
    Paper records remain separate from user-confirmed execution.
    """
    cutoff = stamp(asof)
    if stamp(review["formed_at"]) > cutoff:
        raise ValueError("forum review was not available by the execution cutoff")
    pool, judged, original_predictions = _rows(facts["rows"]), _rows(review["rows"]), _rows(predictions)
    if not pool or set(pool) != set(judged) or set(pool) != set(original_predictions):
        raise ValueError("review and original predictions must retain the exact full pool")
    if any((row["name"], row["boards"]) != (judged[c]["name"], judged[c]["boards"]) for c, row in pool.items()):
        raise ValueError("review candidate identity or board group changed")
    if sessions != sorted(set(sessions)) or facts["signal_date"] not in sessions or facts["target_date"] not in sessions:
        raise ValueError("verified sorted unique exchange sessions required")
    if sessions.index(facts["target_date"]) != sessions.index(facts["signal_date"]) + 1:
        raise ValueError("forecast target must be the next exchange session")
    payload = deepcopy(evidence or {})
    records = {key: payload.get(key, []) for key in LISTS}
    if any(not isinstance(value, list) for value in records.values()):
        raise ValueError("execution evidence lists required")
    binding = payload.get("source_bindings", {})
    _binding(binding)
    kind = payload.get("kind", "user_confirmed")
    if kind not in {"user_confirmed", "paper"}:
        raise ValueError("explicit execution account kind required")
    for key, items in records.items():
        for item in items:
            if item.get("code") not in pool:
                raise ValueError("execution or valuation record outside the original full pool")
            _evidence(item, binding)
            if key not in {"marks", "benchmarks"} and item.get("kind") != kind:
                raise ValueError("paper and actual execution cannot mix")
    plans = {p["plan_id"]: p for p in records["plans"]}
    if len(plans) != len(records["plans"]) or len({p["code"] for p in plans.values()}) != len(plans):
        raise ValueError("one unique frozen plan per candidate required")
    for plan in plans.values():
        if plan.get("trade_date") != facts["target_date"]:
            raise ValueError("entry plan differs from the original forecast target")
        if not stamp(review["formed_at"]) <= stamp(plan["frozen_at"]) <= stamp(plan["available_at"]) <= cutoff:
            raise ValueError("execution plan cannot precede the bound forum review or follow the cutoff")
        if not plan.get("entry_start") or not plan.get("entry_end"):
            raise ValueError("frozen entry window required")
        start, end = _market_clock(plan["entry_start"], sessions), _market_clock(plan["entry_end"], sessions)
        if start > end or start.date().isoformat() != facts["target_date"] or end.date() != start.date():
            raise ValueError("entry window must stay in the forecast target session")
    fill_ids, order_ids = set(), set()
    orders = {r["order_id"]: r for r in records["orders"]}
    confirmations = {r["confirmation_id"]: r for r in records["confirmations"]}
    if len(orders) != len(records["orders"]) or len(confirmations) != len(records["confirmations"]):
        raise ValueError("duplicate order or confirmation")
    for order in orders.values():
        order_ids.add(order["order_id"])
        plan = plans.get(order.get("plan_id"))
        confirmation = confirmations.get(order.get("confirmation_id"))
        if plan is None or confirmation is None or plan["code"] != order["code"] or confirmation["code"] != order["code"]:
            raise ValueError("order must bind its candidate plan and confirmation")
        submitted = _market_clock(order["submitted_at"], sessions)
        if not stamp(plan["available_at"]) <= stamp(confirmation["observed_at"]) <= stamp(confirmation["available_at"]) < submitted:
            raise ValueError("observation must become available before the order; the observed open cannot be its own execution")
        if not stamp(plan["entry_start"]) <= submitted <= stamp(plan["entry_end"]):
            raise ValueError("order outside its frozen entry window")
        if stamp(order["expires_at"]) > stamp(plan["entry_end"]) or stamp(order["expires_at"]) > stamp(confirmation["expires_at"]):
            raise ValueError("order expiry exceeds its entry or confirmation window")
    for fill in records["fills"]:
        if fill["evidence_id"] in fill_ids:
            raise ValueError("duplicate execution fill")
        fill_ids.add(fill["evidence_id"])
        quantity = trade_account_ledger.number(fill["shares"], True)
        trade_account_ledger.number(fill["price"], True)
        if not quantity.is_integer() or (fill.get("side") == "BUY" and quantity % 100):
            raise ValueError("positive integer fills and A-share buy board lots required")
        if fill.get("fee_cash") is not None:
            trade_account_ledger.number(fill["fee_cash"])
        executed = _market_clock(fill["filled_at"], sessions)
        if fill.get("side") == "BUY":
            order = orders.get(fill.get("order_id"))
            if order is None or fill["code"] != order["code"] or fill.get("plan_id") != order["plan_id"]:
                raise ValueError("buy fill must reference its exact submitted order")
            if not stamp(order["submitted_at"]) < executed <= stamp(order["expires_at"]):
                raise ValueError("fill must follow its order and stay in its frozen window")
            if fill.get("confirmation_id") != order["confirmation_id"]:
                raise ValueError("buy fill must bind its exact confirmation")
        elif fill.get("side") == "SELL":
            buys = [f for f in records["fills"] if f.get("side") == "BUY" and f.get("trade_id") == fill.get("trade_id")]
            if not buys or executed.date() <= min(stamp(b["filled_at"]).date() for b in buys):
                raise ValueError("T+1: no same-session exit")
        else:
            raise ValueError("explicit BUY or SELL fill side required")
        if not executed <= stamp(fill["available_at"]) <= cutoff:
            raise ValueError("fill not available by the execution cutoff")
    plan_by_code = {p["code"]: p for p in plans.values()}
    entry_fills = [r for r in records["fills"] if r["side"] == "BUY"]
    entry_audit = entry_plan_audit.audit(records["fills"], list(plans.values()), records["confirmations"])
    if entry_audit["issue_count"]:
        raise ValueError("declared entry rules did not pass: " + json.dumps(entry_audit, ensure_ascii=False))
    exit_audit = exit_plan_audit.audit(records["fills"], list(plans.values()), records["exit_confirmations"], sessions, asof)
    # Preserve sell blocks, overdue exits and supplied off-rule exits as issues.
    # Accounting is separate from strategy compliance and never erases them.
    decisions = []
    for code, row in pool.items():
        plan = plan_by_code.get(code)
        when = plan["frozen_at"] if plan else review["formed_at"]
        known = plan["available_at"] if plan else review["formed_at"]
        decisions.append(dict(decision_id="review-" + code, trade_date=facts["target_date"], signal_date=facts["signal_date"],
            frozen_at=when, available_at=known, source="bound forum review and explicit execution plan", data_status="verified", kind=kind,
            code=code, board_group="CONSECUTIVE_BOARD", decision_state="FOCUS" if plan else "WATCH", plan_id=plan["plan_id"] if plan else None))
    manifest = dict(trade_date=facts["target_date"], signal_date=facts["signal_date"], candidate_count=len(pool), universe_complete=True,
        data_status="verified", source="exact bound complete candidate pool", kind=kind,
        observed_at=facts["signal_date"] + "T15:00:00+08:00", available_at=review["formed_at"])
    funnel = execution_funnel_audit.audit([manifest], decisions, records["confirmations"], records["orders"], entry_fills,
        [facts["target_date"]], asof)
    if funnel["issues"]:
        raise ValueError("execution funnel did not pass: " + "; ".join(funnel["issues"]))
    fees_complete = all(r.get("fee_cash") is not None and r.get("fee_evidence_id") for r in records["fills"])
    portfolio = None
    if payload.get("initial_cash") is not None:
        cash = payload.get("initial_cash_evidence", {})
        _evidence(cash, binding)
        if cash.get("initial_cash") != payload["initial_cash"] or cash.get("opening_positions_empty") is not True:
            raise ValueError("isolated cash ledger requires a bound starting cash and no prior holdings")
        if cash.get("session") != facts["target_date"] or cash.get("kind") != kind:
            raise ValueError("initial account scope or session mismatch")
        last_fill_close = max((stamp(f["filled_at"]).date().isoformat() + "T15:00:00+08:00" for f in records["fills"]), default=facts["target_date"] + "T15:00:00+08:00")
        if fees_complete and stamp(last_fill_close) <= cutoff:
            account_sessions = [d for d in sessions if facts["target_date"] <= d and stamp(d + "T15:00:00+08:00") <= cutoff]
            portfolio = trade_account_ledger.account(payload["initial_cash"], records["fills"], records["marks"], account_sessions,
                kind, asof, list(plans.values()), records["confirmations"], payload.get("risk_policy"), records["exit_confirmations"])
    slippage = []
    for quote in records["benchmarks"]:
        fill = next((f for f in records["fills"] if f["evidence_id"] == quote.get("fill_id")), None)
        if fill is None or fill["code"] != quote["code"]:
            raise ValueError("slippage benchmark has no matching actual fill")
        if not stamp(quote["observed_at"]) <= stamp(quote["available_at"]) <= stamp(fill["filled_at"]):
            raise ValueError("slippage benchmark was unavailable before the actual fill")
        reference = trade_account_ledger.number(quote["price"], True)
        delta = float(fill["price"]) - reference
        if fill["side"] == "SELL":
            delta = -delta
        slippage.append(dict(fill_id=fill["evidence_id"], benchmark_price=reference, cost_cash=delta * float(fill["shares"]),
            separately_subtracted_from_actual_pnl=False))
    # Per-trade charged cash is known without knowing total account capital.
    # Full account equity remains unavailable without a bound starting balance.
    closed = {}
    for code in pool:
        own = [f for f in records["fills"] if f["code"] == code]
        buys = [f for f in own if f["side"] == "BUY"]
        sells = [f for f in own if f["side"] == "SELL"]
        bought = sum(float(f["shares"]) for f in buys)
        sold = sum(float(f["shares"]) for f in sells)
        if sold > bought:
            raise ValueError("cumulative sale exceeds actually bought quantity")
        if bought and bought == sold and all(f.get("fee_cash") is not None and f.get("fee_evidence_id") for f in own):
            cost = sum(float(f["shares"]) * float(f["price"]) + float(f["fee_cash"]) for f in buys)
            proceeds = sum(float(f["shares"]) * float(f["price"]) - float(f["fee_cash"]) for f in sells)
            closed[code] = dict(code=code, net_return=proceeds / cost - 1, net_profit=proceeds - cost, buy_cost_cash=cost, sell_proceeds_cash=proceeds)
    exited = {r["plan_id"]: r for r in exit_audit["trades"]}
    rows = []
    for code, fact in pool.items():
        plan = plan_by_code.get(code)
        ownfills = [f for f in records["fills"] if f["code"] == code]
        buys = [f for f in ownfills if f["side"] == "BUY"]
        sells = [f for f in ownfills if f["side"] == "SELL"]
        policy = exited.get(plan["plan_id"]) if plan else None
        complete = closed.get(code)
        rows.append(dict(code=code, name=fact["name"], boards=fact["boards"],
            original_prediction=original_predictions[code], forum_independent_direction=judged[code].get("promotion_judgement"),
            planned_conditions=plan, opening_confirmation_status="SUPPLIED" if any(c["code"] == code for c in records["confirmations"]) else "UNKNOWN",
            execution_status=("DECLARED_FILL" if kind == "user_confirmed" else "PAPER_DECLARED_FILL") if buys else "UNKNOWN_NO_FILL_RECORD",
            actual_buy_quantity=sum(float(f["shares"]) for f in buys) if buys else None,
            actual_sold_quantity=sum(float(f["shares"]) for f in sells) if buys else None,
            remaining_quantity=sum(float(f["shares"]) * (1 if f["side"] == "BUY" else -1) for f in ownfills) if buys else None,
            exit_review=policy, actual_complete_trade_net_return=complete["net_return"] if complete and kind == "user_confirmed" else None,
            paper_complete_trade_net_return=complete["net_return"] if complete and kind == "paper" else None,
            completed_charged_cash_projection=complete,
            strategy_complete_trade_eligible=bool(complete and policy and not policy["issues"]),
            strategy_net_return=complete["net_return"] if complete and policy and not policy["issues"] else None,
            fee_cash=sum(float(f["fee_cash"]) for f in ownfills) if ownfills and all(f.get("fee_cash") is not None for f in ownfills) else None,
            daily_proxy_return=None, unknown_execution_is_cash=False))
    supplied_coverage = payload.get("complete_execution_coverage", [])
    if supplied_coverage:
        if set(_rows(supplied_coverage)) != set(pool):
            raise ValueError("execution completeness must cover the exact full pool")
        for cover in supplied_coverage:
            _evidence(cover, binding)
            if cover.get("execution_records_complete") is not True or stamp(cover["available_at"]) > cutoff:
                raise ValueError("invalid execution completeness declaration")
    complete_day = bool(supplied_coverage and cutoff >= stamp(facts["target_date"] + "T15:00:00+08:00"))
    funnel["known_cash_day_count"] = 1 if complete_day and not entry_fills else 0
    funnel["cash_day_rate"] = (1.0 if not entry_fills else 0.0) if complete_day else None
    proxies = payload.get("daily_proxy", [])
    if any(r.get("code") not in pool for r in proxies):
        raise ValueError("daily proxy outside the exact candidate pool")
    if proxies:
        raise ValueError("daily proxy returns must use a separately bound diagnostic report, never this actual execution input")
    return dict(version=VERSION, status="EXECUTION_EVIDENCE_REVIEW" if records["fills"] else "FULL_POOL_PENDING_EXECUTION",
        signal_date=facts["signal_date"], target_date=facts["target_date"], asof=cutoff.isoformat(), pool_size=len(pool), rows=rows,
        funnel=funnel, entry_audit=entry_audit, exit_audit=exit_audit, supplied_account_projection=portfolio,
        account_status="SUPPLIED_CASH_LEDGER" if portfolio else "UNKNOWN_CASH_FEES_OR_CLOSE_MARKS_PENDING",
        execution_fee_data_complete=fees_complete if records["fills"] else None, observed_slippage=slippage,
        actual_execution_records=len(records["fills"]) if kind == "user_confirmed" else 0, paper_execution_records=len(records["fills"]) if kind == "paper" else 0,
        account_scope="ISOLATED_SUPPLIED_REVIEW_FILLS_NOT_VERIFIED_BROKER_ACCOUNT",
        no_fill_implies_zero_return=False, touch_or_open_price_implies_fill=False, predictions_changed=False, orders_submitted_by_this_tool=False,
        execution_independently_authenticated=False, new_fits=0,
        limitation="Saved user/paper declarations are hash-bound, not broker-authenticated. Daily bars are no fill evidence. Missing fees/marks retain unknown returns. Supplied isolated account excludes deposits, dividends and corporate actions.")


def run(facts_path, review_path, original_predictions_path, calendar_path, output_dir, *, asof, evidence_path=None):
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError("choose a new immutable execution review output")
    facts, review = forum._validate_paths(facts_path, review_path)
    predictions, calendar = read(original_predictions_path), read(calendar_path)
    if predictions.get("signal_date") != facts["signal_date"] or predictions.get("target_date") != facts["target_date"]:
        raise ValueError("original prediction date differs from the forum review")
    if calendar.get("status") != "verified" or not calendar.get("source"):
        raise ValueError("verified source-bound exchange calendar required")
    evidence = read(evidence_path) if evidence_path else None
    paths = [facts_path, review_path, original_predictions_path, calendar_path, __file__, forum.__file__, entry_plan_audit.__file__,
        execution_funnel_audit.__file__, exit_plan_audit.__file__, trade_account_ledger.__file__]
    if evidence_path:
        paths.append(evidence_path)
        wanted = {"facts_sha256": sha(facts_path), "review_sha256": sha(review_path), "original_prediction_sha256": sha(original_predictions_path), "calendar_sha256": sha(calendar_path)}
        if evidence.get("version") != VERSION or any(evidence.get(k) != v for k, v in wanted.items()):
            raise ValueError("execution input must bind this exact original forecast, review and calendar")
    result = project(facts, review, predictions["rows"], calendar["sessions"], asof, evidence)
    result["source_bindings"] = facts["source_bindings"] | review["source_bindings"] | {str(Path(p).resolve()): sha(p) for p in paths} | (evidence or {}).get("source_bindings", {})
    _binding(result["source_bindings"])
    out.mkdir(parents=True)
    write_new(out / "execution_review.json", result)
    lines = ["# 完整候选的预测、确认与成交收益审阅", "", f"信号日 {result['signal_date']}，目标日 {result['target_date']}；完整 {result['pool_size']} 只。", "",
        "原预测与本次股吧分析分栏。计划条件不是开盘确认；开盘确认不是成交。没有真实成交、费用和退出记录时，实际收益保持未知。", "",
        "| 股票 | 原独立方向 | 本次论坛方向 | 开盘确认 | 成交 | 实际完整收益 |", "|---|---|---|---|---|---|"]
    for row in result["rows"]:
        call = row["original_prediction"].get("prediction_judgement", row["original_prediction"].get("promotion_judgement"))
        lines.append(f"| {row['name']} {row['code']} | {call or 'UNKNOWN'} | {row['forum_independent_direction'] or 'UNKNOWN'} | {row['opening_confirmation_status']} | {row['execution_status']} | {row['actual_complete_trade_net_return'] if row['actual_complete_trade_net_return'] is not None else 'UNKNOWN'} |")
    lines += ["", "已有 entry_plan_audit、execution_funnel_audit、exit_plan_audit、trade_account_ledger 被实际调用；T+1、卖出受阻、费用、现金和未平仓盯市沿用现有审阅。日线代理结果须在独立诊断页保存，不能填到实际收益。", "",
        "本次工具未下单、未写旧登记或账本、未拟合。存在实际成交时仍需真实源声明，工具不认证券商来源真伪。"]
    with (out / "完整池成交与收益待结算.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("facts", "review", "original-predictions", "calendar", "output-dir", "asof"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--evidence")
    args = parser.parse_args()
    report = run(args.facts, args.review, args.original_predictions, args.calendar, args.output_dir, asof=args.asof, evidence_path=args.evidence)
    print(json.dumps({k: report[k] for k in ("status", "pool_size", "actual_execution_records", "account_status", "new_fits")}, ensure_ascii=False))
