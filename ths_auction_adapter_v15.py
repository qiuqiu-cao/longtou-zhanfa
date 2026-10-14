#!/usr/bin/env python3
"""Strict Tonghuashun 09:25 export adapter for a preregistered relay candidate.

The adapter converts common Chinese export columns into the canonical auction
fields, checks the frozen V10/V11/V14 chain, and emits only auction-stage
WATCH/CANCEL/INELIGIBLE/DATA_LIMITED outcomes.  It cannot emit ENTER because
VWAP, pullback volume, theme synchronization and reseal are not known at 09:25.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

import announcement_forward_v14 as announcement_forward
import execution_contract_v4 as execution
import intraday_forward_v10 as v10_forward
import open_confirmation_v4 as opening
import theme_forward_v11 as theme_forward
from leader_system_v3 import ACTIVE_STATES


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "v15" / "ths_auction"
VERSION = "v15-ths-auction-adapter-1"
SHANGHAI = ZoneInfo("Asia/Shanghai")

ALIASES = {
    "code": ("code", "股票代码", "代码", "证券代码"),
    "name": ("name", "股票简称", "简称", "证券简称"),
    "auction_gap": ("auction_gap", "竞价涨幅", "今日竞价涨幅", "集合竞价涨幅"),
    "auction_amount": ("auction_amount", "今日竞价金额", "竞价金额", "集合竞价成交额"),
    "previous_amount": ("previous_amount", "昨日成交额", "前一日成交额"),
    "match_amount_0920": (
        "match_amount_0920", "9:20匹配金额", "09:20匹配金额", "九点二十分匹配金额",
    ),
    "match_amount_092450": (
        "match_amount_092450", "9:24:50匹配金额", "09:24:50匹配金额",
        "九点二十四分五十秒匹配金额",
    ),
}

QUERY_TEXT = """沪深主板，非ST，非退市整理，昨日连续涨停2天到5天，\
今日竞价涨幅大于等于-3%小于等于7%，显示今日竞价金额、昨日成交额、\
股票代码、股票简称、连续涨停天数、所属概念，按今日竞价金额占昨日成交额从大到小排序

使用规则：
1. 只在前一晚已经登记的候选代码中取交集，不准临盘新增后排。
2. 另外记录9:20与9:24:50匹配金额；问财无法导出时手工加入模板。
3. 竞价成交占比=今日竞价金额/昨日成交额。
4. 匹配额衰减率=max(0,(9:20匹配金额-9:24:50匹配金额)/9:20匹配金额)。
5. 9:25结果只能是观察或取消，真正确认最早在9:35。
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code6(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = text.split(".")[0] if "." in text and len(text.split(".")[0]) == 6 else text
    return text.zfill(6)


def parse_percent(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan
    text = str(value).strip().replace(",", "")
    if not text:
        return math.nan
    explicit = text.endswith("%")
    if explicit:
        text = text[:-1]
    number = float(text)
    return number / 100 if explicit or abs(number) > 1 else number


def parse_money(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan
    text = str(value).strip().replace(",", "").replace("元", "")
    if not text or text.lower() in {"nan", "none", "--", "-"}:
        return math.nan
    multipliers = {"亿": 100_000_000.0, "万": 10_000.0, "千": 1_000.0}
    multiplier = 1.0
    if text[-1:] in multipliers:
        multiplier = multipliers[text[-1]]
        text = text[:-1]
    return float(text) * multiplier


def alias_column(frame: pd.DataFrame, canonical: str) -> str | None:
    matches = [name for name in ALIASES[canonical] if name in frame.columns]
    if len(matches) > 1:
        raise ValueError(f"multiple columns map to {canonical}: {matches}")
    return matches[0] if matches else None


def normalize_export(frame: pd.DataFrame) -> pd.DataFrame:
    required = ("code", "name", "auction_gap", "auction_amount", "previous_amount")
    resolved = {field: alias_column(frame, field) for field in ALIASES}
    missing = [field for field in required if resolved[field] is None]
    if missing:
        raise ValueError(f"Tonghuashun export missing required fields: {missing}")
    result = pd.DataFrame(index=frame.index)
    result["code"] = frame[resolved["code"]].map(code6)
    result["name"] = frame[resolved["name"]].astype(str).str.strip()
    result["auction_gap"] = frame[resolved["auction_gap"]].map(parse_percent)
    for field in ("auction_amount", "previous_amount", "match_amount_0920", "match_amount_092450"):
        source = resolved[field]
        result[field] = frame[source].map(parse_money) if source else math.nan
    if result.duplicated("code").any():
        raise ValueError("Tonghuashun export contains duplicate stock codes")
    if result[["auction_gap", "auction_amount", "previous_amount"]].isna().any().any():
        raise ValueError("Tonghuashun export contains invalid required numeric values")
    if (result[["auction_amount", "previous_amount"]] < 0).any().any() or (result["previous_amount"] == 0).any():
        raise ValueError("auction/previous amount must be nonnegative and previous amount positive")
    result["auction_amount_ratio"] = result["auction_amount"] / result["previous_amount"]
    valid_match = result["match_amount_0920"].gt(0) & result["match_amount_092450"].notna()
    result["auction_cancel_ratio"] = math.nan
    result.loc[valid_match, "auction_cancel_ratio"] = (
        (result.loc[valid_match, "match_amount_0920"] - result.loc[valid_match, "match_amount_092450"])
        / result.loc[valid_match, "match_amount_0920"]
    ).clip(lower=0, upper=1)
    return result


def local_timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize(SHANGHAI)
    return stamp.tz_convert(SHANGHAI)


def validate_capture_time(
    trade_date: str, captured_at: Any, now: Any | None = None, enforce_wall_clock: bool = True,
) -> pd.Timestamp:
    captured = local_timestamp(captured_at)
    trade = pd.Timestamp(trade_date).date()
    start = local_timestamp(datetime.combine(trade, time(9, 25)))
    deadline = local_timestamp(datetime.combine(trade, time(9, 27)))
    if captured < start or captured > deadline:
        raise ValueError("09:25 export must be captured between 09:25:00 and 09:27:00")
    if enforce_wall_clock:
        actual = local_timestamp(now or datetime.now(SHANGHAI))
        if actual < captured:
            raise ValueError("capture timestamp is in the future")
        if actual > deadline:
            raise ValueError("09:25 screen cannot be created after the 09:27 evidence deadline")
    return captured


def load_registered_candidate(plan_path: Path) -> tuple[dict[str, Any], pd.Series, dict[str, Any]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    registration = announcement_forward.validate_registered_v10_plan(plan_path, plan)
    scores_path = plan_path.parent / "frozen_scores.csv"
    if not scores_path.exists():
        raise ValueError("registered V10 frozen score file is missing")
    scores = pd.read_csv(scores_path, dtype={"code": "string"})
    scores["code"] = scores["code"].map(code6)
    selected = scores[scores["code"].eq(code6(plan["candidate_code"]))]
    if len(selected) != 1:
        raise ValueError("registered candidate is not unique in frozen scores")
    return plan, selected.iloc[0], registration


def load_gate_bindings(
    plan: dict[str, Any], theme_binding_path: Path, announcement_binding_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    theme_check = theme_forward.verify_binding(theme_binding_path)
    if theme_check.get("status") != "PASS":
        raise ValueError("V11 theme binding is invalid")
    theme = json.loads(theme_binding_path.read_text(encoding="utf-8"))
    announcement_errors = announcement_forward.verify_one(announcement_binding_path.parent)
    if announcement_errors:
        raise ValueError("V14 announcement binding is invalid: " + "; ".join(announcement_errors))
    announcement = json.loads(announcement_binding_path.read_text(encoding="utf-8"))
    for label, binding in (("theme", theme), ("announcement", announcement)):
        if str(binding.get("session_id")) != str(plan.get("session_id")):
            raise ValueError(f"{label} binding session differs from V10 plan")
        if binding.get("candidate_code") and code6(binding["candidate_code"]) != code6(plan["candidate_code"]):
            raise ValueError(f"{label} binding candidate differs from V10 plan")
    return theme, announcement


def structural_position_cap(candidate: pd.Series, auction_gap: float) -> float:
    boards = int(float(candidate["boards"]))
    emotion = str(candidate.get("emotion_state", ""))
    gap_cap = 0.02 if auction_gap > 0.07 else 0.03 if auction_gap > 0.05 else 0.05
    return max(0.0, min(
        execution.RISK_BUDGET_PCT / execution.STRESS_LOSS_PCT,
        execution.BOARD_CAPS.get(boards, 0.0),
        execution.EMOTION_CAPS.get(emotion, 0.0) if emotion in ACTIVE_STATES else 0.0,
        gap_cap,
    ))


def screen_candidate(
    plan: dict[str, Any], candidate: pd.Series, export: pd.DataFrame,
    theme_decision: str, announcement_decision: str,
    portfolio: pd.Series | None = None,
) -> dict[str, Any]:
    code = code6(plan["candidate_code"])
    matches = export[export["code"].eq(code)]
    base = {
        "session_id": plan["session_id"], "signal_date": plan["signal_date"],
        "trade_date": plan["trade_date"], "asof": "09:25", "code": code,
        "name": plan.get("candidate_name"), "boards": int(float(candidate["boards"])),
        "emotion_state": candidate.get("emotion_state"), "theme_decision": theme_decision,
        "announcement_decision": announcement_decision,
        "execution_permission": "NO_ENTRY_AT_0925",
    }
    if len(matches) != 1:
        return {**base, "auction_status": "NOT_IN_THS_RESULT", "reasons": "候选未出现在同花顺竞价导出中",
                "structural_position_cap_pct": 0.0, "final_position_cap_pct": 0.0}
    row = matches.iloc[0]
    reasons: list[str] = []
    eligible, eligibility_reasons = opening.preopen_eligibility(candidate)
    if theme_decision != "THEME_SYNC_ELIGIBLE":
        eligible = False
        eligibility_reasons.append(f"题材绑定为{theme_decision}")
    if announcement_decision == "HARD_RISK_REJECT":
        eligible = False
        eligibility_reasons.append("公告绑定存在硬风险")
    if announcement_decision != "NO_HARD_RISK_FOUND" and announcement_decision != "HARD_RISK_REJECT":
        reasons.append(f"公告绑定为{announcement_decision}")

    gap = float(row["auction_gap"])
    amount_ratio = float(row["auction_amount_ratio"])
    cancel_ratio = float(row["auction_cancel_ratio"]) if pd.notna(row["auction_cancel_ratio"]) else math.nan
    if not eligible:
        status = "INELIGIBLE"
        reasons.extend(eligibility_reasons)
    elif announcement_decision != "NO_HARD_RISK_FOUND":
        status = "DATA_LIMITED"
    elif math.isnan(cancel_ratio):
        status = "DATA_LIMITED"
        reasons.append("缺少9:20或9:24:50匹配金额，不能验证撤单衰减")
    elif gap > 0.05 and cancel_ratio > 0.45:
        status = "CANCEL"
        reasons.append("高开超过5%且匹配额衰减超过45%")
    elif gap > 0.05 and amount_ratio < 0.02:
        status = "CANCEL"
        reasons.append("高开超过5%但竞价成交占昨日成交额不足2%")
    elif gap > 0.07:
        status = "WATCH_OVERHEATED"
        reasons.append("高开超过7%，必须等待09:35极强承接例外")
    elif amount_ratio < 0.03:
        status = "WATCH_BELOW_CONFIRMATION_FLOOR"
        reasons.append("竞价成交占比低于3%，暂不具备后续确认基础")
    else:
        status = "WATCH_AUCTION"
        reasons.append("竞价风险初筛通过，仍需09:35换手回封确认")

    structural = structural_position_cap(candidate, gap) if eligible else 0.0
    if portfolio is not None and structural > 0:
        snapshot = pd.Series({"auction_gap": gap})
        sized = execution.size_position(candidate, snapshot, portfolio)
        final_cap = float(sized["maximum_position_pct"])
        maximum_notional = float(sized["maximum_notional"])
        position_status = "PORTFOLIO_BOUND_PROVISIONAL"
    else:
        final_cap = math.nan if structural > 0 else 0.0
        maximum_notional = math.nan
        position_status = "PORTFOLIO_REQUIRED" if structural > 0 else "NO_POSITION"
    if status not in {"WATCH_AUCTION", "WATCH_OVERHEATED", "WATCH_BELOW_CONFIRMATION_FLOOR"}:
        final_cap = 0.0
        maximum_notional = 0.0
        position_status = "NO_POSITION"
    return {
        **base, "auction_gap": gap, "auction_amount_ratio": amount_ratio,
        "auction_cancel_ratio": None if math.isnan(cancel_ratio) else cancel_ratio,
        "auction_status": status, "reasons": "；".join(dict.fromkeys(reasons)),
        "structural_position_cap_pct": structural,
        "final_position_cap_pct": None if math.isnan(final_cap) else final_cap,
        "maximum_notional": None if math.isnan(maximum_notional) else maximum_notional,
        "position_status": position_status,
        "next_required_confirmation": "09:35收复VWAP、首次回落缩量、题材同步>=60%、真实换手回封",
    }


def run_screen(
    plan_path: Path, theme_binding_path: Path, announcement_binding_path: Path,
    export_path: Path, captured_at: Any, source: str, portfolio_path: Path | None,
    output: Path = DEFAULT_OUTPUT, now: Any | None = None, enforce_wall_clock: bool = True,
) -> dict[str, Any]:
    plan, candidate, registration = load_registered_candidate(plan_path)
    captured = validate_capture_time(plan["trade_date"], captured_at, now, enforce_wall_clock)
    theme, announcement = load_gate_bindings(plan, theme_binding_path, announcement_binding_path)
    export = normalize_export(pd.read_csv(export_path, dtype="string"))
    portfolio = None
    if portfolio_path is not None:
        portfolio_frame = pd.read_csv(portfolio_path)
        portfolio = execution.validate_portfolio_state(portfolio_frame, captured.to_pydatetime())
    result = screen_candidate(
        plan, candidate, export, str(theme["theme_decision"]), str(announcement["decision"]), portfolio,
    )
    target = output / str(plan["trade_date"])
    target.mkdir(parents=True, exist_ok=True)
    screen_path = target / "auction_screen_0925.json"
    normalized_path = target / "normalized_ths_export_0925.csv"
    if screen_path.exists() or normalized_path.exists():
        raise FileExistsError("immutable 09:25 Tonghuashun screen already exists for trade date")
    export.to_csv(normalized_path, index=False, encoding="utf-8-sig")
    payload = {
        "version": VERSION, "captured_at": captured.isoformat(), "source": source,
        "data_status": "verified_terminal_export", "input_sha256": sha256_file(export_path),
        "normalized_sha256": sha256_file(normalized_path), "plan_sha256": sha256_file(plan_path),
        "v10_registration": registration, "theme_binding_hash": theme["binding_hash"],
        "announcement_binding_hash": announcement["binding_hash"], "result": result,
        "warning": "09:25 never emits ENTER; live order queue and actual fill remain unavailable",
    }
    payload["event_hash"] = announcement_forward.hash_value(payload)
    screen_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**payload, "screen_path": str(screen_path.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tonghuashun 09:25 export adapter for board relay")
    sub = parser.add_subparsers(dest="command", required=True)
    query = sub.add_parser("query")
    query.add_argument("--output", type=Path)
    screen = sub.add_parser("screen")
    screen.add_argument("--plan", type=Path, required=True)
    screen.add_argument("--theme-binding", type=Path, required=True)
    screen.add_argument("--announcement-binding", type=Path, required=True)
    screen.add_argument("--export", type=Path, required=True)
    screen.add_argument("--captured-at", required=True)
    screen.add_argument("--source", default="同花顺问财手工导出")
    screen.add_argument("--portfolio", type=Path)
    screen.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "query":
        if args.output:
            if args.output.exists():
                raise FileExistsError("query output already exists")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(QUERY_TEXT, encoding="utf-8")
        result = {"version": VERSION, "query": QUERY_TEXT, "output": str(args.output) if args.output else None}
    else:
        result = run_screen(
            args.plan, args.theme_binding, args.announcement_binding, args.export,
            args.captured_at, args.source, args.portfolio, args.output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
