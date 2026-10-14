#!/usr/bin/env python3
"""Frozen-threshold audit for V4 large-loss rejection rules.

This module is deliberately independent from the frozen V3 model and the locked
V4 emotion challenger.  It evaluates domain rules whose thresholds are fixed in
source before the audit is run.  Passing the audit makes a rule a *research
candidate* only; it never mutates the production scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


AUDIT_VERSION = "v4-risk-rule-audit-1"
RESEARCH_STATUS = "RESEARCH_ONLY"
SUPPORTED_BOARDS = (2, 3, 4, 5)
EXECUTED_STATUSES = {
    "inferred_daily_limit_touch_fill",
    "sell_blocked_one_word_down",
}


@dataclass(frozen=True)
class RiskRule:
    rule_id: str
    label: str
    stage: str
    description: str
    required_fields: tuple[str, ...]
    predicate: Callable[[pd.DataFrame], pd.Series]


def _num(frame: pd.DataFrame, field: str) -> pd.Series:
    return pd.to_numeric(frame[field], errors="coerce")


def _text(frame: pd.DataFrame, field: str) -> pd.Series:
    return frame[field].astype("string").str.strip()


# Thresholds below are fixed domain definitions already used by V3 emotion/risk
# language or conservative trading rules.  They must not be selected from the
# audit output.  Any threshold change requires a new audit version and a future
# evaluation period.
RULES: tuple[RiskRule, ...] = (
    RiskRule(
        "EOD_EMOTION_ADVERSE",
        "大分歧或退潮",
        "EOD",
        "盘后情绪状态已经进入大分歧或退潮。",
        ("emotion_state",),
        lambda f: _text(f, "emotion_state").isin(["大分歧", "退潮"]),
    ),
    RiskRule(
        "EOD_FALSE_ACCELERATION",
        "假加速",
        "EOD",
        "高度抬升但至少两个市场确认项恶化。",
        ("false_acceleration",),
        lambda f: _num(f, "false_acceleration").eq(1),
    ),
    RiskRule(
        "EOD_FRAGILITY_HIGH",
        "环境脆弱度高",
        "EOD",
        "regime_fragility达到V3既定高风险阈值0.68。",
        ("regime_fragility",),
        lambda f: _num(f, "regime_fragility").ge(0.68),
    ),
    RiskRule(
        "EOD_LOSS_EFFECT_WEAK",
        "亏钱效应恶化",
        "EOD",
        "loss_effect_health低于既定弱势阈值0.35。",
        ("loss_effect_health",),
        lambda f: _num(f, "loss_effect_health").lt(0.35),
    ),
    RiskRule(
        "EOD_TAIL_LOSS_SEVERE",
        "昨日涨停尾部重伤",
        "EOD",
        "昨日涨停股表现的10%分位数不高于-5%。",
        ("previous_limit_tail_loss",),
        lambda f: _num(f, "previous_limit_tail_loss").le(-0.05),
    ),
    RiskRule(
        "EOD_RISK_LEVEL_HIGH",
        "综合情绪高风险",
        "EOD",
        "V3因果情绪层给出的HIGH风险状态。",
        ("emotion_risk_level",),
        lambda f: _text(f, "emotion_risk_level").eq("HIGH"),
    ),
    RiskRule(
        "EOD_FIVE_BOARD",
        "五板高位",
        "EOD",
        "五板处于2—5板交易范围的风险上沿。",
        ("boards",),
        lambda f: _num(f, "boards").eq(5),
    ),
    RiskRule(
        "EOD_ONE_WORD_STREAK",
        "连续一字",
        "EOD",
        "此前连续一字板不少于2个，首次换手时结构脆弱。",
        ("one_word_streak",),
        lambda f: _num(f, "one_word_streak").ge(2),
    ),
    RiskRule(
        "OPEN_GAP_OVERHEATED",
        "竞价高开过热代理",
        "DAILY_OPEN_PROXY",
        "次日开盘涨幅不低于7%；这里只能用日线open代理。",
        ("next_open_gap",),
        lambda f: _num(f, "next_open_gap").ge(0.07),
    ),
    RiskRule(
        "OPEN_GAP_ADVERSE_COMBO",
        "高开叠加情绪逆风",
        "DAILY_OPEN_PROXY",
        "次日开盘涨幅不低于5%，且盘后处于大分歧或退潮。",
        ("next_open_gap", "emotion_state"),
        lambda f: _num(f, "next_open_gap").ge(0.05)
        & _text(f, "emotion_state").isin(["大分歧", "退潮"]),
    ),
)


def validate_input(frame: pd.DataFrame) -> None:
    required = {
        "date",
        "trade_date",
        "code",
        "boards",
        "large_loss",
        "promoted_next_day",
        "mark_to_market_return",
        "execution_status",
        "large_loss_label_status",
        "execution_source",
    }
    required.update(field for rule in RULES for field in rule.required_fields)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"candidate table missing required fields: {','.join(missing)}")


def eligible_executions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the exact audit population without treating missing labels as zero."""
    validate_input(frame)
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce")
    result["boards"] = pd.to_numeric(result["boards"], errors="coerce")
    result["large_loss"] = pd.to_numeric(result["large_loss"], errors="coerce")
    result["promoted_next_day"] = pd.to_numeric(
        result["promoted_next_day"], errors="coerce"
    )
    result["mark_to_market_return"] = pd.to_numeric(
        result["mark_to_market_return"], errors="coerce"
    )
    mask = (
        result["boards"].isin(SUPPORTED_BOARDS)
        & result["execution_status"].isin(EXECUTED_STATUSES)
        & result["large_loss"].notna()
    )
    result = result.loc[mask].copy()
    if result["date"].isna().any() or result["trade_date"].isna().any():
        raise ValueError("eligible executions contain invalid date/trade_date values")
    if (result["trade_date"] <= result["date"]).any():
        raise ValueError("trade_date must be strictly after signal date")
    duplicated = result.duplicated(["date", "trade_date", "code"], keep=False)
    if duplicated.any():
        examples = result.loc[duplicated, ["date", "trade_date", "code"]].head(5)
        raise ValueError(f"duplicate audit keys detected: {examples.to_dict('records')}")
    if not result["large_loss"].isin([0.0, 1.0]).all():
        raise ValueError("large_loss labels must be binary")
    return result.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def assign_contiguous_folds(frame: pd.DataFrame, fold_count: int = 3) -> pd.DataFrame:
    """Assign whole signal dates to disjoint chronological folds."""
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    dates = pd.Index(sorted(frame["date"].dropna().unique()))
    if len(dates) < fold_count:
        raise ValueError("not enough unique dates for requested folds")
    chunks = np.array_split(dates.to_numpy(), fold_count)
    mapping = {
        pd.Timestamp(value): idx + 1
        for idx, chunk in enumerate(chunks)
        for value in chunk
    }
    result = frame.copy()
    result["fold"] = result["date"].map(mapping).astype(int)
    return result


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1 + (z * z / total)
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0 or math.isnan(denominator):
        return None
    return numerator / denominator


def evaluate_partition(frame: pd.DataFrame, flagged: pd.Series) -> dict:
    flagged = flagged.reindex(frame.index).fillna(False).astype(bool)
    yes = frame.loc[flagged]
    no = frame.loc[~flagged]
    yes_losses = int(yes["large_loss"].sum())
    no_losses = int(no["large_loss"].sum())
    yes_rate = float(yes["large_loss"].mean()) if len(yes) else math.nan
    no_rate = float(no["large_loss"].mean()) if len(no) else math.nan
    yes_low, yes_high = wilson_interval(yes_losses, len(yes))
    no_low, no_high = wilson_interval(no_losses, len(no))
    return {
        "assessed_n": int(len(frame)),
        "flagged_n": int(len(yes)),
        "unflagged_n": int(len(no)),
        "flagged_dates": int(yes["date"].nunique()),
        "baseline_large_loss_rate": float(frame["large_loss"].mean()),
        "flagged_large_loss_rate": None if math.isnan(yes_rate) else yes_rate,
        "unflagged_large_loss_rate": None if math.isnan(no_rate) else no_rate,
        "risk_difference": None
        if math.isnan(yes_rate) or math.isnan(no_rate)
        else yes_rate - no_rate,
        "risk_ratio": None
        if math.isnan(yes_rate) or math.isnan(no_rate)
        else _safe_ratio(yes_rate, no_rate),
        "flagged_loss_wilson_low": None if math.isnan(yes_low) else yes_low,
        "flagged_loss_wilson_high": None if math.isnan(yes_high) else yes_high,
        "unflagged_loss_wilson_low": None if math.isnan(no_low) else no_low,
        "unflagged_loss_wilson_high": None if math.isnan(no_high) else no_high,
        "flagged_promotion_rate": float(yes["promoted_next_day"].mean())
        if len(yes) and yes["promoted_next_day"].notna().any()
        else None,
        "unflagged_promotion_rate": float(no["promoted_next_day"].mean())
        if len(no) and no["promoted_next_day"].notna().any()
        else None,
        "flagged_mean_return": float(yes["mark_to_market_return"].mean())
        if len(yes) and yes["mark_to_market_return"].notna().any()
        else None,
        "unflagged_mean_return": float(no["mark_to_market_return"].mean())
        if len(no) and no["mark_to_market_return"].notna().any()
        else None,
    }


def _available_mask(frame: pd.DataFrame, fields: Iterable[str]) -> pd.Series:
    available = pd.Series(True, index=frame.index)
    for field in fields:
        if pd.api.types.is_numeric_dtype(frame[field]):
            available &= pd.to_numeric(frame[field], errors="coerce").notna()
        else:
            text = frame[field].astype("string").str.strip()
            available &= text.notna() & text.ne("") & text.str.lower().ne("nan")
    return available


def stability_gate(folds: list[dict], pooled: dict) -> tuple[bool, list[str]]:
    """Conservative, pre-declared consistency gate; no threshold fitting."""
    reasons: list[str] = []
    if any(item["flagged_n"] < 30 for item in folds):
        reasons.append("某个时间段标记样本少于30")
    if pooled["flagged_n"] < 150:
        reasons.append("全样本标记样本少于150")
    if any(item["risk_difference"] is None or item["risk_difference"] <= 0 for item in folds):
        reasons.append("风险差未在全部时间段同向")
    if any(item["risk_ratio"] is None or item["risk_ratio"] < 1.10 for item in folds):
        reasons.append("某个时间段风险比低于1.10")
    if pooled["risk_ratio"] is None or pooled["risk_ratio"] < 1.20:
        reasons.append("全样本风险比低于1.20")
    if pooled["risk_difference"] is None or pooled["risk_difference"] < 0.03:
        reasons.append("全样本风险差低于3个百分点")
    return not reasons, reasons


def audit_rules(frame: pd.DataFrame, rules: Iterable[RiskRule] = RULES) -> dict:
    eligible = assign_contiguous_folds(eligible_executions(frame), 3)
    results: list[dict] = []
    for rule in rules:
        available = _available_mask(eligible, rule.required_fields)
        assessed = eligible.loc[available].copy()
        raw_flags = rule.predicate(assessed).fillna(False).astype(bool)
        fold_results: list[dict] = []
        for fold in sorted(assessed["fold"].unique()):
            part = assessed.loc[assessed["fold"].eq(fold)]
            metrics = evaluate_partition(part, raw_flags.loc[part.index])
            metrics.update(
                {
                    "fold": int(fold),
                    "start_date": part["date"].min().date().isoformat(),
                    "end_date": part["date"].max().date().isoformat(),
                }
            )
            fold_results.append(metrics)
        pooled = evaluate_partition(assessed, raw_flags)
        stable, failure_reasons = stability_gate(fold_results, pooled)
        results.append(
            {
                "rule_id": rule.rule_id,
                "label": rule.label,
                "stage": rule.stage,
                "description": rule.description,
                "required_fields": list(rule.required_fields),
                "eligible_n": int(len(eligible)),
                "assessed_n": int(len(assessed)),
                "missing_n": int(len(eligible) - len(assessed)),
                "pooled": pooled,
                "folds": fold_results,
                "stable_candidate": stable,
                "failure_reasons": failure_reasons,
            }
        )
    source_counts = eligible["execution_source"].fillna("unavailable").value_counts().to_dict()
    label_counts = eligible["large_loss_label_status"].fillna("unavailable").value_counts().to_dict()
    fold_windows = []
    for fold, part in eligible.groupby("fold", sort=True):
        fold_windows.append(
            {
                "fold": int(fold),
                "start_date": part["date"].min().date().isoformat(),
                "end_date": part["date"].max().date().isoformat(),
                "n": int(len(part)),
                "dates": int(part["date"].nunique()),
                "large_loss_rate": float(part["large_loss"].mean()),
            }
        )
    passed = [item["rule_id"] for item in results if item["stable_candidate"]]
    return {
        "audit_version": AUDIT_VERSION,
        "status": RESEARCH_STATUS,
        "decision": "STABLE_RESEARCH_CANDIDATES" if passed else "NO_STABLE_RULE",
        "scope": "沪深主板非ST连续2—5板的日线触板成交代理",
        "label_definition": "模拟入场价至T+1实际可得退出价，扣费后收益<=-7%",
        "execution_caveat": (
            "日线触及涨停即按涨停价成交是乐观上界代理；没有排队、逐笔或真实委托证据。"
        ),
        "threshold_policy": (
            "阈值固定在源代码中；本审计不搜索阈值，也不自动修改冻结模型。"
        ),
        "population": {
            "n": int(len(eligible)),
            "dates": int(eligible["date"].nunique()),
            "start_date": eligible["date"].min().date().isoformat(),
            "end_date": eligible["date"].max().date().isoformat(),
            "large_losses": int(eligible["large_loss"].sum()),
            "large_loss_rate": float(eligible["large_loss"].mean()),
            "execution_source_counts": {str(k): int(v) for k, v in source_counts.items()},
            "label_status_counts": {str(k): int(v) for k, v in label_counts.items()},
        },
        "fold_windows": fold_windows,
        "stable_rule_ids": passed,
        "rules": results,
    }


def _fmt_rate(value: float | None) -> str:
    return "NA" if value is None or math.isnan(value) else f"{value:.1%}"


def render_markdown(report: dict) -> str:
    pop = report["population"]
    lines = [
        "# V4 大面风险规则审计",
        "",
        f"- 状态：`{report['status']}`",
        f"- 结论：`{report['decision']}`",
        f"- 样本：{pop['n']} 条，{pop['start_date']} 至 {pop['end_date']}，共 {pop['dates']} 个信号日",
        f"- 基准大面率：{_fmt_rate(pop['large_loss_rate'])}（{pop['large_losses']} 笔）",
        f"- 标签：{report['label_definition']}",
        f"- 执行限制：{report['execution_caveat']}",
        f"- 阈值纪律：{report['threshold_policy']}",
        "",
        "## 时间分段",
        "",
        "| 分段 | 日期 | 样本 | 大面率 |",
        "|---:|---|---:|---:|",
    ]
    for fold in report["fold_windows"]:
        lines.append(
            f"| {fold['fold']} | {fold['start_date']}—{fold['end_date']} | "
            f"{fold['n']} | {_fmt_rate(fold['large_loss_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## 规则结果",
            "",
            "| 规则 | 阶段 | 可评估/缺失 | 标记数 | 标记大面率 | 未标记大面率 | 风险比 | 风险差 | 稳定候选 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in report["rules"]:
        pooled = item["pooled"]
        ratio = pooled["risk_ratio"]
        difference = pooled["risk_difference"]
        lines.append(
            f"| {item['label']} (`{item['rule_id']}`) | {item['stage']} | "
            f"{item['assessed_n']}/{item['missing_n']} | {pooled['flagged_n']} | "
            f"{_fmt_rate(pooled['flagged_large_loss_rate'])} | "
            f"{_fmt_rate(pooled['unflagged_large_loss_rate'])} | "
            f"{'NA' if ratio is None else f'{ratio:.2f}'} | "
            f"{_fmt_rate(difference)} | {'是' if item['stable_candidate'] else '否'} |"
        )
    lines.extend(["", "## 稳定性明细", ""])
    for item in report["rules"]:
        lines.append(f"### {item['label']} (`{item['rule_id']}`)")
        lines.append("")
        lines.append(item["description"])
        lines.append("")
        for fold in item["folds"]:
            fold_ratio = "NA" if fold["risk_ratio"] is None else f"{fold['risk_ratio']:.2f}"
            lines.append(
                f"- 第{fold['fold']}段 {fold['start_date']}—{fold['end_date']}："
                f"标记 {fold['flagged_n']}，大面率 {_fmt_rate(fold['flagged_large_loss_rate'])}；"
                f"未标记 {_fmt_rate(fold['unflagged_large_loss_rate'])}；"
                f"风险比 {fold_ratio}。"
            )
        if item["stable_candidate"]:
            lines.append("- 结果：通过预设稳定性闸门，但仍只能作为未来样本验证候选。")
        else:
            lines.append("- 未通过：" + "；".join(item["failure_reasons"]) + "。")
        lines.append("")
    lines.extend(
        [
            "## 使用边界",
            "",
            "- `EOD`规则只使用信号日收盘前已经形成的字段。",
            "- `DAILY_OPEN_PROXY`规则使用次日开盘价代理，不能替代9:25竞价快照或Level-2撤单数据。",
            "- 缺失字段不按0处理，对应样本从该条规则的分母中剔除并单列缺失数。",
            "- 通过不等于上线：还需冻结规则后，在新的前向样本中验证；本报告不会改变V3或V4锁定文件。",
            "- 未通过的规则不得因为全样本看起来有效就进入实盘拒绝层。",
            "",
        ]
    )
    return "\n".join(lines)


def flatten_results(report: dict) -> pd.DataFrame:
    rows = []
    for item in report["rules"]:
        for scope, metrics in [("pooled", item["pooled"])] + [
            (f"fold_{fold['fold']}", fold) for fold in item["folds"]
        ]:
            rows.append(
                {
                    "audit_version": report["audit_version"],
                    "rule_id": item["rule_id"],
                    "label": item["label"],
                    "stage": item["stage"],
                    "scope": scope,
                    "start_date": metrics.get("start_date", report["population"]["start_date"]),
                    "end_date": metrics.get("end_date", report["population"]["end_date"]),
                    "assessed_n": metrics["assessed_n"],
                    "flagged_n": metrics["flagged_n"],
                    "unflagged_n": metrics["unflagged_n"],
                    "flagged_large_loss_rate": metrics["flagged_large_loss_rate"],
                    "unflagged_large_loss_rate": metrics["unflagged_large_loss_rate"],
                    "risk_ratio": metrics["risk_ratio"],
                    "risk_difference": metrics["risk_difference"],
                    "flagged_promotion_rate": metrics["flagged_promotion_rate"],
                    "unflagged_promotion_rate": metrics["unflagged_promotion_rate"],
                    "flagged_mean_return": metrics["flagged_mean_return"],
                    "unflagged_mean_return": metrics["unflagged_mean_return"],
                    "stable_candidate": item["stable_candidate"],
                    "failure_reasons": "；".join(item["failure_reasons"]),
                }
            )
    return pd.DataFrame(rows)


def run(input_path: Path, output_dir: Path) -> dict:
    frame = pd.read_csv(input_path, dtype={"code": "string"}, low_memory=False)
    report = audit_rules(frame)
    report["assurance"] = "LOCAL_REPRODUCIBLE_HASHED_INPUT_ONLY"
    report["input_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "risk_rule_audit.json"
    md_path = output_dir / "risk_rule_audit.md"
    csv_path = output_dir / "risk_rule_audit.csv"
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["input_path"] = str(input_path.resolve())
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    flatten_results(report).to_csv(csv_path, index=False, encoding="utf-8-sig")
    report["output_paths"] = {
        "json": str(json_path.resolve()),
        "markdown": str(md_path.resolve()),
        "csv": str(csv_path.resolve()),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit fixed V4 large-loss risk rules")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/v3/leader_candidates.csv"),
        help="V3 candidate table",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/v4/risk_rules"),
        help="Audit output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.input, args.output_dir)
    summary = {
        "audit_version": report["audit_version"],
        "status": report["status"],
        "decision": report["decision"],
        "population_n": report["population"]["n"],
        "large_loss_rate": report["population"]["large_loss_rate"],
        "stable_rule_ids": report["stable_rule_ids"],
        "output_paths": report["output_paths"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
