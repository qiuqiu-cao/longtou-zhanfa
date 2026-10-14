from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leader_system_v3 import is_mainboard, parse_clock, safe_float, zfill_code


ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = ROOT / "data" / "v4" / "open_confirmation_snapshots.csv"
DEFAULT_THEME_SYNC = ROOT / "data" / "v4" / "theme_open_sync.csv"
CANDIDATES = ROOT / "data" / "v3" / "leader_candidates.csv"
OUTPUT = ROOT / "results" / "v4" / "open_confirmation"
SNAPSHOT_TIMES = ("09:20", "09:24:50", "09:25", "09:30", "09:35", "10:00")
VERIFIED_STATUSES = {
    "verified", "verified_import", "verified_broker_export", "verified_terminal_export",
    "verified_derived_intraday_v9",
}
VERIFIED_THEME_BASES = {
    "verified_theme_registry",
}

OPEN_FIELDS = (
    "auction_gap", "auction_amount_ratio", "auction_cancel_ratio", "expectation_gap",
    "vwap_slope_5m", "vwap_slope_15m", "pullback_volume_ratio", "vwap_reclaimed",
    "reseal_minute", "theme_sync_open", "rivals_eliminated", "below_vwap_expanding",
    "failed_seal_twice",
)

EARLIEST_FIELD_TIME = {
    "auction_amount_ratio": "09:20",
    "auction_cancel_ratio": "09:24:50",
    "auction_gap": "09:25",
    "expectation_gap": "09:25",
    "vwap_slope_5m": "09:35",
    "pullback_volume_ratio": "09:35",
    "vwap_reclaimed": "09:35",
    "reseal_minute": "09:35",
    "theme_sync_open": "09:35",
    "rivals_eliminated": "09:35",
    "below_vwap_expanding": "09:35",
    "failed_seal_twice": "09:35",
    "vwap_slope_15m": "10:00",
}

REQUIRED_BY_ASOF = {
    "09:20": ("auction_amount_ratio",),
    "09:24:50": ("auction_amount_ratio", "auction_cancel_ratio"),
    "09:25": ("auction_gap", "auction_amount_ratio", "auction_cancel_ratio"),
    "09:30": ("auction_gap",),
    "09:35": (
        "auction_gap", "auction_amount_ratio", "auction_cancel_ratio", "vwap_slope_5m",
        "pullback_volume_ratio", "vwap_reclaimed",
        "reseal_minute", "theme_sync_open", "below_vwap_expanding", "failed_seal_twice",
    ),
    "10:00": (
        "auction_gap", "auction_amount_ratio", "auction_cancel_ratio", "vwap_slope_5m",
        "vwap_slope_15m", "pullback_volume_ratio",
        "vwap_reclaimed", "reseal_minute", "theme_sync_open", "rivals_eliminated",
        "below_vwap_expanding", "failed_seal_twice",
    ),
}


def clock_seconds(value: str) -> int:
    parts = [int(part) for part in str(value).split(":")]
    if len(parts) == 2:
        parts.append(0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def normalize_asof(value: Any) -> str:
    text = str(value).strip()
    if text in SNAPSHOT_TIMES:
        return text
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"invalid asof {value}")
    candidate = parsed.strftime("%H:%M:%S")
    candidate = candidate[:-3] if candidate.endswith(":00") else candidate
    if candidate not in SNAPSHOT_TIMES:
        raise ValueError(f"asof must be one of {SNAPSHOT_TIMES}; got {value}")
    return candidate


def _load_theme_sync_registry(source: Path | pd.DataFrame | None) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if source is None or not source.exists() or not source.stat().st_size:
        return pd.DataFrame()
    return pd.read_csv(source, dtype={"code": "string"})


def validate_snapshots(
    frame: pd.DataFrame,
    theme_sync_registry: Path | pd.DataFrame | None = DEFAULT_THEME_SYNC,
) -> pd.DataFrame:
    required = {
        "signal_date", "trade_date", "asof", "code", "name", "source",
        "source_timestamp", "data_status", "theme_sync_basis", "theme_evidence_id",
        "theme_group_id", "theme_member_coverage", "theme_observed_peers",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"snapshot input missing columns: {missing}")
    checked = frame.copy()
    checked["signal_date"] = pd.to_datetime(checked["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    checked["trade_date"] = pd.to_datetime(checked["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if checked[["signal_date", "trade_date"]].isna().any().any():
        raise ValueError("snapshot input contains invalid signal_date/trade_date")
    checked["asof"] = checked["asof"].map(normalize_asof)
    checked["code"] = checked["code"].map(zfill_code)
    if not checked.apply(lambda row: is_mainboard(row["code"], row["name"]), axis=1).all():
        raise ValueError("snapshot input contains non-mainboard/ST/delisting-risk code")
    if checked.duplicated(["signal_date", "trade_date", "asof", "code"]).any():
        raise ValueError("snapshot input contains duplicate signal/trade/asof/code keys")
    for field in OPEN_FIELDS:
        if field not in checked:
            checked[field] = math.nan
        checked[field] = pd.to_numeric(checked[field], errors="coerce")
    checked["theme_member_coverage"] = pd.to_numeric(checked["theme_member_coverage"], errors="coerce")
    checked["theme_observed_peers"] = pd.to_numeric(checked["theme_observed_peers"], errors="coerce")
    binary_fields = ("vwap_reclaimed", "below_vwap_expanding", "failed_seal_twice")
    for field in binary_fields:
        invalid = checked[field].notna() & ~checked[field].isin([0, 1])
        if invalid.any():
            raise ValueError(f"{field} must be 0/1 or blank")
    bounded = {
        "auction_gap": (-0.20, 0.20), "auction_amount_ratio": (0, 5),
        "auction_cancel_ratio": (0, 1), "pullback_volume_ratio": (0, 10),
        "theme_sync_open": (0, 1), "rivals_eliminated": (0, 1),
        "theme_member_coverage": (0, 1), "theme_observed_peers": (0, 1000),
    }
    for field, (low, high) in bounded.items():
        invalid = checked[field].notna() & ~checked[field].between(low, high)
        if invalid.any():
            raise ValueError(f"{field} outside allowed range [{low}, {high}]")
    for index, row in checked.iterrows():
        asof_seconds = clock_seconds(row["asof"])
        cutoff = pd.Timestamp(row["trade_date"]) + pd.Timedelta(seconds=asof_seconds)
        timestamp = pd.to_datetime(row["source_timestamp"], errors="coerce")
        if pd.isna(timestamp):
            raise ValueError("source_timestamp must be present and parseable")
        timestamp = pd.Timestamp(timestamp)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("Asia/Shanghai").tz_localize(None)
        if timestamp > cutoff:
            raise ValueError(f"row {index} source_timestamp is later than asof")
        if timestamp.date() != pd.Timestamp(row["trade_date"]).date():
            raise ValueError(f"row {index} source_timestamp date differs from trade_date")
        for field, earliest in EARLIEST_FIELD_TIME.items():
            if pd.notna(row[field]) and asof_seconds < clock_seconds(earliest):
                raise ValueError(f"row {index} field {field} is unavailable at {row['asof']} (future leakage)")
        reseal = safe_float(row.get("reseal_minute"))
        if not math.isnan(reseal):
            elapsed = max(0, (asof_seconds - clock_seconds("09:30")) / 60)
            if reseal < 0 or reseal > elapsed:
                raise ValueError(f"row {index} reseal_minute exceeds elapsed open time")
        if pd.notna(row.get("theme_sync_open")):
            basis = str(row.get("theme_sync_basis", "")).strip().lower()
            if basis not in VERIFIED_THEME_BASES:
                raise ValueError(f"row {index} theme_sync_open lacks a verified theme basis")
            if not str(row.get("theme_evidence_id", "")).strip() or not str(row.get("theme_group_id", "")).strip():
                raise ValueError(f"row {index} theme_sync_open lacks bound evidence/group id")
            if safe_float(row.get("theme_member_coverage")) < 0.67:
                raise ValueError(f"row {index} theme member coverage is below 67%")
            if safe_float(row.get("theme_observed_peers")) < 2:
                raise ValueError(f"row {index} theme sync has fewer than 2 observed peers")
    sync_rows = checked[checked["theme_sync_open"].notna()].copy()
    if not sync_rows.empty:
        registry = _load_theme_sync_registry(theme_sync_registry)
        if registry.empty:
            raise ValueError("verified theme sync registry is unavailable")
        keys = ["signal_date", "trade_date", "asof", "code"]
        required_registry = set(keys + [
            "theme_sync_open", "theme_sync_basis", "theme_evidence_id", "theme_group_id",
            "theme_member_coverage", "theme_observed_peers",
        ])
        if not required_registry.issubset(registry.columns):
            raise ValueError("verified theme sync registry is missing required linkage fields")
        registry = registry.copy()
        registry["code"] = registry["code"].map(zfill_code)
        registry["signal_date"] = pd.to_datetime(registry["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        registry["trade_date"] = pd.to_datetime(registry["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        registry["asof"] = registry["asof"].map(normalize_asof)
        if registry.duplicated(keys).any():
            raise ValueError("verified theme sync registry contains duplicate keys")
        lookup = registry.set_index(keys)
        for index, row in sync_rows.iterrows():
            key = tuple(str(row[field]) for field in keys)
            if key not in lookup.index:
                raise ValueError(f"row {index} has no matching verified theme sync registry row")
            bound = lookup.loc[key]
            if abs(safe_float(bound.get("theme_sync_open")) - safe_float(row.get("theme_sync_open"))) > 1e-9:
                raise ValueError(f"row {index} theme_sync_open differs from verified registry")
            for field in ("theme_sync_basis", "theme_evidence_id", "theme_group_id"):
                if str(bound.get(field, "")).strip() != str(row.get(field, "")).strip():
                    raise ValueError(f"row {index} {field} differs from verified registry")
            for field in ("theme_member_coverage", "theme_observed_peers"):
                if abs(safe_float(bound.get(field)) - safe_float(row.get(field))) > 1e-9:
                    raise ValueError(f"row {index} {field} differs from verified registry")
    checked["data_status"] = checked["data_status"].astype(str).str.lower()
    return checked


def import_snapshots(
    source: Path,
    archive: Path = DEFAULT_ARCHIVE,
    theme_sync_registry: Path | pd.DataFrame | None = DEFAULT_THEME_SYNC,
) -> dict[str, Any]:
    incoming = validate_snapshots(pd.read_csv(source, dtype={"code": str}), theme_sync_registry)
    keys = ["signal_date", "trade_date", "asof", "code"]
    if archive.exists():
        existing = validate_snapshots(pd.read_csv(archive, dtype={"code": str}), theme_sync_registry)
        existing_keys = set(map(tuple, existing[keys].astype(str).to_numpy()))
        duplicate = [tuple(row) for row in incoming[keys].astype(str).to_numpy() if tuple(row) in existing_keys]
        if duplicate:
            raise ValueError(f"immutable snapshot keys already exist: {duplicate[:5]}")
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming
    archive.parent.mkdir(parents=True, exist_ok=True)
    combined.sort_values(keys).to_csv(archive, index=False, encoding="utf-8-sig")
    return {"imported": len(incoming), "archive_rows": len(combined), "archive": str(archive)}


def preopen_eligibility(
    row: pd.Series,
    allow_height_confirmed_research: bool = False,
    allow_countertrend_research: bool = False,
) -> tuple[bool, list[str]]:
    reasons = []
    boards = int(safe_float(row.get("boards")))
    if boards < 1:
        reasons.append("板位必须为已核验的首板或连续板")
    identity_missing = safe_float(row.get("valid_identity")) != 1
    # First boards do not need an already established height-leader identity,
    # but they still need a verified pioneer/core identity from the close
    # review. Higher boards are not mechanically excluded: their risk is
    # handled by confirmation, sizing and announcement gates.
    role = str(row.get("competition_identity", "")).strip().lower()
    first_board_role_ok = boards == 1 and role in {
        "pioneer", "turnover_core", "unique_survivor", "capacity_core",
    }
    research_height_ok = allow_height_confirmed_research and boards >= 3
    countertrend_role_ok = (
        allow_countertrend_research
        and 2 <= boards <= 5
        and role in {"height_anchor", "unique_survivor", "turnover_core", "capacity_core", "pioneer"}
        and str(row.get("theme_status", "")) in {
            "verified_same_day_group", "verified_explicit_source_roster",
        }
    )
    if boards == 1 and not first_board_role_ok:
        reasons.append("首板须具备先锋或核心竞争身份")
    elif boards >= 2 and identity_missing and not research_height_ok:
        reasons.append("盘后身份闸门未通过")
    one_word = safe_float(row.get("one_word_proxy"))
    if allow_countertrend_research and math.isnan(one_word):
        reasons.append("逆情绪研究须明确验证前一日不是一字不可买")
    elif one_word == 1:
        reasons.append("前一日一字不可买，仅作情绪锚")
    if allow_countertrend_research and not countertrend_role_ok:
        reasons.append("逆情绪研究仅限2至5板且具有已核验核心竞争身份和具体题材")
    if str(row.get("emotion_state")) in {"大分歧", "退潮"} and not countertrend_role_ok:
        reasons.append(f"盘后情绪为{row.get('emotion_state')}")
    if safe_float(row.get("announcement_risk")) >= 0.8:
        reasons.append("公告硬风险")
    return not reasons, reasons


def required_fields(asof: str) -> tuple[str, ...]:
    return REQUIRED_BY_ASOF[normalize_asof(asof)]


def evaluate_row(
    candidate: pd.Series,
    snapshot: pd.Series | None,
    asof: str,
    allow_height_confirmed_research: bool = False,
    allow_countertrend_research: bool = False,
) -> dict[str, Any]:
    eligible, reasons = preopen_eligibility(
        candidate, allow_height_confirmed_research, allow_countertrend_research,
    )
    base = {
        "signal_date": str(candidate.get("date", "")),
        "trade_date": "" if snapshot is None else snapshot.get("trade_date", ""),
        "asof": asof,
        "code": zfill_code(candidate.get("code")),
        "name": candidate.get("name", ""),
        "boards": int(safe_float(candidate.get("boards"))),
        "emotion_state": candidate.get("emotion_state", ""),
        "preopen_eligible": eligible,
        "evidence_level": "VERIFIED_OPEN_SNAPSHOT_REQUIRED",
        "execution_mode": "PAPER_ONLY",
        "board_scope_policy": "ALL_VERIFIED_FIRST_AND_CONSECUTIVE_BOARDS",
        "identity_policy": "HEIGHT_CONFIRMED_RESEARCH" if allow_height_confirmed_research else "STRICT_IDENTITY",
        "countertrend_policy": "RESEARCH_ONLY" if allow_countertrend_research else "DISABLED",
    }
    if not eligible:
        return {**base, "confirmation_status": "INELIGIBLE", "missing_fields": "",
                "reasons": "；".join(reasons), "entry_contract": "禁止进入开盘确认"}
    if snapshot is None:
        return {**base, "confirmation_status": "DATA_LIMITED",
                "missing_fields": ",".join(required_fields(asof)),
                "reasons": "没有对应时点的真实开盘快照", "entry_contract": "仅观察，不得确认"}
    data_status = str(snapshot.get("data_status", "")).lower()
    needed = required_fields(asof)
    missing = [field for field in needed if math.isnan(safe_float(snapshot.get(field)))]
    theme_binding_invalid = False
    if data_status not in VERIFIED_STATUSES:
        reasons.append(f"数据状态{data_status or 'missing'}不是可核验实时导入")
    if pd.notna(snapshot.get("theme_sync_open")):
        if str(snapshot.get("theme_sync_basis", "")).strip().lower() != "verified_theme_registry":
            reasons.append("题材同步没有绑定冻结成员名册")
            theme_binding_invalid = True
        if not str(snapshot.get("theme_evidence_id", "")).strip():
            reasons.append("题材同步缺少证据ID")
            theme_binding_invalid = True
        if safe_float(snapshot.get("theme_member_coverage")) < 0.67:
            reasons.append("题材成员覆盖不足67%")
            theme_binding_invalid = True
        if safe_float(snapshot.get("theme_observed_peers")) < 2:
            reasons.append("题材同步观察成员少于2只")
            theme_binding_invalid = True
    if missing:
        reasons.append("关键字段缺失")
    if data_status not in VERIFIED_STATUSES or missing or theme_binding_invalid:
        return {**base, "trade_date": snapshot.get("trade_date", ""),
                "confirmation_status": "DATA_LIMITED", "missing_fields": ",".join(missing),
                "reasons": "；".join(reasons), "entry_contract": "仅观察，不得确认"}

    gap = safe_float(snapshot.get("auction_gap"))
    amount_ratio = safe_float(snapshot.get("auction_amount_ratio"))
    cancel_ratio = safe_float(snapshot.get("auction_cancel_ratio"))
    slope5 = safe_float(snapshot.get("vwap_slope_5m"))
    slope15 = safe_float(snapshot.get("vwap_slope_15m"))
    pullback = safe_float(snapshot.get("pullback_volume_ratio"))
    reclaimed = safe_float(snapshot.get("vwap_reclaimed"))
    reseal = safe_float(snapshot.get("reseal_minute"))
    sync = safe_float(snapshot.get("theme_sync_open"))
    rivals = safe_float(snapshot.get("rivals_eliminated"))
    below = safe_float(snapshot.get("below_vwap_expanding"))
    failed_twice = safe_float(snapshot.get("failed_seal_twice"))

    cancel_reasons = []
    if failed_twice == 1:
        cancel_reasons.append("两次冲板不封")
    if below == 1:
        cancel_reasons.append("跌破VWAP且卖量扩大")
    if not math.isnan(sync) and sync < 0.35:
        cancel_reasons.append("真实题材同步率低于35%")
    if not math.isnan(cancel_ratio) and cancel_ratio > 0.45 and gap > 0.05:
        cancel_reasons.append("高开同时竞价匹配额衰减超过45%")
    if not math.isnan(amount_ratio) and amount_ratio < 0.02 and gap > 0.05:
        cancel_reasons.append("高开超过5%但竞价成交占比不足2%")
    if not math.isnan(pullback) and pullback > 1.25 and reclaimed == 0:
        cancel_reasons.append("首次回落放量且未收复VWAP")
    if gap > 0.07 and reclaimed == 0:
        cancel_reasons.append("高开超过7%后未收复VWAP")
    if cancel_reasons:
        return {**base, "trade_date": snapshot.get("trade_date", ""),
                "confirmation_status": "CANCEL", "missing_fields": "",
                "reasons": "；".join(cancel_reasons), "entry_contract": "取消接力，禁止补仓"}

    if clock_seconds(asof) < clock_seconds("09:35"):
        status = "WATCH_AUCTION"
        reasons.append("竞价只能筛除风险，不能完成换手回封确认")
    else:
        common = (
            amount_ratio >= 0.03 and slope5 > 0 and pullback <= 0.75
            and reclaimed == 1 and sync >= 0.60 and not math.isnan(reseal)
        )
        if asof == "10:00":
            common = common and slope15 >= 0 and rivals >= 0.50
        normal = 0 <= gap <= 0.05 and common
        weak_to_strong = gap < 0 and gap >= -0.03 and common and sync >= 0.70
        hot = 0.05 < gap <= 0.07 and common and pullback <= 0.60 and sync >= 0.70 and reseal <= 20
        exceptional_hot = (
            gap > 0.07 and common and pullback <= 0.50 and sync >= 0.80
            and reseal <= 10 and (math.isnan(cancel_ratio) or cancel_ratio <= 0.20)
        )
        countertrend = (
            allow_countertrend_research
            and -.01 <= gap <= .05
            and amount_ratio >= .05
            and not math.isnan(cancel_ratio) and cancel_ratio <= .20
            and slope5 > 0 and pullback <= .60 and reclaimed == 1
            and sync >= .70 and rivals >= .50 and not math.isnan(reseal) and reseal <= 15
            and below == 0 and failed_twice == 0
            and (asof != "10:00" or slope15 >= 0)
        )
        riskoff_countertrend_path = (
            allow_countertrend_research
            and str(candidate.get("emotion_state")) in {"大分歧", "退潮"}
        )
        if countertrend:
            status = "CONFIRM_COUNTERTREND_RESEAL_RESEARCH"
            reasons.append("逆情绪核心通过更严格的竞价、VWAP、题材、竞争者淘汰与回封链")
        elif riskoff_countertrend_path:
            status = "WATCH_COUNTERTREND_NOT_CONFIRMED"
            reasons.append("逆情绪核心未同时通过更严格的成交、匹配额净变化、VWAP、题材和竞争确认")
        elif normal or weak_to_strong or hot or exceptional_hot:
            status = "CONFIRM_RESEAL"
            reasons.append("真实换手、VWAP收复、题材同步与回封条件同时成立")
        elif gap > 0.07:
            status = "WATCH_OVERHEATED"
            reasons.append("高开超过7%，未达到极强承接例外条件")
        else:
            status = "WATCH"
            reasons.append("尚未形成完整的缩量回落—VWAP收复—题材同步—回封链")
    contract = (
        "仅纸面确认；必须在涨停价可成交且封单真实稳定时记录成交"
        if status in {"CONFIRM_RESEAL", "CONFIRM_COUNTERTREND_RESEAL_RESEARCH"}
        else "继续观察，不得视为买入确认"
    )
    return {**base, "trade_date": snapshot.get("trade_date", ""),
            "confirmation_status": status, "missing_fields": "",
            "reasons": "；".join(reasons), "entry_contract": contract}


def evaluate(
    signal_date: str,
    trade_date: str,
    asof: str,
    archive: Path = DEFAULT_ARCHIVE,
    candidates_path: Path = CANDIDATES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    signal = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
    trade = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    point = normalize_asof(asof)
    candidates = pd.read_csv(candidates_path, dtype={"code": str}, low_memory=False)
    candidates = candidates[candidates["date"].astype(str) == signal].copy()
    if candidates.empty:
        raise ValueError(f"no candidates for signal date {signal}")
    snapshots = pd.DataFrame()
    if archive.exists() and archive.stat().st_size:
        snapshots = validate_snapshots(pd.read_csv(archive, dtype={"code": str}))
        snapshots = snapshots[
            (snapshots["signal_date"] == signal)
            & (snapshots["trade_date"] == trade)
            & (snapshots["asof"].map(clock_seconds) <= clock_seconds(point))
        ].sort_values("asof").drop_duplicates("code", keep="last")
    snapshot_map = snapshots.set_index("code").to_dict("index") if not snapshots.empty else {}
    rows = [evaluate_row(candidate, pd.Series(snapshot_map[code]) if code in snapshot_map else None, point)
            for _, candidate in candidates.iterrows() for code in [zfill_code(candidate["code"])]]
    result = pd.DataFrame(rows)
    counts = result["confirmation_status"].value_counts().to_dict()
    manifest = {
        "signal_date": signal, "trade_date": trade, "asof": point,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_rows": len(result), "snapshot_rows": len(snapshots),
        "status_counts": {str(key): int(value) for key, value in counts.items()},
        "verified_confirmation_rows": int((result["confirmation_status"] == "CONFIRM_RESEAL").sum()),
        "decision": "paper-only conditional confirmation",
        "warning": "no Level-2 queue or actual fill is inferred; industry proxy cannot fill theme_sync_open",
    }
    return result, manifest


def write_report(result: pd.DataFrame, manifest: dict[str, Any], output: Path = OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{manifest['signal_date']}_to_{manifest['trade_date']}_{manifest['asof'].replace(':', '')}"
    csv_path = output / f"{stem}.csv"
    json_path = output / f"{stem}.json"
    md_path = output / f"{stem}.md"
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    payload = {**manifest, "rows": result.to_dict("records")}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        f"# {manifest['trade_date']} {manifest['asof']} 开盘确认", "",
        f"- 盘后信号日：{manifest['signal_date']}",
        f"- 快照行数：{manifest['snapshot_rows']}",
        f"- 状态统计：{manifest['status_counts']}",
        "- 只有真实时点快照可以确认；日线开盘、行业代理、缺失值均不能替代。", "",
        "| 代码 | 名称 | 板位 | 状态 | 原因 | 执行契约 |", "|---|---|---:|---|---|---|",
    ]
    for row in result.sort_values(["boards", "code"], ascending=[False, True]).itertuples(index=False):
        lines.append(f"| {row.code} | {row.name} | {row.boards} | {row.confirmation_status} | {row.reasons} | {row.entry_contract} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "report": str(md_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict point-in-time auction/open confirmation for board relay")
    sub = parser.add_subparsers(dest="command", required=True)
    imported = sub.add_parser("import")
    imported.add_argument("file", type=Path)
    imported.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    validate = sub.add_parser("validate")
    validate.add_argument("file", type=Path)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--signal-date", required=True)
    evaluate_parser.add_argument("--trade-date", required=True)
    evaluate_parser.add_argument("--asof", required=True, choices=SNAPSHOT_TIMES)
    evaluate_parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    try:
        if args.command == "import":
            payload = import_snapshots(args.file, args.archive)
        elif args.command == "validate":
            checked = validate_snapshots(pd.read_csv(args.file, dtype={"code": str}))
            payload = {"status": "PASS", "rows": len(checked)}
        else:
            result, manifest = evaluate(args.signal_date, args.trade_date, args.asof, args.archive)
            payload = {**manifest, "files": write_report(result, manifest)}
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
