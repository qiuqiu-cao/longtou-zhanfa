#!/usr/bin/env python3
"""Point-in-time, auditable theme/catalyst evidence for A-share board relay.

The frozen V3 scorer is intentionally untouched.  This companion layer turns
primary-source or terminal-verified theme membership into an immutable registry,
then derives candidate roles and intraday theme synchronisation without using an
industry label as a substitute for a real catalyst.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from leader_system_v3 import is_mainboard, zfill_code


ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY = ROOT / "data" / "v4" / "theme_evidence_registry.csv"
DEFAULT_CANDIDATES = ROOT / "data" / "v3" / "leader_candidates.csv"
DEFAULT_ROSTER = ROOT / "data" / "v4" / "verified_theme_roster.csv"
DEFAULT_OPEN_SYNC = ROOT / "data" / "v4" / "theme_open_sync.csv"
OUTPUT_ROOT = ROOT / "results" / "v4" / "themes"
VERSION = "v4-theme-evidence-1"

TRUSTED_SOURCE_TYPES = {
    "exchange_announcement": "A",
    "company_announcement": "A",
    "government_policy": "A",
    "regulator_release": "A",
    "official_media": "B",
    "broker_terminal": "B",
    "verified_market_concept_snapshot": "B",
}
TRUSTED_MEMBERSHIP_BASES = {
    "explicit_company_disclosure",
    "official_policy_scope",
    "verified_concept_membership",
    "broker_theme_classification",
    "manual_primary_source_mapping",
}
VERIFIED_EVIDENCE_STATUSES = {"verified", "verified_import", "verified_terminal_export"}
VERIFIED_OPEN_STATUSES = {"verified", "verified_import", "verified_broker_export", "verified_terminal_export"}
GENERIC_THEME_TOKENS = (
    "融资融券", "沪股通", "深股通", "MSCI", "富时罗素", "机构重仓", "基金重仓",
    "昨日涨停", "昨日连板", "低价股", "高股息", "国企改革", "预亏预减",
)
BROAD_THEME_GRANULARITIES = {
    "板块题材", "行业板块", "宽泛概念", "一级行业", "大类题材",
}
REGISTRY_COLUMNS = (
    "signal_date", "code", "name", "theme_group_id", "theme", "theme_granularity",
    "driver", "evidence_kind", "source_type", "source_ref", "source_published_at",
    "timestamp_precision", "membership_basis", "evidence_status", "collected_at",
)
OPEN_MEMBER_COLUMNS = (
    "signal_date", "trade_date", "asof", "theme_group_id", "member_code", "member_name",
    "supportive", "source", "source_timestamp", "data_status",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_id(row: pd.Series) -> str:
    values = [str(row.get(field, "")).strip() for field in REGISTRY_COLUMNS]
    return hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()[:20]


def is_generic_theme(value: Any) -> bool:
    text = str(value).strip()
    return not text or any(token.lower() in text.lower() for token in GENERIC_THEME_TOKENS)


def _normalize_timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    parsed = pd.Timestamp(parsed)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("Asia/Shanghai").tz_localize(None)
    return parsed


def validate_evidence(frame: pd.DataFrame, decision_clock: str = "15:00") -> pd.DataFrame:
    missing = sorted(set(REGISTRY_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"theme evidence missing columns: {missing}")
    checked = frame.loc[:, REGISTRY_COLUMNS].copy()
    checked["signal_date"] = pd.to_datetime(checked["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if checked["signal_date"].isna().any():
        raise ValueError("theme evidence contains invalid signal_date")
    checked["code"] = checked["code"].map(zfill_code)
    if not checked.apply(lambda row: is_mainboard(row["code"], row["name"]), axis=1).all():
        raise ValueError("theme evidence contains non-mainboard/ST/delisting-risk code")
    required_text = (
        "name", "theme_group_id", "theme", "theme_granularity", "driver", "evidence_kind",
        "source_type", "source_ref", "timestamp_precision", "membership_basis",
        "evidence_status", "collected_at",
    )
    for field in required_text:
        checked[field] = checked[field].astype("string").str.strip()
        if checked[field].isna().any() or checked[field].eq("").any():
            raise ValueError(f"theme evidence field {field} cannot be blank")
    if checked["theme_group_id"].str.contains(r"[^a-zA-Z0-9_.:-]", regex=True).any():
        raise ValueError("theme_group_id must use stable ASCII letters/numbers/_.:-")
    if checked["theme_group_id"].str.len().gt(80).any():
        raise ValueError("theme_group_id is too long")
    if checked["theme"].map(is_generic_theme).any():
        raise ValueError("generic style/index concepts cannot be verified trade themes")
    checked["source_type"] = checked["source_type"].str.lower()
    checked["timestamp_precision"] = checked["timestamp_precision"].str.lower()
    checked["membership_basis"] = checked["membership_basis"].str.lower()
    checked["evidence_status"] = checked["evidence_status"].str.lower()
    checked["evidence_kind"] = checked["evidence_kind"].str.upper()
    if (~checked["evidence_kind"].isin(["CATALYST", "MEMBERSHIP"])).any():
        raise ValueError("evidence_kind must be CATALYST or MEMBERSHIP")
    if (~checked["timestamp_precision"].isin(["exact", "date_only"])).any():
        raise ValueError("timestamp_precision must be exact or date_only")
    published = checked["source_published_at"].map(_normalize_timestamp)
    collected = checked["collected_at"].map(_normalize_timestamp)
    if published.isna().any() or collected.isna().any():
        raise ValueError("source_published_at/collected_at must be parseable")
    exact = checked["timestamp_precision"].eq("exact")
    invalid_collection_order = (exact & (collected < published)) | (
        ~exact & (collected.dt.normalize() < published.dt.normalize())
    )
    if invalid_collection_order.any():
        raise ValueError("collected_at cannot precede source publication date/time")
    date_only = ~exact
    published.loc[date_only] = published.loc[date_only].dt.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    cutoff = pd.to_datetime(checked["signal_date"] + " " + decision_clock, errors="raise")
    trusted_source = checked["source_type"].isin(TRUSTED_SOURCE_TYPES)
    trusted_membership = checked["membership_basis"].isin(TRUSTED_MEMBERSHIP_BASES)
    verified_status = checked["evidence_status"].isin(VERIFIED_EVIDENCE_STATUSES)
    known_by_cutoff = published <= cutoff
    checked["effective_published_at"] = published.dt.strftime("%Y-%m-%d %H:%M:%S")
    checked["decision_cutoff"] = cutoff.dt.strftime("%Y-%m-%d %H:%M:%S")
    checked["evidence_tier"] = checked["source_type"].map(TRUSTED_SOURCE_TYPES).fillna("UNTRUSTED")
    checked["usable_at_signal"] = trusted_source & trusted_membership & verified_status & known_by_cutoff
    reasons = []
    for idx in checked.index:
        row_reasons = []
        if not trusted_source.loc[idx]:
            row_reasons.append("untrusted_source_type")
        if not trusted_membership.loc[idx]:
            row_reasons.append("untrusted_membership_basis")
        if not verified_status.loc[idx]:
            row_reasons.append("evidence_not_verified")
        if not known_by_cutoff.loc[idx]:
            row_reasons.append("not_known_by_signal_cutoff")
        reasons.append("usable" if not row_reasons else ";".join(row_reasons))
    checked["usable_reason"] = reasons
    checked["evidence_id"] = checked.apply(event_id, axis=1)
    if checked.duplicated(["evidence_id"]).any():
        raise ValueError("theme evidence input contains exact duplicate evidence rows")
    return checked


def import_evidence(source: Path, registry: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    incoming = validate_evidence(pd.read_csv(source, dtype={"code": "string"}))
    if registry.exists() and registry.stat().st_size:
        existing = pd.read_csv(registry, dtype={"code": "string"})
        existing_ids = set(existing["evidence_id"].astype(str))
        duplicates = sorted(existing_ids.intersection(incoming["evidence_id"].astype(str)))
        if duplicates:
            raise ValueError(f"immutable evidence_ids already exist: {duplicates[:5]}")
        # A source reference is immutable for one date/code/theme even if another
        # field is later edited.  Corrections require a new source reference.
        key = ["signal_date", "code", "theme_group_id", "source_ref"]
        old_keys = set(map(tuple, existing[key].astype(str).to_numpy()))
        conflicts = [tuple(row) for row in incoming[key].astype(str).to_numpy() if tuple(row) in old_keys]
        if conflicts:
            raise ValueError(f"source reference already registered with different content: {conflicts[:5]}")
        combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    else:
        combined = incoming
    registry.parent.mkdir(parents=True, exist_ok=True)
    combined.sort_values(["signal_date", "theme_group_id", "code", "evidence_id"]).to_csv(
        registry, index=False, encoding="utf-8-sig"
    )
    return {
        "version": VERSION,
        "imported": int(len(incoming)),
        "usable_at_signal": int(incoming["usable_at_signal"].sum()),
        "archive_rows": int(len(combined)),
        "registry": str(registry.resolve()),
        "registry_sha256": sha256_file(registry),
    }


def load_limit_pool(signal_date: str, path: Path | None = None) -> pd.DataFrame:
    path = path or ROOT / "data" / "v3" / "raw" / signal_date.replace("-", "") / "limit_up.csv"
    if not path.exists():
        raise FileNotFoundError(f"limit-up pool not found: {path}")
    pool = pd.read_csv(path, dtype={"代码": "string", "code": "string"})
    rename = {"代码": "code", "名称": "name", "连板数": "boards", "所属行业": "industry"}
    pool = pool.rename(columns=rename)
    required = {"code", "name", "boards"}
    if not required.issubset(pool.columns):
        raise ValueError(f"limit-up pool missing columns: {sorted(required - set(pool.columns))}")
    pool["code"] = pool["code"].map(zfill_code)
    pool["boards"] = pd.to_numeric(pool["boards"], errors="coerce")
    pool["signal_date"] = signal_date
    return pool


def _tier_rank(value: Any) -> int:
    return {"A": 2, "B": 1}.get(str(value), 0)


def build_theme_roster(
    signal_date: str,
    registry: pd.DataFrame,
    limit_pool: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_date = pd.Timestamp(signal_date).date().isoformat()
    evidence = registry.copy()
    if not evidence.empty:
        evidence = evidence[
            evidence["signal_date"].astype(str).eq(signal_date)
            & evidence["usable_at_signal"].astype(str).str.lower().isin(["true", "1"])
        ].copy()
    pool = limit_pool.copy()
    pool["code"] = pool["code"].map(zfill_code)
    pool = pool.drop_duplicates("code")
    candidates = candidates.copy()
    candidates["code"] = candidates["code"].map(zfill_code)
    candidates = candidates[pd.to_datetime(candidates["date"], errors="coerce").dt.strftime("%Y-%m-%d").eq(signal_date)]
    candidate_cols = ["date", "code", "name", "boards", "valid_identity", "buyable_eod", "emotion_state"]
    for field in candidate_cols:
        if field not in candidates:
            candidates[field] = math.nan
    if evidence.empty:
        unavailable = candidates[candidate_cols].copy()
        unavailable["theme_group_id"] = ""
        unavailable["theme"] = ""
        unavailable["theme_status"] = "unavailable"
        unavailable["theme_role"] = "UNVERIFIED"
        unavailable["theme_member_count"] = math.nan
        unavailable["theme_support_ex_self"] = math.nan
        unavailable["theme_first_support_ex_self"] = math.nan
        unavailable["theme_middle_support_ex_self"] = math.nan
        unavailable["theme_high_support_ex_self"] = math.nan
        unavailable["same_height_rivals"] = math.nan
        unavailable["theme_max_board"] = math.nan
        unavailable["theme_ladder_verified"] = False
        unavailable["primary_evidence_id"] = ""
        unavailable["source_tier"] = "UNAVAILABLE"
        unavailable["warning"] = "no point-in-time verified theme evidence; industry proxy is not substituted"
        return pd.DataFrame(), unavailable
    evidence["code"] = evidence["code"].map(zfill_code)
    # Multiple source rows for one membership collapse only after preserving the
    # strongest source and all evidence IDs for audit.
    grouped_evidence = []
    for (group_id, code), part in evidence.groupby(["theme_group_id", "code"], sort=True):
        part = part.copy()
        part["_tier_rank"] = part["evidence_tier"].map(_tier_rank)
        best = part.sort_values(["_tier_rank", "effective_published_at"], ascending=[False, True]).iloc[0]
        grouped_evidence.append({
            "theme_group_id": group_id,
            "code": code,
            "theme": best["theme"],
            "theme_granularity": best["theme_granularity"],
            "driver": best["driver"],
            "source_tier": best["evidence_tier"],
            "primary_evidence_id": best["evidence_id"],
            "all_evidence_ids": ";".join(sorted(part["evidence_id"].astype(str).unique())),
        })
    membership = pd.DataFrame(grouped_evidence)
    roster = membership.merge(pool[["signal_date", "code", "name", "boards"]], on="code", how="inner")
    if roster.empty:
        return roster, build_theme_roster(signal_date, pd.DataFrame(), pool, candidates)[1]
    summaries = []
    for group_id, part in roster.groupby("theme_group_id", sort=True):
        boards = pd.to_numeric(part["boards"], errors="coerce")
        granularities = set(part["theme_granularity"].astype(str).str.strip())
        specific_enough = bool(granularities) and not bool(granularities.intersection(BROAD_THEME_GRANULARITIES))
        summaries.append({
            "theme_group_id": group_id,
            "theme_specific_enough": specific_enough,
            "theme_member_count": int(part["code"].nunique()),
            "theme_first_count": int(boards.eq(1).sum()),
            "theme_middle_count": int(boards.between(2, 4).sum()),
            "theme_high_count": int(boards.ge(5).sum()),
            "theme_max_board": int(boards.max()),
            "theme_height_levels": ",".join(str(int(v)) for v in sorted(boards.dropna().unique())),
        })
    summary = pd.DataFrame(summaries)
    summary["theme_ladder_verified"] = (
        summary["theme_specific_enough"]
        &
        summary["theme_member_count"].ge(3)
        & summary["theme_first_count"].ge(1)
        & (summary["theme_middle_count"].ge(1) | summary["theme_high_count"].ge(1))
    )
    roster = roster.merge(summary, on="theme_group_id", how="left")
    roster["signal_date"] = signal_date
    rows = []
    for candidate in candidates[candidate_cols].itertuples(index=False):
        matches = roster[roster["code"].eq(candidate.code)]
        if matches.empty:
            rows.append({
                **dict(zip(candidate_cols, candidate)),
                "theme_group_id": "", "theme": "", "theme_status": "unavailable",
                "theme_role": "UNVERIFIED", "theme_member_count": math.nan,
                "theme_support_ex_self": math.nan, "theme_first_support_ex_self": math.nan,
                "theme_middle_support_ex_self": math.nan, "theme_high_support_ex_self": math.nan,
                "same_height_rivals": math.nan, "theme_max_board": math.nan,
                "theme_ladder_verified": False, "primary_evidence_id": "",
                "source_tier": "UNAVAILABLE",
                "warning": "candidate has no usable point-in-time theme membership",
            })
            continue
        options = []
        for match in matches.itertuples(index=False):
            group = roster[roster["theme_group_id"].eq(match.theme_group_id)]
            peers = group[~group["code"].eq(candidate.code)]
            candidate_board = int(float(candidate.boards))
            support = int(peers["code"].nunique())
            first = int(pd.to_numeric(peers["boards"], errors="coerce").eq(1).sum())
            middle = int(pd.to_numeric(peers["boards"], errors="coerce").between(2, 4).sum())
            high = int(pd.to_numeric(peers["boards"], errors="coerce").ge(5).sum())
            rivals = int(pd.to_numeric(peers["boards"], errors="coerce").eq(candidate_board).sum())
            verified = int(match.theme_member_count) >= 2 and support >= 1
            if not verified:
                role = "ISOLATED"
            elif not bool(match.theme_specific_enough):
                role = "BROAD_THEME_REFERENCE"
            elif candidate_board == int(match.theme_max_board):
                if rivals >= 1:
                    role = "CO_HEIGHT_CLUSTER"
                else:
                    role = "HEIGHT_CORE" if support >= 2 else "PIONEER"
            elif candidate_board >= 2:
                role = "MIDDLE_CORE" if middle + high >= 1 else "FOLLOWER"
            else:
                role = "FIRST_BOARD_SUPPORT"
            options.append({
                **dict(zip(candidate_cols, candidate)),
                "theme_group_id": match.theme_group_id,
                "theme": match.theme,
                "theme_status": (
                    "verified_same_day_group" if verified and bool(match.theme_specific_enough)
                    else "verified_same_day_broad_group" if verified
                    else "verified_membership_isolated"
                ),
                "theme_role": role,
                "theme_member_count": int(match.theme_member_count),
                "theme_support_ex_self": support,
                "theme_first_support_ex_self": first,
                "theme_middle_support_ex_self": middle,
                "theme_high_support_ex_self": high,
                "same_height_rivals": rivals,
                "theme_max_board": int(match.theme_max_board),
                "theme_ladder_verified": bool(match.theme_ladder_verified),
                "theme_specific_enough": bool(match.theme_specific_enough),
                "primary_evidence_id": match.primary_evidence_id,
                "source_tier": match.source_tier,
                "warning": (
                    "" if verified and bool(match.theme_specific_enough)
                    else "broad theme cannot supply positive identity or open-sync evidence"
                    if verified
                    else "membership verified but no same-day peer support"
                ),
                "_sort_verified": int(verified),
                "_sort_ladder": int(bool(match.theme_ladder_verified)),
                "_sort_support": support,
                "_sort_tier": _tier_rank(match.source_tier),
            })
        options.sort(
            key=lambda x: (x["_sort_verified"], x["_sort_ladder"], x["_sort_support"], x["_sort_tier"]),
            reverse=True,
        )
        selected = options[0]
        for field in ["_sort_verified", "_sort_ladder", "_sort_support", "_sort_tier"]:
            selected.pop(field, None)
        rows.append(selected)
    return roster.sort_values(["theme_group_id", "boards", "code"], ascending=[True, False, True]), pd.DataFrame(rows)


def validate_open_members(frame: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(OPEN_MEMBER_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"theme open member snapshot missing columns: {missing}")
    checked = frame.loc[:, OPEN_MEMBER_COLUMNS].copy()
    for field in ["signal_date", "trade_date"]:
        checked[field] = pd.to_datetime(checked[field], errors="coerce").dt.strftime("%Y-%m-%d")
    if checked[["signal_date", "trade_date"]].isna().any().any():
        raise ValueError("invalid signal_date/trade_date in theme open member snapshot")
    checked["asof"] = checked["asof"].astype(str).str.strip()
    if (~checked["asof"].isin(["09:35", "10:00"])).any():
        raise ValueError("theme sync is only valid at 09:35 or 10:00")
    checked["member_code"] = checked["member_code"].map(zfill_code)
    checked["supportive"] = pd.to_numeric(checked["supportive"], errors="coerce")
    if checked["supportive"].isna().any() or (~checked["supportive"].isin([0, 1])).any():
        raise ValueError("supportive must be verified 0/1, never blank")
    checked["data_status"] = checked["data_status"].astype(str).str.lower()
    if (~checked["data_status"].isin(VERIFIED_OPEN_STATUSES)).any():
        raise ValueError("theme open member snapshot must have verified data_status")
    if checked.duplicated(["signal_date", "trade_date", "asof", "theme_group_id", "member_code"]).any():
        raise ValueError("duplicate theme member open snapshot keys")
    roster_keys = set(map(tuple, roster[["signal_date", "theme_group_id", "code"]].astype(str).to_numpy()))
    unknown = [
        tuple(row) for row in checked[["signal_date", "theme_group_id", "member_code"]].astype(str).to_numpy()
        if tuple(row) not in roster_keys
    ]
    if unknown:
        raise ValueError(f"open snapshot contains member outside verified signal-date roster: {unknown[:5]}")
    timestamps = checked["source_timestamp"].map(_normalize_timestamp)
    if timestamps.isna().any():
        raise ValueError("source_timestamp must be parseable")
    cutoffs = pd.to_datetime(checked["trade_date"] + " " + checked["asof"], errors="raise")
    if (timestamps > cutoffs).any():
        raise ValueError("theme member source_timestamp is later than asof")
    if any(ts.date() != pd.Timestamp(day).date() for ts, day in zip(timestamps, checked["trade_date"])):
        raise ValueError("theme member source_timestamp date differs from trade_date")
    return checked


def compute_open_sync(open_members: pd.DataFrame, roster: pd.DataFrame, candidate_roles: pd.DataFrame) -> pd.DataFrame:
    checked = validate_open_members(open_members, roster)
    rows = []
    for key, part in checked.groupby(["signal_date", "trade_date", "asof", "theme_group_id"], sort=True):
        signal_date, trade_date, asof, group_id = key
        expected = roster[
            roster["signal_date"].astype(str).eq(signal_date)
            & roster["theme_group_id"].astype(str).eq(group_id)
        ]
        candidates = candidate_roles[
            candidate_roles["date"].astype(str).eq(signal_date)
            & candidate_roles["theme_group_id"].astype(str).eq(group_id)
        ]
        for candidate in candidates.itertuples(index=False):
            peer_expected = expected[~expected["code"].eq(candidate.code)]
            peer_observed = part[~part["member_code"].eq(candidate.code)]
            expected_n = int(peer_expected["code"].nunique())
            observed_n = int(peer_observed["member_code"].nunique())
            coverage = observed_n / expected_n if expected_n else math.nan
            identity_usable = (
                str(getattr(candidate, "theme_status", "")) == "verified_same_day_group"
                and bool(getattr(candidate, "theme_specific_enough", False))
            )
            usable = identity_usable and expected_n >= 2 and observed_n >= 2 and coverage >= 0.67
            sync = float(peer_observed["supportive"].mean()) if usable else math.nan
            rows.append({
                "signal_date": signal_date,
                "trade_date": trade_date,
                "asof": asof,
                "code": candidate.code,
                "name": candidate.name,
                "theme_group_id": group_id,
                "theme": candidate.theme,
                "theme_evidence_id": candidate.primary_evidence_id,
                "theme_sync_open": sync,
                "theme_sync_basis": "verified_theme_registry" if usable else "insufficient_verified_theme_members",
                "theme_expected_peers": expected_n,
                "theme_observed_peers": observed_n,
                "theme_member_coverage": coverage,
                "theme_supportive_peers": int(peer_observed["supportive"].sum()),
                "data_status": "verified_derived_from_member_snapshots" if usable else "unavailable",
                "source_timestamp": part["source_timestamp"].max(),
                "warning": (
                    "" if usable
                    else "broad or unverified theme cannot supply open-sync evidence"
                    if not identity_usable
                    else "need at least 2 observed peers and 67% roster coverage"
                ),
            })
    return pd.DataFrame(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def render_report(signal_date: str, roster: pd.DataFrame, candidates: pd.DataFrame, metadata: dict[str, Any]) -> str:
    verified = candidates["theme_status"].eq("verified_same_day_group") if not candidates.empty else pd.Series(dtype=bool)
    lines = [
        f"# {signal_date} 真实题材证据报告",
        "",
        f"- 状态：`{metadata['status']}`",
        f"- 可用原始证据：{metadata['usable_evidence_rows']} 条",
        f"- 题材涨停成员：{len(roster)} 条",
        f"- 候选覆盖：{int(verified.sum())}/{len(candidates)}",
        "- 证据纪律：行业代理、社区热度、收益截图和无时点来源均不能升级为真实题材。",
        "",
        "| 代码 | 名称 | 板位 | 题材 | 角色 | 去自身支持 | 首板/中位/高位支持 | 梯队 | 证据级别 | 状态 |",
        "|---|---|---:|---|---|---:|---|---|---|---|",
    ]
    for row in candidates.itertuples(index=False):
        def show(value: Any) -> str:
            return "NA" if pd.isna(value) or value == "" else str(value)
        supports = f"{show(row.theme_first_support_ex_self)}/{show(row.theme_middle_support_ex_self)}/{show(row.theme_high_support_ex_self)}"
        lines.append(
            f"| {row.code} | {row.name} | {show(row.boards)} | {show(row.theme)} | {row.theme_role} | "
            f"{show(row.theme_support_ex_self)} | {supports} | "
            f"{'是' if bool(row.theme_ladder_verified) else '否'} | {row.source_tier} | {row.theme_status} |"
        )
    lines.extend([
        "",
        "只有 `verified_same_day_group` 才可提供题材正向证据；`verified_membership_isolated` 仍按孤立题材处理。",
        "本报告不修改冻结V3概率模型。盘中同步还必须由09:35/10:00成员快照重新计算。",
        "",
    ])
    return "\n".join(lines)


def build(
    signal_date: str,
    registry_path: Path = DEFAULT_REGISTRY,
    candidates_path: Path = DEFAULT_CANDIDATES,
    pool_path: Path | None = None,
    output_root: Path = OUTPUT_ROOT,
    commit_roster: bool = False,
) -> dict[str, Any]:
    signal_date = pd.Timestamp(signal_date).date().isoformat()
    if registry_path.exists() and registry_path.stat().st_size:
        registry = pd.read_csv(registry_path, dtype={"code": "string"}, low_memory=False)
    else:
        registry = pd.DataFrame(columns=[*REGISTRY_COLUMNS, "effective_published_at", "decision_cutoff", "evidence_tier", "usable_at_signal", "usable_reason", "evidence_id"])
    candidates = pd.read_csv(candidates_path, dtype={"code": "string"}, low_memory=False)
    pool = load_limit_pool(signal_date, pool_path)
    roster, roles = build_theme_roster(signal_date, registry, pool, candidates)
    usable_rows = 0
    if not registry.empty and "usable_at_signal" in registry:
        usable_rows = int((
            registry["signal_date"].astype(str).eq(signal_date)
            & registry["usable_at_signal"].astype(str).str.lower().isin(["true", "1"])
        ).sum())
    verified_count = int(roles["theme_status"].eq("verified_same_day_group").sum()) if not roles.empty else 0
    metadata = {
        "version": VERSION,
        "status": "VERIFIED_THEME_AVAILABLE" if verified_count else "THEME_EVIDENCE_UNAVAILABLE",
        "signal_date": signal_date,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "usable_evidence_rows": usable_rows,
        "roster_rows": int(len(roster)),
        "candidate_rows": int(len(roles)),
        "verified_candidate_rows": verified_count,
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": sha256_file(registry_path) if registry_path.exists() else None,
        "candidates_sha256": sha256_file(candidates_path),
        "pool_source": str((pool_path or ROOT / "data" / "v3" / "raw" / signal_date.replace("-", "") / "limit_up.csv").resolve()),
        "assurance": "POINT_IN_TIME_SOURCE_BOUND_LOCAL_HASHES",
        "warning": "verified theme coverage is required; industry proxy is never substituted",
    }
    output_dir = output_root / signal_date
    output_dir.mkdir(parents=True, exist_ok=True)
    roster_path = output_dir / "verified_theme_roster.csv"
    roles_path = output_dir / "candidate_theme_roles.csv"
    report_path = output_dir / "theme_evidence_report.md"
    manifest_path = output_dir / "theme_evidence_manifest.json"
    roster.to_csv(roster_path, index=False, encoding="utf-8-sig")
    roles.to_csv(roles_path, index=False, encoding="utf-8-sig")
    report_path.write_text(render_report(signal_date, roster, roles, metadata), encoding="utf-8")
    metadata["files"] = {
        "roster": {"path": str(roster_path.resolve()), "sha256": sha256_file(roster_path)},
        "candidate_roles": {"path": str(roles_path.resolve()), "sha256": sha256_file(roles_path)},
        "report": {"path": str(report_path.resolve()), "sha256": sha256_file(report_path)},
    }
    if commit_roster:
        keys = ["signal_date", "theme_group_id", "code"]
        if DEFAULT_ROSTER.exists() and DEFAULT_ROSTER.stat().st_size:
            old = pd.read_csv(DEFAULT_ROSTER, dtype={"code": "string"})
            old_keys = set(map(tuple, old[keys].astype(str).to_numpy()))
            duplicate = [tuple(row) for row in roster[keys].astype(str).to_numpy() if tuple(row) in old_keys]
            if duplicate:
                raise ValueError(f"immutable verified roster keys already exist: {duplicate[:5]}")
            roster = pd.concat([old, roster], ignore_index=True, sort=False)
        DEFAULT_ROSTER.parent.mkdir(parents=True, exist_ok=True)
        roster.sort_values(keys).to_csv(DEFAULT_ROSTER, index=False, encoding="utf-8-sig")
        metadata["committed_roster"] = str(DEFAULT_ROSTER.resolve())
    else:
        metadata["committed_roster"] = None
    manifest_path.write_text(json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["manifest"] = str(manifest_path.resolve())
    return metadata


def run_open_sync(source: Path, roster_path: Path, roles_path: Path, output: Path) -> dict[str, Any]:
    roster = pd.read_csv(roster_path, dtype={"code": "string"})
    roles = pd.read_csv(roles_path, dtype={"code": "string"})
    members = pd.read_csv(source, dtype={"member_code": "string"})
    sync = compute_open_sync(members, roster, roles)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size:
        old = pd.read_csv(output, dtype={"code": "string"})
        keys = ["signal_date", "trade_date", "asof", "code"]
        old_keys = set(map(tuple, old[keys].astype(str).to_numpy()))
        duplicate = [tuple(row) for row in sync[keys].astype(str).to_numpy() if tuple(row) in old_keys]
        if duplicate:
            raise ValueError(f"immutable theme sync keys already exist: {duplicate[:5]}")
        sync = pd.concat([old, sync], ignore_index=True, sort=False)
    sync.to_csv(output, index=False, encoding="utf-8-sig")
    return {"rows": int(len(sync)), "output": str(output.resolve()), "sha256": sha256_file(output)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate point-in-time theme evidence and open synchronisation")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("source", type=Path)
    validate.add_argument("--decision-clock", default="15:00")
    import_cmd = sub.add_parser("import")
    import_cmd.add_argument("source", type=Path)
    import_cmd.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    build_cmd = sub.add_parser("build")
    build_cmd.add_argument("--signal-date", required=True)
    build_cmd.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    build_cmd.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    build_cmd.add_argument("--pool", type=Path)
    build_cmd.add_argument("--commit-roster", action="store_true")
    sync_cmd = sub.add_parser("open-sync")
    sync_cmd.add_argument("source", type=Path)
    sync_cmd.add_argument("--roster", type=Path, required=True)
    sync_cmd.add_argument("--roles", type=Path, required=True)
    sync_cmd.add_argument("--output", type=Path, default=DEFAULT_OPEN_SYNC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "validate":
        checked = validate_evidence(pd.read_csv(args.source, dtype={"code": "string"}), args.decision_clock)
        result = {"rows": len(checked), "usable_at_signal": int(checked["usable_at_signal"].sum())}
    elif args.command == "import":
        result = import_evidence(args.source, args.registry)
    elif args.command == "build":
        result = build(args.signal_date, args.registry, args.candidates, args.pool, commit_roster=args.commit_roster)
    else:
        result = run_open_sync(args.source, args.roster, args.roles, args.output)
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
