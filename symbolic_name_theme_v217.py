"""Detect cross-sectional stock-name/symbol clusters without calling them real themes.

This layer is deliberately descriptive.  A repeated character or phrase can become
an A-share short-term market symbol, but it is not evidence that the companies share
an industry, business driver, or catalyst.  The output therefore cannot rank stocks,
authorize entry, or satisfy the project's verified-theme gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd


VERSION = "v217-symbolic-name-cluster-1"

# Corporate-form and generic descriptor fragments create large mechanical clusters
# (for example, almost every abbreviated name ending in “股份”).  They are excluded
# before any threshold is evaluated.  The list is frozen for V217 and is recorded in
# every artifact so later changes cannot silently rewrite historical detections.
STOP_TOKENS = frozenset({
    "股", "份", "股份", "科", "技", "科技", "集", "团", "集团",
    "控", "控股", "实", "业", "实业", "发", "展", "发展",
    "公", "司", "公司",
})

THRESHOLDS = {
    "min_member_count": 4,
    "min_member_share": 0.08,
    "min_relay_member_count": 2,
    "min_distinct_board_levels": 2,
    "min_distinct_industries": 3,
    "max_ngram_length": 2,
}


def _truth(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _timestamp(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("Asia/Shanghai")
    else:
        stamp = stamp.tz_convert("Asia/Shanghai")
    return stamp


def _clean_name(value) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^(?:\*?ST|N|C)", "", text, flags=re.IGNORECASE)
    return "".join(re.findall(r"[\u4e00-\u9fff]", text))


def _name_tokens(name: str) -> set[tuple[str, int]]:
    clean = _clean_name(name)
    tokens: set[tuple[str, int]] = set()
    for length in range(1, THRESHOLDS["max_ngram_length"] + 1):
        for index in range(0, len(clean) - length + 1):
            token = clean[index:index + length]
            if token not in STOP_TOKENS:
                tokens.add((token, length))
    return tokens


def _member_position(name: str, token: str) -> str:
    clean = _clean_name(name)
    if clean == token:
        return "whole_name"
    if clean.startswith(token):
        return "prefix"
    if clean.endswith(token):
        return "suffix"
    return "infix"


def _scope(snapshot: pd.DataFrame, manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "signal_date", "code", "name", "boards", "pool_assignment",
        "is_mainboard", "industry_provider",
    }
    missing = required - set(snapshot)
    if missing:
        raise ValueError(f"symbolic-name snapshot fields missing: {sorted(missing)}")
    signal_date = str(manifest.get("signal_date", ""))
    rows = snapshot[
        snapshot.signal_date.astype(str).eq(signal_date)
        & snapshot.is_mainboard.map(_truth)
    ].copy()
    rows["code"] = rows.code.astype(str).str.zfill(6)
    rows["boards"] = pd.to_numeric(rows.boards, errors="coerce")
    if rows.code.duplicated().any():
        # A code may occur once in each disjoint pool, but never twice in the same
        # normalized snapshot.  The upstream contract promises disjoint pools.
        raise ValueError("symbolic-name snapshot contains duplicate mainboard codes")
    closing = rows[rows.pool_assignment.eq("CLOSING_LIMIT_UP")].copy()
    failed = rows[rows.pool_assignment.eq("FAILED_BOARD_OBSERVATION")].copy()
    if len(closing) != int(manifest.get("mainboard_closing_limit_rows", -1)):
        raise ValueError("symbolic-name closing scope differs from manifest")
    if len(failed) != int(manifest.get("mainboard_failed_board_rows", -1)):
        raise ValueError("symbolic-name failed-board scope differs from manifest")
    if closing.boards.isna().any():
        raise ValueError("symbolic-name closing scope has invalid board levels")
    return closing, failed


def _previous_clusters(previous: dict | None, signal_date: str) -> dict[tuple[str, int], dict]:
    if previous is None:
        return {}
    if previous.get("version") != VERSION:
        raise ValueError("previous symbolic-name evidence version mismatch")
    if previous.get("thresholds") != THRESHOLDS:
        raise ValueError("previous symbolic-name thresholds differ from current frozen rules")
    if str(previous.get("signal_date", "")) >= signal_date:
        raise ValueError("previous symbolic-name evidence is not from an earlier session")
    return {
        (str(row["token"]), int(row["ngram_length"])): row
        for row in previous.get("clusters", [])
    }


def build(snapshot: pd.DataFrame, manifest: dict, asof: str,
          previous: dict | None = None) -> dict:
    closing, failed = _scope(snapshot, manifest)
    signal_date = str(manifest["signal_date"])
    cutoff = _timestamp(asof)
    raw_available_at = manifest.get("available_at")
    if not raw_available_at and "available_at" in snapshot:
        available_values = snapshot.available_at.dropna().astype(str)
        raw_available_at = available_values.max() if len(available_values) else None
    availability_status = "UNAVAILABLE_NOT_TIME_BOUND"
    if raw_available_at:
        available_at = _timestamp(raw_available_at)
        if available_at > cutoff:
            raise ValueError("symbolic-name pool was not available by asof")
        availability_status = "VERIFIED_AVAILABLE_BY_ASOF"

    pool_size = len(closing)
    global_max_board = int(closing.boards.max()) if pool_size else 0
    token_members: dict[tuple[str, int], list[dict]] = {}
    for _, row in closing.iterrows():
        for token, length in _name_tokens(row["name"]):
            token_members.setdefault((token, length), []).append({
                "code": row.code,
                "name": row["name"],
                "boards": int(row.boards),
                "board_group": row.get("board_group"),
                "industry_provider": row.get("industry_provider"),
                "position": _member_position(row["name"], token),
            })

    failed_names = [
        {"code": row.code, "name": row["name"]}
        for _, row in failed.iterrows()
    ]
    prior = _previous_clusters(previous, signal_date)
    clusters = []
    rejected_counts = {
        "member_count_or_share": 0,
        "relay_count": 0,
        "board_levels": 0,
        "industry_diversity": 0,
    }
    minimum_by_share = math.ceil(pool_size * THRESHOLDS["min_member_share"])
    required_members = max(THRESHOLDS["min_member_count"], minimum_by_share)
    for (token, length), members in token_members.items():
        member_count = len(members)
        if member_count < required_members:
            rejected_counts["member_count_or_share"] += 1
            continue
        relay_members = [row for row in members if row["boards"] >= 2]
        if len(relay_members) < THRESHOLDS["min_relay_member_count"]:
            rejected_counts["relay_count"] += 1
            continue
        board_levels = sorted({row["boards"] for row in members})
        if len(board_levels) < THRESHOLDS["min_distinct_board_levels"]:
            rejected_counts["board_levels"] += 1
            continue
        industries = sorted({
            str(row["industry_provider"])
            for row in members
            if pd.notna(row["industry_provider"])
        })
        if len(industries) < THRESHOLDS["min_distinct_industries"]:
            rejected_counts["industry_diversity"] += 1
            continue
        failed_members = [
            row for row in failed_names if token in _clean_name(row["name"])
        ]
        anchor_members = [row for row in members if row["boards"] == global_max_board]
        first_members = [row for row in members if row["boards"] == 1]
        contiguous = board_levels == list(range(min(board_levels), max(board_levels) + 1))
        classification = (
            "SYMBOLIC_LADDER_CANDIDATE"
            if anchor_members and first_members and contiguous
            else "SYMBOLIC_HEIGHT_CLUSTER" if anchor_members
            else "SYMBOLIC_DIFFUSION_CANDIDATE"
        )
        prior_row = prior.get((token, length))
        comparison = {
            "status": "NO_COMPARABLE_PREVIOUS_ARTIFACT",
            "previous_member_count": None,
            "member_count_change": None,
            "previous_member_share": None,
            "member_share_change": None,
            "previous_max_board": None,
            "max_board_change": None,
        }
        if prior_row is not None:
            comparison = {
                "status": "COMPARABLE_FROZEN_RULES",
                "previous_member_count": prior_row["member_count"],
                "member_count_change": member_count - prior_row["member_count"],
                "previous_member_share": prior_row["member_share"],
                "member_share_change": member_count / pool_size - prior_row["member_share"],
                "previous_max_board": prior_row["max_board"],
                "max_board_change": max(board_levels) - prior_row["max_board"],
            }
        clusters.append({
            "cluster_id": f"name_ngram_{length}_{token}",
            "token": token,
            "ngram_length": length,
            "classification": classification,
            "name_match_evidence_status": "VERIFIED_WITHIN_ARCHIVED_PROVIDER_POOL",
            "narrative_status": "INFERRED_SYMBOLIC_CLUSTER_NOT_REAL_THEME",
            "member_count": member_count,
            "member_share": member_count / pool_size,
            "relay_member_count": len(relay_members),
            "first_board_member_count": len(first_members),
            "failed_board_member_count": len(failed_members),
            "distinct_board_levels": board_levels,
            "board_level_count": len(board_levels),
            "max_board": max(board_levels),
            "global_height_anchor_member_count": len(anchor_members),
            "ladder_contiguous": contiguous,
            "distinct_industry_count": len(industries),
            "industries": industries,
            "position_counts": {
                position: sum(row["position"] == position for row in members)
                for position in ("prefix", "infix", "suffix", "whole_name")
            },
            "members": sorted(members, key=lambda row: (-row["boards"], row["code"])),
            "failed_board_members": sorted(failed_members, key=lambda row: row["code"]),
            "previous_session_comparison": comparison,
            "real_theme_eligible": False,
            "candidate_reranking_allowed": False,
            "entry_authorized": False,
        })

    clusters.sort(key=lambda row: (
        -int(row["global_height_anchor_member_count"] > 0),
        -row["max_board"], -row["relay_member_count"],
        -row["member_count"], row["ngram_length"], row["token"],
    ))
    member_map: dict[str, list[dict]] = {str(code): [] for code in closing.code}
    for cluster in clusters:
        for member in cluster["members"]:
            if member["boards"] == global_max_board:
                role = "SYMBOLIC_HEIGHT_ANCHOR"
            elif member["boards"] >= 2:
                role = "SYMBOLIC_RELAY_SUPPORT"
            else:
                role = "SYMBOLIC_FIRST_BOARD_EXPANSION"
            member_map[member["code"]].append({
                "cluster_id": cluster["cluster_id"],
                "token": cluster["token"],
                "classification": cluster["classification"],
                "role": role,
                "evidence_status": "INFERRED_SYMBOLIC_CLUSTER_NOT_REAL_THEME",
            })
    candidate_records = [
        {"code": code, "clusters": values}
        for code, values in sorted(member_map.items())
    ]
    return {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "signal_date": signal_date,
        "asof": cutoff.isoformat(),
        "data_quality": {
            "name_match_status": "VERIFIED_WITHIN_ARCHIVED_PROVIDER_POOL",
            "availability_status": availability_status,
            "provider_pool_status": manifest.get("provider_pool_status"),
            "full_market_completeness": manifest.get("full_market_completeness"),
            "mainboard_closing_limit_count": pool_size,
            "mainboard_failed_board_count": len(failed),
            "point_in_time_security_scope_complete": bool(
                manifest.get("point_in_time_security_scope_complete", False)),
        },
        "scope": {
            "mainboard_only": True,
            "closing_limit_and_failed_board_observed": True,
            "token_lengths": [1, 2],
            "industry_diversity_required": True,
        },
        "thresholds": dict(THRESHOLDS),
        "stop_tokens": sorted(STOP_TOKENS),
        "cluster_count": len(clusters),
        "top_cluster": clusters[0] if clusters else None,
        "clusters": clusters,
        "candidate_records": candidate_records,
        "audit": {
            "unique_tokens_scanned": len(token_members),
            "minimum_members_for_this_pool": required_members,
            "rejected_counts_by_first_failed_gate": rejected_counts,
        },
        "governance": {
            "post_hoc_rule_design_after_2026_09_18_case": True,
            "real_theme_substitution_allowed": False,
            "business_or_catalyst_claim_allowed": False,
            "candidate_reranking_allowed": False,
            "probability_uplift_claim_allowed": False,
            "direct_entry_allowed": False,
            "requires_forward_validation_before_model_use": True,
            "next_session_auction_open_confirmation_still_required": True,
        },
        "interpretation": (
            "名称重复只证明归档涨停池内存在可观察的符号聚集。它可以提示情绪扩散，"
            "但不能替代真实题材、催化、业务归属、可买性或次日确认。"
        ),
    }


def run(snapshot_path: str, manifest_path: str, output_dir: str, asof: str,
        previous_path: str | None = None) -> dict:
    paths = [Path(snapshot_path), Path(manifest_path)]
    previous = None
    if previous_path:
        paths.append(Path(previous_path))
        previous = json.loads(Path(previous_path).read_text(encoding="utf-8"))
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable symbolic-name evidence")
    report = build(
        pd.read_csv(paths[0], dtype={"code": str}),
        json.loads(paths[1].read_text(encoding="utf-8")),
        asof,
        previous,
    )
    report["source_hashes"] = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*paths, Path(__file__)]
    }
    target.mkdir(parents=True, exist_ok=False)
    with (target / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--previous", help="optional earlier V217 report.json")
    parser.add_argument("--asof", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.snapshot, args.manifest, args.output, args.asof,
                         args.previous), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
