"""Build an evidence-tiered theme matrix without altering the leader model.

Real themes, secondary narrative rosters, provider industries and symbolic-name
clusters remain separate evidence lanes.  Missing membership or catalyst data
is never converted into a weak-theme score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v219-theme-screening-matrix-1"
LANE_ORDER = {
    "VERIFIED_REAL_THEME": 0,
    "VERIFIED_MEMBERSHIP_ONLY": 1,
    "OBSERVED_THEME_ROSTER": 2,
    "INDUSTRY_PROXY": 3,
    "SYMBOLIC_NAME_CLUSTER": 4,
    "RUMOR_UNVERIFIED": 5,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_pool(pool: pd.DataFrame) -> pd.DataFrame:
    required = {"signal_date", "code", "name", "boards", "pool_assignment", "is_mainboard"}
    if required - set(pool):
        raise ValueError(f"pool fields missing: {sorted(required - set(pool))}")
    out = pool.copy()
    out["code"] = out.code.astype(str).str.zfill(6)
    out["boards"] = pd.to_numeric(out.boards, errors="coerce")
    out["is_mainboard"] = out.is_mainboard.astype(str).str.lower().isin({"true", "1"})
    out = out[out.is_mainboard].copy()
    out["is_closing"] = out.pool_assignment.eq("CLOSING_LIMIT_UP")
    out["is_failed"] = out.pool_assignment.astype(str).str.startswith("FAILED_BOARD")
    if "industry_provider" not in out:
        out["industry_provider"] = pd.NA
    return out


def industry_memberships(pool: pd.DataFrame) -> pd.DataFrame:
    valid = pool[pool.industry_provider.notna() & pool.industry_provider.astype(str).str.strip().ne("")]
    return pd.DataFrame({
        "signal_date": valid.signal_date,
        "code": valid.code,
        "theme_id": "industry:" + valid.industry_provider.astype(str),
        "theme_name": valid.industry_provider.astype(str),
        "theme_type": "INDUSTRY_PROXY",
        "membership_status": "PROVIDER_INDUSTRY_PROXY",
        "catalyst_status": "UNAVAILABLE",
        "expectation_status": "UNAVAILABLE",
        "available_at": valid.get("available_at", pd.Series(pd.NA, index=valid.index)),
    })


def symbolic_memberships(symbolic: dict, signal_date: str) -> pd.DataFrame:
    records = []
    for cluster in symbolic.get("clusters", []) or ([symbolic.get("top_cluster")] if symbolic.get("top_cluster") else []):
        if not cluster or cluster.get("classification") != "SYMBOLIC_LADDER_CANDIDATE":
            continue
        for member in cluster.get("members", []):
            records.append({
                "signal_date": signal_date,
                "code": str(member["code"]).zfill(6),
                "theme_id": cluster["cluster_id"],
                "theme_name": f"{cluster['token']}字辈",
                "theme_type": "SYMBOLIC_NAME_CLUSTER",
                "membership_status": "INFERRED_FROM_ARCHIVED_NAME",
                "catalyst_status": "NOT_APPLICABLE_SYMBOLIC_NARRATIVE",
                "expectation_status": "UNAVAILABLE",
                "available_at": symbolic.get("asof"),
            })
        for member in cluster.get("failed_board_members", []):
            records.append({
                "signal_date": signal_date,
                "code": str(member["code"]).zfill(6),
                "theme_id": cluster["cluster_id"],
                "theme_name": f"{cluster['token']}字辈",
                "theme_type": "SYMBOLIC_NAME_CLUSTER",
                "membership_status": "INFERRED_FROM_ARCHIVED_NAME",
                "catalyst_status": "NOT_APPLICABLE_SYMBOLIC_NARRATIVE",
                "expectation_status": "UNAVAILABLE",
                "available_at": symbolic.get("asof"),
            })
    return pd.DataFrame(records)


def adapt_external_memberships(memberships: pd.DataFrame) -> pd.DataFrame:
    """Accept either the V219 contract or existing point-in-time theme roles."""
    if "theme_type" in memberships:
        return memberships.copy()
    role_required = {
        "signal_date", "code", "theme_group_id", "theme", "theme_status",
        "evidence_status", "evidence_available_at",
    }
    if role_required - set(memberships):
        return memberships.copy()
    out = memberships.copy()
    status = out.theme_status.astype(str)
    out["theme_id"] = out.theme_group_id
    out["theme_name"] = out.theme
    out["theme_type"] = np.select(
        [status.eq("verified_explicit_source_roster"),
         status.eq("observed_explicit_secondary_roster")],
        ["VERIFIED_MEMBERSHIP_ONLY", "OBSERVED_THEME_ROSTER"],
        default="RUMOR_UNVERIFIED",
    )
    out["membership_status"] = out.evidence_status
    # Existing V168 roles prove an explicit roster, not an active catalyst or
    # expectation state.  Those remain missing unless a richer V219 input says so.
    out["catalyst_status"] = "UNAVAILABLE"
    out["expectation_status"] = "UNAVAILABLE"
    out["available_at"] = out.evidence_available_at
    return out


def validate_external_memberships(memberships: pd.DataFrame) -> pd.DataFrame:
    memberships = adapt_external_memberships(memberships)
    required = {
        "signal_date", "code", "theme_id", "theme_name", "theme_type",
        "membership_status", "catalyst_status", "expectation_status", "available_at",
    }
    if required - set(memberships):
        raise ValueError(f"theme membership fields missing: {sorted(required - set(memberships))}")
    out = memberships.copy()
    out["code"] = out.code.astype(str).str.zfill(6)
    allowed = set(LANE_ORDER)
    if not set(out.theme_type.astype(str)).issubset(allowed):
        raise ValueError("unsupported theme_type")
    if out.duplicated(["signal_date", "code", "theme_id"]).any():
        raise ValueError("theme membership keys must be unique")
    return out


def _contiguous(levels: list[int]) -> bool:
    values = sorted({value for value in levels if value >= 1})
    return bool(values and values == list(range(1, max(values) + 1)))


def build_matrix(pool: pd.DataFrame, memberships: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = normalize_pool(pool)
    memberships = validate_external_memberships(memberships)
    joined = memberships.merge(pool, on=["signal_date", "code"], how="inner", validate="many_to_one")
    records = []
    for (date, theme_id), part in joined.groupby(["signal_date", "theme_id"], sort=True):
        closing = part[part.is_closing]
        failed = part[part.is_failed]
        first = closing[closing.boards.eq(1)]
        relay = closing[closing.boards.gt(1)]
        levels = sorted(closing.boards.dropna().astype(int).unique().tolist())
        theme_type = str(part.theme_type.iloc[0])
        catalyst = set(part.catalyst_status.dropna().astype(str))
        expectations = set(part.expectation_status.dropna().astype(str))
        active = bool(catalyst & {"ACTIVE_DRIVER_VERIFIED", "ACTIVE_DRIVER_OBSERVED"})
        breadth = len(first)
        stage = (
            "FULL_LADDER" if len(relay) and breadth >= 2 and _contiguous(levels)
            else "EMERGING_FIRST_BOARD_BREADTH" if breadth >= 3 and not len(relay)
            else "RELAY_WITH_SUPPORT" if len(relay) and breadth >= 1
            else "ISOLATED_OR_THIN"
        )
        # This score ranks only inside one evidence lane.  It is intentionally
        # structural and does not authorize cross-lane comparisons or entries.
        structural = (
            min(breadth, 7) + 2 * min(len(relay), 4) + min(max(levels or [0]), 5)
            + (2 if _contiguous(levels) else 0) - min(len(failed), 3)
        )
        records.append({
            "signal_date": date,
            "theme_id": theme_id,
            "theme_name": str(part.theme_name.iloc[0]),
            "evidence_lane": theme_type,
            "membership_status": "|".join(sorted(set(part.membership_status.astype(str)))),
            "catalyst_status": "|".join(sorted(catalyst)) if catalyst else "UNAVAILABLE",
            "expectation_status": "|".join(sorted(expectations)) if expectations else "UNAVAILABLE",
            "active_driver_present": active,
            "closing_members": int(len(closing)),
            "first_board_members": int(breadth),
            "relay_members": int(len(relay)),
            "failed_members": int(len(failed)),
            "max_board": int(max(levels or [0])),
            "board_levels": ",".join(map(str, levels)),
            "ladder_contiguous": _contiguous(levels),
            "theme_stage": stage,
            "structural_score_within_lane": float(structural),
            "entry_authorized": False,
        })
    matrix = pd.DataFrame(records)
    if matrix.empty:
        return matrix, pd.DataFrame()
    matrix["lane_order"] = matrix.evidence_lane.map(LANE_ORDER)
    matrix["rank_within_lane"] = matrix.groupby(
        ["signal_date", "evidence_lane"]
    ).structural_score_within_lane.rank(method="min", ascending=False).astype(int)
    matrix = matrix.sort_values(
        ["signal_date", "lane_order", "rank_within_lane", "theme_id"])

    first_map = joined[joined.is_closing & joined.boards.eq(1)].merge(
        matrix[["signal_date", "theme_id", "evidence_lane", "theme_stage",
                "rank_within_lane", "structural_score_within_lane"]],
        on=["signal_date", "theme_id"], how="left", validate="many_to_one")
    seal = pd.to_numeric(first_map.get("first_seal_time"), errors="coerce")
    first_map["theme_pioneer_rank"] = seal.groupby(
        [first_map.signal_date, first_map.theme_id]).rank(method="min", ascending=True)
    first_map["role_in_theme"] = np.where(
        first_map.theme_pioneer_rank.eq(1), "FIRST_BOARD_PIONEER", "FIRST_BOARD_FOLLOWER")
    first_map["entry_authorized"] = False
    keep = [
        "signal_date", "code", "name", "theme_id", "theme_name", "evidence_lane",
        "theme_stage", "rank_within_lane", "structural_score_within_lane",
        "theme_pioneer_rank", "role_in_theme", "entry_authorized",
    ]
    return matrix, first_map[keep]


def run(pool_path: Path, output: Path, memberships_path: Path | None = None,
        symbolic_path: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable v219 output")
    pool = normalize_pool(pd.read_csv(pool_path, dtype={"code": str}))
    groups = [industry_memberships(pool)]
    sources = [pool_path]
    if memberships_path:
        groups.append(adapt_external_memberships(
            pd.read_csv(memberships_path, dtype={"code": str})))
        sources.append(memberships_path)
    if symbolic_path:
        symbolic = json.loads(symbolic_path.read_text(encoding="utf-8"))
        groups.append(symbolic_memberships(symbolic, str(pool.signal_date.iloc[0])))
        sources.append(symbolic_path)
    memberships = pd.concat([group for group in groups if not group.empty], ignore_index=True)
    matrix, first_map = build_matrix(pool, memberships)
    output.mkdir(parents=True)
    matrix.to_csv(output / "theme_matrix.csv", index=False, encoding="utf-8-sig")
    first_map.to_csv(output / "first_board_theme_map.csv", index=False, encoding="utf-8-sig")
    lane_summary = {
        lane: {
            "theme_count": int(len(part)),
            "top_themes": part.head(5)[[
                "theme_id", "theme_name", "theme_stage", "closing_members",
                "first_board_members", "relay_members", "failed_members", "max_board",
            ]].to_dict("records"),
        }
        for lane, part in matrix.groupby("evidence_lane", sort=False)
    }
    verified = matrix[matrix.evidence_lane.eq("VERIFIED_REAL_THEME")]
    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "scope": "THEME_SCREENING_ONLY_LEADER_AND_RELAY_MODELS_UNCHANGED",
        "signal_date": str(pool.signal_date.iloc[0]),
        "theme_rows": int(len(matrix)),
        "first_board_theme_mappings": int(len(first_map)),
        "lane_summary": lane_summary,
        "real_theme_evidence_available": bool(len(verified)),
        "governance": {
            "cross_lane_scores_comparable": False,
            "industry_is_real_theme": False,
            "symbolic_cluster_is_real_theme": False,
            "missing_theme_is_weak_theme": False,
            "active_catalyst_required_for_verified_real_theme": True,
            "entry_authorized": False,
        },
        "limitations": [
            "provider industry can describe co-movement but cannot prove a shared catalyst",
            "symbolic name clusters describe attention only and may reverse abruptly",
            "verified real-theme ranking remains unavailable without point-in-time membership and catalyst inputs",
        ],
        "source_hashes": {str(path.resolve()): _sha256(path) for path in sources},
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--memberships", type=Path)
    parser.add_argument("--symbolic", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.pool, args.output, args.memberships, args.symbolic),
                     ensure_ascii=False, indent=2))
