"""Build a point-in-time leader-competition snapshot from verified close evidence.

The output describes *observed competition at the close*.  It deliberately does
not predict the next-session winner and never authorizes an entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


VERIFIED_THEME_STATUSES = {
    "verified_same_day_group",
    "verified_explicit_source_roster",
    "verified_specific_point_in_time_schema",
}

V150_STATUS = "verified_specific_point_in_time_schema"
V150_FIELDS = {
    "theme_granularity", "driver_type", "driver_summary",
    "driver_first_available_at", "driver_claim_status", "expectation_status",
    "expectation_basis", "positive_catalyst_factor_eligible",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stamp(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("Asia/Shanghai")
    else:
        stamp = stamp.tz_convert("Asia/Shanghai")
    return stamp


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _number(row, field, default=None):
    value = pd.to_numeric(pd.Series([row.get(field)]), errors="coerce").iloc[0]
    return default if pd.isna(value) else float(value)


def _identity(row: pd.Series, global_max: int, global_max_count: int) -> str:
    """Assign only identities supported by close-time facts, in fixed order."""
    tags = {part for part in str(row.get("identity_tags", "")).split("|") if part}
    boards = int(row["boards"])
    role = str(row.get("theme_role", ""))
    rivals = int(_number(row, "same_height_rivals", 0))
    capacity_rank = int(_number(row, "capacity_rank_in_theme", 999999))
    if boards == global_max and global_max_count == 1:
        return "height_anchor"
    if role == "THEME_HEIGHT_LEADER" and rivals == 0:
        return "unique_survivor"
    if role == "SAME_HEIGHT_CORE" and capacity_rank == 1:
        return "capacity_core"
    if "FIRST_SEAL_PIONEER" in tags:
        return "pioneer"
    if capacity_rank == 1:
        return "capacity_core"
    return "supplement"


def build(pool: pd.DataFrame, pool_manifest: dict, themes: pd.DataFrame,
          theme_manifest: dict, asof: str) -> tuple[pd.DataFrame, dict]:
    signal_date = str(pool_manifest["signal_date"])
    if str(theme_manifest.get("signal_date")) != signal_date:
        raise ValueError("pool and theme manifests have different signal dates")

    required_pool = {
        "signal_date", "code", "name", "boards", "pool_assignment",
        "is_mainboard", "amount", "first_seal_time", "observed_at", "available_at",
    }
    required_theme = {
        "signal_date", "code", "theme_status", "theme_role", "theme_group_id",
        "theme", "primary_evidence_id", "source_tier", "theme_support_ex_self",
        "same_height_rivals", "theme_ladder_verified", "evidence_published_at",
        "identity_tags", "capacity_rank_in_theme",
    }
    if required_pool - set(pool):
        raise ValueError(f"pool fields missing: {sorted(required_pool - set(pool))}")
    if required_theme - set(themes):
        raise ValueError(f"theme fields missing: {sorted(required_theme - set(themes))}")

    frame = pool.copy()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame[
        frame.signal_date.eq(signal_date)
        & frame.pool_assignment.eq("CLOSING_LIMIT_UP")
        & frame.is_mainboard.map(_truthy)
    ].copy()
    if frame.code.duplicated().any():
        raise ValueError("closing pool contains duplicate codes")
    expected = int(pool_manifest["mainboard_closing_limit_rows"])
    if len(frame) != expected:
        raise ValueError("closing pool count differs from manifest")
    if not _truthy(pool_manifest.get("complete_within_archived_roster")):
        raise ValueError("competition scope is not complete within its declared roster")

    frame["boards"] = pd.to_numeric(frame["boards"], errors="coerce")
    if frame.boards.isna().any() or frame.boards.lt(1).any() or not frame.boards.mod(1).eq(0).all():
        raise ValueError("closing pool has invalid board heights")
    frame["boards"] = frame.boards.astype(int)

    theme = themes.copy()
    theme["code"] = theme["code"].astype(str).str.zfill(6)
    theme["signal_date"] = pd.to_datetime(theme["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    theme = theme[theme.signal_date.eq(signal_date)].copy()
    theme = theme[theme.theme_status.isin(VERIFIED_THEME_STATUSES)].copy()
    if theme.theme_status.eq(V150_STATUS).any():
        missing_v150 = V150_FIELDS - set(theme)
        if missing_v150:
            raise ValueError(f"V150 theme fields missing: {sorted(missing_v150)}")
    if theme.code.duplicated().any():
        raise ValueError("verified theme roles contain duplicate codes")
    unknown_codes = set(theme.code) - set(frame.code)
    if unknown_codes:
        raise ValueError(f"theme roles contain codes outside closing scope: {sorted(unknown_codes)}")

    cutoff = _stamp(asof)
    pool_available = max(_stamp(value) for value in frame.available_at)
    theme_available = max((_stamp(value) for value in theme.evidence_published_at),
                          default=_stamp(f"{signal_date} 15:00:00"))
    available = max(pool_available, theme_available)
    if available > cutoff:
        raise ValueError("competition inputs were not available by asof")
    observed = _stamp(f"{signal_date} 15:00:00")
    if observed > available:
        raise ValueError("competition evidence availability precedes observation")

    merged = theme.merge(
        frame[["code", "name", "boards", "amount", "first_seal_time"]],
        on="code", how="left", validate="one_to_one",
    )
    global_max = int(frame.boards.max())
    global_max_count = int(frame.boards.eq(global_max).sum())
    all_by_board = {int(level): sorted(group.code.tolist()) for level, group in frame.groupby("boards")}
    rows = []
    for _, row in merged.sort_values(["boards", "code"], ascending=[False, True]).iterrows():
        code, boards = row.code, int(row.boards)
        same_board = [peer for peer in all_by_board[boards] if peer != code]
        higher = sorted(frame.loc[frame.boards.gt(boards), "code"].tolist())
        same_theme = merged[merged.theme_group_id.eq(row.theme_group_id)]
        theme_higher = sorted(same_theme.loc[same_theme.boards.gt(boards), "code"].tolist())
        theme_same = sorted(same_theme.loc[
            same_theme.boards.eq(boards) & same_theme.code.ne(code), "code"
        ].tolist())
        support_count = int(_number(row, "theme_support_ex_self", 0))
        tags = {part for part in str(row.identity_tags).split("|") if part}
        events = []
        if boards == global_max and global_max_count == 1:
            events.append({"type": "GLOBAL_UNIQUE_HEIGHT", "boards": boards})
        if same_board:
            events.append({"type": "GLOBAL_SAME_BOARD_RIVALS", "codes": same_board})
        if higher:
            events.append({"type": "GLOBAL_HIGHER_BOARD_PRESSURE", "codes": higher})
        if theme_higher:
            events.append({"type": "THEME_HIGHER_BOARD_PRESSURE", "codes": theme_higher})
        elif not theme_same:
            events.append({"type": "THEME_UNIQUE_HEIGHT_AT_CLOSE", "boards": boards})
        if theme_same:
            events.append({"type": "THEME_SAME_HEIGHT_RIVALS", "codes": theme_same})
        if support_count:
            events.append({"type": "VERIFIED_THEME_SUPPORT", "count_ex_self": support_count})
        if "CAPACITY_CORE_BY_CLOSE_AMOUNT" in tags:
            events.append({"type": "CAPACITY_LEADER_BY_CLOSE_AMOUNT", "rank": 1})
        if "FIRST_SEAL_PIONEER" in tags:
            events.append({"type": "FIRST_SEAL_PIONEER_WITHIN_VERIFIED_THEME"})
        if _truthy(row.theme_ladder_verified):
            events.append({"type": "THEME_LADDER_PRESENT_AT_CLOSE"})
        rows.append({
            "signal_date": signal_date,
            "code": code,
            "name": row["name"],
            "boards": boards,
            "amount": _number(row, "amount"),
            "first_seal_time": _number(row, "first_seal_time"),
            "theme_role": row.theme_role,
            "competition_identity": _identity(row, global_max, global_max_count),
            "scope_id": f"mainboard_closing_limit_reconciled_roster:{signal_date}",
            "source": "reconciled_closing_pool+verified_point_in_time_theme_roles",
            "observed_at": observed.isoformat(),
            "available_at": available.isoformat(),
            "data_status": "verified",
            "evidence_status": "VERIFIED_CLOSE_TIME_FACTS_ONLY",
            "universe_complete": True,
            "same_board_peer_count": len(same_board),
            "higher_board_peer_count": len(higher),
            "same_theme_peer_count": int(len(same_theme) - 1),
            "same_theme_board_peer_count": len(theme_same),
            "higher_theme_peer_count": len(theme_higher),
            "theme_group_id": row.theme_group_id,
            "theme": row.theme,
            "theme_granularity": row.get("theme_granularity"),
            "driver_type": row.get("driver_type"),
            "driver_summary": row.get("driver_summary"),
            "driver_first_available_at": row.get("driver_first_available_at"),
            "driver_claim_status": row.get("driver_claim_status"),
            "expectation_status": row.get("expectation_status"),
            "expectation_basis": row.get("expectation_basis"),
            "positive_catalyst_factor_eligible": (
                _truthy(row.get("positive_catalyst_factor_eligible"))
                if row.theme_status == V150_STATUS else False
            ),
            "theme_factor_schema_status": (
                "V150_COMPLETE" if row.theme_status == V150_STATUS else "LEGACY_INCOMPLETE"
            ),
            "events": json.dumps(events, ensure_ascii=False, separators=(",", ":")),
            "entry_authorized": False,
        })
    result = pd.DataFrame(rows)
    metadata = {
        "status": "VERIFIED_PARTIAL_CANDIDATE_COMPETITION_AT_CLOSE",
        "signal_date": signal_date,
        "asof": cutoff.isoformat(),
        "observed_at": observed.isoformat(),
        "available_at": available.isoformat(),
        "scope_id": f"mainboard_closing_limit_reconciled_roster:{signal_date}",
        "closing_candidate_count": int(len(frame)),
        "verified_theme_candidate_count": int(len(result)),
        "v150_factor_schema_candidate_count": int(
            result.theme_factor_schema_status.eq("V150_COMPLETE").sum()
        ) if len(result) else 0,
        "coverage": float(len(result) / len(frame)) if len(frame) else None,
        "full_market_completeness": pool_manifest.get("full_market_completeness"),
        "complete_within_declared_roster": True,
        "unknown_candidate_count": int(len(frame) - len(result)),
        "unknown_candidate_codes": sorted(set(frame.code) - set(result.code)),
        "identity_rule": "deterministic_close_time_precedence_v114",
        "forward_winner_claimed": False,
        "entry_authorized": False,
        "limitations": [
            "competition identities describe close-time structure, not next-session winners",
            "candidates without verified specific-theme evidence remain unavailable",
            "full-market completeness is inherited from the pool manifest and is not upgraded",
            "no auction, opening, Level-2, queue, fill, or exit evidence is used",
        ],
    }
    return result, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--pool-manifest", required=True)
    parser.add_argument("--theme-roles", required=True)
    parser.add_argument("--theme-manifest", required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    pool_path, pool_manifest_path = Path(args.pool), Path(args.pool_manifest)
    theme_path, theme_manifest_path = Path(args.theme_roles), Path(args.theme_manifest)
    result, metadata = build(
        pd.read_csv(pool_path, dtype={"code": str}),
        json.loads(pool_manifest_path.read_text(encoding="utf-8")),
        pd.read_csv(theme_path, dtype={"code": str}),
        json.loads(theme_manifest_path.read_text(encoding="utf-8")),
        args.asof,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "candidate_competition_snapshot.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    metadata["source_hashes"] = {
        str(path.resolve()): _sha256(path)
        for path in (pool_path, pool_manifest_path, theme_path, theme_manifest_path)
    }
    metadata["output_sha256"] = _sha256(output_path)
    (output_dir / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
