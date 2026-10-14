"""Build confidence-aware cross-theme races for first boards and relays.

First boards compete for new-cycle/low-position attention. Relays compete for
height and ladder capital.  Their fields are deliberately separate and are not
collapsed into a common winner score. Multiple evidence files may describe the
same candidate; stronger evidence deterministically replaces weaker evidence.

Unknown themes stay unknown. Secondary narratives and verified business
membership without an active driver are descriptive only. Model eligibility
requires a complete market pool and fully verified active-theme evidence for
every closing-limit candidate in the session.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v170-confidence-aware-cross-theme-race-1"
FULLY_VERIFIED_STATUSES = {
    "verified_same_day_group", "verified_explicit_source_roster",
    "verified_specific_point_in_time_schema", "verified_membership_and_driver",
}
MEMBERSHIP_ONLY_STATUSES = {"verified_membership_driver_unverified"}
SECONDARY_STATUSES = {
    "observed_explicit_secondary_roster", "observed_specific_point_in_time_schema",
    "observed_membership_unverified",
}
RUMOR_STATUSES = {
    "rumor_explicit_roster", "rumor_specific_point_in_time_schema", "rumor_membership",
}
CONFIDENCE_PRIORITY = {
    "UNKNOWN": 0, "RUMOR": 1, "SECONDARY_NARRATIVE": 2,
    "VERIFIED_MEMBERSHIP_DRIVER_UNVERIFIED": 3,
    "VERIFIED_ACTIVE_THEME": 4,
}


def _truthy(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _confidence(row: pd.Series) -> str:
    status = str(row.get("theme_status", ""))
    if status in FULLY_VERIFIED_STATUSES:
        positive = row.get("positive_catalyst_factor_eligible", pd.NA)
        if pd.isna(positive) or not _truthy(positive):
            # Legacy "verified" rows do not prove the V169 active-driver and
            # expectation gates. They stay descriptive rather than being
            # silently promoted to model-eligible evidence.
            return "VERIFIED_MEMBERSHIP_DRIVER_UNVERIFIED"
        return "VERIFIED_ACTIVE_THEME"
    if status in MEMBERSHIP_ONLY_STATUSES:
        return "VERIFIED_MEMBERSHIP_DRIVER_UNVERIFIED"
    if status in SECONDARY_STATUSES:
        return "SECONDARY_NARRATIVE"
    if status in RUMOR_STATUSES:
        return "RUMOR"
    return "UNKNOWN"


def _local_naive(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return pd.NaT
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return stamp


def _prepare_roles(role_frames: list[pd.DataFrame], signal_date: str,
                   asof: str) -> pd.DataFrame:
    required = {
        "signal_date", "code", "theme_status", "theme_group_id", "theme",
    }
    prepared = []
    cutoff = _local_naive(asof)
    for source_index, raw in enumerate(role_frames):
        if required - set(raw):
            raise ValueError(f"theme role fields missing: {sorted(required - set(raw))}")
        frame = raw.copy()
        frame["code"] = frame.code.astype(str).str.zfill(6)
        frame["signal_date"] = pd.to_datetime(
            frame.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
        frame = frame[frame.signal_date.eq(signal_date)].copy()
        if frame.code.duplicated().any():
            raise ValueError("a theme role source contains duplicate candidate codes")
        if "evidence_available_at" in frame:
            available = frame.evidence_available_at.map(_local_naive)
            if available.isna().any() or available.gt(cutoff).any():
                raise ValueError("theme evidence is unavailable or after asof")
        elif "evidence_published_at" in frame:
            published = frame.evidence_published_at.fillna("").astype(str).str.strip()
            if published.eq("").any():
                raise ValueError("theme evidence needs an availability timestamp")
            parsed = published.map(_local_naive)
            if parsed.isna().any() or parsed.gt(cutoff).any():
                raise ValueError("theme evidence publication is after asof")
        else:
            raise ValueError("theme evidence needs publication or availability time")
        frame["theme_evidence_confidence"] = frame.apply(_confidence, axis=1)
        frame["_confidence_priority"] = frame.theme_evidence_confidence.map(
            CONFIDENCE_PRIORITY)
        frame["_source_index"] = source_index
        prepared.append(frame)
    if not prepared:
        return pd.DataFrame(columns=list(required) + ["theme_evidence_confidence"])
    merged = pd.concat(prepared, ignore_index=True, sort=False)
    # Higher confidence wins; for equal confidence the later CLI source wins,
    # making deliberate corrections reproducible without averaging conflicts.
    merged = merged.sort_values(
        ["code", "_confidence_priority", "_source_index"],
        ascending=[True, False, False], kind="mergesort")
    return merged.drop_duplicates("code", keep="first")


def build(pool: pd.DataFrame, pool_manifest: dict, role_frames: list[pd.DataFrame],
          asof: str) -> tuple[pd.DataFrame, dict]:
    required_pool = {
        "signal_date", "code", "name", "boards", "amount", "pool_assignment",
        "is_mainboard",
    }
    if required_pool - set(pool):
        raise ValueError(f"pool fields missing: {sorted(required_pool - set(pool))}")
    signal_date = str(pool_manifest.get("signal_date", ""))
    frame = pool.copy()
    frame["code"] = frame.code.astype(str).str.zfill(6)
    frame["signal_date"] = pd.to_datetime(
        frame.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
    mainboard = frame.is_mainboard.map(_truthy)
    frame = frame[
        frame.signal_date.eq(signal_date) & mainboard
        & frame.pool_assignment.eq("CLOSING_LIMIT_UP")
    ].copy()
    if (frame.empty or frame.code.duplicated().any()
            or len(frame) != int(pool_manifest.get("mainboard_closing_limit_rows", -1))):
        raise ValueError("pool is not the manifest-bound unique mainboard closing pool")
    frame["boards"] = pd.to_numeric(frame.boards, errors="coerce")
    frame["amount"] = pd.to_numeric(frame.amount, errors="coerce")
    if frame.boards.isna().any() or not frame.boards.mod(1).eq(0).all():
        raise ValueError("boards must be integral")
    frame["boards"] = frame.boards.astype(int)
    roles = _prepare_roles(role_frames, signal_date, asof)
    role_columns = [
        "code", "theme_status", "theme_group_id", "theme",
        "theme_evidence_confidence", "business_branch", "driver_verified",
        "expectation_status", "positive_catalyst_factor_eligible",
        "primary_evidence_id", "evidence_available_at",
    ]
    for column in role_columns:
        if column not in roles:
            roles[column] = pd.NA
    result = frame.merge(roles[role_columns], on="code", how="left", validate="one_to_one")
    result["theme_evidence_confidence"] = result.theme_evidence_confidence.fillna("UNKNOWN")
    known = (
        result.theme_group_id.notna()
        & result.theme_group_id.astype(str).str.strip().ne("")
        & result.theme_evidence_confidence.ne("UNKNOWN")
    )
    known_frame = result[known].copy()
    result["race_lane"] = np.where(
        result.boards.eq(1), "FIRST_BOARD_NEW_CYCLE_RACE", "RELAY_HEIGHT_RACE")
    result["race_peer_evidence_scope"] = "SAME_CONFIDENCE_ONLY"
    result["cross_confidence_known_candidate_count"] = pd.NA
    result["cross_confidence_known_theme_count"] = pd.NA

    columns = [
        "first_cross_theme_competing_theme_count",
        "first_cross_theme_competing_candidate_count",
        "first_own_theme_first_count", "first_own_theme_relay_count",
        "first_own_theme_max_board", "first_existing_relay_backed",
        "first_isolated_theme_first",
        "relay_cross_theme_same_height_candidate_count",
        "relay_cross_theme_same_height_theme_count",
        "relay_cross_theme_higher_candidate_count",
        "relay_cross_theme_higher_theme_count",
        "relay_own_theme_first_support_count",
        "relay_own_theme_middle_support_count",
        "relay_own_theme_lower_support_count",
        "relay_same_theme_same_height_rival_count", "relay_global_height_gap",
    ]
    for column in columns:
        result[column] = pd.NA
    events = {}
    global_max = int(result.boards.max())
    for index, row in result[known].iterrows():
        theme_id = str(row.theme_group_id)
        confidence = str(row.theme_evidence_confidence)
        compatible = known_frame[
            known_frame.theme_evidence_confidence.astype(str).eq(confidence)]
        own = compatible[compatible.theme_group_id.astype(str).eq(theme_id)]
        other = compatible[compatible.theme_group_id.astype(str).ne(theme_id)]
        incompatible = known_frame[
            known_frame.theme_evidence_confidence.astype(str).ne(confidence)]
        result.at[index, "cross_confidence_known_candidate_count"] = int(len(incompatible))
        result.at[index, "cross_confidence_known_theme_count"] = int(
            incompatible.theme_group_id.nunique())
        board = int(row.boards)
        row_events = []
        if board == 1:
            competing_first = other[other.boards.eq(1)]
            result.at[index, "first_cross_theme_competing_theme_count"] = int(
                competing_first.theme_group_id.nunique())
            result.at[index, "first_cross_theme_competing_candidate_count"] = int(
                len(competing_first))
            result.at[index, "first_own_theme_first_count"] = int(own.boards.eq(1).sum())
            result.at[index, "first_own_theme_relay_count"] = int(own.boards.ge(2).sum())
            result.at[index, "first_own_theme_max_board"] = int(own.boards.max())
            result.at[index, "first_existing_relay_backed"] = bool(own.boards.ge(2).any())
            result.at[index, "first_isolated_theme_first"] = bool(len(own) == 1)
            if len(competing_first):
                row_events.append({
                    "type": "FIRST_BOARD_CROSS_THEME_NEW_CYCLE_RACE",
                    "other_theme_count": int(competing_first.theme_group_id.nunique()),
                    "other_candidate_count": int(len(competing_first)),
                })
        else:
            same_height = other[other.boards.eq(board)]
            higher = other[other.boards.gt(board)]
            lower_own = own[own.boards.lt(board)]
            result.at[index, "relay_cross_theme_same_height_candidate_count"] = int(
                len(same_height))
            result.at[index, "relay_cross_theme_same_height_theme_count"] = int(
                same_height.theme_group_id.nunique())
            result.at[index, "relay_cross_theme_higher_candidate_count"] = int(len(higher))
            result.at[index, "relay_cross_theme_higher_theme_count"] = int(
                higher.theme_group_id.nunique())
            result.at[index, "relay_own_theme_first_support_count"] = int(
                own.boards.eq(1).sum())
            result.at[index, "relay_own_theme_middle_support_count"] = int(
                ((own.boards.ge(2)) & (own.boards.lt(board))).sum())
            result.at[index, "relay_own_theme_lower_support_count"] = int(len(lower_own))
            result.at[index, "relay_same_theme_same_height_rival_count"] = int(
                (own.code.ne(row.code) & own.boards.eq(board)).sum())
            result.at[index, "relay_global_height_gap"] = global_max - board
            if len(same_height):
                row_events.append({
                    "type": "RELAY_CROSS_THEME_SAME_HEIGHT_RACE",
                    "other_theme_count": int(same_height.theme_group_id.nunique()),
                    "other_candidate_count": int(len(same_height)),
                })
            if len(higher):
                row_events.append({
                    "type": "RELAY_CROSS_THEME_HIGHER_PRESSURE",
                    "higher_theme_count": int(higher.theme_group_id.nunique()),
                    "higher_candidate_count": int(len(higher)),
                })
        events[index] = json.dumps(row_events, ensure_ascii=False, separators=(",", ":"))
    result["cross_theme_race_events"] = pd.Series(events, dtype="object")

    full_market = pool_manifest.get("full_market_completeness") == "verified"
    complete_theme_scope = bool(known.all())
    all_active_verified = bool(
        result.theme_evidence_confidence.eq("VERIFIED_ACTIVE_THEME").all())
    training_scope = full_market and complete_theme_scope and all_active_verified
    result["cross_theme_race_status"] = "UNKNOWN_THEME_NO_RACE_INFERENCE"
    result.loc[known, "cross_theme_race_status"] = "DESCRIPTIVE_KNOWN_THEME_SUBSET"
    result.loc[
        known & result.theme_evidence_confidence.eq(
            "VERIFIED_MEMBERSHIP_DRIVER_UNVERIFIED"),
        "cross_theme_race_status",
    ] = "DESCRIPTIVE_MEMBERSHIP_ONLY_DRIVER_UNVERIFIED"
    result.loc[
        known & result.theme_evidence_confidence.eq("SECONDARY_NARRATIVE"),
        "cross_theme_race_status",
    ] = "DESCRIPTIVE_SECONDARY_NARRATIVE_ONLY"
    if training_scope:
        result["cross_theme_race_status"] = "MODEL_ELIGIBLE_COMPLETE_SCOPE"
    result["eligible_for_cross_theme_race_training"] = training_scope
    result["missing_theme_is_weak_theme"] = False
    result["winner_claimed"] = False
    result["entry_authorized"] = False
    result["race_asof"] = asof
    result["cross_theme_race_version"] = VERSION
    counts = result.theme_evidence_confidence.value_counts().to_dict()
    metadata = {
        "version": VERSION, "signal_date": signal_date, "asof": asof,
        "status": "MODEL_ELIGIBLE_COMPLETE_SCOPE" if training_scope
        else "DESCRIPTIVE_ONLY_INCOMPLETE_OR_UNVERIFIED_SCOPE",
        "closing_candidate_count": int(len(result)),
        "first_board_count": int(result.boards.eq(1).sum()),
        "relay_count": int(result.boards.ge(2).sum()),
        "known_theme_candidate_count": int(known.sum()),
        "unknown_theme_candidate_count": int((~known).sum()),
        "confidence_counts": {str(key): int(value) for key, value in counts.items()},
        "full_market_complete": full_market,
        "complete_theme_scope": complete_theme_scope,
        "all_active_theme_verified": all_active_verified,
        "model_eligible_count": int(result.eligible_for_cross_theme_race_training.sum()),
        "missing_theme_is_weak_theme": False, "winner_claimed": False,
        "entry_authorized": False,
        "lane_contract": {
            "first_board": "new-cycle and low-position attention across themes",
            "relay": "height, ladder support and market-recognition capital across themes",
            "cross_lane_common_winner_score_forbidden": True,
        },
        "policy": (
            "Competition counts use same-confidence peers only and describe pressure, not a "
            "winner. Cross-confidence names are counted separately. First-board and relay "
            "fields must be validated separately against B+2/B+3 complete-trade outcomes."
        ),
    }
    return result, metadata


def run(pool_path: Path, manifest_path: Path, theme_paths: list[Path],
        asof: str, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable V170 race archive")
    pool = pd.read_csv(pool_path, dtype={"code": str})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = [pd.read_csv(path, dtype={"code": str}) for path in theme_paths]
    frame, metadata = build(pool, manifest, roles, asof)
    output.mkdir(parents=True)
    output_path = output / "candidate_cross_theme_races.csv"
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    metadata["source_hashes"] = {
        str(path.resolve()): _sha256(path)
        for path in [pool_path, manifest_path, *theme_paths]
    }
    metadata["output_sha256"] = _sha256(output_path)
    (output / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--theme-roles", required=True, action="append", type=Path)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.pool, args.manifest, args.theme_roles,
                         args.asof, args.output), ensure_ascii=False, indent=2))
