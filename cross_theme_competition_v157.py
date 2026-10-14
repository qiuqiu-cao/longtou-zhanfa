"""Derive cross-theme capital-competition factors for first boards and relays.

The factors are structural facts, not a hand-made winner score.  Model training
is allowed only when the full closing-limit pool and every primary specific
theme are known; otherwise the known subset remains descriptive and missing
themes never become weak-theme negatives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v157-cross-theme-competition-1"


def _truthy(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(archive: pd.DataFrame, archive_manifest: dict) -> tuple[pd.DataFrame, dict]:
    required = {
        "candidate_id", "signal_date", "code", "name", "boards", "board_group",
        "amount", "theme_group_id", "theme", "theme_factor_status",
        "eligible_for_leader_factor_training", "full_market_complete",
    }
    missing = required - set(archive)
    if missing:
        raise ValueError(f"archive fields missing: {sorted(missing)}")
    frame = archive.copy()
    frame["code"] = frame.code.astype(str).str.zfill(6)
    frame["boards"] = pd.to_numeric(frame.boards, errors="coerce")
    frame["amount"] = pd.to_numeric(frame.amount, errors="coerce")
    signal_dates = pd.to_datetime(frame.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
    if (
        frame.empty or frame.code.duplicated().any() or frame.candidate_id.duplicated().any()
        or frame.boards.isna().any() or not frame.boards.mod(1).eq(0).all()
        or signal_dates.nunique() != 1
        or int(archive_manifest.get("closing_candidate_count", -1)) != len(frame)
        or str(archive_manifest.get("signal_date", "")) != signal_dates.iloc[0]
    ):
        raise ValueError("archive is not a unique one-session closing-limit pool")
    frame["boards"] = frame.boards.astype(int)
    expected_group = np.where(frame.boards.eq(1), "FIRST_BOARD", "CONSECUTIVE_BOARD")
    if not frame.board_group.astype(str).eq(expected_group).all():
        raise ValueError("board_group conflicts with board height")

    known = frame.theme_group_id.notna() & frame.theme_group_id.astype(str).str.strip().ne("")
    known_frame = frame[known].copy()
    full_market = bool(frame.full_market_complete.map(_truthy).all())
    full_theme_scope = bool(known.all())
    cross_theme_scope_complete = full_market and full_theme_scope

    theme_summary = []
    for theme_id, group in known_frame.groupby("theme_group_id", sort=True):
        boards = group.boards
        theme_summary.append({
            "theme_group_id": theme_id,
            "theme_candidate_count": int(len(group)),
            "theme_first_board_count": int(boards.eq(1).sum()),
            "theme_relay_count": int(boards.ge(2).sum()),
            "theme_max_board": int(boards.max()),
            "theme_total_amount": float(group.amount.sum(min_count=1)) if group.amount.notna().any() else np.nan,
            "theme_has_first_relay_ladder": bool(boards.eq(1).any() and boards.ge(2).any()),
        })
    themes = pd.DataFrame(theme_summary)
    if not themes.empty:
        themes["theme_height_rank"] = themes.theme_max_board.rank(
            method="dense", ascending=False).astype(int)
        themes["theme_breadth_rank"] = themes.theme_candidate_count.rank(
            method="dense", ascending=False).astype(int)
        themes["theme_relay_rank"] = themes.theme_relay_count.rank(
            method="dense", ascending=False).astype(int)
        known_total_amount = themes.theme_total_amount.sum(min_count=1)
        themes["theme_amount_share_known"] = (
            themes.theme_total_amount / known_total_amount
            if pd.notna(known_total_amount) and known_total_amount > 0 else np.nan
        )
        themes["theme_limitup_share_known"] = themes.theme_candidate_count / len(known_frame)

    result = frame.copy()
    if themes.empty:
        for column in [
            "theme_candidate_count", "theme_first_board_count", "theme_relay_count",
            "theme_max_board", "theme_total_amount", "theme_has_first_relay_ladder",
            "theme_height_rank", "theme_breadth_rank", "theme_relay_rank",
            "theme_amount_share_known", "theme_limitup_share_known",
        ]:
            result[column] = pd.NA
    else:
        result = result.merge(themes, on="theme_group_id", how="left", validate="many_to_one")

    factor_columns = [
        "cross_theme_other_theme_count", "cross_theme_same_height_candidate_count",
        "cross_theme_higher_candidate_count", "cross_theme_higher_theme_count",
        "cross_theme_same_max_height_theme_count", "cross_theme_height_gap_to_global",
        "first_board_other_theme_first_board_count", "first_board_other_theme_count",
        "first_board_theme_has_existing_relay", "first_board_theme_is_singleton",
        "relay_other_theme_relay_count", "relay_cross_theme_same_height_count",
        "relay_cross_theme_higher_count", "relay_same_theme_lower_support_count",
        "relay_same_theme_same_or_higher_competitor_count",
    ]
    for column in factor_columns:
        result[column] = pd.NA
    all_known_theme_ids = set(known_frame.theme_group_id.astype(str))
    global_max = int(frame.boards.max())
    events_by_index = {}
    for index, row in result[known].iterrows():
        theme_id = str(row.theme_group_id)
        boards = int(row.boards)
        other = known_frame[known_frame.theme_group_id.astype(str).ne(theme_id)]
        same_theme = known_frame[known_frame.theme_group_id.astype(str).eq(theme_id)]
        other_theme_max = other.groupby("theme_group_id").boards.max() if len(other) else pd.Series(dtype=float)
        base = {
            "cross_theme_other_theme_count": len(all_known_theme_ids - {theme_id}),
            "cross_theme_same_height_candidate_count": int(other.boards.eq(boards).sum()),
            "cross_theme_higher_candidate_count": int(other.boards.gt(boards).sum()),
            "cross_theme_higher_theme_count": int((other_theme_max > boards).sum()),
            "cross_theme_same_max_height_theme_count": int((other_theme_max == int(row.theme_max_board)).sum()),
            "cross_theme_height_gap_to_global": global_max - int(row.theme_max_board),
        }
        for column, value in base.items():
            result.at[index, column] = value
        events = []
        if base["cross_theme_same_height_candidate_count"]:
            events.append({"type": "CROSS_THEME_SAME_HEIGHT_RACE",
                           "count": base["cross_theme_same_height_candidate_count"]})
        if base["cross_theme_higher_theme_count"]:
            events.append({"type": "CROSS_THEME_HIGHER_PRESSURE",
                           "theme_count": base["cross_theme_higher_theme_count"]})
        if boards == 1:
            other_first = int(other.boards.eq(1).sum())
            result.at[index, "first_board_other_theme_first_board_count"] = other_first
            result.at[index, "first_board_other_theme_count"] = len(all_known_theme_ids - {theme_id})
            result.at[index, "first_board_theme_has_existing_relay"] = bool(
                same_theme.boards.ge(2).any())
            result.at[index, "first_board_theme_is_singleton"] = bool(len(same_theme) == 1)
            if other_first:
                events.append({"type": "FIRST_BOARD_CROSS_THEME_PIONEER_RACE",
                               "other_candidate_count": other_first})
        else:
            result.at[index, "relay_other_theme_relay_count"] = int(other.boards.ge(2).sum())
            result.at[index, "relay_cross_theme_same_height_count"] = int(other.boards.eq(boards).sum())
            result.at[index, "relay_cross_theme_higher_count"] = int(other.boards.gt(boards).sum())
            result.at[index, "relay_same_theme_lower_support_count"] = int(
                same_theme.boards.lt(boards).sum())
            result.at[index, "relay_same_theme_same_or_higher_competitor_count"] = int(
                (same_theme.code.ne(row.code) & same_theme.boards.ge(boards)).sum())
            if int(other.boards.eq(boards).sum()):
                events.append({"type": "RELAY_CROSS_THEME_HEIGHT_RACE",
                               "other_candidate_count": int(other.boards.eq(boards).sum())})
        events_by_index[index] = json.dumps(events, ensure_ascii=False, separators=(",", ":"))

    result["cross_theme_events"] = pd.Series(events_by_index, dtype="object")
    result["cross_theme_scope_complete"] = cross_theme_scope_complete
    candidate_v150 = result.theme_factor_status.eq("V150_POINT_IN_TIME")
    result["eligible_for_cross_theme_factor_training"] = (
        candidate_v150 & cross_theme_scope_complete
    )
    result["cross_theme_factor_status"] = "UNAVAILABLE"
    result.loc[known, "cross_theme_factor_status"] = "DESCRIPTIVE_KNOWN_THEME_SUBSET"
    result.loc[result.eligible_for_cross_theme_factor_training,
               "cross_theme_factor_status"] = "MODEL_ELIGIBLE_COMPLETE_SCOPE"
    result["missing_theme_is_weak_theme"] = False
    result["cross_theme_version"] = VERSION

    metadata = {
        "version": VERSION,
        "status": "MODEL_FACTOR_ARCHIVE_READY" if cross_theme_scope_complete and candidate_v150.all()
                  else "DESCRIPTIVE_ONLY_INCOMPLETE_CROSS_THEME_SCOPE",
        "signal_date": signal_dates.iloc[0],
        "candidate_count": int(len(result)),
        "known_theme_candidate_count": int(known.sum()),
        "known_theme_count": int(len(all_known_theme_ids)),
        "known_theme_coverage": float(known.mean()),
        "full_market_complete": full_market,
        "full_theme_scope": full_theme_scope,
        "cross_theme_scope_complete": cross_theme_scope_complete,
        "model_eligible_count": int(result.eligible_for_cross_theme_factor_training.sum()),
        "first_board_count": int(result.boards.eq(1).sum()),
        "relay_count": int(result.boards.ge(2).sum()),
        "missing_theme_is_weak_theme": False,
        "winner_claimed": False,
        "entry_authorized": False,
        "interpretation": {
            "first_board": "competes for new-cycle and low-position attention across themes",
            "relay": "competes for height, middle-board support and theme leadership across themes",
        },
    }
    return result, metadata


def run(archive_path, manifest_path, output_dir):
    archive_source, manifest_source, target = map(
        Path, (archive_path, manifest_path, output_dir))
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable cross-theme archive")
    result, metadata = build(
        pd.read_csv(archive_source, dtype={"code": str}),
        json.loads(manifest_source.read_text(encoding="utf-8")),
    )
    target.mkdir(parents=True)
    output = target / "candidate_cross_theme_competition.csv"
    result.to_csv(output, index=False, encoding="utf-8-sig")
    metadata["source_hashes"] = {
        str(path.resolve()): _sha256(path) for path in (archive_source, manifest_source)}
    metadata["output_sha256"] = _sha256(output)
    (target / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.archive, args.manifest, args.output),
                     ensure_ascii=False, indent=2))
