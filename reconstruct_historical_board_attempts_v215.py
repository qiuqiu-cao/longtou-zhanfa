"""Reconstruct closing boards and failed attempted heights from daily bars.

The reconstruction is useful for same-height competition research but does not
repair survivorship bias in a current-security universe.  A failed board is an
N+1 attempt only when the same code closed at N boards in the immediately
preceding market session.  Missing sessions reset the chain; they are never
skipped to manufacture continuity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v215-historical-board-attempts-1"
MAIN_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool(series: pd.Series) -> pd.Series:
    return series.map(lambda value: value is True or str(value).strip().lower() in {"true", "1", "1.0"})


def _limit_price(pre_close: pd.Series) -> pd.Series:
    values = pd.to_numeric(pre_close, errors="coerce")
    return np.floor(values * 110 + .5 + 1e-9) / 100


def reconstruct(bars: pd.DataFrame, inventory: pd.DataFrame | None = None):
    required = {"date", "code", "name", "high", "close", "pre_close",
                "st", "new_listing", "status_verified"}
    if required - set(bars):
        raise ValueError(f"daily bar fields missing: {sorted(required - set(bars))}")
    rows = bars.copy()
    rows["code"] = rows.code.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    rows["date"] = pd.to_datetime(rows.date, errors="raise").dt.strftime("%Y-%m-%d")
    rows = rows.drop_duplicates(["date", "code"], keep="last")
    rows = rows[rows.code.str.startswith(MAIN_PREFIXES)].copy()
    rows["st_flag"] = _bool(rows.st)
    rows["new_listing_flag"] = _bool(rows.new_listing)
    rows["status_verified_flag"] = _bool(rows.status_verified)
    # Keep unverified rows for descriptive reconstruction, but never promote them
    # to a formal non-ST universe.  Requiring status_verified here would erase the
    # whole current-universe history because the source truthfully marks that
    # point-in-time status was not independently archived.
    rows["eligible_scope"] = (~rows.st_flag & ~rows.new_listing_flag)
    rows["scope_evidence_status"] = np.where(
        rows.status_verified_flag,
        "POINT_IN_TIME_STATUS_VERIFIED",
        "CURRENT_UNIVERSE_STATUS_RECONSTRUCTION_UNVERIFIED",
    )
    rows["limit_price_10"] = _limit_price(rows.pre_close)
    high = pd.to_numeric(rows.high, errors="coerce")
    close = pd.to_numeric(rows.close, errors="coerce")
    rows["closing_limit"] = rows.eligible_scope & close.sub(rows.limit_price_10).abs().le(.011)
    rows["touched_limit"] = rows.eligible_scope & high.ge(rows.limit_price_10.sub(.011))
    rows["failed_board"] = rows.touched_limit & ~rows.closing_limit

    calendar = sorted(rows.date.unique())
    date_index = {date: index for index, date in enumerate(calendar)}
    rows["session_index"] = rows.date.map(date_index)
    rows = rows.sort_values(["code", "session_index"]).reset_index(drop=True)
    grouped = rows.groupby("code", sort=False)
    previous_index = grouped.session_index.shift()
    previous_limit = grouped.closing_limit.shift().fillna(False).astype(bool)
    adjacent = rows.session_index.sub(previous_index).eq(1)
    rows["adjacent_previous_bar"] = adjacent
    link = rows.closing_limit & previous_limit & adjacent
    run = (~link).groupby(rows.code).cumsum()
    rows["boards"] = rows.closing_limit.astype(int).groupby([rows.code, run]).cumsum()
    previous_boards = rows.groupby("code", sort=False).boards.shift()
    rows["attempted_height"] = np.where(
        rows.failed_board, np.where(previous_limit & adjacent, previous_boards + 1, 1), np.nan)
    candidates = rows[rows.closing_limit | rows.failed_board].copy()
    candidates["signal_date"] = candidates.date
    candidates["pool_assignment"] = np.where(
        candidates.closing_limit, "CLOSING_LIMIT_UP", "FAILED_BOARD_OBSERVATION")
    candidates["board_height"] = np.where(
        candidates.closing_limit, candidates.boards, candidates.attempted_height).astype(int)
    candidates["board_group"] = np.where(
        candidates.pool_assignment.eq("FAILED_BOARD_OBSERVATION"), "FAILED_BOARD",
        np.where(candidates.board_height.eq(1), "FIRST_BOARD", "CONSECUTIVE_BOARD"))
    candidates["height_basis"] = np.where(
        candidates.closing_limit, "CONSECUTIVE_DAILY_CLOSE_LIMITS",
        np.where(candidates.board_height.gt(1), "IMMEDIATE_PREVIOUS_SESSION_CLOSE_LIMIT_PLUS_ONE",
                 "NO_IMMEDIATE_PREVIOUS_CLOSE_LIMIT_FIRST_BOARD_ATTEMPT"))
    candidates["universe_scope"] = "CURRENT_SECURITY_UNIVERSE_HISTORY_SURVIVORSHIP_BIAS"
    keep = ["signal_date", "code", "name", "pool_assignment", "board_height", "board_group",
            "height_basis", "limit_price_10", "high", "close", "pre_close",
            "adjacent_previous_bar", "status_verified_flag", "scope_evidence_status",
            "universe_scope"]
    candidates = candidates[keep].sort_values(["signal_date", "pool_assignment", "board_height", "code"])

    comparison = {"inventory_rows": None, "matched_rows": None, "inventory_recall": None,
                  "reconstruction_precision": None}
    if inventory is not None:
        required_inventory = {"signal_date", "code"}
        if required_inventory - set(inventory):
            raise ValueError("inventory keys missing")
        inv = inventory.copy()
        inv["code"] = inv.code.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        inv_keys = set(zip(inv.signal_date.astype(str), inv.code))
        close_rows = candidates[candidates.pool_assignment.eq("CLOSING_LIMIT_UP")]
        rebuilt_keys = set(zip(close_rows.signal_date.astype(str), close_rows.code))
        matched = len(inv_keys & rebuilt_keys)
        comparison = {
            "inventory_rows": len(inv_keys), "reconstructed_closing_rows": len(rebuilt_keys),
            "matched_rows": matched,
            "inventory_recall": matched / len(inv_keys) if inv_keys else None,
            "reconstruction_precision": matched / len(rebuilt_keys) if rebuilt_keys else None,
        }
    report = {
        "version": VERSION, "status": "DIAGNOSTIC_ONLY",
        "bar_rows": int(len(rows)), "codes": int(rows.code.nunique()),
        "sessions": int(len(calendar)), "candidate_rows": int(len(candidates)),
        "closing_limit_rows": int(candidates.pool_assignment.eq("CLOSING_LIMIT_UP").sum()),
        "failed_board_rows": int(candidates.pool_assignment.eq("FAILED_BOARD_OBSERVATION").sum()),
        "multi_board_failed_attempt_rows": int((
            candidates.pool_assignment.eq("FAILED_BOARD_OBSERVATION")
            & candidates.board_height.gt(1)).sum()),
        "point_in_time_status_verified_candidate_rows": int(
            candidates.status_verified_flag.sum()),
        "status_unverified_candidate_rows": int(
            (~candidates.status_verified_flag).sum()),
        "inventory_comparison": comparison,
        "full_historical_market_complete": False,
        "formal_factor_allowed": False,
        "governance": {
            "missing_session_skipped_for_height": False,
            "rounded_10pct_daily_proxy": True,
            "current_universe_survivorship_bias_resolved": False,
            "point_in_time_nonst_scope_required_for_formal_use": True,
            "failed_attempts_mixed_with_closing_candidates": False,
        },
    }
    return candidates, report


def run(bars_path, output_dir, inventory_path=None):
    source, target = Path(bars_path), Path(output_dir)
    inventory_source = Path(inventory_path) if inventory_path else None
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable board-attempt reconstruction")
    usecols = ["date", "code", "name", "high", "close", "pre_close", "st",
               "new_listing", "status_verified"]
    bars = pd.read_csv(source, dtype={"code": str}, usecols=usecols, low_memory=False)
    inventory = (pd.read_csv(inventory_source, dtype={"code": str})
                 if inventory_source else None)
    rows, report = reconstruct(bars, inventory)
    target.mkdir(parents=True)
    output = target / "historical_closing_and_failed_board_attempts.csv"
    rows.to_csv(output, index=False, encoding="utf-8-sig")
    paths = [source, Path(__file__)] + ([inventory_source] if inventory_source else [])
    report["source_hashes"] = {str(path.resolve()): _sha256(path) for path in paths}
    report["output_sha256"] = _sha256(output)
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", required=True)
    parser.add_argument("--inventory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.bars, args.output, args.inventory), ensure_ascii=False, indent=2))
