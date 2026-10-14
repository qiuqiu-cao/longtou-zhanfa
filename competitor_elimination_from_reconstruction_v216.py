"""Map reconstructed same-height elimination pressure onto candidate rows.

This closes a descriptive-coverage gap, not the formal evidence gate.  The
underlying daily bars contain the current security universe and unverified
point-in-time ST status, so values are explicitly non-formal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v216-reconstructed-competitor-elimination-1"
STATUS = "CURRENT_UNIVERSE_RECONSTRUCTION_UNVERIFIED"
KEYS = ["signal_date", "code"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(attempts: pd.DataFrame, candidates: pd.DataFrame):
    required_attempts = {"signal_date", "code", "pool_assignment", "board_height"}
    required_candidates = {"signal_date", "code", "name", "boards"}
    if required_attempts - set(attempts):
        raise ValueError(f"attempt fields missing: {sorted(required_attempts - set(attempts))}")
    if required_candidates - set(candidates):
        raise ValueError(f"candidate fields missing: {sorted(required_candidates - set(candidates))}")

    pool = attempts.copy()
    pool["signal_date"] = pool.signal_date.astype(str).str[:10]
    pool["board_height"] = pd.to_numeric(pool.board_height, errors="coerce")
    pool["is_survivor"] = pool.pool_assignment.eq("CLOSING_LIMIT_UP").astype(int)
    pool["is_failed"] = pool.pool_assignment.eq("FAILED_BOARD_OBSERVATION").astype(int)
    counts = pool.groupby(["signal_date", "board_height"], as_index=False).agg(
        same_height_survivor_count=("is_survivor", "sum"),
        same_height_failed_attempt_count=("is_failed", "sum"),
    )

    rows = candidates.copy()
    rows["code"] = rows.code.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    rows["signal_date"] = rows.signal_date.astype(str).str[:10]
    rows["boards"] = pd.to_numeric(rows.boards, errors="coerce")
    rows = rows.merge(
        counts, left_on=["signal_date", "boards"],
        right_on=["signal_date", "board_height"], how="left", validate="many_to_one")
    denominator = rows.same_height_survivor_count + rows.same_height_failed_attempt_count
    available = denominator.gt(0)
    rows["competitor_elimination"] = (
        rows.same_height_failed_attempt_count / denominator).where(available)
    rows["competitor_elimination_evidence_status"] = np.where(
        available, STATUS, "UNAVAILABLE")
    rows["competitor_available_at"] = np.where(
        available, rows.signal_date + "T15:05:00+08:00", None)
    rows["formal_factor_allowed"] = False
    keep = KEYS + ["name", "boards", "competitor_elimination",
                   "competitor_elimination_evidence_status",
                   "same_height_failed_attempt_count", "same_height_survivor_count",
                   "competitor_available_at", "formal_factor_allowed"]
    rows = rows[keep].sort_values(KEYS).reset_index(drop=True)
    report = {
        "version": VERSION, "status": "DIAGNOSTIC_ONLY",
        "candidate_rows": int(len(rows)),
        "sessions": int(rows.signal_date.nunique()),
        "descriptive_reconstruction_rows": int(available.sum()),
        "formal_point_in_time_rows": 0,
        "full_historical_market_complete": False,
        "point_in_time_nonst_scope_verified": False,
        "formal_factor_allowed": False,
        "missing_values_treated_as_zero": False,
    }
    return rows, report


def run(attempts_path, candidates_path, output_dir):
    attempts_source, candidates_source, target = map(
        Path, (attempts_path, candidates_path, output_dir))
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable factor evidence")
    attempts = pd.read_csv(attempts_source, dtype={"code": str}, low_memory=False)
    candidates = pd.read_csv(candidates_source, dtype={"code": str}, low_memory=False)
    rows, report = build(attempts, candidates)
    target.mkdir(parents=True)
    output = target / "competitor_elimination_evidence.csv"
    rows.to_csv(output, index=False, encoding="utf-8-sig")
    paths = [attempts_source, candidates_source, Path(__file__)]
    report["source_hashes"] = {str(path.resolve()): _sha256(path) for path in paths}
    report["output_sha256"] = _sha256(output)
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.attempts, args.candidates, args.output),
                     ensure_ascii=False, indent=2))
