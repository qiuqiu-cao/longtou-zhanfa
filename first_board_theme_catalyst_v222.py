"""V222 point-in-time theme membership and catalyst verifier."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


VERSION = "v222-first-board-theme-catalyst-1"
MEMBERSHIP_OK = {"VERIFIED_PRIMARY_SOURCE", "VERIFIED_COMPANY_DISCLOSURE"}
CATALYST_OK = {"VERIFIED_ACTIVE", "VERIFIED_NEW_INFORMATION"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(matrix: pd.DataFrame, evidence: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    required = {
        "signal_date", "theme_name", "membership_status", "membership_source_url",
        "membership_available_at", "catalyst_status", "catalyst_source_url",
        "catalyst_published_at", "catalyst_available_at", "expectation_status",
    }
    missing = required - set(evidence)
    if missing:
        raise ValueError(f"theme evidence fields missing: {sorted(missing)}")
    if evidence.duplicated(["signal_date", "theme_name"]).any():
        raise ValueError("theme evidence keys must be unique")
    cutoff_ts = pd.Timestamp(cutoff)
    if cutoff_ts.tzinfo is None:
        raise ValueError("cutoff must include timezone")
    evidence = evidence.rename(columns={
        name: f"{name}_evidence" for name in required
        if name not in {"signal_date", "theme_name"}
    })
    merged = matrix.merge(evidence, on=["signal_date", "theme_name"], how="left",
                          validate="one_to_one")
    mtime = pd.to_datetime(merged.membership_available_at_evidence, errors="coerce", utc=True)
    ctime = pd.to_datetime(merged.catalyst_available_at_evidence, errors="coerce", utc=True)
    ptime = pd.to_datetime(merged.catalyst_published_at_evidence, errors="coerce", utc=True)
    cutoff_utc = cutoff_ts.tz_convert("UTC")
    membership = (
        merged.membership_status_evidence.fillna("UNAVAILABLE").isin(MEMBERSHIP_OK)
        & merged.membership_source_url_evidence.fillna("").str.startswith(("http://", "https://"))
        & mtime.notna() & mtime.le(cutoff_utc)
    )
    catalyst = (
        merged.catalyst_status_evidence.fillna("UNAVAILABLE").isin(CATALYST_OK)
        & merged.catalyst_source_url_evidence.fillna("").str.startswith(("http://", "https://"))
        & ctime.notna() & ptime.notna() & ctime.le(cutoff_utc) & ptime.le(cutoff_utc)
    )
    merged["membership_verified_point_in_time"] = membership
    merged["catalyst_verified_point_in_time"] = catalyst
    merged["verified_real_theme"] = membership & catalyst
    merged["verified_lane_v222"] = "UNVERIFIED_OR_INCOMPLETE"
    merged.loc[membership & ~catalyst, "verified_lane_v222"] = "VERIFIED_MEMBERSHIP_ONLY"
    merged.loc[membership & catalyst, "verified_lane_v222"] = "VERIFIED_REAL_THEME"
    merged["entry_authorized"] = False
    return merged


def run(matrix_path: str, evidence_path: str, cutoff: str, output_dir: str) -> dict:
    sources = [Path(matrix_path), Path(evidence_path)]
    matrix = pd.read_csv(sources[0], dtype={"code": str})
    evidence = pd.read_csv(sources[1], dtype={"code": str})
    rows = verify(matrix, evidence, cutoff)
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError("refusing to overwrite immutable V222 output")
    out.mkdir(parents=True)
    csv_path = out / "verified_theme_evidence.csv"
    rows.to_csv(csv_path, index=False, encoding="utf-8-sig")
    report = {
        "version": VERSION,
        "status": "RESEARCH_ONLY",
        "cutoff": cutoff,
        "themes": int(len(rows)),
        "verified_membership": int(rows.membership_verified_point_in_time.sum()),
        "verified_real_theme": int(rows.verified_real_theme.sum()),
        "unavailable_or_incomplete": int((~rows.verified_real_theme).sum()),
        "missing_is_not_negative": True,
        "observed_roster_is_not_verified_catalyst": True,
        "entry_authorized": False,
        "source_hashes": {str(p.resolve()): _sha(p) for p in sources},
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--output", required=True)
    ns = ap.parse_args()
    print(json.dumps(run(ns.matrix, ns.evidence, ns.cutoff, ns.output), ensure_ascii=False, indent=2))
