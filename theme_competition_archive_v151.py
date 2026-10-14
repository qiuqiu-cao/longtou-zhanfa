"""Create one-row-per-candidate theme/competition archives for future backtests.

The archive always starts from the declared complete closing pool and left joins
theme evidence.  Candidates without evidence remain explicit UNAVAILABLE rows;
this prevents both survivorship-by-theme and missing-as-zero leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


VERSION = "v151-theme-competition-archive-1"
FULL_MARKET_VALUES = {
    "verified_full_market", "VERIFIED_FULL_MARKET",
    "proven_full_market", "PROVEN_FULL_MARKET",
}
FACTOR_COLUMNS = [
    "theme_group_id", "theme", "theme_granularity", "theme_role",
    "competition_identity", "theme_factor_schema_status", "driver_type",
    "driver_summary", "driver_first_available_at", "driver_claim_status",
    "expectation_status", "expectation_basis", "positive_catalyst_factor_eligible",
    "same_board_peer_count", "higher_board_peer_count", "same_theme_peer_count",
    "same_theme_board_peer_count", "higher_theme_peer_count", "events",
]


def _stamp(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("invalid timestamp")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("Asia/Shanghai")
    else:
        stamp = stamp.tz_convert("Asia/Shanghai")
    return stamp


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _truthy(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def build(pool, pool_manifest, competition, competition_manifest, asof):
    cutoff = _stamp(asof)
    required_pool = {
        "candidate_id", "signal_date", "code", "name", "boards", "board_group",
        "pool_assignment", "is_mainboard", "is_st", "is_delisting_period",
        "security_status", "limit_up_status", "identity_status", "observed_at",
        "available_at", "amount", "turnover", "first_seal_time", "failed_seal_count",
    }
    if required_pool - set(pool):
        raise ValueError(f"pool fields missing: {sorted(required_pool - set(pool))}")
    signal_date = str(pool_manifest.get("signal_date", ""))
    frame = pool.copy()
    frame["signal_date"] = pd.to_datetime(frame.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
    frame["code"] = frame.code.astype(str).str.zfill(6)
    frame = frame[frame.pool_assignment.eq("CLOSING_LIMIT_UP")].copy()
    if (
        not signal_date or frame.empty or not frame.signal_date.eq(signal_date).all()
        or frame.code.duplicated().any() or frame.candidate_id.duplicated().any()
        or len(frame) != int(pool_manifest.get("mainboard_closing_limit_rows", -1))
        or not _truthy(pool_manifest.get("complete_within_archived_roster"))
    ):
        raise ValueError("pool is not a unique complete declared closing roster")
    valid_scope = (
        frame.is_mainboard.map(_truthy)
        & ~frame.is_st.map(_truthy)
        & ~frame.is_delisting_period.map(_truthy)
        & frame.security_status.eq("verified")
        & frame.limit_up_status.astype(str).str.startswith(("VERIFIED", "verified"))
        & frame.identity_status.eq("verified")
    )
    if not valid_scope.all():
        raise ValueError("pool contains ineligible or unverified candidates")
    for _, row in frame.iterrows():
        if not _stamp(row.observed_at) <= _stamp(row.available_at) <= cutoff:
            raise ValueError("pool evidence was unavailable by archive asof")

    comp = competition.copy()
    if comp.empty:
        comp = pd.DataFrame(columns=["signal_date", "code"] + FACTOR_COLUMNS)
    if not {"signal_date", "code"} <= set(comp):
        raise ValueError("competition rows require signal_date and code")
    comp["signal_date"] = pd.to_datetime(comp.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
    comp["code"] = comp.code.astype(str).str.zfill(6)
    if (
        comp.code.duplicated().any() or not comp.signal_date.eq(signal_date).all()
        or not set(comp.code) <= set(frame.code)
        or str(competition_manifest.get("signal_date", "")) != signal_date
        or int(competition_manifest.get("closing_candidate_count", -1)) != len(frame)
        or int(competition_manifest.get("verified_theme_candidate_count", -1)) != len(comp)
    ):
        raise ValueError("competition scope does not match the closing pool")
    if len(comp):
        if not {"observed_at", "available_at", "data_status"} <= set(comp):
            raise ValueError("competition timestamps/status missing")
        for _, row in comp.iterrows():
            if (
                not _stamp(row.observed_at) <= _stamp(row.available_at) <= cutoff
                or row.data_status != "verified"
            ):
                raise ValueError("competition evidence was unavailable or unverified")

    for column in FACTOR_COLUMNS:
        if column not in comp:
            comp[column] = pd.NA
    comp_view = comp[["code"] + FACTOR_COLUMNS].copy()
    result = frame.merge(comp_view, on="code", how="left", validate="one_to_one")
    has_theme = result.theme_group_id.notna() & result.theme_group_id.astype(str).ne("")
    result["theme_factor_status"] = "UNAVAILABLE"
    result.loc[has_theme, "theme_factor_status"] = "LEGACY_DESCRIPTIVE_ONLY"
    v150 = has_theme & result.theme_factor_schema_status.eq("V150_COMPLETE")
    result.loc[v150, "theme_factor_status"] = "V150_POINT_IN_TIME"
    # Nullable numeric/boolean theme fields stay missing on unknown candidates.
    result["theme_membership_known"] = has_theme.astype("boolean")
    result.loc[~has_theme, "theme_membership_known"] = pd.NA
    full_market = str(pool_manifest.get("full_market_completeness", "")) in FULL_MARKET_VALUES
    result["full_market_complete"] = full_market
    result["eligible_for_leader_factor_training"] = v150 & full_market
    result["missing_theme_is_negative"] = False
    result["archive_version"] = VERSION
    result["archive_asof"] = cutoff.isoformat()

    metadata = {
        "version": VERSION,
        "status": "FULL_MARKET_FACTOR_ARCHIVE_READY" if result.eligible_for_leader_factor_training.all()
                  else "DESCRIPTIVE_ARCHIVE_ONLY",
        "signal_date": signal_date,
        "closing_candidate_count": int(len(result)),
        "first_board_count": int((pd.to_numeric(result.boards) == 1).sum()),
        "relay_count": int((pd.to_numeric(result.boards) >= 2).sum()),
        "theme_known_count": int(has_theme.sum()),
        "v150_factor_count": int(v150.sum()),
        "theme_coverage": float(has_theme.mean()),
        "v150_factor_coverage": float(v150.mean()),
        "full_market_complete": full_market,
        "model_eligible_count": int(result.eligible_for_leader_factor_training.sum()),
        "missing_theme_is_negative": False,
        "unavailable_codes": sorted(result.loc[~has_theme, "code"].tolist()),
        "entry_authorized": False,
    }
    return result, metadata


def run(pool_path, pool_manifest_path, competition_path, competition_manifest_path,
        asof, output_dir):
    paths = list(map(Path, (pool_path, pool_manifest_path, competition_path,
                            competition_manifest_path)))
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable archive")
    result, metadata = build(
        pd.read_csv(paths[0], dtype={"code": str}),
        json.loads(paths[1].read_text(encoding="utf-8")),
        pd.read_csv(paths[2], dtype={"code": str}),
        json.loads(paths[3].read_text(encoding="utf-8")),
        asof,
    )
    target.mkdir(parents=True)
    output = target / "candidate_theme_competition.csv"
    result.to_csv(output, index=False, encoding="utf-8-sig")
    metadata["source_hashes"] = {str(path.resolve()): _sha(path) for path in paths}
    metadata["output_sha256"] = _sha(output)
    (target / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--pool-manifest", required=True)
    parser.add_argument("--competition", required=True)
    parser.add_argument("--competition-manifest", required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.pool, args.pool_manifest, args.competition,
                         args.competition_manifest, args.asof, args.output),
                     ensure_ascii=False, indent=2))

