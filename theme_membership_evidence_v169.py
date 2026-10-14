"""Audit member-level theme evidence without confusing business membership with a catalyst.

V169 deliberately separates three questions that were previously easy to mix:
1. Does the company really belong to the stated business branch?
2. Was an active driver for that branch public before the decision cutoff?
3. Was the expectation assessment frozen before that cutoff?

Verified business membership may support a descriptive same-chain roster.  It is
not a positive factor, formal high-to-low mapping, or entry permission unless the
driver and expectation gates independently pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


VERSION = "v169-member-theme-evidence-1"
MEMBERSHIP_STATUSES = {
    "verified_primary", "verified_official_comparator", "secondary", "rumor",
}
VERIFIED_MEMBERSHIP = {"verified_primary", "verified_official_comparator"}
DRIVER_STATUSES = {
    "verified_primary", "verified_official_media",
    "secondary_market_narrative", "rumor_unverified",
}
VERIFIED_DRIVER = {"verified_primary", "verified_official_media"}
EXPECTATION_STATUSES = {
    "underrecognized", "partially_traded", "fully_traded", "unknown",
}
POSITIVE_EXPECTATIONS = {"underrecognized", "partially_traded"}
OUTPUT_COLUMNS = [
    "signal_date", "code", "theme_status", "theme_role", "theme_group_id", "theme",
    "business_branch", "primary_evidence_id", "source_tier", "source_claim_status",
    "membership_claim_status", "membership_source_url", "membership_published_at",
    "membership_available_at", "membership_time_basis", "membership_verified",
    "driver_summary", "driver_source_url", "driver_first_available_at",
    "driver_claim_status", "driver_verified", "expectation_status",
    "expectation_assessed_at", "theme_support_ex_self", "same_height_rivals",
    "theme_ladder_verified", "identity_tags", "capacity_rank_in_theme",
    "positive_catalyst_factor_eligible", "formal_high_low_mapping_eligible",
    "evidence_status", "evidence_available_at", "evidence_time_basis",
]


def _url(value: str, field: str) -> str:
    value = str(value or "").strip()
    if urlparse(value).scheme not in {"http", "https"}:
        raise ValueError(f"{field} must be an http(s) URL")
    return value


def _time(value: str, cutoff: pd.Timestamp, field: str) -> tuple[pd.Timestamp, str]:
    """Parse evidence time; same-day date-only evidence fails closed."""
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"blank timestamp: {field}")
    date_only = len(text) == 10 and text[4] == "-" and text[7] == "-"
    stamp = pd.Timestamp(text)
    if pd.isna(stamp):
        raise ValueError(f"invalid timestamp: {field}")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("Asia/Shanghai")
    else:
        stamp = stamp.tz_convert("Asia/Shanghai")
    if date_only:
        if stamp.date() >= cutoff.date():
            raise ValueError(f"date-only evidence not admissible on asof date: {field}")
        stamp = stamp + pd.Timedelta(hours=23, minutes=59, seconds=59)
        basis = "date_only_prior_day_end_of_day_conservative"
    else:
        basis = "exact_timestamp"
    if stamp > cutoff:
        raise ValueError(f"future evidence relative to asof: {field}")
    return stamp, basis


def _required(record: dict, fields: set[str], label: str) -> None:
    missing = fields - set(record)
    if missing:
        raise ValueError(f"{label} missing fields: {sorted(missing)}")
    for field in fields:
        if not isinstance(record[field], (list, dict)) and not str(record[field]).strip():
            raise ValueError(f"blank {label} field: {field}")


def validate(evidence: dict, asof: str) -> None:
    top = {"schema_version", "signal_date", "drivers", "members"}
    if set(evidence) != top or evidence.get("schema_version") != VERSION:
        raise ValueError("V169 requires only schema_version, signal_date, drivers and members")
    cutoff = pd.Timestamp(asof)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("Asia/Shanghai")
    else:
        cutoff = cutoff.tz_convert("Asia/Shanghai")
    driver_fields = {
        "theme_group_id", "theme", "business_branch", "driver_summary",
        "driver_source_url", "driver_first_available_at", "driver_claim_status",
        "expectation_status", "expectation_assessed_at",
    }
    drivers = {}
    for driver in evidence["drivers"]:
        _required(driver, driver_fields, "driver")
        group_id = str(driver["theme_group_id"]).strip()
        if group_id in drivers:
            raise ValueError("duplicate theme_group_id")
        if driver["driver_claim_status"] not in DRIVER_STATUSES:
            raise ValueError("invalid driver_claim_status")
        if driver["expectation_status"] not in EXPECTATION_STATUSES:
            raise ValueError("invalid expectation_status")
        _url(driver["driver_source_url"], "driver_source_url")
        _time(driver["driver_first_available_at"], cutoff, "driver_first_available_at")
        _time(driver["expectation_assessed_at"], cutoff, "expectation_assessed_at")
        drivers[group_id] = driver
    member_fields = {
        "code", "theme_group_id", "business_branch", "membership_claim",
        "membership_source_url", "membership_source_tier", "membership_published_at",
        "membership_available_at", "membership_claim_status",
    }
    seen = set()
    for member in evidence["members"]:
        _required(member, member_fields, "member")
        code = str(member["code"]).zfill(6)
        if not code.isdigit() or len(code) != 6 or code in seen:
            raise ValueError("member codes must be unique six-digit codes")
        seen.add(code)
        if member["theme_group_id"] not in drivers:
            raise ValueError("member references unknown theme_group_id")
        if member["business_branch"] != drivers[member["theme_group_id"]]["business_branch"]:
            raise ValueError("member business branch does not match driver branch")
        if member["membership_claim_status"] not in MEMBERSHIP_STATUSES:
            raise ValueError("invalid membership_claim_status")
        _url(member["membership_source_url"], "membership_source_url")
        _time(member["membership_published_at"], cutoff, "membership_published_at")
        _time(member["membership_available_at"], cutoff, "membership_available_at")


def build(pool: pd.DataFrame, evidence: dict, asof: str) -> pd.DataFrame:
    validate(evidence, asof)
    required = {"signal_date", "code", "name", "boards", "amount", "first_seal_time",
                "pool_assignment"}
    if required - set(pool):
        raise ValueError(f"pool fields missing: {sorted(required - set(pool))}")
    cutoff = pd.Timestamp(asof)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("Asia/Shanghai")
    else:
        cutoff = cutoff.tz_convert("Asia/Shanghai")
    signal_date = str(evidence["signal_date"])
    closing = pool.copy()
    closing["signal_date"] = pd.to_datetime(closing.signal_date).dt.strftime("%Y-%m-%d")
    closing["code"] = closing.code.astype(str).str.zfill(6)
    closing = closing[
        closing.signal_date.eq(signal_date) & closing.pool_assignment.eq("CLOSING_LIMIT_UP")
    ].copy()
    if closing.empty or closing.code.duplicated().any():
        raise ValueError("unique nonempty closing pool required")
    for field in ("boards", "amount", "first_seal_time"):
        closing[field] = pd.to_numeric(closing[field], errors="coerce")
    if closing[["boards", "amount", "first_seal_time"]].isna().any().any():
        raise ValueError("role construction requires boards, amount and first_seal_time")
    closing = closing.set_index("code")
    drivers = {row["theme_group_id"]: row for row in evidence["drivers"]}
    members_by_group: dict[str, list[dict]] = {}
    for member in evidence["members"]:
        code = str(member["code"]).zfill(6)
        if code in closing.index:
            members_by_group.setdefault(member["theme_group_id"], []).append(member | {"code": code})
    rows = []
    for group_id, members in members_by_group.items():
        if len(members) < 2:
            continue
        driver = drivers[group_id]
        codes = [member["code"] for member in members]
        levels = {code: int(closing.loc[code, "boards"]) for code in codes}
        max_level = max(levels.values())
        height_codes = [code for code in codes if levels[code] == max_level]
        amount_order = sorted(codes, key=lambda code: (-float(closing.loc[code, "amount"]), code))
        first_time = min(float(closing.loc[code, "first_seal_time"]) for code in codes)
        distinct = sorted(set(levels.values()))
        ladder = any(right == left + 1 for left, right in zip(distinct, distinct[1:]))
        driver_verified = driver["driver_claim_status"] in VERIFIED_DRIVER
        driver_time, driver_basis = _time(
            driver["driver_first_available_at"], cutoff, "driver_first_available_at")
        expectation_time, expectation_basis = _time(
            driver["expectation_assessed_at"], cutoff, "expectation_assessed_at")
        for member in members:
            code = member["code"]
            membership_verified = member["membership_claim_status"] in VERIFIED_MEMBERSHIP
            positive = bool(
                membership_verified and driver_verified
                and driver["expectation_status"] in POSITIVE_EXPECTATIONS
            )
            if membership_verified and driver_verified:
                theme_status = "verified_membership_and_driver"
            elif membership_verified:
                theme_status = "verified_membership_driver_unverified"
            elif member["membership_claim_status"] == "rumor":
                theme_status = "rumor_membership"
            else:
                theme_status = "observed_membership_unverified"
            level = levels[code]
            role = ("THEME_HEIGHT_LEADER" if level == max_level and len(height_codes) == 1
                    else "SAME_HEIGHT_CORE" if level == max_level else "THEME_SUPPORT")
            tags = [role]
            if code == amount_order[0]:
                tags.append("CAPACITY_CORE_BY_CLOSE_AMOUNT")
            if float(closing.loc[code, "first_seal_time"]) == first_time:
                tags.append("FIRST_SEAL_PIONEER")
            published, published_basis = _time(
                member["membership_published_at"], cutoff, "membership_published_at")
            available, available_basis = _time(
                member["membership_available_at"], cutoff, "membership_available_at")
            complete_available = max(available, driver_time, expectation_time)
            rows.append({
                "signal_date": signal_date, "code": code, "theme_status": theme_status,
                "theme_role": role, "theme_group_id": group_id, "theme": driver["theme"],
                "business_branch": driver["business_branch"],
                "primary_evidence_id": member["membership_source_url"],
                "source_tier": member["membership_source_tier"],
                "source_claim_status": driver["driver_claim_status"],
                "membership_claim_status": member["membership_claim_status"],
                "membership_source_url": member["membership_source_url"],
                "membership_published_at": published.isoformat(),
                "membership_available_at": available.isoformat(),
                "membership_time_basis": f"published:{published_basis};available:{available_basis}",
                "membership_verified": membership_verified,
                "driver_summary": driver["driver_summary"],
                "driver_source_url": driver["driver_source_url"],
                "driver_first_available_at": driver_time.isoformat(),
                "driver_claim_status": driver["driver_claim_status"],
                "driver_verified": driver_verified,
                "expectation_status": driver["expectation_status"],
                "expectation_assessed_at": expectation_time.isoformat(),
                "theme_support_ex_self": len(codes) - 1,
                "same_height_rivals": sum(value == level for value in levels.values()) - 1,
                "theme_ladder_verified": ladder, "identity_tags": "|".join(tags),
                "capacity_rank_in_theme": amount_order.index(code) + 1,
                "positive_catalyst_factor_eligible": positive,
                "formal_high_low_mapping_eligible": positive,
                "evidence_status": "member_driver_expectation_separately_audited",
                "evidence_available_at": complete_available.isoformat(),
                "evidence_time_basis": (
                    f"membership:{available_basis};driver:{driver_basis};"
                    f"expectation:{expectation_basis}"
                ),
            })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS).sort_values(
        ["theme_group_id", "capacity_rank_in_theme", "code"]
    ).reset_index(drop=True) if rows else pd.DataFrame(columns=OUTPUT_COLUMNS)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(pool_path: Path, evidence_path: Path, asof: str, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable V169 evidence")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    frame = build(pd.read_csv(pool_path, dtype={"code": str}), evidence, asof)
    output.mkdir(parents=True)
    roles_path = output / "candidate_theme_roles.csv"
    frame.to_csv(roles_path, index=False, encoding="utf-8-sig")
    manifest = {
        "version": VERSION, "signal_date": evidence["signal_date"], "asof": asof,
        "status": "MEMBERSHIP_DRIVER_EXPECTATION_SEPARATED",
        "candidate_rows": int(len(frame)),
        "verified_membership_rows": int(frame.membership_verified.sum()) if len(frame) else 0,
        "verified_driver_rows": int(frame.driver_verified.sum()) if len(frame) else 0,
        "positive_factor_rows": int(frame.positive_catalyst_factor_eligible.sum()) if len(frame) else 0,
        "formal_high_low_mapping_rows": int(frame.formal_high_low_mapping_eligible.sum()) if len(frame) else 0,
        "entry_authorized": False,
        "scope_warning": (
            "verified membership alone supports only a descriptive same-chain roster; "
            "an unverified active driver cannot authorize ranking, high-low mapping, or entry"
        ),
        "source_hashes": {
            str(pool_path.resolve()): _sha256(pool_path),
            str(evidence_path.resolve()): _sha256(evidence_path),
            str(roles_path.resolve()): _sha256(roles_path),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.pool, args.evidence, args.asof, args.output),
                     ensure_ascii=False, indent=2))
