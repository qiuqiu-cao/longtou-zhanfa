"""Build model-auditable specific-theme roles from point-in-time evidence.

V150 replaces broad post-hoc concept grouping with an explicit evidence schema:
specific subtheme, catalyst type and first availability, membership source, and
an as-of expectation assessment.  Narrative/rumour evidence can be archived
but cannot become a positive catalyst factor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


VERSION = "v150-specific-theme-roles-1"
DRIVER_TYPES = {"policy", "industry", "event", "announcement", "pure_emotion"}
CLAIM_STATUSES = {"verified_fact", "market_narrative", "rumor"}
EXPECTATION_STATUSES = {"underrecognized", "partially_traded", "fully_traded", "unknown"}
POSITIVE_SOURCE_TIERS = {"A_PRIMARY", "B_OFFICIAL_MEDIA"}
OUTPUT_COLUMNS = [
    "signal_date", "code", "theme_status", "theme_role", "theme_group_id", "theme",
    "primary_evidence_id", "evidence_published_at", "evidence_status",
    "theme_granularity", "specificity_basis", "driver_type", "driver_summary",
    "driver_source_url", "driver_first_available_at", "driver_claim_status",
    "expectation_status", "expectation_basis", "expectation_assessed_at",
    "membership_source_url", "membership_published_at", "source_tier",
    "theme_support_ex_self", "same_height_rivals", "theme_ladder_verified",
    "identity_tags", "capacity_rank_in_theme", "positive_catalyst_factor_eligible",
]
REQUIRED_GROUP_FIELDS = {
    "theme_group_id", "theme", "theme_granularity", "specificity_basis", "codes",
    "driver_type", "driver_summary", "driver_source_url", "driver_first_available_at",
    "driver_claim_status", "expectation_status", "expectation_basis",
    "expectation_assessed_at", "membership_source_url", "membership_published_at",
    "source_tier",
}


def _stamp(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("invalid timestamp")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("Asia/Shanghai")
    else:
        stamp = stamp.tz_convert("Asia/Shanghai")
    return stamp


def _nonempty(group, field):
    value = group.get(field)
    return isinstance(value, str) and bool(value.strip())


def validate_evidence(evidence, asof):
    if set(evidence) - {"signal_date", "groups", "schema_version"}:
        raise ValueError("unknown top-level evidence fields")
    if evidence.get("schema_version") != VERSION or not evidence.get("signal_date"):
        raise ValueError("V150 schema_version and signal_date required")
    if not isinstance(evidence.get("groups"), list):
        raise ValueError("groups must be a list")
    cutoff = _stamp(asof)
    seen_codes, seen_groups = set(), set()
    for group in evidence["groups"]:
        missing = REQUIRED_GROUP_FIELDS - set(group)
        if missing:
            raise ValueError(f"theme group missing fields: {sorted(missing)}")
        if set(group) - REQUIRED_GROUP_FIELDS:
            raise ValueError("unknown group fields")
        group_id = str(group["theme_group_id"]).strip()
        if not group_id or group_id in seen_groups:
            raise ValueError("duplicate or blank theme_group_id")
        seen_groups.add(group_id)
        for field in REQUIRED_GROUP_FIELDS - {"codes"}:
            if not _nonempty(group, field):
                raise ValueError(f"blank theme field: {field}")
        if group["theme_granularity"] != "specific_subtheme":
            raise ValueError("only specific_subtheme evidence is admitted")
        if group["driver_type"] not in DRIVER_TYPES:
            raise ValueError("invalid driver_type")
        if group["driver_claim_status"] not in CLAIM_STATUSES:
            raise ValueError("invalid driver_claim_status")
        if group["expectation_status"] not in EXPECTATION_STATUSES:
            raise ValueError("invalid expectation_status")
        codes = [str(code).zfill(6) for code in group["codes"]]
        if len(codes) < 2 or len(codes) != len(set(codes)) or not all(
            code.isdigit() and len(code) == 6 for code in codes
        ):
            raise ValueError("at least two unique six-digit theme members required")
        overlap = seen_codes.intersection(codes)
        if overlap:
            raise ValueError(f"primary theme membership duplicated: {sorted(overlap)}")
        seen_codes.update(codes)
        for field in ("driver_first_available_at", "membership_published_at",
                      "expectation_assessed_at"):
            if _stamp(group[field]) > cutoff:
                raise ValueError(f"future evidence relative to asof: {field}")
        if (group["driver_type"] == "pure_emotion"
                and group["driver_claim_status"] == "verified_fact"):
            raise ValueError("pure emotion cannot be encoded as a verified factual catalyst")


def build(pool: pd.DataFrame, evidence: dict, asof: str):
    validate_evidence(evidence, asof)
    required = {"signal_date", "code", "name", "boards", "pool_assignment", "amount",
                "first_seal_time"}
    if required - set(pool):
        raise ValueError(f"pool fields missing: {sorted(required - set(pool))}")
    signal_date = str(evidence["signal_date"])
    closing = pool.copy()
    closing["signal_date"] = pd.to_datetime(closing.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
    closing["code"] = closing.code.astype(str).str.zfill(6)
    closing = closing[
        closing.signal_date.eq(signal_date) & closing.pool_assignment.eq("CLOSING_LIMIT_UP")
    ].copy()
    if closing.code.duplicated().any() or closing.empty:
        raise ValueError("unique nonempty closing pool required")
    closing["boards"] = pd.to_numeric(closing.boards, errors="coerce")
    closing["amount"] = pd.to_numeric(closing.amount, errors="coerce")
    closing["first_seal_time"] = pd.to_numeric(closing.first_seal_time, errors="coerce")
    if closing[["boards", "amount", "first_seal_time"]].isna().any().any():
        raise ValueError("role construction requires boards, amount and first seal time")
    closing = closing.set_index("code")
    rows = []
    for group in evidence["groups"]:
        members = [str(code).zfill(6) for code in group["codes"] if str(code).zfill(6) in closing.index]
        if len(members) < 2:
            continue
        levels = {code: int(closing.loc[code, "boards"]) for code in members}
        max_level = max(levels.values())
        max_codes = [code for code in members if levels[code] == max_level]
        amount_order = sorted(members, key=lambda code: (-float(closing.loc[code, "amount"]), code))
        first_time = min(float(closing.loc[code, "first_seal_time"]) for code in members)
        distinct = sorted(set(levels.values()))
        ladder = any(right == left + 1 for left, right in zip(distinct, distinct[1:]))
        positive = (
            group["driver_type"] != "pure_emotion"
            and group["driver_claim_status"] == "verified_fact"
            and group["source_tier"] in POSITIVE_SOURCE_TIERS
            and group["expectation_status"] in {"underrecognized", "partially_traded"}
        )
        theme_status = (
            "verified_specific_point_in_time_schema"
            if group["driver_claim_status"] == "verified_fact"
            and group["source_tier"] in POSITIVE_SOURCE_TIERS
            else "rumor_specific_point_in_time_schema"
            if group["driver_claim_status"] == "rumor"
            else "observed_specific_point_in_time_schema"
        )
        complete_evidence_time = max(
            _stamp(group["driver_first_available_at"]),
            _stamp(group["membership_published_at"]),
            _stamp(group["expectation_assessed_at"]),
        )
        for code in members:
            level = levels[code]
            role = ("THEME_HEIGHT_LEADER" if level == max_level and len(max_codes) == 1
                    else "SAME_HEIGHT_CORE" if level == max_level else "THEME_SUPPORT")
            tags = [role]
            if code == amount_order[0]:
                tags.append("CAPACITY_CORE_BY_CLOSE_AMOUNT")
            if float(closing.loc[code, "first_seal_time"]) == first_time:
                tags.append("FIRST_SEAL_PIONEER")
            rows.append({
                "signal_date": signal_date,
                "code": code,
                "theme_status": theme_status,
                "theme_role": role,
                "theme_group_id": group["theme_group_id"],
                "theme": group["theme"],
                "primary_evidence_id": group["membership_source_url"],
                "evidence_published_at": complete_evidence_time.isoformat(),
                "evidence_status": "v150_point_in_time_specific_theme_record",
                "theme_granularity": group["theme_granularity"],
                "specificity_basis": group["specificity_basis"],
                "driver_type": group["driver_type"],
                "driver_summary": group["driver_summary"],
                "driver_source_url": group["driver_source_url"],
                "driver_first_available_at": _stamp(group["driver_first_available_at"]).isoformat(),
                "driver_claim_status": group["driver_claim_status"],
                "expectation_status": group["expectation_status"],
                "expectation_basis": group["expectation_basis"],
                "expectation_assessed_at": _stamp(group["expectation_assessed_at"]).isoformat(),
                "membership_source_url": group["membership_source_url"],
                "membership_published_at": _stamp(group["membership_published_at"]).isoformat(),
                "source_tier": group["source_tier"],
                "theme_support_ex_self": len(members) - 1,
                "same_height_rivals": sum(value == level for value in levels.values()) - 1,
                "theme_ladder_verified": ladder,
                "identity_tags": "|".join(tags),
                "capacity_rank_in_theme": amount_order.index(code) + 1,
                "positive_catalyst_factor_eligible": positive,
            })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def run(pool_path, evidence_path, asof, output_dir):
    pool_source, evidence_source, target = map(Path, (pool_path, evidence_path, output_dir))
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V150 roles")
    evidence = json.loads(evidence_source.read_text(encoding="utf-8"))
    roles = build(pd.read_csv(pool_source, dtype={"code": str}), evidence, asof)
    target.mkdir(parents=True)
    roles_path = target / "candidate_theme_roles.csv"
    roles.to_csv(roles_path, index=False, encoding="utf-8-sig")
    manifest = {
        "version": VERSION,
        "status": "SPECIFIC_THEME_POINT_IN_TIME_SCHEMA_ARCHIVED",
        "signal_date": evidence["signal_date"],
        "asof": asof,
        "candidate_rows": int(len(roles)),
        "theme_groups": int(roles.theme_group_id.nunique()) if len(roles) else 0,
        "positive_catalyst_factor_rows": int(roles.positive_catalyst_factor_eligible.sum()) if len(roles) else 0,
        "missing_candidates_are_unavailable_not_negative": True,
        "entry_authorized": False,
        "source_hashes": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (pool_source, evidence_source)
        },
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.pool, args.evidence, args.asof, args.output),
                     ensure_ascii=False, indent=2))
