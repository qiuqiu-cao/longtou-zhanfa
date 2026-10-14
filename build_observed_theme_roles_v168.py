"""Build theme roles while preserving fact-vs-narrative source confidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "signal_date", "code", "theme_status", "theme_role", "theme_group_id", "theme",
    "primary_evidence_id", "source_tier", "theme_support_ex_self",
    "same_height_rivals", "theme_ladder_verified", "evidence_published_at",
    "evidence_available_at", "evidence_time_basis", "evidence_status",
    "source_claim_status", "source_page_sha256", "identity_tags",
    "capacity_rank_in_theme",
]
CLAIM_TO_STATUS = {
    "verified_primary_or_official": "verified_explicit_source_roster",
    "secondary_market_narrative": "observed_explicit_secondary_roster",
    "rumor_unverified": "rumor_explicit_roster",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def validate(evidence: dict, asof: str) -> None:
    if {"signal_date", "sources", "groups"} - set(evidence):
        raise ValueError("evidence requires signal_date, sources and groups")
    cutoff = _time(asof)
    sources = {source["source_id"]: source for source in evidence["sources"]}
    if len(sources) != len(evidence["sources"]):
        raise ValueError("source ids must be unique")
    for source in sources.values():
        required = {
            "url", "available_at", "timestamp_basis", "source_tier",
            "claim_status", "page_sha256",
        }
        if required - set(source):
            raise ValueError("source observation fields missing")
        if source["claim_status"] not in CLAIM_TO_STATUS:
            raise ValueError("unsupported source claim status")
        if _time(source["available_at"]) > cutoff:
            raise ValueError("source observation is after asof")
        if not str(source["url"]).startswith(("https://", "http://")):
            raise ValueError("source URL must be http(s)")
        if not str(source["page_sha256"]).isalnum() or len(str(source["page_sha256"])) != 64:
            raise ValueError("source page hash is invalid")
    seen = set()
    for group in evidence["groups"]:
        required = {"theme_group_id", "theme", "codes", "source_id", "claim_status", "source_tier"}
        if required - set(group):
            raise ValueError("theme group fields missing")
        if group["source_id"] not in sources:
            raise ValueError("theme group references unknown source")
        source = sources[group["source_id"]]
        if group["claim_status"] != source["claim_status"]:
            raise ValueError("group and source claim status differ")
        if len(group["codes"]) < 2:
            raise ValueError("a theme group requires at least two explicitly named codes")
        for raw in group["codes"]:
            code = str(raw).zfill(6)
            if code in seen:
                raise ValueError(f"code assigned to multiple primary themes: {code}")
            seen.add(code)


def build(pool: pd.DataFrame, evidence: dict, asof: str) -> pd.DataFrame:
    validate(evidence, asof)
    closing = pool.loc[pool.pool_assignment.eq("CLOSING_LIMIT_UP")].copy()
    closing["code"] = closing.code.astype(str).str.zfill(6)
    closing = closing.drop_duplicates("code").set_index("code")
    sources = {source["source_id"]: source for source in evidence["sources"]}
    rows = []
    for group in evidence["groups"]:
        source = sources[group["source_id"]]
        explicit = [str(code).zfill(6) for code in group["codes"]]
        present = [code for code in explicit if code in closing.index]
        if len(present) < 2:
            continue
        levels = {code: int(closing.loc[code, "boards"]) for code in present}
        amounts = {code: float(closing.loc[code, "amount"]) for code in present}
        max_level = max(levels.values())
        height_codes = [code for code, level in levels.items() if level == max_level]
        amount_order = sorted(present, key=lambda code: (-amounts[code], code))
        distinct = sorted(set(levels.values()))
        ladder = any(right == left + 1 for left, right in zip(distinct, distinct[1:]))
        for code in present:
            level = levels[code]
            role = ("THEME_HEIGHT_LEADER" if level == max_level and len(height_codes) == 1
                    else "SAME_HEIGHT_CORE" if level == max_level else "THEME_SUPPORT")
            tags = [role]
            if code == amount_order[0]:
                tags.append("CAPACITY_CORE_BY_CLOSE_AMOUNT")
            if closing.loc[code, "first_seal_time"] == min(
                    closing.loc[peer, "first_seal_time"] for peer in present):
                tags.append("FIRST_SEAL_PIONEER")
            published = source.get("published_at") or ""
            rows.append({
                "signal_date": evidence["signal_date"], "code": code,
                "theme_status": CLAIM_TO_STATUS[group["claim_status"]],
                "theme_role": role, "theme_group_id": group["theme_group_id"],
                "theme": group["theme"], "primary_evidence_id": source["url"],
                "source_tier": group["source_tier"],
                "theme_support_ex_self": len(present) - 1,
                "same_height_rivals": sum(value == level for value in levels.values()) - 1,
                "theme_ladder_verified": ladder,
                "evidence_published_at": published,
                "evidence_available_at": source["available_at"],
                "evidence_time_basis": source["timestamp_basis"],
                "evidence_status": "source_page_explicitly_names_security_and_theme",
                "source_claim_status": group["claim_status"],
                "source_page_sha256": source["page_sha256"],
                "identity_tags": "|".join(tags),
                "capacity_rank_in_theme": amount_order.index(code) + 1,
            })
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS).sort_values(
        ["theme_group_id", "capacity_rank_in_theme", "code"]
    ).reset_index(drop=True)


def run(pool_path: Path, evidence_path: Path, asof: str, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable observed theme roles")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    frame = build(pd.read_csv(pool_path, dtype={"code": str}), evidence, asof)
    output.mkdir(parents=True)
    roles_path = output / "candidate_theme_roles.csv"
    frame.to_csv(roles_path, index=False, encoding="utf-8-sig")
    counts = frame.theme_status.value_counts().to_dict() if len(frame) else {}
    manifest = {
        "version": "v168-confidence-preserving-theme-roles-1",
        "signal_date": evidence["signal_date"], "asof": asof,
        "status": "SOURCE_CONFIDENCE_PRESERVED",
        "candidate_rows": int(len(frame)),
        "theme_groups": int(frame.theme_group_id.nunique()) if len(frame) else 0,
        "theme_status_counts": {str(key): int(value) for key, value in counts.items()},
        "verified_positive_factor_rows": int(
            frame.theme_status.eq("verified_explicit_source_roster").sum()) if len(frame) else 0,
        "secondary_narrative_rows": int(
            frame.theme_status.eq("observed_explicit_secondary_roster").sum()) if len(frame) else 0,
        "high_low_mapping_authorized_rows": int(
            frame.theme_status.eq("verified_explicit_source_roster").sum()) if len(frame) else 0,
        "source_pool": str(pool_path.resolve()), "source_pool_sha256": _sha256(pool_path),
        "evidence_file": str(evidence_path.resolve()), "evidence_sha256": _sha256(evidence_path),
        "output_sha256": _sha256(roles_path),
        "scope_warning": (
            "secondary narratives describe what a page grouped together; they do not verify "
            "company facts, catalyst first-public time, expectation saturation or a buy signal"
        ),
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
