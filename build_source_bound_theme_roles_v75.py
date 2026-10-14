"""Build point-in-time theme and competition evidence from reviewed source rosters.

The input JSON is deliberately human-reviewed: a security is admitted only when
the cited page explicitly puts that code in the stated specific theme.  Missing
securities remain unavailable rather than being treated as negative evidence.
"""
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
    "evidence_status", "identity_tags", "capacity_rank_in_theme",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def validate_evidence(evidence: dict, asof: str) -> None:
    required = {"signal_date", "groups"}
    missing = required - set(evidence)
    if missing:
        raise ValueError(f"evidence missing fields: {sorted(missing)}")
    cutoff = _parse_time(asof)
    seen_codes: set[str] = set()
    for group in evidence["groups"]:
        fields = {"theme_group_id", "theme", "codes", "url", "published_at", "source_tier"}
        absent = fields - set(group)
        if absent:
            raise ValueError(f"theme group missing fields: {sorted(absent)}")
        if len(group["codes"]) < 2:
            raise ValueError("a verified theme group requires at least two explicitly named codes")
        published = _parse_time(group["published_at"])
        if published > cutoff:
            raise ValueError(f"future evidence relative to asof: {group['url']}")
        for raw_code in group["codes"]:
            code = str(raw_code).zfill(6)
            if code in seen_codes:
                raise ValueError(f"code assigned to multiple primary themes: {code}")
            seen_codes.add(code)


def build(pool: pd.DataFrame, evidence: dict, asof: str) -> pd.DataFrame:
    validate_evidence(evidence, asof)
    closing = pool.loc[pool.pool_assignment.eq("CLOSING_LIMIT_UP")].copy()
    closing["code"] = closing.code.astype(str).str.zfill(6)
    closing = closing.drop_duplicates("code").set_index("code")
    rows: list[dict] = []
    for group in evidence["groups"]:
        explicit = [str(code).zfill(6) for code in group["codes"]]
        present = [code for code in explicit if code in closing.index]
        if len(present) < 2:
            continue
        levels = {code: int(closing.loc[code, "boards"]) for code in present}
        amounts = {code: float(closing.loc[code, "amount"]) for code in present}
        max_level = max(levels.values())
        height_codes = [code for code, level in levels.items() if level == max_level]
        amount_order = sorted(present, key=lambda code: (-amounts[code], code))
        distinct_levels = sorted(set(levels.values()))
        ladder = any(b == a + 1 for a, b in zip(distinct_levels, distinct_levels[1:]))
        for code in present:
            level = levels[code]
            if level == max_level and len(height_codes) == 1:
                role = "THEME_HEIGHT_LEADER"
            elif level == max_level:
                role = "SAME_HEIGHT_CORE"
            else:
                role = "THEME_SUPPORT"
            tags = [role]
            if code == amount_order[0]:
                tags.append("CAPACITY_CORE_BY_CLOSE_AMOUNT")
            if closing.loc[code, "first_seal_time"] == min(
                    closing.loc[member, "first_seal_time"] for member in present):
                tags.append("FIRST_SEAL_PIONEER")
            rows.append({
                "signal_date": evidence["signal_date"],
                "code": code,
                "theme_status": "verified_explicit_source_roster",
                "theme_role": role,
                "theme_group_id": group["theme_group_id"],
                "theme": group["theme"],
                "primary_evidence_id": group["url"],
                "source_tier": group["source_tier"],
                "theme_support_ex_self": len(present) - 1,
                "same_height_rivals": sum(v == level for v in levels.values()) - 1,
                "theme_ladder_verified": ladder,
                "evidence_published_at": group["published_at"],
                "evidence_status": "verified_page_text_explicitly_names_security",
                "identity_tags": "|".join(tags),
                "capacity_rank_in_theme": amount_order.index(code) + 1,
            })
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS).sort_values(
        ["theme_group_id", "capacity_rank_in_theme", "code"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pool_path, evidence_path, output = Path(args.pool), Path(args.evidence), Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable evidence directory")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    frame = build(pd.read_csv(pool_path, dtype={"code": str}), evidence, args.asof)
    output.mkdir(parents=True)
    csv_path = output / "candidate_theme_roles.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    manifest = {
        "version": "v75-source-bound-theme-roles-1",
        "signal_date": evidence["signal_date"],
        "asof": args.asof,
        "status": "PARTIAL_VERIFIED_EXPLICIT_SOURCE_ROSTER",
        "scope_warning": "absence is unavailable, not negative; roles use only closing-pool structure",
        "candidate_rows": len(frame),
        "theme_groups": int(frame.theme_group_id.nunique()) if len(frame) else 0,
        "closing_pool_rows": int((pd.read_csv(pool_path).pool_assignment == "CLOSING_LIMIT_UP").sum()),
        "source_pool": str(pool_path.resolve()),
        "source_pool_sha256": _sha256(pool_path),
        "evidence_file": str(evidence_path.resolve()),
        "evidence_sha256": _sha256(evidence_path),
        "output_sha256": _sha256(csv_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
