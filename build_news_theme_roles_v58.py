"""Build a source-bound, same-day theme roster for the 2026-09-08 review.

Only securities explicitly named by the cited same-day reports are admitted.
This is a reviewed evidence overlay, not a historical point-in-time database and
not a model probability.  Static provider industries are deliberately ignored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


GROUPS = {
    "agriculture.planting": {
        "theme": "种植业/粮食",
        "codes": ["600108", "600354", "601952"],
        "url": "https://www.nbd.com.cn/articles/2026-09-08/4575767.html",
        "published_at": "2026-09-08T16:10:29+08:00",
    },
    "agriculture.fertilizer": {
        "theme": "农化/化肥",
        "codes": ["000912", "002274", "002312", "002470", "002999",
                  "600227", "600596", "600691", "603077"],
        "url": "https://finance.jrj.com.cn/2026/09/08123058382356.shtml",
        "published_at": "2026-09-08T12:30:07+08:00",
    },
    "agriculture.processing": {
        "theme": "农产品加工/糖",
        "codes": ["000930", "002286", "600737"],
        "url": "https://finance.jrj.com.cn/2026/09/08123058382356.shtml",
        "published_at": "2026-09-08T12:30:07+08:00",
    },
    "media.publishing": {
        "theme": "文化传媒/出版",
        "codes": ["002343", "600880", "601595", "601949", "603999"],
        "url": "https://m.jrj.com.cn/madapter/stock/2026/09/08180658388633.shtml",
        "published_at": "2026-09-08T18:06:47+08:00",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(pool: pd.DataFrame) -> pd.DataFrame:
    closing = pool[pool.pool_assignment.eq("CLOSING_LIMIT_UP")].copy()
    closing["code"] = closing.code.astype(str).str.zfill(6)
    by_code = closing.set_index("code")
    rows = []
    for group_id, spec in GROUPS.items():
        present = [code for code in spec["codes"] if code in by_code.index]
        if len(present) < 2:
            continue
        levels = {code: int(by_code.loc[code, "boards"]) for code in present}
        high = max(levels.values())
        high_count = sum(level == high for level in levels.values())
        distinct = sorted(set(levels.values()))
        ladder = any(b == a + 1 for a, b in zip(distinct, distinct[1:]))
        for code in present:
            level = levels[code]
            if level == high and high_count == 1:
                role = "THEME_HEIGHT_LEADER"
            elif level == high:
                role = "SAME_HEIGHT_CORE"
            else:
                role = "THEME_SUPPORT"
            rows.append({
                "signal_date": "2026-09-08",
                "code": code,
                "theme_status": "verified_same_day_group",
                "theme_role": role,
                "theme_group_id": group_id,
                "theme": spec["theme"],
                "primary_evidence_id": spec["url"],
                "source_tier": "B_FINANCIAL_NEWS_EXPLICIT_ROSTER",
                "theme_support_ex_self": len(present) - 1,
                "same_height_rivals": high_count - 1 if level == high else
                                       sum(other == level for other in levels.values()) - 1,
                "theme_ladder_verified": ladder,
                "evidence_published_at": spec["published_at"],
                "evidence_status": "verified_page_text_explicitly_names_security",
            })
    columns = [
        "signal_date", "code", "theme_status", "theme_role", "theme_group_id", "theme",
        "primary_evidence_id", "source_tier", "theme_support_ex_self",
        "same_height_rivals", "theme_ladder_verified", "evidence_published_at",
        "evidence_status",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["theme_group_id", "code"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.pool)
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable evidence directory")
    target.mkdir(parents=True)
    frame = build(pd.read_csv(source, dtype={"code": str}))
    out = target / "candidate_theme_roles.csv"
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    manifest = {
        "signal_date": "2026-09-08",
        "status": "PARTIAL_VERIFIED_SAME_DAY_NEWS_ROSTER",
        "scope_warning": "only explicitly named securities; absence is unavailable, not negative",
        "candidate_rows": len(frame),
        "theme_groups": frame.theme_group_id.nunique(),
        "source_pool": str(source.resolve()),
        "source_pool_sha256": sha256(source),
        "output_sha256": sha256(out),
        "groups": GROUPS,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
