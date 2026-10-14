"""Apply an immutable, source-bound full-text clearance overlay to 2026-09-08 notices."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REVIEWS = [
    {
        "code": "000912", "contains": "减持股份计划实施期限届满",
        "decision": "CLOSED_COMPLETED", "evidence": "https://www.futunn.com/stock/000912-SZ/news",
        "reason": "full-text mirror states the plan was completed and the holding fell from 14.92% to 14.00%",
    },
    {
        "code": "002264", "contains": "高级管理人员减持股份实施情况",
        "decision": "CLOSED_COMPLETED", "evidence": "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12585797&stockid=002264",
        "reason": "full text shows the planned 25,000 shares were sold on 2026-09-03",
    },
    {
        "code": "603077", "contains": "减持已回购股份达1%暨减持结果",
        "decision": "CLOSED_RESULT", "evidence": "https://news.futunn.com/post/78513236",
        "reason": "the notice is explicitly the result of the repurchased-share sale",
    },
    {
        "code": "603077", "contains": "减持已回购股份的进展公告",
        "decision": "SUPERSEDED_BY_LATER_RESULT", "evidence": "https://news.futunn.com/post/78513236",
        "reason": "the 2026-08-04 progress notice is superseded by the 2026-08-29 result for the same repurchased-share sale",
    },
]


def apply(frame: pd.DataFrame):
    result = frame.copy()
    result["risk_hard_original"] = result.risk_hard
    result["fulltext_review_status"] = "NOT_REVIEWED"
    result["review_evidence_url"] = ""
    result["review_reason"] = ""
    ledger = []
    for item in REVIEWS:
        mask = (result.code.astype(str).str.zfill(6).eq(item["code"])
                & result.title.astype(str).str.contains(item["contains"], regex=False))
        if int(mask.sum()) != 1:
            raise ValueError(f"review target must match exactly once: {item['code']} {item['contains']}")
        index = result.index[mask][0]
        before = int(result.at[index, "risk_hard"])
        result.at[index, "risk_hard"] = 0
        result.at[index, "risk_type"] = "historical_corporate_action_closed"
        result.at[index, "risk_window_status"] = "fulltext_closed_or_superseded"
        result.at[index, "fulltext_review_status"] = item["decision"]
        result.at[index, "review_evidence_url"] = item["evidence"]
        result.at[index, "review_reason"] = item["reason"]
        ledger.append({**item, "title": result.at[index, "title"],
                       "risk_hard_before": before, "risk_hard_after": 0})
    return result, ledger


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, target = Path(args.source), Path(args.output)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable full-text overlay")
    target.mkdir(parents=True)
    result, ledger = apply(pd.read_csv(source, dtype={"code": str}))
    out = target / "announcements_fulltext_overlay.csv"
    result.to_csv(out, index=False, encoding="utf-8-sig")
    manifest = {
        "source": str(source.resolve()), "source_sha256": sha(source),
        "output_sha256": sha(out), "reviewed_rows": len(ledger), "ledger": ledger,
        "scope": "manual source-bound clearance; unreviewed rows retain title/index classification",
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
