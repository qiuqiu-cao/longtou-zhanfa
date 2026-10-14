"""Classify archived announcement text without leaking late evidence backwards.

V228 consumes a V227 immutable archive, verifies every recorded hash, assigns a
transparent content-risk class, and records whether the text was actually
available by the requested decision cutoff.  A public document mirror can
support soft research classification, but it cannot create an authenticated
hard veto or authorise a trade by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


VERSION = "v228-announcement-fulltext-review-1"
TZ = ZoneInfo("Asia/Shanghai")

SUBSTANTIVE_HARD_PATTERNS = (
    r"(?:被|遭|收到).{0,8}立案", r"立案调查", r"行政处罚(?:决定|事先告知)",
    r"退市风险警示", r"可能被终止上市", r"股票终止上市", r"债务逾期",
    r"资金占用", r"违规担保", r"控制权(?:存在)?不稳定", r"破产清算",
)
HOT_THEME_NEGATIVE_PATTERNS = (
    r"未开展.{0,20}业务", r"不涉及.{0,20}(?:概念|业务)",
    r"与公司主营业务无关", r"不存在.{0,20}相关业务",
    r"(?:相关)?产品.{0,24}(?:均为|属于).{0,12}(?:代理|经销)产品",
    r"(?:非|不是).{0,12}(?:公司)?(?:自研|自主研发|自产)(?:产品)?",
    r"不进行.{0,20}(?:生产|制造)",
)
MARKET_RISK_PATTERNS = (
    r"股票交易风险提示", r"股价波动", r"累计涨幅", r"市盈率.{0,20}高于",
    r"估值.{0,12}(?:较高|偏高)", r"换手率较高", r"非理性炒作",
)
PERFORMANCE_RISK_PATTERNS = (
    r"净利润.{0,48}同比下降", r"归属于上市公司股东的净利润.{0,48}下降",
    r"营业收入.{0,48}同比下降", r"经营业绩.{0,24}(?:下降|下滑)",
)
MATERIAL_PERFORMANCE_PATTERNS = (
    r"预计.{0,36}(?:减少|影响).{0,24}(?:净利润|归属于上市公司股东的净利润)",
    r"(?:净利润|归属于上市公司股东的净利润).{0,36}(?:减少|损失).{0,16}(?:亿元|万元)",
)
AUTHENTICATED_TIERS = {
    "EXCHANGE_AUTHENTICATED", "MANUALLY_VERIFIED_EXCHANGE_DOCUMENT",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=TZ)
    return parsed.astimezone(TZ)


def classify_text(text: str) -> dict:
    text = re.sub(r"\s+", "", str(text or ""))
    hard_hits = [pattern for pattern in SUBSTANTIVE_HARD_PATTERNS if re.search(pattern, text)]
    theme_hits = [pattern for pattern in HOT_THEME_NEGATIVE_PATTERNS if re.search(pattern, text)]
    market_hits = [pattern for pattern in MARKET_RISK_PATTERNS if re.search(pattern, text)]
    performance_hits = [pattern for pattern in PERFORMANCE_RISK_PATTERNS if re.search(pattern, text)]
    material_performance_hits = [
        pattern for pattern in MATERIAL_PERFORMANCE_PATTERNS if re.search(pattern, text)
    ]
    if hard_hits:
        classification, score = "SUBSTANTIVE_HARD_RISK", -20.0
    elif material_performance_hits:
        classification, score = "SUBSTANTIVE_HARD_RISK", -20.0
    elif theme_hits:
        classification, score = "HOT_THEME_CLARIFICATION_NEGATIVE", -8.0
    elif performance_hits:
        classification, score = "ROUTINE_MARKET_AND_PERFORMANCE_RISK", -4.0
    elif market_hits:
        classification, score = "ROUTINE_MARKET_RISK", -2.0
    else:
        classification, score = "NO_CLASSIFIED_SUBSTANTIVE_RISK", 0.0
    return {
        "content_risk_class": classification,
        "research_score": score,
        "hard_pattern_hits": "|".join(hard_hits),
        "hot_theme_pattern_hits": "|".join(theme_hits),
        "market_pattern_hits": "|".join(market_hits),
        "performance_pattern_hits": "|".join(performance_hits),
        "material_performance_pattern_hits": "|".join(material_performance_hits),
        "theme_thesis_veto": bool(theme_hits),
    }


def review(archive_dir: str, decision_cutoff: str, output_dir: str) -> dict:
    archive, target = Path(archive_dir), Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V228 review")
    evidence_path, archive_report_path = archive / "fulltext_evidence.csv", archive / "report.json"
    archive_report = json.loads(archive_report_path.read_text(encoding="utf-8-sig"))
    if archive_report.get("evidence_sha256") != _sha(evidence_path):
        raise ValueError("V227 evidence CSV hash mismatch")
    cutoff = _parse_time(decision_cutoff)
    evidence = pd.read_csv(evidence_path, dtype={"code": str})
    rows = []
    for source in evidence.to_dict("records"):
        item = {
            "code": str(source.get("code", "")).zfill(6),
            "name": source.get("name"), "notice_id": source.get("notice_id"),
            "title": source.get("title"), "published_at": source.get("published_at"),
            "document_source_tier": source.get("document_source_tier"),
            "archive_full_text_status": source.get("full_text_status"),
            "full_text_available_at": source.get("full_text_available_at"),
            "decision_cutoff": cutoff.isoformat(), "entry_authorized": False,
        }
        text = ""
        if str(source.get("full_text_status", "")).startswith("MIRROR_PDF"):
            pdf_path = Path(str(source.get("pdf_path")))
            if not pdf_path.exists() or _sha(pdf_path) != source.get("pdf_sha256"):
                raise ValueError(f"PDF hash mismatch for {source.get('notice_id')}")
        if source.get("full_text_status") == "MIRROR_PDF_TEXT_RECEIVED":
            text_path = Path(str(source.get("text_path")))
            if not text_path.exists() or _sha(text_path) != source.get("text_sha256"):
                raise ValueError(f"text hash mismatch for {source.get('notice_id')}")
            text = text_path.read_text(encoding="utf-8", errors="strict")
            classified = classify_text(text)
            available = _parse_time(str(source.get("full_text_available_at")))
            point_in_time = available <= cutoff
            authenticated = source.get("document_source_tier") in AUTHENTICATED_TIERS
            item.update(classified)
            item.update({
                "fulltext_review_status": (
                    "POINT_IN_TIME_REVIEWED" if point_in_time
                    else "RETROSPECTIVE_REVIEW_NOT_POINT_IN_TIME"
                ),
                "point_in_time_eligible": point_in_time,
                "fulltext_risk_score": classified["research_score"] if point_in_time else 0.0,
                "fulltext_hard_veto": bool(
                    point_in_time and authenticated
                    and classified["content_risk_class"] == "SUBSTANTIVE_HARD_RISK"
                ),
                "source_authenticated_for_hard_veto": authenticated,
                "text_sha256": source.get("text_sha256"),
            })
        else:
            item.update({
                "content_risk_class": "UNAVAILABLE", "research_score": 0.0,
                "hard_pattern_hits": "", "hot_theme_pattern_hits": "",
                "market_pattern_hits": "", "performance_pattern_hits": "",
                "material_performance_pattern_hits": "", "theme_thesis_veto": False,
                "fulltext_review_status": "UNAVAILABLE", "point_in_time_eligible": False,
                "fulltext_risk_score": 0.0, "fulltext_hard_veto": False,
                "source_authenticated_for_hard_veto": False, "text_sha256": "",
            })
        rows.append(item)
    reviewed = pd.DataFrame(rows)
    target.mkdir(parents=True)
    review_path = target / "fulltext_review.csv"
    reviewed.to_csv(review_path, index=False, encoding="utf-8-sig")
    announcement_hashes = archive_report.get("source_hashes", {})
    announcement_index_hash = next(
        (value for key, value in announcement_hashes.items() if key.lower().endswith("announcements.csv")),
        None,
    )
    report = {
        "version": VERSION, "status": "REVIEWED_NOT_TRADE_CLEARANCE",
        "signal_date": archive_report.get("signal_date"),
        "decision_cutoff": cutoff.isoformat(), "reviewed_rows": int(len(reviewed)),
        "point_in_time_eligible_rows": int(reviewed.point_in_time_eligible.sum()) if len(reviewed) else 0,
        "retrospective_rows": int(reviewed.fulltext_review_status.eq(
            "RETROSPECTIVE_REVIEW_NOT_POINT_IN_TIME").sum()) if len(reviewed) else 0,
        "unavailable_rows": int(reviewed.fulltext_review_status.eq("UNAVAILABLE").sum()) if len(reviewed) else 0,
        "announcement_index_sha256": announcement_index_hash,
        "source_hashes": {str(evidence_path.resolve()): _sha(evidence_path),
                          str(archive_report_path.resolve()): _sha(archive_report_path)},
        "review_sha256": _sha(review_path),
        "policy": {
            "late_evidence_backdated": False,
            "mirror_alone_can_authorize_hard_veto": False,
            "entry_authorized": False,
        },
        "entry_authorized": False,
    }
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--decision-cutoff", required=True)
    ap.add_argument("--output", required=True)
    ns = ap.parse_args()
    print(json.dumps(review(ns.archive, ns.decision_cutoff, ns.output),
                     ensure_ascii=False, indent=2))
