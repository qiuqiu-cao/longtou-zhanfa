"""Recover candidate driver claims from a hash-matched secondary review page.

V168 preserved the page URL, retrieval time and SHA-256 but only retained theme
tags.  V231 re-fetches the exact same bytes, refuses any hash mismatch, archives
the page, and extracts the per-stock explanatory text.  These are secondary
market narratives: they may improve auditability and reveal negative
clarifications, but they do not prove company facts, a common theme catalyst,
the catalyst's first-public time, expectation gap or a buy signal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


VERSION = "v231-secondary-driver-claim-archive-1"

CATEGORY_PATTERNS = {
    "NEGATIVE_OR_LIMITED_EXPOSURE": (
        r"不生产", r"不涉及", r"未开展", r"尚未", r"少量", r"小量验证",
        r"未形成.{0,8}(?:收入|营收|利润)", r"与公司.{0,12}无关",
    ),
    "CONTROL_OR_CAPITAL_ACTION": (
        r"(?:控股股东|实际控制人|实控人).{0,24}(?:拟|将|变更|转让|增持|减持|收购)",
        r"股权转让", r"协议转让", r"资产重组", r"收购",
        r"回购", r"增持", r"减持", r"定增", r"控制权",
    ),
    "PERFORMANCE_CLAIM": (
        r"预计.{0,24}(?:净利润|业绩)", r"净利润.{0,24}(?:增长|下降)",
        r"业绩预增", r"扭亏", r"同比增长", r"同比下降",
    ),
    "ORDER_OR_PROJECT": (r"中标", r"订单", r"签署.{0,12}合同", r"项目落地"),
    "POLICY_OR_LIST_INCLUSION": (
        r"纳入.{0,24}(?:指南|目录|名单)", r"入选.{0,24}(?:目录|名单)",
        r"获批", r"政策", r"国家.{0,16}(?:发布|印发)",
    ),
    "PRODUCT_OR_BUSINESS_DESCRIPTION": (
        r"公司是", r"主营", r"提供", r"研发", r"生产", r"销售", r"布局",
        r"已融合", r"拥有", r"可适配", r"主要产品",
    ),
}

RECENCY_PATTERNS = (
    r"近期", r"拟", r"预计", r"公告", r"中标", r"签署", r"回购", r"增持",
    r"变更", r"纳入", r"入选", r"获批", r"今年", r"上半年", r"本次",
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_claim(text: str) -> dict:
    normalized = re.sub(r"\s+", "", str(text or ""))
    hits = {
        category: [pattern for pattern in patterns if re.search(pattern, normalized)]
        for category, patterns in CATEGORY_PATTERNS.items()
    }
    categories = [category for category, values in hits.items() if values]
    if "NEGATIVE_OR_LIMITED_EXPOSURE" in categories:
        primary = "NEGATIVE_OR_LIMITED_EXPOSURE"
    elif categories:
        primary = categories[0]
    else:
        primary = "UNCLASSIFIED_SECONDARY_CLAIM"
    recency_hits = [pattern for pattern in RECENCY_PATTERNS if re.search(pattern, normalized)]
    return {
        "driver_claim_category": primary,
        "all_driver_claim_categories": "|".join(categories),
        "negative_or_limited_exposure": "NEGATIVE_OR_LIMITED_EXPOSURE" in categories,
        "recency_language_status": (
            "RECENCY_LANGUAGE_PRESENT_NOT_FIRST_TIME_VERIFIED"
            if recency_hits else "STATIC_OR_UNDATED_DESCRIPTION"),
        "classification_pattern_hits": "|".join(
            pattern for values in hits.values() for pattern in values),
        "recency_pattern_hits": "|".join(recency_hits),
    }


def parse_page(content: bytes, signal_date: str, source: dict) -> pd.DataFrame:
    soup = BeautifulSoup(content, "html.parser")
    rows = []
    for card in soup.select("[data-stock-card]"):
        code = None
        for anchor in card.find_all("a"):
            match = re.fullmatch(r"\d{6}", anchor.get_text(strip=True))
            if match:
                code = match.group(0)
                break
        if not code:
            continue
        name_link = next((
            anchor for anchor in card.find_all("a", href=re.compile(r"/stock/detail/"))
            if anchor.get_text(strip=True)
            and not re.fullmatch(r"榜\d+", anchor.get_text(strip=True))
        ), None)
        name = name_link.get_text(strip=True) if name_link else ""
        description = card.select_one("div.text-sm.line-clamp-2")
        tags, claim = [], ""
        if description:
            for tag in description.select("span.font-bold[title]"):
                tags.extend(value.strip() for value in tag.get("title", "").split("|") if value.strip())
            claim_spans = [span for span in description.select("span.text-secondary[title]")
                           if "font-bold" not in (span.get("class") or [])]
            claim = " ".join(span.get("title", "").strip() for span in claim_spans if span.get("title"))
        classified = classify_claim(claim)
        card_text = card.get_text(" ", strip=True)
        rows.append({
            "signal_date": signal_date, "code": code, "name": name,
            "theme_tags": "|".join(dict.fromkeys(tags)),
            "driver_claim": claim,
            "driver_claim_status": (
                "OBSERVED_SECONDARY_DRIVER_CLAIM" if claim else "UNAVAILABLE"),
            "membership_fact_status": "UNVERIFIED_SECONDARY_NARRATIVE",
            "catalyst_first_public_time_status": "UNAVAILABLE",
            "source_id": source.get("source_id"), "source_url": source.get("url"),
            "source_tier": source.get("source_tier"),
            "source_claim_status": source.get("claim_status"),
            "source_available_at": source.get("available_at"),
            "source_published_at": source.get("published_at"),
            "source_page_sha256": source.get("page_sha256"),
            "card_text_sha256": _sha_bytes(card_text.encode("utf-8")),
            "positive_buy_reason_allowed": False,
            "entry_authorized": False,
            **classified,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no stock cards parsed from hash-matched page")
    if frame.code.duplicated().any():
        merged = []
        for code, part in frame.groupby("code", sort=False):
            if part.name.nunique(dropna=False) != 1 or part.driver_claim.nunique(dropna=False) != 1:
                raise ValueError(f"conflicting duplicated stock cards: {code}")
            item = part.iloc[0].to_dict()
            tags = []
            for value in part.theme_tags.fillna(""):
                tags.extend(tag for tag in str(value).split("|") if tag)
            item["theme_tags"] = "|".join(dict.fromkeys(tags))
            item["card_occurrence_count"] = int(len(part))
            item["card_text_sha256"] = "|".join(sorted(set(part.card_text_sha256)))
            merged.append(item)
        frame = pd.DataFrame(merged)
    else:
        frame["card_occurrence_count"] = 1
    return frame


def theme_summary(claims: pd.DataFrame, source_doc: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_by_id = {row["source_id"]: row for row in source_doc.get("sources", [])}
    summary_rows, overlay_rows = [], []
    for group in source_doc.get("groups", []):
        codes = {str(code).zfill(6) for code in group.get("codes", [])}
        part = claims[claims.code.isin(codes)]
        source = source_by_id.get(group.get("source_id"), {})
        claim_part = part[part.driver_claim_status.eq("OBSERVED_SECONDARY_DRIVER_CLAIM")]
        normalized = claim_part.driver_claim.fillna("").map(lambda value: re.sub(r"\s+", "", value))
        counts = normalized.value_counts()
        repeated = int(counts.iloc[0]) if len(counts) else 0
        repeated_share = repeated / len(claim_part) if len(claim_part) else 0.0
        common = (
            "REPEATED_SECONDARY_DRIVER_CLAIM"
            if repeated >= 2 and repeated_share >= .5
            else "INDIVIDUAL_DRIVER_CLAIMS_COMMON_CATALYST_NOT_DEMONSTRATED"
        )
        categories = claim_part.driver_claim_category.value_counts().to_dict()
        negatives = int(claim_part.negative_or_limited_exposure.map(bool).sum())
        fresh = int(claim_part.recency_language_status.eq(
            "RECENCY_LANGUAGE_PRESENT_NOT_FIRST_TIME_VERIFIED").sum())
        summary_rows.append({
            "signal_date": source_doc.get("signal_date"),
            "theme_group_id": group.get("theme_group_id"), "theme_name": group.get("theme"),
            "roster_member_count": len(codes), "parsed_member_count": int(len(part)),
            "driver_claim_count": int(len(claim_part)), "unique_driver_claim_count": int(len(counts)),
            "recency_language_claim_count": fresh,
            "negative_or_limited_exposure_count": negatives,
            "driver_category_counts_json": json.dumps(categories, ensure_ascii=False, sort_keys=True),
            "common_driver_status": common,
            "common_driver_verified": False,
            "catalyst_first_public_time_verified": False,
            "expectation_status": "UNAVAILABLE",
            "entry_authorized": False,
        })
        overlay_rows.append({
            "signal_date": source_doc.get("signal_date"), "theme_name": group.get("theme"),
            "membership_status": "OBSERVED_SECONDARY_ROSTER",
            "membership_source_url": source.get("url", ""),
            "membership_available_at": source.get("available_at", ""),
            "catalyst_status": common,
            "catalyst_source_url": source.get("url", ""),
            "catalyst_published_at": "",
            "catalyst_available_at": source.get("available_at", ""),
            "expectation_status": "UNAVAILABLE",
            "notes": (
                f"secondary driver claims={len(claim_part)}; unique={len(counts)}; "
                f"recency language={fresh}; negative/limited exposure={negatives}; "
                "page publication time and catalyst first-public time unavailable"
            ),
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(overlay_rows)


def collect(source_path: str, output_dir: str, timeout: int = 30,
            archived_pages: list[str] | None = None) -> dict:
    source_file, target = Path(source_path), Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V231 archive")
    source_doc = json.loads(source_file.read_text(encoding="utf-8-sig"))
    sources = source_doc.get("sources", [])
    if not sources:
        raise ValueError("secondary source document contains no sources")
    if archived_pages is not None and len(archived_pages) != len(sources):
        raise ValueError("archived page count must equal source count")
    target.mkdir(parents=True)
    all_claims, page_records = [], []
    for index, source in enumerate(sources, 1):
        supplied_archive = (Path(archived_pages[index - 1])
                            if archived_pages is not None else None)
        if supplied_archive is not None:
            content = supplied_archive.read_bytes()
        else:
            response = requests.get(source["url"], timeout=timeout)
            response.raise_for_status()
            content = response.content
        actual = _sha_bytes(content)
        if actual != source.get("page_sha256"):
            raise ValueError(f"page hash mismatch for {source.get('source_id')}")
        html_path = target / f"source_{index}_{source.get('source_id', 'page')}.html"
        html_path.write_bytes(content)
        claims = parse_page(content, str(source_doc.get("signal_date")), source)
        all_claims.append(claims)
        page_records.append({
            "source_id": source.get("source_id"), "url": source.get("url"),
            "archived_path": str(html_path.resolve()), "sha256": _sha(html_path),
            "hash_matches_prior_archive": True,
            "prior_retrieved_at": source.get("retrieved_at"),
            "prior_available_at": source.get("available_at"),
            "recovered_from_prior_exact_archive": supplied_archive is not None,
            "publication_time_status": (
                "AVAILABLE" if source.get("published_at") else "UNAVAILABLE"),
        })
    claims = pd.concat(all_claims, ignore_index=True)
    groups, overlay = theme_summary(claims, source_doc)
    claims_path = target / "candidate_driver_claims.csv"
    groups_path = target / "theme_driver_summary.csv"
    overlay_path = target / "theme_catalyst_overlay_v222.csv"
    claims.to_csv(claims_path, index=False, encoding="utf-8-sig")
    groups.to_csv(groups_path, index=False, encoding="utf-8-sig")
    overlay.to_csv(overlay_path, index=False, encoding="utf-8-sig")
    report = {
        "version": VERSION, "status": "SECONDARY_CLAIMS_ARCHIVED_NOT_VERIFIED_CATALYST",
        "signal_date": source_doc.get("signal_date"), "source_pages": len(page_records),
        "hash_matched_pages": sum(row["hash_matches_prior_archive"] for row in page_records),
        "candidate_cards": int(len(claims)),
        "candidate_driver_claims": int(claims.driver_claim_status.eq(
            "OBSERVED_SECONDARY_DRIVER_CLAIM").sum()),
        "negative_or_limited_exposure_claims": int(claims.negative_or_limited_exposure.sum()),
        "theme_groups": int(len(groups)),
        "verified_common_catalysts": 0,
        "verified_catalyst_first_public_times": 0,
        "page_records": page_records,
        "source_hashes": {str(source_file.resolve()): _sha(source_file)},
        "artifact_hashes": {claims_path.name: _sha(claims_path),
                            groups_path.name: _sha(groups_path),
                            overlay_path.name: _sha(overlay_path)},
        "governance": {
            "secondary_claim_is_company_fact": False,
            "secondary_claim_is_verified_active_catalyst": False,
            "retrieval_time_substitutes_first_public_time": False,
            "positive_buy_reason_allowed": False, "entry_authorized": False,
        },
        "entry_authorized": False,
    }
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--archived-page", action="append",
                    help="existing exact archived page; repeat in source order")
    ns = ap.parse_args()
    print(json.dumps(collect(ns.source, ns.output, ns.timeout, ns.archived_page),
                     ensure_ascii=False, indent=2))
