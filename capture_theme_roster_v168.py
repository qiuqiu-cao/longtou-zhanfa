"""Capture a structured, source-bound market-theme narrative roster.

The capture stores only code/name/theme tags, retrieval metadata and a page
hash.  It does not turn a secondary review page into verified company facts.
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
import requests
from bs4 import BeautifulSoup


THEME_IDS = {
    "PCB板": "pcb_board",
    "国产芯片": "domestic_semiconductor",
    "光通信": "optical_communication",
    "福建自贸-海西概念": "fujian_haixi",
    "智能电网": "smart_grid_power",
    "机器人": "robotics",
    "黄酒": "yellow_rice_wine",
    "被动元件": "passive_components",
    "大农业": "agriculture",
    "房地产": "real_estate",
    "航天": "aerospace",
    "锂电池": "lithium_battery",
    "炭黑": "carbon_black",
    "液冷服务器": "liquid_cooling_server",
    "云计算数据中心": "cloud_data_center",
    "大消费": "consumer",
}
NON_DRIVER_TAGS = {"其他", "业绩增长"}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _theme_id(theme: str) -> str:
    if theme in THEME_IDS:
        return THEME_IDS[theme]
    digest = hashlib.sha256(theme.encode("utf-8")).hexdigest()[:12]
    return f"source_theme_{digest}"


def select_primary_theme(tags: list[str]) -> tuple[str | None, str]:
    candidates = [tag for tag in tags if tag not in NON_DRIVER_TAGS]
    if len(candidates) == 1:
        return candidates[0], "unique_specific_tag_after_generic_filter"
    if not candidates:
        return None, "no_specific_tag_after_generic_filter"
    return None, "multiple_specific_tags_primary_driver_ambiguous"


def parse_page(html: str, pool: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    required = {"code", "name", "pool_assignment", "is_mainboard"}
    if required - set(pool):
        raise ValueError("pool is missing fields needed for source capture")
    eligible = pool.copy()
    eligible["code"] = eligible.code.astype(str).str.zfill(6)
    mainboard = eligible.is_mainboard.astype(str).str.lower().isin({"true", "1"})
    eligible = eligible[eligible.pool_assignment.eq("CLOSING_LIMIT_UP") & mainboard]
    pool_codes = set(eligible.code)
    soup = BeautifulSoup(html, "html.parser")
    observations = []
    seen = set()
    for card in soup.select("div[data-stock-card]"):
        quote = card.find("a", href=re.compile(r"quote\.eastmoney\.com/"))
        if quote is None:
            continue
        match = re.search(r"\b(\d{6})\b", quote.get_text(" ", strip=True))
        if not match:
            continue
        code = match.group(1)
        if code not in pool_codes or code in seen:
            continue
        theme_span = card.select_one("div.line-clamp-2 span[title]")
        if theme_span is None:
            continue
        raw_theme = str(theme_span.get("title", "")).strip()
        tags = [item.strip() for item in raw_theme.split("|") if item.strip()]
        if not tags:
            continue
        seen.add(code)
        pool_row = eligible.loc[eligible.code.eq(code)].iloc[0]
        primary, basis = select_primary_theme(tags)
        observations.append({
            "code": code,
            "name": str(pool_row["name"]),
            "primary_theme": primary,
            "all_source_tags": tags,
            "primary_selection_basis": basis,
            "primary_driver_ambiguous": primary is None,
            "claim_status": "secondary_market_narrative",
        })
    by_theme: dict[str, list[str]] = {}
    for row in observations:
        if row["primary_theme"]:
            by_theme.setdefault(row["primary_theme"], []).append(row["code"])
    groups = [
        {
            "theme_group_id": _theme_id(theme),
            "theme": theme,
            "codes": sorted(codes),
            "claim_status": "secondary_market_narrative",
            "source_tier": "C_PUBLIC_REVIEW_EXPLICIT_ROSTER",
        }
        for theme, codes in sorted(by_theme.items()) if len(codes) >= 2
    ]
    return observations, groups


def capture(url: str, signal_date: str, pool_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable theme source capture")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    raw = response.content
    response.encoding = response.apparent_encoding or response.encoding
    html = response.text
    pool = pd.read_csv(pool_path, dtype={"code": str})
    observations, groups = parse_page(html, pool)
    retrieved = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    source_id = "secondary_review_lwwhy"
    payload = {
        "version": "v168-secondary-theme-source-capture-1",
        "signal_date": pd.Timestamp(signal_date).date().isoformat(),
        "sources": [{
            "source_id": source_id,
            "url": url,
            "retrieved_at": retrieved,
            "available_at": retrieved,
            "published_at": None,
            "timestamp_basis": "retrieval_time_only_publication_time_unavailable",
            "source_tier": "C_PUBLIC_REVIEW_EXPLICIT_ROSTER",
            "claim_status": "secondary_market_narrative",
            "page_sha256": _sha256_bytes(raw),
            "page_bytes": len(raw),
        }],
        "groups": [dict(group, source_id=source_id) for group in groups],
        "observations": observations,
        "unassigned_or_singleton_codes": sorted(
            row["code"] for row in observations
            if not row["primary_theme"] or sum(
                row["primary_theme"] == other["primary_theme"] for other in observations
            ) < 2
        ),
        "governance": {
            "business_fact_verified": False,
            "catalyst_first_public_time_verified": False,
            "positive_model_factor_allowed": False,
            "high_low_mapping_authorized": False,
            "absence_means_negative": False,
            "multi_specific_tag_primary_driver_forced": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "signal_date": payload["signal_date"],
        "observations": len(observations),
        "groups": len(groups),
        "grouped_codes": sum(len(group["codes"]) for group in groups),
        "output": str(output.resolve()),
        "page_sha256": payload["sources"][0]["page_sha256"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(capture(args.url, args.signal_date, args.pool, args.output),
                     ensure_ascii=False, indent=2))
