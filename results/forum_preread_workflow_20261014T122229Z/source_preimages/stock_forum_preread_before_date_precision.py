"""Read checked forum evidence before stock judgement; retain the whole pool."""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import stock_catalyst_context as catalyst
import stock_judgement_review as original
from public_outcome_sources_v293 import next_session


def project(candidates, claims, sources, signal_date):
    expected = original.unique_rows(candidates)
    if len(expected) != len(candidates):
        raise ValueError("duplicate candidate in pre-read pool")
    if any(c["scope"] == "STOCK" and c["code"] not in expected for c in claims):
        raise ValueError("forum evidence outside the complete candidate pool")
    rows = []
    for code, stock in sorted(expected.items()):
        evidence = []
        for claim in claims:
            source = sources[claim["source_id"]]
            if (claim["scope"] != "STOCK" or claim["code"] != code or claim["kind"] != "NARRATIVE"
                    or source["kind"] != "ORIGINAL_FORUM"):
                continue
            section = claim.get("source_section", "UNKNOWN_SECTION")
            if section not in {"AUTHOR_BODY", "COMMENT", "UNKNOWN_SECTION"}:
                raise ValueError("author text and comments must remain separate")
            publication = claim.get("comment_published_at") if section == "COMMENT" else source["published_at"]
            if publication is not None and original.stamp(publication) > original.stamp(source["obtained_at"]):
                raise ValueError("comment publication follows actual source capture")
            publication_day = original.stamp(publication).astimezone(original.SHANGHAI).date().isoformat() if publication else None
            author = claim.get("comment_author", "UNKNOWN_COMMENT_AUTHOR") if section == "COMMENT" else source.get("author", "UNKNOWN")
            evidence.append({"source_id": claim["source_id"], "author": author,
                "url": source["url"], "published_at": source["published_at"],
                "discussion_published_at": publication,
                "obtained_at": source["obtained_at"], "source_section": section,
                "excerpt": claim["excerpt"], "offset": claim["offset"],
                "published_on_signal_day": publication_day == signal_date,
                "use": "SIGNAL_DAY_DISCUSSION" if publication_day == signal_date else "OLDER_DISCUSSION_BACKGROUND" if publication_day else "COMMENT_DATE_UNKNOWN",
                "review_focus": claim.get("review_focus", []), "opinion_is_verified_market_fact": False})
        current = [e for e in evidence if e["published_on_signal_day"]]
        rows.append({"code": code, "name": stock["name"], "boards": stock.get("boards", stock.get("consecutive_boards")),
            "coverage": "SIGNAL_DAY_LIMITED_SOURCE" if current else "BACKGROUND_ONLY" if evidence else "NOT_COVERED_IN_CHECKED_SOURCES",
            "forum_evidence": evidence, "narrative_is_absent": None,
            "unknown_is_negative_evidence": False, "unknown_is_positive_evidence": False,
            "author_review_required": ["实际交易叙事与行业身份", "高位情绪锚及低位补涨连接",
                "完整同位与必要同题材竞争", "支持、反证和作者之间的分歧", "公告/行情待核与取消条件"],
            "promotion_direction_supplied": False, "role_identity_certified": False})
    return rows


def build(pool_path, bank_path, calendar_path, output_dir, exposure):
    moment = original.now()
    pool = original.read(pool_path)
    calendar = original.read(calendar_path)
    if calendar.get("status") != "verified" or next_session(calendar, pool["signal_date"]) != pool["target_date"]:
        raise ValueError("exact official signal/target sessions required")
    if exposure not in {"MODEL_OUTPUT_ALREADY_SEEN", "MODEL_NOT_REVIEWED_SELF_ATTESTED"}:
        raise ValueError("actual model exposure must be explicit")
    candidates = pool.get("candidates", pool.get("rows"))
    bank, sources, claims, proof = catalyst.validate_bank(bank_path, candidates, moment)
    if (bank["signal_date"], bank["target_date"]) != (pool["signal_date"], pool["target_date"]):
        raise ValueError("forum bank and complete pool dates differ")
    rows = project(candidates, claims, sources, pool["signal_date"])
    proof |= original.bind(pool_path) | original.bind(calendar_path) | original.bind(__file__) | original.bind(catalyst.__file__)
    original.check_sources(proof)
    report = {"status": "FORUM_PREREAD_CONTEXT_ONLY" if exposure == "MODEL_NOT_REVIEWED_SELF_ATTESTED" else "FORUM_CONTEXT_AFTER_MODEL_EXPOSURE_NOT_PRIOR_BLIND_REVIEW",
        "formed_at": moment.isoformat(), "signal_date": pool["signal_date"], "target_date": pool["target_date"],
        "model_exposure": exposure, "blinding_verified": False, "pool_size": len(rows), "rows": rows,
        "signal_day_covered_count": sum(r["coverage"] == "SIGNAL_DAY_LIMITED_SOURCE" for r in rows),
        "background_only_count": sum(r["coverage"] == "BACKGROUND_ONLY" for r in rows),
        "uncovered_count": sum(r["coverage"] == "NOT_COVERED_IN_CHECKED_SOURCES" for r in rows),
        "source_bindings": proof, "new_fits": 0, "new_formal_forward_samples": 0,
        "new_scores_or_directions": False, "next_stage": "Actual source/market/mainline and author judgement before model display on a new date"}
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError("new immutable pre-read output required")
    out.mkdir(parents=True)
    original.write_new(out / "report.json", report)
    lines = ["# 逐股题材预读卡", "", f"信号{pool['signal_date']}，目标{pool['target_date']}；形成{moment.isoformat()}。",
        "本文仅提供已核论坛讨论线索，旧文、评论和当前正文分别标识；未覆盖不代表没有题材，也不自动转为晋级。", "",
        "| 股票 | 讨论覆盖 | 已核信息和待核关系 |", "|---|---|---|"]
    for row in rows:
        texts = "；".join(f"{e['author']}（{e['source_section']}，{e['use']}）：{e['excerpt']}" for e in row["forum_evidence"]) or "限定已读来源未覆盖；题材保持未知"
        lines.append(f"| {row['name']} {row['code']} | {row['coverage']} | {texts.replace(chr(10), ' ').replace('|', '/')} |")
    (out / "逐股题材预读卡.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("pool", "bank", "calendar", "output-dir", "model-exposure"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    r = build(a.pool, a.bank, a.calendar, a.output_dir, a.model_exposure)
    import json
    print(json.dumps({k: r[k] for k in ("status", "pool_size", "signal_day_covered_count", "background_only_count", "uncovered_count")}, ensure_ascii=False))
