import datetime as dt
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[4]
OUT = pathlib.Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def binding(path):
    path = (ROOT / path).resolve()
    return {"path": str(path), "sha256": sha(path)}


BANK_PATH = "results/forum_preread_workflow_20261014T122229Z/checked_forum_bank_qualified.json"
POOL_PATH = "results/sunshine_jiuyang_relation_20261014T152814Z/15日完整15只角色补充输出.json"
DAILY_PATH = "results/daily_review_20261014T1118Z/focused_daily/normalized_daily.json"
RELATION_PATH = "results/deep_review_20261014T115540Z/jiuyang_baiyang_relation_correction_20261014T121615Z/relation_review.json"
SUN_QUAL_PATH = "results/sunshine_jiuyang_relation_20261014T152814Z/source_qualification.json"
SUN_WEB_PATH = "results/sunshine_jiuyang_relation_20261014T152814Z/web_selected_sources_tool_response.json"
SUN_RECEIPT_PATH = "results/sunshine_jiuyang_relation_20261014T152814Z/completion_receipt.json"
JUNYAO_NAME_PATH = "results/v293/daily_lifecycle_20261002_v1/2026-09-30_2c74dc406eef7dcf.md"
bank, pool, daily, relation = [read(x) for x in (BANK_PATH, POOL_PATH, DAILY_PATH, RELATION_PATH)]
sun_qual, sun_text, sun_receipt = [read(x) for x in (SUN_QUAL_PATH, SUN_WEB_PATH, SUN_RECEIPT_PATH)]
assert len(pool["rows"]) == 15 and len({r["code"] for r in pool["rows"]}) == 15
bindings = [binding(x) for x in (BANK_PATH, POOL_PATH, DAILY_PATH, RELATION_PATH, SUN_QUAL_PATH, SUN_WEB_PATH, SUN_RECEIPT_PATH)]
name_line = (ROOT / JUNYAO_NAME_PATH).read_text(encoding="utf-8-sig").splitlines()[64]
assert "605388 均瑶健康" in name_line
bindings.append(binding(JUNYAO_NAME_PATH))
sources = []
texts = {}
for source in bank["sources"]:
    text_path = pathlib.Path(source["text_path"])
    assert sha(text_path) == source["text_sha256"]
    assert sha(source["receipt_path"]) == source["receipt_sha256"]
    text = text_path.read_text(encoding="utf-8-sig")
    texts[source["source_id"]] = text
    sources.append({**source, "source_content_scope": "CHECKED_OCT12_AUTHOR_BODY_AND_SEPARATE_COMMENT", "body_available": True})
    bindings.append({"path": str(text_path), "sha256": sha(text_path)})
    bindings.append({"path": source["receipt_path"], "sha256": sha(source["receipt_path"])})

claims = []
for n, claim in enumerate(bank["claims"]):
    source_id = claim["source_id"]
    assert texts[source_id][claim["offset"]:claim["offset"] + len(claim["excerpt"])] == claim["excerpt"]
    source = next(x for x in sources if x["source_id"] == source_id)
    claims.append({
        **claim,
        "claim_id": f"checked_bank_{n + 1}",
        "publication_time": claim.get("comment_published_at", source["published_at"]),
        "parent_post_publication_time": source["published_at"],
        "available_at": source["obtained_at"],
        "span_verified": True,
        "is_market_fact": False,
        "is_role_truth": False,
    })

author_question = "1时代和九阳是装糖还是真糖？细节是什么？它们7板意义大吗？二板非传媒里远程百洋都是Y？有没有独立走出来的可能性？"
question_offset = texts["xueqiu_1012"].index(author_question)
claims.append({"claim_id": "xueqiu_1012_author_y_question", "source_id": "xueqiu_1012",
               "source_section": "AUTHOR_BODY", "offset": question_offset, "excerpt": author_question,
               "publication_time": "2026-10-12T16:23:00+08:00",
               "available_at": "2026-10-14T12:19:01.705127+00:00", "span_verified": True,
               "is_market_fact": False, "is_role_truth": False,
               "claim_scope": "AUTHOR_ASKS_Y_LINK_AND_INDEPENDENCE_NOT_AUTHOR_ASSERTS_JIUYANG_FUNDS_CAUSALITY"})

# Extract checked Sept30 author paragraphs from the exact saved decoded tool string.
# The offset is against the decoded JSON string, explicitly not raw JSON bytes.
start930 = sun_text.index("930_延边刺客_淘股吧")
end930 = len(sun_text)
text930 = sun_text[start930:end930]
source930_id = "tgb_0930_background"
sun_available = sun_receipt["completed_at"]
sources.append({
    "source_id": source930_id,
    "kind": "ORIGINAL_FORUM",
    "url": "https://m.tgb.cn/a/2vvUuPqu2Tl",
    "author": "延边刺客",
    "published_at": "2026-09-30T16:39:00+08:00",
    "publication_precision": "MINUTE",
    "obtained_at": sun_available,
    "obtained_at_precision": "CONSERVATIVE_COMPLETION_RECEIPT_NOT_EXACT_TOOL_FETCH",
    "capture_kind": "PUBLIC_WEB_TOOL_SAVED_JSON_STRING",
    "text_path": str((ROOT / SUN_WEB_PATH).resolve()),
    "text_sha256": sha(ROOT / SUN_WEB_PATH),
    "offset_encoding": "DECODED_JSON_STRING_UNICODE_CHARACTERS",
    "body_available": True,
    "source_content_scope": "SEPT30_BACKGROUND_NOT_CURRENT_DAY_CONSENSUS",
})
for suffix, first_line, last_line, focus in (
    ("consumption", 51, 57, "消费段明确点名阳光，九阳/均瑶发酵、后排承接未明同时保留"),
    ("y_context", 59, 64, "化工段Y拼音联想背景，不把其直接归属阳光"),
    ("anchor_alternatives", 109, 146, "均瑶/Y、九阳/Y、新华全局背景及对手/承接反证；只保留背景范围"),
):
    a = start930 + text930.index(f"L{first_line}:")
    marker = f"L{last_line + 1}:"
    b = start930 + text930.index(marker)
    excerpt = sun_text[a:b]
    assert excerpt.startswith(f"L{first_line}:")
    claims.append({
        "claim_id": f"tgb_0930_{suffix}",
        "source_id": source930_id,
        "source_section": "AUTHOR_BODY",
        "offset": a,
        "excerpt": excerpt,
        "original_line_range": f"L{first_line}-L{last_line}",
        "publication_time": "2026-09-30T16:39:00+08:00",
        "available_at": sun_available,
        "review_focus": focus,
        "span_verified": True,
        "is_market_fact": False,
        "is_role_truth": False,
    })

all_rows = sorted(pool["rows"], key=lambda x: x["code"])
by_code = {r["code"]: r for r in all_rows}
rows = []
for r in all_rows:
    peers = sorted(x["code"] for x in all_rows if x["boards"] == r["boards"] and x["code"] != r["code"])
    assert peers == sorted(r["complete_same_height_peers"])
    relevant = [c["claim_id"] for c in claims if c.get("code") == r["code"]]
    if r["code"] == "001318":
        relevant = ["tgb_0930_consumption", "tgb_0930_y_context", "tgb_0930_anchor_alternatives"]
    rows.append({
        "code": r["code"], "name": r["name"], "boards": r["boards"],
        "board_scope": r["reported_board_scope"],
        "resume_calendar_qualification_gap": r["resume_calendar_qualification_gap"],
        "same_height_peers": peers,
        "coverage": "OLDER_CHECKED_DISCUSSION_BACKGROUND" if relevant else "UNKNOWN_IN_BOUNDED_CHECKED_SOURCES",
        "claim_ids": relevant,
        "narrative_absent": None,
        "unknown_is_negative_evidence": False,
        "unknown_is_positive_evidence": False,
        "confirmed_catchup_dragon": False,
        "promotion_direction": None,
        "tradeability_verified": False,
    })

# Dates stop at the Oct14 signal close; no Oct15 outcome is read or emitted.
dates = ["2026-10-09", "2026-10-12", "2026-10-13", "2026-10-14"]
known_boards = {"002242": {dates[0]: 5, dates[1]: 6, dates[2]: None, dates[3]: None},
                "002696": dict(zip(dates, [1, 2, 3, 4])),
                "001318": {dates[0]: None, dates[1]: None, dates[2]: 1, dates[3]: 2}}
for r in all_rows:
    known_boards.setdefault(r["code"], {})["2026-10-14"] = r["boards"]
temporal = []
for code in sorted(set(by_code) | {"002242"}):
    if code not in daily:
        continue
    d = daily[code]
    raw_path = pathlib.Path(d["raw_path"])
    if raw_path.exists():
        bindings.append({"path": str(raw_path), "sha256": sha(raw_path)})
    for date in dates:
        day = d["days"].get(date)
        if day is None:
            temporal.append({"code": code, "date": date, "status": "UNKNOWN_IN_SAVED_DAILY"})
            continue
        equal = len({day[x] for x in ("open", "high", "low", "close")}) == 1
        board_source = None
        if date == "2026-10-14" and code in by_code:
            pool_index = next(i for i, x in enumerate(pool["rows"]) if x["code"] == code)
            board_source = {**binding(POOL_PATH), "json_pointer": f"/rows/{pool_index}/boards"}
        elif code in ("002242", "002696"):
            board_source = {**binding(RELATION_PATH), "json_pointer": f"/timeline/{dates.index(date)}/{'jiuyang_archived_boards' if code == '002242' else 'baiyang_archived_boards'}"}
        elif code == "001318" and date == "2026-10-13":
            board_source = {**binding("results/sunshine_jiuyang_relation_20261014T152814Z/阳光九阳补涨关系与候选比较.md"), "scope": "EXPLICIT_TABLE_10_13_RESTART_FIRST_BOARD"}
        temporal.append({
            "code": code, "date": date, "ohlc_cents": {x: day[x] for x in ("open", "high", "low", "close")},
            "one_price_ohlc": equal,
            "one_price_is_execution_confirmation": False,
            "archived_boards": known_boards.get(code, {}).get(date),
            "board_status": "EXPLICIT_PRIOR_REVIEW_REUSED" if date in known_boards.get(code, {}) else "PRIOR_BOARD_IDENTITY_UNKNOWN",
            "board_source": board_source,
            "available_at": d["available_at"],
            "data_scope": "CURRENT_KNOWLEDGE_RETROSPECTIVE_SIGNAL_DAY_CHRONOLOGY_NOT_PRIOR_CUTOFF_ELIGIBILITY",
            "source_path": str((ROOT / DAILY_PATH).resolve()),
            "source_json_pointer": f"/{code}/days/{date}",
            "raw_path": d["raw_path"],
            "auction_verified": d.get("auction_verified", False),
            "reseal_verified": d.get("reseal_verified", False),
        })

base_edge = {"capital_migration_confirmed": False, "unique_parent_confirmed": False,
             "confirmed_catchup_dragon": False, "promotion_direction": None, "training_role_label": None,
             "fit_authorized": False, "original_prediction_backfill_allowed": False}
edges = [
    {**base_edge, "edge_id": "baiyang_jiuyang", "child_code": "002696", "anchor_code": "002242",
     "relation_type": "CROSS_INDUSTRY_NAME_EMOTIONAL_CATCHUP_HYPOTHESIS", "evidence_grade": "EXPLICIT_NAME_CHAIN_AND_DISTINCT_COMMENT_WITH_COMPATIBLE_RETROSPECTIVE_TIMING",
     "claim_ids": ["checked_bank_1", "xueqiu_1012_author_y_question", "checked_bank_6"], "independent_catalyst_claim_ids": ["checked_bank_2"],
     "known_timing": {"child_start_date": "2026-10-09", "anchor_boards_at_child_start": 5, "anchor_first_streak_broken_date": "2026-10-13", "child_boards_at_anchor_break": 3, "child_boards_signal_close": 4},
     "alternative_anchors_or_explanations": ["三羊/圣阳同链背景，但无已核父子时序", "农业避险/魚粉等独立叙事", "新华全局情绪周期，不能自动等同直接父龙"],
     "cancel_or_weaken_conditions": ["完整名称关联池持续弱反馈，候选失去相对连续性", "消费/名字线缺扩散而独立行业解释更充分", "开盘实际承接或可成交未核"]},
    {**base_edge, "edge_id": "sunshine_jiuyang", "child_code": "001318", "anchor_code": "002242",
     "relation_type": "CONSUMPTION_NAME_LINE_OLD_FOLLOWER_RESTART_CATCHUP_HYPOTHESIS", "evidence_grade": "EXPLICIT_GROUP_PARAGRAPH_BACKGROUND_NOT_DIRECT_CURRENT_DAY_PAIR_BODY",
     "claim_ids": ["tgb_0930_consumption", "tgb_0930_y_context", "tgb_0930_anchor_alternatives"],
     "known_timing": {"prior_first_board_date": "2026-09-30", "restart_first_board_date": "2026-10-13", "anchor_first_streak_broken_date": "2026-10-13", "child_boards_signal_close": 2},
     "alternative_anchors_or_explanations": ["均瑶健康消费/Y背景锚", "百洋4板名称线领先候选，并非已经证实阳光的新父龙", "乳品/消费独立修复，尚未绑定全部上涨原因"],
     "cancel_or_weaken_conditions": ["补涨关系当日正文和资金因果仍未核", "次日相对二板全池弱化且消费无扩散", "盘口回封稳定性与成交条件未核"]},
    {**base_edge, "edge_id": "sunshine_junyao_alternative", "child_code": "001318", "anchor_code": "605388",
     "relation_type": "ALTERNATIVE_CONSUMPTION_NAME_LINE_ANCHOR", "evidence_grade": "SEPT30_GROUP_BACKGROUND_TIMELINE_UNKNOWN",
     "claim_ids": ["tgb_0930_consumption", "tgb_0930_anchor_alternatives"], "anchor_market_chronology_available": False,
     "cancel_or_weaken_conditions": ["需同口径父锚日线与节点验证，未核前不得排除或定为唯一父龙"]},
    {**base_edge, "edge_id": "xinjingdian_sector_role", "child_code": "603096", "anchor_code": None,
     "relation_type": "MEDIA_CATCHUP_AUTHOR_HYPOTHESIS_PARENT_UNKNOWN", "evidence_grade": "EXPLICIT_ROLE_OPINION_NO_DIRECT_PARENT_OR_CAUSAL_PROOF", "claim_ids": ["checked_bank_4"],
     "alternative_anchors_or_explanations": ["新华传媒为原文全局传媒锚但缺逐股父子绑定", "同期慈文、龙版等互相竞争和共同晋级路径"]},
    {**base_edge, "edge_id": "ciwen_sector_role", "child_code": "002343", "anchor_code": None,
     "relation_type": "MEDIA_FOLLOWING_AUTHOR_HYPOTHESIS_PARENT_UNKNOWN", "evidence_grade": "EXPLICIT_FOLLOWING_OPINION_NOT_CATCHUP_DRAGON_LABEL", "claim_ids": ["checked_bank_3"],
     "alternative_anchors_or_explanations": ["新华传媒全局锚背景；具体父子连接未知", "传媒自身轮动/相对承接，不能排除"]},
    {**base_edge, "edge_id": "cuhua_sector_context", "child_code": "603968", "anchor_code": None,
     "relation_type": "CHEMICAL_SECTOR_PARAGRAPH_CONTEXT_NOT_STOCK_PAIR", "evidence_grade": "STOCK_NAMED_BUT_SHANSHUI_XINHUA_RELATION_NOT_TRANSFERABLE_TO_CUHUA", "claim_ids": ["checked_bank_5"],
     "alternative_anchors_or_explanations": ["善水—新华是正文直接指称主体；不能偷换为醋化—新华", "独立化工/乙腈叙事需逐股公告核验"]},
]

chain_unknowns = [
    {"token": "三羊", "resolved_code": None, "status": "NAME_CHAIN_ONLY_CODE_AND_ANCHOR_TIMING_UNKNOWN", "claim_ids": ["checked_bank_1"]},
    {"token": "圣阳", "resolved_code": None, "status": "NAME_CHAIN_ONLY_CODE_AND_ANCHOR_TIMING_UNKNOWN", "claim_ids": ["checked_bank_1"]},
    {"token": "远程", "resolved_code": None, "status": "AUTHOR_Y_QUESTION_AND_COMMENT_BACKGROUND_NOT_CURRENT_POOL", "claim_ids": ["checked_bank_6"]},
]
unknown_sources = []
for q in sun_qual["sources"]:
    if q.get("status", "").startswith("AUTHOR_BODY_CHECKED"):
        continue
    unknown_sources.append({**q, "is_relation_confirmation": False})

# All existing input bindings are checked only; no shared files are edited.
unique_bindings = {b["path"]: b for b in bindings}
fixture = {
    "version": "supplement-recognition-source-fixture-1",
    "formed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "signal_date": "2026-10-14", "target_date": "2026-10-15",
    "review_timing": "AFTER_CURRENT_AND_PRIOR_MODEL_AND_OUTCOME_EXPOSURE_APPEND_ONLY_RECOGNITION_RESEARCH",
    "new_data_requests": 0, "new_fits": 0, "new_formal_forward_labels": 0,
    "current_pool_scope": "FULL15_ALTERNATIVE_REPORTED_TRADING_STREAK_NOT_FORMAL_SOURCE_QUALIFICATION",
    "model_scores_or_original_directions_included": False,
    "original_prediction_or_training_backfill_allowed": False,
    "name_match_scope": "CHECKED_AUTHOR_EXPLICIT_TEXT_ONLY_NO_NEW_PINYIN_OR_FULL_FORUM_EXPANSION",
    "sources": sources, "claims": claims, "rows": rows, "relation_hypotheses": edges,
    "background_chain_unresolved_entities": chain_unknowns,
    "unqualified_source_leads": unknown_sources,
    "temporal_facts": temporal,
    "source_bindings": list(unique_bindings.values()),
    "counts": {"complete_pool": len(rows), "older_discussion_background": sum(bool(r["claim_ids"]) for r in rows),
               "unknown_in_bounded_sources": sum(not r["claim_ids"] for r in rows), "current_signal_day_checked_relation_body": 0,
               "relation_hypotheses": len(edges), "temporal_rows": len(temporal), "confirmed_catchup_dragons": 0},
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "relation_fixture.json").write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "temporal_facts.json").write_text(json.dumps({"signal_date": fixture["signal_date"], "rows": temporal, "source_bindings": list(unique_bindings.values())}, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "full15_score_free_pool.json").write_text(json.dumps({"signal_date": "2026-10-14", "target_date": "2026-10-15", "rows": rows, "source_bindings": [binding(POOL_PATH)]}, ensure_ascii=False, indent=2), encoding="utf-8")

source_map = {s["source_id"]: s for s in sources}
claim_map = {c["claim_id"]: c for c in claims}
compact_edges = []
for e in edges:
    child = by_code[e["child_code"]]
    types = (["NAME_EMOTION"] if e["child_code"] == "002696" else
             ["CONSUMER", "NAME_EMOTION"] if e["child_code"] == "001318" else ["THEME", "BACKGROUND"])
    citations = []
    for cid in e["claim_ids"]:
        c = claim_map[cid]
        s = source_map[c["source_id"]]
        citations.append({"claim_id": cid, "source_id": c["source_id"], "source_section": c["source_section"],
                          "text_path": s["text_path"], "text_sha256": s["text_sha256"],
                          "offset_encoding": s.get("offset_encoding", "TEXT_UNICODE_CHARACTERS"),
                          "offset": c["offset"], "excerpt": c["excerpt"],
                          "published_at": c["publication_time"], "available_at": c["available_at"],
                          "is_role_truth": False})
    compact_edges.append({**e, "child": {"code": child["code"], "name": child["name"]},
                          "parent": {"code": e["anchor_code"], "name": "九阳股份" if e["anchor_code"] == "002242" else "均瑶健康" if e["anchor_code"] == "605388" else None},
                          "relation_types": types,
                          "connection_strength": "EXPLICIT_CO_MENTION" if e["edge_id"] == "baiyang_jiuyang" else "GROUP_ASSOCIATION" if e["child_code"] == "001318" else "SECTOR_CONTEXT_PARENT_UNKNOWN",
                          "source_citations": citations})
compact = {
    "version": "supplement-relation-bank-fixture-1", "signal_date": fixture["signal_date"], "target_date": fixture["target_date"],
    "formed_at": fixture["formed_at"], "full_pool_path": str((OUT / "full15_score_free_pool.json").resolve()),
    "anchors_are_forecast_candidates": False,
    "anchors": [{"code": "002242", "name": "九阳股份", "scope": "HISTORICAL_HIGH_EMOTION_ANCHOR_HYPOTHESIS", "temporal_facts_path": str((OUT / "temporal_facts.json").resolve())},
                {"code": "605388", "name": "均瑶健康", "scope": "ALTERNATIVE_BACKGROUND_ANCHOR", "market_timing": "UNKNOWN_IN_THIS_BOUNDED_FIXTURE", "name_code_binding": {**binding(JUNYAO_NAME_PATH), "line": 65, "used_fields_only": ["code", "name"], "existing_role_opinion_or_prediction_not_used_as_truth": True}},
                {"code": None, "name": "三羊", "scope": "NAME_CHAIN_ALTERNATIVE_CODE_AND_TIMING_UNKNOWN"},
                {"code": None, "name": "圣阳", "scope": "NAME_CHAIN_ALTERNATIVE_CODE_AND_TIMING_UNKNOWN"}],
    "edges": compact_edges,
    "temporal_facts_path": str((OUT / "temporal_facts.json").resolve()),
    "explicit_pair_chronology_binding": {**binding(RELATION_PATH), "json_pointer": "/timeline", "scope": "CURRENT_REVIEW_REUSED_NOT_OLD_PREDICTION"},
    "restart_vs_original_start": [{"code": "002696", "original_start_date": "2026-10-09", "restart_date": None, "source": binding(RELATION_PATH)},
                                  {"code": "001318", "original_first_observed_board_date": "2026-09-30", "restart_date": "2026-10-13", "source": binding("results/sunshine_jiuyang_relation_20261014T152814Z/阳光九阳补涨关系与候选比较.md"), "not_all_lifetime_first_start": True}],
    "unqualified_source_leads": unknown_sources,
    "unknown_or_uncovered_pool_codes": [r["code"] for r in rows if not r["claim_ids"]],
    "confirmed_catchup_dragon_count": 0, "new_fits": 0, "training_role_labels_available": False,
    "not_prior_blind_review": True, "source_bindings": fixture["source_bindings"],
}
(OUT / "compact_relation_bank.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
check = {"counts": fixture["counts"], "bank_span_checks": len(bank["claims"]), "sept30_span_checks": 3,
         "peer_completeness_checks": 15, "all_input_hashes_recomputed": len(unique_bindings),
         "no_oct15_temporal_rows": all(r["date"] <= "2026-10-14" for r in temporal),
         "no_model_scores_or_original_direction_emitted": all("raw_score" not in r and "predicted_positive" not in r for r in rows),
         "output_sha256": sha(OUT / "relation_fixture.json")}
(OUT / "evidence_checks.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(check, ensure_ascii=False))
