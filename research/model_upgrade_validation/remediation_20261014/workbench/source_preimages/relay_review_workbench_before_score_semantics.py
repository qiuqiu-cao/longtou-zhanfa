"""One additive entry for full-pool forum, parent-anchor and author review.

Frozen engines retain their original hashes and guards. Shared prose references
are only a formatting convenience; directions and review decisions stay explicit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path

import forum_reasoning_review as forum
import forum_reasoning_outcome_context as feedback
import stock_judgement_review as original

VERSION = "relay-review-workbench-1"
# Only author-written prose can be replaced by a reference. Enum decisions,
# identities, source paths, state, timestamps and forecasts cannot be referenced.
TEXT_FIELDS = {"reason", "synthesis", "agreement_reason", "author_claim", "author_reasoning",
    "testable_expectation", "role_hypothesis", "relative_thesis", "support", "rebuttal",
    "cancel_condition", "requirement", "counterevidence", "cancel_conditions",
    "alternative_explanations", "conditions", "counter_claims", "opening_confirmation", "missing"}


def _receipt(path, status, **values):
    result = {"version": VERSION, "status": status, "formed_at": original.now().isoformat(),
        "new_fits": 0, "directions_autofilled": False, **values}
    original.write_new(path, result)
    return result


def prepare(features, theme_report, mainline_report, compatibility, bank, calendar, output_dir,
            pool, relation_bank, exposure="MODEL_OUTPUT_ALREADY_SEEN", role_evidence=None,
            history_datasets=(), mainline_history=(), narrative_reviews=None):
    """Forward every history input to the guarded current review pipeline."""
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError("new immutable workbench directory required")
    try:
        result = forum.prepare(features=features, theme_report=theme_report, mainline_report=mainline_report,
            compatibility=compatibility, bank=bank, calendar=calendar, output_dir=out / "pipeline",
            pool=pool, relation_bank=relation_bank, exposure=exposure, role_evidence=role_evidence,
            history_datasets=tuple(history_datasets), mainline_history=tuple(mainline_history),
            narrative_reviews=narrative_reviews)
        facts = out / "pipeline" / "reasoning" / "facts_with_forum_reasoning.json"
        render(facts, out / "review")
    except Exception as exc:
        _receipt(out / "prepare_failure.json", "WORKBENCH_PREPARATION_FAILED_OR_RENDER_INCOMPLETE",
                 error_type=type(exc).__name__, error=str(exc), original_pipeline_artifacts_preserved=True)
        raise
    _receipt(out / "receipt.json", "PREPARED_FULL_POOL_AUTHOR_REVIEW_PENDING", facts_path=str(facts.resolve()),
        facts_sha256=original.sha(facts), pool_size=len(result["rows"]),
        history_datasets=[str(Path(p).resolve()) for p in history_datasets],
        mainline_history=[str(Path(p).resolve()) for p in mainline_history],
        narrative_reviews=str(Path(narrative_reviews).resolve()) if narrative_reviews else None,
        source_bindings=original.bind(__file__) | original.bind(facts))
    return result


def attach(facts_path, output_dir, draft_path=None):
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError("new immutable workbench directory required")
    result = forum.attach(facts_path, out / "pipeline", draft_path)
    render(out / "pipeline" / "facts_with_forum_reasoning.json", out / "review")
    return result


def freeze(facts_path, judgements_path, output):
    # This always goes through the old full role/catalyst/clock/registry checks.
    return forum.freeze(facts_path, judgements_path, output)


def review_context(facts_path, judgements_path, output):
    return forum.review_context(facts_path, judgements_path, output)


def compare(record, baseline, outcome_archive, output, feedback_dir=None):
    """Keep the original comparison even if additive argument feedback fails."""
    output = Path(output)
    receipt = output.with_name(output.stem + ".workbench.json")
    if receipt.exists():
        raise FileExistsError("workbench comparison receipt already exists")
    try:
        result = forum.compare(record, baseline, outcome_archive, output)
    except Exception as exc:
        _receipt(receipt, "ORIGINAL_COMPARISON_REJECTED", comparison_path=str(output.resolve()),
            error_type=type(exc).__name__, error=str(exc), original_artifacts_preserved=True)
        raise
    destination = Path(feedback_dir) if feedback_dir else output.with_name(output.stem + "_argument_feedback")
    try:
        report = feedback.project(record, output, destination)
    except Exception as exc:
        _receipt(receipt, "COMPARISON_SAVED_ARGUMENT_FEEDBACK_FAILED", comparison_path=str(output.resolve()),
            comparison_sha256=original.sha(output), feedback_path=str(destination.resolve()),
            error_type=type(exc).__name__, error=str(exc), original_comparison_preserved=True)
        raise
    _receipt(receipt, "COMPARISON_AND_ARGUMENT_FEEDBACK_SAVED", comparison_path=str(output.resolve()),
        comparison_sha256=original.sha(output), feedback_path=str(destination.resolve()),
        feedback_status=report["status"], source_bindings=original.bind(output) | original.bind(__file__))
    return result


def _prose_walk(value, transform, allowed=False, path=()):
    if isinstance(value, dict):
        if "$reason" in value:
            if not allowed or set(value) != {"$reason"}:
                raise ValueError("reason reference is only valid in an author prose field")
            return transform(value, path)
        # Structured conditions contain dates, metrics and expected values;
        # prose ancestry must not make those enum/clock fields referenceable.
        return {k: _prose_walk(v, transform, k in TEXT_FIELDS or
            (bool(path) and path[-1] == "opening_confirmation" and k in {"09:25", "09:35", "10:00"}), (*path, k))
            for k, v in value.items()}
    if isinstance(value, list):
        return [_prose_walk(v, transform, allowed, (*path, str(i))) for i, v in enumerate(value)]
    return transform(value, path) if allowed and isinstance(value, str) and value.strip() else deepcopy(value)


def compact(facts_path, draft_path, output):
    """Factor existing prose once; retain the whole scaffold and all decisions."""
    facts, draft = original.read(facts_path), original.read(draft_path)
    forum.checked(facts)
    _draft_binding(facts_path, facts, draft)
    bank = {}
    def compress(value, path):
        if not isinstance(value, str):
            raise ValueError("source draft must contain plain prose, not nested reason references")
        rid = "reason_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
        entry = bank.setdefault(rid, {"text": value, "original_locations": []})
        if entry["text"] != value:
            raise ValueError("reason identity collision")
        entry["original_locations"].append("/" + "/".join(path))
        return {"$reason": rid}
    # Do not compact provenance metadata: only row author prose is reusable.
    reusable = deepcopy(draft)
    reusable["rows"] = _prose_walk(draft["rows"], compress)
    result = {"version": VERSION, "status": "AUTHOR_PROSE_REFERENCE_DRAFT_NOT_A_FORECAST",
        "facts_path": str(Path(facts_path).resolve()), "facts_sha256": original.sha(facts_path),
        "base_draft_path": str(Path(draft_path).resolve()), "base_draft_sha256": original.sha(draft_path),
        "source_bindings": facts["source_bindings"] | original.bind(facts_path) | original.bind(draft_path) | original.bind(__file__),
        "reason_bank": bank, "draft": reusable, "new_fits": 0,
        "directions_roles_and_unknowns_autofilled": False,
        "instructions": ["作者可编辑共享理由或复用引用；仅作用于理由文本。",
            "方向、角色状态、采纳决定、是否完成审阅必须逐股由作者填写。",
            "expand会还原完整稿并执行原校验；freeze再次走原全部登记门。"]}
    original.write_new(output, result)
    return result


def _draft_binding(facts_path, facts, draft):
    if draft.get("facts_path") != str(Path(facts_path).resolve()) or draft.get("facts_sha256") != original.sha(facts_path):
        raise ValueError("draft must bind its exact facts")
    if draft.get("signal_date") != facts["signal_date"] or draft.get("target_date") != facts["target_date"]:
        raise ValueError("author draft dates changed")
    expected, actual = original.unique_rows(facts["rows"]), original.unique_rows(draft["rows"])
    if set(expected) != set(actual) or any((actual[c]["name"], actual[c]["boards"]) != (expected[c]["name"], expected[c]["boards"]) for c in expected):
        raise ValueError("draft must retain every original identity and candidate")
    if draft.get("model_exposure") != facts["supplement_model_exposure"]:
        raise ValueError("actual model exposure cannot change")


def validate_completed(facts_path, draft):
    """Read-only completed-draft checks; no simulated clock or registration."""
    facts = original.read(facts_path)
    _draft_binding(facts_path, facts, draft)
    if draft.get("author_review_completed") is not True:
        raise ValueError("author review is still pending; templates are not decisions")
    original.check_sources(draft.get("source_bindings", {}))
    if any(draft.get("source_bindings", {}).get(p) != sha for p, sha in
           (facts["source_bindings"] | original.bind(facts_path)).items()):
        raise ValueError("completed draft must retain all original source bindings")
    forum.validate(forum.checked(facts), draft, facts["source_bindings"])
    forum.supplement.validate_review(facts, draft)
    if facts["supplement_context_diagnostic_only"]:
        if any(row.get("promotion_judgement") is not None for row in draft["rows"]):
            raise ValueError("diagnostic review cannot introduce a retroactive forecast")
        return "DIAGNOSTIC_APPEND_ONLY_REVIEW_VALIDATED"
    legacy = forum.supplement.legacy
    if draft.get("status") in {"PENDING_AUTHOR_REVIEW_TEMPLATE", "DIAGNOSTIC_TEMPLATE_NO_FORECAST"}:
        raise ValueError("pending template is not a completed role review")
    legacy.check_role_facts(facts)
    legacy.short_term.validate_assessments(legacy.short_term.checked_projection(facts, required=True), draft)
    legacy.expectation.validate_plans(facts, draft)
    narrative = legacy.narrative.checked(facts)
    if narrative is None:
        raise ValueError("typed narrative context required")
    legacy.narrative.validate_assessments(narrative, draft)
    bycode = original.unique_rows(facts["rows"])
    for row in draft["rows"]:
        roles = row.get("role_assessment", {})
        if set(roles) != set(legacy.e.ROLES):
            raise ValueError("all eleven role assessments required")
        for role, value in roles.items():
            if value.get("state") not in {"CANDIDATE", "EVIDENCE_SUPPORTED", "COMPETING", "INVALIDATED", "UNKNOWN"}:
                raise ValueError("unsupported role state")
            if bycode[row["code"]]["leader_role_evidence"]["roles"][role]["missing"] and value["state"] != "UNKNOWN":
                raise ValueError("missing required role facts cannot be certified")
            if not isinstance(value.get("reason"), str) or not value["reason"].strip():
                raise ValueError("explicit reason for each role required")
    catalyst = forum.supplement.catalyst
    catalyst.validate_catalyst_assessments(catalyst.checked_context(facts), draft)
    original.validate_fact_projection(facts)
    original.validate_judgements(facts, draft)
    original.pre_auction(facts["signal_date"], facts["target_date"], facts["sessions"], original.now())
    return "FULL_DRAFT_VALIDATED_FINAL_FREEZE_STILL_REQUIRED"


def expand(compact_path, output):
    packet = original.read(compact_path)
    if packet.get("version") != VERSION:
        raise ValueError("unsupported compact draft schema")
    original.check_sources(packet["source_bindings"])
    for path, key in ((packet["facts_path"], "facts_sha256"), (packet["base_draft_path"], "base_draft_sha256")):
        if original.sha(path) != packet[key]:
            raise ValueError("compact original draft or facts changed")
    base = original.read(packet["base_draft_path"])
    draft = deepcopy(packet["draft"])
    # Prose and explicit author row decisions may change; source clocks, review
    # mode, exposure, facts and provenance must retain the original values.
    for key, value in base.items():
        if key not in {"rows", "status", "author_review_completed", "reviewed_at"} and draft.get(key) != value:
            raise ValueError("compact draft changed original metadata: " + key)
    def resolve(value, path):
        if isinstance(value, str):
            return value
        rid = value["$reason"]
        entry = packet["reason_bank"].get(rid)
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str) or not entry["text"].strip():
            raise ValueError("missing or empty shared author reason: " + str(rid))
        return entry["text"]
    draft["rows"] = _prose_walk(draft["rows"], resolve)
    validation = validate_completed(packet["facts_path"], draft)
    draft["source_bindings"] = draft["source_bindings"] | original.bind(compact_path)
    original.write_new(output, draft)
    _receipt(Path(output).with_name(Path(output).stem + ".expansion.json"), "SHARED_PROSE_EXPANDED_AND_VALIDATED",
        expanded_draft_path=str(Path(output).resolve()), expanded_draft_sha256=original.sha(output),
        compact_path=str(Path(compact_path).resolve()), validation=validation,
        reason_count=len(packet["reason_bank"]),
        source_bindings=original.bind(compact_path) | original.bind(output) | original.bind(__file__))
    return draft


def review_projection(facts, draft=None):
    """Allowlisted card projection: no model role scores or baseline calls."""
    candidates = original.unique_rows(facts["rows"])
    def peers(codes):
        return [{"code": c, "name": candidates[c]["name"] if c in candidates else "池外锚/待核"} for c in codes]
    reviewed = original.unique_rows(draft["rows"]) if draft else {}
    cards = []
    for row in facts["rows"]:
        rc = row.get("supplement_relation_context", {})
        relations = [{k: deepcopy(edge.get(k)) for k in ("relation_id", "anchor_code", "anchor_name",
            "anchor_in_current_pool", "relation_types", "connection_strength", "recognition_state", "timing",
            "alternative_anchors_or_explanations", "cancel_or_weaken_conditions", "discussion_evidence")}
            for edge in rc.get("relations", [])]
        opinions = [{k: deepcopy(opinion.get(k)) for k in ("opinion_id", "author", "source_section", "url",
            "discussion_published_at", "parent_post_published_at", "obtained_at", "reviewed_at", "available_at",
            "use", "excerpt", "text_path", "source_contains_question")}
            for opinion in row.get("forum_reasoning_context", {}).get("opinions", [])]
        author_fields = {"code", "name", "boards", "promotion_judgement", "role_hypothesis", "relative_thesis",
            "support", "rebuttal", "cancel_condition", "missing", "comparison_peers", "fact_references",
            "opening_confirmation", "catalyst_assessment", "turnover_comparison_claim", "role_assessment",
            "narrative_type_assessment", "short_term_assessment", "pairwise_competition_assessment",
            "forum_preread_assessment", "supplement_assessment", "forum_reasoning_assessment"}
        actual_review = reviewed.get(row["code"], {})
        cards.append({"code": row["code"], "name": row["name"], "boards": row["boards"],
            "same_height_peers": peers(row.get("same_height_peers", [])),
            "same_anchor_candidate_peers": peers(rc.get("same_anchor_candidate_peers", [])),
            "daily_price_volume": deepcopy(row.get("daily_price_volume", {})),
            "actual_one_word_daily": row.get("actual_one_word_daily"),
            "feature_semantics": deepcopy(row.get("feature_semantics", {})),
            "catalyst": deepcopy(row.get("catalyst_context", {})),
            "discussion_coverage": row.get("forum_reasoning_context", {}).get("coverage", "UNKNOWN"),
            "relations": relations, "opinions": opinions,
            "independent_review": {k: deepcopy(v) for k, v in actual_review.items() if k in author_fields} if draft else None})
    return {"version": VERSION, "signal_date": facts["signal_date"], "target_date": facts["target_date"],
        "pool_size": len(cards), "cards": cards,
        "diagnostic_only": facts.get("supplement_context_diagnostic_only", False),
        "model_exposure": facts.get("supplement_model_exposure", "UNKNOWN"),
        "model_output_displayed": False, "author_review_attached": draft is not None,
        "registered_by_this_view": False, "new_fits": 0}


def _escape(value):
    return html.escape(str(value if value is not None else "未知"))


def _md(value):
    return str(value if value is not None else "未知").replace("|", "\\|").replace("\n", "<br>")


def markdown(report):
    lines = ["# 完整个股审阅工作台", "",
        f"信号日 {report['signal_date']} → {report['target_date']}；完整 {report['pool_size']} 只。",
        f"诊断附页：{report['diagnostic_only']}；真实模型暴露：{report['model_exposure']}。",
        "页面隐藏全部模型分数与原模型方向；作者审阅与未知均须实际填写。", "",
        "| 个股 | 板位 | 同位 | 讨论覆盖 | 父锚 |", "|---|---:|---|---|---|"]
    for c in report["cards"]:
        names = lambda items: "、".join(p["name"] + " " + p["code"] for p in items) or "无"
        anchors = "、".join((r["anchor_name"] or "父锚未知") + "：" + str(r["recognition_state"]) for r in c["relations"]) or "未知"
        lines.append(f"| {c['name']} {c['code']} | {c['boards']} | {_md(names(c['same_height_peers']))} | {c['discussion_coverage']} | {_md(anchors)} |")
    for c in report["cards"]:
        lines += ["", f"## {c['name']} {c['code']}（{c['boards']}板）", "",
            "同位：" + names(c["same_height_peers"]), "同父锚候选：" + names(c["same_anchor_candidate_peers"]),
            "日线价量：" + _md(json.dumps(c["daily_price_volume"], ensure_ascii=False)),
            "真实一字身份：" + str(c["actual_one_word_daily"])]
        if c["feature_semantics"]:
            lines += ["字段语义：" + _md(json.dumps(c["feature_semantics"], ensure_ascii=False))]
        for r in c["relations"]:
            lines += ["", "父锚假设：" + _md(r["anchor_name"] or "未知") + "；" + _md(r["connection_strength"]),
                "节点：" + _md(json.dumps(r["timing"], ensure_ascii=False)),
                "替代解释：" + _md("；".join(r["alternative_anchors_or_explanations"] or [])),
                "削弱／取消条件：" + _md("；".join(r["cancel_or_weaken_conditions"] or []))]
        if not c["opinions"]:
            lines += ["", "限定来源未覆盖；未点名不等于无催化，不自动看空或转正。"]
        for p in c["opinions"]:
            lines += ["", f"讨论 {p['opinion_id']}：{_md(p['author'])}／{p['source_section']}／{p['use']}",
                "正文／评论各自发布时间：" + _md(p["discussion_published_at"]) + "；主帖：" + _md(p["parent_post_published_at"]),
                "取得：" + _md(p["obtained_at"]) + "；核阅后可用：" + _md(p["available_at"]),
                "> " + _md(p["excerpt"]), "来源：" + _md(p["url"])]
        if c["independent_review"]:
            lines += ["", "作者实际审阅：", "", "```json", json.dumps(c["independent_review"], ensure_ascii=False, indent=2), "```"]
        else:
            lines += ["", "独立方向／采纳决定／反证／取消条件：待作者审阅。"]
    return "\n".join(lines) + "\n"


def html_page(report):
    def paragraph(label, value):
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        return '<p><strong>' + _escape(label) + '：</strong>' + _escape(value) + '</p>'
    def items(values):
        return '<ul>' + ''.join('<li>' + _escape(x) + '</li>' for x in (values or [])) + '</ul>' if values else '<p class="muted">未知／未提供</p>'
    def names(values):
        return '、'.join(v["name"] + ' ' + v["code"] for v in values) or '无'
    def source_link(url):
        if isinstance(url, str) and url.startswith(("https://", "http://")):
            return '<a href="' + _escape(url) + '" target="_blank" rel="noopener noreferrer">核对原讨论</a>'
        return '<span>原讨论链接未知</span>'
    cards = []
    nav = []
    for card in report["cards"]:
        # A filtered view keeps all originals in the DOM and in the JSON/MD.
        nav.append('<a href="#stock-' + card['code'] + '">' + _escape(card['name']) + ' · ' + str(card['boards']) + '</a>')
        relation_panels = []
        for r in card['relations']:
            t = r['timing'] or {}
            relation_panels.append('<div class="relation"><h4>' + _escape(r['anchor_name'] or '父锚未知') + '</h4>' +
                paragraph('关系证据', r['connection_strength']) + paragraph('候选类型', r['recognition_state']) +
                paragraph('本轮启动', t.get('current_streak_start')) + paragraph('旧首板观察', t.get('prior_first_observed_board')) +
                paragraph('锚首次观察断板', t.get('anchor_first_observed_streak_break')) +
                '<strong>替代解释</strong>' + items(r['alternative_anchors_or_explanations']) +
                '<strong>削弱／取消条件</strong>' + items(r['cancel_or_weaken_conditions']) + '</div>')
        review = card['independent_review'] or {}
        argument_reviews = {a['opinion_id']: a for a in review.get('forum_reasoning_assessment', {}).get('opinions', [])}
        opinions = []
        for p in card['opinions']:
            a = argument_reviews.get(p['opinion_id'])
            body = '<div class="opinion"><h4>' + _escape(p['author']) + ' <span class="tag">' + _escape(p['source_section']) + '</span></h4>'
            body += '<blockquote>' + _escape(p['excerpt']) + '</blockquote>'
            body += '<details><summary>来源和各自时间</summary>' + paragraph('正文／评论各自发布时间', p['discussion_published_at']) + paragraph('主帖', p['parent_post_published_at']) + paragraph('取得', p['obtained_at']) + paragraph('核阅后可用', p['available_at']) + paragraph('用途', p['use']) + source_link(p['url']) + '</details>'
            if a:
                body += '<div class="argument"><span class="tag">' + _escape(a.get('decision')) + '</span>'
                for k, label in (('author_claim', '作者命题'), ('author_reasoning', '作者推理'), ('testable_expectation', '可检验预期'), ('reason', '独立采纳理由'), ('independent_effect', '对预期的影响')):
                    body += paragraph(label, a.get(k))
                body += paragraph('核验', a.get('verification')) + '<strong>反证</strong>' + items(a.get('counterevidence')) + '<strong>取消条件</strong>' + items(a.get('cancel_conditions')) + '</div>'
            else:
                body += '<p class="pending">独立采纳决定、依据、反证和取消条件：待作者审阅。</p>'
            opinions.append(body + '</div>')
        synthesis = review.get('forum_reasoning_assessment', {})
        review_panel = '<p class="pending">独立方向与实际审阅：待作者填写。</p>'
        if review:
            review_panel = paragraph('实际独立方向', review.get('promotion_judgement')) + paragraph('综合判断', synthesis.get('synthesis')) + paragraph('与作者分歧', synthesis.get('agreement')) + paragraph('分歧依据', synthesis.get('agreement_reason'))
            review_panel += '<details><summary>完整作者审阅字段</summary><pre>' + _escape(json.dumps(review, ensure_ascii=False, indent=2)) + '</pre></details>'
        cards.append('<article id="stock-' + card['code'] + '" data-search="' + _escape(card["name"] + " " + card["code"]) + '"><header><h2>' +
            _escape(card["name"] + " " + card["code"]) + '</h2><span class="board">' + str(card['boards']) + '板</span></header>' +
            '<div class="grid"><section><h3>事实与完整竞争</h3>' + paragraph('同位', names(card['same_height_peers'])) + paragraph('同父锚候选', names(card['same_anchor_candidate_peers'])) +
            paragraph('真实一字身份', card['actual_one_word_daily']) + paragraph('日线价量', card['daily_price_volume'] or '本附页未携带原行情，保持未知') + paragraph('讨论覆盖', card['discussion_coverage']) +
            '<details><summary>催化与字段语义</summary>' + paragraph('催化', card['catalyst']) + paragraph('字段语义', card['feature_semantics']) + '</details></section>' +
            '<section><h3>跨日父锚假设</h3>' + (''.join(relation_panels) or '<p class="muted">限定来源没有本股直接父锚证据；不等于无催化。</p>') + '</section></div>' +
            '<section><h3>股吧命题与独立分析</h3>' + (''.join(opinions) or '<p class="muted">限定来源未覆盖；不自动看空或转正。</p>') + '</section>' +
            '<section class="review"><h3>实际审阅与预期</h3>' + review_panel + '</section></article>')
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>完整个股审阅工作台</title><style>body{font:15px/1.65 system-ui;margin:24px auto;padding:0 18px;max-width:1200px;color:#17313c;background:#f3f6f8}'
        'h1,h2,h3,h4{line-height:1.3}h4{margin-bottom:8px}article{background:white;border:1px solid #ccd8dd;border-radius:12px;padding:22px;margin:22px 0;scroll-margin-top:100px}'
        'header{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #dde5e9}section{overflow-wrap:anywhere;margin:12px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:26px}'
        'pre{font:inherit;white-space:pre-wrap;overflow-wrap:anywhere}input{font:inherit;padding:8px;width:300px;max-width:90%;border:1px solid #9fb2bd;border-radius:8px}.status,.review{padding:16px;background:#e2eef3;border-radius:8px}'
        '.toolbar{position:sticky;top:0;background:#f3f6f8;padding:12px 0;z-index:1}nav{display:flex;flex-wrap:wrap;gap:8px}nav a,.tag,.board{background:#eaf0f3;padding:3px 8px;border-radius:5px}.board{font-size:20px;font-weight:700;background:#d7e9e5}'
        'a{color:#126a7a;text-decoration:none}a:hover{text-decoration:underline}.muted,.pending{color:#607880}.relation,.opinion{border-left:3px solid #8db6b2;padding:4px 14px;margin:14px 0}'
        'blockquote{margin:10px 0;background:#f6f8fa;padding:12px;border-radius:6px}.argument{background:#f7faf9;padding:10px;margin-top:12px}details{margin:12px 0}summary{cursor:pointer;color:#126a7a}p{margin:7px 0}ul{padding-left:22px}@media(max-width:750px){.grid{grid-template-columns:1fr;gap:0}article{padding:15px}}</style></head><body>'
        '<h1>完整个股审阅工作台</h1><p>' + _escape(report["signal_date"] + " → " + report["target_date"]) +
        '；完整 ' + str(report["pool_size"]) + ' 只</p><p class="status">诊断附页：' + str(report["diagnostic_only"]) +
        '；真实模型暴露：' + _escape(report["model_exposure"]) + '。模型分数与原方向隐藏。此页不登记预测。</p><nav>' + ''.join(nav) + '</nav>'
        '<div class="toolbar"><label>查找股票 <input id="query" placeholder="名称或代码"></label><span id="count"></span></div>' + ''.join(cards) +
        '<script>const q=document.getElementById("query"),a=[...document.querySelectorAll("article")];function filter(){let n=0;for(const e of a){e.hidden=!e.dataset.search.includes(q.value.trim());if(!e.hidden)n++}document.getElementById("count").textContent=" 显示 "+n+" / "+a.length}q.addEventListener("input",filter);filter()</script></body></html>')


def render(facts_path, output_dir, review_path=None):
    facts = original.read(facts_path)
    forum.checked(facts)
    draft = None
    if review_path:
        draft = original.read(review_path)
        validate_completed(facts_path, draft)
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError("new immutable review view required")
    report = review_projection(facts, draft)
    out.mkdir(parents=True)
    original.write_new(out / "review.json", report)
    (out / "review.md").write_text(markdown(report), encoding="utf-8")
    (out / "review.html").write_text(html_page(report), encoding="utf-8")
    _receipt(out / "receipt.json", "FULL_POOL_SCORE_MASKED_READ_ONLY_VIEW", facts_path=str(Path(facts_path).resolve()),
        pool_size=report["pool_size"], source_bindings=facts["source_bindings"] | original.bind(facts_path) |
        (original.bind(review_path) if review_path else {}) | original.bind(__file__))
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    a = s.add_parser("prepare")
    for field in ("features", "theme-report", "mainline-report", "compatibility", "bank", "calendar", "output-dir", "pool", "relation-bank"):
        a.add_argument("--" + field, required=True)
    a.add_argument("--model-exposure", choices=sorted(original.EXPOSURES), default="MODEL_OUTPUT_ALREADY_SEEN")
    a.add_argument("--role-evidence")
    a.add_argument("--history-dataset", action="append", default=[])
    a.add_argument("--mainline-history", action="append", default=[])
    a.add_argument("--narrative-reviews")
    a = s.add_parser("attach")
    a.add_argument("--facts", required=True); a.add_argument("--output-dir", required=True); a.add_argument("--draft")
    for name in ("freeze", "review-context"):
        a = s.add_parser(name)
        for field in ("facts", "judgements", "output"):
            a.add_argument("--" + field, required=True)
    a = s.add_parser("compare")
    for field in ("record", "baseline", "outcome-archive", "output"):
        a.add_argument("--" + field, required=True)
    a.add_argument("--feedback-dir")
    a = s.add_parser("compact")
    for field in ("facts", "draft", "output"):
        a.add_argument("--" + field, required=True)
    a = s.add_parser("expand")
    a.add_argument("--compact", required=True); a.add_argument("--output", required=True)
    a = s.add_parser("render")
    a.add_argument("--facts", required=True); a.add_argument("--output-dir", required=True); a.add_argument("--review")
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.command == "prepare":
        result = prepare(features=a.features, theme_report=a.theme_report, mainline_report=a.mainline_report,
            compatibility=a.compatibility, bank=a.bank, calendar=a.calendar, output_dir=a.output_dir,
            pool=a.pool, relation_bank=a.relation_bank, exposure=a.model_exposure, role_evidence=a.role_evidence,
            history_datasets=a.history_dataset, mainline_history=a.mainline_history, narrative_reviews=a.narrative_reviews)
    elif a.command == "attach":
        result = attach(a.facts, a.output_dir, a.draft)
    elif a.command in {"freeze", "review-context"}:
        result = globals()[a.command.replace("-", "_")](a.facts, a.judgements, a.output)
    elif a.command == "compare":
        result = compare(a.record, a.baseline, a.outcome_archive, a.output, a.feedback_dir)
    elif a.command == "compact":
        result = compact(a.facts, a.draft, a.output)
    elif a.command == "expand":
        result = expand(a.compact, a.output)
    else:
        result = render(a.facts, a.output_dir, a.review)
    print(json.dumps({"status": result.get("status"), "pool_size": result.get("pool_size", len(result.get("rows", []))), "new_fits": 0}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
