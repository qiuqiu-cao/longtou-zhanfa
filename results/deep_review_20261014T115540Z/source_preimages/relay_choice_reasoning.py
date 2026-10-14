"""Explain immutable relay rankings, full peer comparisons and author theses.

This optional read-only delivery entry never replaces a forecast or edits the
protected report renderer. Logit terms explain the estimator, not market cause.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import stock_judgement_review as original
from next_session_expectation_contract_v271 import load_registered
from relay_error_analysis import RELAY_FIELDS, components, explain

VERSION = "relay-choice-reasoning-1"
FIELD_NAMES = {
    "boards": "当前连板高度", "height_gap": "距归档池最高板的距离",
    "same_height_rivals": "同身位对手数量", "unique_space": "旧空间唯一性特征",
    "height_share": "旧高度份额特征", "is_space_leader": "旧空间指示特征",
    "emotion_score": "原市场情绪特征", "promotion_2to3": "原二进三反馈",
    "promotion_3to4": "原三进四反馈", "promotion_mid": "原中位晋级反馈",
    "previous_limit_tail_loss": "原昨涨停尾部损失", "failed_rate": "原炸板比例",
    "down_limit_rate": "旧跌停密度特征", "breadth_health": "原涨停广度特征",
    "ladder_health": "原梯队特征", "loss_effect_health": "原亏钱效应特征",
    "crowding_risk": "原拥挤特征", "regime_fragility": "原环境脆弱特征",
    "turnover": "当日换手", "signal_amount": "当日成交额",
    "amplitude": "日线振幅", "turnover_vs_prev": "相对上次入连板池的换手比",
    "price": "股价特征", "price_same_height_percentile": "股价在同位组的相对位置",
    "signal_amount_same_height_percentile": "成交额在同位组的相对位置",
    "turnover_same_height_percentile": "换手在同位组的相对位置",
    "turnover_vs_prev_same_height_percentile": "上次入池换手比在同位组的相对位置",
    "signal_one_word": "旧低换手代理", "log_signal_amount": "成交额对数特征",
}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _same(a, b):
    return a is None and b is None or _finite(a) and _finite(b) and abs(a-b) < 1e-10


def _sigmoid(value):
    return 1/(1+math.exp(-value)) if value >= 0 else math.exp(value)/(1+math.exp(value))


def _stamp(value):
    return original.stamp(str(value).replace(" UTC", "+00:00").replace(" ", "T", 1))


def _unique(rows):
    result = original.unique_rows(rows)
    if len(result) != len(rows):
        raise ValueError("duplicate candidate identity")
    return result


def _label(field):
    if field.startswith("missingindicator_"):
        return "缺失标记：" + FIELD_NAMES.get(field[17:], field[17:])
    return FIELD_NAMES.get(field, field)


def check_forum_bank(path):
    bank = original.read(path)
    if (bank.get("version") != "choice-forum-reasoning-1"
            or any(bank.get(k) is not False for k in (
                "model_weights_or_calls_authorized", "historical_prediction_backfill_allowed", "trading_rule_authorized"))):
        raise ValueError("checked author-opinion bank only; no numeric or trade authority")
    original.check_sources(bank["source_bindings"])
    sources = {s["source_id"]: s for s in bank["sources"]}
    if len(sources) != len(bank["sources"]) or not bank["lessons"]:
        raise ValueError("complete unique forum sources and actual lessons required")
    for source in sources.values():
        if original.sha(source["body_path"]) != source["body_sha256"] or original.sha(source["receipt_path"]) != source["receipt_sha256"]:
            raise ValueError("forum body or original receipt changed")
        if not _stamp(source["published_at"]) <= _stamp(source["obtained_at"]) <= _stamp(bank["formed_at"]) <= datetime.now(timezone.utc):
            raise ValueError("forum publication, acquisition and review clocks differ")
        receipt = original.read(source["receipt_path"])
        body = Path(source["body_path"]).read_text(encoding="utf-8")
        if source["capture_kind"] == "PUBLIC_WEB_TOOL_LINE_EXTRACT":
            response = receipt["response"]
            boundary = source["author_body_line_boundaries"]
            segment = response[response.index(boundary["start"]):response.index(boundary["end"])]
            import re
            expected = "\n".join(m.group(1) for line in segment.splitlines()
                if (m := re.match(r"^L\d+: (.*)$", line))) + "\n"
            if source["url"] not in response or expected != body:
                raise ValueError("forum body differs from the original web-tool author lines")
        elif not any(item.get("url") == source["url"]
                and item.get("body_path") == source["body_path"]
                and item.get("body_sha256") == source["body_sha256"] for item in receipt.get("sources", [])):
            raise ValueError("saved author body does not match its original receipt")
    for lesson in bank["lessons"]:
        source = sources[lesson["source_id"]]
        text = Path(source["body_path"]).read_text(encoding="utf-8")
        anchor = lesson["anchor"]
        if not _stamp(source["obtained_at"]) <= _stamp(lesson["reviewed_at"]) <= _stamp(bank["formed_at"]):
            raise ValueError("lesson review cannot backdate source acquisition")
        if text[anchor["offset"]:anchor["offset"]+len(anchor["text"])] != anchor["text"]:
            raise ValueError("forum lesson must trace to the exact checked author span")
        if any(not isinstance(lesson.get(k), str) or not lesson[k].strip() for k in (
            "thinking_to_check", "expression_to_apply", "unverified_claims_not_imported")):
            raise ValueError("reasoning, expression and rejected claims must be separate")
    return bank


def project(base, explanations, facts, record):
    """Pure same-pool projection. build() additionally replays the estimator."""
    for value in (explanations, facts, record):
        if any(value.get(k) != base[k] for k in ("signal_date", "target_date")):
            raise ValueError("reasoning and original forecast dates differ")
    if explanations.get("status") != "SOURCE_REBUILT_EXACT_ORIGINAL_LOGITS":
        raise ValueError("exact original model explanation required")
    if explanations["original_knowledge_cutoff"] != base["knowledge_cutoff"]:
        raise ValueError("original model cutoff differs")
    model = _unique(base["relay"]["candidates"])
    terms = _unique(explanations["rows"])
    observed = _unique(facts["rows"])
    authored = _unique(record["rows"])
    if set(model) != set(terms) or set(model) != set(observed) or set(model) != set(authored):
        raise ValueError("reasoning requires the complete same original candidate pool")
    ranking = sorted(model, key=lambda c: (-model[c]["raw_score"], c))
    if base["relay"]["ranking"] != ranking:
        raise ValueError("do not change the original score-desc/code-asc ranking")
    for code, row in model.items():
        detail = terms[code]
        if not _finite(row["raw_score"]) or type(row["predicted_positive"]) is not bool:
            raise ValueError("original finite raw score and boolean call required")
        if row["predicted_positive"] != (row["raw_score"] > .5):
            raise ValueError("original fixed positive-class threshold changed")
        if (any(row[k] != detail[k] for k in ("name", "boards", "actual_one_word_daily", "predicted_positive"))
                or not _same(row["raw_score"], detail["raw_score"])
                or not _same(_sigmoid(detail["logit"]), row["raw_score"])
                or not _same(detail["intercept"] + sum(t["logit_contribution"] for t in detail["terms"]), detail["logit"])):
            raise ValueError("explanation must reconstruct the exact original score/call")
        if len({t["field"] for t in detail["terms"]}) != len(detail["terms"]):
            raise ValueError("duplicate explanatory term")
        if any(not _finite(t["logit_contribution"]) for t in detail["terms"]):
            raise ValueError("nonfinite explanatory term")
        if any(observed[code].get(k) != row.get(k) for k in ("name", "boards", "actual_one_word_daily")):
            raise ValueError("actual price/board identity differs from the masked original facts")
        expected_peers = sorted(k for k, other in model.items() if k != code and other["boards"] == row["boards"])
        if observed[code]["same_height_peers"] != expected_peers:
            raise ValueError("complete original same-height peers required")
        if authored[code]["promotion_judgement"] not in {"PROMOTE", "FAIL", "UNKNOWN"}:
            raise ValueError("preserve explicit actual author directions and abstentions")
    result_rows = []
    for rank, code in enumerate(ranking, 1):
        row, detail, fact, author = model[code], terms[code], observed[code], authored[code]
        termmap = {t["field"]: t for t in detail["terms"]}
        peer_codes = fact["same_height_peers"]
        pairs = []
        for peer in sorted(peer_codes, key=ranking.index):
            other = terms[peer]
            othermap = {t["field"]: t for t in other["terms"]}
            if set(othermap) != set(termmap) or not _same(detail["intercept"], other["intercept"]):
                raise ValueError("pairwise explanations must use the same frozen estimator")
            differences = [{"field": field, "label": _label(field),
                "self_signal_value": termmap[field]["signal_value"], "peer_signal_value": othermap[field]["signal_value"],
                "logit_difference": termmap[field]["logit_contribution"]-othermap[field]["logit_contribution"]}
                for field in termmap]
            delta = detail["logit"]-other["logit"]
            if not _same(sum(t["logit_difference"] for t in differences), delta):
                raise ValueError("complete peer differences do not reconstruct the model gap")
            differences.sort(key=lambda t: (-abs(t["logit_difference"]), t["field"]))
            pairs.append({"peer_code": peer, "peer_name": model[peer]["name"],
                "peer_score": model[peer]["raw_score"], "score_display_gap": 100*(row["raw_score"]-model[peer]["raw_score"]),
                "logit_gap": delta, "all_term_differences": differences,
                "proxy_differs_while_actual_one_word_same": (
                    termmap["signal_one_word"]["signal_value"] != othermap["signal_one_word"]["signal_value"]
                    and row["actual_one_word_daily"] is not None
                    and row["actual_one_word_daily"] == model[peer]["actual_one_word_daily"]),
                "market_core_absorption_or_entry_cause_confirmed": False})
        sorted_terms = sorted(detail["terms"], key=lambda t: (-abs(t["logit_contribution"]), t["field"]))
        proxy = termmap["signal_one_word"]["signal_value"]
        mismatch = (row["actual_one_word_daily"] is not None and proxy is not None
                    and bool(row["actual_one_word_daily"]) != bool(proxy))
        positive = [{**t, "label": _label(t["field"])} for t in sorted_terms if t["logit_contribution"] > 0]
        negative = [{**t, "label": _label(t["field"])} for t in sorted_terms if t["logit_contribution"] < 0]
        result_rows.append({"code": code, "name": row["name"], "boards": row["boards"], "rank": rank,
            "raw_score": row["raw_score"], "display_score": round(100*row["raw_score"], 1),
            "model_positive": row["predicted_positive"], "actual_one_word_daily": row["actual_one_word_daily"],
            "low_turnover_proxy": proxy, "proxy_actual_identity_disagrees": mismatch,
            "same_height_peer_codes": peer_codes, "all_peer_model_comparisons": pairs,
            "positive_model_terms": positive, "negative_model_terms": negative,
            "missing_model_fields": detail["missing_fields"],
            "independent_call": author["promotion_judgement"],
            "independent_model_conflict": (author["promotion_judgement"] != "UNKNOWN"
                and (author["promotion_judgement"] == "PROMOTE") != row["predicted_positive"]),
            "role_hypothesis": author["role_hypothesis"], "support": author["support"],
            "counterevidence": author["rebuttal"], "cancel": author["cancel_condition"],
            "opening_confirmation": author["opening_confirmation"], "missing_facts": author["missing"],
            "observed_price_volume": fact.get("daily_price_volume", {}),
            "provider_industry": fact.get("provider_industry"),
            "narrative_presence": fact.get("reviewed_narrative", {}).get("current_trade_narrative_presence", "UNKNOWN"),
            "role_or_cause_certified": False, "entry_selection_authorized": False})
    positive_nonword = [c for c in ranking if model[c]["predicted_positive"] and model[c]["actual_one_word_daily"] == 0]
    return {"version": VERSION, "status": "READ_ONLY_FULL_POOL_CHOICE_REASONING_NOT_NEW_FORECAST",
        "signal_date": base["signal_date"], "target_date": base["target_date"],
        "original_model_knowledge_cutoff": base["knowledge_cutoff"],
        "model_version": base["model_version"], "candidate_count": len(ranking), "ranking": ranking,
        "model_rank_first": ranking[0] if ranking else None,
        "non_one_word_positive_first": positive_nonword[0] if positive_nonword else None,
        "trade_primary": None, "trade_primary_reason": "No qualified original executable entry is supplied by this explanation",
        "market_model": base["market"], "model_exposure": record["model_exposure"], "blinding_verified": False,
        "original_pool_counts": facts.get("closing_and_failed_pool_counts", {}),
        "rows": result_rows, "score_is_calibrated_probability": False,
        "score_call_order_changed": False, "numeric_fits": 0, "new_mature_forward_samples": 0,
        "limits": ["Logit terms and their differences explain the estimator, not the market or money-flow cause",
            "Common same-day features can support model calls without distinguishing peers",
            "Old low-turnover proxy, actual daily one-word, absorption and execution are distinct",
            "Author/forum interpretation is retained separately; no target labels or retrospective new calls"]}


def _fmt(value):
    if value is None:
        return "未知"
    if _finite(value):
        return f"{value:.4g}"
    return str(value)


def _field_value(field, value):
    if value is None:
        return "未知"
    if field == "signal_one_word":
        return "触发" if value == 1 else "未触发" if value == 0 else _fmt(value)
    if field == "turnover":
        return f"{100*value:.2f}%"
    if field == "signal_amount":
        return f"{value/1e8:.2f}亿元"
    if field == "price":
        return _fmt(value)+"元"
    if field == "amplitude":
        return _fmt(value)+"%"
    return _fmt(value)


def render(report, bank):
    """Connected evidence-first prose, with all contenders and counterarguments."""
    rows = report["rows"]
    first = next((r for r in rows if r["code"] == report["model_rank_first"]), None)
    lines = ["# 模型为什么把它排在前面", "",
        f"信号日{report['signal_date']}，目标日{report['target_date']}；原模型知识截止{report['original_model_knowledge_cutoff']}。",
        "当前解释沿原分数、方向和完整候选生成；解释取得时间独立留档。", ""]
    highest = max((r["boards"] for r in rows), default=None)
    highest_names = [r["name"] for r in rows if r["boards"] == highest]
    counts = report["original_pool_counts"]
    if counts:
        lines.append(f"先看环境：原归档池封板{counts.get('closing', '未知')}只、炸板{counts.get('failed', '未知')}只；完整连板{report['candidate_count']}只。这个范围的最高组为{highest}板，包含{'、'.join(highest_names)}。最高组多股并存，不能先预设唯一赢家。")
    lines.append("原市场模型方向为" + {"WORSEN": "偏恶化", "IMPROVE": "偏改善", "CONTINUE": "偏延续"}.get(report["market_model"].get("prediction"), "未知")
        + "，输入状态为" + report["market_model"].get("input_status", "未知") + "。市场方向、个股晋级、主题身份和可成交分开审阅。")
    if first:
        lines.append(f"模型首位是{first['name']}（{first['display_score']:.1f}）。先看原模型的数值偏好，再看独立判断的支持、反证和次日确认。")
    lines += ["分数是100×原内部原始分数；不是晋级概率或交易胜率。模型排名首位与可交易主选分别判断。",
        f"非一字正向首选：{report['non_one_word_positive_first'] or '空'}；原正式交易主选：{report['trade_primary'] or '空'}。", "",
        "从限定原帖借鉴的是推理顺序：先说明环境与位置，再比较完整对手，给出支持和反解释，最后声明怎样观察会推翻判断。",
        "论坛给出的唯一赢家、固定板数上限和必然回流没有变成模型依据。", ""]
    for source in bank["sources"]:
        lines.append(f"参考原作者[{source['author']}]({source['url']})；正文发布时间{source['published_at']}，真实取得{source['obtained_at']}。")
    lines += ["", "| 原排名 | 股票 | 板位目标 | 原分数 | 模型方向 | 已登记独立方向 |", "| --- | --- | --- | ---: | --- | --- |"]
    for row in rows:
        direction = {"PROMOTE": "偏晋级", "FAIL": "偏断板", "UNKNOWN": "未知/弃判"}[row["independent_call"]]
        lines.append(f"| {row['rank']} | {row['name']} {row['code']} | {row['boards']}→{row['boards']+1} | {row['display_score']:.1f} | {'偏晋级' if row['model_positive'] else '偏断板'} | {direction} |")
    for row in rows:
        direction = {"PROMOTE": "偏晋级", "FAIL": "偏断板", "UNKNOWN": "未知/弃判"}[row["independent_call"]]
        lines += ["", f"## {row['name']} {row['code']}", "",
            f"原模型将它排第{row['rank']}，{'偏晋级' if row['model_positive'] else '偏断板'}；独立登记为{direction}。角色仍是候选假设：{row['role_hypothesis']}。"]
        observed = row["observed_price_volume"]
        actual = row["actual_one_word_daily"]
        daytype = "真实一字" if actual == 1 else "真实非一字" if actual == 0 else "一字身份未知"
        daily = [f"当日{row['boards']}板、{daytype}"]
        for field in ("turnover", "signal_amount", "amplitude"):
            if observed.get(field) is not None:
                daily.append(FIELD_NAMES[field] + _field_value(field, observed[field]))
        lines.append("盘面先看这些事实：" + "，".join(daily) + "。")
        if actual == 1:
            lines.append("这里的一字说明当日未呈现价格打开，首次打开后的供给和排队成交仍待核；模型高分不能提前回答这两个问题。")
        else:
            lines.append("换手和振幅说明参与与波动；是否消化抛压、相对对手是否主动，仍要看真实顺序和完整比较，不能由总量替代。")
        lines += ["把分数推向晋级侧的模型项主要是" + "、".join(t["label"] for t in row["positive_model_terms"][:3]) + "；压低分数的模型项主要是" + "、".join(t["label"] for t in row["negative_model_terms"][:3]) + "。各项和原截距共同决定分数，不能只挑正项解释；这些项不认证资金承接或主线地位。"]
        if row["same_height_peer_codes"]:
            pairs = row["all_peer_model_comparisons"]
            lines.append("完整同位对手：" + "、".join(p["peer_name"] for p in pairs) + "。同位可以共同晋级、共同失败或分化，模型高分不证明必然卡位。")
            closest = min(pairs, key=lambda p: (abs(p["score_display_gap"]), p["peer_code"]))
            strongest = pairs[0]
            for pair in [strongest] + ([] if closest is strongest else [closest]):
                different = [t for t in pair["all_term_differences"] if abs(t["logit_difference"]) > 1e-12]
                detail = "；".join(t["label"] + "=" + _field_value(t["field"], t["self_signal_value"]) + " / " + _field_value(t["field"], t["peer_signal_value"])
                    + ("，模型相对偏向本股" if t["logit_difference"] > 0 else "，模型相对偏向对手")
                    for t in different[:3]) or "模型输入贡献没有差异"
                lines.append(f"与{pair['peer_name']}相比，本股分数{'高' if pair['score_display_gap'] >= 0 else '低'}{abs(pair['score_display_gap']):.1f}；最大的模型差异是{detail}。完整逐项差值均已核对，不把差值当可叠加的交易评分。")
                if pair["proxy_differs_while_actual_one_word_same"]:
                    lines.append("两股真实一字身份相同，但旧低换手代理不同。这项可以解释模型排序差，不能解释成真实一字更强或承接更好。")
        else:
            lines.append("原归档池内没有同位对手；独一板位也不自动认证市场核心。")
        if row["proxy_actual_identity_disagrees"]:
            lines.append("本股真实OHLC一字身份与旧低换手代理不一致；按真实身份审阅，原代理的数值影响仍如实保留。")
        if row["missing_model_fields"]:
            lines.append("模型缺失：" + "、".join(_label(k) for k in row["missing_model_fields"]) + "；沿原训练插补，缺值不证明价格转弱。")
        lines += ["独立支持："+row["support"], "独立反证："+row["counterevidence"],
            "取消条件："+row["cancel"], "09:25："+row["opening_confirmation"]["09:25"],
            "09:35："+row["opening_confirmation"]["09:35"], "10:00："+row["opening_confirmation"]["10:00"],
            "这些是原事前条件，实际确认另留带时间来源。来源不足保持未知，观测不能回填原方向。"]
    lines += ["", "本输出没有拟合、改变原分数或增加成熟样本。解释更清楚是否带来预测增益，仍需新日期实际独立判断与完整同池结果检验。"]
    return "\n".join(lines)+"\n"


def build(base_path, explanation_path, record_path, forum_path, output, reader_policy=None, registered_path=None):
    out = Path(output)
    if out.exists():
        raise FileExistsError("new immutable reasoning output required")
    base, record = original.read(base_path), original.read(record_path)
    original.check_sources(base["source_hashes"])
    original.check_sources(record["source_bindings"])
    receipts = [r for r in original.registrations() if r["path"] == str(Path(record_path).resolve())]
    if len(receipts) != 1 or receipts[0]["sha256"] != original.sha(record_path):
        raise ValueError("original independent registered record required")
    if original.sha(record["facts_path"]) != record["facts_sha256"]:
        raise ValueError("original masked facts changed")
    facts = original.read(record["facts_path"])
    original.validate_fact_projection(facts)
    original.validate_judgements(facts, record)
    if base["feature_sha256"] != original.sha(Path(record["feature_archive"])/"features.json"):
        raise ValueError("author and original model use different feature inputs")
    bank = check_forum_bank(forum_path)
    if explanation_path is None:
        out.mkdir(parents=True)
        explanation_path = out/"original_explanation.json"
        explain(base_path, explanation_path, reader_policy=reader_policy)
    explanations = original.read(explanation_path)
    original.check_sources(explanations["source_bindings"])
    bound_base = str(Path(base_path).resolve())
    if explanations["source_bindings"].get(bound_base) != original.sha(base_path):
        raise ValueError("explanation belongs to a different saved BASE artifact")
    if registered_path:
        registered = load_registered(registered_path)
        core = registered["plan"]
        actual = _unique(core["candidates"])
        if (any(core[k] != base[k] for k in ("signal_date", "target_date"))
                or set(actual) != {r["code"] for r in base["relay"]["candidates"]}):
            raise ValueError("formal original dates or candidate denominator differ")
        for r in base["relay"]["candidates"]:
            if (not _same(actual[r["code"]].get("raw_score"), r["raw_score"])
                    or actual[r["code"]]["promotion_call"] != ("PROMOTE" if r["predicted_positive"] else "FAIL")):
                raise ValueError("formal original score or call differs")
    report = project(base, explanations, facts, record)
    if registered_path:
        report["trade_primary"] = core.get("primary_candidate")
        report["trade_primary_reason"] = "Preserved original formal selection; this explanation does not authorize or change an entry"
    report["formed_at"] = datetime.now(timezone.utc).isoformat()
    paths = [base_path, explanation_path, record_path, record["facts_path"], forum_path, __file__]
    if registered_path:
        paths.append(registered_path)
    bindings = {}
    for p in paths:
        bindings.update(original.bind(p))
    bindings |= base["source_hashes"] | record["source_bindings"] | explanations["source_bindings"] | bank["source_bindings"]
    # Recompute all terms from the actual frozen estimator; a self-consistent
    # edited decomposition cannot pass merely by keeping the same term sum.
    import joblib
    import pandas as pd
    feature_rows = _unique(original.read(Path(record["feature_archive"])/"features.json")["relay"])
    selected = _unique(explanations["rows"])
    ordered = report["ranking"]
    estimator = joblib.load(base["model_bindings"]["relay"]["model_path"])
    x = pd.DataFrame([feature_rows[c]["features"] for c in ordered])
    names, replayed, logits, scores, imputed, means, intercept = components(estimator, x, RELAY_FIELDS)
    for i, code in enumerate(ordered):
        tm = {t["field"]: t for t in selected[code]["terms"]}
        if set(tm) != set(names) or not _same(float(scores[i]), selected[code]["raw_score"]):
            raise ValueError("saved model terms do not match the actual frozen estimator")
        for j, name in enumerate(names):
            observed = feature_rows[code]["features"].get(name) if name in RELAY_FIELDS else None
            if (not _same(tm[name]["logit_contribution"], float(replayed[i, j]))
                    or not _same(tm[name]["signal_value"], observed)
                    or not _same(tm[name]["imputed_value"], float(imputed[i, j]))
                    or not _same(tm[name]["training_mean"], float(means[j]))):
                raise ValueError("individual model contribution changed")
    original.check_sources(bindings)
    report["source_bindings"] = bindings
    report["forum_expression_lessons"] = bank["lessons"]
    report["all_model_terms_replayed"] = True
    report["prediction_ledger_or_protected_report_changed"] = False
    out.mkdir(parents=True, exist_ok=True)
    original.write_new(out/"report.json", report)
    with (out/"为什么选它与逐股反证.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(render(report, bank))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "record", "forum-lessons", "output-dir"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--explanations")
    p.add_argument("--reader-policy")
    p.add_argument("--registered")
    a = p.parse_args()
    report = build(a.base, a.explanations, a.record, a.forum_lessons, a.output_dir, a.reader_policy, a.registered)
    print(json.dumps({k: report[k] for k in ("status", "candidate_count", "model_rank_first", "non_one_word_positive_first", "numeric_fits")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
