"""Independent G1 observation overlay; never alter model scores or stock calls."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

from market_mainline_analysis import normalize, sha, stamp


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_sources(sources, protocol):
    """Old executed source versions remain checkable through explicit preimages."""
    used_preimages = {}
    for filename, wanted in sources.items():
        path = Path(filename)
        if path.is_file() and sha(path) == wanted:
            continue
        saved = protocol.get("source_preimages", {}).get(str(path.resolve()))
        if (not saved or path.suffix != ".py" or saved["sha256"] != wanted
                or not Path(saved["path"]).is_file() or sha(saved["path"]) != wanted):
            raise ValueError("original source hash conflict: " + filename)
        used_preimages[filename] = saved
    return used_preimages


def load_protocol(path):
    p = read(path)
    if (p.get("version") != "regime-theme-g1-fixed-1" or p.get("arms") != ["BASE", "G1"]
            or p["rules"]["isolated_max_sealed_members_including_self"] != 2
            or p["rules"]["risk_strictly_greater_than"] != .4 or p["numerical_refits"] != 0):
        raise ValueError("unsupported G1 fixed protocol")
    from datetime import datetime, timezone
    if stamp(p["formed_at"]) > datetime.now(timezone.utc):
        raise ValueError("protocol follows real current time")
    for b, keys in ((p["model_binding"], [("manifest_path", "manifest_sha256"), ("model_path", "model_sha256")]),
                    (p["theme_policy"], [("path", "sha256")])):
        for pk, hk in keys:
            if not Path(b[pk]).is_file() or sha(b[pk]) != b[hk]:
                raise ValueError("protocol model/policy binding conflict")
    return p


def _flag(value):
    if type(value) is bool:
        return int(value)
    if type(value) in (int, float) and value in (0, 1):
        return int(value)
    return None


def selectors(rows):
    unknown_score = any(type(r.get("raw_score")) not in (int, float) or not math.isfinite(r["raw_score"]) for r in rows)
    identity_unknown = any(_flag(r.get("actual_one_word_daily")) is None for r in rows)
    first = lambda seq: seq[0]["code"] if seq else None
    nonword = [r for r in rows if _flag(r.get("actual_one_word_daily")) == 0]
    return {"all_rank_first": None if unknown_score else first(rows),
        "one_word_rank_first": None if unknown_score or identity_unknown else first([r for r in rows if _flag(r.get("actual_one_word_daily")) == 1]),
        "non_one_word_rank_first": None if unknown_score or identity_unknown else first(nonword),
        "non_one_word_positive_first": None if unknown_score or identity_unknown else first([r for r in nonword if r.get("predicted_positive") is True]),
        "selection_status": "NO_CANDIDATES" if not rows else "UNKNOWN_SCORE" if unknown_score else "UNKNOWN_IDENTITY" if identity_unknown else "KNOWN",
        "trade_primary": "原作者计划保持；取消不自动替换"}


def analyse(theme, base, policy):
    """Signal-only pure function. Does not even consult next-day truth fields."""
    if theme["signal_date"] != base["signal_date"]:
        raise ValueError("theme/BASE signal mismatch")
    if stamp(base["knowledge_cutoff"]) > stamp(theme["knowledge_cutoff"]):
        raise ValueError("original score follows theme cutoff")
    aliases = {}
    for canonical, names in policy.get("aliases", {}).items():
        for name in [canonical] + names:
            key = normalize(name)
            if key in aliases and aliases[key] != normalize(canonical):
                raise ValueError("ambiguous fixed alias")
            aliases[key] = normalize(canonical)
    canon = lambda name: aliases.get(normalize(name), normalize(name))
    exclusions = {normalize(n) for n in policy.get("excluded_names", [])}
    patterns = policy.get("excluded_patterns", [])
    facts = {}
    for row in theme["all_candidates"]:
        code = row["code"]
        if code in facts:
            raise ValueError("duplicate fact stock")
        facts[code] = row
    closing = {c for c, r in facts.items() if r["pool_kind"] == "limit_up"}
    failed = {c for c, r in facts.items() if r["pool_kind"] == "failed_board"}
    unknown_closing = sorted(c for c in closing if facts[c]["category_status"] == "UNKNOWN")
    unknown_failed = sorted(c for c in failed if facts[c]["category_status"] == "UNKNOWN")
    groups, rejected = {}, []
    for group in theme["groups"]:
        lane, name = group["evidence_lane"], canon(group["group_name"])
        if lane == "PROVIDER_SEMANTIC_FAMILY":
            spec = next((s for s in policy.get("families", {}).values() if s["source_lane"] == lane and s["group_id"] == group["group_id"]), None)
            if spec is None:
                rejected.append({"name": name, "lane": lane, "reason": "UNREGISTERED_FAMILY"})
                continue
            name = canon(spec["canonical"])
        if name in exclusions or any(re.search(pattern, name) for pattern in patterns):
            rejected.append({"name": name, "lane": lane, "reason": "GENERIC_EXCLUDED"})
            continue
        if lane not in {"PROVIDER_CURRENT_CATEGORY", "PROVIDER_SEMANTIC_FAMILY", "PROVIDER_INDUSTRY_PROXY"}:
            rejected.append({"name": name, "lane": lane, "reason": "INELIGIBLE_SYMBOL_OR_UNKNOWN_LANE"})
            continue
        codes = set(group["member_codes"])
        if not codes <= facts.keys():
            raise ValueError("theme group includes extra stock")
        key = ("INDUSTRY_PROXY" if lane == "PROVIDER_INDUSTRY_PROXY" else "PROVIDER_THEME", name)
        merged = groups.setdefault(key, {"name": name, "lane": key[0], "codes": set(), "source_lanes": set()})
        merged["codes"].update(codes)
        merged["source_lanes"].add(lane)
    # Parent memberships use fixed child definitions and observed membership only.
    parent_of = {}
    for spec in policy.get("families", {}).values():
        parent = canon(spec["canonical"])
        for child in spec.get("children", []):
            child = canon(child)
            parent_of[child] = parent
            child_group = groups.get(("PROVIDER_THEME", child))
            if child_group:
                parent_group = groups.setdefault(("PROVIDER_THEME", parent), {"name": parent, "lane": "PROVIDER_THEME", "codes": set(), "source_lanes": set()})
                parent_group["codes"].update(child_group["codes"])
                parent_group["source_lanes"].add("FIXED_CHILD_MEMBERSHIP_UNION")
    table = []
    for (lane, name), group in sorted(groups.items()):
        seals, failures = sorted(group["codes"] & closing), sorted(group["codes"] & failed)
        industry_missing = sorted(c for c in facts if not facts[c].get("industry_provider"))
        gaps = industry_missing if lane == "INDUSTRY_PROXY" else sorted(set(unknown_closing+unknown_failed))
        complete = not gaps
        ratio = len(failures)/len(seals) if seals and complete else None
        table.append({"group_name": name, "evidence_lane": lane, "source_lanes": sorted(group["source_lanes"]),
            "parent_theme": parent_of.get(name), "eligible_for_G1": lane == "PROVIDER_THEME",
            "closing_codes": seals, "failed_codes": failures, "closing_count": len(seals), "known_failed_count": len(failures),
            "first_board_count": sum(facts[c]["boards"] == 1 for c in seals),
            "relay_count": sum(facts[c]["boards"] >= 2 for c in seals),
            "board_levels": sorted({facts[c]["boards"] for c in seals}),
            "membership_coverage_complete": complete, "classification_known": len(facts)-len(gaps), "classification_total": len(facts),
            "unknown_membership_codes": gaps, "failed_over_sealed_ratio": ratio,
            "risk": "UNKNOWN" if ratio is None else "分歧风险" if ratio > .4 else "未触发",
            "ratio_definition": "炸板股票数/收盘封板股票数；唯一代码；严格大于40%",
            "first_board_change": "UNKNOWN_NO_EXACT_PREDECESSOR_COMPARISON", "confirmed_mainline": False})
    candidate_rows = base["relay"]["candidates"]
    expected = {c for c in closing if facts[c]["boards"] >= 2}
    if len(candidate_rows) != len({r["code"] for r in candidate_rows}) or {r["code"] for r in candidate_rows} != expected:
        raise ValueError("BASE relay roster differs from complete signal-day relay group")
    enriched = []
    for original in candidate_rows:
        code = original["code"]
        if original["boards"] != facts[code]["boards"]:
            raise ValueError("candidate board count conflict")
        memberships = [g for g in table if g["eligible_for_G1"] and code in g["closing_codes"]]
        ambiguous = facts[code].get("category_status") != "PROVIDER_CURRENT_CATEGORY_ONLY" or not memberships
        if unknown_closing or ambiguous:
            state = "UNKNOWN"
        else:
            state = "结构孤立" if all(g["closing_count"] <= 2 for g in memberships) else "有结构支持"
        score = original.get("raw_score")
        if type(score) not in (int, float) or not math.isfinite(score):
            score = None
        enriched.append({**original, "structure_status": state,
            "G1_excluded": state == "结构孤立", "eligible_theme_groups": [g["group_name"] for g in memberships],
            "industry_proxies": [g["group_name"] for g in table if g["evidence_lane"] == "INDUSTRY_PROXY" and code in g["closing_codes"]],
            "name_symbols": "独立观察；不属于题材筛选输入",
            "model_promotion_score": round(100*score, 1) if score is not None else None,
            "theme_risks": [{"theme": g["group_name"], "risk": g["risk"], "known_failed": g["known_failed_count"],
                              "sealed": g["closing_count"], "ratio": g["failed_over_sealed_ratio"]} for g in memberships],
            "observation_0925": "待发生：确认竞价价格、可成交性与原作者触发/取消条件",
            "observation_0935": "待发生：确认开板、承接与原预案；不改原分数",
            "observation_1000": "必要时路径确认；原主选取消不自动改选备用",
            "trigger_cancel_conditions": "沿用具体作者计划；附页不增加买入条件",
            "pending_fields": ["目标日竞价与盘口", "题材炸板完整分类"] if unknown_failed else ["目标日竞价与盘口"]})
    ranking_known = all(type(r.get("raw_score")) in (int, float) and math.isfinite(r["raw_score"]) for r in enriched)
    if ranking_known:
        enriched.sort(key=lambda r: (-r["raw_score"], r["code"]))
        for rank, row in enumerate(enriched, 1):
            if row.get("rank") is not None and row["rank"] != rank:
                raise ValueError("original rank differs from frozen V280 ordering")
    else:
        enriched.sort(key=lambda r: r["code"])
    filtered = [r for r in enriched if not r["G1_excluded"]]
    return {"status": "G1_SHADOW_ONLY_NO_SCORE_CHANGE", "signal_date": theme["signal_date"],
        "target_date": base["target_date"], "knowledge_cutoff": theme["knowledge_cutoff"],
        "model_version": base.get("model_version"), "pool_counts": theme["pool_counts"],
        "all_candidates": enriched, "full_theme_risk_table": table, "excluded_generic_groups": rejected,
        "original_model_top5": enriched[:5] if ranking_known else [],
        "G1_shadow_top5": filtered[:5] if ranking_known else [],
        "original_selectors": selectors(enriched), "G1_shadow_selectors": selectors(filtered),
        "unknown_closing_categories": unknown_closing, "unknown_failed_categories": unknown_failed,
        "ranking_status": "KNOWN" if ranking_known else "UNKNOWN_SCORE",
        "model_or_prediction_changed": False, "original_complete_candidate_count": len(enriched),
        "calibrated_probability": None, "new_forward_samples": 0, "prediction_effect_verified": False,
        "accuracy_change": "NO_ACCURACY_CHANGE", "automatic_entry_allowed": False}


def run(theme_path, base_path, protocol_path, output):
    from datetime import datetime, timezone
    p = load_protocol(protocol_path)
    theme, base = read(theme_path), read(base_path)
    cutoff = datetime.now(timezone.utc).isoformat()
    if max(stamp(theme["generated_at"]), stamp(base["generated_at"]), stamp(theme["knowledge_cutoff"])) > stamp(cutoff):
        raise ValueError("overlay input follows actual current cutoff")
    saved = verify_sources(theme.get("source_hashes", {}), p)
    saved.update(verify_sources(base.get("source_hashes", {}), p))
    binding = theme["original_base_prediction_binding"]
    if Path(binding["path"]).resolve() != Path(base_path).resolve() or binding["sha256"] != sha(base_path):
        raise ValueError("theme report does not bind supplied original BASE")
    mb = base["model_bindings"]["relay"]
    if mb["model_sha256"] != p["model_binding"]["model_sha256"] or mb["sha256"] != p["model_binding"]["manifest_sha256"]:
        raise ValueError("frozen G1 model changed; no automatic migration")
    result = analyse(theme, base, read(p["theme_policy"]["path"]))
    result.update(generated_at=cutoff, source_signal_context_cutoff=theme["knowledge_cutoff"],
        original_prediction_knowledge_cutoff=base["knowledge_cutoff"], knowledge_cutoff=cutoff,
        protocol_binding={"path": str(Path(protocol_path).resolve()), "sha256": sha(protocol_path)},
        model_binding=p["model_binding"], theme_policy=p["theme_policy"], preserved_executed_sources=saved,
        prospective_eligible=stamp(base["knowledge_cutoff"]) > stamp(p["formed_at"]),
        source_hashes={str(Path(path).resolve()): sha(path) for path in (theme_path, base_path, protocol_path, __file__, p["theme_policy"]["path"])})
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    (directory/"report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    import csv
    with (directory/"theme_risks.csv").open("x", encoding="utf-8-sig", newline="") as f:
        cols = ["group_name", "evidence_lane", "parent_theme", "closing_count", "first_board_count", "relay_count", "known_failed_count", "classification_known", "classification_total", "failed_over_sealed_ratio", "risk"]
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); writer.writeheader(); writer.writerows(result["full_theme_risk_table"])
    lines = ["# 周期与题材G1影子观察附页", "", f"信号日{result['signal_date']}，目标日{result['target_date']}；实际形成{cutoff}。",
        "原分数、排序、晋级判断和交易主选保持。内部原始分数×100仅为显示尺度。", "",
        "逐题材风险使用炸板股票数÷收盘封板股票数，严格大于40%；覆盖不足显示未知。"]
    for title, key, sel in (("原模型前5", "original_model_top5", "original_selectors"), ("G1影子前5", "G1_shadow_top5", "G1_shadow_selectors")):
        lines += ["", "## "+title, "", "|股票|代码|板位|原始分数|显示分|原判断|结构|分歧提示|", "|---|---|---:|---:|---:|---|---|---|"]
        for r in result[key]:
            risks = "；".join(g["theme"]+":"+g["risk"] for g in r["theme_risks"])
            lines.append(f"|{r.get('name')}|{r['code']}|{r['boards']}|{r.get('raw_score')}|{r['model_promotion_score']}|{r.get('promotion_call', r.get('predicted_positive'))}|{r['structure_status']}|{risks or '未知'}|")
        s = result[sel]
        lines += ["", f"模型首选{s['all_rank_first'] or '未知/空'}；一字首选{s['one_word_rank_first'] or '未知/空'}；非一字排名首选{s['non_one_word_rank_first'] or '未知/空'}；非一字正向首选{s['non_one_word_positive_first'] or '空/未知'}。",
            "09:25确认竞价、可成交性及原触发/取消条件；09:35确认承接与开板；必要时10:00路径确认。均待发生；原主选取消不自动换备用。"]
    lines += ["", "全部原候选及来源/模型哈希、截止、完整风险表见report.json；题材未知不筛除，行业代理与名称符号分别保留。",
        "本附页为已冻结日期的功能回放，不是新预测或独立前向；全候选Accuracy没有变化，预测效果尚待固定120日验证。"]
    (directory/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result
