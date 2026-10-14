"""Read-only BASE diagnostics; no fitting, rescoring or prediction changes.

Signal diagnostics project a fixed allowlist before inspecting values. Historical
truth is accessed only after the exact official target close has matured.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MARKET_CLASSES = ("IMPROVE", "CONTINUE", "WORSEN")
MARKET_BASE = ["limit_count", "limit_change", "seal_rate", "failed_rate", "max_board",
    "height_change", "middle_density", "ladder_gap_count", "unique_space", "one_word_share",
    "breadth_height_divergence", "promotion_overall", "promotion_mid", "yesterday_premium",
    "previous_limit_tail_loss", "down_limit_count", "breadth_health", "relay_appetite",
    "ladder_health", "loss_effect_health", "regime_fragility", "emotion_quality_score"]
MARKET_FIELDS = MARKET_BASE + [name for key in ["emotion_quality_score", "promotion_mid", "yesterday_premium", "failed_rate", "down_limit_count"]
    for name in (key+"_delta_1", key+"_delta_3", key+"_mean_5")] + [
    "current_REPAIR", "current_RISK_OFF", "current_RISK_ON"] + [name for key in ["sh_a_return_pct", "sz_a_return_pct"]
    for name in (key, key+"_lag_1", key+"_lag_3")]
RELAY_FIELDS = ["boards", "height_gap", "same_height_rivals", "unique_space", "height_share", "is_space_leader",
    "emotion_score", "promotion_2to3", "promotion_3to4", "promotion_mid", "previous_limit_tail_loss",
    "failed_rate", "down_limit_rate", "breadth_health", "ladder_health", "loss_effect_health",
    "crowding_risk", "regime_fragility", "turnover", "signal_amount", "amplitude", "turnover_vs_prev", "price",
    "price_same_height_percentile", "signal_amount_same_height_percentile", "turnover_same_height_percentile",
    "turnover_vs_prev_same_height_percentile", "signal_one_word", "log_signal_amount"]
EXTRA_MARKET_FACTS = ("nonpositive_premium_promotion_down", "nonpositive_premium_failure_up")
FOLDS = [{"id": "2025H2", "start": "2025-07-01", "end": "2025-12-31"},
         {"id": "2026JanAug", "start": "2026-01-01", "end": "2026-08-31"},
         {"id": "2026Sep", "start": "2026-09-01", "end": "2026-09-30"}]


def now():
    return datetime.now(timezone.utc).isoformat()


def stamp(value):
    d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise ValueError("timezone-aware knowledge cutoff required")
    return d


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    number = _number(value)
    if number in (0., 1.):
        return bool(number)
    return None


def _one_word(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    number = _number(value)
    return int(number) if number in (0., 1.) else None


def _manifest_fields(manifest, module):
    fields = list(manifest.get("fields", []))
    allowed = MARKET_FIELDS if module == "market" else RELAY_FIELDS
    if len(fields) != len(set(fields)) or set(fields) - set(allowed):
        raise ValueError("manifest includes fields outside original BASE signal allowlist")
    if manifest.get("module", module) != module or manifest.get("arm", "BASE") != "BASE":
        raise ValueError("diagnostics require original BASE manifest")
    return fields


def _features(source, fields):
    # Never copy or iterate all input fields: target/exit/future fields are unread.
    return {key: _number(source.get(key)) for key in fields}


def _base(value):
    return value.get("arms", {}).get("BASE", {}) if "arms" in value else value


def signal_diagnostics(featurepack, market_base, relay_base, market_manifest, relay_manifest, asof=None):
    """Pure signal-only facts, preserving original scores, calls and V280 ties."""
    market_fields = _manifest_fields(market_manifest, "market")
    relay_fields = _manifest_fields(relay_manifest, "relay")
    market_features = _features(featurepack.get("market", {}).get("features", {}), market_fields)
    market_extra = _features(featurepack.get("market", {}).get("features", {}), EXTRA_MARKET_FACTS)
    mb, rb = _base(market_base), _base(relay_base)
    prediction = mb.get("prediction", mb.get("predicted"))
    prediction = prediction if prediction in MARKET_CLASSES else None
    raw = mb.get("raw_scores", {})
    market_scores = {key: _number(raw.get(key)) for key in MARKET_CLASSES}
    roster, base_rows = featurepack.get("relay", []), rb.get("candidates", [])
    facts = {}
    for row in roster:
        code = row.get("code")
        if not isinstance(code, str) or len(code) != 6 or not code.isdigit() or code in facts:
            raise ValueError("invalid or duplicate original candidate code")
        boards = _number(row.get("boards"))
        if boards is None or boards < 2 or not boards.is_integer():
            raise ValueError("invalid original board height")
        facts[code] = {"code": code, "name": row.get("name"), "boards": int(boards),
            "actual_one_word_daily": _one_word(row.get("actual_one_word_daily")),
            "features": _features(row.get("features", {}), relay_fields)}
    scores = {}
    for row in base_rows:
        code = row.get("code")
        if code in scores or code not in facts:
            raise ValueError("BASE output changes complete original candidate scope")
        scores[code] = {"raw_score": _number(row.get("raw_score", row.get("probability"))),
            "predicted_positive": _boolean(row.get("predicted_positive", row.get("predicted"))),
            "boards": _number(row.get("boards")), "actual_one_word_daily": _one_word(row.get("actual_one_word_daily")),
            "original_rank": _number(row.get("rank", row.get("original_rank")))}
    if set(scores) != set(facts):
        raise ValueError("BASE output omits original candidates, including unknowns")
    ordered = sorted(facts, key=lambda c: (scores[c]["raw_score"] is None,
        -(scores[c]["raw_score"] or 0.), c))
    subset_rank = {code: i+1 for i, code in enumerate(ordered) if scores[code]["raw_score"] is not None}
    score_scope_complete = all(scores[c]["raw_score"] is not None for c in facts)
    rank = subset_rank if score_scope_complete else {}
    counts = {}
    for item in facts.values():
        counts[item["boards"]] = counts.get(item["boards"], 0)+1
    candidates = []
    for code in ordered:
        fact, score = facts[code], scores[code]
        issues = []
        missing = [key for key, value in fact["features"].items() if value is None]
        if missing:
            issues.append("MODEL_INPUT_MISSING_OR_IMPUTED")
        word = fact["actual_one_word_daily"]
        if word is None:
            issues.append("ACTUAL_OHLC_ONE_WORD_UNKNOWN")
        elif word == 1:
            issues.append("ONE_WORD_DAILY_DOES_NOT_PROVE_ENTRY_FILL")
        proxy = fact["features"].get("signal_one_word")
        if word is not None and proxy in (0., 1.) and word != proxy:
            issues.append("LEGACY_PROXY_DIFFERS_FROM_ACTUAL_OHLC_DEFINITION")
        rivals = fact["features"].get("same_height_rivals")
        if rivals is not None and rivals != counts[fact["boards"]]-1:
            issues.append("SAME_HEIGHT_COHORT_COUNT_CONFLICT")
        if score["boards"] is not None and score["boards"] != fact["boards"]:
            issues.append("BASE_BOARD_METADATA_CONFLICT")
        if score["actual_one_word_daily"] is not None and word is not None and score["actual_one_word_daily"] != word:
            issues.append("BASE_OHLC_METADATA_CONFLICT")
        if score_scope_complete and score["original_rank"] is not None and score["original_rank"] != rank.get(code):
            raise ValueError("bound original rank conflicts with unchanged BASE score/code ranking")
        value, call = score["raw_score"], score["predicted_positive"]
        candidates.append({"code": code, "name": fact["name"], "boards": fact["boards"],
            "raw_score": value, "model_promotion_score": round(value*100, 1) if value is not None else None,
            "promotion_call": "PROMOTE" if call is True else "FAIL" if call is False else "UNKNOWN",
            "predicted_positive": call, "rank": rank.get(code), "bound_original_rank": score["original_rank"],
            "known_score_subset_position": subset_rank.get(code),
            "actual_one_word_daily": word, "legacy_signal_one_word": proxy,
            "signal_facts": fact["features"],
            "missing_model_fields": missing, "fact_prompts": issues, "theme": None, "identity": None,
            "theme_status": "UNKNOWN", "identity_status": "UNKNOWN", "execution_status": "UNVERIFIED"})
    eligible = [c for c in candidates if c["raw_score"] is not None]
    nonword = [c for c in eligible if c["actual_one_word_daily"] == 0]
    one_word = [c for c in eligible if c["actual_one_word_daily"] == 1]
    positive = [c for c in eligible if c["predicted_positive"] is True]
    nonword_positive = [c for c in nonword if c["predicted_positive"] is True]
    word_scope_complete = all(c["actual_one_word_daily"] is not None for c in candidates)
    call_scope_complete = all(c["predicted_positive"] is not None for c in candidates)
    first = lambda rows: rows[0]["code"] if rows else None
    premium = market_features.get("yesterday_premium")
    market_prompts = []
    if prediction == "IMPROVE" and premium is not None and premium <= 0:
        market_prompts.append("CURRENT_NONPOSITIVE_FEEDBACK_WITH_NEXT_SESSION_IMPROVE_CALL")
    return {"status": "SIGNAL_ONLY_DIAGNOSTIC_NO_PREDICTION_CHANGE", "signal_date": featurepack.get("signal_date"),
        "target_date": featurepack.get("target_date"), "knowledge_cutoff": asof,
        "market": {"original_prediction": prediction, "original_raw_scores": market_scores,
            "signal_facts": market_features,
            "missing_model_fields": [k for k, v in market_features.items() if v is None],
            "current_nonpositive_premium": premium <= 0 if premium is not None else None,
            "existing_exact_previous_flags": market_extra, "fact_prompts": market_prompts,
            "prompt_scope": "当日事实与次日判断的复核提示，不是逻辑否决或修改预测"},
        "candidates": candidates, "all_rank_first": first(eligible) if score_scope_complete else None,
        "all_rank_status": "KNOWN" if score_scope_complete and eligible else "EMPTY" if not candidates else "UNKNOWN_SCORE",
        "positive_model_primary": first(positive) if score_scope_complete and call_scope_complete else None,
        "known_score_subset_rank_first": first(eligible),
        "one_word_rank_first": first(one_word) if word_scope_complete and score_scope_complete else None,
        "one_word_rank_status": ("UNKNOWN_SCORE" if not score_scope_complete else
                                  "UNKNOWN_IDENTITY" if not word_scope_complete else
                                  "KNOWN" if one_word else "EMPTY"),
        "non_one_word_rank_first": first(nonword) if word_scope_complete and score_scope_complete else None,
        "non_one_word_positive_first": first(nonword_positive) if word_scope_complete and score_scope_complete and call_scope_complete else None,
        "non_one_word_primary_status": ("UNKNOWN_SCORE" if not score_scope_complete else
                                        "UNKNOWN_IDENTITY" if not word_scope_complete else
                                        "UNKNOWN_MODEL_CALL" if not call_scope_complete else
                                        "KNOWN" if nonword_positive else "EMPTY"),
        "known_nonword_subset_rank_first": first(nonword),
        "known_nonword_subset_positive_first": first(nonword_positive),
        "known_subset_scope": "仅已知字段子集观察；不是完整组首选或交易主选",
        "non_one_word_scope_complete": word_scope_complete, "raw_score_scope_complete": score_scope_complete,
        "ranking_rule": "raw_score descending; code ascending; missing raw_score has unknown rank",
        "score_scope": "100×BASE内部原始晋级分数；未校准；不是盈利概率或买入阈值",
        "model_or_prediction_changed": False, "accuracy_change": "NO_ACCURACY_CHANGE",
        "trading_primary_status": "保留原作者交易计划；本附页不计算交易主选",
        "automatic_entry_allowed": False}


def _calendar(path):
    calendar = _read(path)
    sessions = calendar.get("sessions", [])
    if calendar.get("status") != "verified" or not calendar.get("source") or sessions != sorted(set(sessions)):
        raise ValueError("verified unique official calendar required")
    return calendar


def theme_ladder_diagnostics(closing_pool, failed_pool, context_view, signal_date, asof,
                             base_candidates=None):
    """Count signal-day provider groups; never score themes or alter BASE calls.

    The battery family is a descriptive union of category/industry names that
    contain 电池. 储能 and 新能源 are not added. Raw category groups remain intact.
    Current categories do not establish an active trading theme or a mainline.
    """
    cutoff = stamp(asof)
    if cutoff < stamp(signal_date+"T15:00:00+08:00"):
        raise ValueError("theme appendix requires a completed signal session")
    facts = {}
    for pool, kind in ((closing_pool, "limit_up"), (failed_pool, "failed_board")):
        if (pool.get("date") != signal_date or pool.get("kind") != kind
                or pool.get("capture_complete_within_provider_response") is not True):
            raise ValueError("exact-date complete provider closing/failed pools required")
        if stamp(pool["available_at"]) > cutoff:
            raise ValueError("pool input follows diagnostic cutoff")
        for row in pool["rows"]:
            code = row.get("code")
            if not isinstance(code, str) or len(code) != 6 or not code.isdigit() or code in facts:
                raise ValueError("duplicate/invalid code or closing and failed pools overlap")
            boards = row.get("boards") if kind == "limit_up" else None
            if kind == "limit_up" and (type(boards) is not int or boards < 1):
                raise ValueError("invalid closing board height")
            industry = row.get("industry")
            facts[code] = {"code": code, "name": row.get("name"), "boards": boards,
                "pool_kind": kind, "industry_provider": industry if isinstance(industry, str) and industry.strip() else None,
                "category_status": "UNKNOWN", "category_missing_reason": "NO_CURRENT_CATEGORY_ROW",
                "provider_categories": [], "active_trading_theme_verified": False}
    if stamp(context_view["knowledge_cutoff"]) != cutoff:
        raise ValueError("context view uses a different diagnostic cutoff")
    seen_context = set()
    for row in context_view["candidates"]:
        code = row["code"]
        if code in seen_context or code not in facts:
            raise ValueError("duplicate/extra context candidate")
        seen_context.add(code)
        category = row.get("categories")
        if category is None:
            facts[code]["category_missing_reason"] = row.get("unavailable", {}).get("categories", "MISSING")
            continue
        available = stamp(category["available_at"])
        if available > cutoff:
            raise ValueError("category input follows diagnostic cutoff")
        if (cutoff-available).total_seconds() > 86400:
            facts[code]["category_missing_reason"] = "CAPTURE_OLDER_THAN_24H_POLICY"
            continue
        if available.astimezone(stamp(signal_date+"T00:00:00+08:00").tzinfo).date().isoformat() != signal_date:
            facts[code]["category_missing_reason"] = "CATEGORY_NOT_OBSERVED_ON_SIGNAL_DATE"
            continue
        identities = set()
        projected = []
        for item in category["rows"]:
            identity, name = item.get("board_code"), item.get("board_name")
            if not isinstance(identity, str) or not isinstance(name, str) or not name.strip() or identity in identities:
                raise ValueError("invalid/duplicate provider category")
            identities.add(identity)
            projected.append({"board_code": identity, "board_name": name})
        facts[code].update(category_status="PROVIDER_CURRENT_CATEGORY_ONLY",
            category_missing_reason=None, provider_categories=projected)
    calls = {}
    if base_candidates is not None:
        for row in base_candidates:
            code = row.get("code")
            if code in calls or code not in facts or facts[code]["pool_kind"] != "limit_up" or facts[code]["boards"] < 2:
                raise ValueError("BASE roster differs from complete original relay candidates")
            if row.get("boards") != facts[code]["boards"]:
                raise ValueError("BASE board metadata differs from signal pool")
            calls[code] = {"predicted_positive": _boolean(row.get("predicted_positive")),
                "original_raw_score": _number(row.get("raw_score")), "original_rank": _number(row.get("rank"))}
        expected = {c for c, r in facts.items() if r["pool_kind"] == "limit_up" and r["boards"] >= 2}
        if set(calls) != expected:
            raise ValueError("BASE omits original relay candidates")
    groups = {}
    for code, row in facts.items():
        memberships = []
        if row["industry_provider"]:
            memberships.append(("PROVIDER_INDUSTRY_PROXY", row["industry_provider"], row["industry_provider"]))
        memberships.extend(("PROVIDER_CURRENT_CATEGORY", item["board_code"], item["board_name"])
                           for item in row["provider_categories"])
        family_basis = []
        for lane, identity, name in memberships:
            key = (lane, identity)
            if key in groups and groups[key]["name"] != name:
                raise ValueError("provider category identity has conflicting names")
            groups.setdefault(key, {"lane": lane, "name": name, "codes": set()})["codes"].add(code)
            if "电池" in name:
                family_basis.append(lane)
        if family_basis:
            key = ("PROVIDER_SEMANTIC_FAMILY", "battery")
            groups.setdefault(key, {"lane": key[0], "name": "电池", "codes": set()})["codes"].add(code)
        row["battery_family_membership_basis"] = sorted(set(family_basis))
        row["original_base"] = calls.get(code)
    unknown = sorted(c for c, r in facts.items() if r["category_status"] == "UNKNOWN")
    relay_denominator = sum(r["pool_kind"] == "limit_up" and r["boards"] >= 2 for r in facts.values())
    group_rows = []
    for (lane, identity), group in sorted(groups.items()):
        codes = sorted(group["codes"])
        closing = [c for c in codes if facts[c]["pool_kind"] == "limit_up"]
        first = [c for c in closing if facts[c]["boards"] == 1]
        relay = [c for c in closing if facts[c]["boards"] >= 2]
        failed = [c for c in codes if facts[c]["pool_kind"] == "failed_board"]
        levels = sorted({facts[c]["boards"] for c in closing})
        negative = [c for c in relay if calls.get(c, {}).get("predicted_positive") is False]
        group_rows.append({"evidence_lane": lane, "group_id": identity, "group_name": group["name"],
            "member_codes": codes, "closing_codes": closing, "first_board_codes": first,
            "relay_codes": relay, "failed_codes": failed, "closing_count": len(closing),
            "first_board_count": len(first), "relay_count": len(relay), "failed_count": len(failed),
            "board_levels": levels, "max_board": max(levels) if levels else None,
            "ladder_contiguous": bool(levels and levels == list(range(1, max(levels)+1))),
            "has_relay_and_first_board_support": bool(relay and first),
            "relay_count_over_original_relay_count": [len(relay), relay_denominator],
            "original_negative_relay_codes": negative,
            "all_known_original_relay_calls_negative": bool(relay and len(negative) == len(relay)),
            "category_unknown_codes": unknown, "failed_membership_coverage_complete": not any(facts[c]["pool_kind"] == "failed_board" for c in unknown),
            "active_trading_theme_verified": False, "mainline_confirmation_authorized": False})
    return {"status": "SIGNAL_ONLY_THEME_CONTEXT_NO_PREDICTION_CHANGE", "signal_date": signal_date,
        "knowledge_cutoff": asof, "scope": "PROVIDER_MAINBOARD_NAME_NON_ST_CLOSING_AND_FAILED_POOLS_NOT_EXCHANGE_CERTIFIED",
        "official_full_market": False, "all_candidates": [facts[c] for c in sorted(facts)],
        "pool_counts": {"closing": len(closing_pool["rows"]), "failed": len(failed_pool["rows"]),
            "relay": relay_denominator, "first_board": len(closing_pool["rows"])-relay_denominator},
        "category_unknown_codes": unknown, "groups": group_rows,
        "semantic_family_definition": {"battery": "分类名称或供应商行业名称含电池；股票去重；不纳入仅储能或新能源的成员"},
        "group_order": "evidence_lane and provider identity lexical order; no strength ranking",
        "model_or_prediction_changed": False, "new_forward_samples": 0,
        "prediction_effect_verified": False, "accuracy_change": "NO_ACCURACY_CHANGE",
        "automatic_entry_allowed": False,
        "limits": ["供应商当前分类与行业代理不是主动交易题材、业务真实性或主线身份认证",
            "未知分类不等于非成员；已知炸板数不代表完整同题材炸板数量",
            "语义族定义在本次诊断固定；本次已见数据，不是独立盲测或预测改善"]}


def theme_from_archives(pool_archive, context_archive, signal_date, output, asof=None, base_predictions=None):
    """Reconstruct existing receipts and issue a new immutable context appendix."""
    import public_outcome_sources_v293 as outcome
    import point_context_v295 as context
    import point_context_reconciled_v295 as reconciled
    cutoff = asof or now()
    if stamp(cutoff) > stamp(now()):
        raise ValueError("diagnostic cutoff is in the future")
    original_manifest = _read(Path(pool_archive)/"manifest.json")
    if original_manifest.get("target_date", "9999-12-31") > signal_date:
        raise ValueError("pool archive contains a target session after signal date")
    manifest, pools, _daily, _failures = outcome.reconstruct(pool_archive)
    if stamp(manifest["recorded_at"]) > stamp(cutoff):
        raise ValueError("pool archive formed after diagnostic cutoff")
    if signal_date not in manifest["calendar"]["sessions"]:
        raise ValueError("signal date absent from official calendar")
    archive = Path(context_archive)
    cm = _read(archive / "manifest.json")
    current = (reconciled.reconstruct(archive) if cm["version"] == reconciled.VERSION else context.reconstruct(archive))
    if stamp(current["recorded_at"]) > stamp(cutoff):
        raise ValueError("context archive formed after diagnostic cutoff")
    scope = current["scope"]
    if (scope["price_signal_date"] != signal_date
            or Path(scope["source_path"]).resolve() != Path(pool_archive).resolve()
            or scope.get("manifest_sha256") != sha(Path(pool_archive)/"manifest.json")):
        raise ValueError("context is not bound to this signal pool archive")
    closing, failed = pools.get((signal_date, "limit_up")), pools.get((signal_date, "failed_board"))
    if closing is None or failed is None:
        raise ValueError("both exact signal-day closing and failed pools are required")
    if set(scope["codes"]) != {r["code"] for r in closing["rows"]} or set(scope["codes"]) != {r["code"] for r in current["candidates"]}:
        raise ValueError("context does not preserve original complete closing roster")
    sources = {str(Path(p).resolve()): sha(p) for p in [Path(pool_archive)/"manifest.json", Path(pool_archive)/"normalized.json", archive/"manifest.json", archive/"context.json", __file__]}
    for request in manifest["requests"]:
        if request.get("raw_path"):
            sources[str(Path(request["raw_path"]).resolve())] = request["raw_sha256"]
    base = None
    if base_predictions:
        base = _read(base_predictions)
        if base.get("signal_date") != signal_date or base.get("production_model_changed") is not False:
            raise ValueError("original unchanged BASE predictions required for same signal date")
        if max(stamp(base["knowledge_cutoff"]), stamp(base["generated_at"])) > stamp(cutoff):
            raise ValueError("BASE prediction follows diagnostic cutoff")
        for path, wanted in base.get("source_hashes", {}).items():
            if not Path(path).is_file() or sha(path) != wanted:
                raise ValueError("original BASE source hash mismatch")
        sources[str(Path(base_predictions).resolve())] = sha(base_predictions)
    result = theme_ladder_diagnostics(closing, failed, context.asof_view(current, cutoff),
        signal_date, cutoff, base["relay"]["candidates"] if base else None)
    result["generated_at"] = now()
    result["source_hashes"] = sources
    result["original_base_prediction_binding"] = ({"path": str(Path(base_predictions).resolve()),
        "sha256": sha(base_predictions), "model_version": base.get("model_version"),
        "knowledge_cutoff": base["knowledge_cutoff"], "target_date": base.get("target_date")} if base else None)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    _write(directory/"report.json", result)
    return result


def _bound_history(root, module, asof):
    directory = Path(root) / (module+"_history")
    report_path, source = directory / "report.json", directory / "predictions.csv"
    report = _read(report_path)
    if report.get("module") != module or report.get("folds") != FOLDS or report.get("arms") != ["BASE", "C1", "C2"]:
        raise ValueError("historical report scope differs from fixed original comparison")
    if sha(source) != report["predictions_sha256"]:
        raise ValueError("saved BASE source CSV hash differs from report")
    if stamp(report["formed_at"]) > stamp(asof):
        raise ValueError("historical output formed after actual diagnostic cutoff")
    manifest_path = Path(root) / "models" / module / "BASE/manifest.json"
    manifest = _read(manifest_path)
    fields = _manifest_fields(manifest, module)
    model_path = Path(manifest["model_path"])
    if sha(model_path) != manifest["sha256"]:
        raise ValueError("BASE model artifact hash differs from manifest")
    allowed = set(fields + ["date", "target_date", "status", "label_valid", "target", "predicted", "confidence",
        "raw_score", "probability", "arm", "fold", "code", "name", "boards", "actual_one_word_daily",
        "consecutive_count", "data_status", "market_data_status", "one_word_proxy", "emotion_state"] +
        ["raw_score_"+c for c in MARKET_CLASSES])
    frame = pd.read_csv(source, dtype={"code": str}, usecols=lambda c: c in allowed)
    frame = frame[frame.arm.eq("BASE")].copy()
    keys = ["fold", "date"] + (["code"] if module == "relay" else [])
    if frame.duplicated(keys).any():
        raise ValueError("duplicate BASE historical keys")
    return frame, report, manifest, {str(p.resolve()): sha(p) for p in [source, report_path, manifest_path, model_path]}


def _truth(row, module, day, next_day, cutoff):
    if next_day is None:
        return None, "OFFICIAL_TARGET_UNAVAILABLE"
    if stamp(next_day+"T15:00:00+08:00") > stamp(cutoff):
        return None, "TARGET_NOT_MATURED"
    if row is None:
        return None, "SOURCE_ROW_UNAVAILABLE"
    if row.get("target_date") != next_day:
        return None, "TARGET_NOT_EXACT_OFFICIAL_NEXT_SESSION"
    if _boolean(row.get("label_valid")) is not True:
        return None, "LABEL_UNKNOWN"
    value = row.get("target")
    if module == "market":
        return (value, "KNOWN") if value in MARKET_CLASSES else (None, "LABEL_UNKNOWN")
    boolean = _boolean(value)
    return (boolean, "KNOWN") if boolean is not None else (None, "LABEL_UNKNOWN")


def _classification(rows, module):
    known = [r for r in rows if r["evaluation_status"] == "SCORED"]
    result = {"original_rows": len(rows), "scored_rows": len(known), "accuracy":
        sum(r["error"] is False for r in known)/len(known) if known else None,
        "unknown_or_pending_rows": len(rows)-len(known), "calibrated_brier": None}
    if module == "relay":
        tp = sum(r["original_prediction"] is True and r["truth"] is True for r in known)
        tn = sum(r["original_prediction"] is False and r["truth"] is False for r in known)
        fp = sum(r["original_prediction"] is True and r["truth"] is False for r in known)
        fn = sum(r["original_prediction"] is False and r["truth"] is True for r in known)
        positives = sum(r["truth"] is True for r in rows)
        precision = tp/(tp+fp) if tp+fp else None
        complete_recall = tp/positives if positives else None
        f1 = (2*precision*complete_recall/(precision+complete_recall) if precision+complete_recall else 0.) \
            if precision is not None and complete_recall is not None else None
        result.update({"TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "scored_FN": fn, "complete_positive_misses": positives-tp,
            "truth_positive_support": positives,
            "precision": precision, "complete_positive_recall": complete_recall,
            "F1": f1})
    else:
        result["confusion"] = {a: {b: sum(r["truth"] == a and r["original_prediction"] == b for r in known)
                                  for b in MARKET_CLASSES} for a in MARKET_CLASSES}
        result["by_class"] = {}
        for cls in MARKET_CLASSES:
            support = sum(r["truth"] == cls for r in rows)
            tp = sum(r["truth"] == cls and r["original_prediction"] == cls for r in known)
            result["by_class"][cls] = {"truth_support": support,
                "predicted_calls": sum(r["original_prediction"] == cls for r in rows), "TP": tp,
                "complete_recall": tp/support if support else None}
    return result


def historical(root, calendar_path, output, protocol_path=None, asof=None):
    """New immutable error appendix from saved BASE rows; does not run models."""
    cutoff = asof or now()
    calendar = _calendar(calendar_path)
    sessions = calendar["sessions"]
    successor = dict(zip(sessions[:-1], sessions[1:]))
    market, mreport, mm, sources = _bound_history(root, "market", cutoff)
    relay, rreport, rm, bound = _bound_history(root, "relay", cutoff)
    sources.update(bound)
    sources[str(Path(calendar_path).resolve())] = sha(calendar_path)
    if mreport["protocol_sha256"] != rreport["protocol_sha256"]:
        raise ValueError("market and relay historical reports belong to different protocols")
    if protocol_path:
        if sha(protocol_path) != mreport["protocol_sha256"]:
            raise ValueError("saved historical protocol hash mismatch")
        sources[str(Path(protocol_path).resolve())] = sha(protocol_path)
    sources[str(Path(__file__).resolve())] = sha(__file__)
    error_rows, candidate_rows, folds = [], [], {}
    for fold in FOLDS:
        days = [d for d in sessions if fold["start"] <= d <= fold["end"]]
        by_market = {r["date"]: r for r in market[market.fold.eq(fold["id"])].to_dict("records")}
        by_relay = {d: g.to_dict("records") for d, g in relay[relay.fold.eq(fold["id"])].groupby("date")}
        market_errors, stock_errors = [], []
        for day in days:
            target = successor.get(day)
            mr = by_market.get(day)
            raw_candidates = by_relay.get(day, [])
            pack = {"signal_date": day, "target_date": target,
                "market": {"features": {k: mr.get(k) if mr else None for k in mm["fields"]}},
                "relay": [{"code": r["code"], "name": r.get("name"), "boards": r["boards"],
                    "actual_one_word_daily": r.get("actual_one_word_daily"),
                    "features": {k: r.get(k) for k in rm["fields"]}} for r in raw_candidates]}
            mb = {"prediction": mr.get("predicted") if mr else None,
                  "raw_scores": {c: mr.get("raw_score_"+c) if mr else None for c in MARKET_CLASSES}}
            rb = {"candidates": [{"code": r["code"], "boards": r["boards"],
                "actual_one_word_daily": r.get("actual_one_word_daily"), "raw_score": r.get("raw_score"),
                "predicted_positive": r.get("predicted") if r.get("status") == "PREDICTED_UNCALIBRATED" else None}
                for r in raw_candidates]}
            diag = signal_diagnostics(pack, mb, rb, mm, rm, cutoff)
            truth, reason = _truth(mr, "market", day, target, cutoff)
            call = diag["market"]["original_prediction"] if mr and mr.get("status") == "PREDICTED_UNCALIBRATED" else None
            scored = truth is not None and call is not None
            entry = {"module": "market", "fold": fold["id"], "signal_date": day, "target_date": target,
                "original_prediction": call, "truth": truth, "truth_status": reason,
                "evaluation_status": "SCORED" if scored else "UNKNOWN", "error": call != truth if scored else None,
                "original_score": _number(mr.get("confidence")) if mr else None,
                "missing_model_fields": json.dumps(diag["market"]["missing_model_fields"], ensure_ascii=False),
            "fact_prompts": json.dumps(diag["market"]["fact_prompts"], ensure_ascii=False),
                **{"raw_score_"+c: diag["market"]["original_raw_scores"][c] for c in MARKET_CLASSES},
                **{k: diag["market"]["signal_facts"].get(k) for k in MARKET_FIELDS},
                "original_candidates": len(raw_candidates)}
            error_rows.append(entry); market_errors.append(entry)
            stock_day = []
            original_by_code = {r["code"]: r for r in raw_candidates}
            for candidate in diag["candidates"]:
                raw = original_by_code[candidate["code"]]
                truth, reason = _truth(raw, "relay", day, target, cutoff)
                call = candidate["predicted_positive"]
                scored = truth is not None and call is not None
                row = {"module": "relay", "fold": fold["id"], "signal_date": day, "target_date": target,
                    **candidate, "original_prediction": call, "truth": truth, "truth_status": reason,
                    "evaluation_status": "SCORED" if scored else "UNKNOWN", "error": call != truth if scored else None}
                row["missing_model_fields"] = json.dumps(row["missing_model_fields"], ensure_ascii=False)
                row["fact_prompts"] = json.dumps(row["fact_prompts"], ensure_ascii=False)
                row["signal_facts"] = json.dumps(row["signal_facts"], ensure_ascii=False)
                candidate_rows.append(row); stock_day.append(row); stock_errors.append(row)
            error_rows.append({"module": "relay", "fold": fold["id"], "signal_date": day, "target_date": target,
                "evaluation_status": "CANDIDATE_GROUP" if stock_day else "NO_SAVED_CANDIDATE_ROWS",
                "original_candidates": len(stock_day), "scored_candidates": sum(r["evaluation_status"] == "SCORED" for r in stock_day),
                "error_candidates": sum(r["error"] is True for r in stock_day),
                "all_rank_first": diag["all_rank_first"], "positive_model_primary": diag["positive_model_primary"],
                "one_word_rank_first": diag["one_word_rank_first"],
                "one_word_rank_status": diag["one_word_rank_status"] if raw_candidates else "UNKNOWN_SOURCE",
                "non_one_word_rank_first": diag["non_one_word_rank_first"],
                "non_one_word_positive_first": diag["non_one_word_positive_first"],
                "non_one_word_scope_complete": diag["non_one_word_scope_complete"] if raw_candidates else None,
                "non_one_word_primary_status": diag["non_one_word_primary_status"] if raw_candidates else "UNKNOWN_SOURCE"})
        folds[fold["id"]] = {"official_signal_dates": len(days), "market": _classification(market_errors, "market"),
            "relay_candidates": _classification(stock_errors, "relay"),
            "by_boards": {str(b): _classification([r for r in stock_errors if r["boards"] == b], "relay")
                          for b in sorted({r["boards"] for r in stock_errors})},
            "true_ohlc_one_word_known_candidates": sum(r["actual_one_word_daily"] is not None for r in stock_errors),
            "relay_saved_candidate_dates": len(by_relay)}
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(error_rows).to_csv(directory / "full_error_rows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(candidate_rows).to_csv(directory / "relay_candidate_diagnostics.csv", index=False, encoding="utf-8-sig")
    report = {"status": "BASE_HISTORICAL_ERROR_APPENDIX", "evidence_class": "HISTORICAL_REPLAY_NOT_FORWARD",
        "accuracy_change": "NO_ACCURACY_CHANGE", "generated_at": now(), "knowledge_cutoff": cutoff,
        "protocol_sha256": mreport["protocol_sha256"], "program_sha256": sha(__file__), "source_hashes": sources,
        "folds": folds, "outputs": {p.name: sha(p) for p in directory.iterdir()},
        "new_forward_samples": 0, "model_prediction_or_threshold_changed": False,
        "说明": "只读保存BASE结果的独立误差附页；原分数、判断、候选、失败与未知全部保留。历史已看，不是盲测或前向改善。旧年份真实OHLC一字缺失不以低换手代理回填。",
        "limits": ["Provider scope; historical labels reused without new cent/limit certification",
            "Missing predictions/labels remain unknown in the official date grid", "Uncalibrated scores; calibrated Brier unavailable",
            "Daily one-price evidence does not establish executable fills or leader identity"]}
    _write(directory / "report.json", report)
    return report


def signal_from_registrations(market_registration, relay_registration, feature_archive, output):
    """Validate native immutable BASE registrations, then issue a separate appendix."""
    from model_upgrade_shadow import _registered
    from model_upgrade_inputs import load_feature_archive
    market, *_ = _registered(market_registration)
    relay, *_ = _registered(relay_registration)
    if market["module"] != "market" or relay["module"] != "relay":
        raise ValueError("both native module registrations required")
    for key in ("signal_date", "target_date", "feature_projection_sha256", "protocol_sha256"):
        if market[key] != relay[key]:
            raise ValueError("module registrations have different signal/feature bindings")
    if any(Path(r["feature_archive"]).resolve() != Path(feature_archive).resolve() for r in (market, relay)):
        raise ValueError("feature archive is not the original registered archive")
    cutoff = min(stamp(r["knowledge_cutoff"]) for r in (market, relay)).isoformat()
    pack = load_feature_archive(feature_archive, cutoff=cutoff)
    projection = hashlib.sha256(_canonical(pack).encode()).hexdigest()
    if projection != market["feature_projection_sha256"]:
        raise ValueError("featurepack differs from native registered projection")
    if pack["signal_date"] != market["signal_date"] or pack["target_date"] != market["target_date"]:
        raise ValueError("featurepack dates disagree with native registrations")
    result = signal_diagnostics(pack, market["arms"]["BASE"], relay["arms"]["BASE"],
        market["model_bindings"]["BASE"]["manifest"], relay["model_bindings"]["BASE"]["manifest"], asof=cutoff)
    for candidate in result["candidates"]:
        if any(p in candidate["fact_prompts"] for p in ("BASE_BOARD_METADATA_CONFLICT", "BASE_OHLC_METADATA_CONFLICT", "SAME_HEIGHT_COHORT_COUNT_CONFLICT")):
            raise ValueError("native BASE registration metadata conflicts with source-bound featurepack")
    result["generated_at"] = now()
    result["original_module_cutoffs"] = {"market": market["knowledge_cutoff"], "relay": relay["knowledge_cutoff"]}
    sources = {str(Path(p).resolve()): sha(p) for p in [market_registration, relay_registration, __file__]}
    sources.update(pack["sources"])
    for registration in (market, relay):
        binding = registration["model_bindings"]["BASE"]
        sources.update({binding["path"]: binding["sha256"], binding["model_path"]: binding["model_sha256"]})
    result["source_hashes"], result["program_sha256"] = sources, sha(__file__)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(result["candidates"]).to_csv(directory / "relay_candidate_diagnostics.csv", index=False, encoding="utf-8-sig")
    result["outputs"] = {"relay_candidate_diagnostics.csv": sha(directory / "relay_candidate_diagnostics.csv")}
    _write(directory / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    hist = sub.add_parser("historical")
    hist.add_argument("--root", required=True); hist.add_argument("--calendar", required=True)
    hist.add_argument("--output", required=True); hist.add_argument("--protocol")
    signal = sub.add_parser("signal")
    signal.add_argument("--market-registration", required=True); signal.add_argument("--relay-registration", required=True)
    signal.add_argument("--feature-archive", required=True); signal.add_argument("--output", required=True)
    theme = sub.add_parser("theme")
    theme.add_argument("--pool-archive", required=True); theme.add_argument("--context-archive", required=True)
    theme.add_argument("--signal-date", required=True); theme.add_argument("--output", required=True)
    theme.add_argument("--knowledge-cutoff"); theme.add_argument("--base-predictions")
    theme.add_argument("--g1-protocol")
    regime = sub.add_parser("regime")
    regime.add_argument("--registry", required=True); regime.add_argument("--ledger", required=True)
    regime.add_argument("--calendar", required=True); regime.add_argument("--output", required=True)
    regime.add_argument("--history-root"); regime.add_argument("--knowledge-cutoff")
    args = parser.parse_args()
    if args.command == "historical":
        result = historical(args.root, args.calendar, args.output, args.protocol)
    elif args.command == "signal":
        result = signal_from_registrations(args.market_registration, args.relay_registration, args.feature_archive, args.output)
    elif args.command == "regime":
        from cycle_regime_diagnostics import run
        result = run(args.registry, args.ledger, args.calendar, args.output, args.history_root, args.knowledge_cutoff)
    else:
        result = theme_from_archives(args.pool_archive, args.context_archive, args.signal_date,
            args.output, args.knowledge_cutoff, args.base_predictions)
        if args.g1_protocol:
            if not args.base_predictions:
                raise ValueError("G1 requires unchanged full original BASE predictions")
            from theme_structure_g1 import run
            run(Path(args.output)/"report.json", args.base_predictions, args.g1_protocol, Path(args.output)/"g1")
    print(json.dumps({"status": result["status"], "output": args.output, "accuracy_change": "NO_ACCURACY_CHANGE"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
