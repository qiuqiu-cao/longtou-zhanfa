"""Frozen daily research extension and independent, missing-aware scoring.

The V271 original remains authoritative. This extension must be inside its
plan BEFORE registration. A legacy view cannot add predictions retroactively.
No trades, automatic model promotion, or invented probabilities are supported.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from next_session_expectation_contract_v271 import (
    band_result, canonical, digest, finite, next_session, stamp, validate_plan,
)

VERSION = "v292-daily-research-1"
PHASES = ["强修复", "弱修复", "延续主升", "分歧", "强分歧", "退潮", "冰点", "混沌"]
DRIVERS = ["指数主导", "情绪主导", "共同驱动"]
THEME_STATES = ["继续加强", "正常分歧", "分歧后回流", "高潮兑现", "退潮", "一日游", "新旧主线切换"]
ROLES = ["周期龙", "空间龙", "题材龙", "换手龙", "补涨龙", "卡位龙", "穿越龙", "日内龙", "中军", "高标", "后排", "跟风"]
EVENTS = ["promotion", "break", "reseal", "high_open_fade", "weak_to_strong", "strong_to_weak", "rebound"]
ANCHORS = ["emotion", "highest", "main_leader", "main_middle", "middle_risk", "new_cycle"]
METRICS = {
    "index_return": "HIGHER_BETTER", "turnover_amount": "HIGHER_BETTER",
    "limit_up_count": "HIGHER_BETTER", "limit_down_count": "LOWER_BETTER",
    "failed_rate": "LOWER_BETTER", "max_boards": "HIGHER_BETTER",
    "promotion_rate": "HIGHER_BETTER", "previous_limit_premium": "HIGHER_BETTER",
    "high_stock_feedback": "HIGHER_BETTER", "theme_strength": "HIGHER_BETTER",
}
FEATURES = ["theme_strength", "leader_position", "ladder", "middle_feedback", "board_type",
    "turnover", "amount", "seal_time", "open_count", "reseal_count", "seal_order",
    "absorption", "historical_auction", "emotion", "historical_promotion", "height_pressure",
    "same_height_competition", "position_challenge", "supplement_relationship"]
LEADER_FEATURES = ["market_position", "theme_position", "initiative", "influence", "recognition",
    "turnover_quality", "absorption", "middle_linkage", "ladder", "flow_persistence",
    "expectation_gap", "space", "market_node"]
CONSENSUS_FIELDS = ["forum_heat", "discussion_change", "agreement", "attention", "bull_bear_extremes"]
TODAY_FIELDS = ["emotion", "cycle", "index", "amount", "money_effect", "loss_effect", "max_boards", "promotion_rate"]


def unknown(reason="未取得同口径、可核验的时点证据"):
    return {"status": "UNKNOWN", "value": None, "reason": reason}


def rule(spec):
    """Bind a predeclared semantic label to its complete definition."""
    return {"spec": deepcopy(spec), "sha256": digest(spec)}


def check_rule(value):
    if not isinstance(value, dict) or not value.get("spec") or value.get("sha256") != digest(value["spec"]):
        raise ValueError("label rule must bind its predeclared full specification")


def checked_probability(item, plan, target, scope):
    p = item.get("probability")
    if p is None:
        if not item.get("probability_unavailable_reason"):
            raise ValueError("unknown probability needs an explicit reason")
        return
    if not finite(p) or not 0 <= p <= 1:
        raise ValueError("probability must be finite and in [0,1]")
    # Evidence is included in the frozen input, not fetched after the outcome.
    evidence = item.get("calibration", {})
    check_rule(evidence.get("method", {}))
    if (evidence.get("target"), evidence.get("scope")) != (target, scope):
        raise ValueError("calibration target/scope differs from forecast")
    if evidence.get("status") != "OUT_OF_SAMPLE_CALIBRATED" or not evidence.get("source_sha256"):
        raise ValueError("numerical probability requires hash-bound OOS calibration evidence")
    source = Path(evidence.get("source", ""))
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != evidence["source_sha256"]:
        raise ValueError("calibration artifact missing or hash mismatch")
    receipt = json.loads(source.read_text(encoding="utf-8"))
    if any(receipt.get(k) != evidence.get(k) for k in ["status", "target", "scope", "available_at", "max_label_date", "evaluation_n", "brier", "bins", "method"]):
        raise ValueError("calibration diagnostics differ from immutable artifact")
    if stamp(evidence["available_at"]) > stamp(plan.get("knowledge_cutoff", plan["evidence_cutoff"])):
        raise ValueError("calibration evidence arrived after freeze")
    if evidence["max_label_date"] >= plan["target_date"] or evidence["max_label_date"] > plan["signal_date"]:
        raise ValueError("future calibration labels")
    if evidence.get("evaluation_n", 0) < 60 or not finite(evidence.get("brier")) or not evidence.get("bins"):
        raise ValueError("calibration diagnostics or sample coverage missing")
    # This is a research evidence gate, not a proof that probabilities are stable.
    lo, hi = item.get("probability_interval", [p, p])
    if not finite(lo) or not finite(hi) or not 0 <= lo <= p <= hi <= 1:
        raise ValueError("invalid probability interval")


def check_fact(fact, plan):
    if fact.get("status") == "UNKNOWN":
        if fact.get("value") is not None or not fact.get("reason"):
            raise ValueError("missing fact cannot contain a value")
        return
    if fact.get("status") not in {"verified", "inferred", "delayed"} or not fact.get("source"):
        raise ValueError("fact source/status missing")
    observed, available = stamp(fact["observed_at"]), stamp(fact["available_at"])
    cutoff = stamp(plan.get("knowledge_cutoff", plan["evidence_cutoff"]))
    if observed > available or available > cutoff:
        raise ValueError("future fact in daily research")
    if observed.date().isoformat() >= plan["target_date"]:
        raise ValueError("target-session observations cannot enter frozen research")


def check_claim(item, labels, plan, target, scope):
    value = item.get("call")
    if value is None:
        if not item.get("reason"):
            raise ValueError("unmade forecast needs a reason")
    else:
        if value not in labels:
            raise ValueError("unsupported categorical call")
        check_rule(item.get("label_rule", {}))
        if not item.get("basis") or not item.get("invalidation"):
            raise ValueError("forecast basis and invalidation missing")
    checked_probability(item, plan, target, scope)
    if value is None and item.get("probability") is not None:
        raise ValueError("probability without a declared target call")


def validate_extension(plan, recorded_at):
    validate_plan(plan, recorded_at)
    ext = plan.get("daily_research")
    if ext is None:
        return {"status": "LEGACY_NO_EXTENSION", "missing_layers": ["emotion", "theme", "leader", "events"]}
    if ext.get("version") != VERSION:
        raise ValueError("unsupported daily research schema")
    if set(ext["today"]) != set(TODAY_FIELDS):
        raise ValueError("all eight current-session observations must be represented")
    for fact in ext["today"].values():
        check_fact(fact, plan)
    for name, labels in [("market_phase", PHASES), ("emotion", PHASES), ("driver", DRIVERS)]:
        check_claim(ext[name], labels, plan, name, "MARKET")
    if set(ext["metrics"]) != set(METRICS):
        raise ValueError("all ten expectation baselines must be represented")
    for metric, item in ext["metrics"].items():
        check_fact(item["baseline"], plan)
        if item.get("band") is None:
            if not item.get("reason"):
                raise ValueError("unavailable expected range needs a reason")
        else:
            b = item["band"]
            if b.get("metric") != metric or b.get("direction") != METRICS[metric] or b.get("stage") != "CLOSE":
                raise ValueError("range semantics differ from the frozen metric")
            if not all(b.get(k) for k in ["unit", "scope", "basis"]) or not finite(b.get("low")) or not finite(b.get("high")) or b["low"] > b["high"]:
                raise ValueError("expected range needs finite bounds, scope, units and basis")
    expected = {c["code"]: c for c in plan["candidates"]}
    actual = {c["code"]: c for c in ext["candidates"]}
    if len(actual) != len(ext["candidates"]) or set(actual) != set(expected):
        raise ValueError("daily extension must preserve EVERY frozen relay candidate")
    for code, candidate in actual.items():
        if candidate.get("boards") != expected[code]["boards"]:
            raise ValueError("board count differs from frozen denominator")
        if set(candidate["features"]) != set(FEATURES) or set(candidate["events"]) != set(EVENTS):
            raise ValueError("required feature/event coverage missing")
        for fact in candidate["features"].values():
            check_fact(fact, plan)
        for event, item in candidate["events"].items():
            check_claim(item, [True, False], plan, event, "STOCK:"+code)
            if item.get("call") is not None and not isinstance(item["call"], bool):
                raise ValueError("event call must be a strict boolean")
        promotion, broken = (candidate["events"][n] for n in ["promotion", "break"])
        core = expected[code]["promotion_call"]
        if promotion["call"] != ({"PROMOTE": True, "FAIL": False, "UNCERTAIN": None}[core]):
            raise ValueError("extension cannot silently change original promotion call")
        if broken["call"] is not None and (promotion["call"] is None or broken["call"] != (not promotion["call"])):
            raise ValueError("promotion and consecutive-board break are complements")
        if promotion["probability"] is not None and broken["probability"] is not None and abs(promotion["probability"]+broken["probability"]-1) > 1e-9:
            raise ValueError("promotion and break probabilities must sum to one")
    if len(ext["leaders"]) != 3:
        raise ValueError("represent all three leader-candidate slots, including unknown slots")
    leader_codes = []
    for leader in ext["leaders"]:
        if leader.get("code") is None:
            if not leader.get("reason"):
                raise ValueError("empty leader slot needs a reason")
            continue
        leader_codes.append(leader["code"])
        if leader.get("role") not in ROLES or leader.get("identity_status") not in {"CANDIDATE", "CONFIRMED"}:
            raise ValueError("explicit leader role and confirmation status required")
        if set(leader["features"]) != set(LEADER_FEATURES):
            raise ValueError("leader feature coverage missing")
        for fact in leader["features"].values():
            check_fact(fact, plan)
        for name in ["normal", "strong", "weak", "hold", "watch", "invalidate", "confirm"]:
            if not leader.get(name):
                raise ValueError("leader conditions missing")
        if leader["identity_status"] == "CONFIRMED" and any(leader["features"][k]["status"] != "verified" for k in ["theme_position", "influence", "absorption", "ladder"]):
            raise ValueError("height alone cannot confirm a leader")
        check_claim(leader["forecast"], [True, False], plan, "leader_identity", "STOCK:"+leader["code"])
    if len(leader_codes) != len(set(leader_codes)):
        raise ValueError("duplicate leader slots")
    theme_ids = []
    for theme in ext["themes"]:
        theme_ids.append(theme["theme_id"])
        check_fact(theme["membership"], plan)
        check_claim(theme["forecast"], THEME_STATES, plan, "theme_state", "THEME:"+theme["theme_id"])
        if set(theme["relationships"]) != {"leader", "middle", "supplement", "rear", "new_first"}:
            raise ValueError("theme ladder relationships missing")
    if len(theme_ids) != len(set(theme_ids)):
        raise ValueError("duplicate theme prediction")
    if not ext["themes"] and not ext.get("themes_unavailable_reason"):
        raise ValueError("missing theme denominator needs an explicit reason")
    if set(ext["anchors"]) != set(ANCHORS) or set(ext["plans"]) != {"A", "B", "C"}:
        raise ValueError("six anchors and three contingency plans required")
    for anchor in ext["anchors"].values():
        if not anchor.get("reason") or not anchor.get("observe"):
            raise ValueError("anchor evidence and observation criteria missing")
        if anchor.get("code") is not None and (not isinstance(anchor["code"], str) or len(anchor["code"]) != 6 or not anchor["code"].isdigit()):
            raise ValueError("anchor must name an explicit six-digit stock code")
        check_fact(anchor["evidence"], plan)
    for scenario in ext["plans"].values():
        if not all(scenario.get(k) for k in ["conditions", "observe", "action", "cancel"]):
            raise ValueError("scenario conditions/action/cancel missing")
    if set(ext["consensus"]) != set(CONSENSUS_FIELDS):
        raise ValueError("consensus data coverage missing")
    for fact in ext["consensus"].values():
        check_fact(fact, plan)
    for hypothesis in ext["hypotheses"]:
        if hypothesis.get("grade") not in {"A", "B", "C", "D"} or hypothesis.get("core_rule") is not False:
            raise ValueError("unpromoted hypotheses cannot act as core trading rules")
        if hypothesis["grade"] == "A" and not hypothesis.get("statistical_evidence"):
            raise ValueError("grade A requires historical statistical evidence")
    return {"status": "EXTENSION_VALIDATED", "automatic_trading_allowed": False}


def verified_outcome(block, target, recorded_at):
    if not block or block.get("status") != "verified":
        return False
    if not block.get("source"):
        raise ValueError("verified outcome requires a source")
    observed, available = stamp(block["observed_at"]), stamp(block["available_at"])
    close = stamp(target+"T15:00:00+08:00")
    if observed < close or observed.date() != close.date() or observed > available or available > stamp(recorded_at):
        raise ValueError("outcome is not an available, completed exact-target result")
    return True


def result_row(layer, key, call, truth=None, *, probability=None, boards=None, highest=False, reason=None, partial=False):
    known, decided = truth is not None, call is not None
    return {"layer": layer, "key": key, "call": call, "truth": truth,
        "probability": probability, "boards": boards, "highest": highest,
        "outcome_known": known, "forecast_made": decided,
        "correct": call == truth if known and decided else None,
        "partial_correct": bool(partial) if known and decided else None,
        "reason": reason if reason else (None if known else "目标日结果未完整核验")}


def score(registered, outcomes, recorded_at):
    plan = registered["plan"]
    validate_extension(plan, registered["recorded_at"])
    target = plan["target_date"]
    if outcomes.get("example_only"):
        raise ValueError("example or synthetic outcomes cannot enter the real scoring ledger")
    if stamp(recorded_at) < stamp(target+"T15:00:00+08:00") or outcomes.get("date") != target:
        raise ValueError("scoring requires the exact next completed trading session")
    pool = outcomes.get("limit_pool", {})
    complete_pool = verified_outcome(pool, target, recorded_at)
    if complete_pool and (pool.get("complete") is not True or pool.get("scope") != "MAINBOARD_NON_ST_CLOSING_LIMIT_UP"):
        raise ValueError("closing pool scope/completeness differs from frozen denominator")
    codes = {}
    if complete_pool:
        for item in pool.get("rows", []):
            if item["code"] in codes or type(item.get("boards")) is not int or item["boards"] < 1:
                raise ValueError("duplicate or invalid closing-pool board count")
            codes[item["code"]] = item
    rows, bands = [], []
    market = outcomes.get("market", {})
    market_ok = verified_outcome(market, target, recorded_at)
    for channel, labels in [("primary", ["RISK_ON", "REPAIR", "RISK_OFF"]), ("direction", ["IMPROVE", "CONTINUE", "WORSEN"])]:
        truth = market.get(channel) if market_ok else None
        if truth is not None and truth not in labels:
            raise ValueError("invalid realized market label")
        rows.append(result_row("market_"+channel, "MARKET", plan["market"].get(channel), truth))
    maximum = max((c["boards"] for c in plan["candidates"]), default=0)
    ext = plan.get("daily_research")
    for candidate in plan["candidates"]:
        code, height = candidate["code"], candidate["boards"]
        promoted = (code in codes and codes[code]["boards"] == height+1) if complete_pool else None
        call = {"PROMOTE": True, "FAIL": False, "UNCERTAIN": None}[candidate["promotion_call"]]
        item = next((c for c in ext["candidates"] if c["code"] == code), None) if ext else None
        probability = item["events"]["promotion"]["probability"] if item else None
        rows.append(result_row("promotion", code, call, promoted, probability=probability, boards=height, highest=height == maximum))
        if item:
            rows.append(result_row("break", code, item["events"]["break"]["call"], None if promoted is None else not promoted,
                probability=item["events"]["break"]["probability"], boards=height))
        else:
            for event in EVENTS[1:]:
                rows.append(result_row(event, code, None, reason="原预测未包含该事件，不事后补造", boards=height))
    # Qualitative and path-event labels require the exact predeclared rule hash.
    label_blocks = outcomes.get("labels", [])
    label_index = {}
    for label in label_blocks:
        key = label["layer"], label["key"]
        if key in label_index:
            raise ValueError("duplicate outcome label")
        label_index[key] = label
    def add_claim(layer, key, item, allowed):
        label = label_index.get((layer, key))
        truth, partial = None, False
        if verified_outcome(label, target, recorded_at):
            if item.get("call") is not None and label.get("rule_sha256") != item["label_rule"]["sha256"]:
                raise ValueError("realized qualitative/event label changed frozen semantics")
            truth = label.get("value")
            if truth not in allowed or (allowed == [True, False] and type(truth) is not bool):
                raise ValueError("outcome outside predeclared label classes")
            if layer == "emotion" and item.get("label_rule"):
                adjacency = item["label_rule"]["spec"].get("partial_pairs", [])
                partial = [item.get("call"), truth] in adjacency
        rows.append(result_row(layer, key, item.get("call"), truth,
            probability=item.get("probability"), partial=partial, reason=item.get("reason")))
    if ext:
        for layer, allowed in [("market_phase", PHASES), ("emotion", PHASES), ("driver", DRIVERS)]:
            add_claim(layer, "MARKET", ext[layer], allowed)
        for theme in ext["themes"]:
            add_claim("theme", theme["theme_id"], theme["forecast"], THEME_STATES)
        for leader in ext["leaders"]:
            if leader.get("code"):
                add_claim("leader", leader["code"], leader["forecast"], [True, False])
        for candidate in ext["candidates"]:
            for event in EVENTS[2:]:
                add_claim(event, candidate["code"], candidate["events"][event], [True, False])
    else:
        for layer in ["market_phase", "emotion", "driver", "theme", "leader"]:
            rows.append(result_row(layer, "UNPREDICTED_LEGACY", None, reason="原预测未包含该层，不事后补造"))
    observations = {}
    for obs in outcomes.get("observations", []):
        key = obs["metric"], obs["scope"]
        if key in observations:
            raise ValueError("duplicate numeric observation")
        if verified_outcome(obs, target, recorded_at):
            observations[key] = obs
    expected_bands = [b for b in plan.get("bands", []) if b["stage"] == "CLOSE"]
    if ext:
        expected_bands += [v["band"] for v in ext["metrics"].values() if v["band"]]
    seen = set()
    for band in expected_bands:
        key = band["metric"], band["scope"]
        if key in seen:
            raise ValueError("duplicate frozen close expectation")
        seen.add(key)
        obs = observations.get(key)
        bands.append({"metric": band["metric"], "scope": band["scope"], "low": band["low"], "high": band["high"],
            "actual": obs["value"] if obs else None, "result": band_result(band, obs)})
    errors = []
    explanations = outcomes.get("error_evidence", [])
    for row in rows:
        if row["correct"] is not False:
            continue
        evidence = [e for e in explanations if (e.get("layer"), e.get("key")) == (row["layer"], row["key"])]
        for e in evidence:
            if not e.get("source") or stamp(e["available_at"]) > stamp(recorded_at):
                raise ValueError("unavailable error explanation")
        cutoff = stamp(plan.get("knowledge_cutoff", plan["evidence_cutoff"]))
        errors.append({"layer": row["layer"], "key": row["key"], "original": row["call"], "actual": row["truth"],
            "observable_by_freeze": any(stamp(e["available_at"]) <= cutoff for e in evidence) if evidence else None,
            "evidence": evidence, "explanation": "有证据待研究" if evidence else "归因证据不足，不强行解释"})
    return {"version": VERSION, "model_version": plan["model_version"], "registration_hash": registered["sha256"], "signal_date": plan["signal_date"],
        "target_date": target, "recorded_at": recorded_at, "forecast_status": registered["forecast_status"],
        "outcome_hash": digest(outcomes), "rows": rows, "bands": bands, "errors": errors,
        "scoring_status": "PARTIAL" if any(r["forecast_made"] and not r["outcome_known"] for r in rows) else "AVAILABLE_DECLARED_CALLS_SCORED",
        "trade_win_rate_claim_allowed": False, "automatic_model_upgrade": False}


def wilson(correct, n):
    if not n:
        return None
    z = 1.959963984540054
    p = correct/n
    denom = 1+z*z/n
    center = (p+z*z/(2*n))/denom
    spread = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0., center-spread), min(1., center+spread)]


def binary_metrics(rows):
    eligible = [r for r in rows if r["correct"] is not None]
    tp = sum(r["call"] is True and r["truth"] is True for r in eligible)
    fp = sum(r["call"] is True and r["truth"] is False for r in eligible)
    fn = sum(r["call"] is False and r["truth"] is True for r in eligible)
    tn = sum(r["call"] is False and r["truth"] is False for r in eligible)
    known_positive = sum(r["truth"] is True for r in rows)
    uncalled_positive = sum(r["truth"] is True and not r["forecast_made"] for r in rows)
    probs = [r for r in rows if r["outcome_known"] and finite(r["probability"])]
    bins = []
    for i in range(10):
        group = [r for r in probs if int(min(r["probability"]*10, 9)) == i]
        bins.append({"low": i/10, "high": (i+1)/10, "n": len(group),
            "mean_probability": sum(r["probability"] for r in group)/len(group) if group else None,
            "actual_rate": sum(r["truth"] for r in group)/len(group) if group else None})
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    return {"total_frozen_rows": len(rows), "known_outcomes": sum(r["outcome_known"] for r in rows),
        "unknown_outcomes": sum(not r["outcome_known"] for r in rows),
        "unmade_forecasts": sum(not r["forecast_made"] for r in rows), "scored": len(eligible),
        "correct": tp+tn, "wrong": fp+fn, "accuracy": (tp+tn)/len(eligible) if eligible else None,
        "accuracy_wilson_95": wilson(tp+tn, len(eligible)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
        "all_known_positive_recall": tp/known_positive if known_positive else None,
        "precision_wilson_95": wilson(tp, tp+fp), "all_known_positive_recall_wilson_95": wilson(tp, known_positive),
        "uncalled_known_positives": uncalled_positive,
        "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        "all_fail_baseline_accuracy": (tn+fp)/len(eligible) if eligible else None,
        "probability_scored_n": len(probs), "brier": sum((r["probability"]-r["truth"])**2 for r in probs)/len(probs) if probs else None,
        "calibration_bins": bins}


def classification_metrics(rows):
    scored = [r for r in rows if r["correct"] is not None]
    classes = sorted({r["call"] for r in scored} | {r["truth"] for r in rows if r["truth"] is not None}, key=str)
    per_class = {}
    for cls in classes:
        tp = sum(r["call"] == cls and r["truth"] == cls for r in scored)
        fp = sum(r["call"] == cls and r["truth"] != cls for r in scored)
        fn = sum(r["call"] != cls and r["truth"] == cls for r in scored)
        per_class[str(cls)] = {"precision": tp/(tp+fp) if tp+fp else None, "recall": tp/(tp+fn) if tp+fn else None,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "support": sum(r["truth"] == cls for r in rows)}
    probs = [r for r in scored if finite(r["probability"])]
    # Scalar probabilities here mean reliability of the declared class, not a
    # complete multiclass probability vector. Its Brier is reported separately.
    return {"total_frozen_rows": len(rows), "scored": len(scored),
        "known_outcomes": sum(r["outcome_known"] for r in rows),
        "unknown_outcomes": sum(not r["outcome_known"] for r in rows),
        "unmade_forecasts": sum(not r["forecast_made"] for r in rows),
        "correct": sum(r["correct"] for r in scored), "wrong": sum(not r["correct"] for r in scored),
        "partial": sum(r["partial_correct"] and not r["correct"] for r in scored),
        "accuracy": sum(r["correct"] for r in scored)/len(scored) if scored else None, "per_class": per_class,
        "accuracy_wilson_95": wilson(sum(r["correct"] for r in scored), len(scored)),
        "top_call_reliability_brier": sum((r["probability"]-r["correct"])**2 for r in probs)/len(probs) if probs else None,
        "probability_scored_n": len(probs), "multiclass_brier": None}


def statistics(settlements, registrations, asof, calendar):
    """20/60 actual exchange sessions, including unresolved/missed dates."""
    cutoff = stamp(asof)
    sessions = [d for d in calendar["sessions"] if stamp(d+"T15:00:00+08:00") <= cutoff]
    if calendar.get("status") != "verified" or not sessions:
        raise ValueError("verified completed trading-session calendar required")
    by_hash = {s["registration_hash"]: s for s in settlements}
    if len(by_hash) != len(settlements):
        raise ValueError("statistics requires one current settlement view per registration")
    reports = {}
    forward = [r for r in registrations if r["forecast_status"] == "FORWARD_REGISTERED"]
    monitoring_start = min((r["plan"]["target_date"] for r in forward), default=None)
    for name, full_dates in [("20", sessions[-20:]), ("60", sessions[-60:]), ("all", sessions)]:
        dates = [d for d in full_dates if monitoring_start is not None and d >= monitoring_start]
        relevant = [r for r in registrations if r["forecast_status"] == "FORWARD_REGISTERED" and r["plan"]["target_date"] in dates]
        scored, pending = [], []
        for registered in relevant:
            value = by_hash.get(registered["sha256"])
            if value:
                scored.append(value)
            else:
                # Account for every missing day's full original denominator.
                pending.append(registered["sha256"])
                blank = score(registered, {"date": registered["plan"]["target_date"]}, asof)
                scored.append(blank)
        rows = [r for s in scored for r in s["rows"]]
        layers = {}
        for layer in sorted({r["layer"] for r in rows}):
            group = [r for r in rows if r["layer"] == layer]
            binary = layer in ["promotion", "break", "leader", *EVENTS[2:]]
            layers[layer] = binary_metrics(group) if binary else classification_metrics(group)
        promotion = [r for r in rows if r["layer"] == "promotion"]
        tiers = {str(height)+"进"+str(height+1): binary_metrics([r for r in promotion if r["boards"] == height]) for height in [2, 3, 4, 5]}
        tiers["5板以上"] = binary_metrics([r for r in promotion if r["boards"] >= 5])
        tiers["最高板"] = binary_metrics([r for r in promotion if r["highest"]])
        report = {"requested_window": name, "monitoring_start": monitoring_start,
            "window_completed_sessions": len(dates), "start": dates[0] if dates else None, "end": dates[-1] if dates else None,
            "registered_forward_dates": len(relevant), "missing_registration_sessions": len(dates)-len(relevant),
            "unsettled_registration_hashes": pending, "layers": layers, "tiers": tiers,
            "future_pending": sum(r["forecast_status"] == "FORWARD_REGISTERED" and r["plan"]["target_date"] > sessions[-1] for r in registrations)}
        reports[name] = report
    return {"version": VERSION, "asof": asof, "windows": reports, "stable_60_verified": False,
        "notes": ["回放与事后附页不进入前向统计", "缺结果、未预测及未登记交易日单列", "二分类准确率不能代替正向精确率和召回率", "未校准概率不计算Brier",
            "Wilson区间为描述性二项区间，同日股票相关性需另做日期聚类检验，不能据此宣布稳定60%"]}
