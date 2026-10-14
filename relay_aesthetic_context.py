"""Source-qualified price-path interpretation, never a score or buy signal.

The signs and equalities below describe observed prices. They have no fitted
weights and do not establish queue absorption, leadership or promotion odds.
Current surviving peers and previously declared rivals retain separate scopes.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import leader_role_engine as e

VERSION = "relay-aesthetic-context-1"


def _value(row, group, qualification_field):
    if e.number(row.get("features", {}).get(qualification_field)) is None:
        return None
    return row.get("source_groups", {}).get(group, {}).get("value")


def _proof(row, group):
    cell = row.get("source_groups", {}).get(group, {})
    return {k: cell[k] for k in ("source_path", "source_sha256", "pointer", "observed_at", "available_at", "reconstructed") if k in cell}


def _relative(value, scope):
    result = {"scope": scope, "complete_comparison": False, "state": "UNKNOWN",
              "peer_codes": [], "missing_peer_codes": [], "relative_min_pp": None,
              "relative_last_pp": None, "positive_bar_share": None,
              "active_displacement_or_core_identity_confirmed": False}
    if value is None:
        result["reason"] = "NO_QUALIFIED_PEER_PATH_SOURCE"
        return result
    data = value[scope]
    result.update(peer_codes=data["peer_codes"], missing_peer_codes=data["missing_peer_codes"],
                  declared_scope_complete=data["scope_complete"],
                  complete_comparison=data["complete_price_comparison"])
    features = data["features"]
    if not data["complete_price_comparison"]:
        result["reason"] = "NO_PEERS_OR_INCOMPLETE_REFERENCE_PRICES_GRID_OR_DECLARED_SCOPE"
        return result
    result.update({k: features.get(k) for k in ("relative_min_pp", "relative_last_pp", "positive_bar_share")})
    minimum, last = e.number(result["relative_min_pp"]), e.number(result["relative_last_pp"])
    if minimum is None or last is None:
        result.update(state="UNKNOWN", complete_comparison=False, reason="MISSING_COMPUTED_RELATIVE_VALUES")
    elif minimum > 0:
        result["state"] = "RELATIVE_PRICE_AHEAD_AT_EVERY_OBSERVED_POINT"
    elif last > 0:
        result["state"] = "RELATIVE_PRICE_RECOVERED_OR_MIXED_WITH_FINAL_ADVANTAGE"
    elif last == 0:
        result["state"] = "NO_FINAL_RELATIVE_PRICE_ADVANTAGE"
    else:
        result["state"] = "FINAL_RELATIVE_PRICE_BEHIND"
    result["range_mean_is_not_pairwise_victory"] = True
    result["small_positive_gap_may_reflect_limit_price_rounding_not_competitive_dominance"] = True
    return result


def describe(row):
    """Consumes e.analyse's qualified projection, never raw caller flags."""
    sampled = _value(row, "sampled_path", "sampled_grid_complete")
    peers = _value(row, "sampled_peers", "sampled_current_peer_expected")
    path_state = "UNKNOWN_NO_QUALIFIED_INTRADAY_PRICE_PATH"
    prompts = []
    info = {"interval_minutes": None, "grid_complete": None, "sampled_close_drop_observed": None,
            "sampled_recovery_close_observed": None, "after_touch_low_drawdown_pct": None,
            "provider_bomb_count": None, "first_last_reported_gap_minutes": None,
            "first_last_gap_is_not_actual_open_duration": True, "actual_reseal_verified": None,
            "absorption_verified": False, "within_bar_high_low_order": "UNKNOWN"}
    if sampled:
        features = sampled["features"]
        info.update(interval_minutes=sampled["interval_minutes"], grid_complete=sampled["grid_complete"],
                    sampled_close_drop_observed=features.get("sampled_close_drop_after_limit"),
                    sampled_recovery_close_observed=features.get("sampled_later_limit_close_after_drop"),
                    after_touch_low_drawdown_pct=features.get("sampled_after_touch_drawdown_pct"),
                    first_observed_touch_bar=sampled.get("first_observed_touch_bar"),
                    close_below_after_limit_bar=sampled.get("close_below_after_limit_bar"),
                    later_limit_close_bar=sampled.get("later_limit_close_bar"))
        reported = sampled.get("provider_reported_seal", {})
        info.update(provider_bomb_count=e.number(reported.get("bomb_count")),
                    first_last_reported_gap_minutes=e.number(reported.get("first_last_clock_gap_minutes")),
                    original_capture_manifest_hash_present=reported.get("original_capture_manifest_hash_present"))
        if sampled.get("actual_one_word_observed_OHLC") is True:
            path_state = "ONE_WORD_OBSERVED_REOPEN_PRESSURE_NOT_TESTED"
            prompts.append("一字领先可作强度观察；首次真实打开后的承接、盘口和可成交性仍需核对。")
        elif info["sampled_close_drop_observed"] == 1 and info["sampled_recovery_close_observed"] == 1:
            path_state = "OBSERVED_PRICE_DROP_AND_LATER_RECOVERY_NOT_SEAL_ABSORPTION"
            prompts.append("已看到采样价格回落后恢复；核对真实卖压、回封及同步对手，不能单凭恢复确认承接。")
        elif info["sampled_close_drop_observed"] == 1:
            path_state = "OBSERVED_PRICE_DROP_WITHOUT_LATER_SAMPLED_RECOVERY"
            prompts.append("观察窗口出现跌离涨停且未看到后续采样恢复；不以缺失窗口否定未观察到的回封。")
        elif (e.number(info["after_touch_low_drawdown_pct"]) is not None
              and info["after_touch_low_drawdown_pct"] < 0):
            path_state = "INTRABAR_PRICE_PRESSURE_WITHOUT_SAMPLED_CLOSE_BREAK"
            prompts.append("采样收盘未离板但区间低点回落；实际先后、开板时长与承接不能由五分钟柱补造。")
        elif sampled["grid_complete"]:
            path_state = "NO_SAMPLED_POST_TOUCH_BREAK_OBSERVED"
            prompts.append("完整采样未看到触板后价格回落；这不等于逐笔从未开板或次日确定晋级。")
        else:
            path_state = "UNKNOWN_INCOMPLETE_SAMPLED_GRID"
            prompts.append("采样缺口影响未发生判断，保留未知。")
    current, prior = _relative(peers, "current"), _relative(peers, "prior")
    current["comparison_cohort"] = "CURRENT_CLOSING_SURVIVORS_ONLY"
    prior["comparison_cohort"] = "PRIOR_DECLARED_RIVALS_WITH_CURRENT_OBSERVATIONS"
    if current["complete_comparison"]:
        prompts.append("当前同板位比较只覆盖收盘幸存者；前日落败或未入池对手不能从分母中消失。")
        prompts.append("微小相对涨幅差可能来自涨停价取整；领先符号不是主动强度或卡位的认证。")
    if prior["complete_comparison"]:
        prompts.append("原声明对手价格比较完整；领先均值仍需逐对核查，不能自动描述主动卡位或对手核心死亡。")
    gaps = []
    if sampled is None: gaps.append("NO_QUALIFIED_SAMPLED_PATH")
    if not prior["complete_comparison"]: gaps.append("COMPLETE_ORIGINAL_RIVAL_COMPARISON_UNAVAILABLE")
    gaps.extend(["TICK_ORDER_BOOK_ABSORPTION_UNVERIFIED", "EXECUTABLE_FILL_UNVERIFIED",
                 "THEME_CAUSALITY_AND_ROLE_IDENTITY_REQUIRE_SEPARATE_EVIDENCE"])
    return {"version": VERSION, "candidate_id": row["candidate_id"], "code": row["code"],
            "signal_date": row["signal_date"], "name": row.get("name"), "boards": row["boards"],
            "actual_one_word_daily": row.get("actual_one_word_daily"),
            "low_turnover_proxy_is_not_one_word_identity": True,
            "price_path_state": path_state, "price_path_facts": info,
            "current_survivor_comparison": current, "prior_rival_comparison": prior,
            "review_questions": prompts, "missing_evidence": gaps,
            "sampled_source": _proof(row, "sampled_path") if sampled else None,
            "peer_source": _proof(row, "sampled_peers") if peers else None,
            "promotion_judgement": "NOT_CREATED_BY_DESCRIPTOR", "trade_entry": "NOT_CREATED_BY_DESCRIPTOR",
            "aesthetic_score_or_probability": None, "automatic_leader_identity": False,
            "availability_track": "Inherited from parent qualified role projection"}


@e.verified_session
def build(dataset_path, output, track):
    code_bindings = e.bind(__file__) | e.bind(e.__file__)
    e.verify(code_bindings)
    dataset = e.load_dataset(dataset_path)
    if track not in {"STRICT", "DIAGNOSTIC"}:
        raise ValueError("explicit availability track required")
    days = e.calendar(dataset["calendar_path"], dataset["calendar_sha256"])
    originals = dataset["rows"]
    ids = [row["candidate_id"] for row in originals]
    if len(set(ids)) != len(ids) or any(row["signal_date"] not in days
            or row["candidate_id"] != row["signal_date"]+":"+row["code"] for row in originals):
        raise ValueError("complete original candidate IDs or signal calendar changed")
    # All original IDs stay in the denominator; cases with any saved sampled
    # group remain even when that group's qualification fails on this track.
    selected = [row for row in originals if "sampled_path" in row["groups"] or "sampled_peers" in row["groups"]]
    if any(e.stamp(row["available_at"]) > e.stamp(e.now()) or e.stamp(row["decision_at"]) > e.stamp(e.now())
           for row in selected):
        raise ValueError("future sample availability or decision cutoff")
    rows = [describe(e.evidence_row(row, track, days)) for row in selected]
    result = {"version": VERSION, "formed_at": e.now(), "track": track,
              "dataset_path": str(Path(dataset_path).resolve()), "dataset_sha256": e.sha(dataset_path),
              "all_original_candidate_ids": ids,
              "all_original_count": len(originals), "sampled_case_count": len(rows), "rows": rows,
              "path_state_counts": dict(Counter(r["price_path_state"] for r in rows)),
              "source_bindings": dataset["source_bindings"] | e.bind(dataset_path) | code_bindings,
              "numeric_fits": 0, "prediction_registered": False, "new_mature_forward_samples": 0,
              "historical_diagnostic_not_original_forecast": track == "DIAGNOSTIC"}
    e.verify(code_bindings)
    e.write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--track", choices=("STRICT", "DIAGNOSTIC"), required=True)
    args = parser.parse_args()
    report = build(args.dataset, args.output, args.track)
    print(json.dumps({k: report[k] for k in ("all_original_count", "sampled_case_count", "path_state_counts", "prediction_registered")}, ensure_ascii=False))


if __name__ == "__main__": main()
