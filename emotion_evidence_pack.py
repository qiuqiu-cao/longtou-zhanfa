"""Build five-part emotion understanding from causal percentile references.

The result is a descriptive hypothesis, not a calibrated return probability or
an instruction to buy. Missing evidence keeps the affected dimension unavailable.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from leader_target_screen_v17 import timestamp, finite


METRICS = {
    "money_effect": {
        "yesterday_limit_premium": (1, "pct"),
        "first_board_promotion": (1, "ratio"),
        "buyable_profit_rate": (1, "ratio"),
    },
    "broken_loss_effect": {
        "broken_tail_return": (1, "pct"),
        "broken_down_limit_rate": (-1, "ratio"),
        "contagion_scope": (-1, "ratio"),
    },
    "height_feedback": {
        "high_feedback": (1, "pct"),
        "middle_feedback": (1, "pct"),
        "low_feedback": (1, "pct"),
    },
    "theme_divergence": {
        "core_feedback": (1, "pct"),
        "follower_feedback": (1, "pct"),
        "withdrawal_scope": (-1, "ratio"),
    },
    "repair_quality": {
        "repair_close_retention": (1, "ratio"),
        "repair_volume_support": (1, "ratio"),
        "theme_sync": (1, "ratio"),
    },
}


TEXT = {
    "money_effect": {
        "SUPPORTIVE": ("可交易涨停样本的溢价与晋级反馈偏强", "允许保留候选，不单独扩仓"),
        "ADVERSE": ("涨停后溢价或可买样本赚钱效应偏弱", "降低接力意愿，无个股强确认时空仓"),
        "MIXED": ("涨停后反馈分化", "只保留可买样本中的强承接候选"),
        "UNAVAILABLE": ("赚钱效应证据不完整", "不用情绪层支持入场"),
    },
    "broken_loss_effect": {
        "SUPPORTIVE": ("断板尾部与跌停传导受控", "负反馈未构成全面否决"),
        "ADVERSE": ("断板亏钱效应扩散", "默认空仓，除非唯一核心完成强修复"),
        "MIXED": ("断板负反馈局部化", "回避受传导题材和板位"),
        "UNAVAILABLE": ("断板负反馈分母或价格不完整", "不把未观察样本当作无风险"),
    },
    "height_feedback": {
        "SUPPORTIVE": ("高中低位反馈形成正向梯队", "可继续比较先锋与换手核心"),
        "ADVERSE": ("高中低位同步走弱", "视为全面退潮假设，不做接力"),
        "MIXED": ("高中低位反馈不一致", "按高低切换或中位淘汰处理，不用总量掩盖分化"),
        "UNAVAILABLE": ("至少一个高度层级反馈不可用", "不根据最高板或首板数单独判断周期"),
    },
    "theme_divergence": {
        "SUPPORTIVE": ("细分题材核心与跟随梯队共振", "题材可作候选加分，仍需个股承接"),
        "ADVERSE": ("核心与跟风同步撤退", "题材整体禁入，不把单股拉升称为修复"),
        "MIXED": ("题材分歧主要集中在核心或跟风一侧", "只保留未失去梯队支持的核心"),
        "UNAVAILABLE": ("细分题材成员或角色证据不完整", "不用宽泛行业标签推断板块共振"),
    },
    "repair_quality": {
        "SUPPORTIVE": ("修复有收盘保持、成交承接和题材同步支持", "允许观察修复核心的后续确认"),
        "ADVERSE": ("修复缺少收盘保持或群体同步", "按短暂反抽处理，不追高"),
        "MIXED": ("修复强度和结构尚未一致", "等待回封与题材同步再决策"),
        "UNAVAILABLE": ("修复质量证据不完整", "不将盘中翻红或单次拉升称为有效修复"),
    },
}


def _metric(item, direction, unit, cutoff, scope_id):
    try:
        value = finite(item["value"])
        observed, available = timestamp(item["observed_at"]), timestamp(item["available_at"])
        reference = item["reference"]
        p25, median, p75 = [finite(reference[key]) for key in ("p25", "median", "p75")]
        ref_available = timestamp(reference["available_at"])
        ref_start = timestamp(reference["window_start"])
        ref_end = timestamp(reference["window_end"])
        sample = reference["sample_count"]
        denominator, observed_count, missing_count = [item[key] for key in ("denominator", "observed_count", "missing_count")]
        max_age = finite(item.get("max_age_seconds",300))
        if (value is None or any(v is None for v in (p25, median, p75)) or not p25 <= median <= p75
                or item.get("unit") != unit or (unit == "ratio" and (not 0 <= value <= 1 or not 0 <= p25 <= p75 <= 1))
                or max_age is None or max_age < 0 or observed > available or available > cutoff
                or item.get("scope_id") != scope_id or item.get("data_status") not in {"verified", "delayed", "inferred"}
                or not item.get("source") or not item.get("definition_id") or not item.get("cohort_id")
                or any(isinstance(v,bool) or not isinstance(v,int) or v < 0
                       for v in (denominator,observed_count,missing_count))
                or denominator < 1 or observed_count < 1 or observed_count+missing_count != denominator
                or isinstance(sample, bool) or not isinstance(sample, int) or sample < 60
                or reference.get("definition_id") != item.get("definition_id")
                or reference.get("cohort_id") != item.get("cohort_id")
                or not reference.get("source") or not ref_start < ref_end < observed or ref_available > available):
            raise ValueError("invalid metric evidence")
        if direction == 1:
            state = "SUPPORTIVE" if value >= p75 else "ADVERSE" if value <= p25 else "NEUTRAL"
        else:
            state = "SUPPORTIVE" if value <= p25 else "ADVERSE" if value >= p75 else "NEUTRAL"
        current = {key:item[key] for key in ("value", "unit", "observed_at", "available_at", "source",
                                             "data_status", "definition_id", "cohort_id")}
        current.update(metric=item["metric"], max_age_seconds=max_age,
                       denominator=denominator, observed_count=observed_count, missing_count=missing_count,
                       coverage=observed_count/denominator, value_basis="observed_subset" if missing_count else "complete_cohort",
                       percentile_state=state, reference=reference)
        actionable = (item.get("data_status") == "verified" and observed.strftime("%Y-%m-%d") == cutoff.strftime("%Y-%m-%d")
                      and (cutoff-observed).total_seconds() <= max_age and observed_count/denominator >= .9)
        return current, state, actionable
    except (KeyError, TypeError, ValueError):
        return None, "UNAVAILABLE", False


def build(snapshot, asof):
    cutoff = timestamp(asof)
    if timestamp(snapshot["asof"]) != cutoff or not snapshot.get("scope_id"):
        raise ValueError("snapshot asof or scope missing")
    supplied = snapshot.get("metrics", {})
    output = {}
    dimension_states = {}
    for dimension, definitions in METRICS.items():
        evidence, states, actionable_metrics = [], [], []
        for metric, (direction, unit) in definitions.items():
            item = supplied.get(metric)
            if not isinstance(item, dict) or item.get("metric") != metric:
                states.append("UNAVAILABLE")
                actionable_metrics.append(False)
                continue
            normalized, state, actionable = _metric(item, direction, unit, cutoff, snapshot["scope_id"])
            states.append(state)
            actionable_metrics.append(actionable)
            if normalized is not None:
                evidence.append(normalized)
        if "UNAVAILABLE" in states:
            state = "UNAVAILABLE"
        elif states.count("SUPPORTIVE") >= 2 and "ADVERSE" not in states:
            state = "SUPPORTIVE"
        elif states.count("ADVERSE") >= 2:
            state = "ADVERSE"
        else:
            state = "MIXED"
        dimension_states[dimension] = state
        interpretation, trade_effect = TEXT[dimension][state]
        output[dimension] = dict(state=state, interpretation=interpretation, trade_effect=trade_effect,
                                 actionable_for_entry_support=all(actionable_metrics) and state != "UNAVAILABLE",
                                 invalidation="任一指标跨越相反历史分位、分母变化或数据过期即重新评估",
                                 evidence=evidence, missing=[metric for metric in definitions if metric not in {e["metric"] for e in evidence}])
    if any(state == "UNAVAILABLE" for state in dimension_states.values()):
        pattern = "INSUFFICIENT_EVIDENCE"
    elif sum(state == "ADVERSE" for state in dimension_states.values()) >= 3:
        pattern = "BROAD_ADVERSE_HYPOTHESIS"
    elif sum(state == "SUPPORTIVE" for state in dimension_states.values()) >= 4 and "ADVERSE" not in dimension_states.values():
        pattern = "BROAD_SUPPORTIVE_HYPOTHESIS"
    else:
        pattern = "ROTATION_OR_DIVERGENCE_HYPOTHESIS"
    entry_support = all(block["actionable_for_entry_support"] for block in output.values())
    return dict(status="DIAGNOSTIC_ONLY", asof=cutoff.isoformat(), scope_id=snapshot["scope_id"],
                pattern=pattern, dimension_states=dimension_states, emotion_analysis=output,
                entry_support_eligible=entry_support, cycle_state_probability=None, entry_decision=None,
                limitations=["historical percentile comparison is descriptive and not a fitted return model",
                             "verified is the supplier declaration, not independent source authentication",
                             "candidate ranking still requires theme, competition and opening execution evidence"])


def run(input_path, output_path, asof):
    source, target = Path(input_path), Path(output_path)
    if target.exists():
        raise FileExistsError("choose a new immutable emotion evidence output")
    payload = json.loads(source.read_text(encoding="utf-8"))
    result = build(payload, asof)
    result["source_hashes"] = {str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest(),
                               str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asof", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input,args.output,args.asof),ensure_ascii=False,indent=2))
