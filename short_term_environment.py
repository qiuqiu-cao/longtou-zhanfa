"""Evidence-based environment changes; hypotheses, never calibrated trade odds."""
import math

from leader_holding_review_v21 import stamp


# Direction of improvement, not fitted weights or trading thresholds.
METRICS = {
    "relay_premium": (1, "昨日连板股当前平均涨幅", "pct"),
    "broken_tail_return": (1, "昨日断板股当前收益10%分位", "pct"),
    "mid_promotion_rate": (1, "昨日中位股当前封板比例", "ratio"),
    "high_promotion_rate": (1, "昨日高位股当前封板比例", "ratio"),
    "failed_board_rate": (-1, "当日触板后当前未封比例", "ratio"),
    "theme_support_rate": (1, "候选细分题材固定支持池当前封板比例", "ratio"),
}


def analyze(snapshot, asof, previous=None):
    cutoff = stamp(asof)
    if snapshot is None:
        return {"status": "UNAVAILABLE", "missing": list(METRICS),
                "hypothesis": "INSUFFICIENT_EVIDENCE", "premium_probability": None}
    if previous is not None:
        if snapshot.get("scope_id") != previous.get("scope_id"):
            raise ValueError("environment scopes differ")
    def extract(bundle):
        if not bundle.get("scope_id"):
            raise ValueError("explicit environment scope_id required")
        result = {}
        for key, (_, label, unit) in METRICS.items():
            item = bundle.get("metrics", {}).get(key)
            if item is None:
                continue
            observed, available = stamp(item["observed_at"]), stamp(item["available_at"])
            if observed > available or available > cutoff:
                raise ValueError("future or inconsistent environment evidence")
            value = item.get("value")
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("finite numeric evidence required")
            if item.get("unit") != unit or (unit == "ratio" and not 0 <= value <= 1):
                raise ValueError("incorrect metric unit or range")
            n = item.get("sample_count")
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ValueError("positive cohort sample_count required")
            if not item.get("definition_id") or not item.get("cohort_id") or not item.get("source"):
                raise ValueError("definition, cohort and source required")
            if item.get("data_status") not in {"verified", "delayed", "inferred", "unavailable"}:
                raise ValueError("unknown data status")
            result[key] = dict(item, label=label,
                               fresh=observed.date() == cutoff.date() and
                               (cutoff-observed).total_seconds() <= 300)
        return result
    current = extract(snapshot)
    prior = extract(previous) if previous is not None else {}
    evidence, directions = {}, {}
    for key, item in current.items():
        old = prior.get(key)
        reason = "NO_COMPARABLE_PRIOR"
        delta = None
        if old:
            if stamp(old["observed_at"]) >= stamp(item["observed_at"]):
                raise ValueError("previous evidence must precede current")
            comparable = all(item[k] == old[k] for k in
                             ("definition_id", "cohort_id", "sample_count", "source", "unit"))
            same_session = stamp(old["observed_at"]).date() == stamp(item["observed_at"]).date()
            if comparable and same_session:
                delta = item["value"] - old["value"]
                reason = "OBSERVED_CHANGE"
                if item["fresh"] and item["data_status"] == old["data_status"] == "verified":
                    signed = delta * METRICS[key][0]
                    directions[key] = "improving" if signed > 0 else "worsening" if signed < 0 else "unchanged"
        evidence[key] = {"current": item, "previous": old, "delta": delta,
                         "comparison": reason, "direction": directions.get(key, "unconfirmed")}
    core = ("relay_premium", "broken_tail_return", "mid_promotion_rate", "high_promotion_rate")
    pattern = "INSUFFICIENT_EVIDENCE"
    if all(k in directions for k in core):
        if all(directions[k] == "worsening" for k in core):
            pattern = "BROAD_DETERIORATION_HYPOTHESIS"
        elif all(directions[k] == "improving" for k in core):
            pattern = "BROAD_REPAIR_HYPOTHESIS"
        else:
            pattern = "MIXED_FEEDBACK"
    return {"status": "DIAGNOSTIC_ONLY", "scope_id": snapshot["scope_id"],
            "asof": cutoff.isoformat(), "evidence": evidence,
            "missing": [k for k in METRICS if k not in current],
            "unconfirmed": [k for k in METRICS if k not in directions],
            "hypothesis": pattern, "premium_probability": None,
            "counter_explanation": "群体同步变化也可能来自市场波动；不能据此确定资金意图或唯一龙头。",
            "invalidation": "后续同口径反馈反向、支持池变动或数据过期，须重新分析。",
            "candidate_confirmation": "竞争性分歧还需个股承接、同高度竞争和细分题材同步证据；市场情绪不能单独排名。",
            "limitations": ["涨幅是盯市反馈，不是按实际买卖规则可兑现的净溢价",
                            "变化方向未设统计显著性门槛，不等于周期状态或买卖信号",
                            "verified为输入方声明，本模块不独立认证数据来源",
                            "不同交易日或不同样本池不直接比较，避免成分变化伪造改善"]}
