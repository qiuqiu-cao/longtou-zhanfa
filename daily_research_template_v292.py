"""Draft all fields honestly; UNKNOWN is a gap, never a completed forecast."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from daily_research_contract_v292 import (
    VERSION, ANCHORS, CONSENSUS_FIELDS, EVENTS, FEATURES, METRICS, TODAY_FIELDS, rule, unknown,
)


def empty_claim(reason="尚未建立可结算的事前判断", call=None, specification=None):
    value = {"call": call, "reason": reason, "basis": ["沿用原冻结晋级调用"] if call is not None else [],
        "invalidation": ["按完整下一交易日收盘池检验，缺结果不判失败"] if call is not None else [],
        "probability": None, "probability_unavailable_reason": "没有该目标、该范围的合格样本外概率校准",
        "probability_interval": None}
    if specification:
        value["label_rule"] = rule(specification)
    return value


def draft_extension(plan):
    spec = {"name": "CONSECUTIVE_BOARD_PROMOTION", "scope": plan.get("denominator_scope", "MAINBOARD_NON_ST_CLOSING_LIMIT_UP"),
        "positive": "same stock in exact next session complete closing limit-up pool with consecutive boards current+1",
        "negative": "all other verified complete-pool outcomes", "missing": "UNKNOWN"}
    candidates = []
    for c in plan["candidates"]:
        call = {"PROMOTE": True, "FAIL": False, "UNCERTAIN": None}[c["promotion_call"]]
        events = {event: empty_claim() for event in EVENTS}
        events["promotion"] = empty_claim(call=call, specification=spec)
        events["break"] = empty_claim(call=None if call is None else not call,
            specification={"name": "CONSECUTIVE_BOARD_BREAK", "definition": "logical complement of promotion, not intraday limit-touch failure"})
        candidates.append({"code": c["code"], "name": c.get("name"), "boards": c["boards"],
            "theme": unknown("没有经核验的当时题材归属"), "features": {k: unknown() for k in FEATURES}, "events": events})
    return {"version": VERSION, "draft_status": "NEEDS_FACTS_AND_RESEARCH_CALLS",
        "today": {k: unknown() for k in TODAY_FIELDS},
        "market_phase": empty_claim(), "emotion": empty_claim(), "driver": empty_claim(),
        "metrics": {k: {"baseline": unknown(), "band": None, "reason": "缺少同口径区间估计证据"} for k in METRICS},
        "candidates": candidates, "themes": [], "themes_unavailable_reason": "未核验题材完整范围与当时归属",
        "leaders": [{"code": None, "reason": "未取得足以独立判断龙头带动与承接的证据"} for _ in range(3)],
        "anchors": {k: {"code": None, "evidence": unknown(), "reason": "未核验该锚身份", "observe": ["身份和题材证据待核验"]} for k in ANCHORS},
        "plans": {k: {"conditions": ["待研究，未形成可量化的综合触发条件"], "observe": ["按原候选池记录，缺证据保持待确认"],
            "action": ["观察；不得由空白模板自动授权买入"], "cancel": ["原硬否决条件生效或必要证据缺失"]} for k in ["A", "B", "C"]},
        "consensus": {k: unknown("未取得同口径历史论坛观测；论坛意见不能充当资金事实") for k in CONSENSUS_FIELDS},
        "hypotheses": [{"name": name, "grade": "B", "core_rule": False, "status": "UNTESTED", "reason": "需时点历史及反例检验"}
            for name in ["高度压制", "星期/节前/月末", "断板日", "名称/生肖/代码", "地域/实控人", "历史股性/老龙反抽/妖股记忆"]],
        "model_changes": [], "automatic_entry_allowed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if "daily_research" in plan:
        raise ValueError("do not overwrite an existing extension")
    plan["daily_research"] = draft_extension(plan)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({"output": str(path.resolve()), "status": "DRAFT_NOT_REGISTERED"}))


if __name__ == "__main__":
    main()
