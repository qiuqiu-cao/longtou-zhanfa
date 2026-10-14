"""Enrich only a new same-session draft; attention never confirms a leader."""
from copy import deepcopy
from datetime import timezone, timedelta
import argparse
import json
from pathlib import Path

from point_context_v295 import asof_view, sha, write
from point_context_reconciled_v295 import reconstruct
from daily_research_contract_v292 import stamp, unknown

ZONE=timezone(timedelta(hours=8))


def enrich(plan,context,source):
    result=deepcopy(plan)
    ext=result.get("daily_research")
    if not ext or ext.get("draft_status")!="NEEDS_FACTS_AND_RESEARCH_CALLS":
        raise ValueError("only an unreviewed new daily draft can receive context")
    date=plan["signal_date"]
    if context["scope"]["price_signal_date"]!=date:
        raise ValueError("context price scope differs from new draft signal date")
    cutoff=plan.get("knowledge_cutoff",plan["evidence_cutoff"])
    viewed=asof_view(context,cutoff)
    by_code={r["code"]:r for r in viewed["candidates"]}
    if len(by_code)!=len(viewed["candidates"]) or any(c["code"] not in by_code for c in plan["candidates"]):
        raise ValueError("context omits original draft candidates")
    # Current classification has no historical effective date. A later-day
    # capture must not be presented as membership observed on the signal day.
    for row in viewed["candidates"]:
        for key in ["categories","attention"]:
            item=row[key]
            if item is not None and stamp(item["available_at"]).astimezone(ZONE).date().isoformat()!=date:
                raise ValueError("later/earlier-day context cannot establish signal-day membership or attention")
    attention={};clocks=[];filled=[]
    for candidate in ext["candidates"]:
        row=by_code[candidate["code"]]
        item=row["categories"]
        if item:
            candidate["theme"]={"status":"verified","source":source,"observed_at":item["available_at"],"available_at":item["available_at"],
                "value":{"provider_board_memberships":item["rows"],"active_trading_theme":None,"official_business_fact_verified":False},
                "reason":"供应商当前分类已按原始响应核对，尚未确认活跃交易题材或龙头身份"}
            filled.append(candidate["code"])
        item=row["attention"]
        if item:
            attention[candidate["code"]]={"rank":item["rank"],"provider_universe_count":item["provider_universe_count"],"calculated_at":item["calculated_at"]}
            clocks.append((item["calculated_at"],item["available_at"]))
    if attention:
        ext["consensus"]["attention"]={"status":"verified","source":source,
            "observed_at":max((p[0] for p in clocks),key=stamp),"available_at":max((p[1] for p in clocks),key=stamp),
            "value":attention,"reason":"供应商关注度排名；不等于资金流、讨论量、看多一致性或龙头确认"}
    else:ext["consensus"]["attention"]=unknown("该知识截止前没有未过期的供应商关注度观测")
    result["context_binding"]={"source":source,"source_sha256":sha(source),"category_codes_filled":filled,
        "attention_codes_filled":sorted(attention),"history_used_in_live_features":False,"author_review_still_required":True}
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["plan","archive","output"]:parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text(encoding="utf-8"))
    context=reconstruct(args.archive)
    result=enrich(plan,context,str((Path(args.archive)/"context.json").resolve()))
    write(args.output,result)
    print(json.dumps({"status":"UNREGISTERED_DRAFT_ENRICHED_REVIEW_REQUIRED","binding":result["context_binding"]},ensure_ascii=True))
