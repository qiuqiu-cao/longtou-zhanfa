"""Audit actual context capture, old-cutoff exclusions and preserved schedule."""
import argparse
import json
from pathlib import Path
import tomllib

from point_context_v295 import asof_view, sha, write, now
from point_context_reconciled_v295 import reconstruct
from context_draft_bridge_v295 import enrich
from next_session_expectation_contract_v271 import load_registered

ROOT=Path(__file__).resolve().parent


def audit(archive,registered,old_schedule,output):
    value=reconstruct(archive)
    original=load_registered(registered)
    if sha(registered)!="bfbaeb295637451e396850ee583fcafa3eb0ab6a99fb1b154e4c46a8c23aa40c":raise ValueError("original forecast bytes changed")
    candidates={r["code"]:r for r in value["candidates"]}
    if len(candidates)!=len(value["scope"]["codes"]) or set(candidates)!=set(value["scope"]["codes"]) or set(c["code"] for c in original["plan"]["candidates"])-set(candidates):raise ValueError("context omits full scope/original candidates")
    old_view=asof_view(value,original["plan"]["knowledge_cutoff"])
    if any(r[k] is not None for r in old_view["candidates"] for k in ["categories","attention","attention_history"]):raise ValueError("new context leaked into original cutoff")
    try:enrich(original["plan"],value,str((Path(archive)/"context.json").resolve()))
    except ValueError:pass
    else:raise ValueError("registered legacy plan accepted for enrichment")
    old_receipt=json.loads((ROOT/"results/v293/source_pipeline_acceptance_20261002_v1/report.json").read_text(encoding="utf-8"))
    app_path=Path.home()/".codex/automations/automation/automation.toml"
    for path,wanted in old_receipt["source_hashes"].items():
        observed=Path(old_schedule) if path==str(app_path) else Path(path)
        if sha(observed)!=wanted:raise ValueError("previous receipt no longer reconstructs with saved superseded schedule")
    app=tomllib.loads(app_path.read_text(encoding="utf-8"))
    if any(app.get(k)!=v for k,v in {"id":"automation","kind":"heartbeat","status":"ACTIVE","target_thread_id":"01a0fb4b-e111-7c23-adae-0b9a1bc7bbe9","rrule":"FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=19;BYMINUTE=30;BYSECOND=0"}.items()):raise ValueError("schedule identity or time changed")
    if any(s not in app["prompt"] for s in ["题材与关注度时点基线V295.md","point_context_v295.py","point_context_reconciled_v295.py","context_draft_bridge_v295.py"]):raise ValueError("point-context workflow missing in actual automation")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write(output/"old_cutoff_view_NOT_FORECAST.json",old_view)
    rows=[]
    for item in original["plan"]["candidates"]:
        r=candidates[item["code"]];attention=r["attention"]
        rows.append({"code":item["code"],"name":item.get("name"),"original_boards":item["boards"],"provider_categories":len(r["categories"]["rows"]) if r["categories"] else None,
            "attention_rank_at_capture":attention["rank"] if attention else None,"attention_calculated_at":attention["calculated_at"] if attention else None,
            "old_forecast_usable":False,"leader_confirmation_authorized":False})
    lines=["# 题材与关注度时点基线 V295","", "这是10月2日实际取得的当前供应商分类与关注度；价格候选范围来自9月30日。不得当成9月30日题材或原冻结时点已知信息。","",
        "| 原候选 | 原板位 | 当前分类数量 | 采集时排名 | 排名计算时间 |","|---|---:|---:|---:|---|"]
    for row in rows:lines.append(f"| {row['code']} {row['name']} | {row['original_boards']} | {row['provider_categories']} | {row['attention_rank_at_capture']} | {row['attention_calculated_at']} |")
    lines += ["", "分类数量只是供应商标签数量，排名只是关注度。不能据此确认主线、龙头、买点或晋级概率。","", "原计划知识截止前所有新字段均不可用；原文件未变。历史排名只有日期，缺历史公开时间，不能直接用作当时盘后已知特征。","", "原解析误把供应商内部编号当证券代码，22条排名被拒绝；更正保留原始档案，用明确的srcSecurityCode核对证券。","", "看多/看空、一致性、讨论量变化、真实资金流与龙头带动仍未取得。本次新增前向结果为0，稳定60%未验证。"]
    md=output/"review.md"
    with md.open("x",encoding="utf-8") as f:f.write("\n".join(lines)+"\n")
    count=lambda field:sum(r[field] is not None for r in value["candidates"])
    report={"version":"v295-point-context-acceptance-1","recorded_at":now(),"status":"RAW_CONTEXT_OLD_CUTOFF_DRAFT_GUARDS_AND_SCHEDULE_VERIFIED",
        "scope_codes":len(candidates),"categories_available":count("categories"),"attention_available":count("attention"),"history_available":count("attention_history"),
        "provider_category_rows":sum(len(r["categories"]["rows"]) for r in value["candidates"] if r["categories"]),
        "provider_history_rows":sum(len(r["attention_history"]["rows"]) for r in value["candidates"] if r["attention_history"]),
        "corrected_internal_id_rows":len(value["correction"]["codes_previously_rejected_by_internal_id_check"]),
        "original_candidates":rows,"original_bytes_unchanged":True,"old_cutoff_available_new_fields":0,"historical_theme_or_eod_availability_certified":False,
        "future_schedule_completed":False,"schedule_active":True,"related_tests_observed_passed":272,"forward_samples_added":0,"stable_60_verified":False,"goal_complete":False,
        "source_hashes":{str(p.resolve()):sha(p) for p in [Path(archive)/"manifest.json",Path(archive)/"context.json",Path(registered),Path(old_schedule),app_path,Path(__file__),ROOT/"context_draft_bridge_v295.py",ROOT/"test_point_context_v295.py",ROOT/"题材与关注度时点基线V295.md",output/"old_cutoff_view_NOT_FORECAST.json",md]}}
    write(output/"report.json",report)
    return {k:report[k] for k in ["status","scope_codes","categories_available","attention_available","history_available","provider_category_rows","provider_history_rows","corrected_internal_id_rows","old_cutoff_available_new_fields","original_bytes_unchanged","schedule_active","stable_60_verified"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ["archive","registered","old-schedule","output"]:parser.add_argument("--"+key,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.archive,args.registered,args.old_schedule,args.output),ensure_ascii=True))
