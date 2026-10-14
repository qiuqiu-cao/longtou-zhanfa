"""Archive time-bound provider categories and attention, never historical facts."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import requests

from next_session_expectation_contract_v271 import load_registered, digest
from public_outcome_sources_v293 import reconstruct as reconstruct_pool
from daily_research_contract_v292 import stamp

VERSION="v295-point-context-1"
ROOT=Path(__file__).resolve().parent
ZONE=timezone(timedelta(hours=8))
APP={"appId":"appId01","globalId":"786e4c21-70dc-435a-93bb-38","marketType":""}
KINDS=["category_page","category_table","attention_latest","attention_history"]


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open("x",encoding="utf-8") as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def symbol(code):return ("SH" if code.startswith("6") else "SZ")+code
def integer(value,minimum=0):
    if isinstance(value,bool) or not isinstance(value,int) or value<minimum:raise ValueError("integer required")
    return value


def request(code,kind):
    if kind=="category_page":return "GET","https://emweb.securities.eastmoney.com/PC_HSF10/CoreConception/PageAjax",{"code":symbol(code)}
    if kind=="category_table":return "GET","https://datacenter-web.eastmoney.com/api/data/v1/get",{"reportName":"RPT_F10_CORETHEME_BOARDTYPE","columns":"ALL","filter":'(SECURITY_CODE="'+code+'")',"pageSize":1000,"pageNumber":1,"source":"WEB","client":"WEB"}
    endpoint="getCurrentLatest" if kind=="attention_latest" else "getHisList"
    payload={**APP,"srcSecurityCode":symbol(code)}
    if kind=="attention_history":payload["yearType"]="5"
    return "POST","https://emappdata.eastmoney.com/stockrank/"+endpoint,payload


def fetch(code,kind,directory):
    method,url,params=request(code,kind)
    item={"code":code,"kind":kind,"method":method,"url":url,"query":params,"requested_at":now()}
    try:
        response=requests.request(method,url,params=params if method=="GET" else None,json=params if method=="POST" else None,timeout=(5,8))
        path=directory/"raw"/(code+"_"+kind+".response")
        with path.open("xb") as f:f.write(response.content)
        item.update(available_at=now(),http_status=response.status_code,raw_path=str(path.resolve()),raw_sha256=sha(path))
        response.raise_for_status()
        response.json()
        item["status"]="RAW_JSON_RECEIVED"
    except Exception as exc:
        item.update(status="UNAVAILABLE",available_at=now(),error=str(exc)[:650])
    return item


def scope_from_registered(path):
    original=load_registered(path)
    return {"codes":sorted(c["code"] for c in original["plan"]["candidates"]),"price_signal_date":original["plan"]["signal_date"],
        "kind":"ALL_ORIGINAL_REGISTERED_RELAY_CANDIDATES", "source_path":str(Path(path).resolve()),"source_sha256":sha(path),
        "registration_hash":original["sha256"],"official_full_market":False}


def scope_from_pool(directory,date):
    _,pools,_,_=reconstruct_pool(directory)
    pool=pools.get((date,"limit_up"))
    if not pool:raise ValueError("exact-date raw-verified pool required")
    return {"codes":sorted(r["code"] for r in pool["rows"]),"price_signal_date":date,"kind":"FULL_PROVIDER_MAINBOARD_NAME_NON_ST_LIMIT_POOL",
        "source_path":str(Path(directory).resolve()),"manifest_sha256":sha(Path(directory)/"manifest.json"),"official_full_market":False}


def category_rows(rows,code):
    if not isinstance(rows,list):raise ValueError("category array required")
    result={}
    for row in rows:
        if row.get("SECURITY_CODE")!=code or row.get("SECUCODE")!=code+(".SH" if code.startswith("6") else ".SZ"):
            raise ValueError("category security identity mismatch")
        key=row.get("BOARD_CODE");name=row.get("BOARD_NAME")
        if not isinstance(key,str) or not key.isdigit() or not isinstance(name,str) or not name.strip() or key in result:
            raise ValueError("missing or duplicate provider category identity")
        result[key]={"board_code":key,"board_name":name,"precision_flag_provider":row.get("IS_PRECISE"),
            "classification_reason_provider_text":row.get("SELECTED_BOARD_REASON"),"official_business_fact_verified":False}
    return result


def parse_categories(page,table,code):
    if table.get("success") is not True or table.get("code")!=0 or not isinstance(table.get("result"),dict):
        raise ValueError("category response not successful")
    data=table["result"]
    if integer(data.get("pages"))!=1 or integer(data.get("count"))!=len(data.get("data",[])):
        raise ValueError("category pagination/count incomplete")
    left=category_rows(page.get("ssbk"),code);right=category_rows(data["data"],code)
    if {k:r["board_name"] for k,r in left.items()}!={k:r["board_name"] for k,r in right.items()}:
        raise ValueError("provider category endpoints disagree")
    if not right:raise ValueError("empty categories are not certified no-theme")
    return {"status":"PROVIDER_CURRENT_CATEGORIES_RECONCILED","rows":[right[k] for k in sorted(right)],
        "historical_membership_verified":False,"active_trading_theme_verified":False,"independent_provider":False,
        "publication_time":None,"category_effective_date":None}


def attention_data(value):
    if value.get("status")!=0 or value.get("code")!=0 or value.get("message")!="OK":raise ValueError("attention response unsuccessful")
    return value.get("data")


def parse_latest(value,code,available_at):
    data=attention_data(value)
    if not isinstance(data,dict) or data.get("srcSecurityCode")!=symbol(code) or data.get("innerCode")!=code:
        raise ValueError("attention security identity mismatch")
    rank=integer(data.get("rank"),1);total=integer(data.get("marketAllCount"),1)
    if rank>total:raise ValueError("attention rank outside provider universe")
    calculated=datetime.strptime(data["calcTime"],"%Y-%m-%d %H:%M:%S").replace(tzinfo=ZONE)
    age=(stamp(available_at)-calculated).total_seconds()
    if age<0:raise ValueError("attention calculation timestamp is in the future")
    return {"status":"PROVIDER_RANK_TIMESTAMP_CHECKED" if age<=1200 else "DELAYED_PROVIDER_RANK",
        "rank":rank,"provider_universe_count":total,"calculated_at":calculated.isoformat(),"age_seconds_at_capture":age,
        "provider_reported_changes":{k:data.get(k) for k in ["rankChange","hisRankChange"]},"change_semantics_verified":False,
        "bullishness":None,"bearishness":None,"discussion_count":None,"consensus_unanimity":None,
        "interpretation":"provider attention rank only; not order flow, sentiment or leader confirmation"}


def parse_history(value,code,available_at):
    rows=attention_data(value)
    if not isinstance(rows,list) or not rows:raise ValueError("history empty or unavailable")
    result=[];seen=set()
    for row in rows:
        date=row.get("calcTime")
        if not isinstance(date,str) or datetime.strptime(date,"%Y-%m-%d").strftime("%Y-%m-%d")!=date or date in seen:
            raise ValueError("duplicate or invalid attention history date")
        if date>stamp(available_at).astimezone(ZONE).date().isoformat():raise ValueError("future attention history date")
        seen.add(date);result.append({"date":date,"rank":integer(row.get("rank"),1)})
    if [r["date"] for r in result]!=sorted(seen):raise ValueError("unordered attention history")
    return {"status":"RETROSPECTIVE_DATE_ONLY_PROVIDER_HISTORY","rows":result,"code":code,
        "code_binding":"REQUEST_PARAMETER_ONLY; no security code in each returned history row",
        "publication_time":None,"original_eod_availability_verified":False,"forward_feature_authorized":False}


def normalize(scope,entries,recorded_at):
    index={(e["code"],e["kind"]):e for e in entries}
    if len(index)!=len(entries) or set(index)!={(c,k) for c in scope["codes"] for k in KINDS}:
        raise ValueError("request scope missing, extra or duplicate")
    normalized=[]
    for code in scope["codes"]:
        def get(kind):
            entry=index[code,kind]
            if entry["status"]!="RAW_JSON_RECEIVED":raise ValueError(entry.get("error","request unavailable"))
            return json.loads(Path(entry["raw_path"]).read_bytes())
        row={"code":code,"captured_at":recorded_at,"price_signal_date":scope["price_signal_date"],"categories":None,"attention":None,"attention_history":None,"errors":{}}
        for key,func in [("categories",lambda:parse_categories(get("category_page"),get("category_table"),code)),
                         ("attention",lambda:parse_latest(get("attention_latest"),code,index[code,"attention_latest"]["available_at"])),
                         ("attention_history",lambda:parse_history(get("attention_history"),code,index[code,"attention_history"]["available_at"]))]:
            try:
                row[key]=func()
                dependencies=["category_page","category_table"] if key=="categories" else ["attention_latest" if key=="attention" else "attention_history"]
                row[key]["available_at"]=max((index[code,k]["available_at"] for k in dependencies),key=stamp)
                row[key]["raw_sources"]=[index[code,k]["raw_path"] for k in dependencies]
            except (ValueError,KeyError,TypeError) as exc:row["errors"][key]=str(exc)[:700]
        normalized.append(row)
    return {"version":VERSION,"recorded_at":recorded_at,"scope":scope,"candidates":normalized,
        "current_membership_not_backfilled":True,"market_consensus_truth_verified":False,"leader_confirmation_authorized":False,
        "probability_adjustment_authorized":False,"trade_authorized":False}


def reconstruct(directory):
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if manifest["version"]!=VERSION or manifest["source_hashes"].get(str(Path(__file__).resolve()))!=sha(__file__):
        raise ValueError("context implementation hash changed")
    original_scope=manifest["scope"]
    if original_scope["kind"]=="ALL_ORIGINAL_REGISTERED_RELAY_CANDIDATES":fresh=scope_from_registered(original_scope["source_path"])
    else:fresh=scope_from_pool(original_scope["source_path"],original_scope["price_signal_date"])
    if fresh!=original_scope:raise ValueError("original context scope changed")
    for entry in manifest["requests"]:
        method,url,query=request(entry["code"],entry["kind"])
        if (entry["method"],entry["url"],entry["query"])!=(method,url,query):raise ValueError("request code/method/parameters changed")
        if not stamp(entry["requested_at"])<=stamp(entry["available_at"])<=stamp(manifest["recorded_at"]):raise ValueError("request clock invalid")
        if entry.get("raw_path"):
            path=Path(entry["raw_path"]).resolve()
            if not path.is_relative_to((directory/"raw").resolve()) or sha(path)!=entry["raw_sha256"]:raise ValueError("raw context bytes changed")
        if entry["status"]=="RAW_JSON_RECEIVED" and (entry.get("http_status")!=200 or not entry.get("raw_path")):
            raise ValueError("successful context request has no raw evidence")
    fresh=normalize(original_scope,manifest["requests"],manifest["recorded_at"])
    stored=json.loads((directory/"context.json").read_text(encoding="utf-8"))
    if digest(fresh)!=digest(stored):raise ValueError("context fails original raw reconstruction")
    return fresh


def asof_view(value,cutoff):
    limit=stamp(cutoff);rows=[]
    for original in value["candidates"]:
        row={"code":original["code"],"categories":None,"attention":None,"attention_history":None,"unavailable":{}}
        for field in ["categories","attention","attention_history"]:
            item=original[field]
            if item is None:row["unavailable"][field]=original["errors"].get(field,"missing")
            elif stamp(item["available_at"])>limit:row["unavailable"][field]="FIRST_OBSERVED_AFTER_KNOWLEDGE_CUTOFF"
            elif field=="attention" and (limit-stamp(item["calculated_at"])).total_seconds()>1200:
                row["unavailable"][field]="STALE_AT_KNOWLEDGE_CUTOFF; recollect current rank"
            elif field=="categories" and (limit-stamp(item["available_at"])).total_seconds()>86400:
                row["unavailable"][field]="CAPTURE_OLDER_THAN_24H_POLICY; recollect categories"
            else:row[field]=item
        rows.append(row)
    return {"version":VERSION,"knowledge_cutoff":cutoff,"candidates":rows,"original_forecast_rewritten":False,
        "leader_confirmation_authorized":False,"probability_adjustment_authorized":False,"trade_authorized":False}


def collect(scope,output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/"raw").mkdir()
    def one(code):return [fetch(code,kind,output) for kind in KINDS]
    entries=[]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for batch in executor.map(one,scope["codes"]):
            entries.extend(batch)
            print(json.dumps({"finished_codes":len(entries)//4,"total_codes":len(scope["codes"])}),flush=True)
    recorded=now()
    value=normalize(scope,entries,recorded)
    write(output/"context.json",value)
    manifest={"version":VERSION,"scope":scope,"recorded_at":recorded,"requests":entries,
        "source_hashes":{str(Path(__file__).resolve()):sha(__file__)},"no_old_forecast_update":True}
    write(output/"manifest.json",manifest)
    reconstruct(output)
    return {"status":"CONTEXT_RAW_RECONSTRUCTION_PASSED","codes":len(scope["codes"]),
        "categories_available":sum(r["categories"] is not None for r in value["candidates"]),
        "attention_available":sum(r["attention"] is not None for r in value["candidates"]),
        "attention_history_available":sum(r["attention_history"] is not None for r in value["candidates"]),"recorded_at":recorded}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    c=commands.add_parser("collect");group=c.add_mutually_exclusive_group(required=True)
    group.add_argument("--registered");group.add_argument("--pool-archive");c.add_argument("--pool-date");c.add_argument("--output",required=True)
    v=commands.add_parser("view");v.add_argument("--archive",required=True);v.add_argument("--knowledge-cutoff",required=True);v.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="collect":
        if args.pool_archive and not args.pool_date:parser.error("--pool-date required for exact-date scope")
        scope=scope_from_registered(args.registered) if args.registered else scope_from_pool(args.pool_archive,args.pool_date)
        print(json.dumps(collect(scope,args.output),ensure_ascii=True))
    else:
        value=asof_view(reconstruct(args.archive),args.knowledge_cutoff);write(args.output,value)
        print(json.dumps({"codes":len(value["candidates"]),"category_available":sum(r["categories"] is not None for r in value["candidates"])}))
