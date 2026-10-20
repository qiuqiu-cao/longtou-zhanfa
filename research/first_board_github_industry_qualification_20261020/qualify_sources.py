"""Bounded public GitHub metadata/data capture. No repository code execution."""
from pathlib import Path
import concurrent.futures
import datetime as dt
import hashlib
import json
import requests

ROOT=Path(__file__).resolve().parent
REPOS={"stock_basics":"harry2002731/Stock-programe","knowledge_graph":"lemonhu/stock-knowledge-graph"}

def now():return dt.datetime.now(dt.timezone.utc).isoformat()

def capture(name,url):
    receipt=ROOT/(name+"_receipt.json")
    if receipt.exists():
        old=json.loads(receipt.read_text("utf-8"))
        return old
    record=dict(source_url=url,requested_at=now(),one_attempt=True)
    receipt.write_text(json.dumps(record,indent=2),encoding="utf-8")
    try:
        response=requests.get(url,timeout=(5,12),headers={"Accept":"application/vnd.github+json","User-Agent":"Historical-source-qualification"})
        raw=ROOT/(name+".response")
        raw.write_bytes(response.content)
        record.update(received_at=now(),status=response.status_code,bytes=len(response.content),sha256=hashlib.sha256(response.content).hexdigest(),raw_file=raw.name,final_url=response.url)
        if response.headers.get("X-RateLimit-Remaining") is not None:
            record["api_rate_limit_remaining"]=response.headers.get("X-RateLimit-Remaining")
    except Exception as exc:record.update(completed_at=now(),error=repr(exc))
    receipt.write_text(json.dumps(record,indent=2),encoding="utf-8")
    return record

def initialize():
    protocol=ROOT/"qualification_protocol_BEFORE_DATA.json"
    if not protocol.exists():
        value=dict(formed_at=now(),batch_started_at="2026-10-20T14:03:51+00:00",budget_minutes=30,
            previous_goal_turn="PROGRESS:two feasible-training arms rejected after4cores;allmembers/queryweights/scores verified. Need distinct information,not further tuning.",
            task="Qualify immutable historical industry membership data only",repos=REPOS,
            maximum_repositories=2,maximum_static_data_blobs=4,
            qualification="Require immutable content,dated category semantics,full code scope and credible pre-signal publication/effective evidence. Git authored/committer dates alone are not publication proof. Host metadata,release assets or independently archived evidence evaluated separately.",
            freeze_semantics="An old immutable classification can only be a named legacy industry proxy;cannot certify current industry,active theme or fund identities. Newly listed/uncovered codes UNKNOWN.",
            no_downloaded_repository_code_execution=True,no_private_keys_or_tokens=True,no_current_backfill=True,
            source_retrieval_late=True,fits=0,production_changes=False,goal_60_passed=False,
            stop="At2repos/4datablobs,qualification decision or30minutes. No retry of unchanged failed requests,no refitting old schemes.")
        protocol.write_text(json.dumps(value,indent=2),encoding="utf-8")

if __name__=="__main__":
    initialize()
    urls=[]
    for name,repo in REPOS.items():
        urls.extend([(name+"_metadata","https://api.github.com/repos/"+repo),
                     (name+"_releases","https://api.github.com/repos/"+repo+"/releases?per_page=5")])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for name,record in zip([x[0] for x in urls],pool.map(lambda item:capture(*item),urls)):
            if record.get("status")==200:
                value=json.loads((ROOT/record["raw_file"]).read_text("utf-8"))
                result={k:value.get(k) for k in ("full_name","created_at","updated_at","pushed_at","default_branch","private","fork","archived") } if isinstance(value,dict) else {"releases":[{k:v.get(k) for k in ("tag_name","created_at","published_at","assets")} for v in value]}
            else:result=record
            print(json.dumps({"name":name,"result":result},ensure_ascii=True),flush=True)
