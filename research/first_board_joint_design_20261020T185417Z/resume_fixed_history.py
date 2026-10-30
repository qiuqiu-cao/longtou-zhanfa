"""Resume only never-requested dates after the user reauthorized joint training."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
import pandas as pd
import requests

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SOURCE=ROOT/"research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
RAW=OUT/"historical_raw"
def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
frame=pd.read_csv(SOURCE,usecols=["signal_date","code","stage"],dtype={"code":str})
days=sorted(frame.loc[frame.stage.eq(1),"signal_date"].unique())
old=[json.loads(line) for line in (OUT/"COLLECTION_RECEIPTS_APPEND_ONLY.jsonl").read_text("utf-8").splitlines() if line.strip()]
existing={r["date"]:r for r in old}
assert len(existing)==len(old)
for rec in old:
    if rec.get("source_path"): assert sha(rec["source_path"])==rec["sha256"]
for day,name in {"2024-10-15":"kpl_one_historical_day", "2025-01-02":"kpl_2025-01-02", "2026-08-28":"kpl_2026-08-28"}.items():
    if day not in existing:
        p=OUT/"raw"/(name+".response");d=json.loads(p.read_bytes())
        assert d["info"][1]==day and str(d["errcode"])=="0"
        existing[day]=dict(date=day,reused_original_probe=True,source_path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size,response_date=day,records=len(d["info"][0]),qualified_date_envelope=True)
# An unjournalled original response must never trigger a duplicate request.
for p in RAW.glob("*.response"):
    day=p.stem
    if day not in existing:
        d=json.loads(p.read_bytes())
        existing[day]=dict(date=day,recovered_unjournalled_original=True,source_path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size,response_date=d["info"][1],records=len(d["info"][0]),qualified_date_envelope=bool(d["info"][1]==day and str(d["errcode"])=="0"),original_request_clock_unavailable=True)
pending=[day for day in days if day not in existing]
protocol=dict(at=now(),authorization="User:AntigravityGeminiisrestored,monitorlastnightcollaboration,then:youandhimtraintogether", original_live_session="63716 confirmed missing;no process for old collector remained", original_receipts=len(old),preserved_source_dates=len(existing),new_never_requested_dates=pending,source_sha256=sha(SOURCE),
    no_old_date_retry=True,no_original_overwrite=True,no_labels_read=True,fits=0,workers=2,request_spacing_seconds=.4,stop="Single request per pending date;halt on403/429;10min checkpoint;preserve journal if interrupted")
with (OUT/"RESUME_PROTOCOL_BEFORE_REQUESTS_20261021.json").open("x",encoding="utf-8") as f: json.dump(protocol,f,ensure_ascii=False,indent=2)
lock=threading.Lock();halt=threading.Event();local=threading.local();next_time=[0.];began=time.monotonic()
def fetch(day):
    if halt.is_set() or time.monotonic()-began>600:return dict(date=day,status="NOT_REQUESTED_CHECKPOINT_OR_DENIAL")
    with lock:
        delay=max(0,next_time[0]-time.monotonic())
        if delay:time.sleep(delay)
        next_time[0]=time.monotonic()+.4
    if halt.is_set():return dict(date=day,status="NOT_REQUESTED_AFTER_DENIAL")
    if not hasattr(local,"session"):
        local.session=requests.Session();local.session.headers.update({"User-Agent":"LongtouResearch/1.0 (public read-only source qualification)"})
    rec=dict(date=day,requested_at=now(),attempts=1)
    try:
        params=dict(Order=0,st=1000,a="DailyLimitPerformance",c="HisHomeDingPan",PhoneOSNew=1,VerSion="5.7.0.12",Index=0,PidType=1,apiv="w31",Type=4,Day=day)
        response=local.session.get("https://apphis.longhuvip.com/w1/api/index.php",params=params,timeout=12)
        p=RAW/(day+".response")
        with p.open("xb") as f:f.write(response.content)
        rec.update(received_at=now(),http_status=response.status_code,source_path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size)
        if response.status_code in (403,429):halt.set()
        if response.status_code==200:
            d=response.json();info=d.get("info");values=info[0] if isinstance(info,list) and len(info)>1 and isinstance(info[0],list) else []
            rec.update(response_date=info[1] if isinstance(info,list) and len(info)>1 else None,errcode=d.get("errcode"),records=len(values))
            rec["qualified_date_envelope"]=bool(rec["response_date"]==day and str(d.get("errcode"))=="0")
    except Exception as e:rec.update(received_at=now(),error=type(e).__name__,detail=str(e)[:240],qualified_date_envelope=False)
    return rec
with (OUT/"RESUME_RECEIPTS_APPEND_ONLY_20261021.jsonl").open("x",encoding="utf-8") as log,ThreadPoolExecutor(max_workers=2) as pool:
    for future in as_completed([pool.submit(fetch,d) for d in pending]):
        rec=future.result();existing[rec["date"]]=rec;log.write(json.dumps(rec,ensure_ascii=False)+"\n");log.flush()
        if len(existing)%40==0 or len(existing)==len(days):print(json.dumps(dict(at=now(),recorded_dates=len(existing),total_dates=len(days),valid_dates=sum(bool(r.get("qualified_date_envelope")) for r in existing.values()),elapsed_seconds=round(time.monotonic()-began,1))),flush=True)
records=[existing[day] for day in days]
with (OUT/"ALL668_SOURCE_RECEIPTS.json").open("x",encoding="utf-8") as f:json.dump(records,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status="COLLECTION_COMPLETED",dates=len(records),valid_dates=sum(bool(r.get("qualified_date_envelope")) for r in records),fits=0)),flush=True)
