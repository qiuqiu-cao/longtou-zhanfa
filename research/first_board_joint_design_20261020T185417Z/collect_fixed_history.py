"""Acquire each date in the frozen cohort once; no outcomes, retries or changing universe."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
import pandas as pd
import requests

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
RAW = OUT / "historical_raw"
RAW.mkdir(exist_ok=True)
def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
frame = pd.read_csv(SOURCE,usecols=["signal_date","code","stage"],dtype={"code":str})
frame = frame[frame.stage.eq(1)]
days = sorted(frame.signal_date.unique())
assert len(frame)==28503 and len(days)==668 and not frame.duplicated(["signal_date","code"]).any()
protocol = dict(at=now(), task="Complete frozen668date historical first-board public-source qualification", outcomes_read=False, fits=0,
    source_sha256=sha(SOURCE), rows=len(frame), dates=days, request_attempts_per_date=1, workers=2, min_request_spacing_seconds=.4,
    no_retries=True, no_auth_or_real_device_identifier=True, stop="Stop initiating new requests on403/429 or12minute checkpoint;retain partial status,never treat timeout as completion or restart live process",
    fields_pending_joint_review="Proposed first-seal time and reported close/max-seal funds only;quarantine retrospectiveconcepts/names/flowidentity/unmappedflags;no finalseal/bomb count invented",
    fixed_qualification_gate="Development and2025H1 timestamp/close-seal-fund matching coverage>=90%,no outcome-based row/date selection;source failures missing,full28503 retained")
with (OUT / "COLLECTION_PROTOCOL_BEFORE_REQUESTS.json").open("x",encoding="utf-8") as f: json.dump(protocol,f,ensure_ascii=False,indent=2)
cached = {
    "2024-10-15":OUT/"raw/kpl_one_historical_day.response",
    "2025-01-02":OUT/"raw/kpl_2025-01-02.response",
    "2026-08-28":OUT/"raw/kpl_2026-08-28.response",
}
clock_lock=threading.Lock(); halt=threading.Event(); thread_local=threading.local(); next_time=[0.0]; began=time.monotonic()
api="https://apphis.longhuvip.com/w1/api/index.php"
def fetch(day):
    if day in cached:
        path=cached[day]; payload=json.loads(path.read_bytes())
        assert payload["info"][1]==day and str(payload["errcode"])=="0"
        return dict(date=day,reused_original_probe=True,source_path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size,response_date=payload["info"][1],records=len(payload["info"][0]),qualified_date_envelope=True)
    if halt.is_set() or time.monotonic()-began>12*60:
        return dict(date=day,status="NOT_REQUESTED_CHECKPOINT_OR_ACCESS_DENIAL",fits=0)
    with clock_lock:
        wait=max(0,next_time[0]-time.monotonic())
        if wait: time.sleep(wait)
        next_time[0]=time.monotonic()+.4
    if halt.is_set(): return dict(date=day,status="NOT_REQUESTED_AFTER_ACCESS_DENIAL",fits=0)
    rec=dict(date=day,requested_at=now(),attempts=1)
    if not hasattr(thread_local,"session"):
        thread_local.session=requests.Session()
        thread_local.session.headers.update({"User-Agent":"LongtouResearch/1.0 (public read-only source qualification)"})
    params=dict(Order=0,st=1000,a="DailyLimitPerformance",c="HisHomeDingPan",PhoneOSNew=1,VerSion="5.7.0.12",Index=0,PidType=1,apiv="w31",Type=4,Day=day)
    try:
        response=thread_local.session.get(api,params=params,timeout=12)
        path=RAW/(day+".response")
        with path.open("xb") as f: f.write(response.content)
        rec.update(received_at=now(),http_status=response.status_code,source_path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size)
        if response.status_code in (403,429): halt.set()
        if response.status_code==200:
            payload=response.json(); info=payload.get("info")
            rec.update(response_date=info[1] if isinstance(info,list) and len(info)>1 else None,errcode=payload.get("errcode"))
            records=info[0] if isinstance(info,list) and len(info)>1 and isinstance(info[0],list) else []
            rec.update(records=len(records),qualified_date_envelope=bool(str(payload.get("errcode"))=="0" and rec["response_date"]==day))
    except Exception as err: rec.update(received_at=now(),error=type(err).__name__,detail=str(err)[:240],qualified_date_envelope=False)
    return rec

records=[]
journal=OUT/"COLLECTION_RECEIPTS_APPEND_ONLY.jsonl"
with journal.open("x",encoding="utf-8") as log, ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(fetch,day) for day in days]
    for future in as_completed(futures):
        receipt=future.result(); records.append(receipt)
        log.write(json.dumps(receipt,ensure_ascii=False)+"\n"); log.flush()
        if len(records)%40==0 or len(records)==len(days):
            print(json.dumps(dict(at=now(),finished_dates=len(records),total_dates=len(days),valid_date_envelopes=sum(bool(r.get("qualified_date_envelope")) for r in records),last_date=receipt["date"],elapsed_seconds=round(time.monotonic()-began,1),halt=halt.is_set())),flush=True)
records.sort(key=lambda r:r["date"])
(OUT/"ALL668_SOURCE_RECEIPTS.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(dict(at=now(),status="SOURCE_COLLECTION_TERMINAL",recorded_dates=len(records),valid_date_envelopes=sum(bool(r.get("qualified_date_envelope")) for r in records),elapsed_seconds=round(time.monotonic()-began,1),fits=0)),flush=True)
