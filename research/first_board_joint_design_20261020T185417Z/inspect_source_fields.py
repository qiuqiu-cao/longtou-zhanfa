"""Field and timestamp scope before joint design; no label read, no fit."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import requests

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RAW = OUT / "raw"
def now(): return datetime.now(timezone.utc).isoformat()
tree = json.loads((RAW / "alpha_tree.response").read_bytes())
commit = tree["sha"]
paths = ["strategy/first_limit_alpha/README.md", "strategy/first_limit_alpha/labeling.py", "strategy/first_limit_alpha/sequence_dataset.py", "strategy/first_limit_alpha/sequence_model.py", "strategy/first_limit_alpha/train_sequence.py"]
requests_by_name = {"alpha_" + Path(path).name: "https://raw.githubusercontent.com/Shadowell/Alpha/"+commit+"/"+path for path in paths}
api = "https://apphis.longhuvip.com/w1/api/index.php?Order=0&st=1000&a=DailyLimitPerformance&c=HisHomeDingPan&PhoneOSNew=1&VerSion=5.7.0.12&Index=0&PidType=1&apiv=w31&Type=4&Day="
requests_by_name.update({"kpl_"+day:api+day for day in ["2025-01-02", "2026-08-28"]})
with (OUT / "FIELD_SCOPE_PROTOCOL_BEFORE_ADDITIONAL_REQUESTS.json").open("x",encoding="utf-8") as f:
    json.dump(dict(at=now(), fits=0, outcomes_read=False, dates="One validation date only for fields/coverage,one already archivedEMdate for cross-provider time semantics;no label analysis", source_commit=commit,
        requests=requests_by_name, scope="Theme/name/group IDs are potentially retrospectively revised;quarantine them. Single timestamp called limit-time in thirdparty docs is not yet first-time/last-time. Fund field is provider reported quote,not real queue or capital-actor proof. No retry after denial."),f,ensure_ascii=False,indent=2)
def fetch(item):
    name,url=item
    receipt=dict(name=name,url=url,requested_at=now(),attempts=1)
    try:
        response=requests.get(url,timeout=15,headers={"User-Agent":"LongtouResearch/1.0 (public read-only source qualification)"})
        data=response.content
        path=RAW/(name+".response")
        path.write_bytes(data)
        receipt.update(received_at=now(),status=response.status_code,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        if name.startswith("kpl_") and response.status_code==200:
            d=response.json(); records=d.get("info",[[]])[0]
            receipt.update(errcode=d.get("errcode"),response_date=d.get("info",[None,None])[1],records=len(records),array_lengths=sorted(set(map(len,records))),
                timestamp_dates=sorted({datetime.fromtimestamp(row[4],ZoneInfo("Asia/Shanghai")).date().isoformat() for row in records if len(row)>4 and isinstance(row[4],(int,float))}))
    except Exception as e: receipt.update(error=type(e).__name__,detail=str(e)[:300],received_at=now())
    return receipt
with ThreadPoolExecutor(max_workers=3) as pool: receipts=list(pool.map(fetch,requests_by_name.items()))
(OUT/"FIELD_SCOPE_RECEIPTS.json").write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding="utf-8")
for item in receipts: print(json.dumps({k:v for k,v in item.items() if k!="url"},ensure_ascii=True))
