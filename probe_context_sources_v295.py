"""Bounded public-source reconnaissance; preserves bytes, makes no trading labels."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"data/v295/context_source_probe_20261002_v1"
TESTS=[
    ("concept_page","GET","https://emweb.securities.eastmoney.com/PC_HSF10/CoreConception/PageAjax",{"code":"SH600825"}),
    ("concept_datacenter","GET","https://datacenter-web.eastmoney.com/api/data/v1/get",{"reportName":"RPT_F10_CORETHEME_BOARDTYPE","columns":"ALL","filter":'(SECURITY_CODE="600825")',"pageSize":100,"pageNumber":1,"source":"WEB","client":"WEB"}),
    ("popularity","POST","https://emappdata.eastmoney.com/stockrank/getCurrentLatest",{"appId":"appId01","globalId":"786e4c21-70dc-435a-93bb-38","marketType":"","srcSecurityCode":"SH600825"}),
    ("popularity_history","POST","https://emappdata.eastmoney.com/stockrank/getHisList",{"appId":"appId01","globalId":"786e4c21-70dc-435a-93bb-38","marketType":"","srcSecurityCode":"SH600825","yearType":"5"})]


def now():return datetime.now(timezone.utc).isoformat()


def probe(test):
    key,method,url,params=test
    record={"key":key,"method":method,"url":url,"query":params,"requested_at":now()}
    try:
        r=requests.request(method,url,params=params if method=="GET" else None,json=params if method=="POST" else None,timeout=(5,7))
        body=OUT/(key+".response")
        body.write_bytes(r.content)
        record.update(available_at=now(),http_status=r.status_code,raw_path=str(body),raw_sha256=hashlib.sha256(r.content).hexdigest())
        r.raise_for_status()
        value=r.json()
        record.update(status="RAW_JSON_RECEIVED",keys=list(value) if isinstance(value,dict) else None,preview=str(value)[:1100])
    except Exception as exc:
        record.update(status="UNAVAILABLE",available_at=now(),error=str(exc)[:600])
    return record


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=False)
    with ThreadPoolExecutor(max_workers=4) as executor:entries=list(executor.map(probe,TESTS))
    result={"recorded_at":now(),"requests":entries,"source_hashes":{str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    (OUT/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(entries,ensure_ascii=True))
