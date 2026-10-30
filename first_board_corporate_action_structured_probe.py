"""One raw structured dividend-response request per predeclared control company."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,time
import requests

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:json.dump(value,stream,ensure_ascii=False,indent=2)

def run(output):
    output=Path(output)
    protocol=json.loads((output/"JOINT_FROZEN_PROTOCOL.json").read_text("utf-8"))
    assert protocol["model_fits_allowed"]==0 and protocol["max_direct_http_requests"]==4
    assert protocol["collector_source_sha256"]==sha(__file__)
    assert protocol["codes"]==["600361","603203","688368","603214"]
    write(output/"COLLECTION_STARTED.json",dict(at=now(),model_fits=0))
    (output/"raw").mkdir(exist_ok=False)
    receipts=[]
    halt=False
    for code in protocol["codes"]:
        if halt:
            receipts.append(dict(code=code,status="NOT_REQUESTED_AFTER_ACCESS_HALT"))
            continue
        if datetime.now(timezone.utc)>=datetime.fromisoformat(protocol["deadline_utc"]):
            receipts.append(dict(code=code,status="NOT_REQUESTED_AFTER_BUDGET"))
            continue
        url="https://datacenter-web.eastmoney.com/api/data/v1/get"
        params=dict(sortColumns="REPORT_DATE",sortTypes="-1",pageSize=str(protocol["page_size"]),pageNumber="1",
            reportName="RPT_SHAREBONUS_DET",columns="ALL",quoteColumns="",source="WEB",client="WEB",
            filter=f'(SECURITY_CODE="{code}")')
        receipt=dict(code=code,requested_at=now(),url=url,params=params,attempts=1,model_fits=0)
        write(output/f"{code}_REQUEST_BEFORE_HTTP.json",receipt)
        try:
            response=requests.get(url,params=params,timeout=10,headers={"User-Agent":"Mozilla/5.0"})
            path=output/"raw"/f"{code}.response"
            with path.open("xb") as stream:stream.write(response.content)
            receipt.update(received_at=now(),http_status=response.status_code,source_path=str(path.resolve()),
                bytes=len(response.content),sha256=sha(path))
            if response.status_code in (403,429):halt=True
            if response.status_code==200:
                payload=response.json()
                result=payload.get("result") or {}
                data=result.get("data") or []
                receipt.update(success=payload.get("success"),response_code=payload.get("code"),
                    pages=result.get("pages"),reported_count=result.get("count"),received_rows=len(data),
                    complete_single_page=result.get("pages")==1,
                    schema=sorted({key for item in data for key in item}),
                    returned_security_codes=sorted({str(item.get("SECURITY_CODE")) for item in data}))
            receipt["status"]=("CAPTURED_PENDING_SCHEMA_AND_OFFICIAL_COMPARISON" if response.status_code==200
                else "HTTP_FAILED_UNKNOWN_NO_RETRY")
        except (requests.RequestException,ValueError,TypeError) as exc:
            receipt.update(completed_at=now(),status="FAILED_UNKNOWN_NO_RETRY",error=repr(exc))
        receipts.append(receipt)
        with (output/"SOURCE_RECEIPTS_APPEND_ONLY.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(receipt,ensure_ascii=False)+"\n")
        time.sleep(0.6)
    write(output/"ALL_FOUR_STRUCTURED_SOURCE_RECEIPTS.json",dict(at=now(),receipts=receipts,
        actual_requests=sum("requested_at" in r for r in receipts),access_halt=halt,model_fits=0,
        original_targets_changed=0,structured_fields_not_yet_qualified=True))
    print(json.dumps(dict(at=now(),cases=[dict(code=r["code"],status=r["status"],rows=r.get("received_rows"),pages=r.get("pages")) for r in receipts],model_fits=0),ensure_ascii=True))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True)
    run(parser.parse_args().output)
