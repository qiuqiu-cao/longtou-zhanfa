"""Bounded official announcement capture, without model fitting or label edits."""
from __future__ import annotations
import argparse
from datetime import date,datetime,timedelta,timezone
import hashlib
import html
import json
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo
import fitz
import requests

URL="https://www.cninfo.com.cn/new/hisAnnouncement/query"
def now():return datetime.now(timezone.utc).isoformat()
def digest(raw):return hashlib.sha256(raw).hexdigest()
def write_new(path,data):
    with Path(path).open("x",encoding="utf-8") as stream:json.dump(data,stream,ensure_ascii=False,indent=2)
def title(value):return html.unescape(re.sub(r"<[^>]+>","",str(value)))
def publication_date(item):
    value=item.get("announcementTime")
    if value is None:return None
    try:return datetime.fromtimestamp(float(value)/1000,ZoneInfo("Asia/Shanghai")).date().isoformat()
    except (ValueError,TypeError,OverflowError):return None

def run(output):
    output=Path(output)
    protocol=json.loads((output/"JOINT_FROZEN_PROTOCOL.json").read_text("utf-8"))
    assert protocol["model_fits_allowed"]==0
    assert len(protocol["cases"])==3 and protocol["max_direct_http_requests"]==6
    assert protocol["collector_source_sha256"]==digest(Path(__file__).read_bytes())
    write_new(output/"COLLECTION_STARTED.json",dict(at=now(),model_fits=0,
        no_retries=True,protocol_sha256=digest((output/"JOINT_FROZEN_PROTOCOL.json").read_bytes())))
    (output/"raw").mkdir(exist_ok=False)
    receipts=[]
    access_halt=False
    def save(rec):
        receipts.append(rec)
        with (output/"SOURCE_RECEIPTS_APPEND_ONLY.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(rec,ensure_ascii=False)+"\n")
    def retrieve(kind,code,url,**kwargs):
        nonlocal access_halt
        if access_halt:return None,None
        if datetime.now(timezone.utc)>=datetime.fromisoformat(protocol["deadline_utc"]):
            raise TimeoutError("Original fixed source-audit deadline exceeded")
        rec=dict(kind=kind,code=code,url=url,requested_at=now(),attempts=1,model_fits=0)
        if "data" in kwargs:rec["request_body"]=kwargs["data"]
        write_new(output/f"{code}_{kind}_REQUEST_BEFORE_HTTP.json",rec)
        try:
            response=requests.request("POST" if kind=="index" else "GET",url,
                timeout=protocol["timeout_seconds"],headers={"User-Agent":"Mozilla/5.0","Referer":"https://www.cninfo.com.cn/new/index"},**kwargs)
            raw=response.content
            path=output/"raw"/(f"{code}_index.response" if kind=="index" else f"{code}_implementation.pdf")
            with path.open("xb") as stream:stream.write(raw)
            rec.update(received_at=now(),http_status=response.status_code,bytes=len(raw),sha256=digest(raw),source_path=str(path.resolve()))
            if response.status_code in (403,429):access_halt=True
            save(rec)
            return response,rec
        except requests.RequestException as exc:
            rec.update(completed_at=now(),error_type=type(exc).__name__,error=str(exc)[:220])
            save(rec)
            return None,rec
    results=[]
    for case in protocol["cases"]:
        code=case["code"]
        result=dict(case,official_action_verified=False,status="NOT_CAPTURED",model_fits=0)
        if access_halt:
            result["status"]="NOT_REQUESTED_AFTER_ACCESS_HALT"
            results.append(result)
            continue
        ex_date=date.fromisoformat(case["source_date"])
        body=dict(pageNum="1",pageSize="100",column="sse",tabName="fulltext",plate="sh",stock="",
            searchkey=code,secid="",category="",trade="",seDate=f"{ex_date-timedelta(days=120)}~{ex_date}",
            sortName="",sortType="",isHLtitle="true")
        response,receipt=retrieve("index",code,URL,data=body)
        result["index_receipt"]=receipt
        if response is None or response.status_code!=200:
            result["status"]="INDEX_FAILED_OR_UNKNOWN"
            results.append(result)
            continue
        try:
            payload=response.json()
            items=payload.get("announcements") or []
            matching=[item for item in items if str(item.get("secCode")).zfill(6)==code
                and "权益分派" in title(item.get("announcementTitle"))
                and "实施公告" in title(item.get("announcementTitle"))
                and publication_date(item) is not None and publication_date(item)<=case["source_date"]]
            result.update(index_total_announcements=payload.get("totalAnnouncement"),
                index_has_more=payload.get("hasMore"),index_items_received=len(items),
                implementation_candidates=[dict(secCode=item.get("secCode"),title=title(item.get("announcementTitle")),
                    publication_date=publication_date(item),announcement_id=item.get("announcementId"),
                    adjunct_url=item.get("adjunctUrl")) for item in matching])
            # This bounded probe does not page or choose a different company when unavailable.
            if not matching:
                result["status"]="NO_IMPLEMENTATION_IN_BOUNDED_INDEX_UNKNOWN"
                results.append(result)
                continue
            matching.sort(key=lambda item:(publication_date(item),str(item.get("announcementId"))),reverse=True)
            selected=matching[0]
            adjunct=selected.get("adjunctUrl") or ""
            if not adjunct.startswith("finalpage/") or not adjunct.lower().endswith(".pdf"):
                result["status"]="UNQUALIFIED_ADJUNCT_LOCATION_UNKNOWN"
                results.append(result)
                continue
            result["selected_announcement"]=dict(title=title(selected["announcementTitle"]),
                announcement_id=selected.get("announcementId"),publication_date=publication_date(selected),
                url="https://static.cninfo.com.cn/"+adjunct)
            write_new(output/f"{code}_PDF_SELECTION_BEFORE_REQUEST.json",result["selected_announcement"])
            time.sleep(0.6)
            pdf_response,pdf_receipt=retrieve("pdf",code,result["selected_announcement"]["url"])
            result["pdf_receipt"]=pdf_receipt
            if pdf_response is None or pdf_response.status_code!=200 or not pdf_response.content.startswith(b"%PDF"):
                result["status"]="PDF_FAILED_OR_UNKNOWN"
                results.append(result)
                continue
            document=fitz.open(stream=pdf_response.content,filetype="pdf")
            (output/f"{code}_IMPLEMENTATION_TEXT.txt").write_text("\n\n".join(
                f"PAGE {index+1}\n"+page.get_text() for index,page in enumerate(document)),encoding="utf-8")
            for index,page in enumerate(document):
                page.get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(output/f"{code}_PAGE{index+1}.png")
            result.update(status="OFFICIAL_PDF_CAPTURED_PENDING_FIELD_AND_VISUAL_VERIFICATION",pdf_pages=len(document),
                original_signal_time_capture=False,labels_or_features_qualified=False)
        except (ValueError,TypeError,KeyError,RuntimeError) as exc:
            result.update(status="SOURCE_PARSE_ERROR_UNKNOWN",error=repr(exc))
        results.append(result)
        time.sleep(0.6)
    write_new(output/"ALL_THREE_CASE_SOURCE_RECEIPTS.json",dict(at=now(),cases=results,
        actual_direct_http_requests=len(receipts),access_halt=access_halt,
        model_fits=0,old_targets_or_predictions_rewritten=0,
        all_fields_or_formulas_verified=False,complete_action_database=False))
    print(json.dumps(dict(at=now(),cases=[dict(code=r["code"],status=r["status"]) for r in results],
        actual_direct_http_requests=len(receipts),model_fits=0),ensure_ascii=True))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    run(parser.parse_args().output)
