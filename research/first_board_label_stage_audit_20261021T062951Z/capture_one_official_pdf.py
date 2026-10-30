"""Fetch only the official company document identified by the archived index."""
from pathlib import Path
import json
import hashlib
from datetime import datetime,timezone
import requests
import fitz

OUT=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
index=json.loads((OUT/"ONE_COMPANY_DISCLOSURE_RECEIPT.json").read_text("utf-8"))
items=[x for x in index["items"] if x["secCode"]=="603214" and x["announcementTitle"]=="2024年半年度权益分派实施公告"]
assert len(items)==1
url="https://static.cninfo.com.cn/"+items[0]["adjunctUrl"]
record=dict(url=url,requested_at=now(),attempts=1,source_index_sha256=index["sha256"],announcement_id=items[0]["announcementId"])
request_path=OUT/"OFFICIAL_DIVIDEND_PDF_REQUEST.json"
if request_path.exists():raise FileExistsError("No repeated PDF request")
request_path.write_text(json.dumps(record,indent=2),encoding="utf-8")
try:
    r=requests.get(url,timeout=15,headers={"User-Agent":"Mozilla/5.0"})
    filename=OUT/"raw/603214_2024H1_DIVIDEND_IMPLEMENTATION.pdf"
    with filename.open("xb") as f:f.write(r.content)
    record.update(received_at=now(),http_status=r.status_code,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest(),path=str(filename.resolve()))
    if r.status_code==200 and r.content.startswith(b"%PDF"):
        doc=fitz.open(filename)
        text="\n\n".join(f"PAGE {n+1}\n"+page.get_text() for n,page in enumerate(doc))
        (OUT/"603214_DIVIDEND_PDF_TEXT.txt").write_text(text,encoding="utf-8")
        record.update(pages=len(doc),text_path=str((OUT/"603214_DIVIDEND_PDF_TEXT.txt").resolve()))
        doc[0].get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(OUT/"603214_DIVIDEND_PAGE1.png")
        print(text[:11000])
except Exception as exc:record.update(completed_at=now(),error=repr(exc))
(OUT/"OFFICIAL_DIVIDEND_PDF_RECEIPT.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
print(json.dumps(record))
