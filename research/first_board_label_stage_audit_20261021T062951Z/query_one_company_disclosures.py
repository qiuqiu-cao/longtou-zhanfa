"""One bounded official disclosure-index search for the sole negative/report conflict."""
from pathlib import Path
import json
from datetime import datetime,timezone
import hashlib
import requests
OUT=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
if (OUT/"ONE_COMPANY_DISCLOSURE_REQUEST.json").exists():raise FileExistsError("No repeated search")
body=dict(pageNum="1",pageSize="100",column="sse",tabName="fulltext",plate="sh",stock="",
    searchkey="爱婴室",secid="",category="",trade="",seDate="2024-09-01~2024-10-08",sortName="",sortType="",isHLtitle="true")
url="https://www.cninfo.com.cn/new/hisAnnouncement/query"
rec=dict(requested_at=now(),url=url,body=body,scope="Only603214one disclosed-period window to investigate2024-09-30->2024-10-08 conflict",attempts=1,
    inferred_dividend_NOT_VERIFIED=True,model_fits=0,target_rewrites=0)
(OUT/"ONE_COMPANY_DISCLOSURE_REQUEST.json").write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding="utf-8")
try:
    r=requests.post(url,data=body,headers={"User-Agent":"Mozilla/5.0","Referer":"https://www.cninfo.com.cn/new/index"},timeout=15)
    path=OUT/"raw/603214_20240901_20241008_official_index.response"
    with path.open("xb") as f:f.write(r.content)
    rec.update(received_at=now(),http_status=r.status_code,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest(),source_path=str(path.resolve()))
    if r.status_code==200:
        d=r.json(); items=d.get("announcements") or []
        rec.update(total=d.get("totalAnnouncement"),items=[{k:a.get(k) for k in ["secCode","secName","announcementTitle","announcementTime","adjunctUrl","announcementId"]} for a in items])
except Exception as exc:rec.update(completed_at=now(),error_type=type(exc).__name__,error=str(exc)[:200])
(OUT/"ONE_COMPANY_DISCLOSURE_RECEIPT.json").write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in rec.items() if k not in ["body"]},ensure_ascii=True))
