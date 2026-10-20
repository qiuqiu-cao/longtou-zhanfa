"""One bounded GET per named primary source; preserve exact response bytes."""
import concurrent.futures
import datetime as dt
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
import requests

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "capco_index": "https://www.capco.org.cn/pub/zgssgsxh/xhgg/hyfl/hyfljg/index.html",
    "capco_2023h2": "https://capcofile.oss-cn-beijing.aliyuncs.com/2024/file/GwFj/" + quote("2023年下半年上市公司行业分类结果（按行业排序）.pdf"),
    "csrc_classification_index": "https://www.csrc.gov.cn/csrc/c100103/common_list_2.shtml",
    "csrc_2021q3_landing": "https://www.csrc.gov.cn/csrc/c100103/c1558619/content.shtml",
    "csrc_2020q4": "https://www.csrc.gov.cn/csrc/c100103/c1447170/1447170/files/1616066671097_61196.pdf",
}

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def fetch(item):
    name, url = item
    receipt = ROOT / (name + "_http_receipt.json")
    if receipt.exists():
        return {"name": name, "status": "EXISTS_NO_RETRY"}
    record = {"source_url": url, "requested_at": now(), "one_attempt": True, "timeout": [6, 12]}
    receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        response = requests.get(url, timeout=(6, 12), headers={"User-Agent": "Mozilla/5.0"})
        raw = ROOT / (name + (".pdf" if response.content.startswith(b"%PDF") else ".response"))
        raw.write_bytes(response.content)
        record.update(received_at=now(), status_code=response.status_code, final_url=response.url,
                      bytes=len(response.content), sha256=hashlib.sha256(response.content).hexdigest(),
                      content_type=response.headers.get("Content-Type"), raw_file=raw.name,
                      http_last_modified=response.headers.get("Last-Modified"))
    except Exception as exc:
        record.update(completed_at=now(), error=repr(exc))
    receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"name": name, **record}

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(fetch, SOURCES.items()):
            print(json.dumps(result, ensure_ascii=True), flush=True)
