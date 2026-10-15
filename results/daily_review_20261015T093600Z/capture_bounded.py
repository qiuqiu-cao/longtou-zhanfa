"""One bounded public-source check for the requested 15 -> 16 review."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import requests
import public_outcome_sources_v293 as source

OUT = Path(__file__).resolve().parent
CODES = ["000530", "001318", "002343", "002696", "601811", "605189"]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)


def one(item):
    kind, code, url, params = item
    entry = dict(kind=kind, code=code, url=url, parameters=params,
                 requested_at=now(), maximum_attempts=1)
    path = OUT / "bounded_sources" / (kind + ("_" + code if code else "") + ".raw.json")
    try:
        response = requests.get(url, params=params, timeout=(4, 7),
                                headers={"User-Agent": "Mozilla/5.0"})
        path.write_bytes(response.content)
        entry.update(received_at=now(), http_status=response.status_code,
                     raw_path=str(path), sha256=hashlib.sha256(response.content).hexdigest())
        response.raise_for_status()
        data = response.json()
        if code is None:
            parsed = source.parse_pool(data, "2026-10-15", kind)
            save(path.with_name(kind + ".parsed.json"), parsed)
            entry.update(status="EXACT_DATE_POOL_RECEIVED", rows=len(parsed["rows"]))
        else:
            notices = (data.get("data") or {}).get("list")
            if not isinstance(notices, list):
                raise ValueError("notice index is unavailable; no risk clearance")
            entry.update(status="NOTICE_INDEX_RECEIVED_NOT_FULLTEXT_REVIEW",
                         notices=[{k:r.get(k) for k in ("art_code", "title", "notice_date", "codes", "columns")}
                                  for r in notices])
    except Exception as exc:
        entry.update(status="UNAVAILABLE", error=str(exc)[:450], completed_at=now())
    return entry


def main():
    dest = OUT / "bounded_sources"
    dest.mkdir(exist_ok=False)
    jobs = []
    for kind, endpoint in source.ENDPOINTS.items():
        jobs.append((kind, None, source.EM_BASE + endpoint,
                     dict(ut="7eea3edcaed734bea9cbfc24409ed989", dpt="wz.ztzt", Pageindex="0",
                          pagesize="10000", sort="fbt:asc" if kind in {"limit_up", "failed_board"}
                          else "fund:asc" if kind == "limit_down" else "zs:desc", date="20261015")))
    for code in CODES:
        jobs.append(("notices", code, "https://np-anotice-stock.eastmoney.com/api/security/ann",
                     dict(sr="-1", page_size="30", page_index="1", ann_type="A", client_source="web",
                          stock_list=code, begin_time="2026-10-12", end_time="2026-10-15")))
    with ThreadPoolExecutor(max_workers=5) as executor:
        entries = list(executor.map(one, jobs))
    result = dict(formed_at=now(), maximum_attempts_per_request=1, entries=entries,
                  unavailable_not_zero=True, new_fits=0, old_files_modified=False)
    save(dest / "manifest.json", result)
    print(json.dumps([dict(kind=r["kind"], code=r["code"], status=r["status"],
                          notices=r.get("notices"), rows=r.get("rows"), error=r.get("error"))
                      for r in entries], ensure_ascii=False))


if __name__ == "__main__":
    main()
