"""Observe supported request depth/interval combinations without interpolation."""
from datetime import datetime
import argparse
import hashlib
import json
from pathlib import Path
import requests
import public_minute_capture_v1 as minute


def run(output):
    target=Path(output);target.mkdir(parents=True,exist_ok=False)
    checks=[]
    for scale,size in ((1,1500),(1,1969),(5,1023)):
        item=dict(code="000678",interval_minutes=scale,requested_bar_count=size,
                  requested_at=datetime.now().astimezone().isoformat())
        try:
            response=requests.get(minute.SINA_MINUTE_URL,
                params=dict(symbol="sz000678",scale=str(scale),ma="no",datalen=str(size)),
                headers=minute.SINA_HEADERS,timeout=8)
            path=target/(f"m{scale}_{size}.response");path.write_bytes(response.content)
            item.update(url=response.url,http_status=response.status_code,raw_path=str(path.resolve()),
                raw_sha256=hashlib.sha256(response.content).hexdigest(),received_at=datetime.now().astimezone().isoformat())
            response.raise_for_status()
            rows=minute._extract_json_array(response.text)
            item.update(raw_rows=len(rows),first_bar=rows[0].get("day") if rows else None,
                last_bar=rows[-1].get("day") if rows else None,
                distinct_dates=len({r.get("day","")[:10] for r in rows}),
                null_payload="var _data=(null);" in response.text)
        except Exception as exc:item["error"]=str(exc)[:250]
        checks.append(item)
    report=dict(version="v287-minute-interval-depth-1",recorded_at=datetime.now().astimezone().isoformat(),
        checks=checks,scope="Observed request combinations only; five-minute bars do not become one-minute evidence",
        accuracy_claim_allowed=False,forecast_rewrite_allowed=False,
        source_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(__file__),Path(minute.__file__))})
    (target/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    print(json.dumps(run(parser.parse_args().output),ensure_ascii=True))
