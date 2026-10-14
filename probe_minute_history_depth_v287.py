"""Bounded empirical check of recent-minute history depth, no model fitting."""
from datetime import datetime
import argparse
import hashlib
import json
from pathlib import Path

import requests
import public_minute_capture_v1 as minute


def run(output):
    target=Path(output)
    target.mkdir(parents=True,exist_ok=False)
    checks=[]
    for code in ("000678","600825"):
        started=datetime.now().astimezone().isoformat()
        params=dict(symbol=minute.live.sina_symbol(code),scale="1",ma="no",datalen="5000")
        check=dict(code=code,requested_bar_count=5000,started_at=started)
        try:
            response=requests.get(minute.SINA_MINUTE_URL,params=params,headers=minute.SINA_HEADERS,timeout=10)
            path=target/(code+".response")
            path.write_bytes(response.content)
            check.update(url=response.url,http_status=response.status_code,raw_path=str(path.resolve()),
                raw_sha256=hashlib.sha256(response.content).hexdigest(),available_at=datetime.now().astimezone().isoformat())
            response.raise_for_status()
            raw=minute._extract_json_array(response.text)
            frame=minute.parse_minute_response(response.text,code)
            check.update(raw_rows=len(raw),parsed_rows=len(frame),
                first_minute=frame.minute.min() if len(frame) else None,
                last_minute=frame.minute.max() if len(frame) else None,
                distinct_dates=int(frame.minute.str[:10].nunique()) if len(frame) else 0)
            if len(frame):frame.to_csv(target/(code+"_bars_UNRECONCILED.csv"),index=False)
        except Exception as exc:
            check["error"]=str(exc)[:300]
        checks.append(check)
    result=dict(version="v287-minute-depth-probe-1",recorded_at=datetime.now().astimezone().isoformat(),
        status="EMPIRICAL_SOURCE_DEPTH_ONLY",checks=checks,accuracy_claim_allowed=False,
        forecast_rewrite_allowed=False,trade_authorized=False,
        source_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(__file__),Path(minute.__file__))})
    (target/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    print(json.dumps(run(parser.parse_args().output),ensure_ascii=True))
