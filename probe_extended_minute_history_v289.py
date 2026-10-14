"""Bounded longer-window probe: preserve intervals, payloads and missingness."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import pandas as pd
import requests
import public_minute_capture_v1 as minute
from archive_september_5min_v287 import sha


def run(output):
    directory=Path(output);directory.mkdir(parents=True,exist_ok=False)
    checks=[]
    for code in ['000678','600825']:
        for interval in [5,30]:
            item=dict(code=code,interval_minutes=interval,requested_bars=1969,
                requested_at=datetime.now().astimezone().isoformat())
            try:
                response=requests.get(minute.SINA_MINUTE_URL,
                    params=dict(symbol=minute.live.sina_symbol(code),scale=str(interval),ma='no',datalen='1969'),
                    headers=minute.SINA_HEADERS,timeout=8)
                path=directory/f'{code}_{interval}m.response';path.write_bytes(response.content)
                item.update(url=response.url,http_status=response.status_code,raw_path=str(path.resolve()),
                    raw_sha256=sha(path),available_at=datetime.now().astimezone().isoformat())
                response.raise_for_status()
                rows=minute._extract_json_array(response.text)
                stamps=pd.Series([r.get('day') for r in rows],dtype='object')
                item.update(status='RAW_RETURNED_NOT_RECONCILED' if rows else 'EMPTY_NOT_COMPLETE',
                    raw_rows=len(rows),first_bar=stamps.min() if len(rows) else None,
                    last_bar=stamps.max() if len(rows) else None,
                    dates=int(stamps.str[:10].nunique()) if len(rows) else 0,
                    duplicate_raw_times=int(stamps.duplicated().sum()),
                    invalid_raw_times=int(pd.to_datetime(stamps,errors='coerce').isna().sum()))
                if rows:
                    counts=stamps.str[:10].value_counts().sort_index()
                    item['date_row_counts']={k:int(v) for k,v in counts.items()}
            except Exception as exc:item.update(status='UNAVAILABLE',error=str(exc)[:250])
            checks.append(item)
            print(json.dumps({k:v for k,v in item.items() if k not in ['date_row_counts','url','raw_path']},ensure_ascii=True),flush=True)
    report=dict(version='v289-extended-interval-depth-probe-1',recorded_at=datetime.now().astimezone().isoformat(),
        checks=checks,scope='Observed two-code requests only; no universe or interval completeness assertion',
        one_minute_evidence=False,accuracy_claim_allowed=False,original_forecast_rewritten=False,
        source_hashes={str(Path(p).resolve()):sha(p) for p in [Path(__file__),Path(minute.__file__)]})
    (directory/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return {'checks':len(checks),'status':'BOUNDED_PROBE_FINISHED'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    print(json.dumps(run(parser.parse_args().output),ensure_ascii=True))
