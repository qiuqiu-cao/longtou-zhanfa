"""Finite primary-source discovery; save received bytes and actual receipt clocks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import requests
from market_interval_research_v300 import sha,write_json

URLS={
    'library':'https://raw.githubusercontent.com/akfamily/akshare/main/akshare/stock/stock_summary.py',
    'sse_page':'https://www.sse.com.cn/market/stockdata/overview/day/',
    'szse_page':'https://www.szse.cn/market/overview/index.html',
}


def fetch(destination,key,url,params=None):
    item={'key':key,'url':url,'params':params,'requested_at_utc':datetime.now(timezone.utc).isoformat()}
    try:
        response=requests.get(url,params=params,headers={'User-Agent':'Mozilla/5.0','Referer':url},timeout=(5,8))
        path=destination/(key+'.response');path.write_bytes(response.content)
        item.update(received_at_utc=datetime.now(timezone.utc).isoformat(),http_status=response.status_code,
                    bytes=len(response.content),sha256=sha(path),raw_path=path.name,
                    status='RECEIVED' if response.ok else 'HTTP_REJECTED')
    except requests.RequestException as error:
        item.update(status='UNAVAILABLE',error=str(error)[:220],finished_at_utc=datetime.now(timezone.utc).isoformat())
    return item


def run(output):
    directory=Path(output);directory.mkdir(parents=True,exist_ok=False)
    jobs=list(URLS.items())
    with ThreadPoolExecutor(max_workers=3) as executor:
        result=list(executor.map(lambda pair:fetch(directory,*pair),jobs))
    write_json(directory/'manifest.json',{'version':'v301-turnover-source-discovery-1',
        'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'requests':result,
        'source_script_sha256':sha(__file__),'forward_evidence':False,'goal_complete':False})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    print(json.dumps(run(p.parse_args().output),ensure_ascii=False))
