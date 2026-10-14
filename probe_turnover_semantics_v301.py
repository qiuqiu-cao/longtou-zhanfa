"""Fetch only observed official page scripts to verify product and unit semantics."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
from probe_market_turnover_v301 import fetch
from market_interval_research_v300 import sha,write_json


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    result=[]
    for key,url in [('sse_main','https://www.sse.com.cn/xhtml/js/app/main_2021.js'),
                    ('sse_config','https://www.sse.com.cn/xhtml/js/common/table_config.js?v=V3.7.9')]:
        result.append(fetch(out,key,url))
    write_json(out/'manifest.json',{'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'requests':result,
                                   'script_sha256':sha(__file__)})
    print([(r['key'],r['status'],r.get('bytes')) for r in result])
