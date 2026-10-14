"""Independent end-of-day quote cross-check against an archived official roster."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_HALF_UP
import json
import re
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent

def fetch(codes):
    text = requests.get('https://qt.gtimg.cn/q='+','.join(('sh' if c.startswith('6') else 'sz')+c for c in codes), timeout=15).text
    rows=[]
    for raw in re.findall(r'v_\w+="([^"]*)";', text):
        v=raw.split('~')
        if len(v)<49:
            continue
        try:
            rows.append(dict(code=v[2],name=v[1],close=float(v[3]),prev=float(v[4]),open=float(v[5]),timestamp=v[30],pct=float(v[32]),high=float(v[33]),low=float(v[34]),amount_wan=float(v[37]),turnover=float(v[38]),limit_up=float(v[47]),raw=raw))
        except (ValueError,IndexError):
            continue
    return rows

def run(date, output):
    date = pd.Timestamp(date).strftime('%Y%m%d')
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    sz=pd.read_csv(ROOT/'data/v49/security_szse_asof_20260904/security_master_szse_verified.csv',dtype={'code':str})
    sh=pd.read_csv(ROOT/'data/v49/security_sse_asof_20260904_v2/sse_current_mainboard.csv',dtype={'A_STOCK_CODE':str})
    codes=sorted(set(sz.code)|set(sh.A_STOCK_CODE))
    errors=[]; rows=[]
    with ThreadPoolExecutor(max_workers=5) as ex:
        jobs={ex.submit(fetch,codes[i:i+60]):codes[i:i+60] for i in range(0,len(codes),60)}
        for j in as_completed(jobs):
            try: rows.extend(j.result())
            except Exception as exc: errors.append({'codes':jobs[j],'error':str(exc)})
    f=pd.DataFrame(rows)
    f.to_csv(output/'quotes.csv',index=False,encoding='utf-8-sig')
    fresh=f[f.timestamp.str.startswith(date) & ~f.name.str.contains('ST|退') & f.prev.gt(0)].copy()
    fresh['expected_10pct']=fresh.prev.map(lambda x:float((Decimal(str(x))*Decimal('1.1')).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)))
    eligible=fresh[(fresh.limit_up-fresh.expected_10pct).abs().lt(.001)]
    up=eligible[(eligible.close-eligible.limit_up).abs().lt(.001)]
    failed=eligible[(eligible.high-eligible.limit_up).abs().lt(.001)&eligible.close.lt(eligible.limit_up-.001)]
    raw=pd.read_csv(ROOT/f'data/v3/raw/{date}/limit_up.csv',dtype={'代码':str})
    raw=raw[raw['代码'].str.match(r'^(00|60)')]
    rawfail=pd.read_csv(ROOT/f'data/v3/raw/{date}/failed_board.csv',dtype={'代码':str})
    report=dict(version='2-parameterized',trade_date=pd.Timestamp(date).strftime('%Y-%m-%d'),created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),roster_codes=len(codes),quotes_returned=len(f),date_matched_normal_10pct=len(eligible),closing_limit=len(up),failed=len(failed),additional_closing_codes=sorted(set(up.code)-set(raw['代码'])),raw_closing_not_confirmed=sorted(set(raw['代码'])-set(up.code)),additional_failed_codes=sorted(set(failed.code)-set(rawfail['代码'])),raw_failed_not_confirmed=sorted(set(rawfail['代码'])-set(failed.code)),errors=errors,limitation='Independent quote check on archived roster only; official status and new listings after roster cutoff not fully audited. High/limit equality is a touched-limit proxy, not tick verification.')
    up.to_csv(output/'closing.csv',index=False,encoding='utf-8-sig')
    failed.to_csv(output/'failed.csv',index=False,encoding='utf-8-sig')
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.date,args.output),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
