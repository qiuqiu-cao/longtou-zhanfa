"""Post-outcome oracle; explicitly NOT a feasible predictive selection rule."""
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
from concurrent.futures import ThreadPoolExecutor
import json,hashlib,requests,pandas as pd
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent;OUT=BASE/'v3';raw=OUT/'oracle_1030_sina';raw.mkdir(exist_ok=False)
def now():return datetime.now(timezone.utc).isoformat()
x=pd.read_csv(OUT/'FULL_POOL_SCORES_UNQUALIFIED_DIAGNOSTIC.csv',dtype={'code':str},low_memory=False)
x=x[x.research_selection_allowed & ~x.screen_missing_volume_history].copy()
cal=json.loads((ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json').read_text(encoding='utf-8'))['sessions']; pos={v:i for i,v in enumerate(cal)}
prior=json.loads((OUT/'TARGET_1030_SINA_RECEIPTS_V2.json').read_text(encoding='utf-8'));third={r['code']:r['row'] for r in prior if r['row']}
codes=sorted(set(x[x.signal_date.eq('2026-10-29')].code)-set(third))
def fetch(code):
    sym=('sh' if code.startswith('6') else 'sz')+code;p={'symbol':sym,'scale':'240','ma':'no','datalen':'20'}
    rec={'code':code,'requested_at_utc':now(),'url':'https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketData.getKLineData','params':p}
    try:
        r=requests.get(rec['url'],params=p,timeout=(5,12));r.raise_for_status();b=r.content;(raw/f'{code}.response').write_bytes(b)
        rec.update(received_at_utc=now(),sha256=hashlib.sha256(b).hexdigest());a=[a for a in json.loads(b) if a['day']=='2026-10-30'];rec['row']=a[0] if len(a)==1 else None
    except Exception as e:rec.update(received_at_utc=now(),error=repr(e)[:250],row=None)
    return rec
with ThreadPoolExecutor(max_workers=4) as p:received=list(p.map(fetch,codes))
for r in received:
    if r['row']:third[r['code']]=r['row']
(OUT/'oracle_1030_receipts.json').write_text(json.dumps(received,ensure_ascii=False,indent=2),encoding='utf-8')
daily=pd.read_csv(OUT/'supplemented_daily_PROXY.csv',dtype={'code':str},low_memory=False).set_index(['date','code'])
pages={}
for p in (BASE/'pages').glob('*.json'):
    a=json.loads(p.read_text(encoding='utf-8'));pages[a['date']]={r['code'] for r in a['rows'] if r['kind']=='limit_up' and r['scope']}
a=json.loads((ROOT/'research/october_2026_firstboard_check_20261030/20261030_limit_up_raw.json').read_text(encoding='utf-8'))['data'];assert a['qdate']==20261030 and a['tc']==len(a['pool']);pages['2026-10-30']={r['c'] for r in a['pool']}
res=[]
for _,r in x.iterrows():
    day=r.signal_date;target=cal[pos[day]+1];code=r.code; t=None; pre=None
    if target=='2026-10-30':
        z=third.get(code)
        if z:t=float(z['close'])
    else:
        try:z=daily.loc[(target,code)];t=float(z['close']);pre=z['pre_close']
        except KeyError:pass
    cap=float((Decimal(str(r['close']))*Decimal('1.1')).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)); hit=abs(t-cap)<.005 if t is not None else None
    pool=code in pages[target] if target in pages else None
    conflict=(pre is not None and pd.notna(pre) and abs(pre-r['close'])>.011) or (hit is not None and pool is not None and hit!=pool)
    outcome=None if t is None or conflict else int(hit)
    res.append({'signal_date':day,'target':target,'code':code,'name':r['name'],'score':r['s1_score'],'outcome':outcome,'target_close':t,'cap_reference':cap,'page_hit':pool,'conflict':conflict})
out=pd.DataFrame(res);out.to_csv(OUT/'ORACLE_FULL_POOL_LABEL_ONLY.csv',index=False,encoding='utf-8-sig')
daily_summary=[]
for day,g in out.groupby('signal_date',sort=True):
    hits=int(g.outcome.eq(1).sum());unknown=int(g.outcome.isna().sum())
    daily_summary.append({'signal_date':day,'pool':len(g),'wins':hits,'unknown':unknown,'best_one_known_hit':int(hits>0),'best_one_possible_hit':int(hits+unknown>0),'best_three_known_hits':min(3,hits),'best_three_possible_hits':min(3,hits+unknown),'winning_names':g[g.outcome.eq(1)].name.tolist()})
S={'created_at_utc':now(),'meaning':'hindsight oracle only: future labels used, NOT a realizable stock-picking method','date_count':len(daily_summary),'total_pool':len(out),'known':int(out.outcome.notna().sum()),'unknown':int(out.outcome.isna().sum()),'best_one_known_hits':sum(a['best_one_known_hit'] for a in daily_summary),'best_one_possible_hits':sum(a['best_one_possible_hit'] for a in daily_summary),'best_three_known_hits':sum(a['best_three_known_hits'] for a in daily_summary),'best_three_possible_hits':sum(a['best_three_possible_hits'] for a in daily_summary),'daily':daily_summary}
(OUT/'ORACLE_DIAGNOSTIC.json').write_text(json.dumps(S,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(S,ensure_ascii=False,indent=2))
