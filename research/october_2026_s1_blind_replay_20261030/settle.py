"""Score-file is frozen before this outcome-only settlement."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal,ROUND_HALF_UP
import json,hashlib,requests
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
picks=OUT/'TOP3_FROZEN_BEFORE_OUTCOME_REPLAY.csv';receipt=json.loads((OUT/'RANKING_FREEZE_RECEIPT.json').read_text(encoding='utf-8'))
assert sha(picks)==receipt['selected_sha256']
d=pd.read_csv(picks,dtype={'code':str});cal=json.loads((ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json').read_text(encoding='utf-8'))['sessions'];pos={v:i for i,v in enumerate(cal)}
extra_codes=sorted(set(d[d.signal_date.ge('2026-10-27')].code)); raw=OUT/'raw_1030_outcome';raw.mkdir(exist_ok=False)
def fetch(code):
    sym=('sh' if code.startswith('6') else 'sz')+code;p={'param':f'{sym},day,2026-10-29,2026-10-30,10,'}
    rec={'code':code,'requested_at_utc':now(),'url':'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get','params':p}
    try:
        r=requests.get(rec['url'],params=p,timeout=(5,12));r.raise_for_status();blob=r.content;(raw/f'{code}.response').write_bytes(blob)
        rec.update(received_at_utc=now(),sha256=hashlib.sha256(blob).hexdigest());j=json.loads(blob)['data'][sym]['day'];found=[x for x in j if x[0]=='2026-10-30'];rec['row']=found[0] if len(found)==1 else None
    except Exception as e:rec.update(received_at_utc=now(),error=repr(e)[:300],row=None)
    return rec
with ThreadPoolExecutor(max_workers=4) as pool: receipts=list(pool.map(fetch,extra_codes))
(OUT/'TARGET_1030_RECEIPTS_AFTER_FREEZE.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')
daily=pd.read_csv(OUT/'supplemented_daily_PROXY.csv',dtype={'code':str},low_memory=False).set_index(['date','code'])
thirty={x['code']:x['row'] for x in receipts if x['row']}
pages={}
for p in (OUT/'pages').glob('*.json'):
    x=json.loads(p.read_text(encoding='utf-8')); pages[x['date']]={r['code'] for r in x['rows'] if r['kind']=='limit_up' and r['scope']}
src=ROOT/'research/october_2026_firstboard_check_20261030/20261030_limit_up_raw.json'
j=json.loads(src.read_text(encoding='utf-8'))['data'];assert j['qdate']==20261030 and len(j['pool'])==j['tc'];pages['2026-10-30']={r['c'] for r in j['pool']}
def readbar(day,code):
    if day>'2026-10-30':return None
    if day=='2026-10-30':
        a=thirty.get(code)
        if not a:return None
        return {'close':float(a[2]),'high':float(a[3]),'low':float(a[4]),'pre_close':None}
    try:
        a=daily.loc[(day,code)]
        return {'close':a['close'],'high':a['high'],'low':a['low'],'pre_close':a['pre_close']}
    except KeyError:return None
def limit(price):return float((Decimal(str(price))*Decimal('1.1')).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
settled=[]
for _,r in d.iterrows():
    sig=r.signal_date;target=cal[pos[sig]+1];prev=float(r['close']);code=r.code
    path=[];alive=True;known=True
    for h in range(1,4):
        day=cal[pos[sig]+h];bar=readbar(day,code)
        if not alive:path.append({'date':day,'state':'ALREADY_BROKEN'});continue
        if not bar:known=False;path.append({'date':day,'state':'UNKNOWN'});break
        lim=limit(prev); price_hit=abs(bar['close']-lim)<.005
        corp_conflict=bar['pre_close'] is not None and abs(bar['pre_close']-prev)>.011
        pool_hit=code in pages[day] if day in pages else None
        conflict=pool_hit is not None and pool_hit!=price_hit
        if corp_conflict or conflict:
            known=False;path.append({'date':day,'state':'UNKNOWN_PRICE_OR_POOL_CONFLICT','close':bar['close'],'reference':lim,'pool_hit':pool_hit});break
        path.append({'date':day,'state':'HIT' if price_hit else 'MISS','close':bar['close'],'reference':lim,'target_one_price':bool(bar['high']==bar['low']),'pool_hit':pool_hit})
        alive=price_hit;prev=bar['close']
    first=path[0];outcome=1 if first['state']=='HIT' else 0 if first['state']=='MISS' else None
    reach4=0 if any(q['state']=='MISS' for q in path) else 1 if len(path)==3 and all(q['state']=='HIT' for q in path) else None
    settled.append({'signal_date':sig,'target_date':target,'rank':int(r['rank']),'code':code,'name':r['name'],'s1_score':r['s1_score'],'signal_close':r['close'],'signal_one_price':int(r['one_price']),'target_one_price':first.get('target_one_price'),'outcome':outcome,'reach4':reach4,'path_json':json.dumps(path,ensure_ascii=False),'amount_ratio_5_missing':bool(pd.isna(r['amount_ratio_5'])),'return_60_missing':bool(pd.isna(r['return_60']))})
out=pd.DataFrame(settled);out.to_csv(OUT/'TOP3_SETTLED_DIAGNOSTIC.csv',index=False,encoding='utf-8-sig')
def metric(x):
    known=int(x.outcome.notna().sum());hits=int(x.outcome.eq(1).sum());return {'selected':len(x),'known':known,'unknown':len(x)-known,'hits':hits,'hit_rate':hits/known if known else None}
summary={'formed_at_utc':now(),'ranking_sha256':sha(picks),'settled_sha256':sha(OUT/'TOP3_SETTLED_DIAGNOSTIC.csv'),'method':'S1 fixed model reconstructed inputs; not clean blind','days':16,'slots':48,'unfilled':48-len(out),'all':metric(out),'signal_not_one_price':metric(out[out.signal_one_price.eq(0)]),'next_not_one_price':metric(out[out.target_one_price.eq(False)]),'next_one_price_known':metric(out[out.target_one_price.eq(True)]),'reach4_known':int(out.reach4.notna().sum()),'reach4_hits':int(out.reach4.eq(1).sum()),'daily':[{'date':day,**metric(x)} for day,x in out.groupby('signal_date')],'source_1030_limit_pool_sha256':sha(src)}
(OUT/'METRICS_DIAGNOSTIC.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
