from pathlib import Path
import csv, json, hashlib
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
spec = [
 ('2026-10-23','2026-10-26','daily_review_20261023T1621Z','daily_review_20261026T1323Z'),
 ('2026-10-26','2026-10-27','daily_review_20261026T1323Z','daily_review_20261027T0928Z'),
 ('2026-10-27','2026-10-28','daily_review_20261027T0928Z','daily_review_20261028T1020Z'),
 ('2026-10-28','2026-10-29','daily_review_20261028T1020Z','daily_review_20261029T1912Z'),
 ('2026-10-29','2026-10-30','daily_review_20261029T1912Z',None),
]

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

source=ROOT/'research/first_board_top3_40_20261030/results'
a=pd.read_csv(source/'FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv',dtype={'code':str})
b=pd.read_csv(source/'OUTCOMES_DIAGNOSTIC_ONLY.csv',dtype={'code':str})
data=a.merge(b,on=['signal_date','code'],validate='one_to_one')
old=data[data.signal_date.str.startswith('2025-10') & data.saved_s1_rank.le(3)].copy()
assert len(old)==51 and old.signal_date.nunique()==17
old.sort_values(['signal_date','saved_s1_rank']).to_csv(OUT/'2025_oct_s1_top3.csv',index=False,encoding='utf-8-sig')

url='https://push2ex.eastmoney.com/getTopicZTPool'
params=dict(ut='7eea3edcaed734bea9cbfc24409ed989',dpt='wz.ztzt',Pageindex='0',pagesize='10000',sort='fbt:asc',date='20261030')
receipt={'url':url,'params':params,'requested_at_utc':datetime.now(timezone.utc).isoformat()}
try:
    res=requests.get(url,params=params,timeout=12)
    receipt.update(received_at_utc=datetime.now(timezone.utc).isoformat(),status=res.status_code)
    res.raise_for_status(); blob=res.content; (OUT/'20261030_limit_up_raw.json').write_bytes(blob)
    receipt['sha256']=hashlib.sha256(blob).hexdigest()
    pool=json.loads(blob)['data']; assert pool['qdate']==20261030 and pool['tc']==len(pool['pool'])
    current={r['c']:r for r in pool['pool']}; receipt['reported_total']=pool['tc']
except Exception as e:
    receipt['error']=repr(e); current={}
(OUT/'market_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')

out=[]
for signal,target,origin,dest in spec:
    cards_path=next((ROOT/'results'/origin).glob('FINAL_SIX_EXECUTION_CARDS_*.json'))
    cards=read(cards_path); codes=cards.get('first_board_top3',cards.get('first_final3'))
    assert len(codes)==3 and len(set(codes))==3
    mapping={x['code']:x['name'] for x in cards['cards']}
    signalbars=read(ROOT/'results'/origin/'facts/normalized_daily.json')
    if dest:
        rpath=ROOT/'results'/dest/'facts'/f'{target}_limit_up_primary.json'
        rp=read(rpath); hits={r['code']:r for r in rp['rows']}
        bars=read(ROOT/'results'/dest/'facts/normalized_daily.json')
    else: hits=current; bars={}
    for rank,code in enumerate(codes,1):
        s=signalbars.get(code,{}).get('days',{}).get(signal)
        t=bars.get(code,{}).get('days',{}).get(target)
        limit=int((Decimal(str(s['close']))*Decimal('1.1')).quantize(Decimal('1'),rounding=ROUND_HALF_UP)) if s else None
        # 10/23 first pick has no exact signal-day daily bar; previous report explicitly resolved its outcome using two minute sources.
        if code in hits:
            value=1; evidence='provider_close_limit_pool'
        elif t and limit is not None:
            value=int(t['close']==limit); evidence='exact_unadjusted_daily_price'
        elif signal=='2026-10-23' and code=='603677':
            value=0; evidence='prior_report_two_minute_sources'
        else: value=None; evidence='UNKNOWN'
        out.append(dict(signal=signal,target=target,rank=rank,code=code,name=mapping.get(code,''),outcome=value,evidence=evidence,signal_one_price=(s['high']==s['low']) if s else None, source=str(cards_path.relative_to(ROOT)),signal_close_cents=s['close'] if s else None))

# Supplement unknown targets with independently obtained unadjusted daily bars.
extra=[]
for row in out:
    if row['outcome'] is not None or row['signal_close_cents'] is None: continue
    symbol=('sh' if row['code'].startswith('6') else 'sz')+row['code']
    p={'param':f"{symbol},day,{row['signal']},{row['target']},10,"}
    item={'code':row['code'],'target':row['target'],'requested_at_utc':datetime.now(timezone.utc).isoformat(),'url':'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get','params':p}
    try:
        r=requests.get(item['url'],params=p,timeout=12);r.raise_for_status()
        raw=r.content; (OUT/f"{row['code']}_{row['target']}_daily_raw.json").write_bytes(raw)
        item.update(received_at_utc=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(raw).hexdigest(),status=r.status_code)
        days=json.loads(raw)['data'][symbol]['day']; found=[x for x in days if x[0]==row['target']]
        if len(found)==1:
            close=int((Decimal(found[0][2])*100).quantize(Decimal('1')))
            lim=int((Decimal(row['signal_close_cents'])*Decimal('1.1')).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
            row['outcome']=int(close==lim);row['evidence']='fresh_tencent_exact_unadjusted_daily';item.update(close_cents=close,reference_limit_cents=lim)
    except Exception as e: item['error']=repr(e)
    extra.append(item)
(OUT/'supplement_daily_receipt.json').write_text(json.dumps(extra,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'manual_20261023_29_diagnostic.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print(json.dumps({'old_2025_10':{'selected':len(old),'known':int(old.promotion_next_LABEL_ONLY.count()),'hits':int(old.promotion_next_LABEL_ONLY.sum()),'nonone_known':int(old[old.one_price.eq(0)].promotion_next_LABEL_ONLY.count()),'nonone_hits':int(old[old.one_price.eq(0)].promotion_next_LABEL_ONLY.sum()),'reach4_hits':int(old.reach4_LABEL_ONLY.sum()),'reach4_known':int(old.reach4_LABEL_ONLY.count())},'manual':out,'receipt':receipt},ensure_ascii=False,indent=2))
