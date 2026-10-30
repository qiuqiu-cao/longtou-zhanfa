from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
O=Path(__file__).resolve().parent;R=O.parents[1];OLD=R/'results/daily_review_20261023T1621Z'
def write(p,v):
    assert not p.exists(),p
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
s=(OLD/'prepare_features.py').read_text(encoding='utf-8').replace("DAY='2026-10-23';PRIOR='2026-10-22';TARGET='2026-10-26'","DAY='2026-10-26';PRIOR='2026-10-23';TARGET='2026-10-27'").replace("oldfacts=b.read(ROOT/'results/daily_review_20261022T1608Z/facts/qualified_full_facts_SCORES_MASKED.json')","oldfacts=b.read(ROOT/'results/daily_review_20261023T1621Z/facts/qualified_full_facts_SCORES_MASKED.json')").replace("('2026-10-22','daily_review_20261022T1608Z')]","('2026-10-22','daily_review_20261022T1608Z'),('2026-10-23','daily_review_20261023T1621Z')]")
(O/'prepare_features.py').write_text(s,encoding='utf-8')
s=(OLD/'score_original_BASE.py').read_text(encoding='utf-8').replace("'2026-10-26'","'2026-10-27'").replace("'2026-10-23'","'2026-10-26'").replace("prep['full_relay']==11","prep['full_relay']==14").replace('complete11=True,independent_full81_unchanged=True','complete14=True,independent_full90_unchanged=True')
(O/'score_original_BASE.py').write_text(s,encoding='utf-8')
records=read(O/'minutes/capture_receipt.json')['records'];minmap={}
for x in records:
    if x.get('rows'):minmap[x['symbol'][-6:]]={r['time']:r['price'] for r in x['rows']}
for x in read(O/'minutes/secondary_capture.json'):
    if x.get('data',{}).get('trends'):
        minmap[x['code']]={r.split(',')[0][-5:].replace(':',''):float(r.split(',')[2]) for r in x['data']['trends']}
        if x['code']=='603089':
            rows=[r.split(',') for r in x['data']['trends']];print('正裕minute low',min(float(r[4]) for r in rows),'nonlimit',[(r[0],r[2],r[4]) for r in rows if float(r[4])<13.08][:12])
for c in ['603278','603677','002009','603081']:
    p=minmap[c];limit={'603278':17.56,'603677':17.93,'002009':19.03,'603081':11.72}[c]
    hit=[t for t,v in p.items() if v==limit];after=[t for t,v in p.items() if hit and t>hit[0] and v<limit]
    print(c,'first limit minute',hit[0] if hit else None,'first subsequent lower',after[0] if after else None,'close',p.get('1500'))
windows=[]
for node in ['0935','0940','1000','1015','1030','1400']:
    n=int(node[:2])*60+int(node[2:]);times=[f'{(n+k)//60:02d}{(n+k)%60:02d}' for k in [-5,-3,-1,0,1,3,5]]
    windows.append(dict(node=node,times=times,prices={c:[minmap.get(c,{}).get(t) for t in times] for c in ['000001','399001','399006','603278','002724','600838','603528','603089','000678','603677','002009','603081','001219','600558','002178','002086','002696','002185','600825']}))
write(O/'minutes/RELATIONSHIP_WINDOWS.json',dict(formed_at=datetime.now(timezone.utc).isoformat(),frequency='one_minute_points_unless_secondary_OHLC',offsets_minutes=[-5,-3,-1,0,1,3,5],windows=windows,causality='No account migration or fund identity inferred'))
for x in records:
    if x['symbol'] in ['sh000001','sz399001','sz399006']:
        print('INDEX',x['name'],x['previous_close'],x['last_quote'],'last amount',x['rows'][-1]['cum_amount'],'low',min(x['rows'],key=lambda r:r['price']))
print('WINDOWS',json.dumps([w for w in windows if w['node'] in ['0940','1015']],ensure_ascii=False))
v=read(OLD/'features/FIRSTBOARD_BASE_QUALIFICATION.json')
print('OLD FIRST QUAL KEYS',list(v))
facts=read(O/'facts/complete_enriched_facts_SCORES_MASKED.json')
write(O/'facts/PRIOR_TOP3_MINUTE_SUPPLEMENT.json',dict(formed_at=datetime.now(timezone.utc).isoformat(),signal_date='2026-10-23',outcome_date='2026-10-26',all_three_closed_limit_up=False,top3_close_limit_hits=0,total=3,rows=[dict(code=c,close=minmap[c]['1500'],previous_close=pctbase,change_pct=(minmap[c]['1500']/pctbase-1)*100,closed_limit_up=False) for c,pctbase in [('603677',16.30),('002009',17.30),('603081',10.65)]],qualification='Odd daily bar for 603677 remains unavailable; two exact-date minute feeds and two failed-board pool records corroborate close/failure only. No missing OHLC overwritten.',old_predictions_unchanged=True))
