"""Post-outcome error analysis, labels isolated from model inputs."""
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
import json,pandas as pd
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent;OUT=BASE/'v3'
rows=pd.read_csv(OUT/'FULL_POOL_SCORES_UNQUALIFIED_DIAGNOSTIC.csv',dtype={'code':str},low_memory=False)
rows=rows[rows.research_selection_allowed & ~rows.screen_missing_volume_history & rows.signal_date.le('2026-10-26')].copy()
rows=rows.sort_values(['signal_date','s1_score','code'],ascending=[True,False,True]);rows['s1_rank']=rows.groupby('signal_date').cumcount()+1
daily=pd.read_csv(OUT/'supplemented_daily_PROXY.csv',dtype={'code':str},low_memory=False).set_index(['date','code'])
cal=json.loads((ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json').read_text(encoding='utf-8'))['sessions'];p={v:i for i,v in enumerate(cal)}
pages={}
for file in (BASE/'pages').glob('*.json'):
    a=json.loads(file.read_text(encoding='utf-8'));pages[a['date']]={r['code'] for r in a['rows'] if r['kind']=='limit_up' and r['scope']}
res=[]
for _,r in rows.iterrows():
    prev=r['close'];path=[]; known=True;alive=True
    for h in range(1,4):
        day=cal[p[r.signal_date]+h]
        if not alive:break
        try:
            z=daily.loc[(day,r['code'])];cap=float((Decimal(str(prev))*Decimal('1.1')).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
            hit=abs(z['close']-cap)<.005
            if (pd.notna(z['pre_close']) and abs(z['pre_close']-prev)>.011) or (day in pages and (r['code'] in pages[day])!=hit):known=False;path.append('CONFLICT');break
            path.append('HIT' if hit else 'MISS');alive=hit;prev=z['close']
        except KeyError:known=False;path.append('UNKNOWN');break
    four=0 if 'MISS' in path else 1 if len(path)==3 and all(v=='HIT' for v in path) else None
    res.append({'signal_date':r.signal_date,'code':r.code,'name':r['name'],'s1_rank':int(r.s1_rank),'s1_score':r.s1_score,'reach4':four,'path':'/'.join(path)})
out=pd.DataFrame(res);out.to_csv(OUT/'ALL_POOL_REACH4_LABEL_ONLY_THROUGH_1026.csv',index=False,encoding='utf-8-sig')
z=out[out.reach4.eq(1)];print(json.dumps({'signal_dates':out.signal_date.nunique(),'pool':len(out),'reach4_known':int(out.reach4.notna().sum()),'reach4_unknown':int(out.reach4.isna().sum()),'reach4_hits':len(z),'selected_top3':int(z.s1_rank.le(3).sum()),'missed':z[z.s1_rank.gt(3)].to_dict('records'),'within_top3':z[z.s1_rank.le(3)].to_dict('records')},ensure_ascii=False,indent=2))
