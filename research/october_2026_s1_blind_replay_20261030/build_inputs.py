"""Reconstruct 62-field S1 inputs from time-sliced provider data, with explicit limitations.

This diagnostic uses an Aug-31 security-universe snapshot and inferred later
security status; it cannot certify the original point-in-time universe.
"""
from pathlib import Path
from decimal import Decimal
import json,sys,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));import first_second_stage_model as stage

cols=['date','code','name','open','high','low','close','pre_close','volume','amount','turnover','float_cap','st','new_listing','status_verified','source']
basefile=ROOT/'data/v16/point_in_time_rebuild_cutoff_20260831_v2/point_in_time_current_daily_bars.csv'
print('reading archived history',flush=True)
chunks=[]
for c in pd.read_csv(basefile,usecols=cols,dtype={'code':str},chunksize=300000,low_memory=False):chunks.append(c[c.date.ge('2025-07-01')])
history=pd.concat(chunks,ignore_index=True)
latest=history[history.date.eq('2026-08-31')].set_index('code')
names=latest.name.to_dict();sts=latest.st.to_dict()
pages={};page_first={};page_meta={}
for p in sorted((OUT/'pages').glob('*.json')):
    d=json.loads(p.read_text(encoding='utf-8'));pages[d['date']]=d
    for r in d['rows']:
        if r['scope']:
            names.setdefault(r['code'],r['name'])
            if r['kind']=='limit_up' and r['boards']==1:
                page_first[(d['date'],r['code'])]=r
            if r['kind']=='limit_up':page_meta[(d['date'],r['code'])]=r

sohu={}
for p in (OUT/'raw_sohu_repair').glob('*.receipt.json'):
    r=json.loads(p.read_text(encoding='utf-8'))
    if r.get('valid'):
        sohu[r['code']]=OUT/'raw_sohu_repair'/f"{r['code']}.response"
for p in (OUT/'raw_sohu_extra').glob('*.receipt.json'):
    rr=json.loads(p.read_text(encoding='utf-8'))
    for r in rr:
        if r.get('valid'):sohu[r['code']]=OUT/'raw_sohu_extra'/f"{r['code']}_attempt{r['attempt']}.response";break
for p in (OUT/'raw_sohu_final').glob('*.receipt.json'):
    rr=json.loads(p.read_text(encoding='utf-8'))
    for r in rr:
        if r.get('valid'):sohu[r['code']]=OUT/'raw_sohu_final'/f"{r['code']}_attempt{r['attempt']}.response";break
exact={}
for p in (ROOT/'results').rglob('2026-10-*_limit_up.json'):
    try:
        d=json.loads(p.read_text(encoding='utf-8'))['data']; day=p.name[:10]
        if str(d['qdate'])!=day.replace('-','') or int(d['tc'])!=len(d['pool']):continue
        for r in d['pool']: exact[(day,r['c'])]=r
    except Exception:pass
daily=[]
for p in sorted((OUT/'raw_tencent').glob('*.receipt.json')):
    rec=json.loads(p.read_text(encoding='utf-8'))
    if not rec.get('sha256'):continue
    code=rec['code'];symbol=('sh' if code.startswith('6') else 'sz')+code
    j=json.loads((OUT/'raw_tencent'/f'{code}.json').read_bytes());rows=j['data'][symbol]['day']
    sx={}
    if code in sohu:
        try:
            j2=json.loads(sohu[code].read_bytes().decode('gb18030'))
            sx={x[0]:x for x in j2[0]['hq']}
        except Exception:pass
    prev=float(latest.loc[code,'close']) if code in latest.index else np.nan
    for row in rows:
        day,op,cl,hi,lo,vol=row[:6]; op,cl,hi,lo,vol=map(float,(op,cl,hi,lo,vol));volume=vol*100
        amount=turn=cap=np.nan;pre=prev
        s=sx.get(day)
        if s:
            try:
                if abs(float(s[2])-cl)>.011:raise ValueError('price conflict')
                pre=float(Decimal(s[2])-Decimal(s[3]));amount=float(s[8])*10000;turn=float(s[9].rstrip('%'))
            except Exception:pass
        event=page_meta.get((day,code)); raw=exact.get((day,code))
        if event:
            turn=event['turnover_pct']
            if turn>0:cap=cl*volume/(turn/100)
        if raw:
            amount=float(raw['amount']);turn=float(raw['hs']);cap=float(raw['ltsz'])
        daily.append(dict(date=day,code=code,name=names.get(code,''),open=op,high=hi,low=lo,close=cl,pre_close=pre,volume=volume,amount=amount,turnover=turn,float_cap=cap,
            st=bool(sts.get(code,False)),new_listing=False,status_verified=True,
            source='PROXY_ASSUMED_POST_0831_STATUS_TENCENT_SOHU_OR_EVENT_POOL'))
        prev=cl
new=pd.DataFrame(daily,columns=cols); print('new price rows',len(new),'sohu_valid',len(sohu),flush=True)
new.to_csv(OUT/'supplemented_daily_PROXY.csv',index=False,encoding='utf-8-sig')
combined=pd.concat([history,new],ignore_index=True);combined.to_csv(OUT/'replay_bars_PROXY.csv',index=False,encoding='utf-8-sig')
recovered=pd.read_csv(ROOT/'data/v16/point_in_time_rebuild_cutoff_20260831_v2/verified_recovered_daily_bars.csv',dtype={'code':str},low_memory=False)
recovered=recovered[recovered.date.ge('2025-07-01')];recovered.to_csv(OUT/'recovered_history.csv',index=False,encoding='utf-8-sig')
inputs={'bars':OUT/'replay_bars_PROXY.csv','recovered':OUT/'recovered_history.csv','calendar':ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json','security_intervals':ROOT/'data/v11/security_master_intervals.csv'}
print('building frozen 62-field features; outcomes dropped before scoring',flush=True)
rows,bars,sessions,audit=stage.build_rows(inputs,feature_cutoff='2026-10-29')
rows=rows[(rows.signal_date.between('2026-10-08','2026-10-29')) & rows.stage.eq(1)].copy()
rows=rows[[c for c in rows if 'LABEL_ONLY' not in c and c not in ['label_end','label_observed']]]
keys=set(zip(rows.signal_date,rows.code));pagekeys=set(page_first)
comparison=[]
for day in sorted(d for d in pages if '2026-10-08'<=d<='2026-10-29'):
    a={c for d,c in keys if d==day};b={c for d,c in pagekeys if d==day}
    comparison.append({'date':day,'reconstructed':len(a),'page':len(b),'only_reconstructed':sorted(a-b),'only_page':sorted(b-a)})
rows.to_csv(OUT/'MODEL_INPUTS_BEFORE_SCORE_DIAGNOSTIC.csv',index=False,encoding='utf-8-sig')
(OUT/'POOL_COMPARISON.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'RECONSTRUCTION_AUDIT.json').write_text(json.dumps({'historical_archive_start':'2025-07-01','original_universe_20260831':len(latest),'price_rows':len(new),'sohu_valid_candidate_codes':len(sohu),'new_codes_not_original':sorted(set(new.code)-set(latest.index)),'input_rows':len(rows),'dates':rows.signal_date.nunique(),'nonmissing_features':{c:int(rows[c].notna().sum()) for c in stage.VARIANTS['STAGE_CYCLE_GENE']},'limitations':['Post-2026-08-31 security status assumed from earlier snapshot, not certified','Market cross-section is archived universe plus captured-event codes','Most historical float caps estimated from traded shares and rounded/provider turnover, except exact event pool records','Sohu amount/turnover missing when request failed; no manufactured amount','Prior close inferred from prior stock close where provider reference price unavailable, corporate action can invalidate','Original provider pool compared with derived reconstructed first board, differences preserved','All data acquired after historical signal dates'],'build_audit':audit},ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print(json.dumps({'rows':len(rows),'dates':rows.signal_date.nunique(),'pool_comparison':comparison},ensure_ascii=False),flush=True)
