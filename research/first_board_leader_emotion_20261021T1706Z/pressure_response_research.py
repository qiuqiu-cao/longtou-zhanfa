"""Unfrozen hypothesis: pre-first-board response to a past high-board shock.

Only date/code first-board membership and nominal past OHLC enter construction.
Targets are joined strictly after the feature/prefix outputs have been saved.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SOURCE=ROOT/'data/v16/point_in_time_rebuild_cutoff_20260831_v2'
CALENDAR=ROOT/'data/v279/exchange_calendar_audit_20261002_v1/calendar.json'
CANDIDATES=ROOT/'research/first_board_leader_20261020T114600Z/all_first_boards_features_and_LABEL_ONLY.csv'
FIELDS=['prior_high_shock_exists','prior_high_shock_age',
        'prelaunch_shock_return','prelaunch_shock_drawdown','prelaunch_shock_rebound',
        'prelaunch_shock_return_same_day_rank']

def true(s):return s.astype(str).str.lower().isin(['true','1','1.0'])
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):x=x.item()
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def states(raw,sessions,cutoff=None):
    x=raw[raw.date.le(cutoff)].copy() if cutoff else raw.copy()
    p={d:i for i,d in enumerate(sessions)}
    x['p']=x.date.map(p);x=x[x.p.notna()].sort_values(['code','p']).reset_index(drop=True)
    x['p']=x.p.astype(int)
    prices=x[['open','high','low','close','pre_close']].apply(pd.to_numeric,errors='coerce')
    cents=(prices*100).round()
    precision=((prices*100-cents).abs().le(1e-7)).all(axis=1)
    valid=prices.gt(0).all(axis=1)&np.isfinite(prices).all(axis=1)&precision
    valid &= cents.low.le(cents[['open','close']].min(axis=1))&cents.high.ge(cents[['open','close']].max(axis=1))&cents.high.ge(cents.low)
    cap=(cents.pre_close*110+50)//100;floor=(cents.pre_close*90+50)//100
    valid &= cents.high.le(cap)&cents.low.ge(floor)
    valid &= true(x.status_verified)&~true(x.st)&~true(x.new_listing)&x.code.str.startswith(('000','001','002','003','600','601','603','605'))
    for field in ['volume','amount']:
        x[field]=pd.to_numeric(x[field],errors='coerce')
        valid &= x[field].gt(0)&np.isfinite(x[field])
    x['known']=valid
    x['close_cents']=cents.close;x['low_cents']=cents.low;x['previous_cents']=cents.pre_close
    x['limit']=valid&cents.close.eq(cap)
    x['lower_hit']=valid&cents.close.eq(floor)
    g=x.groupby('code',sort=False)
    linked=x.limit&g.limit.shift().fillna(False).astype(bool)&x.p.sub(g.p.shift()).eq(1)
    streak=(~linked).groupby(x.code).cumsum()
    x['height']=x.limit.astype(int).groupby([x.code,streak]).cumsum()
    return x[['date','p','code','known','close_cents','low_cents','previous_cents','limit','lower_hit','height']]

def construct(first,st,sessions):
    assert list(first.columns)==['signal_date','code']
    assert not first.duplicated(['signal_date','code']).any()
    positions={d:i for i,d in enumerate(sessions)}
    current=st.set_index(['p','code'])
    high=st[st.limit&st.height.ge(5)][['p','code','height']].copy();high['p']+=1
    joined=high.merge(st[['p','code','known','lower_hit']],on=['p','code'],how='left',validate='one_to_one')
    event_rows=[]
    observed=set(st.loc[st.known,'p'])
    for p in range(int(st.p.min())+1,int(st.p.max())+1):
        group=joined[joined.p.eq(p)]
        known=group.known.eq(True)
        hits=known&group.lower_hit.eq(True)
        unknown=int((~known).sum())
        flag=1. if hits.any() else np.nan if unknown or p not in observed or p-1 not in observed else 0.
        event_rows.append(dict(date=sessions[p],p=p,old_high_members=len(group),old_high_unknown=unknown,
            exact_lower_hits=int(hits.sum()),shock=flag,
            old_high_codes=json.dumps(group.code.tolist()),lower_hit_codes=json.dumps(group.loc[hits,'code'].tolist())))
    events=pd.DataFrame(event_rows).set_index('p')
    rows=[]
    for day,g in first.groupby('signal_date',sort=True):
        p=positions[day];codes=g.code.tolist()
        window=events.reindex(range(max(0,p-10),p))
        positive=window[window.shock.eq(1)].index.tolist()
        E=max(positive) if positive else None
        unknown=window.shock.isna()
        uncertain=bool(unknown.loc[unknown.index>E].any()) if E is not None else bool(unknown.any())
        flag=np.nan if uncertain else float(E is not None)
        part=pd.DataFrame({'signal_date':day,'code':codes})
        part['prior_high_shock_exists']=flag
        part['prior_high_shock_age']=float(p-E) if E is not None and not uncertain else np.nan
        for field in FIELDS[2:]:part[field]=np.nan
        part['E']=sessions[E] if E is not None and not uncertain else None
        part['path_unknown_reason']='LATEST_SHOCK_UNCERTAIN' if uncertain else 'NO_PREVIOUS_SHOCK' if E is None else ''
        if E is not None and not uncertain:
            days=list(range(E-1,p))
            keys=pd.MultiIndex.from_product([days,codes],names=['p','code'])
            q=current.reindex(keys)
            observed_path=q.known.eq(True).to_numpy().reshape(len(days),len(codes)).all(axis=0)
            close=q.close_cents.to_numpy().reshape(len(days),len(codes))
            low=q.low_cents.to_numpy().reshape(len(days),len(codes))
            previous=q.previous_cents.to_numpy().reshape(len(days),len(codes))
            consistent=(previous[1:]==close[:-1]).all(axis=0)
            good=observed_path&consistent
            lowest=np.nanmin(low[1:],axis=0)
            ret=close[-1]/close[0]-1;draw=lowest/close[0]-1;rebound=close[-1]/lowest-1
            for field,value in zip(FIELDS[2:5],[ret,draw,rebound]):part[field]=np.where(good,value,np.nan)
            part['path_unknown_reason']=np.where(~observed_path,'MISSING_OR_UNQUALIFIED_PRICE',np.where(~consistent,'NOMINAL_ADJACENCY_CONFLICT',''))
            # Complete same-day cohort retained; do not rank only observed survivors.
            if good.all():part[FIELDS[5]]=part[FIELDS[2]].rank(method='average',pct=True)
        rows.append(part)
    return pd.concat(rows,ignore_index=True),events.reset_index()

def main():
    sessions=json.loads(CALENDAR.read_text(encoding='utf-8'))['sessions']
    use=['date','code','open','high','low','close','pre_close','volume','amount','st','new_listing','status_verified']
    paths=[SOURCE/'point_in_time_current_daily_bars.csv',SOURCE/'verified_recovered_daily_bars.csv']
    raw=pd.concat([pd.read_csv(p,usecols=use,dtype={'code':str},float_precision='round_trip') for p in paths],ignore_index=True)
    raw['date']=raw.date.astype(str).str[:10];raw['code']=raw.code.str.zfill(6)
    dup=raw[raw.duplicated(['date','code'],keep=False)]
    if len(dup):assert dup.groupby(['date','code'])[['open','high','low','close','pre_close']].nunique().max(axis=1).le(1).all()
    raw=raw.drop_duplicates(['date','code'],keep='last')
    first=pd.read_csv(CANDIDATES,usecols=['signal_date','code'],dtype={'code':str})
    f,events=construct(first,states(raw,sessions),sessions)
    f.to_csv(OUT/'UNFROZEN_PRESSURE_RESPONSE_FEATURES_WITHOUT_TARGETS.csv',index=False,encoding='utf-8-sig')
    events.to_csv(OUT/'ALL_DAILY_OLDHIGH_PRESSURE_EVENTS.csv',index=False,encoding='utf-8-sig')
    prefixes={}
    for cutoff in ['2024-12-31','2025-06-30']:
        partial,_=construct(first[first.signal_date.le(cutoff)],states(raw,sessions,cutoff),sessions)
        original=f[f.signal_date.le(cutoff)].reset_index(drop=True)
        pd.testing.assert_frame_equal(partial.reset_index(drop=True),original,check_dtype=False)
        prefixes[cutoff]=dict(rows=len(partial),mismatches=0)
    # Labels are opened only after the no-target tables and prefix tests exist.
    labels=pd.read_csv(OUT/'ALL28503_PATHS_AND_ORIGINAL_SCREEN.csv',dtype={'code':str},low_memory=False)
    paired=f.merge(labels[['signal_date','code','max_height_5_LABEL_ONLY','allowed','path_group_LABEL_ONLY']],on=['signal_date','code'],validate='one_to_one')
    paired.to_csv(OUT/'DESCRIPTIVE_PRESSURE_RESPONSE_AND_FUTURE_PATHS.csv',index=False,encoding='utf-8-sig')
    summary=dict(at_utc=datetime.now(timezone.utc).isoformat(),fits=0,numeric_protocol_frozen=False,
        rows=len(f),shock_dates=int(events.shock.eq(1).sum()),pressure_unknown_dates=int(events.shock.isna().sum()),
        candidate_recent_shock=int(f.prior_high_shock_exists.eq(1).sum()),candidate_shock_unknown=int(f.prior_high_shock_exists.isna().sum()),
        response_observed=int(f.prelaunch_shock_return.notna().sum()),complete_same_day_ranks=int(f.prelaunch_shock_return_same_day_rank.notna().sum()),
        prefix_checks=prefixes,
        feature_definitions='Last observed exact lower-limit close among previous official-day >=5-board cohort in S-10..S-1; candidate E-1 throughS-1 nominal-price response, before first-board launch. Missing later shock or path is unknown.',
        target_use='None during construction; descriptive linkage afterward only. No threshold/window choice or fitting authorized.',
        limitations=['Archived surviving universe, not complete all-market emotion or verified theme labels.','This pressure-event operator is not a completed emotion-cycle classifier.','No-shock rows are retained and are not automatically rejected.'],
        source_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[CALENDAR,CANDIDATES,Path(__file__)]})
    with (OUT/'CODEX_PRESSURE_RESEARCH_RECEIPT.json').open('x',encoding='utf-8') as handle:json.dump(clean(summary),handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(clean(summary),ensure_ascii=True))

if __name__=='__main__':main()
