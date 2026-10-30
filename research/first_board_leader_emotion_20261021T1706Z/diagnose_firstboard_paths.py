"""Complete first-board path and same-date counterexample research; no fits."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
PATHS=ROOT/'research/first_board_leader_20261020T114600Z/all_first_boards_features_and_LABEL_ONLY.csv'
SCREEN=ROOT/'research/first_board_capital_risk_20261020_run1/COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):x=x.item()
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
d=pd.read_csv(PATHS,dtype={'code':str},low_memory=False)
s=pd.read_csv(SCREEN,dtype={'code':str},low_memory=False)
assert len(d)==len(s)==28503 and not d.duplicated(['signal_date','code']).any()
columns=['signal_date','code','research_selection_allowed','exclusion_reason','promotion_next_LABEL_ONLY']
d=d.merge(s[columns],on=['signal_date','code'],validate='one_to_one')
d['allowed']=d.research_selection_allowed.astype(str).str.lower().isin(['true','1','1.0'])
h=pd.to_numeric(d.max_height_5_LABEL_ONLY,errors='coerce')
d['path_group_LABEL_ONLY']=np.select([h.eq(1),h.eq(2),h.eq(3),h.eq(4),h.ge(5)],['STOP1','STOP2','STOP3','STOP4','REACH5PLUS'],default='UNKNOWN')
known=h.notna()
assert d.loc[known,'reach5_LABEL_ONLY'].eq(h[known].ge(5).astype(float)).all()
d['source_scope']='Archived surviving universe; dated reconstruction, not original-time acquisition or real theme identity'
d.to_csv(OUT/'ALL28503_PATHS_AND_ORIGINAL_SCREEN.csv',index=False,encoding='utf-8-sig')
daily=[]
for day,g in d.groupby('signal_date',sort=True):
    row=dict(signal_date=day,rows=len(g),allowed=int(g.allowed.sum()),known=int(g.max_height_5_LABEL_ONLY.notna().sum()))
    row.update(g.path_group_LABEL_ONLY.value_counts().reindex(['STOP1','STOP2','STOP3','STOP4','REACH5PLUS','UNKNOWN'],fill_value=0).to_dict())
    row['mixed_high5_and_stop1']=row['REACH5PLUS']>0 and row['STOP1']>0
    row['complete_path_cohort']=row['UNKNOWN']==0
    row['allowed_high5']=int((g.allowed&g.max_height_5_LABEL_ONLY.ge(5)).sum())
    for field in ['market_max_height','market_failed_rate','previous_top_return','previous_top_broken','previous_first_promotion']:
        assert g[field].nunique(dropna=True)<=1,(day,field)
        row[field]=g[field].dropna().iloc[0] if g[field].notna().any() else np.nan
    daily.append(row)
daily=pd.DataFrame(daily)
daily.to_csv(OUT/'ALL668_DAILY_PATH_COHORTS.csv',index=False,encoding='utf-8-sig')
contrasts=[]
for day,g in d[d.allowed].groupby('signal_date',sort=True):
    winners=g[g.max_height_5_LABEL_ONLY.ge(5)]
    short=g[g.max_height_5_LABEL_ONLY.le(3)]
    if not len(winners) or not len(short):continue
    for f in ['turnover','turnover_rank','volume_ratio_5','return_5','return_20','prior_limits_20','one_price','float_cap']:
        contrasts.append(dict(signal_date=day,field=f,long_count=len(winners),short_count=len(short),
            long_known=int(winners[f].notna().sum()),short_known=int(short[f].notna().sum()),
            long_median=winners[f].median(),short_median=short[f].median(),
            difference=winners[f].median()-short[f].median(),descriptive_not_new_predictive_validation=True))
pd.DataFrame(contrasts).to_csv(OUT/'WITHIN_DATE_EXISTING_FEATURE_CONTRASTS.csv',index=False,encoding='utf-8-sig')
cases=[]
for year in (2024,2025,2026):
    options=daily[daily.signal_date.str.startswith(str(year))&daily.mixed_high5_and_stop1&daily.complete_path_cohort&daily.allowed_high5.gt(0)]
    if not len(options):continue
    day=options.iloc[0].signal_date
    g=d[d.signal_date.eq(day)].copy()
    g.to_csv(OUT/f'COMPLETE_CASE_{day}.csv',index=False,encoding='utf-8-sig')
    cases.append(dict(rule='First chronological complete same-date cohort in year with allowed reach5plus and an observed stop1',
        signal_date=day,complete_rows=len(g),allowed=int(g.allowed.sum()),
        high5=g[g.max_height_5_LABEL_ONLY.ge(5)][['code','name','allowed','turnover','one_price','max_height_5_LABEL_ONLY']].to_dict('records'),
        stop1_count=int(g.max_height_5_LABEL_ONLY.eq(1).sum())))
summary=dict(at_utc=datetime.now(timezone.utc).isoformat(),fits=0,
    scope=dict(rows=len(d),dates=d.signal_date.nunique(),first=d.signal_date.min(),last=d.signal_date.max(),
        known=int(known.sum()),unknown=int((~known).sum()),allowed=int(d.allowed.sum()),excluded=int((~d.allowed).sum())),
    path_groups=d.path_group_LABEL_ONLY.value_counts().to_dict(),
    path_groups_allowed=d[d.allowed].path_group_LABEL_ONLY.value_counts().to_dict(),
    days_with_known_high5=int(daily.REACH5PLUS.gt(0).sum()),
    complete_path_days=int(daily.complete_path_cohort.sum()),
    days_same_emotion_but_high5_and_stop1=int(daily.mixed_high5_and_stop1.sum()),
    cases=cases,
    maturity='Original uniform S+5 official-session horizon; max_height_5 is capped at6 because it includes first board plus5 successors. >=5 is a continuation proxy, not a verified leader label.',
    statistical_status='Descriptive already-seen history, no new model fit/selection/validation; original risk exclusions retained but all excluded winners and unknowns saved.',
    source_hashes={str(p.resolve()):sha(p) for p in [PATHS,SCREEN,Path(__file__)]})
with (OUT/'CODEX_PATH_RESEARCH_SUMMARY.json').open('x',encoding='utf-8') as f:json.dump(clean(summary),f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(clean(summary),ensure_ascii=True))
