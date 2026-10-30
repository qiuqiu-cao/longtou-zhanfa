from pathlib import Path
from datetime import datetime,timezone
import json,sys
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
import first_second_stage_model as stage
d=pd.read_csv(OUT/'ALL28503_PATHS_AND_ORIGINAL_SCREEN.csv',dtype={'code':str},low_memory=False)
oldfields=stage.VARIANTS['STAGE_CYCLE_GENE']
missing=[field for field in oldfields if field not in d]
if missing:
    supplement=pd.read_csv(ROOT/'research/first_board_capital_risk_20261020_run1/COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv',
        usecols=['signal_date','code']+missing,dtype={'code':str})
    d=d.merge(supplement,on=['signal_date','code'],validate='one_to_one')
f=pd.read_csv(OUT/'UNFROZEN_PRESSURE_RESPONSE_FEATURES_WITHOUT_TARGETS.csv',dtype={'code':str})
fields=['prelaunch_shock_return','prelaunch_shock_drawdown','prelaunch_shock_rebound','prelaunch_shock_return_same_day_rank']
x=d.merge(f[['signal_date','code','E','prior_high_shock_exists','prior_high_shock_age']+fields],on=['signal_date','code'],validate='one_to_one')
contrasts=[]
for day,g in x[x.allowed].groupby('signal_date',sort=True):
    long=g[g.max_height_5_LABEL_ONLY.ge(5)];short=g[g.max_height_5_LABEL_ONLY.le(3)]
    if len(long)==0 or len(short)==0:continue
    for field in fields:
        complete=long[field].notna().all() and short[field].notna().all()
        contrasts.append(dict(signal_date=day,field=field,long_count=len(long),short_count=len(short),
            long_observed=int(long[field].notna().sum()),short_observed=int(short[field].notna().sum()),complete_pair=complete,
            difference=float(long[field].median()-short[field].median()) if complete else np.nan))
c=pd.DataFrame(contrasts)
c.to_csv(OUT/'ALL226_DATE_MATCHED_RESPONSE_CONTRASTS.csv',index=False,encoding='utf-8-sig')
byyear=[]
for (year,field),g in c.assign(year=c.signal_date.str[:4]).groupby(['year','field']):
    v=g.difference.dropna()
    byyear.append(dict(year=year,field=field,mixed_dates=len(g),complete_observed_dates=len(v),
        median_difference=float(v.median()) if len(v) else None,long_higher_dates=int(v.gt(0).sum()),long_lower_dates=int(v.lt(0).sum())))
correlations=[]
for added in fields:
    for old in oldfields:
        pair=x[[added,old]].dropna()
        r=pair[added].corr(pair[old],method='spearman') if len(pair)>2 and pair[added].nunique()>1 and pair[old].nunique()>1 else np.nan
        correlations.append(dict(new_field=added,old_field=old,paired_rows=len(pair),spearman=r))
cc=pd.DataFrame(correlations)
cc.to_csv(OUT/'NEW_RESPONSE_VS_OLD62_ASSOCIATIONS.csv',index=False,encoding='utf-8-sig')
top=cc.assign(magnitude=cc.spearman.abs()).sort_values('magnitude',ascending=False).groupby('new_field').head(3).drop(columns='magnitude')
matches=[]
for day in ['2024-01-02','2025-01-02','2026-03-04']:
    g=x[x.signal_date.eq(day)]
    g.to_csv(OUT/f'COMPLETE_CASE_WITH_RESPONSE_{day}.csv',index=False,encoding='utf-8-sig')
    for _,winner in g[g.allowed&g.max_height_5_LABEL_ONLY.ge(5)].iterrows():
        controls=g[g.allowed&g.max_height_5_LABEL_ONLY.le(3)&g.turnover.gt(0)&g.float_cap.gt(0)].copy()
        controls['distance']=np.abs(np.log(controls.turnover/winner.turnover))+np.abs(np.log(controls.float_cap/winner.float_cap))
        controls=controls.sort_values(['distance','code'])
        if not len(controls):continue
        loser=controls.iloc[0]
        selected=['code','name','turnover','float_cap','volume_ratio_5','return_20','path_group_LABEL_ONLY','E']+fields
        matches.append(dict(signal_date=day,match_rule='Nearest observed failed/short-pulse allowed stock by absolute log turnover distance plus absolute log float-cap distance, code breaks ties; posthoc case matching, not a ranker.',
                            long_candidate=winner[selected].to_dict(),short_candidate=loser[selected].to_dict()))
bounds=[]
for name,part,dates in [('all',d,668),('original_allowed',d[d.allowed],668),('2025H1',d[d.signal_date.between('2025-01-01','2025-06-30')],117)]:
    groups=part.groupby('signal_date')
    present=int(groups.max_height_5_LABEL_ONLY.agg(lambda s:s.ge(5).any()).sum())
    possible=int(groups.max_height_5_LABEL_ONLY.agg(lambda s:s.ge(5).any() or s.isna().any()).sum())
    bounds.append(dict(scope=name,full_dates=dates,known_high5_dates=present,known_date_fraction=present/dates,
        possible_high5_dates_with_all_unknowns=possible,possible_date_fraction=possible/dates))
receipt=dict(at_utc=datetime.now(timezone.utc).isoformat(),fits=0,matched_date_response_by_year=byyear,
    largest_correlations_with_existing_62=top.to_dict('records'),posthoc_nearest_controls=matches,
    high5_date_availability=bounds,
    interpretation='Descriptive, already seen data; no correlation/feature contrast is a predictive-validation gain. Response differences vary and do not certify capital intent. Full candidates/unknowns are retained outside complete-pair diagnostics.')
def clean(v):
    if isinstance(v,dict):return {k:clean(z) for k,z in v.items()}
    if isinstance(v,list):return [clean(z) for z in v]
    if isinstance(v,np.generic):v=v.item()
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
with (OUT/'CODEX_RESPONSE_ASSOCIATION_RESEARCH.json').open('x',encoding='utf-8') as handle:json.dump(clean(receipt),handle,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(clean(receipt),ensure_ascii=True))
