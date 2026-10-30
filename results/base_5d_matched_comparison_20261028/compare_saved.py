from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'research/model_upgrade_validation/short_term_thinking_20261015T071000Z/delivery/complete_paired_predictions_VALIDATED.csv'
NEW=ROOT/'research/model_upgrade_validation/codex_5d_causal_repair_20261028/oof_all_candidates.csv'
b=pd.read_csv(BASE,dtype={'code':str},float_precision='round_trip')
b=b[b.arm.eq('BASE')]
d=pd.read_csv(NEW,dtype={'code':str},float_precision='round_trip')
m=d.merge(b[['date','target_date','code','score','target','label_valid']],
          left_on=['date','code'],right_on=['target_date','code'],
          suffixes=('_5d','_base'),validate='one_to_one')
# Label differences never remove a candidate from either ranking.
m['label_conflict']=m.target.notna() & m.target_limit_up.notna() & m.target.ne(m.target_limit_up)
m['common_label']=m.target_limit_up.where(m.label_valid & m.target.notna() & m.target.eq(m.target_limit_up))
m['rank_5d']=m.sort_values(['date_5d','score_5d','code'],ascending=[True,False,True]).groupby('date_5d').cumcount().add(1).reindex(m.index)
m['rank_base']=m.sort_values(['date_5d','score_base','code'],ascending=[True,False,True]).groupby('date_5d').cumcount().add(1).reindex(m.index)
cols=['date_5d','code','common_label','next_return_proxy']
a=m[m.rank_5d.eq(1)][cols]
c=m[m.rank_base.eq(1)][cols]
p=a.merge(c,on='date_5d',suffixes=('_5d','_base')).sort_values('date_5d')
p['both_label_known']=p.common_label_5d.notna() & p.common_label_base.notna()
p['both_return_known']=p.next_return_proxy_5d.notna() & p.next_return_proxy_base.notna()
q=p[p.both_label_known]
diff=(p.common_label_5d-p.common_label_base).to_numpy()
rng=np.random.default_rng(20261028)
boots=[]
for _ in range(2000):
    starts=rng.integers(0,len(diff)-4,size=int(np.ceil(len(diff)/5)))
    sample=np.concatenate([diff[start:start+5] for start in starts])[:len(diff)]
    boots.append(np.nanmean(sample)*100)
ret=p[p.both_return_known]
group_sizes=m.groupby('date_5d').size()
multi=q[q.date_5d.isin(group_sizes[group_sizes>=2].index)]
summary=dict(formed_at_utc=datetime.now(timezone.utc).isoformat(),fits=0,
    scope='Exploratory same-member intersection of saved historical forecasts; BASE signal close vs 5D next-day opening features. Not equal information sets or an independent blind test.',
    date_range=[m.date_5d.min(),m.date_5d.max()],intersection_rows=len(m),intersection_dates=int(m.date_5d.nunique()),
    label_conflict_rows=int(m.label_conflict.sum()),unknown_or_conflicting_label_rows=int(m.common_label.isna().sum()),
    complete_paired_dates=len(q),unknown_paired_dates=int((~p.both_label_known).sum()),
    base=dict(close_limit_hits=int(q.common_label_base.sum()),denominator=len(q),close_limit_pct=round(q.common_label_base.mean()*100,2)),
    model5d=dict(close_limit_hits=int(q.common_label_5d.sum()),denominator=len(q),close_limit_pct=round(q.common_label_5d.mean()*100,2)),
    difference_pp=round(float(q.common_label_5d.mean()-q.common_label_base.mean())*100,3),
    exploratory_paired_5_observation_day_block_bootstrap_95pct_pp=np.round(np.quantile(boots,[.025,.975]),3).tolist(),
    same_pick_complete_dates=int(q.code_5d.eq(q.code_base).sum()),
    multi_candidate_complete_paired_dates=len(multi),
    multi_5d_close_limit_pct=round(multi.common_label_5d.mean()*100,2),
    multi_base_close_limit_pct=round(multi.common_label_base.mean()*100,2),
    paired_next_open_known_dates=len(ret),
    base_next_open_gross_mean_pct=round(ret.next_return_proxy_base.mean()*100,3),
    model5d_next_open_gross_mean_pct=round(ret.next_return_proxy_5d.mean()*100,3),
    original_full_pools_not_replaced=True,production_upgrade_proven=False,
    source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [BASE,NEW]})
OUT.mkdir(exist_ok=True)
m.to_csv(OUT/'all_intersection_candidates.csv',index=False,encoding='utf-8-sig')
p.to_csv(OUT/'daily_paired_top1.csv',index=False,encoding='utf-8-sig')
(OUT/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
