"""Independent causal/semantic audit. No targets, outcomes, model calls or HTTP."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,shutil
import pandas as pd
import numpy as np
import joblib
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import first_board_lifecycle_status_features as f
from historical_symbolic_context import tokens
OUT=Path(__file__).resolve().parent;DESIGN=OUT.parent
PRIOR=ROOT/'research/gemini_codex_joint_theme_feedback_design_20261021T074448Z'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with Path(p).open('x',encoding='utf-8') as h:json.dump(x,h,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))
qual=read(DESIGN/'LIFECYCLE_STATUS_QUALIFICATION.json')
for p,d in qual['bindings'].items():assert sha(p)==d,p
for p in [ROOT/'first_board_lifecycle_status_features.py',DESIGN/'LIFECYCLE_STATUS_QUALIFICATION.json',DESIGN/'gemini/PROPOSAL.md']:
 shutil.copy2(p,OUT/('ORIGINAL_'+p.name))
c=pd.read_csv(PRIOR/'COMPLETE28503_CANDIDATE_HISTORICAL_NAMES_NO_TARGETS.csv',dtype={'code':str},float_precision='round_trip')
n=pd.read_csv(PRIOR/'ALL_DECLARED_ARCHIVE_CLOSING_COHORT_NAMES_NO_TARGETS.csv',dtype={'code':str},float_precision='round_trip')
seal=pd.read_csv(ROOT/'research/first_board_joint_design_20261020T185417Z/qualification_resumed_20261021/COMPLETE28503_NEW_FIELDS_NO_LABELS.csv',dtype={'code':str},float_precision='round_trip')
data=joblib.load(PRIOR/'CAUSAL_PRICE_STATES_WITHOUT_TARGETS.joblib');states,sessions=data['states'],data['sessions']
saved=pd.read_csv(DESIGN/'COMPLETE28503_LIFECYCLE_STATUS_NO_TARGETS.csv',dtype={'code':str},float_precision='round_trip')
full=f.build(c,n,seal,states,sessions)
assert full[f.KEYS].equals(saved[f.KEYS])
exact_match=np.isclose(full[f.ALL_FIELDS].to_numpy(),saved[f.ALL_FIELDS].to_numpy(),atol=0,rtol=0,equal_nan=True)
maxdiff=float(np.nanmax(np.abs(full[f.ALL_FIELDS].to_numpy()-saved[f.ALL_FIELDS].to_numpy())))
np.testing.assert_allclose(full[f.ALL_FIELDS],saved[f.ALL_FIELDS],atol=1e-14,rtol=0,equal_nan=True)
prefixes=[]
for cutoff in ['2024-12-31','2025-06-30']:
 pi=max(i for i,d in enumerate(sessions) if d<=cutoff)
 part=f.build(c[c.signal_date.le(cutoff)],n[n.p.le(pi)],seal[seal.signal_date.le(cutoff)],states[states.p.le(pi)],sessions)
 expected=full[full.signal_date.le(cutoff)].reset_index(drop=True)
 assert part[f.KEYS].equals(expected[f.KEYS]);np.testing.assert_allclose(part[f.ALL_FIELDS],expected[f.ALL_FIELDS],atol=0,rtol=0,equal_nan=True)
 prefixes.append(dict(cutoff=cutoff,rows=len(part),prefix_equal=True))
old=pd.read_csv(PRIOR/'COMPLETE28503_PRIOR_NAME_FEEDBACK_NO_TARGETS.csv',dtype={'code':str},float_precision='round_trip')
joint=full.merge(old[f.KEYS+['name_prev_followers_current_mean_return']],on=f.KEYS,validate='one_to_one')
m=joint[['name_fixed_followers_return_acceleration','name_prev_followers_current_mean_return']].dropna();res=m.iloc[:,1]-m.iloc[:,0]
redundancy=dict(overlap_rows=len(m),correlation=float(m.corr().iloc[0,1]),subtracted_previous_mean_return_min=float(res.min()),subtracted_previous_mean_return_max=float(res.max()),subtracted_previous_mean_return_std=float(res.std()))
cohort_without_nonself=full[full.theme_candidate_cohort_size.notna()&full.name_anchor_broken_age_sessions.isna()]
cohort_without_nonself.to_csv(OUT/'COHORT_WITHOUT_NONSELF_ANCHOR.csv',index=False,encoding='utf-8-sig')
positions={d:p for p,d in enumerate(sessions)};ng={p:g for p,g in n.groupby('p',sort=False)}
lookup={p:{r.code:(bool(r.eligible and r.valid_price and r.regular_price),float(r.day_return),bool(r.limit),int(r.height)) for r in g.itertuples(index=False)} for p,g in states.groupby('p',sort=False)}
cc=c.merge(seal[f.KEYS+['reported_first_seal_trading_minutes','reported_close_seal_log_amount_ratio']],on=f.KEYS,validate='one_to_one')
issues=[]
for day,g in cc.groupby('signal_date',sort=False):
 p=positions[day]
 if any(p-lag not in ng or p-lag not in lookup or not ng[p-lag].name_ok.all() for lag in [1,2,3]):continue
 anchor={};fol={}
 for lag in [3,2,1]:
  for rr in ng[p-lag].itertuples(index=False):
   for t in tokens(rr.historical_name):
    if rr.height>=3:anchor.setdefault(t,set()).add(rr.code)
    if lag==1 and rr.height in (1,2):fol.setdefault(t,set()).add(rr.code)
 records=g.to_dict('records');matched={r['code']:{t for t in tokens(r['historical_name']) if t in anchor} if r['name_ok'] else set() for r in records}
 for r in records:
  code=r['code'];ts=matched[code]
  if not ts:continue
  aa=set().union(*(anchor[t] for t in ts))-{code};ff=set().union(*(fol.get(t,set()) for t in ts))-{code}
  pp=[z for z in records if matched[z['code']]&ts]
  badanchor=[a for a in aa if not lookup.get(p-1,{}).get(a,(False,))[0]]
  badfollower=[a for a in ff if not lookup.get(p,{}).get(a,(False,))[0] or not lookup.get(p-1,{}).get(a,(False,))[0]]
  ft=r['reported_first_seal_trading_minutes'];cs=r['reported_close_seal_log_amount_ratio']
  ties_ft=sum(np.isfinite(z['reported_first_seal_trading_minutes']) and z['reported_first_seal_trading_minutes']==ft for z in pp)
  ties_cs=sum(np.isfinite(z['reported_close_seal_log_amount_ratio']) and z['reported_close_seal_log_amount_ratio']==cs for z in pp)
  if badanchor or badfollower or ties_ft>1 or ties_cs>1 or not np.isfinite(ft) or not np.isfinite(cs):
   issues.append(dict(signal_date=day,code=code,bad_anchor_p1='|'.join(sorted(badanchor)),bad_follower_p1_or_p='|'.join(sorted(badfollower)),cohort_size=len(pp),first_seal_tie_count=ties_ft,close_seal_tie_count=ties_cs,first_missing=not np.isfinite(ft),close_missing=not np.isfinite(cs)))
pd.DataFrame(issues).to_csv(OUT/'ACTUAL_STATE_AND_TIE_ISSUES.csv',index=False,encoding='utf-8-sig')

def fixture(kind):
 ss=[f'D{i}' for i in range(10)];day='D9';codes=['100001','100002']
 cand=pd.DataFrame([dict(signal_date=day,code=x,historical_name='洋甲' if i==0 else '洋乙',name_ok=True) for i,x in enumerate(codes)])
 closing=[]
 for p in [6,7,8]:
  closing.append(dict(signal_date=ss[p],p=p,code='100020',historical_name='洋跟',name_ok=True,height=1))
  if not (kind=='missing_anchor' and p==8):closing.append(dict(signal_date=ss[p],p=p,code='100010',historical_name='洋锚',name_ok=True,height=p-3))
 named=pd.DataFrame(closing)
 sf=pd.DataFrame([dict(signal_date=day,code=x,reported_first_seal_trading_minutes=np.nan if kind=='missing_seal' else 0.,reported_close_seal_log_amount_ratio=2.) for x in codes])
 st=[]
 for p in range(10):
  for code in ['100010','100020']:
   if kind=='missing_anchor' and p==8 and code=='100010':continue
   known=not(kind=='invalid_follower' and p==9 and code=='100020')
   st.append(dict(p=p,code=code,eligible=known,valid_price=True,regular_price=True,day_return=.1 if p<9 else -.05,limit=True,height=p-3 if p>=6 else 0))
 ans=f.build(cand,named,sf,pd.DataFrame(st),ss)
 return ans.replace({np.nan:None}).to_dict('records')
fixtures={k:fixture(k) for k in ['same_seal_tie','missing_anchor','invalid_follower','missing_seal']}
report=dict(formed_at=datetime.now(timezone.utc).isoformat(),status='PRE_FREEZE_REVIEW_REQUIRES_CORRECTIONS_NO_FITS',full28503_numerical_replay=True,replay_tolerance_absolute=1e-14,replay_exact_different_cells=int((~exact_match).sum()),replay_max_absolute_difference=maxdiff,exact_replay_claim=False,prefix_checks=prefixes,no_future_inputs_seen=True,existing_qualification_does_not_cover_semantic_edge_cases=True,actual_rows_with_first_seal_ties=sum(r['first_seal_tie_count']>1 for r in issues),actual_rows_with_close_seal_ties=sum(r['close_seal_tie_count']>1 for r in issues),actual_anchor_price_unknown_rows=sum(bool(r['bad_anchor_p1']) for r in issues),actual_follower_price_unknown_rows=sum(bool(r['bad_follower_p1_or_p']) for r in issues),actual_missing_first_rows=sum(r['first_missing'] for r in issues),actual_missing_close_rows=sum(r['close_missing'] for r in issues),actual_cohort_without_nonself_anchor=len(cohort_without_nonself),validation_cohort_without_nonself_anchor=int(cohort_without_nonself.signal_date.between('2025-01-01','2025-06-30').sum()),acceleration_redundancy_with_old_feedback=redundancy,synthetic_reproducers=fixtures,peak_definition_mismatch='Proposal states [S-10,S-1] peak; implementation max of only [S-3,S-1] anchor closing heights. Reset/rebound rule remains unspecified.',new_fits=0,labels_loaded=False,http_requests=0,original_source_hash=sha(ROOT/'first_board_lifecycle_status_features.py'))
write(OUT/'INDEPENDENT_AUDIT.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='synthetic_reproducers'},ensure_ascii=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))
print(json.dumps(fixtures,ensure_ascii=False))
