"""Replay frozen no-target inputs and check causal prefixes before training."""
from pathlib import Path
import sys,json,hashlib
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import first_board_lifecycle_status_features_v3 as f
from audit_lifecycle_v2 import fixture

OUT=Path(__file__).resolve().parent/'qualification_final'
PRIOR=ROOT/'research/gemini_codex_joint_theme_feedback_design_20261021T074448Z'
INPUTS={
 'candidates':PRIOR/'COMPLETE28503_CANDIDATE_HISTORICAL_NAMES_NO_TARGETS.csv',
 'closing_names':PRIOR/'ALL_DECLARED_ARCHIVE_CLOSING_COHORT_NAMES_NO_TARGETS.csv',
 'seal_facts':ROOT/'research/first_board_joint_design_20261020T185417Z/qualification_resumed_20261021/COMPLETE28503_NEW_FIELDS_NO_LABELS.csv',
 'states':PRIOR/'CAUSAL_PRICE_STATES_WITHOUT_TARGETS.joblib',
 'source':Path(f.__file__),
 'tokens':ROOT/'historical_symbolic_context.py',
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 OUT.mkdir(exist_ok=False)
 c,n,seal=[pd.read_csv(INPUTS[k],dtype={'code':str},float_precision='round_trip')
           for k in ['candidates','closing_names','seal_facts']]
 data=joblib.load(INPUTS['states']);st,ss=data['states'],data['sessions']
 fx=f.build(*fixture())
 assert fx[f.ALL_FIELDS].isna().all().all(), 'mixed known/unknown anchor must remain unknown'
 assert f.fractional_rank([1.,1.])==[.5,.5]
 assert all(np.isnan(x) for x in f.fractional_rank([1.,np.nan]))
 full=f.build(c,n,seal,st,ss)
 assert len(full)==28503 and full[f.KEYS].equals(c[f.KEYS])
 full.to_csv(OUT/'COMPLETE28503_LIFECYCLE_STATUS_NO_TARGETS.csv',index=False,encoding='utf-8-sig')
 prefix=[]
 for cut in ['2024-12-31','2025-06-30']:
  last=max(p for p,day in enumerate(ss) if day<=cut)
  part=f.build(c[c.signal_date.le(cut)].reset_index(drop=True), n[n.p.le(last)],
               seal[seal.signal_date.le(cut)],st[st.p.le(last)],ss[:last+1])
  expect=full[full.signal_date.le(cut)].reset_index(drop=True)
  assert part[f.KEYS].equals(expect[f.KEYS])
  np.testing.assert_allclose(part[f.ALL_FIELDS],expect[f.ALL_FIELDS],atol=0,rtol=0,equal_nan=True)
  prefix.append(dict(cutoff=cut,rows=len(part),exact_prefix_equal=True))
 old=pd.read_csv(ROOT/'research/gemini_codex_joint_lifecycle_status_design_20261021T083600Z/qualification_v2/COMPLETE28503_LIFECYCLE_STATUS_NO_TARGETS.csv',dtype={'code':str},float_precision='round_trip')
 same=np.isclose(full[f.ALL_FIELDS],old[f.ALL_FIELDS],atol=1e-14,rtol=0,equal_nan=True)
 changes=full.loc[~same.all(axis=1), f.KEYS]
 changes.to_csv(OUT/'V3_PRE_FIT_SEMANTIC_CHANGE_KEYS.csv',index=False,encoding='utf-8-sig')
 val=full[full.signal_date.between('2025-01-01','2025-06-30')]
 result=dict(formed_at=datetime.now(timezone.utc).isoformat(),status='PASS_PRE_FIT_COMPLETE_UNKNOWN_QUALIFICATION',
   rows=len(full),dates=int(full.signal_date.nunique()),fields=f.ALL_FIELDS,
   validation_rows=len(val),validation_dates=int(val.signal_date.nunique()),
   validation_nonmissing={k:int(val[k].notna().sum()) for k in f.ALL_FIELDS},
   corrected_rows_vs_v2=len(changes),prefix_checks=prefix,
   mixed_known_unknown_anchor_test='all_seven_fields_unknown',ties_and_missing_rank_pass=True,
   peak_scope='Maximum observed archived closing height in S-10:S-1; not an independently complete theme peak',
   group_scope='historical-name symbolic proxy, not a verified true business theme or identity label',
   bindings={str(p.resolve()):sha(p) for p in INPUTS.values()},
   labels_loaded=False,new_fits=0,http_requests=0)
 (OUT/'QUALIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k!='bindings'},ensure_ascii=True),flush=True)

if __name__=='__main__':main()
