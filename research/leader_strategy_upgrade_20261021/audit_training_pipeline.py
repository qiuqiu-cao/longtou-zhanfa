"""Pre-fit audit of membership, calibration chronology, gates and frozen baselines."""
from pathlib import Path
import sys,json,hashlib
from datetime import datetime,timezone
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import first_board_joint_lifecycle_status_model as m
OUT=Path(__file__).resolve().parent

def main():
 # A preflight cannot accidentally fit either an arm or the baseline.
 def forbidden(*a,**k):raise AssertionError('fit forbidden in pre-training audit')
 m.stage.fit_head=forbidden
 m.stage.fit_core=forbidden
 rows=m.load_rows()
 ref=m.prior.prior.REFERENCE
 fields=m.KEYS+['label_end',m.stage.TARGET]
 windows=[]
 sessions=json.loads(m.base.INPUTS['calendar'].read_text(encoding='utf-8'))['sessions']
 periods=[('validation','2025-01-01',None)]
 for month in sorted(rows.loc[rows.signal_date.ge('2025-07-01'),'signal_date'].str[:7].unique()):
  boundary=month+'-01'
  periods.append((month,boundary,int(np.searchsorted(sessions,boundary))-252))
 assert len(periods)==15
 for name,boundary,start in periods:
  train=m.members.train_members(rows,boundary,start)
  file=ref/'models'/('validation_S1_STAGE_CYCLE_GENE_full_members.csv' if name=='validation'
                    else name+'_S1_STAGE_CYCLE_GENE_full_members.csv')
  old=pd.read_csv(file,dtype={'code':str},float_precision='round_trip')
  assert set(map(tuple,train[fields].to_numpy()))==set(map(tuple,old[fields].to_numpy()))
  assert train.label_end.max()<boundary and train.signal_date.max()<boundary
  dates=sorted(train.signal_date.unique());cutoff=dates[-42]
  inner=train[train.label_end.lt(cutoff)]
  cal=train[train.signal_date.ge(cutoff)]
  assert inner.label_end.max()<cal.signal_date.min()
  assert cal.label_end.max()<boundary and len(cal.signal_date.unique())==42
  assert set(inner[m.stage.TARGET].unique())=={0.,1.}
  for arm,cols in m.ARMS.items():
   assert len(cols)==len(set(cols)) and m.stage.TARGET not in cols and 'label_end' not in cols
   assert all(pd.api.types.is_numeric_dtype(train[c]) for c in cols)
  windows.append(dict(name=name,train_rows=len(train),train_signal_dates=len(dates),
                      max_train_label_end=train.label_end.max(),scoring_boundary=boundary,
                      calibration_cut=cutoff,inner_max_label_end=inner.label_end.max(),
                      calibration_rows=len(cal),calibration_signal_dates=42))
 val=rows[rows.signal_date.between('2025-01-01','2025-06-30')]
 assert len(val)==4724 and val.signal_date.nunique()==117
 baseline=m.risk.attach_frozen(rows,ref/'validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv')
 bm=m.risk.screen_metric(baseline)
 bn=m.risk.screen_metric(baseline[baseline.one_price.eq(0)])
 assert bm['hits']==52 and bn['hits']==48 and bm['full_signal_dates']==117
 params=m.base.make_model('hist_gradient').get_params()
 assert params['early_stopping'] is False and params['random_state']==42
 assert params['max_iter']==160 and params['max_leaf_nodes']==15
 # This intentionally leaves unknown/abstained full dates in the goal denominator.
 proof=dict(example_full_dates=100,example_known=95,example_hits=57,
            known_only_rate=57/95,full_date_success_rate=57/100,
            target60_passes=False)
 result=dict(at=datetime.now(timezone.utc).isoformat(),status='CODEX_REVIEWED_BEFORE_ANY_FIT',
   source_sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
   complete_rows=len(rows),validation_rows=len(val),validation_full_dates=117,
   baseline_primary=bm,baseline_non_one_price=bn,training_windows=windows,
   histogram_parameters=params,calibration='42 last train dates; inner labels strictly mature before calibration starts; all calibration labels mature before scoring boundary',
   goal_denominator='Full signal dates; unknown outcomes do not disappear into known-only hit rate',
   denominator_regression=proof,
   agreements_required=True,protocol_status_rejects_unfinished_audit=True,
   monthly_evaluation_only_after_validation_winner=True,
   no_old_base_or_second_head_refits=True,
   new_model_fits=0,production_changes=False,
   remaining_limits=['Historical replay uses previously seen research periods, not a new pristine blind test',
     'Name-proxy groups are not independently verified real themes',
     'Source and survivor-universe limitations of inherited data remain',
     'Feature prefix checks were at two cutoffs, not proof of all possible leakage',
     'Passing prefit review is not predictive improvement; 60% still unachieved'])
 (OUT/'TRAINING_PIPELINE_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'status':result['status'],'windows':len(windows),'rows':len(rows),
                   'baseline_hits':bm['hits'],'baseline_non_one_price_hits':bn['hits'],'fits':0},ensure_ascii=True),flush=True)

if __name__=='__main__':main()
