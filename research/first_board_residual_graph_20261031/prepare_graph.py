"""Create historical graph without loading outcome labels or fitting models."""
from pathlib import Path
import sys,json,hashlib
from datetime import datetime,timezone
import numpy as np,pandas as pd,joblib
from scipy.special import logit
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
import first_board_long_high_features as source
from graph_features import build_day,FIELDS,KEYS

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def now():return datetime.now(timezone.utc).isoformat()

def main():
    out=HERE/'prepared';out.mkdir(exist_ok=False)
    data=ROOT/'research/first_board_top3_40_20261030/results'
    pool=pd.read_csv(data/'FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv',dtype={'code':str},float_precision='round_trip')
    state=joblib.load(ROOT/'research/gemini_codex_joint_theme_feedback_design_20261021T074448Z/CAUSAL_PRICE_STATES_WITHOUT_TARGETS.joblib')
    bars=source.load_source(); b=bars.merge(state['states'][['date','code','eligible','valid_price','regular_price']],on=['date','code'],how='left',validate='one_to_one')
    b=b.sort_values(['code','p']).reset_index(drop=True);g=b.groupby('code',sort=False)
    prev=g.close.shift();adj=b.p.sub(g.p.shift()).eq(1)
    valid=b.eligible.fillna(False)&b.valid_price.fillna(False)&b.regular_price.fillna(False)&b.volume.gt(0)&b.amount.gt(0)
    prevvalid=valid.groupby(b.code).shift().fillna(False).astype(bool)
    returns=(b.close/prev-1).where(valid&prevvalid&adj&b.pre_close.sub(prev).abs().le(.011))
    b['ret']=returns
    pivot=b.pivot(index='date',columns='code',values='ret').reindex(state['sessions'])
    codes=pivot.columns.tolist(); matrix=pivot.to_numpy(float);positions={d:i for i,d in enumerate(state['sessions'])}
    fields=[];edges=[]; background=[]
    for day,g in pool.groupby('signal_date',sort=True):
        i=positions[day];window=matrix[max(0,i-60):i];scores=dict(zip(g.code,logit(g.s1_score)))
        if len(window)==60:
            frame, pairs=build_day(window,codes,g.code.tolist(),scores)
            background.append({'signal_date':day,'min_background_observed':int(np.isfinite(window).sum(axis=1).min()),'max_background_observed':int(np.isfinite(window).sum(axis=1).max())})
            edges.extend({'signal_date':day,'code_a':a,'code_b':b,'correlation':r,'observations':60} for a,b,r in pairs)
        else:frame=pd.DataFrame({'code':g.code,FIELDS[0]:np.nan,'peer_count':0,'peers':'[]'})
        frame['signal_date']=day;fields.append(frame)
    f=pd.concat(fields,ignore_index=True);assert len(f)==7254 and not f.duplicated(KEYS).any()
    f.to_csv(out/'FACTORS_NO_LABELS.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(edges,columns=['signal_date','code_a','code_b','correlation','observations']).to_csv(out/'QUALIFIED_PAIRS_NO_LABELS.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(background).to_csv(out/'DAILY_BACKGROUND.csv',index=False,encoding='utf-8-sig')
    panel=pool.merge(f,on=KEYS,validate='one_to_one',how='left');panel['fixed_offset']=logit(panel.s1_score)
    panel.to_csv(out/'FULL_POOL_NO_LABELS.csv',index=False,encoding='utf-8-sig')
    summary={'created_at':now(),'rows':len(f),'dates':f.signal_date.nunique(),'finite':int(f[FIELDS[0]].notna().sum()),'days_with_finite':int(f.loc[f[FIELDS[0]].notna(),'signal_date'].nunique()),'qualified_pairs':len(edges),'fits':0,'labels_read':False,'prior_window':'S-60..S-1','market':'archived securities; pair-specific leave-two-out; historical coverage and survivorship not certified','reference':'nominal; no independent corporate action proof'}
    save(out/'QUALIFICATION.json',summary)
    paths=list(HERE.glob('*.py'))+list(out.glob('*'))+[HERE/'PROTOCOL_DEFINITIONS.md',ROOT/'tests/test_residual_graph_competition.py',
       ROOT/'first_board_long_high_features.py',ROOT/'first_board_leader_model.py',ROOT/'first_second_stage_model.py',data/'FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv',data/'OUTCOMES_DIAGNOSTIC_ONLY.csv',
       ROOT/'research/first_second_stage_20261020_run1/S1_ALL_PREDICTIONS.csv',ROOT/'research/first_board_dynamic_rank_20261030/run_experiment.py',ROOT/'research/first_board_dynamic_rank_20261030/pairwise_residual.py',ROOT/'research/first_board_full_public_theme_20261030/rank_offset.py',
       ROOT/'research/first_board_math_20261030/review_gemini_cascade.py',ROOT/'research/first_board_top3_40_20261030/analyze_top3_gap.py',ROOT/'research/first_board_dynamic_rank_20261030/results/A_DYNAMIC_THEME_PICKS_BEFORE_OUTCOME_JOIN.csv',ROOT/'research/gemini_codex_joint_theme_feedback_design_20261021T074448Z/CAUSAL_PRICE_STATES_WITHOUT_TARGETS.joblib']+list(source.base.INPUTS.values())
    save(HERE/'PROTOCOL.json',{'created_at':now(),'arm':'S1_PLUS_PEER_COMPETITION','fields':FIELDS,'bindings':{str(p.relative_to(ROOT)):sha(p) for p in paths},'status':'AWAITING_TWO_ACKS','fits':0})
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
