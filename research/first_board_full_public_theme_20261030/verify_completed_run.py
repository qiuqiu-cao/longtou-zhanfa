"""Read-only result audit: independently reconstruct scores and label counts."""
from pathlib import Path
import hashlib
import json
from datetime import datetime,timezone
import math
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
KEYS=['signal_date','code']
P2='promotion_next_LABEL_ONLY'
P4='reach4_LABEL_ONLY'

def read(p):return pd.read_csv(p,dtype={'code':str},float_precision='round_trip')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    result=HERE/'results'
    protocol=load(HERE/'PROTOCOL_V3.json')
    assert all(sha(ROOT/p)==h for p,h in protocol['bindings'].items())
    started=load(result/'RUN_STARTED_BEFORE_FITS.json')
    for who in ['CODEX','TERRA']:
        ack=load(HERE/f'{who}_FREEZE_ACK.json')
        assert ack['protocol_sha256']==started['protocol_sha256']==sha(HERE/'PROTOCOL_V3.json')
        stamp=ack.get('formed_at',ack.get('created_at'))
        assert datetime.fromisoformat(stamp.replace('Z','+00:00'))<=datetime.fromisoformat(started['created_at'])
    x=read(HERE/'prepared_v3/FULL_POOL_BASES_AND_FEATURES_NO_LABELS.csv')
    labels=read(ROOT/'research/first_board_top3_40_20261030/results/OUTCOMES_DIAGNOSTIC_ONLY.csv')
    maturity=read(ROOT/'research/first_second_stage_20261020_run1/S1_ALL_PREDICTIONS.csv')
    maturity=x[KEYS].merge(maturity[KEYS+['label_end',P2]],on=KEYS,validate='one_to_one',how='left')
    sessions=load(HERE/'prepared_v3/SESSIONS.json');pos={d:i for i,d in enumerate(sessions)}
    assert len(x)==7254 and x.signal_date.nunique()==286 and not x.duplicated(KEYS).any()
    dates=sorted(x.signal_date.unique())
    scores=read(result/'ALL_CANDIDATE_SCORES_NO_LABELS.csv')
    assert len(scores)==14508 and not scores.duplicated(['arm']+KEYS).any()
    report=load(result/'REPORT.json');checks={};model_checks=[]
    models={}
    for path in sorted(result.glob('*_MODEL.json')):
        m=load(path);models[(m['arm'],m['month'])]=m
        first=min(d for d in dates if d.startswith(m['month']))
        cutoff=sessions[pos[first]-1]
        expected=sorted(d for d in dates if pos[d]+1<=pos[cutoff])[-252:]
        assert m['cutoff']==cutoff and m['training_dates']==expected
        members=read(path.with_name(path.name.replace('_MODEL.json','_TRAIN_MEMBERS.csv')))
        pd.testing.assert_frame_equal(members[KEYS].reset_index(drop=True),x.loc[x.signal_date.isin(expected),KEYS].reset_index(drop=True),check_dtype=False)
        mature=maturity[maturity.signal_date.isin(expected)&maturity.label_end.notna()&maturity.label_end.le(cutoff)&maturity[P2].isin([0,1])]
        expected_labels=members[KEYS].merge(mature[KEYS+[P2,'label_end']],on=KEYS,how='left',validate='one_to_one')
        pd.testing.assert_frame_equal(members,expected_labels,check_exact=True,check_dtype=False)
        assert members.label_end.dropna().le(cutoff).all()
        assert len(mature)==m['mature_known_training_labels']
        # Scaling is recomputed from all candidates, including unknown labels.
        train=x[x.signal_date.isin(expected)]
        if m['status']=='FIT':
            values=train[m['fields']].to_numpy(float)
            means=np.array([np.mean(v[np.isfinite(v)]) if np.isfinite(v).any() else 0. for v in values.T])
            scales=np.array([np.std(v[np.isfinite(v)]) if np.isfinite(v).any() else 0. for v in values.T])
            # Original helper disables constants and uses unit scale for inactive fields.
            np.testing.assert_allclose(means,m['means'],rtol=0,atol=1e-14)
            active=np.array(m['active'],bool)
            np.testing.assert_allclose(scales[active],np.array(m['scales'])[active],rtol=0,atol=1e-14)
        model_checks.append({'arm':m['arm'],'month':m['month'],'status':m['status'],'training_dates':len(expected),'timing_and_members_verified':True})
    for arm,base in protocol['arms'].items():
        got=scores[scores.arm.eq(arm)].merge(x[KEYS+[base]+protocol['fields']],on=KEYS,validate='one_to_one')
        for month,g in got.groupby(got.signal_date.str[:7]):
            m=models[(arm,month)]
            residual=np.zeros(len(g))
            if m['status']=='FIT':
                v=g[m['fields']].to_numpy(float)
                z=(v-np.array(m['means']))/np.array(m['scales'])
                z[~np.isfinite(z)]=0.;z[:,~np.array(m['active'],bool)]=0.
                residual=z@np.array(m['coefficients'])
            np.testing.assert_allclose(residual,g.ths_residual,rtol=0,atol=1e-14)
            np.testing.assert_array_equal(g.fixed_offset,g[base])
            np.testing.assert_allclose(g[base]+residual,g.research_score,rtol=0,atol=1e-14)
            empty=g[protocol['fields']].isna().all(axis=1)
            np.testing.assert_array_equal(g.loc[empty,'research_score'],g.loc[empty,base])
        predicted=got.sort_values(['signal_date','research_score','code'],ascending=[True,False,True]).groupby('signal_date').head(3)[KEYS].reset_index(drop=True)
        saved=read(result/f'{arm}_PICKS_BEFORE_OUTCOME_JOIN.csv')
        pd.testing.assert_frame_equal(predicted,saved[KEYS],check_exact=True)
    for arm,m in report['metrics'].items():
        picks=read(result/f'{arm}_PICKS_BEFORE_OUTCOME_JOIN.csv')
        assert len(picks)==858 and sorted(picks.signal_date.unique())==dates
        assert picks.groupby('signal_date').size().eq(3).all() and not picks.duplicated(KEYS).any()
        d=picks.merge(labels[KEYS+[P2,P4]],on=KEYS,validate='one_to_one',how='left').merge(x[KEYS+['one_price']],on=KEYS,validate='one_to_one')
        h,k,u=int(d[P2].eq(1).sum()),int(d[P2].notna().sum()),int(d[P2].isna().sum())
        assert (h,k,u)==(m['hits'],m['known'],m['unknown']) and m['known_precision']==h/k
        assert (int(d[P4].eq(1).sum()),int(d[P4].notna().sum()))==(m['reach4_hits'],m['reach4_known'])
        n=d[d.one_price.eq(0)]
        assert m['non_one_price']=={'selected':len(n),'hits':int(n[P2].eq(1).sum()),'known':int(n[P2].notna().sum()),'unknown':int(n[P2].isna().sum())}
        checks[arm]={'hits':h,'known':k,'unknown':u,'known_precision':h/k,'all858_confirmed_precision':h/858,
            'additional_hits_for_40pct_known_same_denominator':math.ceil(.4*k)-h,
            'additional_hits_for_40pct_all858':math.ceil(.4*858)-h}
    protection=load(ROOT/'research/first_board_dragon_upgrade_20261029/DELETION_RECEIPT.json')['protected_sha256']
    assert all(sha(ROOT/p)==h for p,h in protection.items())
    receipt={'created_at':datetime.now(timezone.utc).isoformat(),'status':'PASS','refits':0,'score_rows_replayed':len(scores),
        'protocol_bindings_verified':len(protocol['bindings']),'protected_hashes_verified':len(protection),'metrics':checks,
        'training_models':model_checks,'audit_script_sha256':sha(Path(__file__))}
    with (HERE/'INDEPENDENT_REPLAY_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
