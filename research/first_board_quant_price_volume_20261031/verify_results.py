"""Independent saved-coefficient replay; no fit and no hyperparameter changes."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
KEYS=['signal_date','code'];P2='promotion_next_LABEL_ONLY';P4='reach4_LABEL_ONLY'
def read(p):return pd.read_csv(p,dtype={'code':str},float_precision='round_trip')
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    protocol=load(HERE/'PROTOCOL.json');out=HERE/'results'
    assert all(sha(ROOT/p)==h for p,h in protocol['bindings'].items())
    started=load(out/'STARTED_BEFORE_FITS.json')
    for who in ['CODEX','TERRA']:
        ack=load(HERE/f'{who}_FREEZE_ACK.json')
        assert ack['protocol_sha256']==started['protocol_sha256']==sha(HERE/'PROTOCOL.json')
        assert datetime.fromisoformat(ack['formed_at'])<datetime.fromisoformat(started['created_at'])
    x=read(HERE/'prepared/FULL_POOL_NO_LABELS.csv')
    outcomes=read(ROOT/'research/first_board_top3_40_20261030/results/OUTCOMES_DIAGNOSTIC_ONLY.csv')
    maturity=read(ROOT/'research/first_second_stage_20261020_run1/S1_ALL_PREDICTIONS.csv')
    maturity=x[KEYS].merge(maturity[KEYS+[P2,'label_end']],on=KEYS,validate='one_to_one')
    s=read(out/'ALL_SCORES_BEFORE_METRICS.csv');assert len(s)==7254
    dates=sorted(x.signal_date.unique());sessions=load(ROOT/'data/v279/exchange_calendar_audit_20261002_v1/calendar.json')['sessions']
    pos={d:i for i,d in enumerate(sessions)};checks=[]
    for month,g in x.groupby(x.signal_date.str[:7]):
        m=load(out/f'{month}_MODEL.json');cutoff=sessions[pos[g.signal_date.min()]-1]
        expected=sorted(d for d in dates if pos[d]+1<=pos[cutoff])[-252:]
        assert m['cutoff']==cutoff and m['training_dates']==expected
        train=x[x.signal_date.isin(expected)]
        y=maturity[maturity.signal_date.isin(expected)&maturity.label_end.notna()&maturity.label_end.le(cutoff)&maturity[P2].isin([0,1])]
        wanted=train[KEYS].merge(y[KEYS+[P2,'label_end']],on=KEYS,how='left',validate='one_to_one')
        pd.testing.assert_frame_equal(read(out/f'{month}_TRAIN_MEMBERS.csv'),wanted.reset_index(drop=True),check_dtype=False,check_exact=True)
        delta=np.zeros(len(g))
        if m['status']=='FIT':
            assert m['solver_success'] and len(expected)>=60
            t=train[m['fields']].to_numpy(float)
            means=np.array([np.mean(v[np.isfinite(v)]) if np.isfinite(v).any() else 0 for v in t.T])
            np.testing.assert_allclose(means,m['means'],rtol=0,atol=1e-14)
            v=g[m['fields']].to_numpy(float)
            z=(v-np.array(m['means']))/np.array(m['scales']);z[~np.isfinite(z)]=0;z[:,~np.array(m['active'],bool)]=0
            delta=z@np.array(m['coefficients'])
        saved=s[s.signal_date.str.startswith(month)].set_index(KEYS).loc[g.set_index(KEYS).index]
        np.testing.assert_allclose(delta,saved.residual,rtol=0,atol=1e-14)
        np.testing.assert_allclose(g.fixed_offset.to_numpy()+delta,saved.research_score,rtol=0,atol=1e-14)
        empty=g[protocol['fields']].isna().all(axis=1)
        np.testing.assert_array_equal(saved.research_score.to_numpy()[empty],g.loc[empty,'fixed_offset'])
        checks.append({'month':month,'status':m['status'],'maturity_and_scores_verified':True})
    pick=s.sort_values(['signal_date','research_score','code'],ascending=[True,False,True]).groupby('signal_date').head(3)[KEYS].reset_index(drop=True)
    pd.testing.assert_frame_equal(pick,read(out/f"{protocol['arm']}_PICKS_BEFORE_METRICS.csv")[KEYS],check_exact=True)
    report=load(out/'REPORT.json');metrics={}
    for name,m in report['metrics'].items():
        p=read(out/f'{name}_PICKS_BEFORE_METRICS.csv')
        assert len(p)==858 and sorted(p.signal_date.unique())==dates and p.groupby('signal_date').size().eq(3).all()
        d=p.merge(outcomes[KEYS+[P2,P4]],on=KEYS,how='left',validate='one_to_one')
        assert (int(d[P2].eq(1).sum()),int(d[P2].notna().sum()),int(d[P2].isna().sum()))==(m['hits'],m['known'],m['unknown'])
        assert (int(d[P4].eq(1).sum()),int(d[P4].notna().sum()))==(m['reach4_hits'],m['reach4_known'])
        metrics[name]={'hits':m['hits'],'known':m['known'],'unknown':m['unknown']}
    swaps=read(out/'SWAP_LEDGER.csv')
    summary={'new_success_replacing_failure':int(((swaps.promoted_promotion==1)&(swaps.displaced_promotion==0)).sum()),
       'new_failure_replacing_success':int(((swaps.promoted_promotion==0)&(swaps.displaced_promotion==1)).sum()),
       'both_success':int(((swaps.promoted_promotion==1)&(swaps.displaced_promotion==1)).sum()),
       'both_failure':int(((swaps.promoted_promotion==0)&(swaps.displaced_promotion==0)).sum()),
       'promoted_success':int(swaps.promoted_promotion.eq(1).sum()),'displaced_success':int(swaps.displaced_promotion.eq(1).sum())}
    protected=load(ROOT/'research/first_board_dragon_upgrade_20261029/DELETION_RECEIPT.json')['protected_sha256']
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    result={'created_at':datetime.now(timezone.utc).isoformat(),'status':'PASS','fits':0,'score_rows_verified':len(s),'metrics':metrics,
       'swap_summary':summary,'monthly_checks':checks,'protected_verified':len(protected),'bindings_verified':len(protocol['bindings']),'verification_script_sha256':sha(__file__)}
    with (HERE/'INDEPENDENT_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
