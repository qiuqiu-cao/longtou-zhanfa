"""Audit the two reported blends against BOTH saved heads. No fit or weight search."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
HEIGHT = ROOT/'research/first_board_leader_20261020T114600Z'
STAGE = ROOT/'research/first_second_stage_20261020_run1'
SCREEN = ROOT/'research/first_board_leader_emotion_20261021T1706Z/ALL28503_PATHS_AND_ORIGINAL_SCREEN.csv'
KEYS = ['signal_date', 'code']
P2, P4 = 'promotion_next_LABEL_ONLY', 'reach4_LABEL_ONLY'
# Two user-reported blends only; endpoints are the indispensable existing controls.
METHODS = {'HEIGHT_ONLY':0., 'REPORTED_50_50':.5, 'REPORTED_70_30':.7, 'S1_ONLY':1.}


def read(path):
    return pd.read_csv(path, dtype={'code':str}, float_precision='round_trip', low_memory=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    with (OUT/name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def selected(rows, weight, k):
    # No label or later-day field determines the ranking.
    score = weight*rows.s1_score + (1-weight)*rows.height_score
    return rows.assign(combination_score=score).sort_values(
        ['signal_date','combination_score','code'], ascending=[True,False,True]).groupby('signal_date').head(k)


def metric(rows, label):
    known, hits = int(rows[label].notna().sum()), int(rows[label].eq(1).sum())
    return dict(selected=len(rows), known=known, hits=hits, unknown=len(rows)-known,
                precision=hits/known if known else None, hits_per_all_slots=hits/len(rows) if len(rows) else None)


def paired(a, b, dates, label):
    aa=a.groupby('signal_date')[label].agg(['sum','count']).reindex(dates,fill_value=0)
    bb=b.groupby('signal_date')[label].agg(['sum','count']).reindex(dates,fill_value=0)
    w=pd.DataFrame(dict(ah=aa['sum'],an=aa['count'],bh=bb['sum'],bn=bb['count']))
    w.index=pd.to_datetime(w.index).to_period('W')
    values=w.groupby(level=0).sum().to_numpy(dtype=float)
    rng=np.random.default_rng(42); samples=[]; n=len(values)
    for _ in range(2000):
        starts=rng.integers(0,n,size=math.ceil(n/5))
        indices=np.concatenate([(start+np.arange(5))%n for start in starts])[:n]
        ah,an,bh,bn=values[indices].sum(axis=0)
        samples.append(ah/an-bh/bn)
    return dict(delta=metric(a,label)['precision']-metric(b,label)['precision'],
        ci95=np.quantile(samples,[.025,.975]).tolist(),replicates=2000,block_weeks=5,seed=42,
        multiplicity_adjusted=False,interpretation='Descriptive interval after reported weight trials; not independent validation')


def main():
    if (OUT/'REVIEW.json').exists(): raise FileExistsError('Preserve prior review')
    paths=[HEIGHT/'evaluation_ALL_PREDICTIONS.csv',STAGE/'S1_ALL_PREDICTIONS.csv',SCREEN,Path(__file__)]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    height,stage,screen=map(read,paths[:3])
    for frame in (height,stage,screen):
        assert not frame.duplicated(KEYS).any()
    assert set(map(tuple,height[KEYS].to_numpy()))==set(map(tuple,stage[KEYS].to_numpy()))
    assert len(height)==13149 and height.signal_date.nunique()==286
    assert stage.stage.eq(1).all() and stage.model_month.eq(stage.signal_date.str[:7]).all()
    monthly=[]
    for month, part in stage.groupby('model_month',sort=True):
        local=read(STAGE/f'{month}_S1_PREDICTIONS_BEFORE_METRICS.csv').sort_values(KEYS)
        original=part.sort_values(KEYS)
        assert local[KEYS].equals(original[KEYS].reset_index(drop=True))
        np.testing.assert_array_equal(local.model_score.to_numpy(),original.model_score.to_numpy())
        hp=height[height.signal_date.str[:7].eq(month)].sort_values(KEYS)
        saved_h=read(HEIGHT/f'predictions_{month}_BEFORE_METRICS.csv').sort_values(KEYS)
        np.testing.assert_array_equal(saved_h.model_score.to_numpy(),hp.model_score.to_numpy())
        for head,path in [('S1',STAGE/'models'/f'{month}_S1_STAGE_CYCLE_GENE_full_BEFORE_FIT.json'),
                          ('HEIGHT',HEIGHT/'models'/f'{month}_BEFORE_FIT.json')]:
            receipt=json.loads(path.read_text(encoding='utf-8'))
            assert receipt['max_label_end']<month+'-01'
            monthly.append(dict(head=head,month=month,train_max_label_end=receipt['max_label_end'],
                receipt_sha256=sha(path),source=str(path.relative_to(ROOT))))
    merged=height.rename(columns={'model_score':'height_score'}).merge(
        stage[KEYS+['model_score',P2]].rename(columns={'model_score':'s1_score'}),on=KEYS,validate='one_to_one')
    merged=merged.merge(screen[KEYS+['allowed',P2]].rename(columns={P2:'screen_promotion'}),on=KEYS,validate='one_to_one')
    assert merged[P2].fillna(-1).equals(merged.screen_promotion.fillna(-1))
    assert np.isfinite(merged[['height_score','s1_score']]).all().all()
    allowed=merged[merged.allowed.eq(True)].copy()
    assert len(allowed)==7254
    metrics=[]; picks={}; comparisons=[]; full_selections=[]
    for scope, rows in [('all',allowed),('2025H2',allowed[allowed.signal_date.lt('2026-01-01')]),
                        ('2026',allowed[allowed.signal_date.ge('2026-01-01')]),
                        ('non_one_price',allowed[allowed.one_price.eq(0)])]:
        dates=sorted(rows.signal_date.unique())
        for method, weight in METHODS.items():
            for k in (1,3):
                take=selected(rows,weight,k)
                picks[(scope,method,k)]=take
                if scope=='all':
                    full_selections.append(take[KEYS+['name','height_score','s1_score',P2,P4,'one_price']].assign(method=method,k=k))
                for label in (P2,P4):
                    metrics.append(dict(scope=scope,method=method,k=k,label=label,**metric(take,label)))
        for method in ('REPORTED_50_50','REPORTED_70_30','S1_ONLY'):
            for control in ('HEIGHT_ONLY','S1_ONLY'):
                if method==control:continue
                for k in (1,3):
                    for label in (P2,P4):
                        comparisons.append(dict(scope=scope,method=method,control=control,k=k,label=label,
                            **paired(picks[(scope,method,k)],picks[(scope,control,k)],dates,label)))
    report=dict(formed_at=datetime.now(timezone.utc).isoformat(),status='REPORTED_BLENDS_REPRODUCED_WITH_MISSING_S1_CONTROL',
        source_hashes=hashes,full_prediction_rows=len(merged),allowed_rows=len(allowed),dates=286,
        original_monthly_predictions_verified=True,training_clock_checks=monthly,
        methods=METHODS,component_fits=0,weight_search_runs=0,new_forward_observations=0,
        historical_status='Previously examined temporal replay; reported weights already evaluated on it',
        score_semantics='Linear combination of two different-target raw scores; not a calibrated probability',
        metrics=metrics,paired_comparisons=comparisons,production_changed=False,
        decision='No chosen replacement. S1-only is essential comparator for next-day accuracy; four-board and non-one-price regressions remain visible.',
        claims_not_supported=['Top3 hit rate reaches30-36% under reported blends','All non-promotions lose money',
            'Late or isolated first boards always fail','Auction screenshots eliminate all remaining failures',
            'Raw50:50 or70:30 mixing is an optimal or calibrated joint probability'],
        original_objective='Improve the S-close nightly shortlist using only information available by that cutoff, with next-day promotion and later height reported separately; auction reviews do not improve the earlier forecast retrospectively.')
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    pd.DataFrame(metrics).to_csv(OUT/'METRICS.csv',index=False,encoding='utf-8-sig')
    pd.concat(full_selections).to_csv(OUT/'ALL286_SELECTIONS.csv',index=False,encoding='utf-8-sig')
    write('REVIEW.json',report)
    print(json.dumps(dict(status=report['status'],monthly_clock_checks=len(monthly),
        metrics=[m for m in metrics if m['scope']=='all'],
        paired=[c for c in comparisons if c['scope']=='all' and c['control']=='HEIGHT_ONLY']),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
