"""One pre-agreed daily-price-distribution factor arm; no tuning or deployment."""
from pathlib import Path
import sys,json,hashlib,argparse
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy.special import logit
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'research/first_board_dynamic_rank_20261030'))
sys.path.insert(0,str(ROOT/'research/first_board_full_public_theme_20261030'))
import run_experiment as old
from rank_offset import fit,score
import first_board_long_high_features as source
import first_second_stage_model as stage
from quant_factors import build,FIELDS,KEYS
TARGET='promotion_next_LABEL_ONLY';ARM='S1_PLUS_DAILY_VWAP_DISTRIBUTION'
DATA=ROOT/'research/first_board_top3_40_20261030/results'

def clock():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p,**kw):return pd.read_csv(p,dtype={'code':str},float_precision='round_trip',**kw)
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,o):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)

def prepare():
    dest=HERE/'prepared';dest.mkdir(exist_ok=False)
    x=read(DATA/'FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv')
    save(HERE/'PREPARATION_STARTED_BEFORE_LABELS.json',{'created_at':clock(),'rows':len(x),'fields':FIELDS,'window':20,'fits':0,'labels_read':False})
    bars=source.load_source();bars=bars[bars.code.isin(x.code.unique())]
    z=build(bars,x[KEYS]);assert len(z)==7254
    facts=read(ROOT/'research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv',
        usecols=KEYS+['stage','close']+stage.VARIANTS['STAGE_CYCLE_GENE'])
    facts=facts[facts.stage.eq(1)]
    joined=z.merge(facts,on=KEYS,validate='one_to_one',how='left')
    assert np.array_equal(np.rint(joined.source_close*100),np.rint(joined.close*100))
    comparisons=[];duplicates=[]
    for field in FIELDS:
        for baseline in stage.VARIANTS['STAGE_CYCLE_GENE']:
            a,b=joined[field],joined[baseline]
            mask=a.notna()&b.notna()
            if mask.sum() and np.array_equal(a[mask],b[mask]):duplicates.append([field,baseline])
            if mask.sum()>1 and a[mask].nunique()>1 and b[mask].nunique()>1:
                rho=a[mask].corr(b[mask],method='spearman')
                comparisons.append({'new_factor':field,'old_field':baseline,'matched':int(mask.sum()),'spearman':float(rho)})
    pd.DataFrame(comparisons).to_csv(dest/'REDUNDANCY_NO_LABELS.csv',index=False,encoding='utf-8-sig')
    prefix=[]
    for cutoff in ['2025-12-31','2026-04-30']:
        a=build(bars[bars.date.le(cutoff)],x.loc[x.signal_date.le(cutoff),KEYS])
        pd.testing.assert_frame_equal(a,z[z.signal_date.le(cutoff)].reset_index(drop=True),check_exact=True)
        prefix.append({'cutoff':cutoff,'rows':len(a),'equal':True})
    z.to_csv(dest/'FACTORS_NO_LABELS.csv',index=False,encoding='utf-8-sig')
    panel=x.merge(z.drop(columns='source_close'),on=KEYS,validate='one_to_one')
    panel['fixed_offset']=logit(panel.s1_score)
    panel.to_csv(dest/'FULL_POOL_NO_LABELS.csv',index=False,encoding='utf-8-sig')
    summary={'created_at':clock(),'rows':len(z),'dates':z.signal_date.nunique(),'fields':FIELDS,
       'coverage':{f:int(z[f].notna().sum()) for f in FIELDS},'status_counts':z.factor_source_status.value_counts().to_dict(),
       'constant_values':{f:int(z[f].dropna().nunique()) for f in FIELDS},'exact_equal_old_fields':duplicates,
       'prefix_checks':prefix,'fits':0,'labels_read':False,
       'source_limitations':'Nominal archive, not independently verified corporate-action history; current universe survivorship; all history previously inspected.'}
    save(dest/'QUALIFICATION.json',summary)
    paths=list(HERE.glob('*.py'))+list(dest.glob('*'))+[HERE/'PROTOCOL_DEFINITIONS.md',ROOT/'tests/test_daily_vwap_distribution.py',
       ROOT/'first_board_long_high_features.py',ROOT/'first_board_leader_model.py',ROOT/'first_second_stage_model.py',
       DATA/'FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv',DATA/'OUTCOMES_DIAGNOSTIC_ONLY.csv',old.S1/'S1_ALL_PREDICTIONS.csv',
       ROOT/'research/first_board_dynamic_rank_20261030/run_experiment.py',ROOT/'research/first_board_dynamic_rank_20261030/pairwise_residual.py',
       ROOT/'research/first_board_full_public_theme_20261030/rank_offset.py',
       ROOT/'research/first_board_math_20261030/review_gemini_cascade.py',ROOT/'research/first_board_top3_40_20261030/analyze_top3_gap.py',
       ROOT/'research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv',
       ROOT/'research/first_board_dynamic_rank_20261030/results/A_DYNAMIC_THEME_PICKS_BEFORE_OUTCOME_JOIN.csv']+list(source.base.INPUTS.values())
    bindings={str(p.relative_to(ROOT)):sha(p) for p in paths}
    save(HERE/'PROTOCOL.json',{'created_at':clock(),'arm':ARM,'fields':FIELDS,'bindings':bindings,'status':'AWAITING_ACTUAL_TWO_ACKS','fit_count':0})
    print(json.dumps(summary,ensure_ascii=False))

def run():
    protocol=load(HERE/'PROTOCOL.json')
    assert protocol['arm']==ARM and protocol['fields']==FIELDS
    for who in ['CODEX','TERRA']:
        ack=load(HERE/f'{who}_FREEZE_ACK.json')
        assert ack['status']=='AGREED_EXPLORATORY_PROTOCOL' and ack['protocol_sha256']==sha(HERE/'PROTOCOL.json')
    assert all(sha(ROOT/p)==h for p,h in protocol['bindings'].items())
    out=HERE/'results';out.mkdir(exist_ok=False)
    save(out/'STARTED_BEFORE_FITS.json',{'created_at':clock(),'protocol_sha256':sha(HERE/'PROTOCOL.json')})
    x=read(HERE/'prepared/FULL_POOL_NO_LABELS.csv');labels=read(DATA/'OUTCOMES_DIAGNOSTIC_ONLY.csv')
    maturity=read(old.S1/'S1_ALL_PREDICTIONS.csv',usecols=KEYS+['label_end',TARGET])
    labels=labels.merge(maturity.rename(columns={TARGET:'saved_target'}),on=KEYS,validate='one_to_one')
    assert (labels[TARGET].eq(labels.saved_target)|(labels[TARGET].isna()&labels.saved_target.isna())).all()
    sessions=load(source.base.INPUTS['calendar'])['sessions'];dates=sorted(x.signal_date.unique())
    predictions=[];picks=[];models=[]
    for month,g in x.groupby(x.signal_date.str[:7]):
        train,y,cutoff,ds=old.maturity_window(x,labels,sessions,g.signal_date.min())
        m={'status':'FALLBACK','reason':'FEWER_THAN_60_MATURE_SIGNAL_DAYS','fields':FIELDS} if len(ds)<60 else fit(train,y,FIELDS)
        m.update(month=month,cutoff=cutoff,training_dates=ds,training_candidates=len(train),mature_known_training_labels=len(y),
                 max_training_label_end=y.label_end.max() if len(y) else None,created_at=clock())
        save(out/f'{month}_MODEL.json',m);models.append(m)
        train[KEYS].merge(y[KEYS+[TARGET,'label_end']],on=KEYS,how='left',validate='one_to_one').to_csv(out/f'{month}_TRAIN_MEMBERS.csv',index=False,encoding='utf-8-sig')
        g=g.copy();g['research_score'],g['residual'],g['score_status']=score(g,m)
        missing=g[FIELDS].isna().all(axis=1)
        np.testing.assert_array_equal(g.loc[missing,'research_score'],g.loc[missing,'fixed_offset'])
        predictions.append(g[KEYS+['s1_score','fixed_offset','research_score','residual','score_status']])
        picks.append(old.slots(old.choose(g,'research_score')))
    prediction=pd.concat(predictions,ignore_index=True);prediction.to_csv(out/'ALL_SCORES_BEFORE_METRICS.csv',index=False,encoding='utf-8-sig')
    selections={'S1_BASELINE':old.slots(old.choose(x,'fixed_offset')),ARM:pd.concat(picks,ignore_index=True),
       'PREVIOUS_THEME_A_REFERENCE':read(ROOT/'research/first_board_dynamic_rank_20261030/results/A_DYNAMIC_THEME_PICKS_BEFORE_OUTCOME_JOIN.csv')}
    for name,p in selections.items():p.to_csv(out/f'{name}_PICKS_BEFORE_METRICS.csv',index=False,encoding='utf-8-sig')
    save(out/'SELECTION_COMPLETE_BEFORE_METRICS.json',{'created_at':clock(),'slots':{k:len(v) for k,v in selections.items()}})
    pool=x.merge(labels.drop(columns='saved_target'),on=KEYS,validate='one_to_one');evaluate=old.evaluator();metrics={};details={};monthly=[]
    for name,p in selections.items():
        m,d=evaluate(pool,p,dates);details[name]=d
        d.to_csv(out/f'{name}_OUTCOMES.csv',index=False,encoding='utf-8-sig')
        sub=d.merge(x[KEYS+['one_price']],on=KEYS,validate='one_to_one');sub=sub[sub.one_price.eq(0)]
        m['non_one_price']={'selected':len(sub),'hits':int(sub[TARGET].eq(1).sum()),'known':int(sub[TARGET].notna().sum()),'unknown':int(sub[TARGET].isna().sum())}
        metrics[name]=m
        for month in sorted({d[:7] for d in dates}):
            ds=[d for d in dates if d.startswith(month)];mm,_=evaluate(pool[pool.signal_date.isin(ds)],p[p.signal_date.isin(ds)],ds)
            monthly.append({'arm':name,'month':month,**{k:mm[k] for k in ['hits','known','unknown','reach4_hits','reach4_known']}})
    swaps=[]
    for day in dates:
        b=details['S1_BASELINE'].query('signal_date==@day');c=details[ARM].query('signal_date==@day')
        removed=b[~b.code.isin(c.code)].sort_values('slot');added=c[~c.code.isin(b.code)].sort_values('slot')
        for (_,a),(_,n) in zip(removed.iterrows(),added.iterrows()):
            swaps.append(dict(signal_date=day,displaced_code=a.code,promoted_code=n.code,displaced_promotion=a[TARGET],promoted_promotion=n[TARGET],
                displaced_reach4=a['reach4_LABEL_ONLY'],promoted_reach4=n['reach4_LABEL_ONLY']))
    pd.DataFrame(swaps,columns=['signal_date','displaced_code','promoted_code','displaced_promotion','promoted_promotion','displaced_reach4','promoted_reach4']).to_csv(out/'SWAP_LEDGER.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(monthly).to_csv(out/'MONTHLY_COMPARISON.csv',index=False,encoding='utf-8-sig')
    protection=load(ROOT/'research/first_board_dragon_upgrade_20261029/DELETION_RECEIPT.json')['protected_sha256']
    assert all(sha(ROOT/p)==h for p,h in protection.items())
    report={'created_at':clock(),'metrics':metrics,'swaps':len(swaps),'changed_dates':len({s['signal_date'] for s in swaps}),
       'block_interval':old.bootstrap(details['S1_BASELINE'],details[ARM],dates),
       'fits':sum(m['status']=='FIT' for m in models),'fallbacks':sum(m['status']!='FIT' for m in models),
       'protected_unchanged':len(protection),'protocol_sha256':sha(HERE/'PROTOCOL.json'),'exploratory_only':True,'production_deployed':False,'post_result_tuning':False}
    save(out/'REPORT.json',report)
    print(json.dumps({a:{k:m[k] for k in ['hits','known','unknown','known_precision','reach4_hits']} for a,m in metrics.items()}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run'])
    prepare() if p.parse_args().action=='prepare' else run()
