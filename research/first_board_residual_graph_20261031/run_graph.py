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
from graph_features import FIELDS,KEYS
TARGET='promotion_next_LABEL_ONLY';ARM='S1_PLUS_PEER_COMPETITION'
DATA=ROOT/'research/first_board_top3_40_20261030/results'

def clock():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p,**kw):return pd.read_csv(p,dtype={'code':str},float_precision='round_trip',**kw)
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,o):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)

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
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run'])
    p.parse_args();run()
