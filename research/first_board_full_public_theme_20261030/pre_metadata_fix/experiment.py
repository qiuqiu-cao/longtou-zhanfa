"""All candidate codes -> source-bound subset features -> two frozen replays."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.special import logit

from rank_offset import fit,score,choose,TARGET,KEYS
from subset_features import build,FIELDS
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PREP=HERE/'prepared_v2'
SOURCES=HERE/'sources_v2'
OLD=ROOT/'research/first_board_dynamic_rank_20261030'
sys.path.insert(0,str(OLD))
spec=importlib.util.spec_from_file_location('old_experiment_utilities',OLD/'run_experiment.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
DATA=old.DATA;S1=old.S1;STATES=old.STATES
read,save_json,sha,slots=old.read,old.save_json,old.sha,old.slots
ARMS={'T_S1_PLUS_THS':'s1_offset','KT_OLD_A_PLUS_THS':'old_a_offset'}


def clock():return datetime.now(timezone.utc).isoformat()


def reconstruct_offsets(x,old_predictions):
    a=old_predictions[old_predictions.arm.eq('A_DYNAMIC_THEME')][KEYS+['residual','research_score','score_status']]
    if a.duplicated(KEYS).any():raise ValueError('duplicate oldA scores')
    out=x.merge(a.rename(columns={'residual':'old_a_residual','research_score':'old_a_saved_score'}),on=KEYS,how='left',validate='one_to_one')
    if not (np.isfinite(out.s1_score).all() and out.s1_score.gt(0).all() and out.s1_score.lt(1).all()
            and np.isfinite(out.old_a_residual).all()):raise ValueError('invalid base data')
    out['s1_offset']=logit(out.s1_score.to_numpy(float))
    out['old_a_offset']=out.s1_offset+out.old_a_residual
    return out


def prepare():
    out=PREP;out.mkdir(exist_ok=False)
    x=read(DATA/'FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv')
    if len(x)!=7254 or x.signal_date.nunique()!=286:raise ValueError('changed pool')
    pred=read(OLD/'results/ALL_CANDIDATE_SCORES_NO_LABELS.csv')
    panel=reconstruct_offsets(x,pred)
    old_features=read(OLD/'prepared_v2/FULL_POOL_THEME_FEATURES_NO_LABELS.csv')
    original=x.merge(old_features[KEYS+old.FIELDS],on=KEYS,validate='one_to_one')
    replay=[]
    for month,g in original.groupby(original.signal_date.str[:7]):
        model=json.loads((OLD/f'results/A_DYNAMIC_THEME_{month}_MODEL.json').read_text(encoding='utf-8'))
        saved=panel[panel.signal_date.str.startswith(month)].set_index(KEYS).loc[g.set_index(KEYS).index]
        original_score,delta,_=old.score(g,model)
        if not np.allclose(delta,saved.old_a_residual.to_numpy(float),rtol=0,atol=1e-14):raise ValueError('oldA replay mismatch')
        if model['status']=='FIT' and (model['cutoff']>=g.signal_date.min() or model['max_training_label_end']>model['cutoff']):raise ValueError('oldA timing')
        replay.append({'month':month,'status':model['status'],'rows':len(g),
                       'max_residual_error':float(np.max(np.abs(delta-saved.old_a_residual.to_numpy(float))))})
    saved_picks=read(OLD/'results/A_DYNAMIC_THEME_PICKS_BEFORE_OUTCOME_JOIN.csv')
    pd.testing.assert_frame_equal(slots(choose(panel,'old_a_offset')),saved_picks,check_exact=True)
    save_json(out/'OLD_A_OFFSET_REPLAY.json',{'rows':len(panel),'months':replay,'old_picks_exactly_reproduced':True})
    state=joblib.load(STATES)
    provider=read(SOURCES/'THS_PROVIDER_NO_LABELS.csv');provider['themes']=provider.themes.map(json.loads)
    provider=provider[provider.signal_date.isin(state['sessions'])]
    features=build(x[KEYS],provider,state['states'],state['sessions'])
    features.to_csv(out/'FULL_POOL_THS_FEATURES_NO_LABELS.csv',index=False,encoding='utf-8-sig')
    panel=panel.merge(features[KEYS+FIELDS],on=KEYS,validate='one_to_one')
    panel.to_csv(out/'FULL_POOL_BASES_AND_FEATURES_NO_LABELS.csv',index=False,encoding='utf-8-sig')
    save_json(out/'SESSIONS.json',state['sessions'])
    summary={'candidate_rows':len(panel),'dates':panel.signal_date.nunique(),
        'context_rows':int(features.theme_count.gt(0).sum()),
        'rows_with_any_numeric_input':int(features[FIELDS].notna().any(axis=1).sum()),
        'numeric_coverage':{f:int(features[f].notna().sum()) for f in FIELDS},
        'days_with_any_numeric_input':int(features.loc[features[FIELDS].notna().any(axis=1),'signal_date'].nunique()),
        'no_labels_loaded':True,'fits':0,'source_scope':'RETROSPECTIVE_THS_OBSERVED_SUBSET_ONLY'}
    save_json(out/'PREPARATION.json',summary)
    bindings={}
    paths=list(HERE.glob('*.py'))+[HERE/'COLLECTION_PLAN.json',HERE/'ALL_SOURCE_INVENTORY.json',HERE/'ALL_EVENTS_PENDING_DATE_CHECK.json',
        SOURCES/'THS_DATED_EVENTS.json',SOURCES/'THS_PROVIDER_NO_LABELS.csv',SOURCES/'SOURCE_COVERAGE_REPORT.json',HERE/'PROTOCOL_DEFINITIONS.md',
        ROOT/'tests/test_full_public_theme.py',DATA/'FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv',DATA/'OUTCOMES_DIAGNOSTIC_ONLY.csv',
        S1/'S1_ALL_PREDICTIONS.csv',STATES,OLD/'run_experiment.py',OLD/'pairwise_residual.py',OLD/'dynamic_features.py',
        OLD/'theme_sources.py',OLD/'results/ALL_CANDIDATE_SCORES_NO_LABELS.csv',OLD/'results/A_DYNAMIC_THEME_PICKS_BEFORE_OUTCOME_JOIN.csv',
        ROOT/'research/first_board_public_theme_sources_20261030/public_theme_sources.py']
    paths+=list(out.glob('*'))+list((HERE/'raw').glob('*.json'))+list((HERE/'link_checks').glob('*.json'))
    paths+=list((OLD/'results').glob('A_DYNAMIC_THEME_*_MODEL.json'))
    prior=json.loads((OLD/'PROTOCOL.json').read_text(encoding='utf-8'))
    bindings.update(prior['bindings'])
    for source in json.loads((HERE/'ALL_SOURCE_INVENTORY.json').read_text(encoding='utf-8')):paths.append(ROOT/source['path'])
    for event in json.loads((SOURCES/'THS_DATED_EVENTS.json').read_text(encoding='utf-8')):
        if event.get('check_path'):paths.append(ROOT/event['check_path'])
    for p in paths:
        if p.is_file():bindings[str(p.relative_to(ROOT))]=sha(p)
    save_json(HERE/'PROTOCOL_V2.json',{'created_at':clock(),'status':'AWAITING_TWO_REAL_ACKS_NO_MARKET_FITS',
        'arms':ARMS,'fields':FIELDS,'hyperparameters':'same original fixed ridge1 monthly252 min60 solver300 ftol1e-12 gtol1e-8',
        'source_scope':'late text, observed subset, NOT PIT or blind','definitions':'PROTOCOL_DEFINITIONS.md','bindings':bindings})
    print(json.dumps(summary,ensure_ascii=False))


def verify():
    p=HERE/'PROTOCOL_V2.json';protocol=json.loads(p.read_text(encoding='utf-8'))
    for who in ['CODEX','TERRA']:
        ack=json.loads((HERE/f'{who}_FREEZE_ACK.json').read_text(encoding='utf-8'))
        if ack['status']!='AGREED_EXPLORATORY_PROTOCOL' or ack['protocol_sha256']!=sha(p):raise ValueError('missing actual agreement')
    for p,h in protocol['bindings'].items():
        if sha(ROOT/p)!=h:raise ValueError('frozen file changed: '+p)
    if protocol['arms']!=ARMS or protocol['fields']!=FIELDS:raise ValueError('spec mismatch')
    return protocol


def run():
    verify();out=HERE/'results';out.mkdir(exist_ok=False)
    save_json(out/'RUN_STARTED_BEFORE_FITS.json',{'created_at':clock(),'protocol_sha256':sha(HERE/'PROTOCOL_V2.json')})
    panel=read(PREP/'FULL_POOL_BASES_AND_FEATURES_NO_LABELS.csv')
    dates=sorted(panel.signal_date.unique());sessions=json.loads((PREP/'SESSIONS.json').read_text())
    labels=read(DATA/'OUTCOMES_DIAGNOSTIC_ONLY.csv')
    maturity=read(S1/'S1_ALL_PREDICTIONS.csv',usecols=KEYS+['label_end',TARGET])
    labels=labels.merge(maturity.rename(columns={TARGET:'saved_S1_target'}),on=KEYS,validate='one_to_one')
    if not (labels[TARGET].eq(labels.saved_S1_target)|(labels[TARGET].isna()&labels.saved_S1_target.isna())).all():raise ValueError('label mismatch')
    picks={'S1_BASELINE':slots(choose(panel,'s1_offset')),'OLD_A_BASELINE':slots(choose(panel,'old_a_offset'))}
    models=[];predictions=[];null_checks=[]
    for arm,base in ARMS.items():
        x=panel.copy();x['fixed_offset']=x[base]
        arm_picks=[]
        for month,future in x.groupby(x.signal_date.str[:7]):
            train,y,cutoff,ds=old.maturity_window(x,labels,sessions,future.signal_date.min())
            model=({'status':'FALLBACK','reason':'FEWER_THAN_60_MATURE_SIGNAL_DAYS','fields':FIELDS} if len(ds)<60 else fit(train,y,FIELDS))
            model.update(arm=arm,month=month,cutoff=cutoff,training_dates=ds,training_candidates=len(train),
                mature_known_training_labels=len(y),max_training_label_end=y.label_end.max() if len(y) else None,created_at=clock())
            save_json(out/f'{arm}_{month}_MODEL.json',model);models.append(model)
            train[KEYS].merge(y[KEYS+[TARGET,'label_end']],on=KEYS,how='left',validate='one_to_one').to_csv(out/f'{arm}_{month}_TRAIN_MEMBERS.csv',index=False,encoding='utf-8-sig')
            for day,g in future.groupby('signal_date'):
                g=g.copy();g['research_score'],g['ths_residual'],status=score(g,model);g['arm']=arm;g['score_status']=status
                predictions.append(g[KEYS+['arm','s1_score','old_a_residual','fixed_offset','research_score','ths_residual','score_status']])
                selected=slots(choose(g,'research_score'));arm_picks.append(selected)
                if not g[FIELDS].notna().any().any():
                    pd.testing.assert_frame_equal(selected,slots(choose(g,'fixed_offset')),check_exact=True)
                    null_checks.append({'arm':arm,'signal_date':day,'exact_base_fallback':True})
        picks[arm]=pd.concat(arm_picks,ignore_index=True)
    pd.concat(predictions,ignore_index=True).to_csv(out/'ALL_CANDIDATE_SCORES_NO_LABELS.csv',index=False,encoding='utf-8-sig')
    for arm,p in picks.items():p.to_csv(out/f'{arm}_PICKS_BEFORE_OUTCOME_JOIN.csv',index=False,encoding='utf-8-sig')
    save_json(out/'SELECTIONS_COMPLETE_BEFORE_EVALUATION.json',{'created_at':clock(),'rows_per_arm':{a:len(p) for a,p in picks.items()},'missing_day_fallback_checks':null_checks})
    pool=panel.merge(labels.drop(columns='saved_S1_target'),on=KEYS,validate='one_to_one')
    evaluate=old.evaluator();metrics={};details={};monthly=[]
    for arm,p in picks.items():
        metric,d=evaluate(pool,p,dates);details[arm]=d
        d.to_csv(out/f'{arm}_SELECTIONS_WITH_OUTCOMES.csv',index=False,encoding='utf-8-sig')
        nonone=d.merge(pool[KEYS+['one_price']],on=KEYS,validate='one_to_one').query('one_price == 0')
        metric['non_one_price']={'selected':len(nonone),'hits':int(nonone[TARGET].eq(1).sum()),'known':int(nonone[TARGET].notna().sum()),'unknown':int(nonone[TARGET].isna().sum())}
        metrics[arm]=metric
        for month in sorted({d[:7] for d in dates}):
            ds=[d for d in dates if d.startswith(month)]
            m,_=evaluate(pool[pool.signal_date.isin(ds)],p[p.signal_date.isin(ds)],ds)
            monthly.append({'arm':arm,'month':month,**{k:m[k] for k in ['dates','slots','hits','known','unknown','known_precision','reach4_hits','reach4_known']}})
    comparison={};ledgers=[]
    for arm,base in [('T_S1_PLUS_THS','S1_BASELINE'),('KT_OLD_A_PLUS_THS','S1_BASELINE'),('KT_OLD_A_PLUS_THS','OLD_A_BASELINE')]:
        rows=[]
        for day in dates:
            b=details[base].query('signal_date == @day');c=details[arm].query('signal_date == @day')
            removed=b[~b.code.isin(c.code)].sort_values('slot');added=c[~c.code.isin(b.code)].sort_values('slot')
            for (_,o),(_,n) in zip(removed.iterrows(),added.iterrows()):
                rows.append({'arm':arm,'baseline':base,'signal_date':day,'displaced_code':o.code,'promoted_code':n.code,
                    'displaced_promotion':o[TARGET],'promoted_promotion':n[TARGET],
                    'displaced_reach4':o['reach4_LABEL_ONLY'],'promoted_reach4':n['reach4_LABEL_ONLY']})
        ledger=pd.DataFrame(rows);ledgers+=rows
        counts={}
        for side in ['promoted','displaced']:
            v=ledger[side+'_promotion'] if len(ledger) else pd.Series(dtype=float)
            counts[side]={'hits':int(v.eq(1).sum()),'failures':int(v.eq(0).sum()),'unknown':int(v.isna().sum())}
        comparison[arm+'_vs_'+base]={'swaps':len(rows),'changed_dates':len(set(r['signal_date'] for r in rows)),**counts,
            'block_interval':old.bootstrap(details[base],details[arm],dates)}
    pd.DataFrame(monthly).to_csv(out/'MONTHLY_COMPARISON.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(ledgers).to_csv(out/'SWAP_LEDGER.csv',index=False,encoding='utf-8-sig')
    protected=json.loads((ROOT/'research/first_board_dragon_upgrade_20261029/DELETION_RECEIPT.json').read_text(encoding='utf-8'))['protected_sha256']
    if any(sha(ROOT/p)!=h for p,h in protected.items()):raise ValueError('protected files changed')
    report={'created_at':clock(),'metrics':metrics,'comparisons':comparison,'fits':sum(m['status']=='FIT' for m in models),
        'fallback_models':sum(m['status']!='FIT' for m in models),'protected_unchanged':len(protected),
        'actual_historical_source_rows':0,'strict_channel_result':'S1_BASELINE; no retrospective trained coefficients',
        'forty_percent_achieved':any(metrics[a]['known_precision']>=.4 for a in ARMS),
        'exploratory_only':True,'unseen_holdout':False,'production_deployed':False,'protocol_sha256':sha(HERE/'PROTOCOL_V2.json')}
    save_json(out/'REPORT.json',report)
    print(json.dumps({a:{k:m[k] for k in ['hits','known','unknown','known_precision','reach4_hits']} for a,m in metrics.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run'])
    prepare() if p.parse_args().action=='prepare' else run()
