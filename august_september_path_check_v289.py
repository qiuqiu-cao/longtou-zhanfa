"""Extend V288's unchanged exploratory rules to August; never select a winner."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import minute_path_diagnostic_v288 as base
from next_session_forecast_lab_v271 import ROOT

AUGUST=ROOT/'data/v289/full_august_five_minute_20261002_v1'
AUGUST_AUDIT=ROOT/'results/v289/august_minute_audit_20261002_v1/report.json'
PREVIOUS_SPEC=ROOT/'results/v288/minute_path_diagnostic_20261002_v1/specification.json'
PERIODS={
    'AUGUST':dict(cohort=AUGUST/'august_cohort_LABEL_ONLY.csv',
        bars=AUGUST/'five_minute_bars_UNRECONCILED.csv',daily=AUGUST/'unadjusted_daily_reference.csv',labels=base.CONTROL),
    'SEPTEMBER':dict(cohort=base.COHORT,bars=base.ARCHIVE/'five_minute_bars_UNRECONCILED.csv',daily=base.DAILY,labels=base.COHORT)}


def verified_sources():
    base.sources_verified()
    record=json.loads(AUGUST_AUDIT.read_text(encoding='utf-8'))
    if record['status']!='AUGUST_RAW_COHORT_AND_SIGNAL_CHECKS_RECONSTRUCTED' or record['original_one_word_reference_disagreements']:
        raise ValueError('August independent reconstruction and true-word agreement required')
    for path,wanted in record['source_hashes'].items():
        if base.sha(path)!=wanted:raise ValueError('audited August source changed')
    original=json.loads(PREVIOUS_SPEC.read_text(encoding='utf-8'))
    if original['arms']!=base.ARMS or original['pools']!=base.POOLS:
        raise ValueError('V288 fixed rules must remain unchanged')


def prepare(output):
    target=Path(output)
    if target.exists():raise FileExistsError('immutable paired-period preparation exists')
    verified_sources()
    baseline=pd.read_csv(base.CONTROL,dtype={'code':str},usecols=['date','code','probability'])
    frames=[]
    for period,paths in PERIODS.items():
        f=pd.read_csv(paths['cohort'],dtype={'code':str},usecols=['date','code','boards','price','daily_true_one_word'])
        controls=baseline[baseline.date.isin(f.date.unique())]
        keys=['date','code']
        if set(map(tuple,f[keys].to_numpy()))!=set(map(tuple,controls[keys].to_numpy())):
            raise ValueError('complete control candidate cohort required')
        bars=pd.read_csv(paths['bars'],dtype={'code':str});daily=pd.read_csv(paths['daily'])
        groups={code:rows for code,rows in bars.groupby('code')}
        features=pd.DataFrame([base.features(groups.get(r.code,bars.iloc[:0]),daily,r.code,r.date,r.price)
            for r in f.itertuples()])
        f=f.merge(controls,on=keys,validate='one_to_one').merge(features,on=keys,validate='one_to_one')
        f['period']=period;frames.append(f)
    complete=pd.concat(frames,ignore_index=True)
    if complete.duplicated(['date','code']).any():raise ValueError('periods overlap')
    picks=pd.concat([base.pick(complete,pool,arm) for pool in base.POOLS for arm in base.ARMS],ignore_index=True)
    picks['period']=picks.date.map(complete[['date','period']].drop_duplicates().set_index('date').period)
    target.mkdir(parents=True,exist_ok=False)
    feature_path=target/'features_NO_NEXT_SESSION_LABELS.csv';complete.to_csv(feature_path,index=False,encoding='utf-8-sig')
    picks_path=target/'ranked_choices_BEFORE_LABEL_JOIN.csv';picks.to_csv(picks_path,index=False,encoding='utf-8-sig')
    sources=[PREVIOUS_SPEC,AUGUST_AUDIT,base.AUDIT,base.CONTROL,base.REGISTRY,Path(__file__),Path(base.__file__)]
    sources += [path for paths in PERIODS.values() for path in paths.values()]
    spec=dict(version='v289-unchanged-august-september-path-check-1',recorded_at=base.now().isoformat(),
        status='VIEWED_HISTORY_EXTENSION_NOT_BLIND_PREREGISTRATION',arms=base.ARMS,pools=base.POOLS,
        periods=list(PERIODS),candidate_rows=len(complete),candidate_dates=int(complete.date.nunique()),
        price_path_complete_rows=int(complete.price_path_complete.sum()),
        full_day_vwap_rows=int(complete.full_day_vwap.notna().sum()),
        prior_rule_specification_sha256=base.sha(PREVIOUS_SPEC),
        rule_reselected=False,target_labels_enter_features=False,target_labels_enter_ranking=False,
        original_forecast_rewritten=False,goal_verified=False,trade_authorized=False,
        source_hashes={str(p.resolve()):base.sha(p) for p in sources},
        artifact_hashes={p.name:base.sha(p) for p in [feature_path,picks_path]})
    base.dump(target/'specification.json',spec)
    return {k:spec[k] for k in ['status','candidate_rows','candidate_dates','price_path_complete_rows','full_day_vwap_rows']}


def summary_and_pairs(scored):
    summaries={};paired={}
    for period in [*PERIODS,'COMBINED']:
        frame=scored if period=='COMBINED' else scored[scored.period.eq(period)]
        summaries[period],paired[period]={},{}
        for pool in base.POOLS:
            valid=frame[frame.pool.eq(pool)&frame.evaluated]
            summaries[period][pool],paired[period][pool]={},{}
            control=valid[valid.arm.eq('CONTROL')][['date','code','target']]
            baseline=control[['date','target']].rename(columns={'target':'promoted_next_day'})
            for arm in base.ARMS:
                part=valid[valid.arm.eq(arm)]
                n,k=len(part),int(part.target.sum())
                choices=part[['date','code']].merge(control[['date','code']],on='date',validate='one_to_one',suffixes=('_rule','_control'))
                summaries[period][pool][arm]=dict(dates=n,successes=k,rate=k/n if n else None,
                    wilson95=base.wilson(k,n),path_rule_dates=int(part.path_rule_used.sum()),
                    fallback_dates=int((~part.path_rule_used).sum()) if arm!='CONTROL' else 0,
                    changed_choice_dates=int(choices.code_rule.ne(choices.code_control).sum()))
                candidate=part[['date','target']].rename(columns={'target':'promoted_next_day'})
                if set(candidate.date)!=set(baseline.date):raise ValueError('paired date scope changed')
                if arm!='CONTROL':paired[period][pool][arm]=base._paired(candidate,baseline,seed=289)
    return summaries,paired


def evaluate(directory):
    directory=Path(directory)
    if (directory/'report.json').exists():raise FileExistsError('immutable paired-period result exists')
    spec=json.loads((directory/'specification.json').read_text(encoding='utf-8'))
    for path,wanted in spec['source_hashes'].items():
        if base.sha(path)!=wanted:raise ValueError('frozen source changed')
    for name,wanted in spec['artifact_hashes'].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or base.sha(path)!=wanted:raise ValueError('frozen features/choices changed')
    features=pd.read_csv(directory/'features_NO_NEXT_SESSION_LABELS.csv',dtype={'code':str})
    picks=pd.read_csv(directory/'ranked_choices_BEFORE_LABEL_JOIN.csv',dtype={'code':str})
    rebuilt=pd.concat([base.pick(features,pool,arm) for pool in base.POOLS for arm in base.ARMS],ignore_index=True)
    rebuilt['period']=rebuilt.date.map(features[['date','period']].drop_duplicates().set_index('date').period)
    pd.testing.assert_frame_equal(picks,rebuilt,check_dtype=False)
    originals=[]
    for period,paths in PERIODS.items():
        rows=pd.read_csv(paths['labels'],dtype={'code':str},usecols=['date','code','target','label_valid'])
        keys=features.loc[features.period.eq(period),['date','code']]
        rows=keys.merge(rows,on=['date','code'],how='left',validate='one_to_one')
        rows['period']=period;originals.append(rows)
    labels=pd.concat(originals,ignore_index=True)
    scored=base.evaluate_picks(picks,labels.drop(columns='period'))
    output=directory/'scored_choices_LABEL_ONLY.csv';scored.to_csv(output,index=False,encoding='utf-8-sig')
    summaries,paired=summary_and_pairs(scored)
    complete=labels.groupby('date').apply(lambda g:bool(g.label_valid.all() and g.target.notna().all()),include_groups=False)
    result=dict(version=spec['version'],recorded_at=base.now().isoformat(),status='TWO_PERIOD_FIXED_RULE_DIAGNOSTIC_NO_UPGRADE',
        candidate_rows=len(labels),candidate_dates=int(labels.date.nunique()),complete_label_dates=int(complete.sum()),
        unresolved_dates=complete[~complete].index.tolist(),summaries=summaries,paired=paired,
        selected_arm=None,default_model_upgraded=False,original_forecast_rewritten=False,goal_verified=False,trade_authorized=False,
        limitations=['Both periods previously viewed; August extension is not a pristine blind test',
            'Thirty-minute depth probe is separate and supplies no five-minute observations to this result',
            'Missing paths retain original candidates and force whole eligible date/pool to control',
            'Small dependent date sample and three reported rules do not certify stable sixty percent',
            'Promotion of ranked Top1 is not all-candidate binary, actual leader identity or trade success'],
        source_hashes={str((directory/'specification.json').resolve()):base.sha(directory/'specification.json')},
        artifact_hashes={output.name:base.sha(output)})
    base.dump(directory/'report.json',result)
    return {k:result[k] for k in ['status','candidate_rows','complete_label_dates','summaries','paired']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','evaluate']);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.output) if args.phase=='prepare' else evaluate(args.output),ensure_ascii=True))
