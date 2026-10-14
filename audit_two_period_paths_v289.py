"""Reconstruct both periods' features and all unchanged-rule results."""
import argparse
from io import StringIO
import json
from pathlib import Path
import pandas as pd
import august_september_path_check_v289 as engine


def run(directory,output):
    directory,target=Path(directory),Path(output)
    if target.exists():raise FileExistsError('immutable two-period audit exists')
    engine.verified_sources()
    spec=json.loads((directory/'specification.json').read_text(encoding='utf-8'))
    report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    for record in [spec,report]:
        for path,wanted in record['source_hashes'].items():
            if engine.base.sha(path)!=wanted:raise ValueError('source hash mismatch')
        for name,wanted in record['artifact_hashes'].items():
            path=(directory/name).resolve()
            if not path.is_relative_to(directory.resolve()) or engine.base.sha(path)!=wanted:raise ValueError('artifact hash mismatch')
    if spec['arms']!=engine.base.ARMS or spec['pools']!=engine.base.POOLS or spec['periods']!=list(engine.PERIODS):
        raise ValueError('original rules or periods changed')
    baseline=pd.read_csv(engine.base.CONTROL,dtype={'code':str},usecols=['date','code','probability'])
    frames=[];labels=[]
    for period,paths in engine.PERIODS.items():
        f=pd.read_csv(paths['cohort'],dtype={'code':str},usecols=['date','code','boards','price','daily_true_one_word'])
        bars=pd.read_csv(paths['bars'],dtype={'code':str});daily=pd.read_csv(paths['daily'])
        groups={code:part for code,part in bars.groupby('code')}
        features=pd.DataFrame([engine.base.features(groups.get(r.code,bars.iloc[:0]),daily,r.code,r.date,r.price) for r in f.itertuples()])
        full=f.merge(baseline,on=['date','code'],validate='one_to_one').merge(features,on=['date','code'],validate='one_to_one')
        full['period']=period;frames.append(full)
        original=pd.read_csv(paths['labels'],dtype={'code':str},usecols=['date','code','target','label_valid'])
        labels.append(f[['date','code']].merge(original,on=['date','code'],how='left',validate='one_to_one'))
    frame=pd.concat(frames,ignore_index=True);original=pd.concat(labels,ignore_index=True)
    saved=pd.read_csv(directory/'features_NO_NEXT_SESSION_LABELS.csv',dtype={'code':str})
    canonical=pd.read_csv(StringIO(frame.to_csv(index=False)),dtype={'code':str})
    pd.testing.assert_frame_equal(saved,canonical,check_dtype=False,rtol=1e-12,atol=1e-12)
    picks=pd.concat([engine.base.pick(frame,pool,arm) for pool in engine.base.POOLS for arm in engine.base.ARMS],ignore_index=True)
    picks['period']=picks.date.map(frame[['date','period']].drop_duplicates().set_index('date').period)
    pd.testing.assert_frame_equal(pd.read_csv(directory/'ranked_choices_BEFORE_LABEL_JOIN.csv',dtype={'code':str}),picks,check_dtype=False)
    scored=engine.base.evaluate_picks(picks,original)
    stored=pd.read_csv(directory/'scored_choices_LABEL_ONLY.csv',dtype={'code':str})
    if set(stored.columns)!=set(scored.columns):raise ValueError('scored fields changed')
    pd.testing.assert_frame_equal(stored,scored[stored.columns],check_dtype=False)
    summaries,paired=engine.summary_and_pairs(scored)
    if report['summaries']!=summaries or report['paired']!=paired:raise ValueError('summary or paired evidence differs')
    complete=original.groupby('date').apply(lambda g:bool(g.label_valid.all() and g.target.notna().all()),include_groups=False)
    if report['candidate_rows']!=len(original) or report['complete_label_dates']!=int(complete.sum()) \
            or report['unresolved_dates']!=complete[~complete].index.tolist():raise ValueError('date denominator changed')
    for key in ['default_model_upgraded','original_forecast_rewritten','goal_verified','trade_authorized']:
        if report[key] is not False:raise ValueError('exploratory result cannot authorize upgrades')
    if report['selected_arm'] is not None:raise ValueError('post-hoc winner selected')
    result=dict(status='TWO_PERIOD_FEATURE_CHOICES_AND_RESULTS_RECONSTRUCTED',recorded_at=engine.base.now().isoformat(),
        candidate_rows=len(frame),candidate_dates=int(frame.date.nunique()),choice_rows=len(picks),
        evaluated_choice_rows=int(scored.evaluated.sum()),complete_label_dates=int(complete.sum()),
        goal_verified=False,trade_authorized=False,
        source_hashes={str(Path(p).resolve()):engine.base.sha(p) for p in [directory/'specification.json',directory/'report.json',
            directory/'features_NO_NEXT_SESSION_LABELS.csv',directory/'ranked_choices_BEFORE_LABEL_JOIN.csv',
            directory/'scored_choices_LABEL_ONLY.csv',Path(__file__),Path(engine.__file__),Path(engine.base.__file__)]})
    target.mkdir(parents=True,exist_ok=False);engine.base.dump(target/'report.json',result)
    return {k:result[k] for k in ['status','candidate_rows','candidate_dates','choice_rows','evaluated_choice_rows','complete_label_dates']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(args.archive,args.output),ensure_ascii=True))
