"""Rebuild path features, complete-pool choices and results from frozen inputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import minute_path_diagnostic_v288 as engine


def run(directory,output):
    directory,target = Path(directory),Path(output)
    if target.exists(): raise FileExistsError('immutable audit exists')
    spec = json.loads((directory/'specification.json').read_text(encoding='utf-8'))
    report = json.loads((directory/'report.json').read_text(encoding='utf-8'))
    engine.sources_verified()
    for record in [spec,report]:
        for path,wanted in record['source_hashes'].items():
            if engine.sha(path)!=wanted: raise ValueError('source hash mismatch')
        for name,wanted in record['artifact_hashes'].items():
            path = (directory/name).resolve()
            if not path.is_relative_to(directory.resolve()) or engine.sha(path)!=wanted:
                raise ValueError('artifact hash mismatch')
    if spec['arms']!=engine.ARMS or spec['pools']!=engine.POOLS:
        raise ValueError('fixed arms or pools changed')
    keys = ['date','code']
    cohort = pd.read_csv(engine.COHORT,dtype={'code':str})
    baseline = pd.read_csv(engine.CONTROL,dtype={'code':str},usecols=['date','code','probability','fold'])
    baseline = baseline[baseline.fold.eq('SEPTEMBER')].drop(columns='fold')
    bars = pd.read_csv(engine.ARCHIVE/'five_minute_bars_UNRECONCILED.csv',dtype={'code':str})
    daily = pd.read_csv(engine.DAILY)
    groups = {code:rows for code,rows in bars.groupby('code')}
    paths = pd.DataFrame([engine.features(groups[r.code],daily,r.code,r.date,r.price) for r in cohort.itertuples()])
    rebuilt = cohort[['date','code','boards','price','daily_true_one_word']].merge(baseline,on=keys,validate='one_to_one') \
        .merge(paths,on=keys,validate='one_to_one')
    saved = pd.read_csv(directory/'features_NO_NEXT_SESSION_LABELS.csv',dtype={'code':str})
    # Round-trip strings/None through the same CSV representation; price numbers
    # are checked numerically, not by only trusting the output hash.
    from io import StringIO
    canonical = pd.read_csv(StringIO(rebuilt.to_csv(index=False)),dtype={'code':str})
    pd.testing.assert_frame_equal(saved,canonical,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    picks = pd.concat([engine.pick(rebuilt,pool,arm) for pool in engine.POOLS for arm in engine.ARMS],ignore_index=True)
    stored_picks = pd.read_csv(directory/'ranked_choices_BEFORE_LABEL_JOIN.csv',dtype={'code':str})
    pd.testing.assert_frame_equal(stored_picks,picks,check_dtype=False)
    scored = engine.evaluate_picks(picks,cohort[['date','code','label_valid','target']])
    stored_scored = pd.read_csv(directory/'scored_choices_LABEL_ONLY.csv',dtype={'code':str})
    if set(stored_scored.columns)!=set(scored.columns): raise ValueError('scored column scope changed')
    pd.testing.assert_frame_equal(stored_scored,scored[stored_scored.columns],check_dtype=False)
    for pool in engine.POOLS:
        baseline_scored = scored[(scored.pool==pool)&scored.arm.eq('CONTROL')&scored.evaluated]
        baseline_pair = baseline_scored[['date','target']].rename(columns={'target':'promoted_next_day'})
        for arm in engine.ARMS:
            part = scored[(scored.pool==pool)&scored.arm.eq(arm)&scored.evaluated].sort_values('date')
            n,k = len(part),int(part.target.sum());split = (n+1)//2
            expected = dict(dates=n,successes=k,rate=k/n if n else None,wilson95=engine.wilson(k,n),
                path_rule_dates=int(part.path_rule_used.sum()),fallback_dates=int((~part.path_rule_used).sum()) if arm!='CONTROL' else 0,
                first_chronological_part=dict(n=split,k=int(part.iloc[:split].target.sum())),
                remaining_chronological_part=dict(n=n-split,k=int(part.iloc[split:].target.sum())))
            if report['summaries'][pool][arm]!=expected: raise ValueError('summary does not match complete scored choices')
            if arm!='CONTROL':
                actual = engine._paired(part[['date','target']].rename(columns={'target':'promoted_next_day'}),baseline_pair,seed=288)
                if report['paired'][pool][arm]!=actual: raise ValueError('paired result differs')
    complete = cohort.groupby('date').apply(lambda g:bool(g.label_valid.all() and g.target.notna().all()),include_groups=False)
    if (report['candidate_rows']!=len(cohort) or report['candidate_dates']!=cohort.date.nunique()
            or report['complete_label_dates']!=int(complete.sum()) or report['unresolved_dates']!=complete[~complete].index.tolist()):
        raise ValueError('original date denominator changed')
    for key in ['default_model_upgraded','goal_verified','trade_authorized','original_forecast_rewritten']:
        if report[key] is not False: raise ValueError('exploratory diagnostic cannot authorize model or trades')
    if report['selected_arm'] is not None: raise ValueError('no post-hoc winning arm selection')
    if spec['frozen_registry_byte_hash']!=engine.sha(engine.REGISTRY): raise ValueError('original forecast changed')
    result = dict(status='FEATURES_RANKS_LABELS_AND_PAIRED_RESULTS_RECONSTRUCTED',recorded_at=engine.now().isoformat(),
        candidate_rows=len(cohort),feature_rows=len(rebuilt),choice_rows=len(picks),scored_choice_rows=int(scored.evaluated.sum()),
        complete_label_dates=int(complete.sum()),original_forecast_unchanged=True,
        goal_verified=False,trade_authorized=False,
        source_hashes={str(Path(p).resolve()):engine.sha(p) for p in [directory/'specification.json',directory/'report.json',
            directory/'features_NO_NEXT_SESSION_LABELS.csv',directory/'ranked_choices_BEFORE_LABEL_JOIN.csv',
            directory/'scored_choices_LABEL_ONLY.csv',Path(__file__),Path(engine.__file__)]})
    target.mkdir(parents=True,exist_ok=False);engine.dump(target/'report.json',result)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(args.archive,args.output),ensure_ascii=True))
