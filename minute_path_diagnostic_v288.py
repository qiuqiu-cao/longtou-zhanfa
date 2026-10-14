"""Fixed price-path hypotheses on the complete September cohort; diagnostic only.

Two phases persist feature/ranking specifications before scoring viewed labels.
Five-minute OHLC intervals never establish continuous seals, tick order or fills.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from archive_september_5min_v287 import check_signal, sha
from next_session_forecast_lab_v271 import ROOT, top1
from next_session_expectation_contract_v271 import now, load_registered
from relay_promotion_holdout_v256 import wilson, _paired

ARCHIVE = ROOT/'data/v287/september_five_minute_20261002_v1'
AUDIT = ROOT/'results/v287/september_five_minute_audit_20261002_v1/report.json'
COHORT = ROOT/'data/v279/daily_path_clues_20261002_v1/september_relay_paths_LABEL_ONLY.csv'
DAILY = ROOT/'data/v278/september_unadjusted_20261002_v3/unadjusted_daily.csv'
CONTROL = ROOT/'results/v279/daily_path_paired_20261002_v2/CONTROL_predictions_LABEL_ONLY.csv'
REGISTRY = ROOT/'data/v271/expectation_registry/2026-09-30_2c74dc406eef7dcf.json'
PRICE_FIELDS = ['final_cap_price_bar_run', 'cap_price_bar_share', 'negative_max_pullback']
ARMS = {'CONTROL': None, 'FINAL_CAP_PRICE_RUN': PRICE_FIELDS[0],
        'CAP_PRICE_BAR_SHARE': PRICE_FIELDS[1], 'SHALLOW_PULLBACK': PRICE_FIELDS[2]}
POOLS = ['ALL_RELAY', 'TRUE_NONWORD']


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sources_verified():
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    if audit['status'] != 'RAW_PARSE_SIGNALS_AND_COMPLETE_COHORT_VERIFIED':
        raise ValueError('raw reconstruction audit required')
    for path, wanted in audit['source_hashes'].items():
        if sha(path) != wanted: raise ValueError('prior audited source changed')
    report = json.loads((ARCHIVE/'report.json').read_text(encoding='utf-8'))
    for name, wanted in report['artifact_hashes'].items():
        path = (ARCHIVE/name).resolve()
        if not path.is_relative_to(ARCHIVE.resolve()) or sha(path) != wanted:
            raise ValueError('audited archive artifact changed')
    return audit


def features(bars, daily, code, day, cap):
    check = check_signal(bars, daily, code, day)
    result = dict(date=day, code=code, price_path_complete=False,
        **{key: None for key in PRICE_FIELDS}, first_cap_high_bar_end=None,
        last_below_cap_low_bar_end=None, cap_close_transition_count=None,
        full_day_vwap=None, closing_above_vwap_fraction=None,
        observed_bars=check['observed_bars'], price_reconciled=check['price_reconciled'],
        grid_complete=check['grid_complete'], volume_reconciled=check['volume_reconciled'],
        continuous_seal_verified=False, real_reseal_count=None,
        auction_verified=False, actual_fill_verified=False)
    rows = bars[bars.minute.str[:10].eq(day)].sort_values('minute')
    if not (check['price_reconciled'] and check['grid_complete'] and np.isfinite(cap) and cap > 0):
        return result
    if abs(rows.close.iloc[-1]-cap) > .005001 or abs(rows.high.max()-cap) > .005001:
        return result
    at_cap = (rows[['open','high','low','close']]-cap).abs().le(.005001).all(axis=1)
    run = 0
    for value in at_cap.iloc[::-1]:
        if not value: break
        run += 1
    touched = rows.high.ge(cap-.005001)
    below = rows.low.lt(cap-.005001)
    close_cap = rows.close.ge(cap-.005001)
    result.update(price_path_complete=True, final_cap_price_bar_run=run,
        cap_price_bar_share=float(at_cap.mean()), negative_max_pullback=float(rows.low.min()/cap-1),
        first_cap_high_bar_end=rows.loc[touched,'minute'].iloc[0] if touched.any() else None,
        last_below_cap_low_bar_end=rows.loc[below,'minute'].iloc[-1] if below.any() else None,
        cap_close_transition_count=int((close_cap & ~close_cap.shift(1,fill_value=True)).sum()))
    if check['volume_reconciled'] and check.get('amount_volume_unit_consistent'):
        vwap = check['vwap_observed_bars']
        result.update(full_day_vwap=vwap, closing_above_vwap_fraction=float(cap/vwap-1))
    return result


def pick(frame, pool, arm):
    """No target, next-day fields or label-valid mask enters ranking."""
    eligible = frame if pool == 'ALL_RELAY' else frame[frame.daily_true_one_word.eq(0)]
    rows = []
    for day, group in eligible.groupby('date',sort=True):
        field = ARMS[arm]
        usable = field is not None and bool(group.price_path_complete.all() and group[field].notna().all())
        if usable:
            selected = group.sort_values([field,'probability','boards','code'],
                ascending=[False,False,False,True],kind='mergesort').iloc[0]
        else:
            selected = top1(group).iloc[0]
        rows.append(dict(date=day,pool=pool,arm=arm,code=selected.code,
            path_rule_used=usable, fallback_reason='NONE' if usable or field is None else 'ENTIRE_POOL_PATH_INCOMPLETE'))
    return pd.DataFrame(rows)


def prepare(output):
    target = Path(output)
    if target.exists(): raise FileExistsError('immutable specification exists')
    audit = sources_verified()
    # These column allowlists deliberately exclude all future labels.
    cohort = pd.read_csv(COHORT,dtype={'code':str},usecols=['date','code','boards','price','daily_true_one_word'])
    baseline = pd.read_csv(CONTROL,dtype={'code':str},usecols=['date','code','probability','fold'])
    baseline = baseline[baseline.fold.eq('SEPTEMBER')].drop(columns='fold')
    if cohort.duplicated(['date','code']).any() or baseline.duplicated(['date','code']).any():
        raise ValueError('duplicate candidate keys')
    keys = ['date','code']
    if set(map(tuple,cohort[keys].to_numpy())) != set(map(tuple,baseline[keys].to_numpy())):
        raise ValueError('control must retain the exact full candidate cohort')
    bars = pd.read_csv(ARCHIVE/'five_minute_bars_UNRECONCILED.csv',dtype={'code':str})
    daily = pd.read_csv(DAILY)
    groups = {code:rows for code,rows in bars.groupby('code')}
    paths = pd.DataFrame([features(groups[row.code],daily,row.code,row.date,row.price)
        for row in cohort.itertuples()])
    f = cohort.merge(baseline,on=keys,validate='one_to_one').merge(paths,on=keys,validate='one_to_one')
    if len(f) != audit['candidate_rows']: raise ValueError('candidate denominator changed')
    target.mkdir(parents=True,exist_ok=False)
    feature_path = target/'features_NO_NEXT_SESSION_LABELS.csv'
    f.to_csv(feature_path,index=False,encoding='utf-8-sig')
    picks = pd.concat([pick(f,pool,arm) for pool in POOLS for arm in ARMS],ignore_index=True)
    pick_path = target/'ranked_choices_BEFORE_LABEL_JOIN.csv'
    picks.to_csv(pick_path,index=False,encoding='utf-8-sig')
    record = dict(version='v288-fixed-five-minute-path-hypotheses-1',recorded_at=now().isoformat(),
        status='REUSED_PERIOD_SPECIFICATION_NOT_BLIND_PREREGISTRATION',arms=ARMS,pools=POOLS,
        candidate_rows=len(f),candidate_dates=int(f.date.nunique()),price_path_complete_rows=int(f.price_path_complete.sum()),
        full_day_vwap_rows=int(f.full_day_vwap.notna().sum()),
        specification=dict(reference='original reported closing limit-up candidate price; no tick certification',
            final_cap_price_bar_run='number of consecutive final five-minute bars whose entire OHLC equals cap; no elapsed-clock duration claim',
            cap_price_bar_share='fraction of complete grid bars whose entire OHLC equals cap',
            negative_max_pullback='signal-day minimum low / closing cap - 1; higher is shallower',
            fallback='if any eligible candidate lacks a complete price path, the entire date/pool uses control',
            tie_break='existing control probability desc, boards desc, code asc',
            score_target='exact next-session closing promotion of ranked Top1; no buy/sell accuracy',
            feature_rule_uses_labels=False,rank_rule_uses_labels=False,
            evaluation_requires_full_original_date_labels=True,select_winning_arm=False,
            confidence_conversion=False,original_forecast_rewrite=False),
        goal_verified=False,trade_authorized=False,
        frozen_registration_hash=load_registered(REGISTRY)['sha256'],frozen_registry_byte_hash=sha(REGISTRY),
        source_hashes={str(Path(p).resolve()):sha(p) for p in [COHORT,CONTROL,DAILY,AUDIT,ARCHIVE/'report.json',
            ARCHIVE/'five_minute_bars_UNRECONCILED.csv',Path(__file__),ROOT/'archive_september_5min_v287.py',
            ROOT/'next_session_forecast_lab_v271.py',ROOT/'relay_promotion_holdout_v256.py',REGISTRY]},
        artifact_hashes={p.name:sha(p) for p in [feature_path,pick_path]})
    dump(target/'specification.json',record)
    return {key:record[key] for key in ['status','candidate_rows','candidate_dates','price_path_complete_rows','full_day_vwap_rows']}


def evaluate_picks(picks, labels):
    if labels.duplicated(['date','code']).any(): raise ValueError('duplicate labels')
    complete = labels.groupby('date').apply(lambda g:bool(g.label_valid.all() and g.target.notna().all()),include_groups=False)
    result = picks.merge(labels,on=['date','code'],how='left',validate='many_to_one')
    result['whole_original_date_labels_complete'] = result.date.map(complete).fillna(False)
    result['evaluated'] = result.whole_original_date_labels_complete & result.target.notna()
    return result


def evaluate(directory):
    directory = Path(directory)
    report_path = directory/'report.json'
    if report_path.exists(): raise FileExistsError('immutable diagnostic evaluation exists')
    record = json.loads((directory/'specification.json').read_text(encoding='utf-8'))
    for path,wanted in record['source_hashes'].items():
        if sha(path) != wanted: raise ValueError('frozen source changed')
    for name,wanted in record['artifact_hashes'].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != wanted: raise ValueError('frozen feature/ranking changed')
    f = pd.read_csv(directory/'features_NO_NEXT_SESSION_LABELS.csv',dtype={'code':str})
    original = pd.read_csv(COHORT,dtype={'code':str},usecols=['date','code','label_valid','target'])
    picks = pd.read_csv(directory/'ranked_choices_BEFORE_LABEL_JOIN.csv',dtype={'code':str})
    rebuilt = pd.concat([pick(f,pool,arm) for pool in POOLS for arm in ARMS],ignore_index=True)
    pd.testing.assert_frame_equal(picks,rebuilt,check_dtype=False)
    scored = evaluate_picks(picks,original)
    scored.to_csv(directory/'scored_choices_LABEL_ONLY.csv',index=False,encoding='utf-8-sig')
    summaries,paired = {},{}
    for pool in POOLS:
        summaries[pool],paired[pool] = {},{}
        valid = scored[(scored.pool==pool)&scored.evaluated].copy()
        baseline = valid[valid.arm.eq('CONTROL')][['date','target']].rename(columns={'target':'promoted_next_day'})
        for arm in ARMS:
            part = valid[valid.arm.eq(arm)]
            n,k = len(part),int(part.target.sum())
            split = (n+1)//2
            chronological = part.sort_values('date')
            summaries[pool][arm] = dict(dates=n,successes=k,rate=k/n if n else None,wilson95=wilson(k,n),
                path_rule_dates=int(part.path_rule_used.sum()),fallback_dates=int((~part.path_rule_used).sum()) if arm!='CONTROL' else 0,
                first_chronological_part=dict(n=split,k=int(chronological.iloc[:split].target.sum())),
                remaining_chronological_part=dict(n=n-split,k=int(chronological.iloc[split:].target.sum())))
            candidate = part[['date','target']].rename(columns={'target':'promoted_next_day'})
            if set(candidate.date) != set(baseline.date): raise ValueError('paired date denominator mismatch')
            if arm!='CONTROL': paired[pool][arm] = _paired(candidate,baseline,seed=288)
    complete = original.groupby('date').apply(lambda g:bool(g.label_valid.all() and g.target.notna().all()),include_groups=False)
    result = dict(version=record['version'],status='FIXED_PRICE_PATH_DIAGNOSTIC_NO_MODEL_UPGRADE',
        recorded_at=now().isoformat(),candidate_rows=len(original),candidate_dates=int(original.date.nunique()),
        complete_label_dates=int(complete.sum()),unresolved_dates=complete[~complete].index.tolist(),
        summaries=summaries,paired=paired,selected_arm=None,default_model_upgraded=False,
        goal_verified=False,trade_authorized=False,original_forecast_rewritten=False,
        limitations=['All September outcomes previously inspected; no pristine blind or forward score',
            'Three exploratory rules reported together; highest observed rate must not be selected as validated',
            'Five-minute OHLC cannot establish continuous seals, actual reseals, auction strength, theme or queue fills',
            'Only 13 original fully labeled dates; daily dependence and multiple hypotheses limit confidence',
            'Volume mismatch retained; no filtering to volume-verified prefix-biased subset',
            'Ranked Top1 promotion is different from all-candidate binary accuracy and actual leader identity',
            'Original reported provider cohort lacks an exchange-wide completeness certificate'],
        source_hashes={str((directory/'specification.json').resolve()):sha(directory/'specification.json')},
        artifact_hashes={'scored_choices_LABEL_ONLY.csv':sha(directory/'scored_choices_LABEL_ONLY.csv')})
    dump(report_path,result)
    return {key:result[key] for key in ['status','complete_label_dates','summaries','paired']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','evaluate']);parser.add_argument('--output',required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output) if args.phase=='prepare' else evaluate(args.output),ensure_ascii=True))
