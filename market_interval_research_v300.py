"""Fixed causal market bands on an inspected provider archive; never a live forecast."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from historical_ohlc_archive_v278 import verify_archive

ROOT = Path(__file__).resolve().parent
MARKET = ROOT / 'results/v278/september_falsification_20261002_v6/combined_market_input.csv'
CALENDAR = ROOT / 'data/v279/exchange_calendar_audit_20261002_v1/calendar.json'
OLD_INDEX = ROOT / 'data/v203/broad_a_share_indexes_20260917_v1'
DAILY = ROOT / 'data/v278/september_unadjusted_20261002_v3'
OLD_POOL = ROOT / 'data/v16/point_in_time_rebuild_cutoff_20260831_v2/leader_candidates.csv'
NEW_POOL = ROOT / 'data/v275/reported_seal_archive_20261002_v2/reported_seal_rows.csv'
ORIGINAL = ROOT / 'data/v271/expectation_registry/2026-09-30_2c74dc406eef7dcf.json'
ORIGINAL_SHA = 'bfbaeb295637451e396850ee583fcafa3eb0ab6a99fb1b154e4c46a8c23aa40c'
METRICS = {
    'sh_a_return_pct': ('上证A股指数涨跌幅', 'percentage_points', None, None, False),
    'sz_a_return_pct': ('深证A股指数涨跌幅', 'percentage_points', None, None, False),
    'limit_count': ('涨停家数', 'stocks', 0., None, True),
    'down_limit_count': ('跌停家数', 'stocks', 0., None, True),
    'failed_rate': ('炸板率', 'fraction', 0., 1., False),
    'max_board': ('最高板', 'boards', 0., None, True),
    'relay_promotion_rate': ('昨日2板及以上次日晋级率', 'fraction', 0., 1., False),
    'yesterday_premium': ('昨日涨停池当日收益中位数', 'fraction', None, None, False),
}
SLOTS = [
    ('index_return', ['sh_a_return_pct', 'sz_a_return_pct'], '两个A股指数分别评分；非上证综指，也非主板专属指数'),
    ('turnover_amount', [], '缺同范围、完整且可核验的市场成交额序列'),
    ('limit_up_count', ['limit_count'], '供应商00/60且名称非ST/退范围；非交易所全市场认证'),
    ('limit_down_count', ['down_limit_count'], '同供应商范围；旧段部分负反馈由日线重建'),
    ('failed_rate', ['failed_rate'], '炸板家数/(收盘涨停家数+炸板家数)'),
    ('max_boards', ['max_board'], '供应商收盘涨停池最高连续板位'),
    ('promotion_rate', ['relay_promotion_rate'], '昨日所有2板及以上，今日同代码板位恰好+1；两日池数量须与日表吻合'),
    ('previous_limit_premium', ['yesterday_premium'], '昨日涨停池的收益中位数，非平均收益；反馈覆盖不足不评分'),
    ('high_stock_feedback', [], '缺预先固定的高位反馈状态真值，不用空间板晋级率代替'),
    ('theme_strength', [], '缺当时题材归属及题材状态真值'),
]
PROTOCOL = {
    'version': 'v300-fixed-causal-bands-1', 'status': 'INSPECTED_ARCHIVE_DIAGNOSTIC_ONLY',
    'signal_start': '2026-01-01', 'signal_end': '2026-09-30',
    'arms': ['LEVEL', 'PERSISTENCE_DELTA'], 'lookback_official_sessions': 120,
    'minimum_known_observations': 60, 'quantiles': [.1, .5, .9], 'quantile_method': 'linear',
    'nominal_coverage': .8, 'alpha': .2,
    'window': 'last 120 official end-dates <= signal; delta requires exact preceding official session',
    'level': 'empirical quantiles of known metric levels',
    'persistence_delta': 'current observed level plus empirical quantiles of exact-session increments',
    'bounds': 'clip physical bounds, round count lower down and upper up; median clipped but not rounded',
    'scoring': 'width + 2/alpha * distance outside interval; native units, lower is better',
    'bootstrap': {'draws': 5000, 'seed': 300, 'block_official_sessions': 5,
                  'family_channels': 8, 'tails': [.025/8, 1-.025/8]},
    'periods': {'ALL': ['2026-01-01', '2026-09-30'],
                'JAN_FEB': ['2026-01-01', '2026-02-28'],
                'MAR_SEP': ['2026-03-01', '2026-09-30']},
    'threshold_selection_on_test': False, 'missing_labels_imputed': False,
    'pure_blind_test': False, 'genuine_historical_availability_certified': False,
    'provider_scope_exchange_certified': False, 'probability_calibration_proved': False,
    'forward_predictions': 0, 'core_upgrade_authorized': False, 'stable_60_proved': False,
    'original_oct8_plan_may_be_changed': False,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def validate_sessions(sessions):
    if not sessions or sessions != sorted(set(sessions)):
        raise ValueError('official sessions must be unique and ordered')


def raw_indexes():
    """Rebuild index returns from raw closes on the exact calendar, not cached next_* fields."""
    manifest = json.loads((OLD_INDEX / 'manifest.json').read_text(encoding='utf-8'))
    result, sources = {}, []
    daily = verify_archive(DAILY)
    for symbol in ('sh000002', 'sz399107'):
        raw = OLD_INDEX / (symbol+'_tencent.json'); csv = OLD_INDEX / (symbol+'.csv')
        item = manifest['indexes'][symbol]
        if sha(raw) != item['raw_sha256'] or sha(csv) != item['csv_sha256']:
            raise ValueError('old index raw/csv hash mismatch')
        block = json.loads(raw.read_bytes())['data'][symbol]
        values = block.get('qfqday') or block.get('day')
        if not values:
            raise ValueError('index raw close absent')
        old = pd.DataFrame([r[:6] for r in values], columns=['date','open','close','high','low','volume'])
        old = old[['date', 'close']]; old['close'] = pd.to_numeric(old.close, errors='raise')
        saved = pd.read_csv(csv)[['date','close']]
        pd.testing.assert_frame_equal(old.reset_index(drop=True), saved, check_dtype=False, rtol=1e-12)
        new = daily[daily.symbol.eq(symbol)][['date','close']]
        # Overlap is checked before the older series is replaced at September.
        overlap = old.merge(new, on='date', suffixes=('_old','_new'))
        if overlap.empty or not np.allclose(overlap.close_old, overlap.close_new, rtol=1e-12, atol=1e-10):
            raise ValueError('index source seam conflict')
        combined = pd.concat([old[old.date.lt('2026-09-01')], new[new.date.ge('2026-09-01')]])
        if combined.date.duplicated().any() or not combined.close.gt(0).all():
            raise ValueError('duplicate/nonpositive index close')
        result[symbol] = combined.set_index('date').close
        sources.extend([raw, csv])
    sources.extend([OLD_INDEX / 'manifest.json', DAILY / 'report.json', DAILY / 'unadjusted_daily.csv',
                    ROOT / 'historical_ohlc_archive_v278.py', NEW_POOL])
    sources.extend(sorted((DAILY / 'raw').glob('*.json')))
    return result, sources


def relay_observations(market, pool):
    """No future labels: compute today's rate using just today's and official yesterday's pools."""
    if pool.duplicated(['date', 'code']).any() or not pool.boards.ge(2).all():
        raise ValueError('relay pool keys/boards invalid')
    groups = {d: g for d, g in pool.groupby('date')}
    out = pd.Series(np.nan, index=market.index, dtype=float)
    complete = pd.Series(False, index=market.index)
    for day, row in market.iterrows():
        group = groups.get(day, pool.iloc[:0])
        complete.loc[day] = (row.data_status in ('verified','partial')
                             and np.isfinite(row.consecutive_count)
                             and len(group) == row.consecutive_count)
    for pos, day in enumerate(market.index):
        if pos == 0 or not complete.iloc[pos] or not complete.iloc[pos-1]:
            continue
        previous = groups.get(market.index[pos-1], pool.iloc[:0])
        current = groups.get(day, pool.iloc[:0]).set_index('code').boards
        if len(previous):
            out.loc[day] = float(np.mean([
                current.get(r.code, np.nan) == r.boards+1 for r in previous.itertuples()]))
    return out, complete


def prepare(market, sessions, index_closes, pool):
    validate_sessions(sessions)
    if market.date.duplicated().any() or not set(market.date).issubset(set(sessions)):
        raise ValueError('market duplicate or non-official date')
    frame = market.set_index('date').reindex(sessions).copy()
    usable = frame.data_status.isin(['verified','partial'])
    obs = pd.DataFrame(index=frame.index)
    reasons = pd.DataFrame(index=frame.index)
    for symbol, key in [('sh000002','sh_a_return_pct'),('sz399107','sz_a_return_pct')]:
        closes = index_closes[symbol].reindex(sessions)
        obs[key] = closes / closes.shift(1)*100-100
        reasons[key] = np.where(obs[key].notna(), 'KNOWN_PROVIDER_ARCHIVE', 'EXACT_PREVIOUS_INDEX_CLOSE_MISSING')
    for key in ['limit_count','down_limit_count','failed_rate','max_board','yesterday_premium']:
        values = pd.to_numeric(frame[key], errors='coerce')
        valid = usable & np.isfinite(values)
        lo, hi, integer = METRICS[key][2:]
        if lo is not None: valid &= values.ge(lo)
        if hi is not None: valid &= values.le(hi)
        if integer: valid &= values.eq(values.round())
        reasons[key] = np.where(valid, 'KNOWN_PROVIDER_ARCHIVE', 'METRIC_SOURCE_OR_DOMAIN_MISSING')
        if key == 'failed_rate':
            expected = frame.failed_count / (frame.limit_count+frame.failed_count)
            consistent = np.isfinite(expected) & np.isclose(values, expected, rtol=1e-8, atol=1e-10)
            reasons.loc[valid & ~consistent,key] = 'FAILED_RATE_DENOMINATOR_CONFLICT'
            valid &= consistent
        if key == 'yesterday_premium':
            feedback = frame.previous_limit_feedback_coverage.eq(1)
            predecessor = usable.shift(1, fill_value=False) & frame.limit_count.shift(1).gt(0)
            reasons.loc[valid & ~(feedback & predecessor),key] = 'PREVIOUS_POOL_OR_FULL_FEEDBACK_MISSING'
            valid &= feedback & predecessor
        obs[key] = values.where(valid)
    obs['relay_promotion_rate'], complete = relay_observations(frame, pool)
    reasons['relay_promotion_rate'] = np.where(obs.relay_promotion_rate.notna(),
        'KNOWN_PROVIDER_ARCHIVE_RECOMPUTED', 'EXACT_TWO_DAY_RELAY_POOL_OR_NONZERO_DENOMINATOR_MISSING')
    diagnostics = {'relay_pool_complete_dates': int(complete.sum()),
                   'market_rows_input':len(market), 'official_rows_retained':len(frame),
                   'market_status_counts': {str(k):int(v) for k,v in frame.data_status.value_counts(dropna=False).items()},
                   'legacy_verified_is_not_raw_or_exchange_certification': True}
    return obs[list(METRICS)], reasons[list(METRICS)], diagnostics


def bounded(values, metric):
    lo, hi, integer = METRICS[metric][2:]
    values = np.array(values, dtype=float)
    if lo is not None: values = np.maximum(values, lo)
    if hi is not None: values = np.minimum(values, hi)
    if integer:
        values[0] = np.floor(values[0]); values[2] = np.ceil(values[2])
    if not np.isfinite(values).all() or not values[0] <= values[1] <= values[2]:
        raise ValueError('invalid physical interval')
    return values


def forecasts(obs, reasons, sessions, protocol=PROTOCOL):
    validate_sessions(sessions)
    if obs.index.tolist() != sessions or reasons.index.tolist() != sessions:
        raise ValueError('observation grid must include every official session')
    rows = []
    for pos, day in enumerate(sessions):
        if not protocol['signal_start'] <= day <= protocol['signal_end']:
            continue
        target = sessions[pos+1] if pos+1<len(sessions) else None
        start = max(0,pos+1-protocol['lookback_official_sessions'])
        for key in METRICS:
            history = obs[key].iloc[:pos+1]  # hard causal fence, including current close
            for arm in protocol['arms']:
                training = (history if arm == 'LEVEL' else history.diff()).iloc[start:].dropna()
                current = obs[key].iloc[pos]
                status = ('UNKNOWN_CURRENT_LEVEL' if arm == 'PERSISTENCE_DELTA' and not np.isfinite(current)
                          else 'INSUFFICIENT_HISTORY' if len(training)<protocol['minimum_known_observations']
                          else 'KNOWN_DIAGNOSTIC_INTERVAL')
                pred = [np.nan]*3
                if status == 'KNOWN_DIAGNOSTIC_INTERVAL':
                    pred = np.quantile(training.to_numpy(), protocol['quantiles'], method=protocol['quantile_method'])
                    if arm == 'PERSISTENCE_DELTA': pred += current
                    pred = bounded(pred,key)
                rows.append({'signal_date':day, 'target_date':target, 'metric':key,'arm':arm,
                    'status':status, 'low':pred[0], 'median':pred[1], 'high':pred[2],
                    'training_n':len(training), 'window_start':sessions[start],
                    'max_known_observation_date':training.index.max() if len(training) else None,
                    'training_digest':hashlib.sha256(training.to_csv().encode()).hexdigest(),
                    'signal_metric_status':reasons[key].iloc[pos], 'forward_evidence':False})
    return pd.DataFrame(rows)


def interval_score(y, low, high, alpha=.2):
    y, low, high = np.asarray(y), np.asarray(low), np.asarray(high)
    if alpha<=0 or alpha>=1 or np.any(low>high):
        raise ValueError('invalid interval/alpha')
    return high-low + 2/alpha * (np.maximum(low-y,0)+np.maximum(y-high,0))


def score(pred, obs, reasons, alpha=.2):
    result = pred.copy()
    keys = list(zip(result.target_date, result.metric))
    result['actual'] = [obs.at[d,k] if d in obs.index else np.nan for d,k in keys]
    result['target_metric_status'] = [reasons.at[d,k] if d in reasons.index else 'TARGET_NOT_OBSERVED' for d,k in keys]
    result['evaluated'] = result.status.eq('KNOWN_DIAGNOSTIC_INTERVAL') & result.actual.notna()
    result['coverage_hit'] = np.nan; result['interval_score'] = np.nan
    result['median_abs_error'] = np.nan; result['width'] = result.high-result.low
    known = result.evaluated
    result.loc[known,'coverage_hit'] = result.loc[known,'actual'].between(result.loc[known,'low'],result.loc[known,'high']).astype(float)
    result.loc[known,'interval_score'] = interval_score(result.loc[known,'actual'],result.loc[known,'low'],result.loc[known,'high'],alpha)
    result.loc[known,'median_abs_error'] = abs(result.loc[known,'actual']-result.loc[known,'median'])
    result['position'] = 'UNKNOWN'
    result.loc[known,'position'] = 'WITHIN_BAND'
    result.loc[known & result.actual.lt(result.low),'position'] = 'BELOW_BAND'
    result.loc[known & result.actual.gt(result.high),'position'] = 'ABOVE_BAND'
    # Numerical above/below is not automatically good/bad for down-limit counts or failed rate.
    return result


def bootstrap(pairs, dates, spec):
    """Resample contiguous blocks on the FULL official date grid, including holes."""
    grid = pairs.reindex(dates).to_numpy(dtype=float)
    if not np.isfinite(grid).any(): return {'status':'NO_PAIRS'}
    rng = np.random.default_rng(spec['seed']); block=spec['block_official_sessions']
    starts = rng.integers(0,len(dates),size=(spec['draws'],int(np.ceil(len(dates)/block))))
    indexes = (starts[:,:,None]+np.arange(block)) % len(dates)
    samples = grid[indexes.reshape(spec['draws'],-1)[:,:len(dates)]]
    count = np.isfinite(samples).sum(axis=1)
    values = np.divide(np.nansum(samples,axis=1),count,where=count>0,out=np.full(len(count),np.nan))
    finite = values[np.isfinite(values)]
    return {'status':'PAIRED_OFFICIAL_DATE_BLOCK_DIAGNOSTIC', 'pairs':int(np.isfinite(grid).sum()),
            'official_dates':len(dates), 'difference_delta_minus_level':float(np.nanmean(grid)),
            'adjusted_interval':np.quantile(finite,spec['tails']).tolist(),
            'finite_draws':len(finite), 'lower_score_is_better':True,
            'simultaneous_or_live_guarantee':False}


def summarize(scored, sessions, protocol=PROTOCOL):
    records, paired = [], []
    for period, (start,end) in protocol['periods'].items():
        dates = [d for d in sessions if start<=d<=end]
        part = scored[scored.signal_date.isin(dates)]
        for key in METRICS:
            channel = part[part.metric.eq(key)]
            for arm in protocol['arms']:
                g=channel[channel.arm.eq(arm)]; valid=g[g.evaluated]
                records.append({'period':period,'metric':key,'arm':arm,'unit':METRICS[key][1],
                    'official_signal_dates':len(dates),'retained_rows':len(g),
                    'intervals_issued':int(g.status.eq('KNOWN_DIAGNOSTIC_INTERVAL').sum()),
                    'known_targets':int(g.actual.notna().sum()), 'evaluated_dates':len(valid),
                    'missing_target_dates':g.loc[g.actual.isna(),'signal_date'].tolist(),
                    'coverage':float(valid.coverage_hit.mean()) if len(valid) else None,
                    'mean_width':float(valid.width.mean()) if len(valid) else None,
                    'mean_interval_score':float(valid.interval_score.mean()) if len(valid) else None,
                    'median_mae':float(valid.median_abs_error.mean()) if len(valid) else None})
            pivot=channel[channel.evaluated].pivot(index='signal_date',columns='arm',values='interval_score')
            if not set(protocol['arms']).issubset(pivot.columns):
                result={'status':'NO_PAIRS'}
            else:
                result=bootstrap(pivot.PERSISTENCE_DELTA-pivot.LEVEL,dates,protocol['bootstrap'])
            paired.append({'period':period,'metric':key,**result})
    return records, paired


def run(output):
    target=Path(output); target.mkdir(parents=True,exist_ok=False)
    if sha(ORIGINAL)!=ORIGINAL_SHA: raise ValueError('original forecast changed')
    calendar=json.loads(CALENDAR.read_text(encoding='utf-8-sig')); sessions=calendar['sessions']
    if calendar['status']!='verified': raise ValueError('official calendar unavailable')
    indexes, sources=raw_indexes()
    sources.extend([MARKET,CALENDAR,OLD_POOL,ORIGINAL,Path(__file__)])
    inputs={str(p.resolve()):sha(p) for p in dict.fromkeys(sources)}
    # Persist fixed protocol before calculations; historical data were already inspected.
    write_json(target/'protocol.json', {'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
                                       'parameters':PROTOCOL,'source_hashes':inputs})
    market=pd.read_csv(MARKET)
    older=pd.read_csv(OLD_POOL,usecols=['date','code','boards'],dtype={'code':str})
    newer=pd.read_csv(NEW_POOL,usecols=['date','code','boards'],dtype={'code':str})
    pool=pd.concat([older[older.date.lt('2026-09-01')],newer[newer.date.ge('2026-09-01') & newer.boards.ge(2)]])
    obs,reasons,diagnostics=prepare(market,sessions,indexes,pool)
    obs.to_csv(target/'observations_DIAGNOSTIC.csv',index_label='date')
    reasons.to_csv(target/'observation_status.csv',index_label='date')
    pred=forecasts(obs,reasons,sessions)
    pred.to_csv(target/'predictions_NO_LABELS_NOT_FORWARD.csv',index=False)
    write_json(target/'prediction_receipt.json', {'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'predictions_sha256':sha(target/'predictions_NO_LABELS_NOT_FORWARD.csv'),
        'protocol_sha256':sha(target/'protocol.json'),'status':'RETROSPECTIVE_CAUSAL_REPLAY_NOT_REALTIME_FREEZE'})
    scored=score(pred,obs,reasons); scored.to_csv(target/'scores_LABEL_ONLY.csv',index=False)
    records,paired=summarize(scored,sessions)
    pd.DataFrame(records).to_csv(target/'metric_summary.csv',index=False)
    write_json(target/'paired_interval_score.json',paired)
    slots=[{'slot':s,'channels':c,'status':'DIAGNOSTIC_ARCHIVE_CHANNELS' if c else 'UNAVAILABLE',
            'definition_or_gap':why,'live_authorized':False} for s,c,why in SLOTS]
    write_json(target/'requested_ten_slots.json',slots)
    if sha(ORIGINAL)!=ORIGINAL_SHA: raise ValueError('original forecast changed during run')
    if any(sha(p)!=digest for p,digest in inputs.items()): raise ValueError('research inputs changed during run')
    report={'version':PROTOCOL['version'],'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
            'status':'FIXED_CAUSAL_INTERVAL_RESEARCH_NOT_LIVE', 'protocol':PROTOCOL,
            'data':diagnostics, 'requested_slots':10,'diagnostic_slots':7,'unavailable_slots':3,
            'channels':8,'prediction_rows':len(pred), 'metrics':records,'paired_comparisons':paired,
            'source_hashes':inputs,'original_forecast_sha256':sha(ORIGINAL),
            'limitations':['Historical provider data were collected/rebuilt after the nominal signal dates; not true point-in-time certification.',
                'Old pool flags verified and full-feedback coverage are local assertions, not exchange certification.',
                'Seven requested categories have diagnostic channels; three still lack data. No aggregate accuracy substitutes for ten-slot coverage.',
                '80% is a fixed nominal quantile band, not a guaranteed probability or proof of stable 60% prediction accuracy.',
                'Rates/counts/returns are separately scored in native units; above band is not necessarily economically favorable.',
                'No actual fills, return strategy, drawdown or risk upgrade is proved by these intervals.'],
            'forward_evidence':False,'core_upgrade_authorized':False,'stable_60_proved':False,'goal_complete':False}
    report['artifact_hashes']={p.name:sha(p) for p in target.iterdir() if p.is_file()}
    write_json(target/'report.json',report)
    return {'status':report['status'],'prediction_rows':len(pred),'channels':8,'unavailable_slots':3}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',required=True)
    print(json.dumps(run(parser.parse_args().output),ensure_ascii=False))
