"""Audit raw matches, price-path checks and denominators without fitting or ranking changes."""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from first_board_leader_model import FEATURES
from legacy_industry_snapshot import attach_legacy_groups, BANK

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / 'research/data_coverage_7254_qualified_corrected'
SCREEN_PATH = ROOT / 'research/first_board_leader_emotion_20261021T1706Z/ALL28503_PATHS_AND_ORIGINAL_SCREEN.csv'
PREDS_PATH = ROOT / 'research/first_board_leader_20261020T114600Z/evaluation_ALL_PREDICTIONS.csv'
SOURCES = [
    ('30m', ROOT / 'data/v290/full_long_interval_20261002_v1/thirty_minute_bars_UNRECONCILED.csv', 30),
    ('5m_aug', ROOT / 'data/v289/full_august_five_minute_20261002_v1/five_minute_bars_UNRECONCILED.csv', 5),
    ('5m_sep', ROOT / 'data/v287/september_five_minute_20261002_v1/five_minute_bars_UNRECONCILED.csv', 5),
]
KEYS = ['signal_date', 'code']
OHLC = ['open', 'high', 'low', 'close']
PRICE_ATOL = .005001


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def read(path, **kwargs):
    return pd.read_csv(path, dtype={'code': str}, float_precision='round_trip', low_memory=False, **kwargs)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def binary_metric(series):
    if not series.dropna().isin([0, 1]).all(): raise ValueError('Nonbinary outcome')
    known, hits = int(series.notna().sum()), int(series.eq(1).sum())
    return dict(total=len(series), known=known, hits=hits, failures=known-hits, unknown=len(series)-known,
                precision_known=hits/known if known else None, hits_per_all_slots=hits/len(series) if len(series) else None)


def bind_cohort(screen, predictions):
    """Use the complete fixed dates and original OOF score with code-order ties."""
    test = screen[screen.signal_date.between('2025-07-01', '2026-08-31')].copy()
    for frame in (test, predictions):
        if frame.empty or frame.duplicated(KEYS).any(): raise ValueError('Empty or duplicate candidate keys')
        if not frame.code.str.fullmatch(r'\d{6}').all(): raise ValueError('Invalid code')
    if set(map(tuple, test[KEYS].to_numpy())) != set(map(tuple, predictions[KEYS].to_numpy())):
        raise ValueError('Complete prediction cohort mismatch')
    if not np.isfinite(pd.to_numeric(predictions.model_score, errors='coerce')).all():
        raise ValueError('Missing or invalid OOF scores')
    merged = test.merge(predictions[KEYS+['model_score']], on=KEYS, validate='one_to_one')
    flags = merged.allowed.astype(str).str.lower()
    if not flags.isin(['true', 'false', '1', '0', '1.0', '0.0']).all(): raise ValueError('Unknown original scope flag')
    out = merged[flags.isin(['true', '1', '1.0'])].copy()
    out = out.sort_values(['signal_date', 'model_score', 'code'], ascending=[True, False, True]).reset_index(drop=True)
    out['rank'] = out.groupby('signal_date').cumcount()+1
    out['pool_segment'] = np.where(out['rank'].le(3), 'TOP3', np.where(out['rank'].le(10), 'RANK_4_TO_10', 'BEYOND_TOP10'))
    return out


def valid_ohlc(frame):
    p = frame[OHLC].apply(pd.to_numeric, errors='coerce')
    return (np.isfinite(p).all(axis=1) & p.gt(0).all(axis=1) & p.high.ge(p[['open', 'low', 'close']].max(axis=1))
            & p.low.le(p[['open', 'high', 'close']].min(axis=1)))


def expected_grid(interval):
    if interval not in (5, 30): raise ValueError('Unsupported bar interval')
    return {f'{m//60:02d}:{m%60:02d}:00' for start, end in ((570, 690), (780, 900))
            for m in range(start+interval, end+1, interval)}


def inspect_bar_group(rows, daily, interval):
    """Price-path checks do not certify continuous seals, auction data or fills."""
    parsed = pd.to_datetime(rows.minute, errors='coerce', format='%Y-%m-%d %H:%M:%S')
    same_date = parsed.notna().all() and parsed.dt.strftime('%Y-%m-%d').eq(daily['signal_date']).all()
    duplicate = rows.minute.duplicated().any()
    grid = bool(same_date and not duplicate and set(parsed.dt.strftime('%H:%M:%S')) == expected_grid(interval))
    ordered = rows.assign(_stamp=parsed).sort_values('_stamp')
    valid = bool(valid_ohlc(ordered).all())
    observed = dict(open=ordered.open.iloc[0], high=ordered.high.max(), low=ordered.low.min(), close=ordered.close.iloc[-1])
    ref = pd.DataFrame([daily])
    agrees = bool(valid and valid_ohlc(ref).all() and all(abs(float(observed[k])-float(daily[k])) <= PRICE_ATOL for k in OHLC))
    return dict(raw_bar_count=len(rows), grid_complete=grid, duplicate_times=bool(duplicate),
                bar_prices_valid=valid, daily_ohlc_agrees=agrees, price_path_checked=bool(grid and agrees),
                missing_bar_ends='|'.join(sorted(expected_grid(interval)-set(parsed.dt.strftime('%H:%M:%S').dropna()))))


def inspect_source(name, path, interval, candidates):
    wanted = set(map(tuple, candidates[KEYS].to_numpy()))
    pieces = []
    for chunk in read(path, usecols=['code', 'minute']+OHLC, chunksize=200000):
        chunk['signal_date'] = chunk.minute.str[:10]
        mask = [(d, c) in wanted for d, c in zip(chunk.signal_date, chunk.code)]
        if any(mask): pieces.append(chunk.loc[mask].copy())
    bars = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=KEYS+['minute']+OHLC)
    references = candidates.set_index(KEYS)
    checks = []
    for (day, code), group in bars.groupby(KEYS, sort=True):
        ref = references.loc[(day, code)].to_dict() | dict(signal_date=day)
        checks.append(dict(source=name, signal_date=day, code=code, **inspect_bar_group(group, ref, interval)))
    details = pd.DataFrame(checks, columns=['source']+KEYS+['raw_bar_count', 'grid_complete', 'duplicate_times',
        'bar_prices_valid', 'daily_ohlc_agrees', 'price_path_checked', 'missing_bar_ends'])
    report_path = path.parent/'report.json'
    manifest = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    expected = manifest.get('artifact_hashes', {}).get(path.name)
    actual = sha(path)
    if expected is not None and actual != expected: raise ValueError(f'Archive hash mismatch: {path}')
    info = dict(path=str(path.relative_to(ROOT)), sha256=actual, interval_minutes=interval,
        prior_manifest_hash_matches=expected == actual if expected is not None else None,
        original_collection_recorded_at=manifest.get('recorded_at'),
        raw_match_rows=len(details), grid_complete_rows=int(details.grid_complete.sum()),
        price_path_checked_rows=int(details.price_path_checked.sum()),
        scope='Retrospective archive originally collected for relay cohorts; first-board availability is not random',
        raw_http_redecoded_this_run=False, volume_reconciled_this_run=False,
        one_minute_or_queue_evidence=False)
    return details, bars, info


def add_resolution_flags(candidates, checks, bars, suffix):
    out = candidates.copy()
    if checks.empty:
        for field in ['raw_match', 'grid_complete', 'price_path_checked', 'source_price_conflict']:
            out[f'{suffix}_{field}'] = False
        return out
    # Different snapshots must not silently overwrite contradictory prices.
    grouped_bars = bars.groupby(KEYS+['minute'])[OHLC]
    spread = grouped_bars.max()-grouped_bars.min()
    conflicts = spread.gt(PRICE_ATOL).any(axis=1).groupby(level=[0, 1]).any()
    summary = checks.groupby(KEYS)[['grid_complete', 'price_path_checked']].any()
    summary['raw_match'] = True
    summary['source_price_conflict'] = conflicts.reindex(summary.index, fill_value=False)
    summary['price_path_checked'] &= ~summary.source_price_conflict
    summary = summary.add_prefix(suffix+'_').reset_index()
    out = out.merge(summary, on=KEYS, how='left', validate='one_to_one')
    for field in summary.columns.difference(KEYS): out[field] = out[field].eq(True)
    return out


def cohort_metrics(rows):
    return dict(rows=len(rows), dates=int(rows.signal_date.nunique()),
        reach4_uniform_path=binary_metric(rows.reach4_LABEL_ONLY),
        reach2_uniform_path=binary_metric(rows.reach2_LABEL_ONLY),
        promotion_next_separate_stage_label=binary_metric(rows.promotion_next_LABEL_ONLY))


def main(output=OUT_DIR):
    output = Path(output)
    if output.exists(): raise FileExistsError(f'Preserve prior outputs: {output}')
    files = [SCREEN_PATH, PREDS_PATH, Path(__file__), ROOT/'legacy_industry_snapshot.py',
        BANK/'qualified_legacy_snapshot.json', BANK/'legacy_2021_industry_mapping_WITH_UNKNOWNS.csv']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
    allowed = bind_cohort(read(SCREEN_PATH), read(PREDS_PATH, usecols=KEYS+['model_score']))
    if len(allowed) != 7254 or allowed.signal_date.nunique() != 286: raise ValueError('Frozen audit scope changed')
    allowed['daily_ohlc_valid'] = valid_ohlc(allowed)
    numeric = allowed[FEATURES].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    allowed['features44_nonmissing'] = numeric.notna().all(axis=1)
    allowed['missing_feature_count'] = numeric.isna().sum(axis=1)
    details, bars, inventories = {}, {}, []
    for name, path, interval in SOURCES:
        details[name], bars[name], info = inspect_source(name, path, interval, allowed)
        inventories.append(info)
    allowed = add_resolution_flags(allowed, details['30m'], bars['30m'], 'm30')
    allowed = add_resolution_flags(allowed, pd.concat([details['5m_aug'], details['5m_sep']]),
        pd.concat([bars['5m_aug'], bars['5m_sep']]), 'm5')
    allowed = attach_legacy_groups(allowed)
    allowed['has_legacy_industry_proxy'] = allowed.legacy_industry_group_PROXY.notna()
    dates = allowed.groupby('signal_date').size().rename('total_candidates').to_frame()
    coverage = {}
    for col in ['daily_ohlc_valid', 'features44_nonmissing', 'm30_raw_match', 'm30_price_path_checked',
                'm5_raw_match', 'm5_price_path_checked', 'has_legacy_industry_proxy']:
        n = allowed.groupby('signal_date')[col].sum()
        dates[col+'_rows'] = n
        coverage[col] = dict(rows=int(allowed[col].sum()), fraction=float(allowed[col].mean()),
            full_dates=int(n.eq(dates.total_candidates).sum()), zero_dates=int(n.eq(0).sum()),
            partial_dates=int((n.gt(0) & n.lt(dates.total_candidates)).sum()))
    subsets = {name: cohort_metrics(g) for name, g in allowed.groupby('pool_segment')}
    covered_subsets = {name: cohort_metrics(g) for name, g in allowed[allowed.m30_raw_match].groupby('pool_segment')}
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(), status='COVERAGE_AND_PRICE_CHECKS_ONLY_NOT_STRATEGY_VALIDATION',
        fits=0, source_hashes=hashes, total=cohort_metrics(allowed), segment_metrics=subsets,
        m30_raw_match_segment_metrics=covered_subsets, coverage=coverage, source_inventory=inventories,
        feature_missing_counts={c:int(v) for c,v in numeric.isna().sum().items() if v},
        pool_size=dict(min=int(dates.total_candidates.min()), median=float(dates.total_candidates.median()), max=int(dates.total_candidates.max())),
        m30_raw_match_all_outcomes=cohort_metrics(allowed[allowed.m30_raw_match]),
        m30_checked_all_outcomes=cohort_metrics(allowed[allowed.m30_price_path_checked]),
        one_minute_status='NOT_SCANNED_BY_THIS_SCRIPT_NO_PROJECT_WIDE_ZERO_CLAIM',
        industry_semantics='2021_STATIC_GROUP_PROXY_NOT_SIGNAL_DATE_INDUSTRY_OR_ACTIVE_THEME',
        label_note='reach2 and reach4 use the archived uniform path qualification; promotion_next has a different maturity denominator; do not mix.',
        interpretation=[
            'A raw date/code match can contain only one bar. Grid and daily-price checks are separate.',
            'Complete eight-bar price paths support coarse OHLC descriptions, not exact seal/reseal times, order queues or fills.',
            '30-minute data cannot verify a 09:45 exit or 09:25 auction conditions.',
            'Coverage differs by future outcome. Retrospective relay-code sampling cannot establish unbiased first-board performance.',
            'Missing evidence remains in all candidates and dates. Any later rule must specify its fallback before testing.',
            'No feature uplift, promotion rule, training, trade, schedule or production change performed.'
        ])
    if any(sha(ROOT/p) != h for p,h in hashes.items()): raise ValueError('Inputs changed during audit')
    output.mkdir(parents=True, exist_ok=False)
    kept = KEYS+['name', 'rank', 'pool_segment', 'model_score', 'reach2_LABEL_ONLY', 'reach4_LABEL_ONLY',
        'promotion_next_LABEL_ONLY', 'daily_ohlc_valid', 'features44_nonmissing', 'missing_feature_count',
        'm30_raw_match', 'm30_grid_complete', 'm30_price_path_checked', 'm30_source_price_conflict',
        'm5_raw_match', 'm5_grid_complete', 'm5_price_path_checked', 'm5_source_price_conflict',
        'legacy_industry_group_PROXY', 'legacy_industry_source_status']
    allowed[kept].to_csv(output/'QUALIFIED_7254_DATA_COVERAGE.csv', index=False, encoding='utf-8-sig')
    pd.concat(details.values()).to_csv(output/'BAR_SOURCE_CHECKS.csv', index=False, encoding='utf-8-sig')
    dates.to_csv(output/'COMPLETE286_DATE_COVERAGE.csv', encoding='utf-8-sig')
    write(output/'DATA_COVERAGE_REPORT.json', report)
    print(json.dumps({k:report[k] for k in ['status', 'total', 'coverage', 'm30_raw_match_all_outcomes',
        'm30_checked_all_outcomes', 'one_minute_status', 'industry_semantics']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT_DIR)
    main(parser.parse_args().output)
