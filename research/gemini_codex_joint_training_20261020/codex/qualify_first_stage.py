"""Independent raw-price audit for the joint experiment; no fitting or writes upstream."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import first_board_leader_model as base

OUT = Path(__file__).resolve().parent
DATA = ROOT / 'research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv'
TARGET = 'promotion_next_LABEL_ONLY'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write(name, value):
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)


def flag(series):
    return series.astype(str).str.lower().isin(['true', '1', '1.0', 'yes'])


def run():
    bindings = {str(p.resolve()): sha(p) for p in [DATA, *base.INPUTS.values(), Path(__file__)]}
    write('CODEX_AUDIT_BEFORE_DATA_READ.json', dict(formed_at=now(), task='Raw archive price and label qualification independent of Gemini proposal',
        metrics='Complete33725keys, exact-cent signal and next-official-session labels, first/second adjacency and volume-unit proxies',
        price_tolerance='Only1e-7cent representation noise accepted for float serialization; no quote-level one-cent tolerance',
        model_fits=0, source_mutations=False, bindings=bindings))
    sessions = json.loads(base.INPUTS['calendar'].read_text(encoding='utf-8'))['sessions']
    assert sessions == sorted(set(sessions))
    dates = {d: i for i, d in enumerate(sessions)}
    cohort = pd.read_csv(DATA, dtype={'code': str}, float_precision='round_trip')
    assert len(cohort) == 33725 and not cohort.duplicated(['signal_date', 'code']).any()
    cols = ['date', 'code', 'name', 'open', 'high', 'low', 'close', 'pre_close', 'volume', 'amount',
            'st', 'new_listing', 'status_verified', 'source']
    raw = pd.concat([pd.read_csv(base.INPUTS[k], usecols=cols, dtype={'code': str}, float_precision='round_trip', low_memory=False)
                     for k in ['bars', 'recovered']], ignore_index=True)
    raw['code'] = raw.code.str.zfill(6)
    raw['date'] = raw.date.astype(str).str[:10]
    duplicate = raw.duplicated(['date', 'code'], keep=False)
    if duplicate.any():
        assert raw[duplicate].groupby(['date', 'code'])[['open', 'high', 'low', 'close', 'pre_close']].nunique().max(axis=1).le(1).all()
    original_raw = len(raw)
    raw = raw.drop_duplicates(['date', 'code'], keep='last')
    raw['p'] = raw.date.map(dates)
    outside = int(raw.p.isna().sum())
    raw = raw[raw.p.notna()].sort_values(['code', 'p']).reset_index(drop=True)
    raw.p = raw.p.astype(int)
    price = raw[['open', 'high', 'low', 'close', 'pre_close']].to_numpy(dtype=float, copy=True)
    rounded = np.rint(price*100)
    exact_cent = np.isfinite(price).all(axis=1) & (price > 0).all(axis=1) & (np.abs(price*100-rounded) <= 1e-7).all(axis=1)
    op, high, low, close, preclose = rounded.T
    valid = exact_cent & (low <= np.minimum(op, close)) & (high >= np.maximum(op, close)) & (low <= high)
    cap = np.floor((preclose*11+5)/10)
    floor = np.floor((preclose*9+5)/10)
    regular = valid & (high <= cap) & (low >= floor)
    eligible = flag(raw.status_verified) & ~flag(raw.st) & ~flag(raw.new_listing) & raw.code.str.startswith(('000', '001', '002', '003', '600', '601', '603', '605'))
    raw['audit_observed'] = regular & eligible
    raw['audit_limit'] = raw.audit_observed & (close == cap)
    raw['audit_valid'] = valid
    raw['audit_regular'] = regular
    raw['audit_eligible'] = eligible
    raw['audit_cap_cents'] = cap
    raw['audit_close_cents'] = close
    raw['audit_preclose_cents'] = preclose
    group = raw.groupby('code', sort=False)
    adjacent = raw.p.sub(group.p.shift()).eq(1)
    previous_limit = group.audit_limit.shift().fillna(False).astype(bool)
    previous_valid = group.audit_valid.shift().fillna(False).astype(bool)
    previous_eligible = group.audit_eligible.shift().fillna(False).astype(bool)
    raw['audit_first'] = raw.audit_limit & adjacent & previous_valid & previous_eligible & ~previous_limit
    link = raw.audit_limit & adjacent & previous_limit
    runs = (~link).groupby(raw.code).cumsum()
    raw['audit_height'] = raw.audit_limit.astype(int).groupby([raw.code, runs]).cumsum()
    raw['audit_second'] = (raw.audit_height.eq(2) & adjacent & group.audit_first.shift().fillna(False).astype(bool))
    raw['preclose_matches_previous_official_close'] = (raw.audit_preclose_cents.eq(group.audit_close_cents.shift())).where(adjacent)
    # A unit proxy from independent daily totals is not declared metadata.
    # Test native volume as shares or100-share lots against daily VWAP bounds.
    valid_volume = raw.volume.gt(0) & raw.amount.gt(0) & raw.audit_valid
    price_per_native = raw.amount/raw.volume.where(raw.volume.gt(0))
    share_like = valid_volume & price_per_native.between(raw.low-.011, raw.high+.011)
    lot_like = valid_volume & price_per_native.div(100).between(raw.low-.011, raw.high+.011)
    raw['volume_scale_proxy'] = np.where(share_like & ~lot_like, 1., np.where(lot_like & ~share_like, 100., np.nan))
    bycode = raw.groupby('code', sort=False)
    for window in [5, 20]:
        localmin = bycode.volume_scale_proxy.transform(lambda x: x.rolling(window+1, min_periods=window+1).min())
        localmax = bycode.volume_scale_proxy.transform(lambda x: x.rolling(window+1, min_periods=window+1).max())
        continuous = raw.p.sub(bycode.p.shift(window)).eq(window)
        raw[f'mixed_volume_scale_proxy_{window}'] = localmin.ne(localmax).where(localmin.notna() & localmax.notna() & continuous)
    raw_view = raw.set_index(['date', 'code'])
    keys = pd.MultiIndex.from_arrays([cohort.signal_date, cohort.code], names=['date', 'code'])
    signal = raw_view.reindex(keys).reset_index()
    assert signal.p.notna().all()
    np.testing.assert_allclose(signal[['open', 'high', 'low', 'close', 'pre_close']].to_numpy(),
                               cohort[['open', 'high', 'low', 'close', 'pre_close']].to_numpy(), atol=1e-12, rtol=1e-12)
    next_dates = cohort.p.map(lambda p: sessions[int(p)+1] if int(p)+1 < len(sessions) else None)
    future_keys = pd.MultiIndex.from_arrays([next_dates, cohort.code], names=['date', 'code'])
    future = raw_view.reindex(future_keys).reset_index()
    known = future.audit_observed.fillna(False).to_numpy(dtype=bool)
    reconstructed = np.where(known, future.audit_limit.fillna(False).to_numpy(dtype=float), np.nan)
    old = cohort[TARGET].to_numpy(dtype=float)
    same_label = (old == reconstructed) | (np.isnan(old) & np.isnan(reconstructed))
    same_stage = np.where(cohort.stage.eq(1), signal.audit_first, signal.audit_second)
    fields = ['signal_date', 'code', 'stage', 'height', 'close', 'pre_close', 'label_end', TARGET]
    audit = cohort[fields].copy()
    audit['audit_exact_cent_regular_signal'] = signal.audit_observed.to_numpy()
    audit['audit_signal_exact_limit'] = signal.audit_limit.to_numpy()
    audit['audit_stage_identity_matches'] = same_stage
    audit['audit_height'] = signal.audit_height.to_numpy()
    audit['audit_next_official_date'] = next_dates
    audit['audit_next_label'] = reconstructed
    audit['audit_original_label_matches'] = same_label
    audit['audit_target_date_matches'] = cohort.label_end.fillna('').eq(next_dates.fillna(''))
    audit['preclose_matches_previous_official_close'] = signal.preclose_matches_previous_official_close.to_numpy()
    for field in ['volume_scale_proxy', 'mixed_volume_scale_proxy_5', 'mixed_volume_scale_proxy_20']:
        audit[field] = signal[field].to_numpy()
    audit.to_csv(OUT / 'COMPLETE_33725_RAW_PRICE_LABEL_QUALIFICATION.csv', index=False, encoding='utf-8-sig')
    bad = audit[~audit.audit_signal_exact_limit | ~audit.audit_stage_identity_matches | ~audit.audit_original_label_matches | ~audit.audit_target_date_matches]
    bad.to_csv(OUT / 'EXACT_PRICE_STAGE_LABEL_CONFLICTS.csv', index=False, encoding='utf-8-sig')
    summary = dict(formed_at=now(), complete_cohort=len(audit), original_raw_rows=original_raw,
        duplicate_raw_keys_collapsed=int(duplicate.sum()), raw_outside_original_calendar=outside,
        exact_signal_limit_disagreements=int((~audit.audit_signal_exact_limit).sum()),
        first_second_identity_disagreements=int((~audit.audit_stage_identity_matches).sum()),
        next_official_date_disagreements=int((~audit.audit_target_date_matches).sum()),
        next_official_label_disagreements=int((~audit.audit_original_label_matches).sum()),
        stages={str(stage): dict(rows=len(g), known=int(g[TARGET].notna().sum()), unknown=int(g[TARGET].isna().sum()),
                    label_disagreements=int((~g.audit_original_label_matches).sum()),
                    preclose_previous_close_disagreements=int(g.preclose_matches_previous_official_close.eq(False).sum()),
                    volume_scale_proxy=g.volume_scale_proxy.fillna(-1).value_counts().to_dict(),
                    mixed_scale5=int(g.mixed_volume_scale_proxy_5.eq(True).sum()),
                    mixed_scale20=int(g.mixed_volume_scale_proxy_20.eq(True).sum())) for stage, g in audit.groupby('stage')},
        conclusions=['Independent recalculation uses the currently bound historical raw archive, not a second independent quotation provider.',
                     'Any difference is qualification evidence; originals and models remain untouched.',
                     'Previous0.011tolerancelegacyrelaymistakesarenotautomaticallycurrentfirst-boarderrors.',
                     'VWAP-derivedvolumeunitproxyisnotindependentlydeclaredunitmetadata.',
                     'Surviving stock universe, historical eligibility provenance, market-cap definition and late source availability remain.'],
        new_fits=0, protected_source_changes=False)
    for name, digest in bindings.items():
        assert sha(name) == digest, name
    write('CODEX_FIRST_SECOND_DATA_QUALIFICATION.json', summary)
    print(json.dumps(clean(summary), ensure_ascii=True), flush=True)


if __name__ == '__main__':
    run()
