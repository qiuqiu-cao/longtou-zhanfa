"""Causal daily/open-price research engine. No order-flow or execution claims."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    'tail_seal_momentum', 'open_gap', 'open_gap_pct', 'open_vs_prev_high',
    'open_vs_prev_vwap', 'open_vs_prev_open', 'is_golden_auction', 'fake_open_risk',
    'ind_max_streak', 'ind_zt_count', 'ind_first_board_count', 'streak_gap_in_industry',
    'is_industry_space_dragon', 'theme_assist_ratio', 'ind_liquidity_share',
    'prev_turnover', 'prev_amount_yi', 'float_cap_yi', 'chip_cleansing_ratio',
    'vwap_cost_support', 'is_capacity_dragon', 'is_super_capacity_dragon',
    'is_small_cap_trap', 'has_bad_crash_gene', 'bad_zhuang_score',
    'mkt_promote_rate', 'is_cycle_expansion', 'is_cycle_divergence',
    'is_cycle_retraction', 'is_cycle_freezing', 'moe_capacity_dragon_boost',
    'moe_retraction_risk_brake', 'prev_board_streak', 'prev_amplitude_pct',
]


def limit_prices(pre_close, rate):
    # Integer-fen HALF_UP, rather than Python/pandas bankers' rounding.
    cents = np.floor(np.asarray(pre_close, dtype=float) * 100 + 0.5 + 1e-8)
    rate = np.asarray(rate, dtype=float)
    up = np.floor(cents * (1 + rate) + 0.5 + 1e-8) / 100
    down = np.floor(cents * (1 - rate) + 0.5 + 1e-8) / 100
    return up, down


def engineer(raw, sessions, industry_map):
    """Identical transformation for historical scoring and inference.

    Today's OHLC after open is read only for labels; all feature aggregates are
    keyed to the previous official session, never to today's realized members.
    Quote references remain historical supplier proxies, not exchange-certified.
    """
    d = raw.copy()
    d['code'] = d.code.astype(str).str.zfill(6)
    supported = d.code.str.startswith(('000', '001', '002', '003', '600', '601', '603', '605', '300', '301', '688'))
    d = d[supported & d.date.isin(sessions)].sort_values(['code', 'date']).reset_index(drop=True)
    if d.duplicated(['code', 'date']).any():
        raise ValueError('Duplicate date/code rows require source reconciliation')
    sid = {day: i for i, day in enumerate(sessions)}
    d['session_id'] = d.date.map(sid).astype(int)
    d['legacy_group'] = d.code.map(industry_map)
    d['source_known'] = d.source.str.startswith(('Sina', 'Eastmoney')).fillna(False)
    d['volume_shares'] = np.where(d.source.str.startswith('Eastmoney'), d.volume * 100, d.volume)
    d.loc[~d.source_known, 'volume_shares'] = np.nan
    valid_volume = (d.volume_shares > 0) & (d.amount > 0)
    d['vwap'] = (d.amount / d.volume_shares).where(valid_volume)
    # Do not invent a VWAP from close when volume or units fail validation.
    d['vwap'] = d.vwap.where(d.vwap.between(d.low - .02, d.high + .02))
    rate = np.where(d.code.str.startswith(('300', '301', '688')), .20, .10)
    d['upper'], d['lower'] = limit_prices(d.pre_close, rate)
    price_valid = ((d.pre_close > 0) & (d.open > 0) & (d.low > 0) &
                   (d.high >= d.low) & d.close.between(d.low, d.high) &
                   d.open.between(d.low, d.high))
    d['label_known'] = (price_valid & ~d.st & ~d.new_listing &
                        (d.high <= d.upper + .001) & (d.low >= d.lower - .001))
    d['is_zt'] = np.where(d.label_known, (d.close - d.upper).abs() < .001, np.nan)
    d['is_touch_zt'] = np.where(d.label_known, d.high >= d.upper - .001, np.nan)
    d['is_dt'] = np.where(d.label_known, (d.close - d.lower).abs() < .001, np.nan)
    group = d.groupby('code', sort=False)
    delta = group.session_id.diff()
    adjacent = delta.eq(1)
    # Gaps, unknown status, or a non-limit close break the observable streak.
    sealed = d.is_zt.eq(1) & adjacent
    reset_id = (~sealed).cumsum()
    d['board_streak'] = sealed.groupby(reset_id).cumsum().astype(int)
    prior_streak = group.board_streak.shift(1).where(adjacent)
    # Record crashes on the crash day. Shift only after the event is observable.
    crash = (prior_streak >= 3) & d.label_known & ((d.is_dt == 1) | (d.close / d.pre_close <= .91))
    d['crash_event'] = crash.astype(float)
    d['bad_crash_history'] = d.groupby('code', sort=False).crash_event.transform(lambda v: v.rolling(250, min_periods=1).sum())
    lag_columns = ['date', 'close', 'open', 'high', 'low', 'amount', 'turnover', 'vwap',
                   'is_zt', 'board_streak', 'float_cap', 'bad_crash_history', 'volume_shares']
    group = d.groupby('code', sort=False)
    for col in lag_columns:
        d['prev_' + col] = group[col].shift(1).where(adjacent)
    # Industry snapshots are from T-1 actual rows, including peers missing on T.
    known_groups = d[d.legacy_group.notna()].copy()
    ind = known_groups.groupby(['date', 'legacy_group'], observed=True).agg(
        ind_max_streak=('board_streak', 'max'), ind_zt_count=('is_zt', 'sum'),
        ind_first_board_count=('board_streak', lambda v: (v == 1).sum()),
        ind_total_amount=('amount', 'sum'))
    ind_max = known_groups.groupby(['date', 'legacy_group'], observed=True).board_streak.transform('max')
    known_groups['at_max'] = known_groups.board_streak.eq(ind_max).astype(int)
    maxima_count = known_groups.groupby(['date', 'legacy_group'], observed=True).at_max.sum()
    second = known_groups[known_groups.board_streak.lt(ind_max)].groupby(['date', 'legacy_group'], observed=True).board_streak.max()
    ind['maxima_count'] = maxima_count
    ind['second_streak'] = second
    prev_key = pd.MultiIndex.from_arrays([d.prev_date, d.legacy_group])
    for col in ind.columns:
        d[col] = ind[col].reindex(prev_key).to_numpy()
    other_max = np.where((d.prev_board_streak == d.ind_max_streak) & (d.maxima_count == 1), d.second_streak, d.ind_max_streak)
    # A single known industry member has no observed competitor: gap is unknown.
    d['streak_gap_in_industry'] = d.prev_board_streak - other_max
    d['is_industry_space_dragon'] = np.where(d.legacy_group.notna(),
        ((d.prev_board_streak == d.ind_max_streak) & (d.maxima_count == 1) & (d.prev_board_streak >= 2)).astype(float), np.nan)
    d['theme_assist_ratio'] = d.ind_first_board_count / d.ind_zt_count.replace(0, np.nan)
    d['ind_liquidity_share'] = d.prev_amount / d.ind_total_amount.replace(0, np.nan)
    # Yesterday's realized promotion rate, computed on yesterday's eligible denominator.
    d['was_limit_prev'] = d.prev_is_zt.eq(1)
    d['promoted_today'] = d.was_limit_prev & d.is_zt.eq(1)
    d['promote_label_known'] = d.was_limit_prev & d.label_known
    market = d.groupby('date', sort=True).agg(
        total_zt=('is_zt', 'sum'), total_dt=('is_dt', 'sum'),
        promote_den=('was_limit_prev', 'sum'), promote_known=('promote_label_known', 'sum'),
        promoted=('promoted_today', 'sum'))
    market['mkt_promote_rate'] = (market.promoted / market.promote_den.replace(0, np.nan)).where(market.promote_known == market.promote_den)
    d['mkt_promote_rate'] = d.prev_date.map(market.mkt_promote_rate)
    dt = d.prev_date.map(market.total_dt)
    zt = d.prev_date.map(market.total_zt)
    regime = np.where(dt >= 8, 3, np.where((d.mkt_promote_rate >= .28) & (zt >= 35), 1,
                      np.where((d.mkt_promote_rate < .15) & (zt < 25), 0, 2))).astype(float)
    regime[pd.isna(d.mkt_promote_rate) | pd.isna(dt)] = np.nan
    for number, name in enumerate(['freezing', 'expansion', 'divergence', 'retraction']):
        d['is_cycle_' + name] = np.where(np.isfinite(regime), (regime == number).astype(float), np.nan)
    # Real tail orders, matched/unmatched auction volumes and queue histories absent.
    d['tail_seal_momentum'] = np.nan
    d['prev_amplitude_pct'] = (d.prev_high - d.prev_low) / d.prev_close * 100
    d['open_gap'] = d.open / d.pre_close - 1
    d['open_gap_pct'] = d.open_gap * 100
    d['open_vs_prev_high'] = d.open / d.prev_high - 1
    d['open_vs_prev_vwap'] = d.open / d.prev_vwap - 1
    d['open_vs_prev_open'] = d.open / d.prev_open - 1
    d['is_golden_auction'] = np.where(d.prev_vwap.notna(), (d.open_gap.between(.02, .07) & (d.open_vs_prev_vwap >= 0) & (d.open_vs_prev_high >= -.001)).astype(float), np.nan)
    d['fake_open_risk'] = np.where(d.prev_vwap.notna(), ((d.open_gap >= .04) & ((d.open_vs_prev_vwap < 0) | (d.prev_amount < 2.5e8))).astype(float), np.nan)
    span = (d.prev_high - d.prev_low).replace(0, np.nan)
    d['chip_cleansing_ratio'] = d.prev_turnover * (d.prev_vwap - d.prev_low) / span
    d['vwap_cost_support'] = d.open_vs_prev_vwap
    d['prev_amount_yi'] = d.prev_amount / 1e8
    d['float_cap_yi'] = d.prev_float_cap / 1e8
    for name, amt, turn in [('is_capacity_dragon', 5, 10), ('is_super_capacity_dragon', 8, 12)]:
        d[name] = np.where(d.prev_vwap.notna(), ((d.prev_amount_yi >= amt) & d.prev_turnover.between(turn, 28) & (d.open_vs_prev_vwap >= 0)).astype(float), np.nan)
    d['is_small_cap_trap'] = np.where(d.float_cap_yi.notna(), ((d.float_cap_yi < 30) & (d.prev_amount_yi < 3)).astype(float), np.nan)
    d['has_bad_crash_gene'] = np.where(d.prev_bad_crash_history.notna(), (d.prev_bad_crash_history >= 1).astype(float), np.nan)
    d['bad_zhuang_score'] = d.is_small_cap_trap + d.has_bad_crash_gene * 2.5 + d.fake_open_risk * 2
    d['moe_capacity_dragon_boost'] = d.is_capacity_dragon * np.where((regime == 1) | (regime == 2), 2., .5)
    d.loc[~np.isfinite(regime), 'moe_capacity_dragon_boost'] = np.nan
    d['moe_retraction_risk_brake'] = np.where(regime == 3, d.bad_zhuang_score * 2 + (d.prev_board_streak >= 3) * 1.5, 0.)
    d.loc[~np.isfinite(regime), 'moe_retraction_risk_brake'] = np.nan
    # Pool membership does not depend on today's close/high/low or next-day availability.
    d['eligible'] = (~d.st & ~d.new_listing & d.source_known & adjacent &
                     d.prev_is_zt.eq(1) & (d.open < d.upper - .001) &
                     (d.open_gap >= 0) & (d.open > 0))
    d['target_limit_up'] = d.is_zt
    d['target_touch_limit'] = d.is_touch_zt
    d['target_day_return'] = ((d.close / d.open) - 1).where(d.label_known)
    d['target_big_loss'] = np.where(d.label_known, (d.target_day_return <= -.04).astype(float), np.nan)
    d['target_bomb'] = np.where(d.label_known, (d.is_touch_zt.eq(1) & d.is_zt.eq(0) & ((d.high - d.close) / d.open >= .04)).astype(float), np.nan)
    group = d.groupby('code', sort=False)
    nadj = group.session_id.shift(-1).sub(d.session_id).eq(1)
    d['next_date'] = group.date.shift(-1).where(nadj)
    d['next_open'] = group.open.shift(-1).where(nadj)
    d['next_return_proxy'] = d.next_open / d.open - 1
    # Training labels/weights may use future outcomes, but fold boundaries purge them.
    w = np.where(d.target_big_loss == 1, 4., 1.)
    w = np.where(d.target_bomb == 1, 3.5, w)
    w = np.where(d.is_zt.eq(1) & (d.next_open > d.close), 2.5, w)
    d['sample_weight'] = w
    d[FEATURES] = d[FEATURES].replace([np.inf, -np.inf], np.nan)
    return d


def snapshot_inputs(raw, code, date):
    """Only past bars plus today's open/reference enter inference."""
    hist = raw[raw.date.lt(date)].copy()
    current = raw[raw.date.eq(date)].copy()
    # Day-T unknown OHLC outcome fields are deliberately blanked. History stays intact.
    current[['high', 'low', 'close', 'amount', 'volume', 'turnover', 'float_cap']] = np.nan
    return pd.concat([hist, current], ignore_index=True)
