import numpy as np
import pandas as pd
import pytest
import audit_qualified_pool_data_coverage as audit


def bars():
    frame = pd.DataFrame(dict(code='000001', minute='2025-07-01 '+t,
        open=10., high=11., low=9., close=10.) for t in sorted(audit.expected_grid(30)))
    frame.loc[7, 'close'] = 11.
    return frame


def daily():
    return dict(signal_date='2025-07-01', open=10., high=11., low=9., close=11.)


def test_one_record_is_not_complete_price_path():
    result = audit.inspect_bar_group(bars().tail(1), daily(), 30)
    assert result['raw_bar_count'] == 1
    assert result['grid_complete'] is False
    assert result['price_path_checked'] is False


def test_complete_grid_needs_matching_daily_prices():
    assert audit.inspect_bar_group(bars(), daily(), 30)['price_path_checked'] is True
    mismatch = daily() | {'high': 11.5}
    result = audit.inspect_bar_group(bars(), mismatch, 30)
    assert result['grid_complete'] is True
    assert result['daily_ohlc_agrees'] is False
    assert result['price_path_checked'] is False


@pytest.mark.parametrize('bad', ['duplicate', 'missing', 'invalid_price', 'wrong_date', 'extra_seconds'])
def test_invalid_grids_or_prices_do_not_pass(bad):
    frame = bars()
    if bad == 'duplicate': frame = pd.concat([frame, frame.head(1)])
    elif bad == 'missing': frame = frame.drop(index=2)
    elif bad == 'invalid_price': frame.loc[0, 'open'] = 99
    elif bad == 'wrong_date': frame.loc[0, 'minute'] = '2025-07-02 10:00:00'
    elif bad == 'extra_seconds': frame.loc[0, 'minute'] = '2025-07-01 10:00:01'
    assert audit.inspect_bar_group(frame, daily(), 30)['price_path_checked'] is False


def test_label_denominators_remain_explicit():
    uniform = audit.binary_metric(pd.Series([1., 0., np.nan]))
    stage = audit.binary_metric(pd.Series([1., 0., 1.]))
    assert uniform == dict(total=3, known=2, hits=1, failures=1, unknown=1,
                           precision_known=.5, hits_per_all_slots=1/3)
    assert stage['known'] == 3 and stage['precision_known'] == 2/3


def test_tie_order_and_complete_cohort_checks():
    screen = pd.DataFrame(dict(signal_date='2025-07-01', code=c, allowed=True) for c in ['000002', '000001'])
    pred = screen[audit.KEYS].assign(model_score=.5)
    got = audit.bind_cohort(screen, pred)
    assert got.code.tolist() == ['000001', '000002']
    assert got['rank'].tolist() == [1, 2]
    with pytest.raises(ValueError, match='cohort mismatch'): audit.bind_cohort(screen, pred.head(1))
    with pytest.raises(ValueError, match='duplicate'): audit.bind_cohort(screen, pd.concat([pred, pred.head(1)]))
    with pytest.raises(ValueError, match='OOF scores'): audit.bind_cohort(screen, pred.assign(model_score=np.nan))


def test_cross_source_disagreement_is_not_silently_deduplicated():
    candidates = pd.DataFrame([dict(signal_date='2025-07-01', code='000001')])
    checks = pd.DataFrame([dict(signal_date='2025-07-01', code='000001', grid_complete=True,
        price_path_checked=True), dict(signal_date='2025-07-01', code='000001', grid_complete=True,
        price_path_checked=False)])
    first = bars().assign(signal_date='2025-07-01')
    second = first.copy()
    second.loc[0, 'high'] += .1
    got = audit.add_resolution_flags(candidates, checks, pd.concat([first, second]), 'm5')
    assert got.loc[0, 'm5_raw_match']
    assert got.loc[0, 'm5_source_price_conflict']
    assert not got.loc[0, 'm5_price_path_checked']


def test_prior_output_is_immutable(tmp_path):
    with pytest.raises(FileExistsError, match='Preserve prior outputs'): audit.main(tmp_path)
