"""A source-causal co-limit-up graph qualification, without model fitting."""
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

OUT = Path(__file__).resolve().parent / 'network_qualification_precision_repair'
COHORT = ROOT / 'research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv'
FIELDS = ['co_limit_peer_n', 'co_limit_history_weight', 'co_limit_current_observed_weight_fraction',
          'co_limit_peer_limit_rate', 'co_limit_peer_first_rate', 'co_limit_peer_failed_fraction',
          'co_limit_peer_excess_return', 'co_limit_previous_high_survival']
WINDOW, MIN_JOINT, MIN_CO_LIMIT, MAX_PEERS = 126, 60, 2, 8
BAR_FIELDS = ['date', 'code', 'eligible', 'valid_price', 'regular_price', 'amount', 'volume',
              'limit', 'first', 'failed', 'day_return', 'previous_height', 'previous_limit']


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


def grid(bars, sessions):
    b = bars[BAR_FIELDS].copy()
    assert not b.duplicated(['date', 'code']).any()
    codes = np.array(sorted(b.code.unique()))
    codeindex, dateindex = {c: i for i, c in enumerate(codes)}, {d: i for i, d in enumerate(sessions)}
    row = b.date.map(dateindex).to_numpy()
    col = b.code.map(codeindex).to_numpy()
    assert not pd.isna(row).any()
    row = row.astype(int)
    shape = len(sessions), len(codes)
    observed = b.eligible & b.valid_price & b.regular_price & b.amount.gt(0) & b.volume.gt(0)
    result = {}
    for name, values in [('observed', observed), ('limit', observed & b.limit),
                         ('first', observed & b['first']), ('failed', observed & b.failed),
                         ('previous_high', b.previous_limit & b.previous_height.ge(3))]:
        a = np.zeros(shape, dtype=bool)
        a[row, col] = values.to_numpy(dtype=bool)
        result[name] = a
    result['returns'] = np.full(shape, np.nan)
    result['returns'][row, col] = b.day_return.where(observed).to_numpy(dtype=float)
    result['market_median'] = b.day_return.where(observed).groupby(b.date).median().reindex(sessions).to_numpy(dtype=float)
    return codes, result


def build_features(bars, sessions, keys, *, diagnostics=False):
    assert list(keys.columns) == ['signal_date', 'code']
    assert not keys.duplicated().any() and sessions == sorted(set(sessions))
    codes, arrays = grid(bars, sessions)
    dateindex, codeindex = {d: i for i, d in enumerate(sessions)}, {c: i for i, c in enumerate(codes)}
    positions = keys.signal_date.map(dateindex)
    assert positions.notna().all() and keys.code.isin(codes).all()
    groups = {}
    for index, position in enumerate(positions):
        groups.setdefault(int(position), []).append(index)
    bycode = keys.code.map(codeindex).to_numpy(dtype=int)
    n = len(codes)
    count = np.zeros((n, n), dtype=np.int16)
    cap_count = arrays['limit'].sum(axis=1)
    values = np.full((len(keys), len(FIELDS)), np.nan)
    peers = []

    def update(day, sign):
        active = np.flatnonzero(arrays['limit'][day])
        if len(active) < 2:
            return
        ix = np.ix_(active, active)
        count[ix] += sign

    for position in range(int(positions.max())+1):
        if position > 0:
            update(position-1, 1)
        if position-WINDOW-1 >= 0:
            update(position-WINDOW-1, -1)
        if position not in groups:
            continue
        past = arrays['observed'][max(0, position-WINDOW):position]
        for index in groups[position]:
            own = bycode[index]
            if len(past) < MIN_JOINT or past[:, own].sum() < MIN_JOINT:
                continue
            candidates = np.flatnonzero(count[own] >= MIN_CO_LIMIT)
            candidates = candidates[candidates != own]
            if len(candidates):
                common = (past[:, own, None] & past[:, candidates]).sum(axis=0)
                candidates = candidates[common >= MIN_JOINT]
            if len(candidates):
                cap_positions = np.flatnonzero(arrays['limit'][max(0, position-WINDOW):position, own])+max(0, position-WINDOW)
                cap_positions = cap_positions[cap_count[cap_positions] > 1]
                # Direct window arithmetic prevents residuals from expired
                # edges changing otherwise equal-history peer selection.
                candidate_weight = (arrays['limit'][np.ix_(cap_positions, candidates)]/(cap_count[cap_positions]-1)[:, None]).sum(axis=0)
                order = np.lexsort((codes[candidates], -candidate_weight))
                selected = candidates[order[:MAX_PEERS]]
                historic_weight = candidate_weight[order[:MAX_PEERS]]
            else:
                selected = candidates
                historic_weight = np.array([], dtype=float)
            values[index, :2] = len(selected), historic_weight.sum()
            if not len(selected):
                continue
            observed = arrays['observed'][position, selected]
            live_weight = historic_weight*observed
            denominator = live_weight.sum()
            values[index, 2] = denominator/historic_weight.sum()
            if denominator > 0:
                for column, field in [(3, 'limit'), (4, 'first'), (5, 'failed')]:
                    values[index, column] = np.dot(live_weight, arrays[field][position, selected])/denominator
                returns = arrays['returns'][position, selected]
                values[index, 6] = np.dot(live_weight, np.nan_to_num(returns, nan=0.))/denominator-arrays['market_median'][position]
                high = arrays['previous_high'][position, selected] & observed
                if high.any():
                    values[index, 7] = np.dot(historic_weight[high], arrays['limit'][position, selected][high])/historic_weight[high].sum()
            if diagnostics:
                peers.append(dict(signal_date=keys.iloc[index].signal_date, code=keys.iloc[index].code,
                    selected_peers=[dict(code=codes[peer], past_joint_caps=int(count[own, peer]),
                                         historical_weight=float(peer_weight),
                                         joint_observation_days=int((past[:, own] & past[:, peer]).sum())) for peer, peer_weight in zip(selected, historic_weight)]))
        if diagnostics and position % 126 == 0:
            print(json.dumps(dict(progress_official_position=position, completed_rows=int((positions <= position).sum()))), flush=True)
    output = keys.copy()
    output[FIELDS] = values
    return output, peers


def run():
    OUT.mkdir(exist_ok=False)
    inputs = [COHORT, *base.INPUTS.values(), Path(__file__), Path(__file__).parent/'CODEX_PROPOSAL_BEFORE_GEMINI_REVIEW.md']
    bindings = {str(p.resolve()): sha(p) for p in inputs}
    write('protocol_BEFORE_FEATURES_NO_FIT.json', dict(formed_at=now(), purpose='Outcome-blind qualification of independently proposed historical co-limit network',
        window=WINDOW, minimum_common_observations=MIN_JOINT, minimum_joint_caps=MIN_CO_LIMIT, max_peers=MAX_PEERS,
        fields=FIELDS, train_and_validation_minimum_rows_with_observed_peers=100,
        mechanism='Historical co-occurrence proxy; no verified theme, absorption or quantitative-fund identity',
        new_model_fits=0, pending='Actual fitting requires subsequent joint agreement and a separate frozen comparison protocol', bindings=bindings))
    keys = pd.read_csv(COHORT, usecols=['signal_date', 'code', 'stage'], dtype={'code': str})
    keys = keys[keys.stage.eq(1)][['signal_date', 'code']].reset_index(drop=True)
    _, bars, sessions, _ = base.build_dataset()
    bars = bars[BAR_FIELDS].copy()
    rows, peers = build_features(bars, sessions, keys, diagnostics=True)
    rows.to_csv(OUT/'COMPLETE_FIRST_BOARD_CAUSAL_NETWORK_NO_LABELS.csv', index=False, encoding='utf-8-sig')
    write('PAST_SELECTED_PEERS_DIAGNOSTIC.json', peers)
    checks = []
    for cutoff in ['2025-06-30', '2025-12-31']:
        use = keys.signal_date.le(cutoff)
        prefix_keys = keys[use].reset_index(drop=True)
        truncated, _ = build_features(bars[bars.date.le(cutoff)], [d for d in sessions if d <= cutoff], prefix_keys)
        expected = rows[use].reset_index(drop=True)
        np.testing.assert_allclose(truncated[FIELDS], expected[FIELDS], atol=1e-11, rtol=1e-11, equal_nan=True)
        checks.append(dict(cutoff=cutoff, rows=len(truncated), all8fields_equal=True))
    periods = {}
    for name, condition in [('development', rows.signal_date.lt('2025-01-01')),
                            ('validation', rows.signal_date.between('2025-01-01', '2025-06-30')),
                            ('evaluation', rows.signal_date.ge('2025-07-01'))]:
        subset = rows[condition]
        periods[name] = dict(rows=len(subset), signal_dates=subset.signal_date.nunique(),
            rows_with_current_observed_peers=int(subset.co_limit_peer_limit_rate.notna().sum()),
            rows_with_no_historical_qualified_peers=int(subset.co_limit_peer_n.eq(0).sum()),
            coverage=subset[FIELDS].notna().mean().to_dict())
    passed = all(periods[name]['rows_with_current_observed_peers'] >= 100 for name in ['development', 'validation'])
    for path, digest in bindings.items():
        assert sha(path) == digest, path
    write('qualification.json', dict(formed_at=now(), complete_first_board_rows=len(rows), prefix_checks=checks,
         periods=periods, information_coverage_gate_passed=passed, fits=0,
         source_checks='Current bound retrospective prices; not an independent quote source',
         next_action='Gemini review and joint protocol before one fixed fit' if passed else 'No fit; all-unknown or too-sparse information fails fixed qualification'))
    print(json.dumps(clean(dict(coverage_gate=passed, periods=periods)), ensure_ascii=True), flush=True)


if __name__ == '__main__':
    run()
