"""Direct chronological event enumeration; independent of sliding graph updates."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import first_board_leader_model as base

OUT = Path(__file__).resolve().parent/'network_qualification'
FIELDS = ['co_limit_peer_n', 'co_limit_history_weight', 'co_limit_current_observed_weight_fraction',
          'co_limit_peer_limit_rate', 'co_limit_peer_first_rate', 'co_limit_peer_failed_fraction',
          'co_limit_peer_excess_return', 'co_limit_previous_high_survival']


def main():
    expected = pd.read_csv(OUT/'COMPLETE_FIRST_BOARD_CAUSAL_NETWORK_NO_LABELS.csv', dtype={'code': str}, float_precision='round_trip')
    peers = json.loads((OUT/'PAST_SELECTED_PEERS_DIAGNOSTIC.json').read_text(encoding='utf-8'))
    saved_peers = {(r['signal_date'], r['code']): r['selected_peers'] for r in peers}
    _, bars, sessions, _ = base.build_dataset()
    codes = sorted(bars.code.unique())
    dates = {d:i for i,d in enumerate(sessions)}
    indices = {c:i for i,c in enumerate(codes)}
    b = bars.copy()
    row = b.date.map(dates).to_numpy(dtype=int)
    col = b.code.map(indices).to_numpy(dtype=int)
    valid = b.eligible & b.valid_price & b.regular_price & b.amount.gt(0) & b.volume.gt(0)
    shape = len(sessions), len(codes)
    obs, cap, first, fail, prevhigh = [np.zeros(shape, dtype=bool) for _ in range(5)]
    for array, values in [(obs, valid), (cap, valid & b.limit), (first, valid & b['first']),
                          (fail, valid & b.failed), (prevhigh, b.previous_limit & b.previous_height.ge(3))]:
        array[row, col] = values.to_numpy(dtype=bool)
    returns = np.full(shape, np.nan)
    returns[row, col] = b.day_return.where(valid).to_numpy(dtype=float)
    market = b.day_return.where(valid).groupby(b.date).median().reindex(sessions).to_numpy(dtype=float)
    cap_count = cap.sum(axis=1)
    del bars, b
    feature_mismatches = []
    peer_mismatches = []
    checked_peer_edges = 0
    for i,r in enumerate(expected.itertuples()):
        t, own = dates[r.signal_date], indices[r.code]
        start = max(0, t-126)
        h = obs[start:t]
        values = np.full(8, np.nan)
        selected = []
        if len(h) >= 60 and h[:, own].sum() >= 60:
            prior_caps = np.flatnonzero(cap[start:t, own])+start
            if len(prior_caps):
                counts = cap[prior_caps].sum(axis=0)
                cands = np.flatnonzero(counts >= 2)
                cands = cands[cands != own]
                cands = cands[(h[:, own, None] & h[:, cands]).sum(axis=0) >= 60]
                if len(cands):
                    w = (cap[np.ix_(prior_caps, cands)]/(cap_count[prior_caps]-1)[:, None]).sum(axis=0)
                    # Record direct event arithmetic; no tolerance changes saved selection.
                    order = sorted(range(len(cands)), key=lambda j: (-w[j], codes[cands[j]]))[:8]
                    selected = cands[order]
                    weights = w[order]
            values[0] = len(selected)
            values[1] = weights.sum() if len(selected) else 0.
            if len(selected):
                observed = obs[t, selected]
                live = weights*observed
                denominator = live.sum()
                values[2] = denominator/weights.sum()
                if denominator > 0:
                    values[3] = sum(live*cap[t, selected])/denominator
                    values[4] = sum(live*first[t, selected])/denominator
                    values[5] = sum(live*fail[t, selected])/denominator
                    values[6] = sum(live*np.nan_to_num(returns[t, selected], nan=0.))/denominator-market[t]
                    high = prevhigh[t, selected] & observed
                    if high.any():
                        values[7] = sum(weights[high]*cap[t, selected][high])/sum(weights[high])
                got = saved_peers[(r.signal_date, r.code)]
                checked_peer_edges += len(got)
                if [x['code'] for x in got] != [codes[x] for x in selected]:
                    peer_mismatches.append(dict(signal_date=r.signal_date, code=r.code,
                        direct=[codes[x] for x in selected], saved=[x['code'] for x in got]))
        saved = expected.iloc[i][FIELDS].to_numpy(dtype=float)
        if not np.allclose(values, saved, atol=1e-11, rtol=1e-11, equal_nan=True):
            feature_mismatches.append(dict(signal_date=r.signal_date, code=r.code,
                 direct=[None if np.isnan(x) else float(x) for x in values],
                 saved=[None if np.isnan(x) else float(x) for x in saved]))
    result = dict(formed_at=datetime.now(timezone.utc).isoformat(),
                  status='PASS' if not feature_mismatches and not peer_mismatches else 'DIFFERENCES_REQUIRE_TECHNICAL_REVIEW_BEFORE_FIT',
                  complete_rows=len(expected), checked_selected_peer_edges=checked_peer_edges,
                  feature_mismatch_count=len(feature_mismatches), peer_selection_mismatch_count=len(peer_mismatches),
                  feature_mismatches=feature_mismatches, peer_mismatches=peer_mismatches, fits=0)
    with (OUT/'DIRECT_ENUMERATION_VERIFICATION.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ['feature_mismatches','peer_mismatches']}), flush=True)


if __name__ == '__main__':
    main()
