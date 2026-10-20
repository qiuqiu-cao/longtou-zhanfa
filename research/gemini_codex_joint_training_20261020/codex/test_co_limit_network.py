import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('network', HERE/'qualify_co_limit_network.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def toy():
    dates = pd.bdate_range('2024-01-01', periods=150).strftime('%Y-%m-%d').tolist()
    codes = [f'{i:06d}' for i in range(1, 13)]
    rows = []
    for date in dates:
        for code in codes:
            t = dates.index(date)
            cap = t in [20, 30] or (t == 2 and code in codes[:3])
            rows.append(dict(date=date, code=code, eligible=True, valid_price=True, regular_price=True,
                             amount=100., volume=10., limit=cap, first=cap, failed=False,
                             day_return=.1 if cap else 0., previous_height=0., previous_limit=False))
    bars = pd.DataFrame(rows)
    keys = pd.DataFrame([dict(signal_date=dates[140], code=codes[0])])
    return dates, codes, bars, keys


class CausalNetwork(unittest.TestCase):
    def test_expired_edges_cannot_change_code_order_for_equal_recent_history(self):
        dates, codes, bars, keys = toy()
        _, peers = m.build_features(bars, dates, keys, diagnostics=True)
        self.assertEqual([p['code'] for p in peers[0]['selected_peers']], codes[1:9])

    def test_future_events_do_not_change_existing_context(self):
        dates, codes, bars, keys = toy()
        before, _ = m.build_features(bars, dates, keys)
        bars.loc[bars.date.gt(keys.iloc[0].signal_date), ['limit', 'first']] = True
        bars.loc[bars.date.gt(keys.iloc[0].signal_date), 'day_return'] = 99.
        after, _ = m.build_features(bars, dates, keys)
        pd.testing.assert_frame_equal(before, after)

    def test_current_missing_peers_do_not_become_zero_limit_rate(self):
        dates, codes, bars, keys = toy()
        bars = bars[~(bars.date.eq(keys.iloc[0].signal_date) & bars.code.ne(codes[0]))]
        result, _ = m.build_features(bars, dates, keys)
        self.assertEqual(result.iloc[0].co_limit_peer_n, 8)
        self.assertEqual(result.iloc[0].co_limit_current_observed_weight_fraction, 0.)
        self.assertTrue(np.isnan(result.iloc[0].co_limit_peer_limit_rate))


if __name__ == '__main__':
    unittest.main()
