"""Read-only evidence audit of the proposed timing study; no model or rule fitting."""
from pathlib import Path
import hashlib
import json
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'research/execution_upgrade_audit_20261029_corrected/complete13149_original_scores_and_screen.csv'


def counts(rows):
    y = rows.reach4_LABEL_ONLY
    return dict(rows=len(rows), positives=int(y.eq(1).sum()), negatives=int(y.eq(0).sum()),
                unknown=int(y.isna().sum()), dates=int(rows.signal_date.nunique()))


def interval(a, b, dates):
    aa = a.groupby('signal_date').reach4_LABEL_ONLY.agg(['sum', 'count']).reindex(dates, fill_value=0)
    bb = b.groupby('signal_date').reach4_LABEL_ONLY.agg(['sum', 'count']).reindex(dates, fill_value=0)
    w = pd.DataFrame(dict(ah=aa['sum'], an=aa['count'], bh=bb['sum'], bn=bb['count']))
    w.index = pd.to_datetime(w.index).to_period('W')
    v = w.groupby(level=0).sum().to_numpy(dtype=float)
    rng, samples, n = np.random.default_rng(42), [], len(v)
    for _ in range(2000):
        starts = rng.integers(0, n, size=math.ceil(n / 5))
        ix = np.concatenate([(s + np.arange(5)) % n for s in starts])[:n]
        ah, an, bh, bn = v[ix].sum(axis=0)
        samples.append(ah / an - bh / bn)
    return dict(delta=float(aa['sum'].sum()/aa['count'].sum()-bb['sum'].sum()/bb['count'].sum()),
                ci95=np.quantile(samples, [.025, .975]).tolist(), replicates=2000, seed=42,
                circular_block_weeks=5, interpretation='Historical descriptive comparison; not new blind validation')


def main():
    target = OUT / 'PROPOSAL_REVIEW.json'
    if target.exists():
        raise FileExistsError('Preserve existing review')
    d = pd.read_csv(SOURCE, dtype={'code': str}, float_precision='round_trip', low_memory=False)
    d = d.sort_values(['signal_date', 'model_score', 'code'], ascending=[True, False, True])
    z = d[d.original_screen_allowed.eq(True)].copy()
    z['allowed_rank'] = z.groupby('signal_date').cumcount()+1
    beyond = z[z.allowed_rank.gt(10)]
    misses = beyond[beyond.reach4_LABEL_ONLY.eq(1)]
    size = z.groupby('signal_date').size()
    key = lambda x: set(zip(x.signal_date, x.code))
    missed_keys, all_keys = key(misses), key(z)
    summaries = {}
    for name, rows in [('top3_four', z[z.allowed_rank.le(3) & z.reach4_LABEL_ONLY.eq(1)]),
                       ('beyond10_four', misses), ('beyond10_nonfour', beyond[beyond.reach4_LABEL_ONLY.eq(0)])]:
        summaries[name] = dict(counts(rows), turnover_median=float(rows.turnover.median()),
            amount_median_cny=float(rows.amount.median()),
            market_positive_fraction_median=float(rows.market_positive_fraction.median()))
    minute_inventory = []
    for relative, resolution in [
        ('data/v287/september_five_minute_20261002_v1/five_minute_bars_UNRECONCILED.csv', 5),
        ('data/v289/full_august_five_minute_20261002_v1/five_minute_bars_UNRECONCILED.csv', 5),
        ('data/v290/full_long_interval_20261002_v1/thirty_minute_bars_UNRECONCILED.csv', 30)]:
        p = ROOT / relative
        keys = set()
        for chunk in pd.read_csv(p, usecols=['code', 'minute'], dtype=str, chunksize=200000):
            keys.update(zip(chunk.minute.str[:10], chunk.code))
        minute_inventory.append(dict(path=relative, resolution_minutes=resolution,
            raw_missed82_date_code_overlap=len(keys & missed_keys),
            raw_allowed7254_date_code_overlap=len(keys & all_keys),
            status='RAW_UNRECONCILED_OVERLAP_ONLY_NOT_1_MINUTE_OR_ORDER_QUEUE_EVIDENCE'))
    current_root = ROOT / 'results/daily_review_20261028T1020Z'
    receipt = json.loads((current_root/'minutes/capture_receipt.json').read_text(encoding='utf-8'))
    records = [r for r in receipt['records'] if r.get('status') == 'EXACT_DATE_MINUTE_POINTS_SINGLE_SOURCE']
    facts = json.loads((current_root/'facts/RESOLVED_FACTS_v2_SCORES_MASKED.json').read_text(encoding='utf-8'))
    first = {r['code'] for r in facts['rows'] if r.get('kind') == 'limit_up' and r.get('boards') == 1}
    captured = {r['symbol'][2:] for r in records}
    current_keys = {(r['source_date'][:4]+'-'+r['source_date'][4:6]+'-'+r['source_date'][6:], r['symbol'][2:]) for r in records}
    report = dict(recorded_at=datetime.now(timezone.utc).isoformat(), status='DESIGN_REQUIRES_CORRECTION_NO_NEW_FIT',
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), fits=0, new_strategy_tested=False,
        full_evaluation=counts(d), allowed_evaluation=counts(z), beyond_top10=counts(beyond),
        missed_paths=counts(misses), missed_distinct_codes=int(misses.code.nunique()),
        missed_date_range=[misses.signal_date.min(), misses.signal_date.max()],
        pool_size=dict(min=int(size.min()), max=int(size.max()), median=float(size.median()),
                       dates_below15=int(size.lt(15).sum()), dates_above35=int(size.gt(35).sum())),
        beyond_rank35=counts(z[z.allowed_rank.gt(35)]), observed_medians_not_causal=summaries,
        screen_precision_difference=interval(z[z.allowed_rank.le(3)], d.groupby('signal_date').head(3), sorted(d.signal_date.unique())),
        historical_coarse_bar_overlap=minute_inventory,
        october28_minutes=dict(date=receipt['date'], records=len(records), first_boards=len(first),
            first_boards_covered=len(first & captured), missed82_exact_date_code_overlap=len(current_keys & missed_keys)),
        corrections=[
            '82 four-board paths are future-selected labels (79 unique codes), not an ex-ante candidate universe; reach2=100% is definitional.',
            'Evaluate all 7254 allowed candidates across 286 dates, retaining 160 unknown outcomes in the 4397 below-top10 candidates.',
            'Do not impose a 35-candidate ceiling or infer causality from winner-only turnover medians.',
            'October28 minute evidence cannot substitute for July2025-August2026 signal-day evidence; coarse bars do not identify 1/3-minute events or queue changes.',
            'Touching limit-up is a trigger, not a confirmed fill; explicit causal exit rules, queue uncertainty and unavailable exits remain necessary.',
            'Original scenario total return is -52.71%; maximum drawdown is -53.42%.',
            'Freeze feature definitions and decision rules before comparison; no old failed arm, production score, source prediction or task changes.'
        ])
    with target.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
