"""Bound cached original pool observations to first boards, without reading targets."""
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import csv
import hashlib
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import relay_information_qualification as qualification

OUT = Path(__file__).resolve().parent
OLD = ROOT / 'research/model_upgrade_validation/remediation_20261014/data_prediction/actual_local_qualification_v2'
COHORT = ROOT / 'research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(name, data):
    if (OUT / name).exists():
        assert read(OUT / name) == data, name
        return
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def cents(value):
    return int((Decimal(str(value)) * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def main():
    protocol = read(OUT / 'protocol_BEFORE_INTERSECTION.json')
    for name, expected in protocol['inputs'].items():
        assert sha(name) == expected, name
    sources = read(OLD / 'all_source_qualification.json')
    checks = []
    valid_paths = set()
    for source in sources:
        path = Path(source['path'])
        raw_ok = sha(path) == source['sha256']
        receipt = source.get('receipt_path')
        receipt_ok = receipt is None or sha(receipt) == source['receipt_sha256']
        checks.append({'path': str(path), 'raw_hash_matches': raw_ok,
                       'receipt_path': receipt, 'receipt_hash_matches': receipt_ok,
                       'reused_status': source['status']})
        if raw_ok and receipt_ok:
            valid_paths.add(str(path))
    assert all(x['raw_hash_matches'] and x['receipt_hash_matches'] for x in checks)
    write('cached_source_immutability_checks.json', checks)

    # usecols is an explicit outcome firewall for this qualification step.
    cohort = pd.read_csv(COHORT, dtype={'code': str},
                         usecols=['signal_date', 'code', 'stage', 'close'])
    cohort = cohort[cohort.stage.eq(1)].copy()
    assert len(cohort) == 28503 and not cohort.duplicated(['signal_date', 'code']).any()
    original = cohort.set_index(['signal_date', 'code'])['close'].to_dict()
    events = read(OLD / 'all_provider_event_rows.json')
    matches = [x for x in events if (x['date'], x['code']) in original]
    assert len(matches) == 122
    raw_tables = {}
    records = []
    field_columns = {v: k for k, v in qualification.MAP.items() if k not in
                     {'c', 'n', 'lbc', 'fbt', 'lbt', 'zbc', 'fund', 'ltsz'}}
    for event in matches:
        path = event['provider_source_path']
        assert path in valid_paths and sha(path) == event['provider_source_sha256']
        if path not in raw_tables:
            with Path(path).open(encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                rows = [(reader.line_num, row) for row in reader]
            codes = [row[field_columns['code']].zfill(6) for _, row in rows]
            assert len(codes) == len(set(codes))
            raw_tables[path] = {row[field_columns['code']].zfill(6): (line, row)
                                for line, row in rows}
        line, row = raw_tables[path][event['code']]
        close = original[event['date'], event['code']]
        price_matches = cents(row['最新价']) == cents(close)
        fields_equal = all(float(row[field_columns[k]]) == event[k]
                           for k in ('boards', 'first_seal', 'last_seal', 'bomb_count', 'sealed_fund'))
        assert fields_equal
        source = next(x for x in sources if x['path'] == path)
        event_qualified = (price_matches and event['boards'] == 1 and
                           event['reported_clock_contract_valid'] and
                           source['status'] == 'SOURCE_BOUND_PROVIDER_VALUES_DIAGNOSTIC')
        records.append({
            **event, 'signal_date': event['date'], 'cohort_close': close,
            'reported_price': float(row['最新价']), 'exact_cent_close_matches': price_matches,
            'raw_line_number': line,
            'raw_field_columns': {k: field_columns[k] for k in
                                  ('code', 'name', 'boards', 'first_seal', 'last_seal', 'bomb_count', 'sealed_fund')},
            'raw_price_column': '最新价', 'receipt_path': source['receipt_path'],
            'receipt_sha256': source['receipt_sha256'],
            'provider_event_qualified_for_diagnostic': bool(event_qualified),
            'strict_original_signal_time_qualified': False,
            'fund_actor_identity_qualified': False,
            'source_scope': 'PROVIDER_WHOLE_POOL_COUNT_MATCHES_RECEIPT_NOT_EXCHANGE_UNIVERSE_CERTIFICATION',
            'first_board_not_selected_by_future_relay': True,
        })
    write('raw_bound_first_board_observations.json', records)
    data = pd.DataFrame(records)
    keep = ['signal_date', 'code', 'first_seal', 'last_seal', 'bomb_count', 'sealed_fund',
            'provider_event_qualified_for_diagnostic', 'strict_original_signal_time_qualified',
            'fund_actor_identity_qualified', 'provider_source_path', 'provider_source_sha256',
            'raw_line_number', 'available_at', 'market_snapshot_at', 'capture_scope']
    projection = cohort.merge(data[keep], on=['signal_date', 'code'], how='left', validate='one_to_one')
    projection['provider_observation_present'] = projection.provider_source_path.notna()
    # Missing fields are not zero; missing observation is not proof of absent seal activity.
    projection_path = OUT / 'COMPLETE_FIRST_BOARD_PROVIDER_SEAL_PROJECTION_NO_TARGETS.csv'
    projection_bytes = projection.to_csv(index=False).encode('utf-8-sig')
    if projection_path.exists():
        assert projection_path.read_bytes() == projection_bytes
    else:
        with projection_path.open('xb') as f:
            f.write(projection_bytes)

    new_files = read(OUT / 'initial_inventory_and_overlap_NO_TARGETS.json')['new_files']
    new_reports = []
    assert len(new_files) <= 32
    for name in new_files:
        raw = read(name)
        body = raw.get('data') or {}
        pool = body.get('pool') or []
        new_reports.append({
            'path': name, 'sha256': sha(name), 'qdate': body.get('qdate'),
            'rc': raw.get('rc'), 'tc': body.get('tc'), 'pool_rows': len(pool),
            'declared_count_matches': body.get('tc') == len(pool),
            'unique_code_count': len({r.get('c') for r in pool}),
            'first_board_report_count': sum(r.get('lbc') == 1 for r in pool),
            'original_first_board_cohort_intersection': 0,
            'qualification': 'OUTSIDE_ORIGINAL_COHORT_AND_ORIGINAL_CALENDAR_RANGE_NOT_JUDGED_NON_SESSION',
            'note': 'Original cohort ends 2026-08-31. Current raw source cannot fill its missing history.',
            'first_row_identity_pointer': '/data/pool/0/c',
            'first_row_seal_pointers': ['/data/pool/0/fbt', '/data/pool/0/lbt',
                                       '/data/pool/0/zbc', '/data/pool/0/fund'],
        })
    write('new_source_bounded_inspection.json', new_reports)
    periods = [('development_pre2025', projection.signal_date.lt('2025-01-01')),
               ('validation_2025H1', projection.signal_date.between('2025-01-01', '2025-06-30')),
               ('evaluation_2025H2_2026', projection.signal_date.ge('2025-07-01'))]
    coverage = {}
    for name, mask in periods:
        sub = projection.loc[mask]
        coverage[name] = {'full_first_board_rows': len(sub),
                          'provider_observations': int(sub.provider_observation_present.sum()),
                          'qualified_provider_diagnostic': int(sub.provider_event_qualified_for_diagnostic.eq(True).sum())}
    report = {
        'formed_at': datetime.now(timezone.utc).isoformat(),
        'cohort_rows': len(cohort), 'cohort_dates': cohort.signal_date.nunique(),
        'cached_source_files_unchanged': len(checks), 'new_source_files_inspected': len(new_reports),
        'observations': len(data), 'qualified_provider_diagnostic': int(data.provider_event_qualified_for_diagnostic.sum()),
        'price_mismatch_count': int((~data.exact_cent_close_matches).sum()),
        'dates': data.groupby('signal_date').size().to_dict(), 'periods': coverage,
        'no_provider_observation_rows': int((~projection.provider_observation_present).sum()),
        'latest_monthly_training_cutoff': '2026-07-31',
        'provider_rows_before_latest_training_cutoff': int(data.signal_date.le('2026-07-31').sum()),
        'strict_original_signal_time_qualified': 0, 'fund_actor_identity_qualified': 0,
        'model_fits': 0, 'new_forward_results': 0, 'trades': 0, 'goal_60_passed': False,
        'decision': 'NO_SEAL_FEATURE_TRAINING: zero development/validation/monthly-training coverage. All other authorized historical research remains available.',
        'next_action': 'Qualify a causal past co-limit-up network on complete existing daily bars, separately from failed price correlation and legacy-industry arms.',
        'script_sha256': sha(__file__), 'protocol_sha256': sha(OUT / 'protocol_BEFORE_INTERSECTION.json'),
    }
    write('report.json', report)
    print(json.dumps(report, ensure_ascii=True))


if __name__ == '__main__':
    main()
