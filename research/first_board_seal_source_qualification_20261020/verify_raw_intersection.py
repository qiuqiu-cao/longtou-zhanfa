from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
import csv
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
read = lambda path: json.loads(Path(path).read_text(encoding='utf-8-sig'))
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()

expected = pd.read_csv(ROOT / 'research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv',
                       usecols=['signal_date', 'code', 'stage', 'close'], dtype={'code': str})
expected = expected[expected.stage.eq(1)].reset_index(drop=True)
projection = pd.read_csv(OUT / 'COMPLETE_FIRST_BOARD_PROVIDER_SEAL_PROJECTION_NO_TARGETS.csv', dtype={'code': str})
columns = ['signal_date', 'code', 'stage', 'close']
pd.testing.assert_frame_equal(projection[columns], expected[columns], check_dtype=False)
observations = read(OUT / 'raw_bound_first_board_observations.json')
checks = []
for path in sorted({x['provider_source_path'] for x in observations}):
    raw = Path(path)
    lines = raw.read_text(encoding='utf-8-sig').splitlines()
    header = next(csv.reader([lines[0]]))
    rows = list(csv.DictReader(lines))
    manifest = read(raw.parent / 'manifest.json')
    assert len(rows) == manifest['datasets']['limit_up']['rows']
    day = pd.Timestamp(manifest['date']).date().isoformat()
    for record in [v for v in observations if v['provider_source_path'] == path]:
        row = dict(zip(header, next(csv.reader([lines[record['raw_line_number'] - 1]]))))
        assert row['代码'].zfill(6) == record['code']
        assert float(row['连板数']) == 1 and day == record['signal_date']
        for dest, source in [('first_seal', '首次封板时间'), ('last_seal', '最后封板时间'),
                             ('bomb_count', '炸板次数'), ('sealed_fund', '封板资金')]:
            assert float(row[source]) == record[dest]
        cent = lambda value: int((Decimal(str(value)) * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        assert cent(row['最新价']) == cent(record['cohort_close'])
    checks.append({'path': path, 'sha256': sha(path), 'raw_rows': len(rows),
                   'manifest_rows_match': True, 'direct_lines_fields_and_cents_pass': True})
missing = projection.provider_observation_present.eq(False)
assert missing.sum() == 28381
assert projection.loc[missing, ['first_seal', 'last_seal', 'bomb_count', 'sealed_fund']].isna().all().all()
assert projection.loc[~missing, 'strict_original_signal_time_qualified'].eq(False).all()
assert not any('LABEL' in column for column in projection.columns)
for filename, expected_sha in read(OUT / 'failure_and_repair_append.json')['partial_outputs_preserved'].items():
    if filename.endswith(('.json', '.csv')):
        assert sha(OUT / filename) == expected_sha, filename
verification = {
    'formed_at': datetime.now(timezone.utc).isoformat(), 'status': 'PASS',
    'full_cohort_keys_prices_unchanged': len(projection), 'direct_raw_observations': len(observations),
    'sources': checks, 'missing_fields_remain_NaN': int(missing.sum()),
    'development_validation_monthly_training_coverage': 0,
    'strict_time_and_actor_identity_not_certified': True,
    'partial_outputs_from_index_failure_unchanged': True,
    'verifier_first_attempt': 'Column-order assertion mismatch in inline verifier before data comparison; corrected by explicitly selecting identical columns on both sides. No data or model changes.',
    'model_fits': 0, 'new_forward_results': 0,
    'artifacts': {f.name: sha(f) for f in OUT.iterdir() if f.suffix in ('.csv', '.json', '.py')},
}
with (OUT / 'verification.json').open('x', encoding='utf-8') as stream:
    json.dump(verification, stream, ensure_ascii=False, indent=2)
print(json.dumps({'status': 'PASS', 'observations': len(observations), 'missing': int(missing.sum())}))
