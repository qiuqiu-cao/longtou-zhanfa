"""Bound historical provider themes. These are retrospective, not PIT-certified."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'research/first_board_joint_design_20261020T185417Z'
CHINA = timezone(timedelta(hours=8))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def topics(primary, concepts):
    """Exact normalized provider terms only; no name characters or hindsight aliases."""
    values = [('P', primary)] + [('C', x) for x in re.split(r'[、,，+＋;；/|]', str(concepts or ''))]
    clean = {(kind, unicodedata.normalize('NFKC', str(x)).strip()) for kind, x in values if x is not None}
    return sorted(kind + ':' + x for kind, x in clean if x and x not in {'-', '--', 'None', 'nan'})


def seal_time(value, day):
    try:
        t = datetime.fromtimestamp(float(value), CHINA)
        minutes = t.hour * 60 + t.minute + t.second / 60
        if t.date().isoformat() != day:
            return float('nan')
        if 565 <= minutes <= 690: return minutes - 570
        if 780 <= minutes <= 900: return minutes - 660
    except (ValueError, OverflowError, TypeError, OSError):
        pass
    return float('nan')


def load_provider():
    receipt_paths = [ARCHIVE / 'ALL668_SOURCE_RECEIPTS.json',
                     ARCHIVE / 'RESUME_RECEIPTS_APPEND_ONLY_20261021.jsonl']
    records = json.loads(receipt_paths[0].read_text(encoding='utf-8'))
    records += [json.loads(line) for line in receipt_paths[1].read_text(encoding='utf-8').splitlines() if line.strip()]
    valid = {r['date']: r for r in records if r.get('qualified_date_envelope')}
    rows, bindings = [], {str(p.relative_to(ROOT)): sha(p) for p in receipt_paths}
    for day, receipt in sorted(valid.items()):
        path = Path(receipt['source_path'])
        digest = sha(path)
        if digest != receipt['sha256']: raise ValueError(f'source hash changed: {day}')
        obj = json.loads(path.read_bytes())
        if str(obj['errcode']) != '0' or obj['info'][1] != day:
            raise ValueError(f'source date mismatch: {day}')
        bindings[str(path.relative_to(ROOT))] = digest
        seen = set()
        for index, row in enumerate(obj['info'][0]):
            code = str(row[0])
            if code in seen or len(code) != 6 or not code.isdigit() or len(row) < 13:
                raise ValueError(f'duplicate or invalid raw row: {day} {code}')
            seen.add(code)
            rows.append({'signal_date': day, 'code': code,
                'provider_primary_theme': str(row[5] or ''),
                'provider_concepts': str(row[12] or ''), 'themes': topics(row[5], row[12]),
                'first_seal_minutes': seal_time(row[4], day),
                'source_path': str(path.relative_to(ROOT)), 'source_sha256': digest,
                'source_record_pointer': f'/info/0/{index}',
                'source_received_at': receipt.get('received_at'),
                'theme_version_verified_at_signal': False,
                'source_status': 'RETROSPECTIVE_PROVIDER_THEME_NOT_POINT_IN_TIME_CERTIFIED'})
    return pd.DataFrame(rows), bindings


def provider_inventory(frame):
    return {'days': int(frame.signal_date.nunique()), 'rows': len(frame),
        'rows_with_theme': int(frame.themes.map(bool).sum()),
        'primary_theme_count': int(frame.provider_primary_theme.replace('', pd.NA).nunique()),
        'exact_topic_count': len(set(t for row in frame.themes for t in row)),
        'point_in_time_theme_qualified_rows': int(frame.theme_version_verified_at_signal.sum()),
        'all_rows_are_later_retrieved_provider_descriptions': True}
