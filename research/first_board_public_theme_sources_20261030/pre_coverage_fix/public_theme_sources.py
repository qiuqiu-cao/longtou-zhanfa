"""Public text -> dated, source-bound themes; never a probability or buy signal.

THS timeline alignment is checked for a bijection and can then be independently
confirmed against each visible detail link. Event date is NOT publication time.
Forum memberships require reviewed exact spans, never matching name characters.
"""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import unquote_plus

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CHINA = timezone(timedelta(hours=8))
KEYS = ['signal_date', 'code']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unmark(text):
    return re.sub(r'【\d+†([^†】]+)(?:†[^】]+)?】', r'\1', text)


def read_archive(path):
    path = Path(path)
    obj = json.loads(path.read_text(encoding='utf-8'))
    result = obj.get('result', {})
    text = result.get('value', '') if result.get('status') == 'fulfilled' else ''
    if not isinstance(text, str):
        raise ValueError('unsupported web response')
    return obj, text


def page_lines(text):
    return [(int(m[1]), m[2]) for m in re.finditer(r'^L(\d+):\s?(.*)$', text, re.M)]


def instant(value):
    value = value.replace(' UTC', '+00:00').replace('Z', '+00:00')
    d = datetime.fromisoformat(value)
    if d.tzinfo is None:
        raise ValueError('timezone required')
    return d


def split_themes(value):
    # Preserve exact reported labels; no synonym/phonetic auto merging.
    return sorted(set(x.strip() for x in re.split(r'[+＋]', unicodedata.normalize('NFKC', value)) if x.strip()))


def parse_ths(path):
    obj, text = read_archive(path)
    code = obj['code']
    info = {'code': code, 'url': obj['requested_url'], 'path': str(Path(path).relative_to(ROOT)),
            'sha256': sha(path), 'status': 'SOURCE_UNAVAILABLE', 'event_count': 0}
    if '历史涨停时间' not in text:
        return [], info
    lines = page_lines(text)
    heading = [unmark(t) for _, t in lines if t.startswith('### ')]
    if not heading or not re.search(r'\(' + re.escape(code) + r'\)', heading[0]):
        return [], {**info, 'status': 'IDENTITY_MISMATCH'}
    dates = [(n, m[1]) for n, t in lines if (m := re.fullmatch(r'\s*\* [A-Z] (\d{4}\.\d{2}\.\d{2})\s*', t))]
    if not dates or len({d for _, d in dates}) != len(dates):
        return [], {**info, 'status': 'TIMELINE_INVALID'}
    for _, d in dates:
        datetime.strptime(d, '%Y.%m.%d')
    blocks, pending = [], []
    for n, t in lines:
        if n <= dates[-1][0]:
            continue
        link = re.search(r'【(\d+)†查看详情>>', t)
        if link:
            if not pending:
                return [], {**info, 'status': 'EMPTY_DETAIL_BLOCK'}
            blocks.append((pending, int(link[1])))
            pending = []
        elif t.strip() and t.strip() != '[Input]':
            pending.append((n, t))
    if len(dates) != len(blocks) or pending:
        return [], {**info, 'status': 'TIMELINE_DETAIL_COUNT_MISMATCH'}
    citation = re.search(r'cite([^]+)', text)
    rows = []
    for (_, d), (block, link) in zip(dates, blocks):
        title = unmark(block[0][1]).strip()
        if title.startswith(('1、', '行业原因', '公司原因')) or len(block) < 2:
            return [], {**info, 'status': 'DETAIL_TITLE_AMBIGUOUS'}
        rows.append({'signal_date': d.replace('.', '-'), 'code': code,
            'theme_raw': title, 'themes': split_themes(title),
            'source_url': info['url'], 'source_path': info['path'], 'source_sha256': info['sha256'],
            'source_span': '\n'.join(f'L{n}: {t}' for n, t in block),
            'source_ref': citation[1] if citation else None, 'detail_link_id': link,
            'source_received_by': obj['response_saved_after'],
            'publication_time': None, 'version_status': 'LATE_RETRIEVED_VERSION_UNCERTIFIED',
            'event_date_binding': 'ORDERED_TIMELINE_PENDING_DETAIL_LINK_CHECK',
            'is_provider_interpretation': True, 'actual_historical_available': False})
    return rows, {**info, 'status': 'PARSED_ORDERED_TIMELINE', 'event_count': len(rows)}


def validate_detail_link(row, verification):
    """A returned detail URL must name exactly this stock and event day."""
    expected = row['code'] + ' ' + row['signal_date'].replace('-', '') + ' 涨停原因'
    if verification.get('source_ref') != row['source_ref'] or verification.get('detail_link_id') != row['detail_link_id']:
        return False
    r = verification.get('result', {})
    if r.get('status') != 'fulfilled' or not isinstance(r.get('value'), str):
        return False
    values = re.findall(r'https://www\.iwencai\.com/[^\s)\";]+', r['value'])
    queries = [unquote_plus(v).split('&w=', 1)[1].split('&', 1)[0] for v in values if '&w=' in v]
    return bool(queries) and all(q == expected for q in queries)


def clock_gate(claim, cutoff):
    """No event date, body date, or nominal source time can impersonate acquisition."""
    end = instant(cutoff)
    acquired = instant(claim['source_received_by'])
    pub = claim.get('publication_time')
    pub_status = None if not pub else instant(pub) <= end
    archived = acquired <= end
    return {'published_by_cutoff': pub_status, 'acquired_by_cutoff': archived,
            'actual_historical_available': bool(archived and pub_status is True
                and claim.get('publication_timezone_verified') is True
                and claim.get('content_time_conflict') is not True)}


def reviewed_memberships(spec, path):
    """The reviewer selects a paragraph, not a winner; keep every named member."""
    obj, text = read_archive(path)
    if spec['source_sha256'] != sha(path):
        raise ValueError('source changed')
    span = spec['source_span']
    if not span or span not in text:
        raise ValueError('exact source span missing')
    if spec['body_scope'] != 'original_body':
        raise ValueError('comments need their own source and publication record')
    members = spec['members']
    codes = [v['code'] for v in members if v.get('code')]
    if len(set(codes)) != len(codes):
        raise ValueError('duplicate code within source group')
    result = []
    for m in members:
        if m['source_name'] not in unmark(span):
            raise ValueError('member not in reviewed source span')
        if m.get('code') and (len(m['code']) != 6 or not m['code'].isdigit()):
            raise ValueError('invalid stock code')
        result.append({**m, 'theme_id': spec['theme_id'], 'event_date': spec['event_date'],
            'displayed_publication_time': spec['displayed_publication_time'],
            'publication_time': spec.get('publication_time'),
            'publication_timezone_verified': spec.get('publication_timezone_verified', False),
            'source_received_by': obj['response_saved_after'], 'author': spec['author'],
            'source_path': str(Path(path).relative_to(ROOT)), 'source_sha256': sha(path),
            'source_url': obj['url'], 'source_span': span, 'body_scope': 'original_body',
            'version_status': 'LATE_RETRIEVED_VERSION_UNCERTIFIED',
            'role': 'unknown', 'relation': spec['relation'],
            'complete_within_reviewed_group': spec['complete_within_reviewed_group'],
            'complete_market_theme': False, 'independent_identity_gold_label': False,
            'reviewed_at': spec['reviewed_at'], 'counterevidence': spec['counterevidence']})
    return result


def attach_to_pool(pool, events, group_members):
    """Left join, preserve all scores/slots/unknowns; exact event date only.

The result is research context. Late text has no live eligibility and does not
silently alter a frozen scorer. No ffill of forum claims across a market cycle.
"""
    if pool.duplicated(KEYS).any():
        raise ValueError('duplicate pool key')
    if any('LABEL' in c or c in {'promotion_next', 'reach4', 'label_end'} for c in pool):
        raise ValueError('outcome columns are not permitted in source input')
    events_by_key, groups_by_key = {}, {}
    for e in events:
        if e.get('event_date_binding') != 'DETAIL_LINK_DATE_VERIFIED':
            continue
        events_by_key.setdefault((e['signal_date'], e['code']), []).append(e)
    for g in group_members:
        if g.get('code'):
            groups_by_key.setdefault((g['event_date'], g['code']), []).append(g)
    new = []
    for r in pool.itertuples(index=False):
        k = (r.signal_date, r.code)
        e, g = events_by_key.get(k, []), groups_by_key.get(k, [])
        new.append({'public_ths_event_count': len(e), 'public_reviewed_group_count': len(g),
            'public_theme_context_status': 'LATE_RESEARCH_CONTEXT' if e or g else 'UNKNOWN',
            'public_ths_themes_json': json.dumps(sorted({t for x in e for t in x['themes']}), ensure_ascii=False),
            'public_group_ids_json': json.dumps(sorted({x['theme_id'] for x in g}), ensure_ascii=False),
            'public_source_refs_json': json.dumps([{'url': x['source_url'], 'sha256': x['source_sha256']}
                for x in e + g], ensure_ascii=False),
            'public_context_actual_historical_available': False})
    result = pool.copy()
    for col in new[0] if new else []:
        result[col] = [x[col] for x in new]
    return result


def as_provider(events):
    """Compatible input to the frozen dynamic_features.build, not deployed weights."""
    rows = []
    for e in events:
        if e.get('event_date_binding') != 'DETAIL_LINK_DATE_VERIFIED':
            continue
        rows.append({'signal_date': e['signal_date'], 'code': e['code'],
            'themes': ['THS:' + t for t in e['themes']], 'first_seal_minutes': float('nan'),
            'source_path': e['source_path'], 'source_sha256': e['source_sha256'],
            'source_record_pointer': e['source_span'].splitlines()[0].split(':')[0],
            'source_received_at': e['source_received_by'],
            'source_status': 'RETROSPECTIVE_THS_INTERPRETATION_NOT_PIT_CERTIFIED'})
    result = pd.DataFrame(rows)
    if len(result) and result.duplicated(KEYS).any():
        raise ValueError('multiple source versions need explicit reconciliation')
    return result
