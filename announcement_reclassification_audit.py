"""Audit new title rules against an immutable announcement archive, never rescore trades."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import collect_context_v3 as context


def local_time(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError('missing timestamp')
    return stamp.tz_localize('Asia/Shanghai') if stamp.tzinfo is None else stamp.tz_convert('Asia/Shanghai')


def compare(frame, signal_date, asof):
    cutoff = local_time(asof)
    signal = local_time(signal_date).normalize()
    if signal > cutoff.normalize():
        raise ValueError('signal date is after audit cutoff')
    differences, unavailable = [], []
    for row in frame.to_dict('records'):
        try:
            public = local_time(row['published_at'])
            captured = local_time(row['source_timestamp'])
        except (ValueError, TypeError):
            unavailable.append(dict(code=row['code'], reason='INVALID_TIMESTAMP'))
            continue
        # Do not use retrospective receipt as evidence of earlier availability.
        known = max(public, captured) <= cutoff
        label, hard, _ = context.classify_announcement(row['title'], row['announcement_category'])
        label, hard, window = context.apply_risk_window(
            label, hard, row['title'], (signal-public.normalize()).days)
        if not known:
            unavailable.append(dict(code=row['code'], title=row['title'], reason='NOT_KNOWN_AT_CUTOFF'))
        if (label, hard) != (row['risk_type'], int(row['risk_hard'])):
            differences.append(dict(code=row['code'], title=row['title'], source_url=row['url'],
                                    old_label=row['risk_type'], old_hard=int(row['risk_hard']),
                                    new_label=label, new_hard=hard,
                                    risk_window=window, known_at_cutoff=known))
    return differences, unavailable


def run(source, output, manifest):
    source, output = Path(source), Path(output)
    manifest = Path(manifest)
    meta = json.loads(manifest.read_text(encoding='utf-8'))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if meta['files']['announcements']['sha256'] != source_hash:
        raise ValueError('archive does not match manifest hash')
    frame = pd.read_csv(source, dtype={'code': str}).fillna('')
    differences, unavailable = compare(frame, meta['signal_date'], meta['collected_at'])
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(differences, columns=['code','title','source_url','old_label','old_hard','new_label','new_hard','risk_window','known_at_cutoff']).to_csv(
        output/'changes.csv',index=False,encoding='utf-8-sig')
    report=dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), source=str(source.resolve()),
                source_sha256=source_hash, version='2-window-aligned',
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                signal_date=meta['signal_date'], cutoff=meta['collected_at'],
                classifier_sha256=hashlib.sha256(Path(context.__file__).read_bytes()).hexdigest(),
                archive_rows=len(frame), changed_rows=len(differences),
                unavailable_rows=len(unavailable), unavailable=unavailable,
                changes=differences, scope='Title-only diagnostic comparison using archived signal-date risk windows and conservative publication/receipt availability. No historical decisions or forecasts modified; no full-text clearance or ENTER authorization.')
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--manifest',required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.source,args.output,args.manifest),ensure_ascii=False,indent=2))
