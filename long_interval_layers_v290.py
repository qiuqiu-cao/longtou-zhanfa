"""Read parent plus later recovery as a verified view; neither file is rewritten."""
import json
from pathlib import Path
import pandas as pd
import archive_long_interval_v290 as engine
from audit_long_interval_v290 import parse_source


def load(archive,audit,recovery=None,with_bars=False):
    archive,audit=Path(archive),Path(audit)
    proof=json.loads(audit.read_text(encoding='utf-8'))
    if proof['status']!='LONG_INTERVAL_RAW_COHORT_AND_CHECKS_RECONSTRUCTED':raise ValueError('complete source audit required')
    for path,wanted in proof['source_hashes'].items():
        if engine.sha(path)!=wanted:raise ValueError('audited parent source changed')
    scope=pd.read_csv(archive/'candidate_scope_NO_OUTCOMES.csv',dtype={'code':str})
    checks=json.loads((archive/'signal_checks.json').read_text(encoding='utf-8'))
    indexed={(r['date'],r['code']):r for r in checks}
    inputs=[audit,archive/'signal_checks.json',archive/'candidate_scope_NO_OUTCOMES.csv',Path(__file__)]
    recovered_bars=pd.DataFrame(columns=engine.BAR_FIELDS);recovered_days=pd.DataFrame(columns=engine.DAY_FIELDS)
    recovered_codes=set()
    if recovery:
        recovery=Path(recovery)
        report=json.loads((recovery/'report.json').read_text(encoding='utf-8'))
        for path,wanted in report['source_hashes'].items():
            if engine.sha(path)!=wanted:raise ValueError('recovery source changed')
        for name,wanted in report['artifact_hashes'].items():
            path=(recovery/name).resolve()
            if not path.is_relative_to(recovery.resolve()) or engine.sha(path)!=wanted:raise ValueError('recovery artifact changed')
        parent=json.loads((archive/'report.json').read_text(encoding='utf-8'))
        expected={r['code'] for r in parent['audit'] if r['kind']=='interval' and r['status']=='UNAVAILABLE'}
        if set(report['recovery_codes'])!=expected or report['source_hashes'].get(str((archive/'report.json').resolve()))!=engine.sha(archive/'report.json'):
            raise ValueError('recovery is not the exact parent failure cohort')
        recovered_scope=pd.read_csv(recovery/'recovery_scope_NO_OUTCOMES.csv',dtype={'code':str})
        pd.testing.assert_frame_equal(recovered_scope.reset_index(drop=True),scope[scope.code.isin(expected)].reset_index(drop=True),check_dtype=False)
        recovered_bars=pd.read_csv(recovery/'interval_normalized.csv',dtype={'code':str})
        recovered_days=pd.read_csv(recovery/'daily_normalized.csv')
        seen=set()
        for item in report['audit']:
            key=(item['code'],item['kind'])
            if key in seen:raise ValueError('duplicate recovery request')
            seen.add(key)
            if item['status']!='RAW_PARSED':continue
            path=Path(item['raw_path']).resolve()
            if not path.is_relative_to(recovery.resolve()) or engine.sha(path)!=item['raw_sha256']:raise ValueError('recovery raw changed')
            frame,raw_count=parse_source(item)
            saved=recovered_bars[recovered_bars.code.eq(item['code'])] if item['kind']=='interval' else recovered_days[recovered_days.symbol.eq(item['symbol'])]
            pd.testing.assert_frame_equal(saved.reset_index(drop=True),frame.reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-8)
            if len(frame)!=item['parsed_rows'] or (raw_count is not None and raw_count!=item['raw_rows']):raise ValueError('recovery count changed')
        if seen!={(code,kind) for code in expected for kind in ['interval','daily']}:raise ValueError('recovery source request scope changed')
        recovered_checks=[]
        for code in sorted(expected):
            bars=recovered_bars[recovered_bars.code.eq(code)]
            days=recovered_days[recovered_days.symbol.eq(engine.minute.live.sina_symbol(code))]
            for row in recovered_scope[recovered_scope.code.eq(code)].itertuples():
                recovered_checks.append(engine.check_signal(bars,days,row.code,row.date,row.price,row.daily_true_one_word))
        if recovered_checks!=json.loads((recovery/'signal_checks.json').read_text(encoding='utf-8')):
            raise ValueError('recovery candidate checks fail raw reconstruction')
        for row in recovered_checks:indexed[(row['date'],row['code'])]=row
        recovered_codes=expected
        inputs += [recovery/'report.json',recovery/'signal_checks.json',recovery/'recovery_scope_NO_OUTCOMES.csv',
            recovery/'interval_normalized.csv',recovery/'daily_normalized.csv']
    if len(indexed)!=len(scope) or set(indexed)!=set(map(tuple,scope[['date','code']].to_numpy())):
        raise ValueError('full parent candidate denominator changed')
    bars,days=None,None
    if with_bars:
        bars=pd.read_csv(archive/'thirty_minute_bars_UNRECONCILED.csv',dtype={'code':str})
        days=pd.read_csv(archive/'unadjusted_daily_reference.csv')
        inputs += [archive/'thirty_minute_bars_UNRECONCILED.csv',archive/'unadjusted_daily_reference.csv']
        if recovery:
            bars=pd.concat([bars[~bars.code.isin(recovered_codes)],recovered_bars],ignore_index=True)
            replaced={engine.minute.live.sina_symbol(code) for code in recovered_codes}
            days=pd.concat([days[~days.symbol.isin(replaced)],recovered_days],ignore_index=True)
        if bars.duplicated(['code','minute']).any() or days.duplicated(['symbol','date']).any():raise ValueError('composite normalized keys duplicated')
    return scope,[indexed[key] for key in sorted(indexed)],bars,days,inputs
