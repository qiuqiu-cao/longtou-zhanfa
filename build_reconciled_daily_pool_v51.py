"""Reconcile closing-limit and failed-board pools using an independent quote scan."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd


def mainboard(frame):
    return frame[frame['代码'].str.match(r'^(00|60)') & ~frame['名称'].str.contains('ST|退', regex=True)].copy()


def build(raw_up, raw_failed, quote_up, quote_failed, prior_up, date):
    raw_up, raw_failed = mainboard(raw_up), mainboard(raw_failed)
    quote_up = quote_up[quote_up.code.str.match(r'^(00|60)') & ~quote_up.name.str.contains('ST|退', regex=True)].copy()
    quote_failed = quote_failed[quote_failed.code.str.match(r'^(00|60)') & ~quote_failed.name.str.contains('ST|退', regex=True)].copy()
    if raw_up['代码'].duplicated().any() or quote_up.code.duplicated().any():
        raise ValueError('duplicate closing code')
    raw_codes, quote_codes = set(raw_up['代码']), set(quote_up.code)
    missing_quote = sorted(raw_codes-quote_codes)
    extra_quote = sorted(quote_codes-raw_codes)
    prior_levels = prior_up.set_index('代码')['连板数'].to_dict()
    rows=[]
    for row in raw_up.to_dict('records'):
        rows.append(dict(signal_date=date,code=row['代码'],name=row['名称'],boards=int(row['连板数']),
                         first_seal_time=row['首次封板时间'],last_seal_time=row['最后封板时间'],
                         failed_seal_count=int(row['炸板次数']),close=row['最新价'],amount=row['成交额'],turnover=row['换手率'],
                         provider_industry=row['所属行业'],pool_assignment='CLOSING_LIMIT_UP',
                         classification_status='VERIFIED_TWO_SOURCE_CLOSE' if row['代码'] in quote_codes else 'CONFLICT_RAW_ONLY'))
    failed_map=raw_failed.set_index('代码').to_dict('index')
    qmap=quote_up.set_index('code').to_dict('index')
    for code in extra_quote:
        q=qmap[code]; source=failed_map.get(code,{})
        rows.append(dict(signal_date=date,code=code,name=q['name'],boards=int(prior_levels.get(code,0))+1,
                         first_seal_time=source.get('首次封板时间',pd.NA),last_seal_time=pd.NA,
                         failed_seal_count=source.get('炸板次数',pd.NA),close=q['close'],amount=q['amount_wan']*10000,
                         turnover=q['turnover'],provider_industry=source.get('所属行业',''),
                         pool_assignment='CLOSING_LIMIT_UP',classification_status='RECONCILED_QUOTE_CLOSE_OVERRIDES_FAILED_POOL'))
    closing=pd.DataFrame(rows).sort_values(['boards','first_seal_time','code'],ascending=[False,True,True])
    failed=quote_failed.copy()
    failed['signal_date']=date;failed['pool_assignment']='FAILED_BOARD_OBSERVATION'
    failed['classification_status']='INFERRED_TOUCHED_LIMIT_FROM_DAILY_HIGH_TWO_SOURCE_WHERE_AVAILABLE'
    if set(closing.code)&set(failed.code):
        raise ValueError('closing and failed overlap')
    report=dict(signal_date=date,closing_count=len(closing),first_board_count=int(closing.boards.eq(1).sum()),
                relay_count=int(closing.boards.gt(1).sum()),board_counts={str(int(k)):int(v) for k,v in closing.boards.value_counts().sort_index().items()},
                failed_count=len(failed),raw_closing_count=len(raw_up),raw_failed_count=len(raw_failed),
                raw_closing_not_quote_confirmed=missing_quote,quote_closing_not_raw=extra_quote,
                complete_within_archived_roster=not missing_quote,
                full_market_completeness='NOT_PROVEN_ARCHIVED_ROSTER_ONLY',
                eligibility_status='DELAYED_OFFICIAL_ROSTER_AND_NAME_FILTER_NOT_POINT_IN_TIME_COMPLETE')
    return closing, failed, report


def run(args):
    paths=[Path(args.raw_up),Path(args.raw_failed),Path(args.quote_up),Path(args.quote_failed),Path(args.prior_up)]
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    read=lambda p: pd.read_csv(p,dtype={'代码':str,'code':str})
    closing,failed,report=build(*map(read,paths),pd.Timestamp(args.date).strftime('%Y-%m-%d'))
    closing.to_csv(output/'closing_candidates.csv',index=False,encoding='utf-8-sig')
    failed.to_csv(output/'failed_board_observations.csv',index=False,encoding='utf-8-sig')
    report['created_at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    report['source_hashes']={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[Path(__file__)]}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['raw-up','raw-failed','quote-up','quote-failed','prior-up','date','output']:
        p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(run(a),ensure_ascii=False,indent=2))
