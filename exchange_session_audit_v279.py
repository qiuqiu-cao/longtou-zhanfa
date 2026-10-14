"""Bounded historical-session audit against read SSE annual closure notices."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import pandas as pd
from next_session_forecast_lab_v271 import ROOT,DATA,sha

START,END="2023-11-28","2026-10-09"
SOURCES={
 "2023":"https://www.sse.com.cn/disclosure/announcement/general/c/c_20221227_5714458.shtml",
 "2024":"https://www.sse.com.cn/disclosure/announcement/general/c/c_20231226_5733939.shtml",
 "2025":"https://www.sse.com.cn/disclosure/announcement/general/c/c_20241223_10767108.shtml",
 "2026":"https://www.sse.com.cn/disclosure/announcement/general/c/c_20251222_10802507.shtml"}
# 2023's last closure ends Oct 6, before this bounded audit begins.
RANGES=[
 ("2024-01-01","2024-01-01"),("2024-02-09","2024-02-17"),("2024-04-04","2024-04-06"),
 ("2024-05-01","2024-05-05"),("2024-06-10","2024-06-10"),("2024-09-15","2024-09-17"),("2024-10-01","2024-10-07"),
 ("2025-01-01","2025-01-01"),("2025-01-28","2025-02-04"),("2025-04-04","2025-04-06"),
 ("2025-05-01","2025-05-05"),("2025-05-31","2025-06-02"),("2025-10-01","2025-10-08"),
 ("2026-01-01","2026-01-03"),("2026-02-15","2026-02-23"),("2026-04-04","2026-04-06"),
 ("2026-05-01","2026-05-05"),("2026-06-19","2026-06-21"),("2026-09-25","2026-09-27"),("2026-10-01","2026-10-07")]


def sessions(start=START,end=END):
    if start<START or end>END or start>end:
        raise ValueError("outside verified bounded calendar scope")
    closed={d.strftime('%Y-%m-%d') for lo,hi in RANGES for d in pd.date_range(lo,hi)}
    return [d.strftime('%Y-%m-%d') for d in pd.bdate_range(start,end) if d.strftime('%Y-%m-%d') not in closed]


def run(output):
    target=Path(output);target.mkdir(parents=True,exist_ok=False)
    cal=sessions();successor=dict(zip(cal[:-1],cal[1:]))
    market=pd.read_csv(DATA/'market_emotion.csv')
    expected=set(sessions(market.date.min(),market.date.max()));observed=set(market.date)
    candidates=pd.read_csv(DATA/'leader_candidates.csv',dtype={'code':str},low_memory=False)
    candidate_expected=candidates.date.map(successor)
    supplied=candidates.trade_date.fillna('')
    known=supplied.ne('');wrong=known&supplied.ne(candidate_expected)
    report={'version':'v279-sse-annual-calendar-audit-1','recorded_at':datetime.now(timezone.utc).isoformat(),
      'status':'CONSISTENT' if observed==expected and not wrong.any() and not market.date.duplicated().any() else 'MISMATCH',
      'scope':[START,END],'sources':SOURCES,'closure_ranges':RANGES,'market_rows':len(market),
      'market_expected_sessions':len(expected),'missing_official_sessions':sorted(expected-observed),
      'unexpected_market_dates':sorted(observed-expected),'duplicate_market_dates':int(market.date.duplicated().sum()),
      'candidate_known_target_rows':int(known.sum()),'candidate_wrong_next_session_rows':int(wrong.sum()),
      'candidate_target_unknown_rows':int((~known).sum()),'unknown_outcomes_not_imputed':True,
      'calendar_proves':['session membership within bounded range','immediate next session within bounded range'],
      'calendar_does_not_prove':['pool completeness','point-in-time archive receipt','features','promotion outcome','accuracy'],
      'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [DATA/'market_emotion.csv',DATA/'leader_candidates.csv',Path(__file__)]}}
    pd.DataFrame({'date':cal,'next_session':[successor.get(d) for d in cal]}).to_csv(target/'verified_sessions.csv',index=False)
    (target/'calendar.json').write_text(json.dumps({'status':'verified','source':'; '.join(SOURCES.values()),'scope':[START,END],'sessions':cal},ensure_ascii=False,indent=2),encoding='utf-8')
    (target/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return {k:report[k] for k in ['status','market_rows','market_expected_sessions','missing_official_sessions','unexpected_market_dates','candidate_wrong_next_session_rows','candidate_target_unknown_rows']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(args.output),ensure_ascii=True))
