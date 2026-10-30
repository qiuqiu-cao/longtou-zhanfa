from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
f=read(OUT/'facts/complete_enriched_facts_SCORES_MASKED.json');own=read(OUT/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json');freeze=read(OUT/'INDEPENDENT_FREEZE_RECEIPT.json');multi=read(OUT/'MULTIDAY_PREDICTIONS_AND_EXECUTION_CARDS.json');base=read(OUT/'BASE_scores_AFTER_INDEPENDENT.json')
assert len(f['rows'])==len(own['rows'])==97
assert {r['code'] for r in f['rows']}=={r['code'] for r in own['rows']}
assert sha(OUT/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json')==freeze['sha256']
assert own['formed_at']<base['formed_at']<multi['formed_at']
sessions=read(ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json')['sessions'];p=sessions.index('2026-10-22')
assert multi['dates']==sessions[p+1:p+6]
assert len(base['rows'])==9 and base['new_fits']==0
assert all(sha(path)==digest for path,digest in read(OUT/'BASE_inference_BEFORE_LOAD.json')['protected_hashes'].items())
for path,digest in f['source_bindings'].items():assert sha(path)==digest,path
lookup={r['code']:r for r in f['rows']}
for code in own['top3']:
 r=lookup[code]
 assert r['kind']=='limit_up' and r['boards']==1 and r['original_cap_screen'] and not r['thin_proxy_excluded'] and r['qualification']=='EXACT_DAY_OHLC_OBSERVED'
for item in multi['candidates']:
 assert item['signal_close']==lookup[item['code']]['close']
 assert item['next_session_limit_reference']==lookup[item['code']]['next_upper']
assert sum(r['kind']=='limit_up' for r in f['rows'])==30
assert sum(r['kind']=='limit_up' and r['boards']==1 for r in f['rows'])==21
# Reconcile reported field conflicts without selecting the more flattering value.
diffs=[]
for r in f['rows']:
 z=r.get('source_rows',{});page=z.get('page',{});primary=z.get('primary_limit_up',z.get('primary_failed_board',{}))
 if page and primary:
  for field in ['turnover_pct','reported_open_count']:
   a,b=page.get(field),primary.get(field)
   if field=='turnover_pct':different=a is not None and b is not None and abs(float(a)-float(b))>.02
   else:
    try:different=int(str(a).replace('次',''))!=int(b)
    except (TypeError,ValueError):different=False
   if different:diffs.append(dict(code=r['code'],name=r['name'],field=field,page=a,primary=b,policy='Both retained; primary used as provider report, neither is Level2 truth.'))
forum=read(OUT/'context/limited_forum_capture.json')
gaps=[dict(key=r['key'],raw_bytes=Path(r['raw_path']).stat().st_size,body_scope='EMPTY_HTTP_NOT_BODY; separately read web-extracted author body' if Path(r['raw_path']).stat().st_size==0 else 'Body not independently qualified; do not equate HTML with article body') for r in forum['records']]
qa=dict(formed_at=now(),full_rows=97,current_price_known=sum(r['qualification']=='EXACT_DAY_OHLC_OBSERVED' for r in f['rows']),unknown_price_codes=[r['code'] for r in f['rows'] if r['qualification'].startswith('UNKNOWN')],top3_all_within_original_screen=True,full_pool_before_model_verified=True,source_hashes_verified=len(f['source_bindings']),protected_files_unchanged=True,fixed_calendar_verified=True,all_prices_and_limits_match=True,provider_field_differences=diffs,forum_capture_scope_corrections=gaps,
 workbench=dict(emotion_context_calculated=(OUT/'workbench_with_emotion/facts_with_emotion.json').exists(),markdown_render_completed=False,failure='TypeError formatting unknown promotion_rate None in existing workbench markdown; failed output preserved; report independently renders unknown explicitly',core_not_modified=True),
 new_fits=0,new_orders=0,new_formal_registrations=0,predictions_unchanged=True,review_report_sha256=sha(OUT/'10月22日复盘与逐日预期预测.md'))
(OUT/'FINAL_QA.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
receipt=dict(formed_at=now(),status='INDEPENDENT_DAILY_REVIEW_AND_FIVE_SESSION_SCENARIOS_COMPLETE',signal_date='2026-10-22',target_date='2026-10-23',full_pool=97,top3=own['top3'],observations=own['observations'],report=str(OUT/'10月22日复盘与逐日预期预测.md'),qa_path=str(OUT/'FINAL_QA.json'),model_fits=0,original_production_and_tasks_unchanged=True,actual_peer_current_review=False,limits='Uncalibrated independent forecasts; full provider scope not exchange-wide; four current price gaps and one prior first-board price gap retained; no Level2/fills.')
(OUT/'COMPLETION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'all_checks_passed':True,'full_rows':97,'current_price_known':qa['current_price_known'],'provider_field_differences':diffs,'source_hashes':qa['source_hashes_verified'],'fits':0},ensure_ascii=True))
