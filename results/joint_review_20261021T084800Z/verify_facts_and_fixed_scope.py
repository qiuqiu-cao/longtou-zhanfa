from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
import hashlib,json,sys
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
import public_outcome_sources_v293 as public
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
facts=read(OUT/'facts/complete_enriched_facts_SCORES_MASKED.json');daily=read(OUT/'facts/normalized_daily.json')
requests=read(OUT/'facts/pool_capture_manifest.json')['requests']+read(OUT/'facts/daily_manifest.json')['requests']
bindings={}
for r in requests:
 if r.get('raw_path'):
  assert sha(r['raw_path'])==r['raw_sha256'];bindings[r['raw_path']]=r['raw_sha256']
for c,x in daily.items():
 y=public.parse_daily(read(x['raw_path']),c,'2026-09-18','2026-10-21')
 assert all(x[k]==y[k] for k in ['code','days','quote','adjusted'])
assert len(facts['rows'])==67 and len({r['code'] for r in facts['rows']})==67
assert all(r['qualification']=='EXACT_DAY_OHLC_OBSERVED' for r in facts['rows'])
assert not facts['conflicts']
assert sum(r['kind']=='limit_up' for r in facts['rows'])==44
assert len(facts['relay'])==16
continuity_gaps=[{'code':r['code'],'name':r['name'],'reported_boards':r['boards'],'status':'PRIOR_REFERENCE_OFFICIAL_SESSION_BAR_MISSING_NOT_FALSE_FILLED'} for r in facts['relay'] if not r['consecutive_official_sessions_checked']]
calendar=read(ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json')['sessions'];ci=calendar.index('2026-10-21')
rows=[]
for r in facts['rows']:
 if r['kind']!='limit_up':continue
 bars=daily[r['code']]['days'];previous5=calendar[ci-5:ci]
 ratio5=bars['2026-10-21']['volume_native']/(sum(bars[z]['volume_native'] for z in previous5)/5) if all(z in bars for z in previous5) else None
 cap=r['reported_float_cap_cny'];thin=cap<=3e9 and r['turnover_pct']<5 and (r['one_price'] or ratio5 is not None and ratio5<1)
 large=cap>9326859896.52
 rows.append(dict(code=r['code'],name=r['name'],float_cap_cny=cap,volume_ratio_previous5=ratio5,turnover_pct=r['turnover_pct'],one_price=r['one_price'],above_fixed_cap=large,thin_fixed_proxy=thin,selection_allowed_cap_thin=not(large or thin),threeboard_actor_identity='UNKNOWN_NO_DATED_IDENTITY_EVIDENCE',restructuring_exception_preclose='NOT_ASSERTED',pool_row_retained=True))
pd.DataFrame(rows).to_csv(OUT/'facts/完整44只_原市值与薄换手筛选.csv',index=False,encoding='utf-8-sig')
write(OUT/'facts/DATA_AND_SCOPE_VERIFICATION.json',dict(formed_at=datetime.now(timezone.utc).isoformat(),verified_raw_files=len(bindings),all_raw_hashes_pass=True,daily_raw_replays=len(daily),exact_current67=True,reported_relay16=True,official_continuity_confirmed=16-len(continuity_gaps),official_continuity_gaps=continuity_gaps,model_fits=0,cap_threshold_cny=9326859896.52,full44_selection_allowed=sum(r['selection_allowed_cap_thin'] for r in rows),full44_above_cap=sum(r['above_fixed_cap'] for r in rows),full44_thin=sum(r['thin_fixed_proxy'] for r in rows),no_rows_deleted=True,calendar_horizons=facts['horizon_dates'],calendar_source=str(ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json'),known_semantic_differences=[dict(code='603330',field='turnover',tencent=15.1818,pool=16.150888442993165,status='RETAINED_NOT_RESOLVED')],collector_metadata_note='pool_capture_manifest prior_page field inherited old19 path; actual code reads complete Oct20 page for prior codes. Archived source run_collect.py/collect_today.py establishes actual inputs; old19 data not substituted for Oct20. Data pages current-date verified.',source_completeness='Complete returned primary/page union, not independent exchange-wide completeness certification',raw_bindings=bindings))
old=read(ROOT/'results/daily_review_20261020T101630Z/independent_full46_BEFORE_MODEL.json');settled=[]
cur={r['code']:r for r in facts['rows']}
for r in old['rows']:
 b=daily.get(r['code'],{}).get('days',{});a=b.get('2026-10-20');z=b.get('2026-10-21')
 assert a and z
 cap=int((Decimal(a['close'])*Decimal('1.1')).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
 settled.append(dict(code=r['code'],name=r['name'],priority_rank=r.get('priority_rank'),old_call=r['next_close_limit_up_call'],actual_next_close_limit=int(z['close']==cap),close_return_pct=(z['close']/a['close']-1)*100))
write(OUT/'SETTLEMENT_ORIGINAL20_FIRST_VERSION_ALL46.json',dict(formed_at=datetime.now(timezone.utc).isoformat(),source=str(ROOT/'results/daily_review_20261020T101630Z/independent_full46_BEFORE_MODEL.json'),source_sha256=sha(ROOT/'results/daily_review_20261020T101630Z/independent_full46_BEFORE_MODEL.json'),version='Original first independent version only, not later append/debate versions',rows=settled,all46_known=True,old_prediction_unchanged=True,new_formal_matured=0))
print(json.dumps(dict(raw_files=len(bindings),current67_exact=True,reported_relay=16,continuity_gaps=continuity_gaps,allowed=sum(r['selection_allowed_cap_thin'] for r in rows),excluded_cap=sum(r['above_fixed_cap'] for r in rows),thin=sum(r['thin_fixed_proxy'] for r in rows)),ensure_ascii=False))
