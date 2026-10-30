from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib, json
O=Path(__file__).resolve().parent; ROOT=O.parents[1]
DAY='2026-10-29'; PRIOR='2026-10-28'; CAP=9326859896.52
def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x',encoding='utf8') as s: json.dump(v,s,ensure_ascii=False,indent=2,allow_nan=False)
def lim(p,m='1.1'): return int((Decimal(int(p))*Decimal(m)).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
f=read(O/'facts/qualified_full_facts_SCORES_MASKED.json'); daily=read(O/'facts/normalized_daily.json')
raw=read(O/'facts/raw/2026-10-29_limit_up.json'); lookup={r['c']:r for r in raw['data']['pool']}
priorpath=ROOT/'results/daily_review_20261028T1020Z/facts/RESOLVED_FACTS_v2_SCORES_MASKED.json'; prior=read(priorpath)
for r in f['rows']:
 p=lookup.get(r['code']); ds=daily.get(r['code'],{}).get('days',{}); b=ds.get(DAY)
 if p:
  past=[v for k,v in sorted(ds.items()) if k<DAY]
  mean=sum(v['volume_native'] for v in past[-5:])/5 if len(past)>=5 else None
  vr=b['volume_native']/mean if b and mean else None
  thin=p['ltsz']<=3e9 and p['hs']<5 and (r['one_price'] is True or vr is not None and vr<1)
  r.update(reported_float_cap_cny=p['ltsz'],reported_total_cap_cny=p['tshare'],volume_ratio_previous5=vr,
    original_cap_screen=p['ltsz']<=CAP,thin_proxy_excluded=thin,
    screen_status='EXCLUDED_CAP' if p['ltsz']>CAP else 'EXCLUDED_THIN_PROXY' if thin else 'UNKNOWN_DAILY' if vr is None else 'WITHIN_ORIGINAL_RESEARCH_SCREEN')
 else: r.update(screen_status='NOT_CLOSING_LIMIT_CANDIDATE',original_cap_screen=None)
 r.update(seal_delta_1430_to_close=None,seal_delta_status='UNAVAILABLE_NO_COMPARABLE_TIMESTAMPED_QUEUE_PAIR')
f.update(formed_at=now(),resolved_closing_counts={k:sum(r['kind']==k for r in f['rows']) for k in ['limit_up','failed_board','limit_down']},
 market_cap_screen_threshold_cny=CAP,source_status='EASTMONEY_RETURNED_MAINBOARD_NON_ST_SCOPE_TENCENT_OHLC_CROSSCHECKED',
 closing_price_resolutions=[],exchange_wide_completeness=False)
f['source_bindings'][str(O/'facts/qualified_full_facts_SCORES_MASKED.json')]=sha(O/'facts/qualified_full_facts_SCORES_MASKED.json')
write(O/'facts/RESOLVED_FACTS_v2_SCORES_MASKED.json',f)
feedback=[]
for r in prior['rows']:
 ds=daily.get(r['code'],{}).get('days',{}); a=ds.get(PRIOR); b=ds.get(DAY); known=bool(a and b)
 feedback.append(dict(code=r['code'],name=r['name'],prior_kind=r['kind'],prior_boards=r.get('boards'),known=known,
  close=b['close']/100 if b else None,previous_close=a['close']/100 if a else None,
  return_pct=(b['close']/a['close']-1)*100 if known else None,
  open_gap_pct=(b['open']/a['close']-1)*100 if known else None,
  closed_limit_up=int(b['close']==lim(a['close'])) if known else None,
  closed_limit_down=int(b['close']==lim(a['close'],'0.9')) if known else None))
layers={}
for h in sorted({r['prior_boards'] for r in feedback if r['prior_kind']=='limit_up'}):
 a=[r for r in feedback if r['prior_kind']=='limit_up' and r['prior_boards']==h]; k=[r for r in a if r['known']]
 layers[str(h)]=dict(total=len(a),known=len(k),unknown=len(a)-len(k),promoted=sum(r['closed_limit_up'] for r in k),
  mean_return_pct=sum(r['return_pct'] for r in k)/len(k) if k else None,limit_down=sum(r['closed_limit_down'] for r in k))
write(O/'facts/COMPLETE_PRIOR_COHORT_FEEDBACK.json',dict(formed_at=now(),signal_date=DAY,prior_date=PRIOR,source=str(priorpath),sha256=sha(priorpath),layers=layers,rows=feedback,unknown_not_zero=True))
print(json.dumps(dict(counts=f['resolved_closing_counts'],layers=layers,
 first=[{k:r.get(k) for k in ['code','name','close','seal_first','seal_last','reported_open_count','turnover_pct','amount_cny','reported_float_cap_cny','screen_status','industry']} for r in f['rows'] if r['kind']=='limit_up' and r['boards']==1]),ensure_ascii=True))
