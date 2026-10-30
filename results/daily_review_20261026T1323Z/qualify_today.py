from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, ROUND_HALF_UP
import importlib.util
import json
import sys
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
import public_outcome_sources_v293 as public
OLD=ROOT/'results/joint_review_20261019T092500Z'
spec=importlib.util.spec_from_file_location('issuer_collector',OLD/'facts/qualify_and_notices.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
n.OUT=OUT/'facts';n.DAY='2026-10-26'
qs=importlib.util.spec_from_file_location('quote_fields_adapter',ROOT/'results/score_update_20261014T1153Z/supplemental_score_update.py')
q=importlib.util.module_from_spec(qs);qs.loader.exec_module(q);q.DAY=n.DAY
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);n.write(p,x)
def upper(v):return int((Decimal(int(v))*Decimal('1.1')).quantize(Decimal('1'),rounding=ROUND_HALF_UP))

def main():
    facts=n.OUT;page=read(facts/'2026-10-26_complete_page_pools.json');daily=read(facts/'normalized_daily.json')
    prim={k:read(facts/('2026-10-26_'+k+'_primary.json')) for k in ['limit_up','failed_board','limit_down'] if (facts/('2026-10-26_'+k+'_primary.json')).exists()}
    sources={str(facts/p):n.sha(facts/p) for p in ['normalized_daily.json','daily_manifest.json','pool_capture_manifest.json','2026-10-26_complete_page_pools.json']}
    for p in prim:sources[str(facts/('2026-10-26_'+p+'_primary.json'))]=n.sha(facts/('2026-10-26_'+p+'_primary.json'))
    merged={};conflicts=[]
    for r in page['rows']:
        if r['scope']:merged[r['code']]=dict(r,source_kinds=[r['kind']],source_rows={'page':r})
    for kind,pool in prim.items():
        for r in pool['rows']:
            c=r['code']
            if c not in merged:merged[c]=dict(date=n.DAY,code=c,name=r['name'],kind=kind,boards=r.get('boards'),scope=True,turnover_pct=r.get('turnover_pct'),reported_times=None,reason_summary_UNVERIFIED=None,source_kinds=[],source_rows={})
            v=merged[c];v['source_kinds'].append(kind);v['source_rows']['primary_'+kind]=r
            if kind=='limit_up':
                if v['kind']!='limit_up':conflicts.append(dict(code=c,reason='PRIMARY_LIMIT_VS_PAGE_KIND',page_kind=v['kind']))
                if v.get('boards') is not None and v['boards']!=r['boards']:conflicts.append(dict(code=c,reason='BOARD_COUNT_DISAGREEMENT',page=v['boards'],primary=r['boards']))
                v['kind']='limit_up';v['boards']=r['boards']
    calendar=read(ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json')['sessions'];ti=calendar.index(n.DAY);prior=calendar[ti-1]
    rows=[]
    for code,r in merged.items():
        v=dict(r);item=daily.get(code);bar=item.get('days',{}).get(n.DAY) if item else None;before=item.get('days',{}).get(prior) if item else None
        v.update(qualification='UNKNOWN_MISSING_EXACT_DAILY',close=None,one_price=None,amount_cny=None,change_pct=None)
        if bar and before:
            for k in ['open','high','low','close']:v[k]=bar[k]/100
            cap=upper(before['close']);v.update(previous_close=before['close']/100,upper=cap/100,
                next_upper=upper(bar['close'])/100,change_pct=(bar['close']/before['close']-1)*100,
                one_price=bar['open']==bar['high']==bar['low']==bar['close'],
                volume_ratio_previous_official=bar['volume_native']/before['volume_native'] if before['volume_native'] else None,
                amplitude_pct=(bar['high']-bar['low'])/before['close']*100,qualification='EXACT_DAY_OHLC_OBSERVED')
            lower=int((Decimal(before['close'])*Decimal('0.9')).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
            v['lower']=lower/100
            expected='limit_up' if bar['close']==cap else 'failed_board' if bar['high']==cap else 'limit_down' if bar['close']==lower else 'other'
            if v['kind']=='limit_down' and expected!='limit_down':v['qualification']='CONFLICT_DOWN_VS_PRICE';conflicts.append(dict(code=code,reason=v['qualification'],price_kind=expected))
            if v['kind']=='limit_up' and expected!='limit_up':v['qualification']='CONFLICT_POOL_VS_CLOSE';conflicts.append(dict(code=code,reason=v['qualification']))
            if v['kind']=='failed_board' and expected!='failed_board':v['qualification']='CONFLICT_FAILED_VS_PRICE';conflicts.append(dict(code=code,reason=v['qualification'],price_kind=expected))
            try:
                amount,turnover,_=q.quote_fields(code,item)
                v.update(amount_cny=amount,quote_turnover_pct=turnover,quote_observed_at=item['quote']['observed_at'])
                if v.get('turnover_pct') is not None and abs(turnover-v['turnover_pct'])>.02:conflicts.append(dict(code=code,reason='TURNOVER_SOURCE_DIFFERENCE',page=v['turnover_pct'],quote=turnover))
                v['quote_units']='POSITIONAL_INFERRED_CROSSCHECKED_NOT_EXCHANGE_CERTIFIED'
            except Exception as e:v['quote_field_gap']=str(e)[:180]
            if v['kind']=='limit_up' and v.get('boards',0)>=2:
                chain=calendar[ti-int(v['boards'])+1:ti+1];checks=[]
                for day in chain:
                    pre=calendar[calendar.index(day)-1];a=item['days'].get(pre);b=item['days'].get(day)
                    checks.append(bool(a and b and b['close']==upper(a['close'])))
                v['consecutive_official_sessions_checked']=all(checks)
            sources[item['raw_path']]=n.sha(item['raw_path'])
        pr=v['source_rows'].get('primary_limit_up')
        if pr:
            for dest,key in [('seal_first','seal_first'),('seal_last','seal_last'),('reported_open_count','reported_open_count'),('reported_seal_order','reported_seal_order'),('industry','industry')]:v[dest]=pr.get(key)
            v['primary_amount']=pr.get('amount')
        rows.append(v)
    limits=[r for r in rows if r['kind']=='limit_up'];relay=[r for r in limits if r.get('boards',0)>=2]
    result=dict(formed_at=n.now(),signal_date=n.DAY,target_date=calendar[ti+1],horizon_dates={str(h):calendar[ti+h] for h in [1,2,3,5]},rows=rows,relay=relay,conflicts=conflicts,model_outputs_masked=True,exchange_wide_completeness=False,
        full_returned_provider_union=True,source_counts=dict(page_raw=page['raw_pool_counts'],primary_raw={k:p['all_market_raw_count'] for k,p in prim.items()},mainboard_union={k:sum(r['kind']==k for r in rows) for k in ['limit_up','failed_board','limit_down']}),
        source_bindings=sources,unavailable=['Level2 queues','actual fills','intraday complete order path'],new_fits=0)
    write(facts/'qualified_full_facts_SCORES_MASKED.json',result)
    pd.DataFrame([{k:v for k,v in r.items() if k!='source_rows'} for r in rows]).to_csv(facts/'完整首板连板炸板跌停_遮分.csv',encoding='utf-8-sig',index=False)
    print(json.dumps({'counts':result['source_counts'],'relay_count':len(relay),'conflicts':conflicts,'closing_candidates':[{k:r.get(k) for k in ['code','name','boards','close','turnover_pct','one_price','volume_ratio_previous_official','amplitude_pct','seal_first','seal_last','reported_open_count','amount_cny','qualification']} for r in limits]},ensure_ascii=False))


if __name__=='__main__':main()
