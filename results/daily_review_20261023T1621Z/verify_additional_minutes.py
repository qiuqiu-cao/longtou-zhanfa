from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
import json,hashlib
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
a=read(OUT/'minutes/additional_capture_receipt.json')['records'];b=read(OUT/'minutes/capture_receipt.json')['records'];d=read(OUT/'facts/normalized_daily.json');result=[]
for symbol in ['sz002724','sh603721']:
 e=next(x for x in a if x['symbol']==symbol and x.get('source')=='EASTMONEY_MINUTE_OHLC');t=next(x for x in a+b if x['symbol']==symbol and x.get('source')!='EASTMONEY_MINUTE_OHLC')
 cap=float((Decimal(str(e['previous_close']))*Decimal('1.1')).quantize(Decimal('.01'),rounding=ROUND_HALF_UP));by={x['time']:x for x in t['rows']};matched=[x for x in e['rows'] if x['time'] in by];different=[dict(time=x['time'],eastmoney=x['price'],tencent=by[x['time']]['price']) for x in matched if x['price']!=by[x['time']]['price']]
 bar=d[symbol[2:]]['days']['2026-10-23'];oh=dict(open=e['rows'][0]['open'],close=e['rows'][-1]['price'],high=max(x['high'] for x in e['rows']),low=min(x['low'] for x in e['rows']))
 check={k:oh[k]==bar[k]/100 for k in oh};volsum=sum(x['volume_hands'] for x in e['rows']);daily_volume=bar['volume_native']
 result.append(dict(symbol=symbol,upper_reference=cap,minute_first_high_equals_limit=next((x for x in e['rows'] if x['high']==cap),None),minute_first_close_equals_limit=next((x for x in e['rows'] if x['price']==cap),None),common_minutes=len(matched),equal_closes=len(matched)-len(different),different_closes=different,aggregated_minute_ohlc=oh,daily_ohlc_checks=check,minute_volume_hands_sum=volsum,daily_volume_native=daily_volume,volume_sum_difference=volsum-daily_volume,field_decision='Reject pool fbt morning value as first limit-up time; minute bins support afternoon event, no second-level seal/queue certification',raw_hash_checks={r['path']:hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in [e,t]}))
page=read(OUT/'facts/2026-10-23_complete_page_pools.json');pool=read(OUT/'facts/raw/2026-10-23_limit_up.json')['data']['pool'];page_codes={x['code'] for x in page['rows'] if x['kind']=='limit_up'};pool_codes={x['c'] for x in pool};diff=dict(primary_only=[dict(code=x['c'],name=x['n']) for x in pool if x['c'] not in page_codes],page_only=[dict(code=x['code'],name=x['name']) for x in page['rows'] if x['kind']=='limit_up' and x['code'] not in pool_codes])
obj=dict(formed_at=datetime.now(timezone.utc).isoformat(),date='2026-10-23',stocks=result,pool_count_difference=diff,raw_pool_values_preserved=True,no_funds_causality_inferred=True,models_seen_or_loaded=False)
with (OUT/'minutes/SEAL_CONFLICT_CROSSCHECK_APPEND.json').open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
print(json.dumps(obj,ensure_ascii=False))
