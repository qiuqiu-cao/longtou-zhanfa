"""Bounded retrospective minute capture; preserves all raw sources and clocks."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / ('huatian_xinhua_minutes_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
OUT.mkdir(exist_ok=False)
TARGET = '2026-10-22'
HEADERS = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.sina.com.cn/'}

def now():
    return datetime.now(timezone.utc).isoformat()

def fetch(spec):
    provider, symbol, url, params = spec
    item = {'provider': provider, 'symbol': symbol, 'requested_at': now(), 'params': params}
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=12)
        raw = OUT / (provider + '_' + symbol + '.response')
        raw.write_bytes(response.content)
        item.update(url=response.url, received_at=now(), http_status=response.status_code,
                    raw_path=str(raw), bytes=len(response.content), sha256=hashlib.sha256(response.content).hexdigest())
        response.raise_for_status()
        try:
            payload = response.json()
            item['top_keys'] = list(payload) if isinstance(payload, dict) else type(payload).__name__
        except ValueError:
            pass
        item['preview'] = response.text[:450]
        item['status'] = 'RAW_RETURNED_NOT_YET_RECONCILED'
    except Exception as exc:
        item.update(status='UNAVAILABLE', error=str(exc)[:400], finished_at=now())
    return item

specs = []
for symbol in ['sz002185', 'sh600825']:
    specs.append(('sina', symbol, 'https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_data=/CN_MarketDataService.getKLineData',
                  {'symbol': symbol, 'scale': '1', 'ma': 'no', 'datalen': '480'}))
    specs.append(('tencent', symbol, 'https://web.ifzq.gtimg.cn/appstock/app/minute/query', {'code': symbol}))
    specs.append(('eastmoney', symbol, 'https://push2his.eastmoney.com/api/qt/stock/trends2/get',
                  {'secid': ('1.' if symbol.startswith('sh') else '0.') + symbol[2:],
                   'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
                   'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58', 'ndays': '1', 'iscr': '0', 'iscca': '0'}))
with ThreadPoolExecutor(max_workers=6) as pool:
    items = list(pool.map(fetch, specs))
receipt = {'target_date': TARGET, 'formed_at': now(), 'sources': items,
           'interpretation': 'Only raw source acquisition. Date, units and daily OHLC reconciliation remain required.',
           'retrospective_only': True, 'old_predictions_modified': False, 'training_runs': 0}
(OUT / 'capture_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'output': str(OUT), 'sources': items}, ensure_ascii=True), flush=True)
