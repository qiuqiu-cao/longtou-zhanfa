import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

u = "https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPT_BILLBOARD_DAILYDETAILSBUY&columns=ALL&filter=(SECURITY_CODE%3D%22600828%22)(TRADE_DATE%3D%272026-10-27%27)"
req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=5) as resp:
    d = json.loads(resp.read().decode('utf-8'))
    for idx, row in enumerate(d['result']['data']):
        print(f"Row {idx}: {row.get('OPERATEDEPT_NAME')} | Code={row.get('OPERATEDEPT_CODE')} | Buy={row.get('BUY')} | Sell={row.get('SELL')}")
