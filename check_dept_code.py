import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

u = "https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPT_BILLBOARD_DAILYDETAILSBUY&columns=ALL&filter=(OPERATEDEPT_CODE%3D%2210741097%22)&sortTypes=-1&sortColumns=TRADE_DATE&pageNumber=1&pageSize=15"
req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=5) as resp:
        d = json.loads(resp.read().decode('utf-8'))
        items = d.get('result', {}).get('data', [])
        print("Count for 10741097:", len(items))
        for it in items:
            print(f"Date: {it.get('TRADE_DATE')[:10]} | Code: {it.get('SECURITY_CODE')} | Name: {it.get('SECURITY_NAME_ABBR')} | Buy: {it.get('BUY')}")
except Exception as e:
    print("Err:", e)
