import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Query billboard report name
url = "https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPT_DAILYBILLBOARD_DETAILSNEW&columns=ALL&filter=(SECURITY_CODE%3D%22600828%22)(TRADE_DATE%3D%272026-10-27%27)"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=8) as resp:
        d = json.loads(resp.read().decode('utf-8'))
        print(json.dumps(d, ensure_ascii=False, indent=2)[:1500])
except Exception as e:
    print("Err:", e)
