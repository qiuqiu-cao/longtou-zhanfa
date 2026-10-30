import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Query eastmoney billboard seat details
url = "https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPT_BILLBOARD_DAILYDETAILSBUY&columns=ALL&filter=(SECURITY_CODE%3D%22600828%22)(TRADE_DATE%3D%272026-10-27%27)&sortTypes=-1&sortColumns=BUY_AMT"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=8) as resp:
        d = json.loads(resp.read().decode('utf-8'))
        items = d.get('result', {}).get('data', [])
        print("Count:", len(items))
        for it in items:
            print(f"Buyer: {it.get('OPERATOR_CODE_NAME')} | Buy: {it.get('BUY_AMT')} | Sell: {it.get('SELL_AMT')} | Net: {it.get('NET_AMT')}")
except Exception as e:
    print("Err:", e)

# Also let's query the sell side
url_sell = "https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPT_BILLBOARD_DAILYDETAILSSELL&columns=ALL&filter=(SECURITY_CODE%3D%22600828%22)(TRADE_DATE%3D%272026-10-27%27)&sortTypes=-1&sortColumns=SELL_AMT"
req_sell = urllib.request.Request(url_sell, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req_sell, timeout=8) as resp:
        d = json.loads(resp.read().decode('utf-8'))
        items = d.get('result', {}).get('data', [])
        print("\nSell Count:", len(items))
        for it in items:
            print(f"Seller: {it.get('OPERATOR_CODE_NAME')} | Buy: {it.get('BUY_AMT')} | Sell: {it.get('SELL_AMT')} | Net: {it.get('NET_AMT')}")
except Exception as e:
    print("Err:", e)
