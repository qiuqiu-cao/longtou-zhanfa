import urllib.request
import urllib.parse
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

seat_name = "中泰证券股份有限公司常州惠国路证券营业部"
filter_str = f'(OPERATEDEPT_NAME="{seat_name}")'
encoded_filter = urllib.parse.quote(filter_str)

url = f"https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPT_OPERATOR_BILLBOARD_DETAILS&columns=ALL&filter={encoded_filter}&sortTypes=-1&sortColumns=TRADE_DATE&pageNumber=1&pageSize=20"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=8) as resp:
        d = json.loads(resp.read().decode('utf-8'))
        items = d.get('result', {}).get('data', [])
        print("Count:", len(items))
        for it in items:
            print(f"Date: {it.get('TRADE_DATE')[:10]} | Stock: {it.get('SECURITY_NAME_ABBR')} ({it.get('SECURITY_CODE')}) | Buy: {it.get('BUY_AMT')} | Sell: {it.get('SELL_AMT')}")
except Exception as e:
    print("Err:", e)
