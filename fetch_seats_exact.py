import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Query seats for 100433541 or 600828
url = "https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPT_BILLBOARD_DAILYDETAILSBUY&columns=ALL&filter=(SECURITY_CODE%3D%22600828%22)(TRADE_DATE%3D%272026-10-27%27)"
# Alternative report names: RPT_OPERATOR_BILLBOARD_DETAILS, RPT_ORGANIZATION_TRADEDETAILS
for rname in ["RPT_BILLBOARD_DAILYDETAILSBUY", "RPT_DAILYBILLBOARD_SEATDETAILS", "RPT_LHB_BOARDDETAILS"]:
    u = f"https://datacenter-web.eastmoney.com/api/data/v1/get?reportName={rname}&columns=ALL&filter=(SECURITY_CODE%3D%22600828%22)(TRADE_DATE%3D%272026-10-27%27)"
    try:
        req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            d = json.loads(resp.read().decode('utf-8'))
            if d.get('result') and d['result'].get('data'):
                print(f"=== FOUND in {rname} ===")
                for row in d['result']['data']:
                    print(row.get('OPERATOR_CODE_NAME') or row.get('BRANCH_NAME'), row.get('BUY_AMT'), row.get('SELL_AMT'))
                break
            else:
                print(f"Empty in {rname}")
    except Exception as e:
        print(f"Err {rname}: {e}")
