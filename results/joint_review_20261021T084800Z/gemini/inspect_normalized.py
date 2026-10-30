import json
from pathlib import Path

base_p = Path("results/joint_review_20261021T084800Z/facts")

with open(base_p / "normalized_daily.json", encoding="utf-8") as f:
    daily = json.load(f)

focus = ["002086", "002724", "002242", "002696", "603330", "601566", "600838", "603278", "600178", "002929", "002076", "603138", "603332", "002313", "002347", "002835"]

print("=== FOCUS STOCKS FROM NORMALIZED DAILY ===")
for c in focus:
    if c in daily:
        item = daily[c]
        n = item.get("name")
        close = item.get("close")
        pct = item.get("pct")
        to = item.get("turnover_pct")
        amt = item.get("amount", 0) / 1e8 if item.get("amount") else 0
        cap = item.get("float_cap", 0) / 1e8 if item.get("float_cap") else 0
        kline = item.get("history_kline", item.get("kline", []))
        to_str = f"{to:.2f}%" if to is not None else "N/A"
        print(f"{c} {n} | 收盘: {close} | 涨幅: {pct}% | 换手: {to_str} | 成交: {amt:.2f}亿 | 流通值: {cap:.2f}亿 | kline_len: {len(kline) if kline else 0}")
        if kline:
            # print last 3 kline points
            recent = kline[-3:]
            print("   最近K线:", recent)
    else:
        print(f"{c} NOT IN DAILY")
