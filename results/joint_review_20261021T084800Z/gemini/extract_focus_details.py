import json
from pathlib import Path

base_p = Path("results/joint_review_20261021T084800Z/facts")

with open(base_p / "normalized_daily.json", encoding="utf-8") as f:
    daily = json.load(f)

with open(base_p / "2026-10-21_limit_up_primary.json", encoding="utf-8") as f:
    up = json.load(f)

up_dict = {r["code"]: r for r in up["rows"]}

focus = ["002724", "600178", "603278", "002929", "002696", "002242", "002086", "603330", "601566", "600838", "002076", "002313", "002347", "002835", "001239"]

out = []
for c in focus:
    info = up_dict.get(c, {})
    d = daily.get(c, {})
    q = d.get("quote", {}).get("values", {})
    days = d.get("days", {})
    last3 = list(days.items())[-3:] if days else []
    out.append({
        "code": c,
        "name": info.get("name"),
        "boards": info.get("boards"),
        "seal_first": info.get("seal_first"),
        "open_count": info.get("reported_open_count"),
        "turnover_pct": info.get("turnover_pct"),
        "amount_亿": round(info.get("amount", 0) / 1e8, 2),
        "industry": info.get("industry"),
        "close": q.get("close"),
        "pre_close": q.get("previous_close"),
        "pct": round((q.get("close", 0) - q.get("previous_close", 0)) / q.get("previous_close", 1) * 100, 2) if q.get("previous_close") else None,
        "recent_kline": last3
    })

with open("results/joint_review_20261021T084800Z/gemini/FOCUS_STOCKS_DETAILED.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("Saved FOCUS_STOCKS_DETAILED.json")
