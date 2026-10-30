import json
import pandas as pd
from pathlib import Path

base_p = Path("results/joint_review_20261021T084800Z/facts")

with open(base_p / "2026-10-21_limit_up_primary.json", encoding="utf-8") as f:
    up = json.load(f)
with open(base_p / "2026-10-21_failed_board_primary.json", encoding="utf-8") as f:
    failed = json.load(f)
with open(base_p / "2026-10-21_limit_down_primary.json", encoding="utf-8") as f:
    down = json.load(f)
with open(base_p / "2026-10-21_previous_limit_primary.json", encoding="utf-8") as f:
    prev = json.load(f)

out_dir = Path("results/joint_review_20261021T084800Z/gemini")
out_dir.mkdir(parents=True, exist_ok=True)

lines = []
lines.append(f"=== 2026-10-21 市场收盘全景事实 ===")
lines.append(f"涨停总数: {up.get('all_market_raw_count')} (主板/合格池行数: {len(up['rows'])})")
lines.append(f"炸板总数: {failed.get('all_market_raw_count')} (主板行数: {len(failed['rows'])})")
lines.append(f"跌停总数: {down.get('all_market_raw_count')} (主板行数: {len(down['rows'])})")
lines.append(f"昨日涨停今日行数: {len(prev['rows'])}")

lines.append("\n--- 涨停梯队详情 (按板位和首封排序) ---")
df_up = pd.DataFrame(up["rows"])
for boards, grp in sorted(df_up.groupby("boards"), key=lambda x: -x[0]):
    lines.append(f"\n【{boards} 板】共 {len(grp)} 只:")
    for r in grp.sort_values("seal_first").to_dict("records"):
        lines.append(f"  {r['code']} {r['name']} | 首封: {r['seal_first']} | 换手: {r['turnover_pct']:.2f}% | 成交: {r['amount']/1e8:.2f}亿 | 开板: {r['reported_open_count']}次 | 行业: {r['industry']}")

lines.append("\n--- 炸板个股 ---")
df_failed = pd.DataFrame(failed["rows"])
for r in df_failed.to_dict("records"):
    lines.append(f"  {r.get('code')} {r.get('name')} | 涨幅: {r.get('pct')}% | 换手: {r.get('turnover_pct', 0):.2f}% | 成交: {r.get('amount', 0)/1e8:.2f}亿 | 行业: {r.get('industry')}")

lines.append("\n--- 跌停个股 ---")
df_down = pd.DataFrame(down["rows"])
for r in df_down.to_dict("records"):
    lines.append(f"  {r.get('code')} {r.get('name')} | 跌幅: {r.get('pct')}% | 换手: {r.get('turnover_pct', 0):.2f}% | 行业: {r.get('industry')}")

lines.append("\n--- 昨日涨停今日反馈 (Previous Limit) ---")
df_prev = pd.DataFrame(prev["rows"])
for r in df_prev.sort_values("pct", ascending=False).to_dict("records"):
    lines.append(f"  {r.get('code')} {r.get('name')} | 今日涨幅: {r.get('pct')}% | 今日换手: {r.get('turnover_pct', 0):.2f}% | 行业: {r.get('industry')}")

output_text = "\n".join(lines)
with open(out_dir / "FACTS_SUMMARY.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

print("Saved FACTS_SUMMARY.txt successfully")
