import json

with open('eval_output_27.json', 'r', encoding='utf-8') as f:
    items = json.load(f)

print(f"Total evaluated: {len(items)}")

strict = [x for x in items if x['strict_pass']]
print(f"\n--- 严格通过公式 (3只) ---")
for x in strict:
    print(f"[{x['code']}] {x['name']}: 昨收={x['c26']}, 昨成交={x['amt_26']/1e8:.2f}亿, 昨换手={x['to_26']:.1f}%, 高开={x['gap_pct']:+.2f}%, 盘中最高={x['h27']}, 收盘={x['c27']} ({x['day_pct']:+.2f}%), 涨停={x['is_zt']}")

gap_only = [x for x in items if x['gap_pass']]
print(f"\n--- 黄金高开(+2%~+7%)全集 ({len(gap_only)}只) ---")
for x in gap_only:
    zt_str = "封死涨停" if x['is_zt'] else ("触及涨停" if x['is_touch'] else "冲高回落/未板")
    amt_str = f"{x['amt_26']/1e8:.2f}亿" if x['amt_26'] else "未收录"
    to_str = f"{x['to_26']:.1f}%" if x['to_26'] else "未收录"
    print(f"[{x['code']}] {x['name']:<6} | 昨收:{x['c26']:>5.2f} | 昨额:{amt_str:>6} | 昨换:{to_str:>5} | 高开:{x['gap_pct']:>+5.2f}% | 收盘:{x['c27']:>5.2f} ({x['day_pct']:>+6.2f}%) | 状态:{zt_str}")

# Check key rejected stocks
print(f"\n--- 关键被公式排雷的个股 ---")
for x in items:
    if x['code'] in ['603089', '603028', '603528', '001219']:
        print(f"[{x['code']}] {x['name']}: 高开={x['gap_pct']:+.2f}%, 收盘={x['c27']} ({x['day_pct']:+.2f}%), 为什么排除: {'平开/低开走弱(<2%)' if x['gap_pct']<2.0 else ('一字顶板(>7%)' if x['gap_pct']>7.0 else '其他')}")
