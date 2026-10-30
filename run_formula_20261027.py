import json

# 1. Load 2026-10-26 facts (Yesterday)
with open('results/daily_review_20261026T1323Z/facts/complete_enriched_facts_SCORES_MASKED.json', 'r', encoding='utf-8') as f:
    d26_data = json.load(f)
    rows_26 = {r['code']: r for r in d26_data.get('rows', [])}

# 2. Load 2026-10-27 facts (Today)
with open('results/daily_review_20261027T0928Z/facts/normalized_daily.json', 'r', encoding='utf-8') as f:
    daily = json.load(f)

with open('results/daily_review_20261027T0928Z/facts/2026-10-27_previous_limit_primary.json', 'r', encoding='utf-8') as f:
    prev_limit_data = json.load(f)
    prev_limit_rows = {r['code']: r for r in prev_limit_data.get('rows', [])}

with open('results/daily_review_20261027T0928Z/facts/complete_enriched_facts_SCORES_MASKED.json', 'r', encoding='utf-8') as f:
    d27_data = json.load(f)
    rows_27_enriched = {r['code']: r for r in d27_data.get('rows', [])}

print(f"Total 10-26 limit up / previous limit candidate codes: {len(prev_limit_rows)}")

candidates = []

for code, r_prev in prev_limit_rows.items():
    r26 = rows_26.get(code)
    stock_daily = daily.get(code, {}).get('days', {})
    d26_bar = stock_daily.get('2026-10-26')
    d27_bar = stock_daily.get('2026-10-27')
    r27_en = rows_27_enriched.get(code)
    
    if not d26_bar or not d27_bar:
        continue
        
    name = r_prev.get('name')
    if r26 and r26.get('name'):
        name = r26.get('name')
    elif r27_en and r27_en.get('name'):
        name = r27_en.get('name')
        
    c26 = d26_bar['close'] / 100.0
    h26 = d26_bar['high'] / 100.0
    v26 = d26_bar.get('volume_native', 0)
    
    # Amount & turnover on 10-26
    amt_26 = r26.get('amount_cny') if r26 else None
    if amt_26 is None:
        amt_26 = r26.get('primary_amount_cny') if r26 else None
    to_26 = r26.get('turnover_pct') if r26 else None
    
    # 10-27 bars
    o27 = d27_bar['open'] / 100.0
    c27 = d27_bar['close'] / 100.0
    h27 = d27_bar['high'] / 100.0
    l27 = d27_bar['low'] / 100.0
    
    gap_pct = (o27 - c26) / c26 * 100.0
    day_pct = (c27 - c26) / c26 * 100.0
    
    # Limit up price on 10-27
    is_20 = code.startswith('300') or code.startswith('301') or code.startswith('688')
    limit_mult = 1.20 if is_20 else 1.10
    limit_price_27 = round(c26 * limit_mult, 2)
    
    # Criteria evaluation
    c1_yest_zt = True # since in prev_limit_rows
    c2_not_one = o27 < (limit_price_27 - 0.01) # not opening at limit price
    c3_golden_gap = (gap_pct >= 2.0) and (gap_pct <= 7.0)
    c4_break_high = o27 >= h26
    
    # Capacity filter: Amount >= 3亿 (300,000,000) and Turnover in [8.0, 32.0]%
    # If amt_26 or to_26 is known
    capacity_pass = False
    if amt_26 is not None and to_26 is not None:
        capacity_pass = (amt_26 >= 300000000) and (to_26 >= 8.0) and (to_26 <= 32.0)
    else:
        # Check from normalized daily volume * close approx
        est_amt = v26 * 100 * c26
        capacity_pass = (est_amt >= 250000000) # fallback
        
    is_zt = (c27 >= limit_price_27 - 0.01)
    is_touch = (h27 >= limit_price_27 - 0.01)
    
    candidates.append({
        'code': code,
        'name': name,
        'c26': c26,
        'amt_26_yi': amt_26 / 1e8 if amt_26 else None,
        'to_26': to_26,
        'o27': o27,
        'gap_pct': gap_pct,
        'c27': c27,
        'day_pct': day_pct,
        'h27': h27,
        'limit_price': limit_price_27,
        'is_zt': is_zt,
        'is_touch': is_touch,
        'not_one': c2_not_one,
        'golden_gap': c3_golden_gap,
        'break_high': c4_break_high,
        'capacity_pass': capacity_pass
    })

print(f"\nLoaded {len(candidates)} candidates.")

# 1. Check STRICT formula pass (including capacity)
strict_passed = [c for c in candidates if c['not_one'] and c['golden_gap'] and c['break_high'] and c['capacity_pass']]

# 2. Check GOLDEN GAP pass (without capacity restriction, i.e. standard non-one gap)
gap_passed = [c for c in candidates if c['not_one'] and c['golden_gap'] and c['break_high']]

print("\n" + "="*80)
print(f"【严格命中公式的标的 (含容量换手健康度)】: 共 {len(strict_passed)} 只")
print("="*80)
for p in strict_passed:
    zt_str = "【封死涨停】" if p['is_zt'] else ("【盘中触板】" if p['is_touch'] else "【未涨停】")
    print(f"标的代码: {p['code']} | 名称: {p['name']}")
    print(f"  昨日收盘: {p['c26']:.2f}元 | 昨日成交额: {p['amt_26_yi']:.2f}亿元 | 昨日换手率: {p['to_26']:.1f}%")
    print(f"  今日开盘: {p['o27']:.2f}元 (竞价高开: {p['gap_pct']:+.2f}%) | 开盘突破昨高: {p['break_high']}")
    print(f"  今日最高: {p['h27']:.2f}元 | 今日收盘: {p['c27']:.2f}元 (全天涨幅: {p['day_pct']:+.2f}%)")
    print(f"  今日表现: {zt_str}")
    print("-" * 60)

print("\n" + "="*80)
print(f"【所有符合黄金高开 (+2%~+7%) 且非一字的标的】: 共 {len(gap_passed)} 只")
print("="*80)
for p in gap_passed:
    zt_str = "涨停" if p['is_zt'] else ("触板" if p['is_touch'] else "未板")
    amt_s = f"{p['amt_26_yi']:.2f}亿" if p['amt_26_yi'] is not None else "未知"
    to_s = f"{p['to_26']:.1f}%" if p['to_26'] is not None else "未知"
    print(f"{p['code']} {p['name']:<8} | 昨收:{p['c26']:>6.2f} | 昨额:{amt_s:>6} | 昨换:{to_s:>6} | 高开:{p['gap_pct']:>+5.2f}% | 收盘:{p['c27']:>6.2f} | 涨幅:{p['day_pct']:>+6.2f}% | 状态:{zt_str}")
