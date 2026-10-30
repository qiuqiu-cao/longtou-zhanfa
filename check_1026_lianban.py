import json

with open('results/daily_review_20261026T1323Z/facts/complete_enriched_facts_SCORES_MASKED.json', 'r', encoding='utf-8') as f:
    facts_26 = json.load(f)
    rows_26 = facts_26.get('rows', [])

with open('results/daily_review_20261027T0928Z/facts/normalized_daily.json', 'r', encoding='utf-8') as f:
    daily_27 = json.load(f)

print(f"Total rows on 10-26: {len(rows_26)}")

print("\n--- All stocks with >= 2 boards on 2026-10-26 (checking their 10-27 performance) ---")
for r in rows_26:
    b = r.get('boards')
    if b is not None and b >= 2:
        code = r['code']
        name = r['name']
        amt_26 = r.get('amount_cny', 0)
        to_26 = r.get('turnover_pct', 0)
        
        d27_info = daily_27.get(code, {}).get('days', {}).get('2026-10-27', {})
        d26_info = daily_27.get(code, {}).get('days', {}).get('2026-10-26', {})
        
        if d27_info and d26_info:
            c26 = d26_info['close'] / 100.0
            o27 = d27_info['open'] / 100.0
            gap_pct = (o27 - c26) / c26 * 100.0
            not_one = o27 < c26 * 1.098
            golden_gap = (gap_pct >= 3.5) and (gap_pct <= 7.2)
            healthy_dragon = (amt_26 >= 450000000) and (to_26 >= 10.0) and (to_26 <= 28.0)
            
            print(f"[{code}] {name}: 10-26板数={b}, 昨额={amt_26/1e8:.2f}亿, 昨换手={to_26:.1f}%, 10-27高开={gap_pct:+.2f}%, 黄金高开[3.5-7.2]={golden_gap}, 大容量[>=4.5亿,10-28%]={healthy_dragon}")
