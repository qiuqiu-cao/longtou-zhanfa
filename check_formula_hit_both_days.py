import json
import numpy as np
import pandas as pd

# Load facts for 2026-10-27 and 2026-10-28
print("Testing the exact formula on 2026-10-27 and 2026-10-28...")

# Check 2026-10-27 morning (where Yesterday = 10-26, and Day t = 10-27)
# And 2026-10-28 morning (where Yesterday = 10-27, and Day t = 10-28)

with open('results/daily_review_20261027T0928Z/facts/normalized_daily.json', 'r', encoding='utf-8') as f:
    daily_27 = json.load(f)

with open('results/daily_review_20261028T1020Z/facts/normalized_daily.json', 'r', encoding='utf-8') as f:
    daily_28 = json.load(f)

# Combine daily data
all_codes = set(daily_27.keys()).union(set(daily_28.keys()))

def evaluate_day(target_date, prev_date, prev2_date, daily_dict):
    print(f"\n================ EVALUATING TARGET DATE: {target_date} (Yesterday = {prev_date}) ================")
    matched = []
    failed_reasons = {}
    
    for code, info in daily_dict.items():
        # Check mainboard
        if not (code.startswith('000') or code.startswith('001') or code.startswith('002') or code.startswith('003') or code.startswith('600') or code.startswith('601') or code.startswith('603') or code.startswith('605')):
            continue
            
        days = info.get('days', {})
        d_curr = days.get(target_date)
        d_prev1 = days.get(prev_date)
        d_prev2 = days.get(prev2_date)
        
        if not d_curr or not d_prev1 or not d_prev2:
            continue
            
        c_curr = d_curr['close'] / 100.0
        o_curr = d_curr['open'] / 100.0
        h_curr = d_curr['high'] / 100.0
        
        c_p1 = d_prev1['close'] / 100.0
        h_p1 = d_prev1['high'] / 100.0
        v_p1 = d_prev1.get('volume_native', 0)
        
        c_p2 = d_prev2['close'] / 100.0
        h_p2 = d_prev2['high'] / 100.0
        
        # Check ZT1 (yesterday limit up)
        zt1 = (c_p1 / c_p2 >= 1.098) and (c_p1 == h_p1)
        # Check ZT2 (day before yesterday limit up)
        # To check ZT2, we need prev3
        # Let's check streak
        
        # Gap
        gap_pct = (o_curr - c_p1) / c_p1 * 100.0
        golden_gap = (gap_pct >= 3.5) and (gap_pct <= 7.2)
        not_one = o_curr < c_p1 * 1.098
        break_high = o_curr >= h_p1
        
        # Let's inspect stocks that had zt1
        if zt1:
            matched.append({
                'code': code,
                'o_curr': o_curr,
                'c_curr': c_curr,
                'c_p1': c_p1,
                'gap_pct': gap_pct,
                'golden_gap': golden_gap,
                'not_one': not_one,
                'break_high': break_high,
                'days': days
            })
            
    print(f"Total stocks with ZT on {prev_date}: {len(matched)}")
    for m in matched:
        print(f"[{m['code']}]: 昨收={m['c_p1']:.2f}, 今开={m['o_curr']:.2f} (高开={m['gap_pct']:+.2f}%), 黄金高开[3.5-7.2]={m['golden_gap']}, 非一字={m['not_one']}")

evaluate_day('2026-10-27', '2026-10-26', '2026-10-23', daily_27)
evaluate_day('2026-10-28', '2026-10-27', '2026-10-26', daily_28)
