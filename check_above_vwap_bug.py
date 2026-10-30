import json

with open('results/daily_review_20261027T0928Z/facts/normalized_daily.json', 'r', encoding='utf-8') as f:
    daily = json.load(f)

# Let's check 000678 on 10-26:
d = daily.get('000678', {}).get('days', {})
d26 = d.get('2026-10-26')
d27 = d.get('2026-10-27')

with open('results/daily_review_20261026T1323Z/facts/complete_enriched_facts_SCORES_MASKED.json', 'r', encoding='utf-8') as f:
    facts_26 = json.load(f)
    for r in facts_26.get('rows', []):
        if r['code'] == '000678':
            amt_26 = r.get('amount_cny')
            vol_native_26 = d26.get('volume_native')
            # In Tonghuashun, if volume_native is in 手 (lots):
            # OPEN on 10-27 is 15.24
            open_27 = d27['open'] / 100.0
            print("amt_26:", amt_26)
            print("vol_native_26:", vol_native_26)
            print("open_27:", open_27)
            # If in Tonghuashun VOL is in 手:
            # open_27 * vol_native_26 * 100
            val1 = open_27 * vol_native_26 * 100
            print("val1 (open * vol * 100):", val1)
            print("amt_26:", amt_26)
            print("val1 >= amt_26:", val1 >= amt_26)
            
            # BUT what if in Tonghuashun VOL is ALREADY in shares (股)?
            val2 = open_27 * vol_native_26
            print("val2 (open * vol):", val2)
            print("val2 >= amt_26:", val2 >= amt_26)
