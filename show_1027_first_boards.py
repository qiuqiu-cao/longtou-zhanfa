import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('results/daily_review_20261027T0928Z/facts/qualified_full_facts_SCORES_MASKED.json', encoding='utf-8') as f:
    q = json.load(f)

print('=== 1-BOARD CANDIDATES (10-27) ===')
first_boards = [r for r in q['rows'] if r.get('boards') == 1]
for r in first_boards:
    print(f"{r['code']} | {r['name']:8s} | close={r.get('close')} | turnover={r.get('turnover_rate')}% | first_seal={r.get('first_seal_time')} | open_times={r.get('open_times')} | theme={r.get('theme')}")

with open('results/daily_review_20261027T0928Z/facts/raw/2026-10-27_limit_up.json', encoding='utf-8', errors='ignore') as f:
    raw_zt = json.load(f)
print('\nTotal in limit up pool:', len(raw_zt['data']['pool']))
