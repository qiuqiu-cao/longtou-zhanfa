import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('results/daily_review_20261027T0928Z/BASE_scores_AFTER_INDEPENDENT.json', encoding='utf-8') as f:
    d = json.load(f)

print('=== 10-27 BASE SCORES ===')
for r in d['rows']:
    print(f"{r['code']} | {r['name']:8s} | boards={r['boards']} | score={r.get('score_display')} | BASE={r.get('BASE_direction')} | indep={r.get('independent_direction')}")

with open('results/daily_review_20261027T0928Z/facts/qualified_full_facts_SCORES_MASKED.json', encoding='utf-8') as f:
    q = json.load(f)

print('\n=== QUALIFIED FACTS SUMMARY ===')
print('Total rows:', len(q['rows']))
boards_dist = {}
for r in q['rows']:
    b = r.get('boards', 1)
    boards_dist[b] = boards_dist.get(b, 0) + 1
print('Boards distribution:', boards_dist)
