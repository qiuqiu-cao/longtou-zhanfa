import os, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('results/daily_review_20261027T0928Z/20261027复盘与1028预期_完整双榜.md', encoding='utf-8') as fp:
    lines = fp.readlines()
    for idx, l in enumerate(lines[45:180]):
        print(f'{idx+46}: {l}', end='')
