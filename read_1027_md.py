import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

folder = 'results/daily_review_20261027T0928Z'
for f in os.listdir(folder):
    if f.endswith('.md'):
        with open(os.path.join(folder, f), encoding='utf-8') as fp:
            print(f'=== {f} ===')
            lines = fp.readlines()
            for l in lines[:45]:
                print(l, end='')
