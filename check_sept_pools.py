import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

em_dir = Path('data/2026-09_eastmoney')
codes = {
    '002084': '海鸥住工', '2084': '海鸥住工',
    '002855': '捷荣技术', '2855': '捷荣技术',
    '600371': '万向德农',
    '000712': '锦龙股份', '712': '锦龙股份',
    '002396': '星网锐捷', '2396': '星网锐捷',
    '002886': '沃特股份', '2886': '沃特股份',
    '600540': '新赛股份',
    '603269': '海鸥股份'
}

for f in sorted(em_dir.glob('zt_pool_202609*.json')):
    date = f.stem.replace('zt_pool_', '')
    data = json.loads(f.read_text(encoding='utf-8'))
    pool = data.get('data', {}).get('pool', []) if isinstance(data, dict) else []
    found = []
    for r in pool:
        c = str(r.get('c', ''))
        if c in codes:
            found.append(f"{codes[c]}({c}): {r.get('lbc', 1)}板")
    if found:
        print(f"{date}: {found}")
