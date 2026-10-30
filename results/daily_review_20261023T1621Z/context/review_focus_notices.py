"""Bounded 9-document issuer notice read; no old artifact writes."""
import importlib.util, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(r'D:/Documents/ChatGPT/龙头战法')
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('legacy_notice_reader', ROOT / 'results/joint_review_20261019T092500Z/facts/qualify_and_notices.py')
n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n)
n.OUT = OUT
selected = {
 '603278': ['AN202610231830845402'],
 '002724': ['AN202610231830849631'],
 '600838': ['AN202610231830847770'],
 '603089': ['AN202610231830839833'],
 '603280': ['AN202610231830844976'],
 '002535': ['AN202610141830463945'],
 '603528': ['AN202610231830844606'],
 '603933': ['AN202610231830849462', 'AN202610231830849461'],
}
manifest = n.read(OUT / 'notices/index_manifest.json')
jobs = [(row['code'], item) for row in manifest['rows'] if row['code'] in selected
        for item in row['items'] if item['art_code'] in selected[row['code']]]
assert len(jobs) == 9
for code, item in jobs:
    path = OUT / 'notices' / (code + '_' + item['art_code'] + '.pdf')
    if path.exists() or path.with_suffix('.txt').exists():
        raise RuntimeError('Refuse overwrite: ' + str(path))
with ThreadPoolExecutor(max_workers=5) as executor:
    documents = list(executor.map(n.pdf_one, jobs))
n.write(OUT / 'notices/focus_9_fulltext_manifest.json', {'formed_at': n.now(), 'signal_date':'2026-10-23', 'target_date':'2026-10-26', 'limit':9, 'rows': documents, 'post_close_information_separate':True, 'no_regulatory_clearance_inferred':True})
print(json.dumps([{'code':r['code'],'title':r['title'],'status':r['status'],'error':r.get('error')} for r in documents], ensure_ascii=False))
