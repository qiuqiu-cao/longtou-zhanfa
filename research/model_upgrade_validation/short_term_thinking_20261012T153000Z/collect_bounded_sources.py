"""One bounded public request per explicitly reviewed original source."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import requests
from bs4 import BeautifulSoup

OUT = Path(__file__).resolve().parent / 'forum_sources'
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = [
    ('tgb_relative_strength', 'https://tgb.cn/a/2uIZL9baVFa', '2026-09-01T05:20:00+08:00', 'MINUTE', 'ORIGINAL_FORUM'),
    ('eastmoney_cycle_strength', 'https://caifuhao.eastmoney.com/news/20260716062550043397900', '2026-07-16T06:25:00+08:00', 'MINUTE', 'ORIGINAL_FORUM'),
    ('jiuyang_company_disclosure', 'https://data.eastmoney.com/notices/detail/002242/AN202609291830004916.html', '2026-09-30', 'DATE', 'COMPANY_DISCLOSURE_REPRINT'),
]

def fetch(item):
    sid, url, published, precision, kind = item
    entry = dict(source_id=sid, url=url, published_at=published,
                 publication_precision=precision, kind=kind,
                 requested_at=datetime.now(timezone.utc).isoformat())
    try:
        response = requests.get(url, timeout=(10, 25))
        entry.update(obtained_at=datetime.now(timezone.utc).isoformat(),
                     http_status=response.status_code, response_url=response.url)
        raw = OUT / (sid + '.raw.html')
        with raw.open('xb') as handle:
            handle.write(response.content)
        entry.update(raw_path=str(raw), raw_sha256=hashlib.sha256(response.content).hexdigest())
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text('\n', strip=True)
        path = OUT / (sid + '.page_text.txt')
        with path.open('x', encoding='utf-8') as handle:
            handle.write(text)
        entry.update(text_path=str(path), text_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     full_body_verified=False,
                     pdf_links=[a.get('href') for a in soup.find_all('a', href=True)
                                if '.pdf' in a['href'].lower()])
    except Exception as exc:
        entry.update(failed_at=datetime.now(timezone.utc).isoformat(), error=str(exc))
    return entry

results = list(ThreadPoolExecutor(max_workers=3).map(fetch, SOURCES))
payload = dict(formed_at=datetime.now(timezone.utc).isoformat(),
               scope='Three named sources; no retries, comments, media or market collection',
               excluded_from_old_training_and_predictions=True, sources=results)
with (OUT / 'capture_manifest.json').open('x', encoding='utf-8') as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
print(json.dumps([dict(source_id=r['source_id'], status=r.get('http_status'),
                       error=r.get('error'), pdf_links=r.get('pdf_links', [])) for r in results], ensure_ascii=False))
