from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
import importlib.util,json,sys,hashlib,requests
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT)); spec=importlib.util.spec_from_file_location('historical_page_parser',ROOT/'results/joint_review_20261019T092500Z/facts/collect_close.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
cal=json.loads((ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json').read_text(encoding='utf-8'))['sessions']
dates=[d for d in cal if '2026-09-01'<=d<='2026-10-29']
rawdir=OUT/'raw_pages';rawdir.mkdir(exist_ok=True);norm=OUT/'pages';norm.mkdir(exist_ok=True)
def now():return datetime.now(timezone.utc).isoformat()
def get(date):
    rec={'date':date,'kind':'alternative','url':'https://toupage.com/stocks/limit-up','requested_at':now(),'params':{'date':date}}
    try:
        r=requests.get(rec['url'],params=rec['params'],timeout=(6,20),headers={'User-Agent':'Mozilla/5.0'});r.raise_for_status()
        raw=rawdir/f'{date}.html';raw.write_bytes(r.content);rec.update(available_at=now(),http_status=r.status_code,raw_path=str(raw),raw_sha256=hashlib.sha256(r.content).hexdigest(),status='RAW_RECEIVED')
        page=mod.parse_page(rec); (norm/f'{date}.json').write_text(json.dumps(page,ensure_ascii=False,indent=2),encoding='utf-8')
        rec.update(parsed=True,counts=page['raw_pool_counts'])
    except Exception as e: rec.update(parsed=False,error=repr(e),completed_at=now())
    (rawdir/f'{date}.receipt.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
    return rec
items=[]
with ThreadPoolExecutor(max_workers=4) as p:
    for f in as_completed([p.submit(get,d) for d in dates]):
        a=f.result();items.append(a);print(a['date'],'ok' if a['parsed'] else a.get('error'),flush=True)
(OUT/'PAGE_RECEIPTS.json').write_text(json.dumps({'formed_at':now(),'requests':items},ensure_ascii=False,indent=2),encoding='utf-8')
