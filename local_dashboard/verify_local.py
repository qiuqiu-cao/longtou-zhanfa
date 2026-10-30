"""HTTP/data acceptance checks; no training and no external network requests."""
from pathlib import Path
import json,urllib.request,urllib.error,hashlib
import data
HERE=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8766'
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def request(path):
    try:
        r=opener.open(BASE+path,timeout=8);return r.status,r.read(),dict(r.headers)
    except urllib.error.HTTPError as e:return e.code,e.read(),dict(e.headers)
checks={}
for path in ['/','/style.css','/app.js','/api/health','/api/catalog','/api/report?date=2026-10-23','/api/report?date=2026-10-22','/evidence/2026-10-23/relationship.png']:
    status,body,headers=request(path);assert status==200,(path,status,body[:100]);checks[path]=dict(status=status,bytes=len(body))
for path in ['/api/report?date=2099-01-01','/%2e%2e/AGENTS.md','/.runtime/access_key.txt','/server.py']:
    status,body,_=request(path);assert status==404,(path,status);checks[path]=dict(status=status)
report=json.loads(request('/api/report?date=2026-10-23')[1]);original=data.read(data.archives()['2026-10-23']/'BASE_scores_AFTER_INDEPENDENT.json')
assert report['mode']=='ARCHIVE_ONLY' and len(report['rows'])==81
assert report['counts']['first']==53 and report['counts']['relay']==11
assert report['relay_model_top3']==['603028','603933','603280']
assert {r['code']:r['base'] for r in report['rows'] if r['base']}=={r['code']:r for r in original['rows']}
z=next(r for r in report['rows'] if r['code']=='603089')
assert z['base']['BASE_direction']=='NO' and z['base']['rank']==6
assert '分歧观察' in z['card']['gate'] and '第一主选' not in z['card'].get('slot','')
assert '不依据' in z['card']['after0930'] and report['correction']['relay_current_ranked_recommendations']==[]
assert all(r['base'] is None for r in report['rows'] if r['kind']=='limit_up' and r['boards']==1)
before=data.read(HERE/'.runtime/protected_before.json')['files']
assert all(data.digest(data.ROOT/p)==sha for p,sha in before.items())
out=dict(verified_at=data.now(),http_checks=checks,complete_rows=81,raw_BASE_rows_unchanged=11,
    source_hashes_unchanged=len(before),zhengyu_recommendation_withdrawal_present=True,
    firstboard_fake_scores=0,mode='ARCHIVE_ONLY',new_fits=0,orders=0,paid_calls=0,
    mobile_physical_device='NOT_VERIFIED',browser_checks='See browser acceptance receipt',
    not_completed=['实时行情接入','自动每日独立复盘','模型效果提升','手机物理设备连接'])
(HERE/'VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='http_checks'},ensure_ascii=False))
