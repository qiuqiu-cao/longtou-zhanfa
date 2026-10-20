from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,importlib.util
import requests
from bs4 import BeautifulSoup
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def article(job):
    key,url=job;start=now();r={'key':key,'url':url,'requested_at':start,'max_attempts':1}
    try:
        v=requests.get(url,timeout=(4,8),headers={'User-Agent':'Mozilla/5.0'});p=OUT/'context'/(key+'.html');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v.content)
        r.update(available_at=now(),http_status=v.status_code,raw_path=str(p),sha256=sha(p));v.raise_for_status()
        soup=BeautifulSoup(v.content,'html.parser')
        for tag in soup(['script','style']):tag.decompose()
        text=soup.get_text('\n',strip=True);t=p.with_suffix('.txt');t.write_text(text,encoding='utf-8')
        r.update(text_path=str(t),text_sha256=sha(t),characters=len(text),status='TEXT_RECEIVED_NOT_AUTOMATICALLY_QUALIFIED')
    except Exception as e:r.update(status='UNAVAILABLE',error=str(e)[:250],completed_at=now())
    return r

def notice(code):
    url='https://np-anotice-stock.eastmoney.com/api/security/ann'
    params=dict(sr='-1',page_size='20',page_index='1',ann_type='A',client_source='web',stock_list=code,begin_time='2026-10-19',end_time='2026-10-21')
    r=dict(code=code,url=url,parameters=params,requested_at=now(),reason='Check public evening documents bearing next nominal notice date; receipt time, not notice date, establishes knowledge. No next-session prices requested.',max_attempts=1)
    try:
        v=requests.get(url,params=params,timeout=(4,8));p=OUT/'context'/(code+'_evening_index.json');p.write_bytes(v.content)
        r.update(available_at=now(),raw_path=str(p),sha256=sha(p),http_status=v.status_code);v.raise_for_status();items=(v.json().get('data')or{}).get('list')
        if not isinstance(items,list):raise ValueError('missing list')
        r.update(status='INDEX_RECEIVED',items=[{k:x.get(k) for k in ['art_code','title','notice_date']} for x in items])
    except Exception as e:r.update(status='UNAVAILABLE',error=str(e)[:200])
    return r

def main():
    jobs=[('forum_19_rates','https://www.tgb.cn/a/2w1r0zXNKSD'),('forum_19_counter','https://www.tgb.cn/a/2w1lHvU4cKB'),('market_thepaper','https://m.thepaper.cn/newsDetail_forward_34289151'),('market_cls_reprint','https://finance.sina.com.cn/roll/2026-10-20/doc-inivwitp6506343.shtml')]
    with ThreadPoolExecutor(max_workers=4) as e:rows=list(e.map(article,jobs))
    write(OUT/'context/capture_manifest.json',dict(formed_at=now(),rows=rows,preferred_author_search='Bounded exact-date searches for 延边刺客/毛竹下的橙子/林曜01/凯域竟知 yielded no qualified current20 body; latest relevant19/earlier sources remain background, not absence of catalyst.'))
    facts=json.loads((OUT/'facts/qualified_full_facts_SCORES_MASKED.json').read_text(encoding='utf-8'))
    codes=sorted({r['code'] for r in facts['relay']})
    with ThreadPoolExecutor(max_workers=6) as e:indexes=list(e.map(notice,codes))
    write(OUT/'context/evening_index_manifest.json',dict(formed_at=now(),rows=indexes))
    spec=importlib.util.spec_from_file_location('old_notice_functions',ROOT/'results/joint_review_20261019T092500Z/facts/qualify_and_notices.py')
    n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n);n.OUT=OUT/'context';(n.OUT/'notices').mkdir(exist_ok=True)
    previous=json.loads((OUT/'facts/notices/fulltext_manifest.json').read_text(encoding='utf-8'))
    previous_codes={Path(x.get('pdf_path','')).stem.split('_',1)[-1] for x in previous['rows'] if x.get('pdf_path')}
    failed_urls={x['url'] for x in previous['rows'] if x['status']=='UNAVAILABLE'}
    todo=[]
    for r in indexes:
        for it in r.get('items',[]):
            if any(w in it['title'] for w in ['异常','风险提示','业绩','停牌','复牌','减持']) and it['art_code'] not in previous_codes:
                url=f"https://pdf.dfcfw.com/pdf/H2_{it['art_code']}_1.pdf"
                if url not in failed_urls:todo.append((r['code'],it))
    with ThreadPoolExecutor(max_workers=5) as e:docs=list(e.map(n.pdf_one,todo))
    write(OUT/'context/evening_fulltext_manifest.json',dict(formed_at=now(),rows=docs,already_failed_documents_not_retried=True))
    print(json.dumps(dict(context=rows,new_docs=docs),ensure_ascii=False))

if __name__=='__main__':main()
