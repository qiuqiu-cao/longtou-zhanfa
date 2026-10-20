"""One bounded public read per cited page; archive actual responses, no models."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import requests,hashlib,json
from bs4 import BeautifulSoup

OUT=Path(__file__).resolve().parent/'source_checks'
SOURCES={
 'early': 'https://finance.eastmoney.com/a/202610203900809160.html',
 'mid': 'https://www.stcn.com/article/detail/4227544.html',
 'close': 'https://www.stcn.com/article/detail/4227803.html',
 'daye_notice': 'https://www.cfi.net.cn/p20261020001638.html',
 'forum_liaoyin': 'https://m.tgb.cn/a/2w30IRKw3VY',
 'forum_gudu': 'https://www.tgb.cn/a/2w345GlTRHJ',
}
def now():return datetime.now(timezone.utc).isoformat()
def capture(pair):
 key,url=pair;meta={'key':key,'url':url,'requested_at_utc':now()}
 try:
  r=requests.get(url,timeout=(5,12),headers={'User-Agent':'Mozilla/5.0'})
  meta.update(received_at_utc=now(),status=r.status_code,response_url=r.url,sha256=hashlib.sha256(r.content).hexdigest(),bytes=len(r.content))
  r.encoding=r.apparent_encoding
  soup=BeautifulSoup(r.text,'html.parser')
  for n in soup(['script','style','nav','footer']):n.decompose()
  value=soup.get_text('\n',strip=True)
  meta['title']=soup.title.get_text(' ',strip=True) if soup.title else None
  meta['not_automatically_verified']=True
  return key,meta,r.content,value
 except Exception as e:
  meta.update(received_at_utc=now(),error=str(e),not_automatically_verified=True)
  return key,meta,None,None
def main():
 OUT.mkdir(exist_ok=True)
 assert not (OUT/'manifest.json').exists()
 with ThreadPoolExecutor(max_workers=3) as ex:results=list(ex.map(capture,SOURCES.items()))
 manifest=[];summaries=[]
 for key,meta,raw,value in results:
  manifest.append(meta)
  if raw is not None:
   with (OUT/(key+'.raw.html')).open('xb') as f:f.write(raw)
   with (OUT/(key+'.txt')).open('x',encoding='utf-8') as f:f.write(value)
   lines=value.splitlines();snippets=[]
   for i,line in enumerate(lines):
    if any(word in line for word in ['东安','大业','亦未涉及','了因1991','孤渡者']):
     snippets.append('\n'.join(lines[max(0,i-1):i+12]))
   summaries.append(dict(key=key,meta=meta,snippets=snippets[:10]))
  else:summaries.append(meta)
 with (OUT/'manifest.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
 print(json.dumps(summaries,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
