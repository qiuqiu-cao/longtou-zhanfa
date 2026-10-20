"""One bounded evening delta check after the user's delayed continuation."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import short_term_thinking_dataset_builder as b
OUT=Path(__file__).resolve().parent
def main():
 old=b.read(OUT/'facts/notices/index_manifest.json');known={r['code']:{x['art_code'] for x in r.get('items',[])} for r in old['rows']}
 codes=sorted({r['code'] for r in b.read(OUT/'facts/qualified_current_facts_SCORES_MASKED.json')['relay']}|{'600487','600825','002343','605255'})
 dest=OUT/'facts/evening_refresh_20261019T142830Z';(dest/'notices').mkdir(parents=True,exist_ok=False)
 spec=importlib.util.spec_from_file_location('evening_notice_delta',OUT/'facts/qualify_and_notices.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.OUT=dest
 with ThreadPoolExecutor(max_workers=6) as executor:received=list(executor.map(module.notices,codes))
 new=[];jobs=[]
 for entry in received:
  delta=[r for r in entry.get('items',[]) if r['art_code'] not in known.get(entry['code'],set())]
  new.append(dict(code=entry['code'],status=entry['status'],new_articles=delta,baseline_had_index=bool(entry['code'] in known)))
  for item in delta[:2]:jobs.append((entry['code'],item))
 b.write_new(dest/'notices/index_manifest.json',dict(formed_at=b.now(),signal_date='2026-10-19',target_date='2026-10-20',codes=codes,rows=received,max_attempts_per_code=1,previous_receipt=str((OUT/'facts/notices/index_manifest.json').resolve())))
 with ThreadPoolExecutor(max_workers=6) as executor:pdfs=list(executor.map(module.pdf_one,jobs))
 b.write_new(dest/'notices/new_fulltext_manifest.json',dict(formed_at=b.now(),rows=pdfs,max_attempts_per_article=1))
 result=dict(formed_at=b.now(),original_signal_snapshot_preserved=True,rows=new,documents=pdfs,new_document_count=len(pdfs),new_fits=0,original_predictions_changed=False,source_scope='Issuer notice index mirror delta only, no guarantee of exchange-wide disclosure clearance')
 b.write_new(dest/'evening_delta_receipt.json',result)
 print(json.dumps(dict(formed_at=result['formed_at'],delta=new,pdfs=[{k:r.get(k) for k in ['code','title','status','text_path']} for r in pdfs]),ensure_ascii=False))
if __name__=='__main__':main()
