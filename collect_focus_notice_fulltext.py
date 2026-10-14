"""Archive selected notice PDFs and text; receipt is NOT full-text risk clearance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo
import pandas as pd
import requests


IDS = ['AN202609041829023666','AN202609041829025202','AN202609041829025207',
       'AN202609041829020867','AN202609011828887530']

def run(index, output):
    index, output = Path(index), Path(output)
    source=pd.read_csv(index,dtype={'code':str}).fillna('')
    rows=[]
    for notice_id in IDS:
        hit=source[source.url.str.contains(notice_id,regex=False)]
        if len(hit)!=1:
            raise ValueError(f'Expected one archived index match: {notice_id}')
        rows.append(dict(hit.iloc[0],notice_id=notice_id))
    output.mkdir(parents=True,exist_ok=False)
    def fetch(row):
        stamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        url=f"https://pdf.dfcfw.com/pdf/H2_{row['notice_id']}_1.pdf"
        result=dict(code=row['code'],title=row['title'],index_url=row['url'],pdf_url=url,
                    notice_id=row['notice_id'],index_published_at=row['published_at'],
                    published_at_status=row['published_at_status'],request_started_at=stamp,
                    source_kind='issuer_or_regulator_document_on_eastmoney_mirror',
                    review_status='UNREVIEWED',entry_authorized=False)
        try:
            response=requests.get(url,timeout=25)
            response.raise_for_status()
            if not response.content.startswith(b'%PDF'):
                raise ValueError('not PDF')
            pdf=output/(row['notice_id']+'.pdf')
            pdf.write_bytes(response.content)
            txt=pdf.with_suffix('.txt')
            subprocess.run(['pdftotext','-layout',str(pdf),str(txt)],check=True,capture_output=True,timeout=30)
            text=txt.read_text(encoding='utf-8')
            if len(text.strip())<30:
                raise ValueError('text unavailable; requires OCR')
            result.update(receipt_status='PDF_AND_TEXT_RECEIVED',pdf_sha256=hashlib.sha256(response.content).hexdigest(),
                          text_sha256=hashlib.sha256(txt.read_bytes()).hexdigest(),text_length=len(text),
                          pdf_path=str(pdf.resolve()),text_path=str(txt.resolve()))
        except Exception as exc:
            result.update(receipt_status='UNAVAILABLE',error=str(exc))
        result['received_at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        return result
    with ThreadPoolExecutor(max_workers=3) as ex:
        results=list(ex.map(fetch,rows))
    report=dict(index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),notices=results,
                limitation='PDF retrieval/extraction does not validate document authenticity on exchange domain, effective dates, or authorize a trade; manual source-bound review required.')
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    print(json.dumps(run(a.index,a.output),ensure_ascii=False,indent=2))
