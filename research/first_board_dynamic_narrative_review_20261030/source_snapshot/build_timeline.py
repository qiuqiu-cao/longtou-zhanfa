"""Extract a bounded, explicitly incomplete historical event-page timeline."""
from pathlib import Path
import json
import hashlib
import csv

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
PAGES = ROOT / 'research/october_2026_s1_blind_replay_20261030/pages'
DATES = ('2026-10-08','2026-10-09','2026-10-12','2026-10-13','2026-10-14',
         '2026-10-15','2026-10-16','2026-10-19','2026-10-20')
NAMES = ('九阳股份','百洋股份','海洋王','吉大正元','阳光乳业','海欣食品',
         '林州重机','金奥博','长春一东','甘化科工','读者传媒','内蒙新华','文投控股','慈文传媒')

def main():
    output = DEST / 'CASE_EVENT_TIMELINE_LATE_SOURCE.csv'
    receipt = DEST / 'TIMELINE_SOURCES.json'
    if output.exists() or receipt.exists():
        raise FileExistsError('refusing to overwrite prior evidence')
    result=[]; manifest=[]
    for day in DATES:
        p=PAGES / f'{day}.json'
        obj=json.loads(p.read_text(encoding='utf-8'))
        manifest.append({'date':day,'parsed_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                         'raw_sha256':obj['source_sha256'],'retrieved_at':obj['available_at'],
                         'source_path':str(p.relative_to(ROOT))})
        for name in NAMES:
            matches=[r for r in obj['rows'] if r['name']==name]
            if len(matches)>1:
                raise ValueError(f'duplicate event row: {day} {name}')
            r=matches[0] if matches else {}
            result.append({'date':day,'name':name,'code':r.get('code',''),
                           'event_page_state':r.get('kind','NOT_IN_RETURNED_EVENT_LIST_UNKNOWN'),
                           'reported_boards':r.get('boards'), 'reported_times':r.get('reported_times',''),
                           'reported_reason_UNVERIFIED':r.get('reason_summary_UNVERIFIED',''),
                           'row_sha256':r.get('raw_row_sha256',''), 'retrieved_at':obj['available_at']})
    with output.open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=result[0].keys()); w.writeheader(); w.writerows(result)
    receipt.write_text(json.dumps({'dates':len(DATES),'names':len(NAMES),'rows':len(result),
      'source_type':'retrospectively fetched event page; not a full-market universe or real-time archive',
      'absence_means':'not on returned event list; no inference about price, attention or stock trading',
      'ranking_or_fitting':False,'sources':manifest},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'rows':len(result),'present_events':sum(r['row_sha256']!='' for r in result)},ensure_ascii=False))

if __name__=='__main__': main()
