"""Follow the actual 13 October archived provider-category members as a group.

Members come from the pre-existing source, not from later winners. Later page
absence is not a negative observation. No output ranks or model features.
"""
from pathlib import Path
import csv
import hashlib
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    mainline_path=ROOT/'results/daily_review_20261013T075300Z/mainline/report.json'
    mainline=json.loads(mainline_path.read_text(encoding='utf-8'))
    group=[x for x in mainline['candidates'] if x['theme_name']=='冰雪经济']
    if len(group)!=1: raise ValueError('expected exactly one original snow cohort')
    g=group[0]; codes=g['member_codes']
    if len(codes)!=7 or len(set(codes))!=7: raise ValueError('original cohort changed')
    today_path=ROOT/'results/daily_review_20261015T093600Z/完整收盘首板炸板跌停池.csv'
    today=pd.read_csv(today_path,dtype={'code':str})
    late=ROOT/'research/october_2026_s1_blind_replay_20261030/pages'
    rows=[]
    for day in ('2026-10-13','2026-10-14','2026-10-15','2026-10-16'):
        p=json.loads((late/f'{day}.json').read_text(encoding='utf-8'))
        for code in codes:
            m=[x for x in p['rows'] if x['code']==code]
            if len(m)>1: raise ValueError('multiple same-code event rows')
            a=today[today.code.eq(code)] if day=='2026-10-15' else pd.DataFrame()
            if len(a)>1: raise ValueError('ambiguous same-day event')
            rows.append({'signal_date':day,'code':code,'name':m[0]['name'] if m else '',
                'original_group':'13 Oct provider ice-snow candidate, not independently verified full theme',
                'original_group_formed_at':mainline['knowledge_cutoff'],
                'late_state':m[0]['kind'] if m else 'not_on_returned_event_list_unknown',
                'late_boards':m[0]['boards'] if m else None,
                'late_raw_row_sha256':m[0]['raw_row_sha256'] if m else '',
                'late_obtained_at':p['available_at'],
                'original_15_state':('not_on_returned_event_list_unknown' if a.empty else a.iloc[0]['pool']) if day=='2026-10-15' else 'not_joined',
                'source_group_status':mainline['status'],'absence_is_unknown':True})
    # Codes remain available on every row; names recovered from original seed
    # page only, not inferred from later appearances.
    names={r['code']:r['name'] for r in rows if r['signal_date']=='2026-10-13' and r['name']}
    if set(names)!=set(codes): raise ValueError('original cohort identities incomplete')
    for r in rows: r['name']=names[r['code']]
    known=OUT/'ORIGINAL_SNOW_COHORT_1013_1015.csv'
    future=OUT/'ORIGINAL_SNOW_COHORT_1016_FUTURE_ONLY.csv'
    manifest=OUT/'SNOW_COHORT_MANIFEST.json'
    if any(p.exists() for p in (known,future,manifest)): raise FileExistsError('refusing overwrite')
    for path,subset in ((known,[r for r in rows if r['signal_date']!='2026-10-16']),
                        (future,[r for r in rows if r['signal_date']=='2026-10-16'])):
        with path.open('x',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(subset)
    paths=[mainline_path,today_path,*(late/f'{d}.json' for d in ('2026-10-13','2026-10-14','2026-10-15','2026-10-16'))]
    info={'group_members':codes,'original_as_of':mainline['knowledge_cutoff'],'original_mainline_status':mainline['status'],
          'present_day_group_is_complete_theme':False,'original_15_event_page_received_at':'2026-10-15T08:30:43.245783+00:00',
          'previous_day_groups_without_actual_time_archive_are_late_reconstruction':True,
          'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in paths}, 'new_ranking_or_fit':False}
    with manifest.open('x',encoding='utf-8') as f:json.dump(info,f,ensure_ascii=False,indent=2)
    print(json.dumps({'members':len(codes),'observations':len(rows),'future_rows':7}))

if __name__=='__main__':main()
