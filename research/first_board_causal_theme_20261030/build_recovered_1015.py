"""Join a previously archived 15 October event page to the later S1 reconstruction.

This is a source audit and descriptive case ledger, not a model feature or fit.
"""
import csv
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    source_dir=ROOT/'research/model_upgrade_validation/historical_symbolic_context_20261015T081200Z/latest15_alternative_page_once'
    receipt=json.loads((source_dir/'receipt.json').read_text(encoding='utf-8'))
    raw=source_dir/'page.raw.html'
    if sha(raw)!=receipt['raw_sha256']: raise ValueError('archived raw source hash mismatch')
    if receipt['intended_signal_date']!='2026-10-15': raise ValueError('source date mismatch')
    src=ROOT/'results/daily_review_20261015T093600Z/完整收盘首板炸板跌停池.csv'
    verification=json.loads((src.parent/'completion_verification.json').read_text(encoding='utf-8'))
    expected=next((h for p,h in verification['artifacts'].items() if p.endswith(src.name)),None)
    if sha(src)!=expected: raise ValueError('derived source hash mismatch')
    pool_path=ROOT/'research/first_board_dynamic_narrative_review_20261030/corrected/CASE_POOL_NO_LABELS.csv'
    p=pd.read_csv(pool_path,dtype={'code':str},float_precision='round_trip')
    p=p[p.signal_date.eq('2026-10-15')].copy()
    events=pd.read_csv(src,dtype={'code':str})
    if set(events.signal_date)!= {'2026-10-15'}: raise ValueError('mixed date source')
    up=events[events.pool.eq('limit_up')]
    if up.code.duplicated().any(): raise ValueError('duplicate archived limit-up code')
    joined=p.merge(up[['code','reported_boards','unverified_AI_summary_NOT_COMPANY_FACT','source_row','source_sha256']],on='code',how='left',validate='one_to_one',indicator=True)
    if len(joined)!=23 or not joined['_merge'].eq('both').all(): raise ValueError('candidate mapping incomplete')
    joined=joined.drop(columns='_merge')
    joined['source_obtained_at']=receipt['received_at']
    joined['source_scope_note']='16:30 archival provider output; business summary unverified; 10/30 S1 reconstruction'
    # Pool includes no outcome labels and no extracted theme scores.
    out=OUT/'S1015_ALL23_ARCHIVED_TEXT_NO_OUTCOMES.csv'
    ledger=OUT/'FIXED_NAME_COHORT_1013_THROUGH_1015.csv'
    future=OUT/'FIXED_NAME_COHORT_1016_FUTURE_LABEL_ONLY.csv'
    manifest=OUT/'SOURCE_MANIFEST.json'
    if any(x.exists() for x in (out,ledger,future,manifest)): raise FileExistsError('do not overwrite')
    joined.to_csv(out,index=False,encoding='utf-8-sig',mode='x')
    # Five stocks were described together in a 14 October archive; selection is
    # not contemporaneously frozen on the 13th and is not full-market membership.
    codes={'002696':'百洋股份','001318':'阳光乳业','002949':'华阳国际','600281':'华阳新材','002172':'澳洋健康'}
    late=ROOT/'research/october_2026_s1_blind_replay_20261030/pages'
    timeline=[]
    for day in ('2026-10-13','2026-10-14','2026-10-15','2026-10-16'):
        page=json.loads((late/f'{day}.json').read_text(encoding='utf-8'))
        page_rows=page['rows']
        for code,name in codes.items():
            matches=[r for r in page_rows if r['code']==code]
            if len(matches)>1: raise ValueError('ambiguous late event page')
            event=matches[0] if matches else None
            a=events[events.code.eq(code)] if day=='2026-10-15' else pd.DataFrame()
            if len(a)>1: raise ValueError('ambiguous actual archive event')
            archived='not_on_event_page_unknown' if a.empty else a.iloc[0]['pool']
            timeline.append({'date':day,'code':code,'name':name,
                'late_page_state':event['kind'] if event else 'not_on_event_page_unknown',
                'late_page_boards':event.get('boards') if event else None,
                'late_page_row_sha256':event.get('raw_row_sha256') if event else '',
                'late_page_obtained_at':page['available_at'],
                'same_day_archive_state':archived if day=='2026-10-15' else 'not_joined',
                'same_day_archive_obtained_at':receipt['received_at'] if day=='2026-10-15' else '',
                'cohort_selection':'14 Oct retrospective name-character shortlist, not complete verified theme',
                'absence_is_unknown':True})
    cols=list(timeline[0])
    for target, rows in ((ledger,[r for r in timeline if r['date']!='2026-10-16']),
                         (future,[r for r in timeline if r['date']=='2026-10-16'])):
        with target.open('x',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    paths=[raw,source_dir/'receipt.json',src,src.parent/'completion_verification.json',pool_path,
       ROOT/'results/deep_review_20261014T115540Z/jiuyang_baiyang_relation_correction_20261014T121615Z/relation_review.json',
       ROOT/'results/sunshine_jiuyang_relation_20261014T152814Z/阳光九阳补涨关系与候选比较.md',
       *(late/f'{d}.json' for d in ('2026-10-13','2026-10-14','2026-10-15','2026-10-16'))]
    info={'status':'DESCRIPTIVE_RECONSTRUCTION_NO_NEW_RANKING_OR_FIT','archived_15_source_received_at':receipt['received_at'],
          'candidate_rows':len(joined),'candidate_matches':23,'cohort_rows':15,'future_rows':5,
          'future_event_file_is_not_input':True,'theme_membership_is_verified':False,
          'retrospective_s1_not_original_ranking':True,'missing_event_is_not_failure':True,
          'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in paths}}
    with manifest.open('x',encoding='utf-8') as f: json.dump(info,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:info[k] for k in ('candidate_rows','candidate_matches','cohort_rows','future_rows','archived_15_source_received_at')}))

if __name__=='__main__': main()
