"""Build label-separated historical case inventory; no ranking or model fitting."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'research/october_2026_s1_blind_replay_20261030'
OUT = Path(__file__).resolve().parent
DATES = ['2026-10-09', '2026-10-15', '2026-10-16']

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    score_path = SRC / 'v3/FULL_POOL_SCORES_UNQUALIFIED_DIAGNOSTIC.csv'
    outcome_path = SRC / 'v3/ORACLE_FULL_POOL_LABEL_ONLY.csv'
    height_path = SRC / 'v3/ALL_POOL_REACH4_LABEL_ONLY_THROUGH_1026.csv'
    scores = pd.read_csv(score_path, dtype={'code': str})
    subset = scores[scores.signal_date.isin(DATES) & scores.research_selection_allowed.eq(True)].copy()
    subset = subset.sort_values(['signal_date', 's1_score', 'code'], ascending=[True, False, True])
    subset['rank'] = subset.groupby('signal_date').cumcount() + 1
    cols = ['signal_date','code','name','rank','s1_score','close','turnover','amount','float_cap','one_price']
    pool = subset[cols]
    assert len(pool) == 92 and pool.groupby('signal_date').size().to_dict() == {'2026-10-09':33,'2026-10-15':23,'2026-10-16':36}
    assert not pool.duplicated(['signal_date','code']).any()
    pool.to_csv(OUT / 'CASE_POOL_NO_LABELS.csv', index=False, encoding='utf-8-sig')
    outcome = pd.read_csv(outcome_path, dtype={'code':str})
    height = pd.read_csv(height_path, dtype={'code':str})
    labels = pool[['signal_date','code']].merge(outcome[['signal_date','code','target','outcome','conflict']], on=['signal_date','code'], how='left', validate='one_to_one')
    labels = labels.merge(height[['signal_date','code','reach4','path']], on=['signal_date','code'], how='left', validate='one_to_one')
    labels.to_csv(OUT / 'CASE_OUTCOMES_DIAGNOSTIC_ONLY.csv',index=False,encoding='utf-8-sig')
    # Entire returned background, including failed boards. It is NOT a verified market-wide universe.
    page_paths = [SRC / f'pages/{d}.json' for d in ['2026-10-08', *DATES]]
    rows=[]
    for p in page_paths:
        obj=json.loads(p.read_text(encoding='utf-8'))
        for x in obj['rows']:
            rows.append({k:x.get(k) for k in ['date','code','name','kind','boards','row','reported_times','reported_open_count','reason_summary_UNVERIFIED','raw_row_sha256']} | {'retrieved_at':obj['available_at'], 'source_sha256':obj['source_sha256']})
    bg=pd.DataFrame(rows)
    bg.to_csv(OUT / 'RETURNED_BACKGROUND_UNVERIFIED.csv', index=False, encoding='utf-8-sig')
    sources=[score_path,outcome_path,height_path,*page_paths,Path(__file__).resolve()]
    audit={'purpose':'historical descriptive inventory; labels not available at signal cutoff; no fit or selection', 'rows':len(pool),'dates':pool.groupby('signal_date').size().to_dict(),'outcomes':labels.groupby('signal_date').outcome.value_counts(dropna=False).to_dict().__str__(),'background_rows':len(bg),'sources':{str(p.relative_to(ROOT)):digest(p) for p in sources},'qualifications':['2026-10-30 reconstructed S1 ranks, not contemporaneous','provider summaries are unverified AI text, not theme proof','background returned page is not complete market-wide universe','after-close 23:55 pages retrieved 2026-10-30; not original live snapshots','future outcome labels only diagnostic','no dynamic membership established or production rank changed']}
    (OUT/'SOURCE_MANIFEST.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'pool_rows':len(pool),'background_rows':len(bg),'dates':audit['dates']},ensure_ascii=False))

if __name__ == '__main__': main()
