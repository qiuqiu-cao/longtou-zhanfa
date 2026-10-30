import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/first_board_causal_theme_20261030'

def test_archived_source_binding_and_candidate_preservation():
    manifest=json.loads((OUT/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in manifest['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
    frame=pd.read_csv(OUT/'S1015_ALL23_ARCHIVED_TEXT_NO_OUTCOMES.csv',dtype={'code':str})
    assert len(frame)==23 and frame.code.nunique()==23
    assert set(frame['rank'])==set(range(1,24))
    assert frame.source_row.notna().all()
    assert not {'outcome','reach4','target_close','target_one_price'} & set(frame.columns)
    assert frame.loc[frame.code.eq('003029'),'rank'].iloc[0]==14

def test_cohort_retains_all_five_and_future_is_separate():
    known=pd.read_csv(OUT/'FIXED_NAME_COHORT_1013_THROUGH_1015.csv',dtype={'code':str})
    future=pd.read_csv(OUT/'FIXED_NAME_COHORT_1016_FUTURE_LABEL_ONLY.csv',dtype={'code':str})
    assert len(known)==15 and len(future)==5
    assert set(known.date)=={'2026-10-13','2026-10-14','2026-10-15'}
    assert set(future.date)=={'2026-10-16'}
    assert all(len(g)==5 for _,g in known.groupby('date'))
    assert set(future.code)==set(known.code)
    assert known[known.late_page_state.eq('not_on_event_page_unknown')].absence_is_unknown.all()

def test_original_snow_cohort_tracks_every_original_member_without_future_rows():
    known=pd.read_csv(OUT/'ORIGINAL_SNOW_COHORT_1013_1015.csv',dtype={'code':str})
    future=pd.read_csv(OUT/'ORIGINAL_SNOW_COHORT_1016_FUTURE_ONLY.csv',dtype={'code':str})
    manifest=json.loads((OUT/'SNOW_COHORT_MANIFEST.json').read_text(encoding='utf-8'))
    assert manifest['original_mainline_status']=='UNKNOWN'
    assert len(known)==21 and len(future)==7
    assert set(future.code)==set(manifest['group_members'])
    assert all(set(group.code)==set(manifest['group_members']) for _,group in known.groupby('signal_date'))
    assert set(known.signal_date)=={'2026-10-13','2026-10-14','2026-10-15'}
    for name,expected in manifest['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
