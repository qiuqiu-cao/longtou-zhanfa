"""Independent pre-fit replay and mixed-known-anchor check; no labels or HTTP."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import first_board_lifecycle_status_features_v2 as f

OUT = Path(__file__).resolve().parent
PRIOR = ROOT / 'research/gemini_codex_joint_theme_feedback_design_20261021T074448Z'
DESIGN = ROOT / 'research/gemini_codex_joint_lifecycle_status_design_20261021T083600Z'

def fixture():
    ss = [f'D{i}' for i in range(10)]
    c = pd.DataFrame([dict(signal_date='D9', code=x, historical_name=name, name_ok=True)
                      for x, name in [('100001', '洋甲'), ('100002', '洋乙')]])
    n = pd.DataFrame([dict(signal_date=ss[p], p=p, code=code, historical_name=name,
                          name_ok=True, height=3+p-6)
                      for p in [6, 7, 8]
                      for code, name in [('100010', '洋锚'), ('100011', '洋旧')]])
    seal = pd.DataFrame([dict(signal_date='D9', code=x,
                     reported_first_seal_trading_minutes=0.,
                     reported_close_seal_log_amount_ratio=2.) for x in c.code])
    st = pd.DataFrame([dict(p=p, code=code, eligible=not(p==8 and code=='100011'),
                      valid_price=True, regular_price=True, day_return=.1,
                      limit=True, height=max(0, p-3))
                      for p in range(10) for code in ['100010', '100011']])
    return c,n,seal,st,ss

def main():
    fx = f.build(*fixture())
    c = pd.read_csv(PRIOR/'COMPLETE28503_CANDIDATE_HISTORICAL_NAMES_NO_TARGETS.csv',
                    dtype={'code':str}, float_precision='round_trip')
    n = pd.read_csv(PRIOR/'ALL_DECLARED_ARCHIVE_CLOSING_COHORT_NAMES_NO_TARGETS.csv',
                    dtype={'code':str}, float_precision='round_trip')
    seal = pd.read_csv(ROOT/'research/first_board_joint_design_20261020T185417Z/qualification_resumed_20261021/COMPLETE28503_NEW_FIELDS_NO_LABELS.csv',
                       dtype={'code':str}, float_precision='round_trip')
    state_data = joblib.load(PRIOR/'CAUSAL_PRICE_STATES_WITHOUT_TARGETS.joblib')
    st,ss = state_data['states'],state_data['sessions']
    full = f.build(c,n,seal,st,ss)
    saved = pd.read_csv(DESIGN/'qualification_v2/COMPLETE28503_LIFECYCLE_STATUS_NO_TARGETS.csv',
                        dtype={'code':str}, float_precision='round_trip')
    assert full[f.KEYS].equals(saved[f.KEYS])
    np.testing.assert_allclose(full[f.ALL_FIELDS],saved[f.ALL_FIELDS],atol=1e-14,rtol=0,equal_nan=True)
    old_issues = pd.read_csv(DESIGN/'codex_review_20261021T0918Z/ACTUAL_STATE_AND_TIE_ISSUES.csv',dtype={'code':str})
    unknown = old_issues[old_issues.bad_anchor_p1.notna()].merge(full,on=f.KEYS)
    finite = unknown[unknown.name_anchor_decay_from_peak.notna()]
    finite.to_csv(OUT/'V2_UNKNOWN_ANCHOR_ROWS_STILL_NUMERIC.csv',index=False,encoding='utf-8-sig')
    result = dict(full_replay_rows=len(full), dates=int(full.signal_date.nunique()),
                  v2_full_replay_pass=True,
                  candidate_names_unknown=int((~c.name_ok).sum()),
                  actual_original_unknown_anchor_rows_still_numeric=len(finite),
                  mixed_known_unknown_anchor_fixture_has_numeric_lifecycle=bool(fx.name_anchor_decay_from_peak.notna().any()),
                  fixture=fx.replace({np.nan:None}).to_dict('records'),
                  status='REQUIRES_COMPLETE_ANCHOR_UNKNOWN_PROPAGATION_BEFORE_FIT',
                  labels_loaded=False,new_fits=0,http_requests=0,
                  source_sha256=hashlib.sha256(Path(f.__file__).read_bytes()).hexdigest())
    (OUT/'AUDIT_LIFECYCLE_V2.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='fixture'},ensure_ascii=True),flush=True)

if __name__=='__main__': main()
