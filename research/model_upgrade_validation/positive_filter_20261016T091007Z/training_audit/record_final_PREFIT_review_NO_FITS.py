from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[3]
source=REPO/'relay_positive_filter_research.py'
test=OUT.parent/'preflight_tests.xml'
suites=list(ET.parse(test).getroot().iter('testsuite'))
assert suites and all(int(s.get('errors',0))+int(s.get('failures',0))==0 for s in suites)
assert "s['known']-s['positives']" in source.read_text(encoding='utf-8')
(OUT/'reviewed_final_PREFIT_source.py').write_bytes(source.read_bytes())
paths=[source,test,REPO/'tests/test_relay_positive_filter_research.py',REPO/'relay_local_recovery.py',Path(__file__)]
receipt=dict(review_only=True,new_fits=0,blocking_issue_found=False,
    initial_bootstrap_KeyError_fixed_before_fit=True,bootstrap_smoke_test_passed=True,
    passing_preflight_count=sum(int(s.get('tests',0)) for s in suites),
    checks=['48 cached first-stage HGB inference only, complete validation dates and exact original score replay',
        'Prefix first2 blocks labels strictly before third first signal',
        'Final all3 labels strictly before external cutoff',
        'Original frozen B plus verified0 and primary PROMOTE only',
        'Complete original B validation denominator includes unknown',
        'Equal meta-row weights and train-only imputer/scaler',
        '64 research fit ceiling, original first-stage fit0',
        'ConvergenceWarning and max_iter rejection without retry',
        'Raw ranking retained and recommendation fixed3 slots separately assessed',
        'Explicit missing XML recovery only, no source code/data/model hash waiver'],
    known_risks_retained=['Final refit changes score scale so selected prefix threshold may not transfer',
        'Inner first-stage training lengths62/125/188 differ from outer252',
        'Previously seen history, late sources and survivor universe remain'],
    source_bindings={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
(OUT/'final_PREFIT_code_review_PASS.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print('final PREFIT pass saved; fits0')
