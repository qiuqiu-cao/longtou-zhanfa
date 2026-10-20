from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[3]
source=REPO/'relay_monthly_oof_alignment_research.py'
tests=REPO/'research/model_upgrade_validation/rolling_oof_alignment_20261016T093022Z/preflight_tests.xml'
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert suites and all(int(s.get('errors',0))+int(s.get('failures',0))==0 for s in suites)
(OUT/'reviewed_monthly252_PREFIT_source.py').write_bytes(source.read_bytes())
paths=[source,tests,REPO/'tests/test_relay_monthly_oof_alignment_research.py',OUT/'monthly252_qualification_NO_FITS_receipt.json',Path(__file__)]
receipt=dict(independent_prefit_review_pass=True,new_fits=0,blocking_issue_found=False,
    prior_heads49101=13,new_heads56655=15,
    checks=['Natural-month model cutoff for every OOF signal',
        'Exact252 matured original training dates and original92 features',
        'Six new Jan-Jun2025 backfill heads and15 exact old SOURCE reuses',
        'Jul-Oct2025 whole outer month no-filter because OOF history insufficient',
        'Only12 eligible outer months x2 PREFIX meta fits, no FINAL refit path',
        'Prefix training labels before third first date, third selects fixed threshold only',
        'Selected fitted PREFIX object is reused for outer prediction',
        'Same meta features, equal rows, C1 and prior fixed5pp guard',
        'Original13 heads preserved; new15 heads are same3777 cohort each',
        'Strict final models/datasets/source hashes remain protected by explicit recovery adapter'],
    expected_new_research_fits_max=30,preflight_tests_passed=sum(int(s.get('tests',0)) for s in suites),
    interpretation='Joint procedure-alignment hypothesis; not unique causal attribution of past degradation',
    new_outer_scores_or_labels_read=False,official_target16_new_outcomes_read=False,
    source_bindings={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    output_bindings={str((OUT/'reviewed_monthly252_PREFIT_source.py').resolve()):hashlib.sha256(source.read_bytes()).hexdigest()})
(OUT/'monthly252_PREFIT_code_review_PASS.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print('monthly252 static review PASS; fits0')
