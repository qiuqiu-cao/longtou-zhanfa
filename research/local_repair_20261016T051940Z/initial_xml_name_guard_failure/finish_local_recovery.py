"""Record exact recovery and explicit replacement of lost historical test logs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import relay_local_recovery as recovery


def run():
    manifest=OUT/'recovery_manifest.json'
    if manifest.exists():raise FileExistsError('recovery manifest already frozen')
    original=ROOT/'research/model_upgrade_validation/iteration_research_20261016T043944Z/delivery/combined_parameter_delivery_verified_NO_UPGRADE.json'
    value=recovery.read(original);sources=value['source_bindings']
    fresh=OUT/'fresh_recovery_checks.xml'
    suites=list(ET.parse(fresh).getroot().iter('testsuite'))
    count=sum(int(s.get('tests',0)) for s in suites)
    if count!=127 or any(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites):raise ValueError('all127 current checks must pass')
    # The file is never copied to replace an original timestamped test result.
    if re.search(rb'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])',fresh.read_bytes()):
        raise ValueError('fresh test XML contains an address-like pattern; do not export raw')
    missing=[];exact=0
    for name,digest in sources.items():
        path=Path(name)
        if path.exists():
            if recovery.sha(path)!=digest:raise ValueError('existing protected source changed: '+str(path))
            exact+=1
        else:
            if path.suffix.lower()!='.xml' or 'test' not in path.name.lower():raise ValueError('missing non-test protected source: '+str(path))
            missing.append(dict(original_path=str(path),original_expected_sha256=digest,
                status='HISTORICAL_ORIGINAL_BYTES_UNAVAILABLE_AFTER_USER_REQUESTED_IP_CLEANUP',
                replacement='NEW_TEST_RUN_SEPARATE_PATH_NOT_ORIGINAL_BYTES'))
    codepaths=[ROOT/'relay_local_recovery.py',ROOT/'tests/test_relay_local_recovery.py',Path(__file__),OUT/'restore_safe_tracked_tests.py']
    runtimepaths=[ROOT/'relay_pairwise_workbench.py',ROOT/'expectation_345_workbench.py',
        ROOT/'relay_review_workbench.py',ROOT/'short_term_thinking_dataset_builder.py']
    data=dict(formed_at=datetime.now(timezone.utc).isoformat(),type='EXPLICIT_DELETED_PYTEST_RECORD_RECOVERY',
        original_receipt_path=str(original),original_receipt_sha256=recovery.sha(original),
        fresh_tests_path=str(fresh),fresh_tests_sha256=recovery.sha(fresh),fresh_tests_passed=count,
        exact_original_source_files_verified=exact,historical_test_records_unrecoverable=missing,
        recovery_code_bindings={str(p.resolve()):recovery.sha(p) for p in codepaths},
        protected_runtime_bindings={str(p.resolve()):recovery.sha(p) for p in runtimepaths},
        old_receipts_or_hashes_rewritten=False,missing_test_bytes_fabricated=False,
        model_data_code_hash_exceptions=0,network_requests=0,raw_client_address_output=False,
        original_HTML_access_denied_response_left_deleted=True,
        local_Git_history_not_modified=True,original_legacy_verifier_unchanged=True)
    manifest.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    result=recovery.verify_sources(sources,manifest)
    (OUT/'latest_source_chain_recovery_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(exact_original_files=exact,unrecoverable_historical_test_logs=len(missing),fresh_tests=count,
        model_data_code_hash_exceptions=0,IP_values_output=False)))


if __name__=='__main__':run()
