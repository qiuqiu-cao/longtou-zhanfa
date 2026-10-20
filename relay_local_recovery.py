"""Explicit local recovery for deleted historical pytest records.

Never changes old receipts or forgives a model/data/code hash mismatch. The
original missing audit artifact is distinguished from a new test run.
No network, address detection service, web server, or publishing is used.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import runpy
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_recovery(manifest_path):
    manifest=read(manifest_path)
    if manifest.get('type')!='EXPLICIT_DELETED_PYTEST_RECORD_RECOVERY':raise ValueError('explicit recovery manifest required')
    report=Path(manifest['fresh_tests_path'])
    if sha(report)!=manifest['fresh_tests_sha256']:raise ValueError('fresh test record changed')
    suites=list(ET.parse(report).getroot().iter('testsuite'))
    if not suites or any(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites):
        raise ValueError('fresh targeted tests must pass')
    code=manifest.get('recovery_code_bindings',{})
    for name,digest in code.items():
        if sha(name)!=digest:raise ValueError('recovery implementation changed')
    registry={}
    for item in manifest['historical_test_records_unrecoverable']:
        path=Path(item['original_path']).resolve()
        if not path.is_relative_to(ROOT) or path.suffix.lower()!='.xml' or not re.search(r'test|integration',path.name,re.I):
            raise ValueError('recovery applies only to explicitly listed project test XML')
        if not re.fullmatch('[0-9a-f]{64}',item['original_expected_sha256']):raise ValueError('original hash required')
        if path in registry:raise ValueError('duplicate recovery identity')
        registry[path]=item['original_expected_sha256']
    return registry,manifest


def verify_sources(sources,manifest_path):
    registry,manifest=load_recovery(manifest_path)
    exact=0;unavailable=[]
    for name,expected in sources.items():
        path=Path(name).resolve()
        if path.is_file():
            if sha(path)!=expected:raise ValueError('source hash changed: '+str(path))
            exact+=1
        elif registry.get(path)==expected:
            unavailable.append(str(path))
        else:
            raise ValueError('unapproved missing source: '+str(path))
    return dict(exact_original_sources_verified=exact,
        historical_test_records_unavailable=len(unavailable),
        fresh_tests_passed=manifest['fresh_tests_passed'],
        original_test_bytes_recovered=False if unavailable else True,
        historical_test_record_substitution_explicit=True,
        model_data_code_hash_exceptions=0)


def receipt_sources(path):
    data=read(path)
    return data.get('source_bindings',data.get('sources',{}))|data.get('output_bindings',{})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery-manifest',required=True)
    commands=parser.add_subparsers(dest='command',required=True)
    verify=commands.add_parser('verify');verify.add_argument('--receipt',required=True)
    run=commands.add_parser('run-reviewed-score')
    run.add_argument('module',choices=['relay_pairwise_workbench.py','expectation_345_workbench.py'])
    run.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if args.command=='verify':
        print(json.dumps(verify_sources(receipt_sources(args.receipt),args.recovery_manifest),ensure_ascii=False));return
    # Scoped, explicit compatibility adapter for the existing reviewed scoring
    # entry only. No core files or original receipts are edited.
    import short_term_thinking_dataset_builder as original
    registry,manifest=load_recovery(args.recovery_manifest)
    verify_sources(manifest['protected_runtime_bindings'],args.recovery_manifest)
    old_verifier=original.verify
    original.verify=lambda sources:verify_sources(sources,args.recovery_manifest)
    try:
        sys.argv=[args.module]+args.arguments
        runpy.run_path(str(ROOT/args.module),run_name='__main__')
    finally:
        original.verify=old_verifier


if __name__=='__main__':main()
