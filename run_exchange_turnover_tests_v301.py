"""Actual focused tests of source units, scopes and causal amount bands."""
from datetime import datetime,timezone
import argparse
import subprocess
import sys
from pathlib import Path
from market_interval_research_v300 import ROOT,sha,write_json


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    started=datetime.now(timezone.utc).isoformat()
    command=[sys.executable,'-m','unittest','test_exchange_turnover_v301','test_exchange_turnover_intervals_v301',
             'test_market_interval_research_v300','-v']
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    sources=['exchange_turnover_verified_v301.py','exchange_turnover_intervals_v301.py',
             'test_exchange_turnover_v301.py','test_exchange_turnover_intervals_v301.py',
             'market_interval_research_v300.py','test_market_interval_research_v300.py',__file__]
    write_json(out,{'started_at_utc':started,'finished_at_utc':datetime.now(timezone.utc).isoformat(),
        'command':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
        'fixture_tests_are_financial_performance_evidence':False,
        'source_hashes':{str((ROOT/name).resolve()):sha(ROOT/name) for name in sources}})
    print(result.stderr);sys.exit(result.returncode)
