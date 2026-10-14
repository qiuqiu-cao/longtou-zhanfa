"""Persist an actual focused unittest run; fixtures are not market performance evidence."""
from datetime import datetime, timezone
import subprocess
from market_interval_research_v300 import ROOT, sha, write_json


if __name__=='__main__':
    import sys
    output=ROOT/'results/v300/test_run_20261003_v1.json'
    if output.exists(): raise FileExistsError(output)
    started=datetime.now(timezone.utc).isoformat()
    command=[sys.executable,'-m','unittest','test_market_interval_research_v300','-v']
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    write_json(output,{'started_at_utc':started,'finished_at_utc':datetime.now(timezone.utc).isoformat(),
        'command':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
        'fixture_tests_are_market_evidence':False,
        'source_hashes':{str(p.resolve()):sha(p) for p in [ROOT/'market_interval_research_v300.py',
            ROOT/'test_market_interval_research_v300.py',ROOT/'run_market_interval_tests_v300.py']}})
    print(result.stderr);sys.exit(result.returncode)
