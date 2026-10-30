"""Codex-reviewed 5D research entry. Prior Gemini source is preserved in the repair snapshot.
Uses the fixed causal engine and already-completed four-fold backtest. No model deployment.
"""
from pathlib import Path
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
REPAIR=ROOT/'research/model_upgrade_validation/codex_5d_causal_repair_20261028'

if __name__=='__main__':
    summary=REPAIR/'backtest_summary.json'
    if summary.exists():
        sys.stdout.reconfigure(encoding='utf-8')
        print(summary.read_text(encoding='utf-8'))
    else:
        subprocess.run([sys.executable,str(REPAIR/'run_backtest.py')],cwd=ROOT,check=True)
