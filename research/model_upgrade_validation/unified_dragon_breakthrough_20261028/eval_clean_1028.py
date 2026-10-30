"""Causal inference entry; explicit history/date/output required. No invented case inputs."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
REPAIR=ROOT/'research/model_upgrade_validation/codex_5d_causal_repair_20261028'

if __name__=='__main__':
    subprocess.run([sys.executable,str(REPAIR/'score_open_snapshot.py'),*sys.argv[1:]],cwd=ROOT,check=True)
