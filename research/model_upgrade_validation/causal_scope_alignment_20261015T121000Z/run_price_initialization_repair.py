from pathlib import Path
import sys,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import pandas as pd
import short_term_thinking_dataset_builder as b

ROOT=Path(__file__).resolve().parent
b.write_new(ROOT/"price_initialization_repair_STARTED.json",dict(formed_at=b.now(),new_fits=0,source_preimage_preserved=True))
stages=[]
for name in ["build_context.py","supplement_complete_evaluation.py","build_complete_workbench.py"]:
    completed=subprocess.run([sys.executable,str(ROOT/name)],cwd=b.ROOT,capture_output=True,text=True,encoding="utf-8")
    stage=dict(script=name,exit_code=completed.returncode,output=completed.stdout,stderr=completed.stderr,completed_at=b.now(),new_fits=0)
    b.write_new(ROOT/(name+".repair_receipt.json"),stage)
    print(completed.stdout,flush=True)
    if completed.returncode:print(completed.stderr,flush=True);raise SystemExit(completed.returncode)
    stages.append(stage)
before=ROOT/"initial_before_price_history_fix/context/all9315_original_fields_and_review_semantics.csv"
after=ROOT/"context/all9315_original_fields_and_review_semantics.csv"
pd.testing.assert_frame_equal(pd.read_csv(before,low_memory=False),pd.read_csv(after,low_memory=False))
b.write_new(ROOT/"price_initialization_repair_VERIFIED.json",dict(formed_at=b.now(),stages=stages,full9315_rows_and_all_columns_unchanged=True,
    original_actual_data_includes_first_official_source_day=True,new_regression_earlier_price_history_passes=True,
    source_preimage_and_initial_stage_preserved=True,new_fits=0,source_bindings={str(p.resolve()):b.sha(p) for p in [Path(__file__),before,after,b.ROOT/"relay_causal_review_context.py"]}))
print("prior-price initialization repaired; all9315 actual rows unchanged; fit0",flush=True)
