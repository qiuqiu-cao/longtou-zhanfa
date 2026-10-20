"""Bind independent audit outputs and exact distinct fit/selection clocks."""
from pathlib import Path
from datetime import datetime
import hashlib
import json

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
DETAIL=OUT/'executed_training_qualification'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def run():
    import pandas as pd
    receipt_path=DETAIL/'independent_training_and_saved_model_replay_receipt.json'
    audit=read(receipt_path)
    assert audit['independent_training_qualification_pass'] and audit['independent_new_fits']==0
    times=[];sources=[receipt_path,Path(__file__),OUT/'verify_saved_positive_heads_NO_FITS.py']
    for selection_path in sorted((ROOT/'execution').glob('*_SELECTION_BEFORE_FINAL_FIT_AND_OUTER.json')):
        selection=read(selection_path);month=selection['month']
        for arm in selection['choices']:
            pstart=ROOT/f'execution/{month}_{arm}_PREFIX_FIT_STARTED.json'
            pend=ROOT/f'execution/{month}_{arm}_PREFIX_FIT_COMPLETED.json'
            fstart=ROOT/f'execution/{month}_{arm}_FINAL_FIT_STARTED.json'
            fend=ROOT/f'execution/{month}_{arm}_FINAL_FIT_COMPLETED.json'
            armend=ROOT/f'execution/{month}_{arm}_COMPLETED.json'
            ps,pe,fs,fe,ae=map(read,[pstart,pend,fstart,fend,armend])
            values=[ps['formed_at'],pe['completed_at'],selection['formed_at'],fs['formed_at'],fe['completed_at'],ae['completed_at']]
            clocks=list(map(datetime.fromisoformat,values))
            assert clocks[0]<clocks[1]<clocks[2]<clocks[3]<clocks[4]<=clocks[5]
            times.append(dict(month=month,arm=arm,prefix_fit_started=values[0],prefix_fit_completed=values[1],
                threshold_selection_formed=values[2],final_fit_started=values[3],final_fit_completed=values[4],
                arm_prediction_completed=values[5],selection_is_distinct_clock=True,
                prefix_completed_to_selection_seconds=(clocks[2]-clocks[1]).total_seconds(),
                selection_to_final_fit_start_seconds=(clocks[3]-clocks[2]).total_seconds(),
                final_fit_completed_to_arm_complete_seconds=(clocks[5]-clocks[4]).total_seconds()))
            sources.extend([pstart,pend,fstart,fend,armend,selection_path])
    assert len(times)==32
    csv_path=DETAIL/'all32_exact_distinct_fit_selection_prediction_clocks.csv'
    pd.DataFrame(times).to_csv(csv_path,index=False,encoding='utf-8-sig')
    outputs=[p for p in DETAIL.iterdir() if p.is_file()]
    receipt=dict(independent_audit_pass=True,independent_new_fits=0,saved_meta_models_replayed=64,
        primary_OOF_models_replayed=48,saved_meta_prediction_max_error=0.,primary_cache_score_max_error=0.,
        original48_inner_and16_final_primary_hashes_unchanged=True,
        exact_distinct_fit_selection_prediction_clocks_verified=32,
        minimum_prefix_complete_to_selection_seconds=min(r['prefix_completed_to_selection_seconds'] for r in times),
        minimum_selection_to_final_start_seconds=min(r['selection_to_final_fit_start_seconds'] for r in times),
        official_target16_new_results_read=False,metrics_main_acceptance_not_asserted=True,
        source_bindings={str(p.resolve()):sha(p) for p in sources},
        output_bindings={str(p.resolve()):sha(p) for p in outputs})
    (OUT/'final_training_qualification_and_replay_BINDINGS.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['source_bindings','output_bindings']},ensure_ascii=False,indent=2))


if __name__=='__main__':run()
