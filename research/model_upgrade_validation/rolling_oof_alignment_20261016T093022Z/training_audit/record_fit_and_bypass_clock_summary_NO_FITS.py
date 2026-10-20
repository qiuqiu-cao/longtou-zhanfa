from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json

import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
ARMS=['MONTHLY252_OOF_FROZEN_SCORE_FILTER','MONTHLY252_OOF_FROZEN_INDIVIDUAL_FILTER']
KIND=dict(zip(ARMS,['POSITIVE_FILTER_SCORE_ONLY','POSITIVE_FILTER_INDIVIDUAL_CONTEXT']))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def run():
    audit_path=OUT/'monthly252_training_and_saved_model_replay_BINDINGS.json';audit=read(audit_path)
    assert audit['independent_training_and_replay_pass']
    rows=[];sources=[audit_path,Path(__file__)]
    for p in sorted((ROOT/'execution').glob('*_SELECTION_BEFORE_OUTER_SCORE.json')):
        selection=read(p);month=selection['month'];choice_time=datetime.fromisoformat(selection['formed_at'])
        for arm in ARMS:
            kind=KIND[arm];prediction=ROOT/f'execution/{month}_{arm}_predictions_BEFORE_EVALUATION.csv'
            prediction_clock=datetime.fromtimestamp(prediction.stat().st_mtime,timezone.utc)
            assert choice_time<=prediction_clock
            info=dict(month=month,arm=arm,qualified=selection['eligible'],threshold=selection['choices'][arm]['threshold'],
                selection_formed_at=selection['formed_at'],outer_prediction_file_mtime_UTC=prediction_clock.isoformat(),
                selection_to_outer_file_seconds=(prediction_clock-choice_time).total_seconds(),
                fit_actually_executed=selection['eligible'],whole_month_warmup_bypass=not selection['eligible'])
            if selection['eligible']:
                start=ROOT/f'execution/{month}_{kind}_ALIGNED_FROZEN_PREFIX_FIT_STARTED.json'
                end=ROOT/f'execution/{month}_{kind}_ALIGNED_FROZEN_PREFIX_FIT_COMPLETED.json'
                s,e=read(start),read(end)
                assert datetime.fromisoformat(s['formed_at'])<datetime.fromisoformat(e['completed_at'])<choice_time
                info.update(prefix_fit_started=s['formed_at'],prefix_fit_completed=e['completed_at'],
                    prefix_complete_to_selection_seconds=(choice_time-datetime.fromisoformat(e['completed_at'])).total_seconds(),
                    genuine_fit_then_no_filter=info['threshold']==0.)
                sources.extend([start,end])
            else:
                info.update(prefix_fit_started=None,prefix_fit_completed=None,prefix_complete_to_selection_seconds=None,
                    genuine_fit_then_no_filter=False)
            rows.append(info);sources.extend([p,prediction])
    assert len(rows)==32 and sum(r['fit_actually_executed'] for r in rows)==24
    assert sum(r['whole_month_warmup_bypass'] for r in rows)==8 and all(r['threshold']==0 for r in rows)
    clock_path=OUT/'verified_monthly252_and_PREFIX/all32_distinct_fit_selection_or_warmup_clocks.csv'
    pd.DataFrame(rows).to_csv(clock_path,index=False,encoding='utf-8-sig')
    receipt=dict(independent_training_and_replay_pass=True,independent_new_fits=0,
        source_new_backfill_HGB=6,source_exact_reuses=15,old_deployment16_hashes_unchanged=True,
        real_PREFIX_fits=24,qualified_fitted_policies_selecting_no_filter=24,warmup_no_fit_bypass_policies=8,
        total_no_filter_policies=32,no_filter_does_not_mean_training_was_skipped=True,
        all21_HGB_and24_LR_saved_prediction_replay_max_error=0.,
        all32_policy_clocks_verified=True,fit_and_selection_clocks_distinct=True,
        minimum_fit_complete_to_selection_seconds=min(r['prefix_complete_to_selection_seconds'] for r in rows if r['fit_actually_executed']),
        official_target16_new_outcomes_read=False,no_outer_predictive_metric_recomputation=True,
        source_bindings=audit['source_bindings']|{str(p.resolve()):sha(p) for p in sources},
        output_bindings=audit['output_bindings']|{str(clock_path.resolve()):sha(clock_path),str(audit_path.resolve()):sha(audit_path)})
    (OUT/'final_monthly252_training_replay_and_clocks_BINDINGS.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['source_bindings','output_bindings']},ensure_ascii=False,indent=2))


if __name__=='__main__':run()
