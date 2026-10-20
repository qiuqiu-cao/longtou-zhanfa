"""Read-only monthly252 training availability and exact source-model reuse."""
from pathlib import Path
import hashlib
import json
import sys

import joblib
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
REPO=ROOT.parents[2]
sys.path.insert(0,str(REPO))
import model_upgrade_history as history

SOURCE=REPO/'research/model_upgrade_validation/ranking_optimization_20261015T173045Z/source_feature_training'
INPUT=REPO/'research/model_upgrade_validation/positive_filter_20261016T091007Z/fixed_input_BEFORE_ANY_FIT.joblib'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run():
    data=joblib.load(INPUT);full,fields=data['full'],data['fields']
    assert len(full)==9315 and len(fields)==92 and len(data['plans'])==16
    monthly_need=set();outer_rows=[]
    for plan in data['plans']:
        window=history.training_rows(full,plan['cutoff'],'relay');days=sorted(window.date.unique())
        assert len(days)==252 and history._frame_digest(window,fields)==plan['training_digest']
        needed=days[63:];monthly_need.update(d[:7] for d in needed)
        unsupported=[d for d in needed if d<'2025-01-01']
        outer_rows.append(dict(month=plan['month'],outer_cutoff=plan['cutoff'],original252_rows=len(window),
            original252_digest=plan['training_digest'],first_OOF_signal=needed[0],last_OOF_signal=needed[-1],
            prefix_last_signal=days[188],third_first_signal=days[189],unsupported_pre2025_signal_days=len(unsupported),
            all189_OOF_signal_days_have252_history=not unsupported,
            proposed_whole_outer_month_bypass=bool(unsupported)))
    plans={p['month']:p for p in data['plans']}
    monthly_rows=[];bindings={str(p.resolve()):sha(p) for p in [INPUT,Path(__file__),Path(history.__file__)]}
    for month in sorted(monthly_need):
        cutoff=month+'-01'
        mask=full.label_valid.fillna(False).astype(bool)&full.target.notna()&full.date.lt(cutoff)&full.target_date.lt(cutoff)
        mature=full[mask].copy();days=sorted(mature.date.unique());qualified=len(days)>=252
        recent=mature[mature.date.isin(days[-252:])].copy()
        digest=history._frame_digest(recent,fields) if qualified else None
        path=None;reuse=False
        if qualified and month in plans:
            old=plans[month]
            assert old['cutoff']==cutoff and old['training_digest']==digest
            path=Path(old['final_model_path'])
            assert sha(path)==old['final_model_sha256']
            saved=joblib.load(path)
            assert saved['training_digest']==digest and saved['features']==fields and saved['cutoff']==cutoff
            reuse=True;bindings[str(path.resolve())]=sha(path)
        monthly_rows.append(dict(month=month,month_start_cutoff=cutoff,available_mature_signal_days=len(days),
            qualified_exact252=qualified,recent252_rows=len(recent) if qualified else None,
            recent252_first_signal=recent.date.min() if qualified else None,
            recent252_last_signal=recent.date.max() if qualified else None,
            recent252_max_target=recent.target_date.max() if qualified else None,
            expected_training_digest=digest,existing_exact_SOURCE_reuse=reuse,
            reused_model_path=str(path.resolve()) if path else None,reused_model_sha256=sha(path) if path else None,
            new_HGB_needed=qualified and not reuse,insufficient_history_must_bypass=not qualified))
    months=pd.DataFrame(monthly_rows);outer=pd.DataFrame(outer_rows)
    assert months.new_HGB_needed.sum()==6 and months.existing_exact_SOURCE_reuse.sum()==15
    assert outer.proposed_whole_outer_month_bypass.sum()==4
    months.to_csv(OUT/'all25_month_start252_qualification_and15_exact_reuses.csv',index=False,encoding='utf-8-sig')
    outer.to_csv(OUT/'all16_outer_OOF_complete_month_availability.csv',index=False,encoding='utf-8-sig')
    receipt=dict(new_fits=0,network_calls=0,old_files_modified=0,official_target16_new_results_read=False,
        required_OOF_calendar_months=25,qualified_OOF_calendar_months=21,insufficient_OOF_months=4,
        insufficient_month_counts={r['month']:r['available_mature_signal_days'] for r in monthly_rows if not r['qualified_exact252']},
        exact_reusable_first_stage_OOF_models=15,
        required_new_HGB_months=[r['month'] for r in monthly_rows if r['new_HGB_needed']],
        complete_three_block_outer_months=12,whole_outer_month_bypass_months=outer.loc[outer.proposed_whole_outer_month_bypass,'month'].tolist(),
        old_SOURCE_Oct2026_model_is_outer_only_not_OOF=True,all_qualified_train_targets_strictly_before_month_start=True,
        expected_maximum_new_HGB_fits=6,expected_maximum_new_PREFIX_LR_fits=24,expected_total_new_fits_at_most30=True,
        no_new_predictions_generated=True,
        source_bindings=bindings,
        output_bindings={str(p.resolve()):sha(p) for p in OUT.iterdir() if p.suffix=='.csv'})
    (OUT/'monthly252_qualification_NO_FITS_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['source_bindings','output_bindings']},ensure_ascii=False,indent=2))


if __name__=='__main__':run()
