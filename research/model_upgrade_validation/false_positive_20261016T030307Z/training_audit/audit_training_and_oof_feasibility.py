"""Read-only training/OOF feasibility audit. Never fits an estimator."""
from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
import expectation_345_research as original
import model_upgrade_history as history

OUT = Path(__file__).resolve().parent
OLD = REPO / 'research/model_upgrade_validation/ranking_optimization_20261015T173045Z'
SOURCE = OLD / 'source_feature_training/prepared.joblib'
IDENTITY = REPO / 'research/model_upgrade_validation/short_term_thinking_20261015T071000Z/dataset.joblib'
TABLE = OLD / 'source_feature_training/same_source_route_alignment/complete26439_predictions_WITH_SAME_SOURCE_ROUTES.csv'
METRICS = OLD / 'source_feature_training/same_source_route_alignment/same_pool_classification_metrics.csv'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    data = joblib.load(SOURCE)
    full = data['full'].copy()
    historical_identity = joblib.load(IDENTITY)['features'][['date', 'code', 'daily_one_word']].set_index(['date', 'code'])
    # The old full features use the corrected latest engine identity. The audit
    # fixes original B membership explicitly, independently of source recovery.
    membership = full[['date', 'code', 'daily_one_word']].copy()
    for i, row in membership.iterrows():
        if (row.date, row.code) in historical_identity.index:
            membership.loc[i, 'daily_one_word'] = historical_identity.loc[(row.date, row.code), 'daily_one_word']
    outer_refs = pd.read_csv(TABLE, dtype={'code': str}, low_memory=False)
    outer_refs = outer_refs[outer_refs.arm.eq('BASE')][['date', 'code', 'daily_one_word']].set_index(['date', 'code'])
    for i, row in membership.iterrows():
        if (row.date, row.code) in outer_refs.index:
            membership.loc[i, 'daily_one_word'] = outer_refs.loc[(row.date, row.code), 'daily_one_word']
    full['audit_original_frozen_B'] = pd.MultiIndex.from_frame(full[['date','code']]).map(membership.set_index(['date','code']).daily_one_word.eq(0))
    windows, blocks, paths = [], [], []
    for check in data['training_checks']:
        train = history.training_rows(full, check['cutoff'], 'relay')
        days = sorted(train.date.unique())
        assert len(days) == 252 and train.target_date.max() < check['cutoff']
        w1, w15 = original.weights(train, 1), original.weights(train, 1.5)
        day_weight = 1 / train.groupby('date').date.transform('size')
        mass = day_weight.groupby(train.target).sum()
        windows.append(dict(month=check['month'], cutoff=check['cutoff'], rows=len(train), days=len(days),
            natural_row_positive_rate=float(train.target.mean()), natural_equal_day_positive_rate=float((day_weight * train.target).sum() / day_weight.sum()),
            cost1_negative_mass=float(w1[train.target.eq(0)].sum()), cost1_positive_mass=float(w1[train.target.eq(1)].sum()),
            cost1p5_negative_mass=float(w15[train.target.eq(0)].sum()), cost1p5_positive_mass=float(w15[train.target.eq(1)].sum()),
            inverse_day_positive_mass=float(mass[1]), inverse_day_negative_mass=float(mass[0]),
            cost1_positive_to_negative_per_base_weight_multiplier=float(mass[0]/mass[1]),
            raw_score_calibrated=False))
        for block_number in range(3):
            validation_days = days[63 * (block_number + 1):63 * (block_number + 2)]
            first = validation_days[0]
            inner = train[train.date.lt(first) & train.target_date.lt(first)].copy()
            validation = train[train.date.isin(validation_days)]
            calibratable = validation[validation.audit_original_frozen_B]
            removed = train[train.date.lt(first) & ~train.target_date.lt(first)]
            assert inner.target_date.max() < first and inner.date.max() < first
            assert inner.target.nunique() == 2 and calibratable.target.nunique() == 2
            blocks.append(dict(month=check['month'], block=block_number + 1, validation_first=first,
                validation_last=validation_days[-1], validation_signal_days=len(validation_days),
                pre_validation_training_rows=len(inner), pre_validation_training_days=inner.date.nunique(),
                pre_validation_max_target_date=inner.target_date.max(), training_classes=inner.target.nunique(),
                strict_maturity_removed_rows=len(removed), strict_maturity_removed_days=removed.date.nunique(),
                full_validation_rows=len(validation), original_B_known_OOF_rows=len(calibratable),
                original_B_known_OOF_days=calibratable.date.nunique(), original_B_positive=int(calibratable.target.eq(1).sum()),
                original_B_negative=int(calibratable.target.eq(0).sum()), original_B_classes=calibratable.target.nunique()))
        receipt_path = OLD / f"source_feature_training/execution/{check['month']}_COMPLETED.json"
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        model_path = Path(receipt['model_path'])
        assert sha(model_path) == receipt['model_sha256']
        paths.append(dict(month=check['month'], model_path=str(model_path), sha256=sha(model_path),
            reused_original_model=check['reuse_original_model'], cutoff=check['cutoff'],
            training_digest=receipt['training_digest'], features=92, final_head_refit_needed=False))
    windows = pd.DataFrame(windows)
    blocks = pd.DataFrame(blocks)
    windows.to_csv(OUT / 'all16_training_weights_READ_ONLY.csv', index=False, encoding='utf-8-sig')
    blocks.to_csv(OUT / 'all48_OOF_block_feasibility_NO_FITS.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(paths).to_csv(OUT / 'all16_final_source_head_reuse_paths.csv', index=False, encoding='utf-8-sig')
    selected_arms = ['BASE', 'ST_DUAL', 'CORRECTED_CAUSAL', 'CORRECTED_CAUSAL_COST1P5', 'SOURCE_PATCHED_FEATURE_CLASSIFIER']
    stats = pd.read_csv(METRICS)
    stats = stats[stats.arm.isin(selected_arms)].copy()
    stats.to_csv(OUT / 'complete_same_qualification_matrices_NO_RETRAIN.csv', index=False, encoding='utf-8-sig')
    pooled = blocks.groupby('month').agg(OOF_original_B_rows=('original_B_known_OOF_rows','sum'),
        OOF_original_B_days=('original_B_known_OOF_days','sum'), positives=('original_B_positive','sum'),negatives=('original_B_negative','sum'))
    pooled.to_csv(OUT/'all16_combined_OOF_calibration_sample_sizes_NO_FITS.csv',encoding='utf-8-sig')
    summary = dict(fits=0, old_files_modified=0, future_target_results_read=False,
        outer_windows=16, blocks=48, fields=92, outer_signal_days=252, inner_validation_signal_days=63,
        minimum_inner_train_signal_days=int(blocks.pre_validation_training_days.min()),
        maximum_inner_train_signal_days=int(blocks.pre_validation_training_days.max()),
        minimum_inner_train_rows=int(blocks.pre_validation_training_rows.min()),
        minimum_original_B_OOF_rows_per_block=int(blocks.original_B_known_OOF_rows.min()),
        minimum_original_B_positive_per_block=int(blocks.original_B_positive.min()),
        minimum_original_B_negative_per_block=int(blocks.original_B_negative.min()),
        minimum_combined_OOF_original_B_rows=int(pooled.OOF_original_B_rows.min()),
        minimum_combined_OOF_positive=int(pooled.positives.min()), minimum_combined_OOF_negative=int(pooled.negatives.min()),
        source_bindings={str(p.resolve()):sha(p) for p in [SOURCE,IDENTITY,TABLE,METRICS,Path(original.__file__),Path(history.__file__),Path(__file__)]})
    (OUT/'read_only_feasibility_receipt.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    run()
