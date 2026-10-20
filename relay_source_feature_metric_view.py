"""Explicit event-metric applicability view; no fits or prediction mutations."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import short_term_thinking_dataset_builder as b

STRUCTURAL_COLUMNS=frozenset({'period','arm','scope','candidates','known','unknown_outcomes','positives','metric_applicability'})


def applicability_view(original):
    result=original.copy()
    pair=result.arm.str.startswith('PAIRWISE')
    result['structural_unknown_calls_without_absolute_head']=np.nan
    result.loc[pair,'structural_unknown_calls_without_absolute_head']=result.loc[pair,'abstentions']
    result['structural_status']='ABSOLUTE_EVENT_HEAD_METRICS'
    result.loc[pair,'structural_status']='NO_EVENT_HEAD_UNKNOWN_STRUCTURE_IS_NOT_CLASSIFICATION_FAILURE_OR_PROFIT'
    for column in original.columns:
        if column not in STRUCTURAL_COLUMNS and pair.any():
            result.loc[pair,column]=np.nan
    result.loc[pair,'metric_applicability']='NO_ABSOLUTE_EVENT_HEAD'
    return result


def run(metrics_path,output_dir):
    source=Path(metrics_path).resolve();output=Path(output_dir).resolve()
    if output.exists():raise FileExistsError('new immutable performance view directory required')
    original=pd.read_csv(source);view=applicability_view(original)
    pair=view.arm.str.startswith('PAIRWISE')
    pd.testing.assert_frame_equal(original[~pair].reset_index(drop=True),view.loc[~pair,original.columns].reset_index(drop=True),check_exact=True,check_dtype=False)
    absolute=[c for c in original if c not in STRUCTURAL_COLUMNS]
    if not view.loc[pair,absolute].isna().all().all():raise ValueError('pairwise absolute metric not empty')
    output.mkdir()
    path=output/'same_pool_classification_metrics_APPLICABILITY_CLEAN.csv'
    view.to_csv(path,index=False,encoding='utf-8-sig')
    b.write_new(output/'metric_applicability_receipt.json',dict(formed_at=b.now(),source_path=str(source),source_sha256=b.sha(source),
        implementation_path=str(Path(__file__).resolve()),implementation_sha256=b.sha(__file__),output_path=str(path),output_sha256=b.sha(path),
        total_rows=len(view),pairwise_rows=int(pair.sum()),absolute_metric_columns_empty=absolute,
        retained_structural_columns=sorted(STRUCTURAL_COLUMNS),all_absolute_heads_unchanged=True,new_fits=0,
        explanation='PAIRWISE has no absolute event head; structural UNKNOWN is neither a classification failure nor zero realized profit.'))
    return dict(rows=len(view),pairwise_rows=int(pair.sum()),new_fits=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--metrics',required=True);parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();print(json.dumps(run(args.metrics,args.output_dir),ensure_ascii=True))
