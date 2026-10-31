"""THS-observed subset features; missing peers/anchors are unknown, not zero."""
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('old_dynamic_operators',ROOT/'research/first_board_dynamic_rank_20261030/dynamic_features.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
FIELDS=['ths_observed_subset_'+f for f in old.FIELDS]


def build(candidates,provider,states,sessions):
    raw=old.build(candidates,provider,states,sessions)
    rows=[]
    for record in raw.to_dict('records'):
        detail=json.loads(record['theme_details_json'])
        for d in detail:
            if not d['current_peer_codes']:
                d['theme_peer_breadth']=None
                d['relative_reported_first_seal']=None
            if not d['prior_anchor_codes']:
                for f in old.ANCHOR:d[f]=None
            d['scope']='OBSERVED_SUBSET_OF_FIXED_FULL_PERIOD_CANDIDATE_CODES_NOT_FULL_MARKET'
        row={k:v for k,v in record.items() if k not in old.FIELDS and not k.endswith('_valid_themes')}
        row['theme_details_json']=json.dumps(detail,ensure_ascii=False,allow_nan=False)
        for original,prefixed in zip(old.FIELDS,FIELDS):
            vals=[d[original] for d in detail if d[original] is not None and np.isfinite(d[original])]
            row[prefixed]=float(np.mean(vals)) if vals else np.nan
            row[prefixed+'_valid_themes']=len(vals)
        row['full_market_theme_membership_complete']=False
        row['source_version_at_signal_certified']=False
        rows.append(row)
    import pandas as pd
    return pd.DataFrame(rows)
