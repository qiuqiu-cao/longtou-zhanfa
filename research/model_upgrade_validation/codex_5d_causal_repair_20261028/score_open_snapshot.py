"""Score normalized daily history and current-open snapshot with the trained schema."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS','4')
import joblib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from legacy_industry_snapshot import load_snapshot
from causal_engine import engineer, snapshot_inputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bars',required=True,help='Normalized daily CSV; requires historical rows plus target-date open/reference rows.')
    parser.add_argument('--date',required=True)
    parser.add_argument('--model',default=str(HERE/'model_Fold_4.joblib'))
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    raw=pd.read_csv(args.bars,dtype={'code':str},low_memory=False)
    if not raw.date.eq(args.date).any():
        raise ValueError('Requested date absent. No stale-day replacement or fabricated current input is allowed.')
    cal=json.loads((ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json').read_text(encoding='utf-8'))
    if args.date not in cal['sessions']:
        raise ValueError('Requested date outside the bound official calendar.')
    bundle=joblib.load(args.model)
    if bundle['fold']['training_max_weight_label_date']>=args.date:
        raise ValueError('Model training outcomes are not mature strictly before requested scoring date.')
    mapping,_=load_snapshot()
    imap=mapping.set_index('code').legacy_group_PROXY_ALLOWED.to_dict()
    d=engineer(snapshot_inputs(raw,None,args.date),cal['sessions'],imap)
    today=d[d.date.eq(args.date)].copy()
    today['uncalibrated_score']=np.nan
    good=today.eligible
    if good.any():
        today.loc[good,'uncalibrated_score']=bundle['model'].predict_proba(today.loc[good,bundle['features']])[:,1]
    today['status']=np.where(good,'RESEARCH_SCORE_REQUIRES_SCREENSHOT_AUCTION_AUDIT','OUTSIDE_OBSERVABLE_TRAINING_POOL')
    today['tail_order_verified']=False
    today['auction_queue_verified']=False
    target=Path(args.output)
    if target.exists():
        raise FileExistsError('Preserve the previous inference output.')
    out=today[['date','code','name','uncalibrated_score','status','tail_order_verified','auction_queue_verified']].sort_values(['uncalibrated_score','code'],ascending=[False,True],na_position='last')
    out.to_csv(target,index=False,encoding='utf-8-sig')
    print(json.dumps(dict(output=str(target.resolve()),rows=len(out),scored=int(good.sum()),orders=0,calibrated_probability=False),ensure_ascii=False))


if __name__=='__main__':
    main()
