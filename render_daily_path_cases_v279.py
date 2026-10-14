"""Preserve every settled September paired case, including challenger losses."""
import argparse
import json
from pathlib import Path
import pandas as pd
from next_session_forecast_lab_v271 import sha,top1


def run(source,output):
    source=Path(source);target=Path(output);target.mkdir(parents=True,exist_ok=False)
    picks={};hashes={}
    for arm in ['CONTROL','DAILY_CLUES']:
        path=source/(arm+'_predictions_LABEL_ONLY.csv');hashes[arm]=sha(path)
        f=pd.read_csv(path,dtype={'code':str},low_memory=False)
        f=f[f.fold.eq('SEPTEMBER')].copy()
        complete=f.groupby('date').label_valid.all()
        f=f[f.date.isin(complete[complete].index)&f.target.notna()&f.daily_true_one_word.eq(0)]
        picked=top1(f)
        picks[arm]=picked[['date','code','name','boards','target','daily_open_gap','daily_first_open_after_one_word','first_board_turnover']]
    cases=picks['CONTROL'].merge(picks['DAILY_CLUES'],on='date',suffixes=('_control','_clues'),validate='one_to_one')
    cases['promotion_delta']=cases.target_clues-cases.target_control
    cases.to_csv(target/'all_settled_september_cases.csv',index=False,encoding='utf-8-sig')
    report={'status':'PAIRED_CASES_NOT_PROOF_OF_CAUSALITY','dates':len(cases),
            'challenger_only_wins':int(cases.promotion_delta.eq(1).sum()),
            'control_only_wins':int(cases.promotion_delta.eq(-1).sum()),
            'both_fail':int((cases.target_clues.eq(0)&cases.target_control.eq(0)).sum()),
            'source_hashes':hashes,'actual_fills':0,'trade_authorized':False}
    (target/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run(a.source,a.output),ensure_ascii=True))
