from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

os.environ.setdefault('OMP_NUM_THREADS','4')
import joblib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    test=subprocess.run([sys.executable,str(HERE/'test_causal_repair.py')],capture_output=True,text=True,encoding='utf-8',errors='replace')
    (HERE/'test_output.txt').write_text(test.stdout+test.stderr,encoding='utf-8')
    if test.returncode:
        raise RuntimeError(test.stdout+test.stderr)
    summary=json.loads((HERE/'backtest_summary.json').read_text(encoding='utf-8'))
    oof=pd.read_csv(HERE/'oof_all_candidates.csv',dtype={'code':str},float_precision='round_trip')
    errors={}
    for name,rows in oof.groupby('fold'):
        bundle=joblib.load(HERE/f'model_{name.replace(" ","_")}.joblib')
        prediction=bundle['model'].predict_proba(rows[bundle['features']])[:,1]
        errors[name]=float(np.max(np.abs(prediction-rows.score)))
        assert errors[name]<1e-12
        assert bundle['fold']['training_max_weight_label_date']<bundle['fold']['test_start']
    trades=pd.read_csv(HERE/'portfolio_conditional_trades.csv')
    curve=pd.read_csv(HERE/'portfolio_conditional_daily.csv')
    assert (trades.exit_date>trades.entry_date).all()
    assert (curve.cash>=-1e-7).all()
    port=summary['conditional_portfolio']
    open_pnl=sum(p['shares']*p['mark']-p['entry_cost'] for p in port['unclosed_positions'])
    pnl_error=abs(port['end_equity']-port['initial_cash']-trades.pnl.sum()-open_pnl)
    assert pnl_error<1e-5
    assert all(summary['source_and_core_unchanged'].values())
    result=dict(tests=8,tests_passed=True,reloaded_fold_prediction_max_abs_error=errors,
        folds_purged=True,no_duplicate_date_code=not oof.duplicated(['date','code']).any(),
        no_negative_cash=True,tplusone_all_closed_trades=True,pnl_reconciliation_abs_error=pnl_error,
        protected_base_unchanged=True,fits=4,production_replacement=False,
        files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.is_file() and p.name not in ['verification.json']})
    (HERE/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
