"""One frozen bugfix run. No parameter search, formula export, or deployment."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

# Keep resource usage bounded and reproducible on the shared desktop.
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from legacy_industry_snapshot import load_snapshot
from causal_engine import FEATURES, engineer

FOLDS = [
    ('Fold 1', '2024-01-01', '2024-08-31', '2024-09-01', '2025-01-31'),
    ('Fold 2', '2024-06-01', '2025-01-31', '2025-02-01', '2025-07-31'),
    ('Fold 3', '2025-01-01', '2025-07-31', '2025-08-01', '2026-01-31'),
    ('Fold 4', '2025-06-01', '2026-01-31', '2026-02-01', '2026-09-16'),
]
PARAMS = dict(max_iter=150, learning_rate=.06, max_leaf_nodes=31,
              min_samples_leaf=25, l2_regularization=1.5, random_state=42,
              early_stopping=False)  # Fixed iteration budget; no random future validation.


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def stats(s):
    n = len(s)
    known = s.target_limit_up.notna()
    next_known = s.next_return_proxy.notna()
    return dict(selected_rows=n, selected_dates=int(s.date.nunique()),
        label_known=int(known.sum()), label_unknown=int((~known).sum()),
        close_limit_hits=int(s.target_limit_up.sum()),
        close_limit_pct=round(s.loc[known, 'target_limit_up'].mean()*100, 2) if known.any() else None,
        close_limit_lower_bound_full_denominator_pct=round(s.target_limit_up.sum()/n*100, 2) if n else None,
        touch_limit_pct=round(s.loc[known, 'target_touch_limit'].mean()*100, 2) if known.any() else None,
        big_loss_pct=round(s.loc[known, 'target_big_loss'].mean()*100, 2) if known.any() else None,
        next_open_known=int(next_known.sum()), next_open_unknown=int((~next_known).sum()),
        next_open_gross_mean_pct=round(s.loc[next_known, 'next_return_proxy'].mean()*100, 3) if next_known.any() else None,
        next_open_positive_pct=round((s.loc[next_known, 'next_return_proxy'] > 0).mean()*100, 2) if next_known.any() else None)


def portfolio(signals, bars, sessions):
    """Conditional 09:30 open-price proxy; T+1 exits, cash, limits, MTM.

    No real fills are certified. A signal formed after the 09:25 uncross cannot
    join that uncross; the 09:30 execution price is approximated from the daily
    open plus slippage. Daily bars cannot verify that approximation.
    """
    initial = cash = 1_000_000.
    positions, records, trades, events = [], [], [], []
    wanted = set(signals.code)
    small = bars[bars.code.isin(wanted)].set_index(['date', 'code'])
    picks = {r.date: r for r in signals.itertuples()}
    date_set = [d for d in sessions if signals.date.min() <= d <= bars.date.max()]
    for day in date_set:
        active = []
        for pos in positions:
            if day > pos['entry_date'] and (day, pos['code']) in small.index:
                row = small.loc[(day, pos['code'])]
                # No assumed sale at a lower limit opening; retry on later sessions.
                if row.open > row.lower + .001 and row.open > 0:
                    price = max(row.lower, row.open * .999)
                    value = price * pos['shares']
                    fee = max(5., value*.0003) + value*(.0005 + .00001)
                    cash += value-fee
                    ret = (value-fee)/pos['entry_cost'] - 1
                    trades.append(dict(code=pos['code'], entry_date=pos['entry_date'], exit_date=day,
                                       shares=pos['shares'], entry_price=pos['entry_price'], exit_price=price,
                                       return_net=ret, pnl=value-fee-pos['entry_cost']))
                    continue
                events.append(dict(date=day, code=pos['code'], kind='SELL_BLOCKED_OR_NO_OPEN'))
            active.append(pos)
        positions = active
        pick = picks.get(day)
        if pick is not None:
            if len(positions) >= 2 or any(p['code'] == pick.code for p in positions):
                events.append(dict(date=day, code=pick.code, kind='SKIP_CAPACITY_OR_DUPLICATE'))
            else:
                row = small.loc[(day, pick.code)]
                if row.open < row.upper - .001 and row.open > row.lower + .001:
                    price = min(row.upper, row.open*1.001)
                    budget = min(cash, initial/2)
                    lot = 200 if pick.code.startswith('688') else 100
                    shares = int(max(0, budget-10) / (price*1.00031) / lot) * lot
                    value = price*shares
                    fee = max(5., value*.0003)+value*.00001
                    if shares and cash >= value+fee:
                        cash -= value+fee
                        positions.append(dict(code=pick.code, shares=shares, entry_price=price,
                                              entry_cost=value+fee, entry_date=day, mark=price))
                    else:
                        events.append(dict(date=day, code=pick.code, kind='SKIP_CASH_OR_LOT'))
                else:
                    events.append(dict(date=day, code=pick.code, kind='SKIP_OPEN_LIMIT'))
        for pos in positions:
            if (day, pos['code']) in small.index:
                close = small.loc[(day, pos['code']), 'close']
                if pd.notna(close) and close > 0:
                    pos['mark'] = close
        equity = cash + sum(p['shares']*p['mark'] for p in positions)
        records.append(dict(date=day, cash=cash, equity=equity, positions=len(positions)))
    curve = pd.DataFrame(records)
    tt = pd.DataFrame(trades)
    curve.to_csv(HERE/'portfolio_conditional_daily.csv', index=False, encoding='utf-8-sig')
    tt.to_csv(HERE/'portfolio_conditional_trades.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(events).to_csv(HERE/'portfolio_events.csv', index=False, encoding='utf-8-sig')
    returns = tt.return_net if len(tt) else pd.Series(dtype=float)
    pnl = tt.pnl if len(tt) else pd.Series(dtype=float)
    gains, losses = float(pnl[pnl>0].sum()), float(-pnl[pnl<0].sum())
    peaks = np.maximum.accumulate(np.r_[initial, curve.equity.to_numpy()])
    dd = np.r_[initial, curve.equity.to_numpy()]/peaks-1
    return dict(status='CONDITIONAL_DAILY_OPEN_PROXY_NOT_VERIFIED_FILLS', initial_cash=initial,
                end_equity=float(curve.equity.iloc[-1]), net_return_pct=round((curve.equity.iloc[-1]/initial-1)*100,2),
                max_drawdown_pct=round(float(dd.min()*100),2), closed_trades=len(tt),
                open_positions=len(positions), unclosed_positions=positions,
                win_pct=round(float((returns>0).mean()*100),2) if len(tt) else None,
                mean_trade_pct=round(float(returns.mean()*100),3) if len(tt) else None,
                average_win_pct=round(float(returns[returns>0].mean()*100),3) if (returns>0).any() else None,
                average_loss_pct=round(float(returns[returns<0].mean()*100),3) if (returns<0).any() else None,
                profit_factor=round(gains/losses,4) if losses else None,
                worst_trade_pct=round(float(returns.min()*100),3) if len(tt) else None,
                skip_or_block_events=len(events), confirmed_real_fills=0)


def run():
    sys.stdout.reconfigure(encoding='utf-8')
    if (HERE/'frozen_protocol.json').exists():
        raise RuntimeError('Fixed run already started. Preserve it; do not silently rerun.')
    daily = ROOT/'data/v3/imported/daily-bars.csv'
    calendar = ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json'
    source = ROOT/'results/gemini_5d_latest_review_20261028/source_snapshot'
    base = ROOT/'research/model_upgrade_validation/fixed_20261007T112017Z/models/relay/BASE/model.joblib'
    bindings = [daily, calendar, base, HERE/'causal_engine.py', Path(__file__).resolve(),
                ROOT/'legacy_industry_snapshot.py', source/'build_and_train_5d_dragon_model.py',
                ROOT/'research/first_board_github_industry_qualification_20261020/qualified_legacy_snapshot.json']
    hashes = {str(p): digest(p) for p in bindings}
    protocol = dict(formed_at_utc=datetime.now(timezone.utc).isoformat(),
        human_authorization='User requested Codex to recheck, fix bugs, and backtest on 2026-10-28.',
        latest_gemini_repair_not_located=True, starting_source=str(source),
        fixed_arms=1, feature_order=FEATURES, folds=FOLDS, params=PARAMS,
        no_effect_based_tuning=True, blind_test=False, fit_count_planned=4,
        scope='Legacy 10/20 percent board universe; archived current ST/new-list flags retained; survivorship remains.',
        target='Entry-day close at regular price-limit reference; not original prior-close BASE objective.',
        purging='Training sample next_date must be strictly before test_start to mature its next-open-dependent weight.',
        missing_tail_order='NaN; not synthesized from daily amplitude',
        comparators=['same-pool daily highest open gap', 'same-pool daily highest previous board streak'],
        execution='Conditional 09:30 daily-open proxy plus 10bp each side; T+1 or later exit; 2 positions; 500k max each; round lots; blocked down-limit sales; 3bp/min5 commission plus 0.1bp transfer each side and 5bp sell stamp tax.',
        source_hashes=hashes, production_replacement=False)
    write(HERE/'frozen_protocol.json',protocol)
    print('Protocol frozen before fitting.',flush=True)
    snapshot = HERE/'source_before_fix'
    snapshot.mkdir(exist_ok=True)
    shutil.copy2(source/'build_and_train_5d_dragon_model.py',snapshot/'build_and_train_5d_dragon_model.py')
    cal = json.loads(calendar.read_text(encoding='utf-8'))
    sessions = cal['sessions']
    mapping, metadata = load_snapshot()
    imap = mapping.set_index('code').legacy_group_PROXY_ALLOWED.to_dict()
    columns = ['date','code','name','open','high','low','close','pre_close','volume','amount','turnover','float_cap','st','new_listing','source']
    raw = pd.read_csv(daily,usecols=columns,dtype={'code':str},low_memory=False)
    t0 = time.time()
    d = engineer(raw,sessions,imap)
    print(f'Features constructed in {time.time()-t0:.1f}s.',flush=True)
    candidates = d[d.eligible].copy()
    bars = d[['date','code','open','high','low','close','upper','lower']].copy()
    quality = dict(raw_rows=len(raw), engineered_rows=len(d), eligible_rows=len(candidates),
        eligible_label_unknown=int(candidates.target_limit_up.isna().sum()),
        unknown_legacy_group_rows=int(candidates.legacy_group.isna().sum()),
        tail_order_observed_rows=int(candidates.tail_seal_momentum.notna().sum()),
        vwap_unknown_rows=int(candidates.prev_vwap.isna().sum()),
        data_last_date=str(d.date.max()), source_complete_market=False,
        status_point_in_time_verified=False, exchange_reference_independently_verified=False,
        legacy_group_policy=metadata['status'])
    write(HERE/'data_quality.json',quality)
    del d, raw
    predictions, folds = [], []
    for name,start,end,ts,te in FOLDS:
        base_train = candidates.date.between(start,end) & candidates.target_limit_up.notna()
        train = base_train & candidates.next_date.notna() & candidates.next_date.lt(ts)
        test = candidates.date.between(ts,te)
        tr, test_frame = candidates[train], candidates[test].copy()
        clf=HistGradientBoostingClassifier(**PARAMS)
        print(f'{name}: fitting {len(tr)} rows, evaluating {len(test_frame)}.',flush=True)
        clf.fit(tr[FEATURES],tr.target_limit_up.astype(int),sample_weight=tr.sample_weight)
        test_frame['score']=clf.predict_proba(test_frame[FEATURES])[:,1]
        test_frame['fold']=name
        known=test_frame.target_limit_up.notna()
        fold=dict(name=name,training_rows=len(tr),purged_training_rows=int(base_train.sum()-train.sum()),
            training_max_signal_date=str(tr.date.max()),training_max_weight_label_date=str(tr.next_date.max()),
            test_start=ts,test_end=te,test_rows=len(test_frame),test_known=int(known.sum()),
            roc_auc=float(roc_auc_score(test_frame.loc[known,'target_limit_up'],test_frame.loc[known,'score'])),
            pr_auc=float(average_precision_score(test_frame.loc[known,'target_limit_up'],test_frame.loc[known,'score'])))
        assert tr.next_date.max() < ts
        replay=clf.predict_proba(test_frame[FEATURES])[:,1]
        assert np.array_equal(replay,test_frame.score.to_numpy())
        joblib.dump(dict(model=clf,features=FEATURES,params=PARAMS,fold=fold),HERE/f'model_{name.replace(" ","_")}.joblib')
        folds.append(fold)
        keep=['date','code','name','fold','score','open_gap_pct','prev_board_streak',
              'target_limit_up','target_touch_limit','target_big_loss','target_day_return',
              'next_date','next_return_proxy','legacy_group']+FEATURES
        predictions.append(test_frame[list(dict.fromkeys(keep))])
    oof=pd.concat(predictions,ignore_index=True)
    assert not oof.duplicated(['date','code']).any()
    oof=oof.sort_values(['date','score','code'],ascending=[True,False,True]).reset_index(drop=True)
    oof['rank']=oof.groupby('date').cumcount()+1
    oof['rank_gap']=oof.groupby('date').open_gap_pct.rank(ascending=False,method='first')
    oof['rank_height']=oof.groupby('date').prev_board_streak.rank(ascending=False,method='first')
    oof.to_csv(HERE/'oof_all_candidates.csv',index=False,encoding='utf-8-sig')
    top=oof[oof['rank']==1]
    top.to_csv(HERE/'daily_top1.csv',index=False,encoding='utf-8-sig')
    full_dates=[s for s in sessions if FOLDS[0][3] <= s <= FOLDS[-1][4]]
    date_report=pd.DataFrame({'date':full_dates}).merge(oof.groupby('date').size().rename('eligible_count'),on='date',how='left').fillna({'eligible_count':0})
    date_report.to_csv(HERE/'all_official_test_dates.csv',index=False,encoding='utf-8-sig')
    metrics={'pool':stats(oof),'top1':stats(top),'top2':stats(oof[oof['rank']<=2]),
             'top3':stats(oof[oof['rank']<=3]),'gap_top1':stats(oof[oof.rank_gap==1]),
             'height_top1':stats(oof[oof.rank_height==1])}
    by_period={}
    for k,g in oof.groupby(oof.date.str[:4]):
        by_period[k]={'top1':stats(g[g['rank']==1]),'gap_top1':stats(g[g.rank_gap==1])}
    # Date-cluster bootstrap: confidence interval for paired Top1 seal difference.
    paired=top[['date','target_limit_up']].merge(oof[oof.rank_gap==1][['date','target_limit_up']],on='date',suffixes=('_fixed','_gap')).dropna()
    diff=(paired.target_limit_up_fixed-paired.target_limit_up_gap).to_numpy()
    rng=np.random.default_rng(42)
    boots=np.array([rng.choice(diff,size=len(diff),replace=True).mean()*100 for _ in range(2000)])
    compare=dict(paired_known_dates=len(diff),difference_pp=round(float(diff.mean()*100),3),
                 exploratory_paired_date_bootstrap_95pct_pp=np.quantile(boots,[.025,.975]).tolist(),
                 original_base_same_target_comparison_available=False)
    port=portfolio(top,bars,sessions)
    summary=dict(finished_at_utc=datetime.now(timezone.utc).isoformat(),fits=len(folds),folds=folds,
        all_official_test_dates=len(full_dates),no_eligible_observed_candidate_dates=int((date_report.eligible_count==0).sum()),
        metrics=metrics,by_period=by_period,paired_comparison=compare,conditional_portfolio=port,
        accepted_for_production=False,independent_blind_test=False,
        source_and_core_unchanged={p:digest(Path(p))==h for p,h in hashes.items()},
        model_scores_calibrated=False)
    write(HERE/'backtest_summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['fits','metrics','paired_comparison','conditional_portfolio']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    run()
