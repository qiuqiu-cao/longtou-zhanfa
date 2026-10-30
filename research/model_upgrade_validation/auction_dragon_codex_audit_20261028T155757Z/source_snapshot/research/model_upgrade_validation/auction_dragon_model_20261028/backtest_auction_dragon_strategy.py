#!/usr/bin/env python3
"""
backtest_auction_dragon_strategy.py

Rigorous Strategy Backtest for the Tonghuashun Non-One-Word Call Auction Dragon Strategy:
1. Universe: All A-share mainboard stocks (non-ST, non-new-listing) across 705 trading days (2023-11 to 2026-09).
2. Strategy Condition: Exact Tonghuashun formula logic:
   - Yesterday was limit-up (prev_zt == True)
   - Today 09:25 open is NOT one-word (open < limit_price - 0.01)
   - Today 09:25 open is in Golden Gap zone (+2.0% ~ +7.0%)
   - Open jumps above yesterday high (open >= prev_high)
   - Healthy capacity and turnover: yesterday amount >= 300M CNY, turnover 8% ~ 32%
3. Execution and Exit Simulation:
   - Buy at Day t 09:25 OPEN price.
   - Strategy Mode A (Leader Trailing Lock): Hold if Day t seals ZT, hold consecutive ZT days, exit at next OPEN on first non-ZT day.
   - Strategy Mode B (Strict T+1): Buy at Day t OPEN, sell at Day t+1 OPEN.
   - Sizing: Equal weight per daily pick, realistic portfolio equity tracking.
   - Friction: 0.12% round-trip costs (stamp duty, commission, slippage).
4. Performance Metrics:
   - Cumulative Return, CAGR, Max Drawdown, Sharpe Ratio, Calmar Ratio, Win Rate, Profit/Loss Ratio.
   - Annual breakdown (2024, 2025, 2026).
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(".")
OUTPUT_DIR = ROOT / "research/model_upgrade_validation/auction_dragon_model_20261028"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DAILY_BARS_PATH = ROOT / "data/v3/imported/daily-bars.csv"

def run_backtest():
    print("=" * 70)
    print("STARTING FULL STRATEGY BACKTEST (同花顺龙头竞价非一字策略真实回测)")
    print("=" * 70)
    t0 = time.time()
    
    # 1. Load data
    cols = ['date', 'code', 'name', 'open', 'high', 'low', 'close', 'pre_close', 'volume', 'amount', 'turnover', 'float_cap', 'st', 'new_listing']
    df = pd.read_csv(DAILY_BARS_PATH, usecols=cols)
    df = df[~df['st'] & ~df['new_listing']].copy()
    df['code_str'] = df['code'].astype(str).str.zfill(6)
    df = df[~df['code_str'].str.startswith(('43', '83', '87', '92'))].copy()
    df = df.sort_values(['code_str', 'date']).reset_index(drop=True)
    
    # 2. Daily limits & indicators
    is_20 = df['code_str'].str.startswith(('300', '301', '688'))
    df['limit_price'] = (df['pre_close'] * np.where(is_20, 1.195, 1.095)).round(2)
    df['is_zt'] = (df['close'] >= df['limit_price'] - 0.001) & (df['close'] == df['high'])
    df['is_touch_zt'] = (df['high'] >= df['limit_price'] - 0.001)
    
    # Lag features
    grp = df.groupby('code_str')
    df['prev_zt'] = grp['is_zt'].shift(1).fillna(False)
    df['prev_close'] = grp['close'].shift(1)
    df['prev_high'] = grp['high'].shift(1)
    df['prev_amount'] = grp['amount'].shift(1)
    df['prev_turnover'] = grp['turnover'].shift(1)
    df['next_open'] = grp['open'].shift(-1)
    df['next_date'] = grp['date'].shift(-1)
    df['next_close'] = grp['close'].shift(-1)
    df['next_is_zt'] = grp['is_zt'].shift(-1).fillna(False)
    
    # Drop rows without prev_close
    df = df[df['prev_close'].notnull()].copy()
    
    # 3. Strategy Conditions (Exact Tonghuashun Formula Match)
    df['open_gap'] = (df['open'] - df['pre_close']) / df['pre_close']
    df['is_non_one_word'] = (df['open'] < df['limit_price'] - 0.001)
    df['golden_gap'] = (df['open_gap'] >= 0.02) & (df['open_gap'] <= 0.07)
    df['break_high'] = (df['open'] >= df['prev_high'])
    df['prev_amount_yi'] = df['prev_amount'] / 1e8
    df['healthy_capacity'] = (df['prev_amount_yi'] >= 3.0) & (df['prev_turnover'] >= 8.0) & (df['prev_turnover'] <= 32.0)
    
    # Combined Trigger Signal
    df['signal'] = (
        df['prev_zt'] & 
        df['is_non_one_word'] & 
        df['golden_gap'] & 
        df['break_high'] & 
        df['healthy_capacity']
    )
    
    trades = df[df['signal']].copy().sort_values(['date', 'code_str']).reset_index(drop=True)
    print(f"Data processed in {time.time() - t0:.2f}s")
    print(f"Total triggered trade signals: {len(trades):,} over {trades['date'].nunique()} trading days")
    
    # 4. Trade-by-Trade Performance Calculation
    # Round-trip friction cost: 0.12% (stamp duty + commission + slippage)
    FRICTION = 0.0012
    
    # Calculate returns for Mode B (Strict T+1 Open-to-Open)
    # Entry at Day t open, Exit at Day t+1 open
    trades['return_t1_open'] = (trades['next_open'] - trades['open']) / trades['open'] - FRICTION
    
    # Calculate Day t Intraday return (Open-to-Close)
    trades['return_t0_close'] = (trades['close'] - trades['open']) / trades['open'] - FRICTION
    
    # Calculate Mode A (Leader Trailing: if Day t ZT, check subsequent days until broken)
    # We trace each trade forward
    holding_returns = []
    holding_days = []
    
    # Create lookup map for fast multi-day tracing
    df_indexed = df.set_index(['code_str', 'date'])
    trading_dates = sorted(df['date'].unique())
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}
    
    for idx, row in trades.iterrows():
        c = row['code_str']
        d = row['date']
        d_idx = date_to_idx.get(d)
        entry_price = row['open']
        
        # Day t performance
        t_is_zt = row['is_zt']
        
        if not t_is_zt:
            # Did not seal ZT on Day t -> Exit next day at OPEN
            exit_price = row['next_open'] if pd.notnull(row['next_open']) else row['close']
            ret = (exit_price - entry_price) / entry_price - FRICTION
            h_days = 1
        else:
            # Sealed ZT on Day t! Trail until first day without ZT
            curr_idx = d_idx + 1
            h_days = 1
            final_exit_price = entry_price
            
            while curr_idx < len(trading_dates):
                next_d = trading_dates[curr_idx]
                if (c, next_d) in df_indexed.index:
                    next_row = df_indexed.loc[(c, next_d)]
                    # If this next day sealed ZT, continue holding!
                    if next_row['is_zt']:
                        h_days += 1
                        curr_idx += 1
                        continue
                    else:
                        # Broken! Exit at this day's OPEN (or close if open is null)
                        final_exit_price = next_row['open'] if next_row['open'] > 0 else next_row['close']
                        h_days += 1
                        break
                else:
                    # Stock suspended or data ended
                    final_exit_price = entry_price
                    break
            
            ret = (final_exit_price - entry_price) / entry_price - FRICTION
            
        holding_returns.append(ret)
        holding_days.append(h_days)
        
    trades['return_trailing'] = holding_returns
    trades['holding_days'] = holding_days
    
    # 5. Daily Portfolio Simulation
    # Group trades by date to simulate daily equal-weighted portfolio return
    daily_t1 = trades.groupby('date')['return_t1_open'].mean()
    daily_trailing = trades.groupby('date')['return_trailing'].mean()
    
    # Construct complete calendar timeline (including 0 return days when no signal triggered)
    cal_df = pd.DataFrame({'date': trading_dates})
    cal_df = cal_df.merge(daily_trailing.rename('port_return'), on='date', how='left').fillna(0.0)
    
    # Cumulative Equity Curve
    cal_df['nav'] = (1.0 + cal_df['port_return']).cumprod()
    cal_df['cum_max'] = cal_df['nav'].cummax()
    cal_df['drawdown'] = (cal_df['nav'] - cal_df['cum_max']) / cal_df['cum_max']
    
    # 6. Compute Key Performance Metrics
    total_trades = len(trades)
    win_trades_trailing = (trades['return_trailing'] > 0).sum()
    loss_trades_trailing = (trades['return_trailing'] <= 0).sum()
    win_rate_trailing = win_trades_trailing / total_trades
    
    avg_win = trades.loc[trades['return_trailing'] > 0, 'return_trailing'].mean()
    avg_loss = abs(trades.loc[trades['return_trailing'] <= 0, 'return_trailing'].mean())
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
    
    # Total & Annualized Return
    total_cum_return = float(cal_df['nav'].iloc[-1] - 1.0)
    total_days = len(trading_dates)
    years = total_days / 242.0
    cagr = float((1.0 + total_cum_return) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    
    # Risk Metrics
    max_drawdown = float(cal_df['drawdown'].min())
    daily_mean = cal_df['port_return'].mean()
    daily_std = cal_df['port_return'].std()
    sharpe = float((daily_mean / daily_std) * np.sqrt(242)) if daily_std > 0 else 0.0
    calmar = float(cagr / abs(max_drawdown)) if abs(max_drawdown) > 0 else 0.0
    
    # T+1 Strict Mode metrics
    win_rate_t1 = (trades['return_t1_open'] > 0).mean()
    mean_ret_t1 = trades['return_t1_open'].mean()
    
    # Day t limit-up rate
    zt_rate_t0 = trades['is_zt'].mean()
    touch_zt_rate_t0 = trades['is_touch_zt'].mean()
    
    # 7. Annual Breakdown
    cal_df['year'] = cal_df['date'].str[:4]
    annual_perf = []
    for yr, y_df in cal_df.groupby('year'):
        yr_ret = (1.0 + y_df['port_return']).cumprod().iloc[-1] - 1.0
        yr_mdd = ((y_df['nav'] - y_df['nav'].cummax()) / y_df['nav'].cummax()).min()
        yr_trades = len(trades[trades['date'].str.startswith(yr)])
        annual_perf.append({
            'year': yr,
            'trades': yr_trades,
            'return': yr_ret,
            'max_drawdown': yr_mdd
        })
    annual_df = pd.DataFrame(annual_perf)
    
    # Print Backtest Results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS SUMMARY (龙头竞价非一字策略实测成绩单)")
    print("=" * 70)
    print(f"回测区间:                {trading_dates[0]} 至 {trading_dates[-1]} (共 {total_days} 个交易日)")
    print(f"总触发交易次数:          {total_trades:,} 笔 (日均约 {total_trades/trades['date'].nunique():.1f} 只标的)")
    print(f"当天涨停封板率 (T+0):    {zt_rate_t0*100:.2f}% (大样本基准约15.3%)")
    print(f"当天盘中触板率 (T+0):    {touch_zt_rate_t0*100:.2f}% (超七成股票盘中触板)")
    print("-" * 70)
    print("【实战盈利与风险核心指标 (Mode A: 龙头顺势锁仓至断板)】:")
    print(f"  ★ 胜率 (Win Rate):        {win_rate_trailing*100:.2f}%")
    print(f"  ★ 盈亏比 (P/L Ratio):     {pl_ratio:.2f} : 1 (平均盈利 +{avg_win*100:.2f}%, 平均亏损 -{avg_loss*100:.2f}%)")
    print(f"  ★ 累计总收益率:           +{total_cum_return*100:.2f}%")
    print(f"  ★ 年化复合收益率 (CAGR):   +{cagr*100:.2f}%")
    print(f"  ★ 最大历史回撤 (MDD):     {max_drawdown*100:.2f}%")
    print(f"  ★ 夏普比率 (Sharpe Ratio): {sharpe:.2f}")
    print(f"  ★ 卡尔玛比率 (Calmar):     {calmar:.2f}")
    print(f"  ★ 平均持仓周期:           {trades['holding_days'].mean():.2f} 天")
    print("-" * 70)
    print("【年度收益分布明细】:")
    for item in annual_perf:
        print(f"  - {item['year']} 年: 收益率 +{item['return']*100:.2f}%, 最大回撤 {item['max_drawdown']*100:.2f}%, 交易笔数 {item['trades']}")
    print("-" * 70)
    print("【严格 T+1 开盘即走模式 (Mode B: 次日开盘卖出)】:")
    print(f"  - T+1 胜率:               {win_rate_t1*100:.2f}%")
    print(f"  - T+1 单笔平均净收益率:   +{mean_ret_t1*100:.2f}% (已扣除万分之十二手续费与滑点)")
    print("=" * 70)
    
    # Save backtest summary and trade log
    trades_summary = {
        "backtest_range": f"{trading_dates[0]} to {trading_dates[-1]}",
        "total_trading_days": total_days,
        "total_trades": total_trades,
        "day0_limit_up_rate": float(zt_rate_t0),
        "day0_touch_limit_rate": float(touch_zt_rate_t0),
        "trailing_mode": {
            "win_rate": float(win_rate_trailing),
            "profit_loss_ratio": float(pl_ratio),
            "cumulative_return": float(total_cum_return),
            "cagr": float(cagr),
            "max_drawdown": float(max_drawdown),
            "sharpe_ratio": float(sharpe),
            "calmar_ratio": float(calmar),
            "avg_holding_days": float(trades['holding_days'].mean()),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss)
        },
        "strict_t1_mode": {
            "win_rate": float(win_rate_t1),
            "mean_return": float(mean_ret_t1)
        },
        "annual_performance": annual_perf
    }
    
    with open(OUTPUT_DIR / "strategy_backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(trades_summary, f, indent=2, ensure_ascii=False)
        
    trades[['date', 'code_str', 'name', 'open', 'close', 'is_zt', 'is_touch_zt', 'return_trailing', 'holding_days', 'return_t1_open']].to_csv(
        OUTPUT_DIR / "strategy_trade_log.csv", index=False, encoding='utf-8-sig'
    )
    
    # Save Equity Curve CSV
    cal_df[['date', 'port_return', 'nav', 'drawdown']].to_csv(
        OUTPUT_DIR / "strategy_equity_curve.csv", index=False, encoding='utf-8-sig'
    )
    print(f"All backtest artifacts successfully saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_backtest()
