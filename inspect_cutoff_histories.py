import sys
from pathlib import Path
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

codes = ['002084', '002855', '600371', '000712', '002396', '002886', '600540', '603269']
base_dir = Path('data/v3/imported/history-parts')

# Load and concatenate history
for c in codes:
    dfs = []
    # check 20240101_20260828
    f1 = base_dir / '20240101_20260828' / f'{c}.csv'
    if f1.exists():
        dfs.append(pd.read_csv(f1))
    f2 = base_dir / '20260901_20260916' / f'{c}.csv'
    if f2.exists():
        dfs.append(pd.read_csv(f2))
    f3 = base_dir / '20260901_20260901' / f'{c}.csv'
    if f3.exists():
        dfs.append(pd.read_csv(f3))
        
    # special for 002084
    if c == '002084':
        f_spec = Path('data/2026-09_lianban/hist/002084.csv')
        if f_spec.exists():
            dfs.append(pd.read_csv(f_spec))
            
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        date_col = 'date' if 'date' in df.columns else 'trade_date' if 'trade_date' in df.columns else df.columns[0]
        df[date_col] = df[date_col].astype(str).str.replace('-', '')
        df = df.drop_duplicates(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
        # Filter around late August / early September 2026
        sub = df[(df[date_col] >= '20260824') & (df[date_col] <= '20260908')].copy()
        print(f"\n=== Code {c} ===")
        # Calculate pct_chg if not present
        if 'pct_chg' not in sub.columns and 'close' in sub.columns:
            sub['pct_chg'] = sub['close'].pct_change() * 100
        cols_to_print = [c for c in [date_col, 'open', 'high', 'low', 'close', 'pct_chg', 'vol', 'volume'] if c in sub.columns]
        print(sub[cols_to_print].to_string())
    else:
        print(f"Code {c} not found in history parts")
