import sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('research/model_upgrade_validation/dragon_takeover_20261027T074500Z/round3_dataset.csv')
folds = ['2024-08-31', '2024-12-31', '2025-05-31', '2025-11-30']

for f in folds:
    total_d = len(df[df['date'] <= f])
    m_3to4 = len(df[(df['date'] <= f) & (df['determined_date_3to4'] <= f)])
    m_3to5 = len(df[(df['date'] <= f) & (df['determined_date_3to5'] <= f)])
    m_final = len(df[(df['date'] <= f) & (df['determined_date_final'] <= f)])
    print(f"Cutoff {f}: Total up to date={total_d}, Mature 3to4={m_3to4}, Mature 3to5={m_3to5}, Mature Final={m_final}")
