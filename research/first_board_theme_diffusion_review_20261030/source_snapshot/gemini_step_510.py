
import pandas as pd
import numpy as np

diag_df = pd.read_csv('research/first_board_top3_40_20261030/results/OUTCOMES_DIAGNOSTIC_ONLY.csv')
full_df = pd.read_csv('research/first_board_top3_40_20261030/results/FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv')
df = pd.merge(full_df, diag_df, on=['signal_date', 'code'])
df = df.dropna(subset=['promotion_next_LABEL_ONLY']).copy()
df['ind'] = df['legacy_industry_group_PROXY'].fillna('UNKNOWN')

industry_stats = []
for (d, ind), group in df.groupby(['signal_date', 'ind']):
    if ind == 'UNKNOWN': continue
    sorted_group = group.sort_values('reported_first_seal_trading_minutes')
    earliest_time = sorted_group['reported_first_seal_trading_minutes'].min()
    peer_count = len(sorted_group)
    
    for idx, row in sorted_group.iterrows():
        is_first = (row['reported_first_seal_trading_minutes'] == earliest_time)
        followers_30m = 0
        if is_first and peer_count > 1:
            followers = sorted_group[sorted_group['reported_first_seal_trading_minutes'] > earliest_time]
            followers_30m = (followers['reported_first_seal_trading_minutes'] <= earliest_time + 30).sum()
            
        industry_stats.append({
            'signal_date': d,
            'code': row['code'],
            'amount': row['amount'],
            'turnover': row['turnover'],
            'peer_count': peer_count,
            'is_first_in_theme': is_first,
            'followers_30m': followers_30m,
            'promotion': row['promotion_next_LABEL_ONLY']
        })

ind_df = pd.DataFrame(industry_stats)
first_with_followers = ind_df[(ind_df['is_first_in_theme'] == True) & (ind_df['followers_30m'] >= 1)]

# Subdivide by active absorption:
active_leader = first_with_followers[(first_with_followers['amount'] >= 2.0e8) | (first_with_followers['turnover'] >= 6.0)]
thin_leader = first_with_followers[(first_with_followers['amount'] < 2.0e8) & (first_with_followers['turnover'] < 6.0)]

p_act = active_leader['promotion'].mean() * 100
p_thin = thin_leader['promotion'].mean() * 100
print('Theme Leader WITH Followers & Active Volume: {:.2f}% ({}/{})'.format(p_act, int(active_leader['promotion'].sum()), len(active_leader)))
print('Theme Leader WITH Followers & Thin Volume:   {:.2f}% ({}/{})'.format(p_thin, int(thin_leader['promotion'].sum()), len(thin_leader)))
