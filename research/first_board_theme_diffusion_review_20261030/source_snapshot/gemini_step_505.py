
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
            'peer_count': peer_count,
            'is_first_in_theme': is_first,
            'followers_30m': followers_30m,
            'promotion': row['promotion_next_LABEL_ONLY']
        })

ind_df = pd.DataFrame(industry_stats)
print('Total stocks with known industry:', len(ind_df))

solo = ind_df[ind_df['peer_count'] == 1]
cluster = ind_df[ind_df['peer_count'] > 1]
p_solo = solo['promotion'].mean() * 100
p_clust = cluster['promotion'].mean() * 100
print('Solo stocks promotion rate:    {:.2f}% ({}/{})'.format(p_solo, int(solo['promotion'].sum()), len(solo)))
print('Cluster stocks promotion rate: {:.2f}% ({}/{})'.format(p_clust, int(cluster['promotion'].sum()), len(cluster)))

first_with_followers = cluster[(cluster['is_first_in_theme'] == True) & (cluster['followers_30m'] >= 1)]
first_without_followers = cluster[(cluster['is_first_in_theme'] == True) & (cluster['followers_30m'] == 0)]
followers_only = cluster[cluster['is_first_in_theme'] == False]

p_lead_f = first_with_followers['promotion'].mean() * 100
p_lead_nof = first_without_followers['promotion'].mean() * 100
p_foll = followers_only['promotion'].mean() * 100

print('Leader with >=1 followers in 30m: {:.2f}% ({}/{})'.format(p_lead_f, int(first_with_followers['promotion'].sum()), len(first_with_followers)))
print('Leader with 0 followers in 30m:   {:.2f}% ({}/{})'.format(p_lead_nof, int(first_without_followers['promotion'].sum()), len(first_without_followers)))
print('Followers (sealed later):         {:.2f}% ({}/{})'.format(p_foll, int(followers_only['promotion'].sum()), len(followers_only)))
