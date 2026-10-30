
import pandas as pd
import numpy as np

diag_df = pd.read_csv('research/first_board_top3_40_20261030/results/OUTCOMES_DIAGNOSTIC_ONLY.csv')
full_df = pd.read_csv('research/first_board_top3_40_20261030/results/FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv')
df = pd.merge(full_df, diag_df, on=['signal_date', 'code'])
df = df.dropna(subset=['promotion_next_LABEL_ONLY']).copy()
df['ind'] = df['legacy_industry_group_PROXY'].fillna('UNKNOWN')

# Construct follower metric
leader_map = {}
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
        leader_map[(d, row['code'])] = (is_first, followers_30m, peer_count)

df['is_theme_lead'] = df.apply(lambda r: leader_map.get((r['signal_date'], r['code']), (False, 0, 1))[0], axis=1)
df['followers_30m'] = df.apply(lambda r: leader_map.get((r['signal_date'], r['code']), (False, 0, 1))[1], axis=1)
df['peer_count'] = df.apply(lambda r: leader_map.get((r['signal_date'], r['code']), (False, 0, 1))[2], axis=1)
df['month'] = df['signal_date'].str.slice(0, 7)

# Evaluate swap:
# In S1 Top 3, if Rank 2 or 3 is a follower (sealed later) or solo with 0 peers,
# and in Rank 4 to 8 there is a theme leader with >= 1 followers within 30 min,
# swap it!
def test_theme_diffusion_swap(df_subset):
    promoted_w = 0
    promoted_l = 0
    displaced_w = 0
    displaced_l = 0
    
    dates = df_subset['signal_date'].unique()
    for d in dates:
        d_sub = df_subset[df_subset['signal_date'] == d].sort_values('s1_score', ascending=False)
        if len(d_sub) < 3: continue
        r1, r2, r3 = d_sub.iloc[0], d_sub.iloc[1], d_sub.iloc[2]
        
        # Check if r2 or r3 is a follower or zero-diffusion
        target_swap_idx = None
        for idx, r in [(1, r2), (2, r3)]:
            if (not r['is_theme_lead']) or (r['followers_30m'] == 0):
                target_swap_idx = idx
                break
                
        if target_swap_idx is None: continue
        
        # Find replacement in Rank 4-8 that IS a theme leader with followers!
        pool_rep = d_sub.iloc[3:min(8, len(d_sub))]
        good_leads = pool_rep[(pool_rep['is_theme_lead'] == True) & (pool_rep['followers_30m'] >= 1)]
        if len(good_leads) == 0: continue
        
        rep_stock = good_leads.iloc[0]
        displaced_stock = d_sub.iloc[target_swap_idx]
        
        if rep_stock['promotion_next_LABEL_ONLY'] == 1: promoted_w += 1
        else: promoted_l += 1
        
        if displaced_stock['promotion_next_LABEL_ONLY'] == 1: displaced_w += 1
        else: displaced_l += 1
        
    return {
        'promoted_w': promoted_w, 'promoted_l': promoted_l,
        'displaced_w': displaced_w, 'displaced_l': displaced_l,
        'net_gain': promoted_w - displaced_w
    }

print('=== Train (2025-07 to 2026-04) ===')
print(test_theme_diffusion_swap(df[df['month'] <= '2026-04']))

print('=== Test OOT (2026-05 to 2026-08) ===')
print(test_theme_diffusion_swap(df[df['month'] >= '2026-05']))
