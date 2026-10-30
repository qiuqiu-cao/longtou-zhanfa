import sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('research/model_upgrade_validation/dragon_stage_leader_20261027T071500Z/stage_leader_dataset.csv')
b3 = df[df['boards'] == 3].copy()
b3['is_already_space_leader'] = (b3['height_gap'] == 0).astype(int)

case_b = b3[b3['is_already_space_leader'] == 0].copy()
total_b = len(case_b)
takeover_space = int(case_b['is_market_space_leader'].sum())
takeover_unique = int(case_b['is_unique_space_leader'].sum())
dragon = int(case_b['is_emotion_dragon_proxy'].sum())

print(f"Total Case B (Not Space Leader at 3 boards): {total_b}")
print(f"Case B subsequently TOOK OVER Space Leader: {takeover_space} ({takeover_space/total_b:.2%})")
print(f"Case B subsequently became UNIQUE Space Leader: {takeover_unique} ({takeover_unique/total_b:.2%})")
print(f"Case B became Emotion Dragon: {dragon} ({dragon/total_b:.2%})")
