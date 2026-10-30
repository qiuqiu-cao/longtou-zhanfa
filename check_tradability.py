import sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

daily = pd.read_csv('research/model_upgrade_validation/dragon_stage_leader_20261027T071500Z/daily_competition_round2.csv')
total = len(daily)
veto_count = int(daily['top1_audit_veto'].sum())
oneword_count = int((daily['top1_one_word_proxy'] == 1).sum())
tradable = daily[(daily['top1_audit_veto'] == False) & (daily['top1_one_word_proxy'] == 0)]

print(f"Total competitive days: {total}")
print(f"Top-1 Audit Veto count: {veto_count} ({veto_count/total:.2%})")
print(f"Top-1 One-Word Proxy count: {oneword_count} ({oneword_count/total:.2%})")
print(f"Fully Tradable Top-1 count (No Veto, Non-One-Word): {len(tradable)} ({len(tradable)/total:.2%})")
print(f"Fully Tradable Top-1 Space Leader Rate: {tradable['top1_is_space_leader'].mean():.2%}")
print(f"Fully Tradable Top-1 Emotion Dragon Rate: {tradable['top1_is_emotion_dragon'].mean():.2%}")
print(f"Fully Tradable Top-1 5-Board Rate: {tradable['top1_is_3to5'].mean():.2%}")
print(f"Fully Tradable Top-1 4-Board Rate: {tradable['top1_is_3to4'].mean():.2%}")
