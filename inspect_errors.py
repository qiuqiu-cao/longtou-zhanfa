import sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

daily = pd.read_csv('research/model_upgrade_validation/dragon_stage_leader_20261027T071500Z/daily_competition_round2.csv')
oof = pd.read_csv('research/model_upgrade_validation/dragon_stage_leader_20261027T071500Z/oof_round2_predictions.csv')

fn_days = daily[(daily['actual_dragon_count'] > 0) & (~daily['top1_is_emotion_dragon'])]
print(f"Total False Negative competitive days: {len(fn_days)}")

for _, r in fn_days.head(8).iterrows():
    d = r['date']
    cohort = oof[oof['date'] == d].sort_values('score_composite_dragon', ascending=False)
    actual_dragons = cohort[cohort['is_emotion_dragon_proxy'] == 1]
    top1_name = r['top1_name']
    top1_code = r['top1_code']
    top1_score = r['top1_score']
    top1_tb = r['top1_terminal_board']
    print(f"Date {d}: Top-1 Pick: {top1_name} ({top1_code}), Score={top1_score:.4f}, Terminal Board={top1_tb}")
    for _, ad in actual_dragons.iterrows():
        print(f"   --> Missed Dragon: {ad['name']} ({ad['code']}), Score={ad['score_composite_dragon']:.4f}, Terminal Board={ad['terminal_board']}")
