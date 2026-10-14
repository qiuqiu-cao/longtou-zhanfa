"""Explain exclusions without using them to silently improve rankings."""
import json
from pathlib import Path

import pandas as pd

from target_proxy_experiment_v24 import audit_proxy_labels


if __name__ == "__main__":
    root=Path(__file__).resolve().parent
    target=root/"results/v26/unresolved_audit_20260904"
    target.mkdir(parents=True,exist_ok=False)
    features=pd.read_csv(root/"data/v23/candidate_emotion_features_strict.csv",dtype={"code":str})
    episodes=pd.read_csv(root/"data/v20/episode_inventory_20260904_strict/episode_outcomes_LABEL_ONLY.csv",dtype={"code":str})
    events=json.loads((root/"data/v19/target_event_library_20260904.json").read_text(encoding="utf-8"))["events"]
    calendar=pd.read_csv(root/"data/v18/benchmarks_20260904/sz399107.csv").date.tolist()
    audit=audit_proxy_labels(features,episodes,events,calendar)
    fields=["code","signal_date","episode_id","label_status","proxy_target","label_end"]
    audit[fields].to_csv(target/"all_candidate_label_reasons.csv",index=False,encoding="utf-8-sig")
    predictions=pd.read_csv(root/"results/v24/proxy_baseline_full_pool_20260904/exploratory_predictions_NOT_LIVE.csv",dtype={"code":str})
    top=predictions.sort_values(["signal_date","proxy_probability_NOT_LIVE","code"],ascending=[True,False,True]).drop_duplicates("signal_date")
    top=top.sort_values(["signal_date","code"]).drop_duplicates("episode_id")
    top=top.drop(columns="label_status").merge(audit[["code","signal_date","label_status"]],on=["code","signal_date"],validate="one_to_one")
    unresolved=top[top.proxy_target.isna()]
    unresolved.to_csv(target/"selected_unresolved_episodes.csv",index=False,encoding="utf-8-sig")
    report=dict(status="DIAGNOSTIC_ONLY",all_rows=len(audit),
                all_reason_counts=audit.label_status.value_counts().to_dict(),
                selected_unresolved_episodes=len(unresolved),
                selected_unresolved_reasons=unresolved.label_status.value_counts().to_dict(),
                note="Already-occurred proxy events require contemporaneous threshold evidence before entry exclusion; no labels repaired here.")
    (target/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
