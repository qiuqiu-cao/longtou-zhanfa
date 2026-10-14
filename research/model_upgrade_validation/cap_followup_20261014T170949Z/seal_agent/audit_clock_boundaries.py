"""Supplemental read-only contract/coverage audit; no parser change."""
from pathlib import Path
import json
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
rows = pd.read_csv(ROOT / "data/v275/reported_seal_archive_20261002_v2/reported_seal_rows.csv", dtype={"code":str})
relay = rows[rows.boards.ge(2)]
full = joblib.load(ROOT / "research/model_upgrade_validation/relay_context_20261011T151204Z/prepared_source_reconstruction.joblib")["relay"]
bad = relay[relay.reported_seal_status.ne("PROVIDER_REPORTED_VALID_CLOCKS")]
assert len(bad) == 26
assert bad.first_seal.isin([92501,92502,92503,92504]).all()
coverage = []
for date, r in relay.groupby("date"):
    original = full[full.date.eq(date)]
    same = set(r.code) == set(original.code)
    assert same
    coverage.append({"date":date,"archived_relay":len(r),"original_pool":len(original),"complete_candidate_set_exact":same})
result = {"status":"SOURCE_OBSERVATION_CONTRACT_DIAGNOSTIC_ONLY","original_contract_invalid_relay_rows":len(bad),"invalid_first_clock_counts":{str(k):int(v) for k,v in bad.first_seal.value_counts().items()},"all_invalid_reason":"reported first-seal09:25:01 through09:25:04 excluded by exact09:25:00 admission rule; raw reported value is retained","invalid_same_first_last":int(bad.first_seal.eq(bad.last_seal).sum()),"invalid_positive_reported_bomb_count":int(bad.bomb_count.gt(0).sum()),"not_zero_or_garbage":True,"provider_clock_semantics_confirmed":False,"do_not_silently_expand_clock_gate":True,"full_original_candidate_coverage":coverage,"sealed_fund_amount_ratio_observed_min":float((relay.sealed_fund/relay.amount).min()),"sealed_fund_amount_ratio_observed_max":float((relay.sealed_fund/relay.amount).max()),"ratio_not_queue_safety_or_buyability":True,"delayed_capture_specific_example":{"signal_date":"2026-09-17","candidate_rows":9,"received_manifest":"2026-09-21T18:46:37","original_decision_time_qualified":False},"no_source_or_score_changes":True,"new_model_fits":0}
bad[["date","code","first_seal","last_seal","bomb_count","reported_seal_status","source_path","captured_at"]].to_csv(OUT / "all26_auction_adjacent_clocks_UNKNOWN.csv", index=False, encoding="utf-8-sig")
(OUT / "clock_boundary_and_candidate_coverage.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
print(json.dumps(result,ensure_ascii=False,indent=2))
