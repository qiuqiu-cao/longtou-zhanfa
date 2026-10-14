"""Read-only bounded source audit. Never fit models or fetch market data."""
from pathlib import Path
import hashlib
import json
import re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def write(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def clock(value):
    if value is None or pd.isna(value):
        return None
    token = str(value).strip().split(".")[0].replace(":", "").zfill(6)
    if not re.fullmatch(r"\d{6}", token):
        return None
    hh, mm, ss = map(int, (token[:2], token[2:4], token[4:]))
    if hh > 23 or mm > 59 or ss > 59:
        return None
    secs = 3600 * hh + 60 * mm + ss
    return secs if secs == 33900 or 34200 <= secs <= 41400 or 46800 <= secs <= 54000 else None

archive = ROOT / "data/v275/reported_seal_archive_20261002_v2/reported_seal_rows.csv"
rows = pd.read_csv(archive, dtype={"code": str})
report = read(archive.parent / "report.json")
relay = rows[rows.boards.ge(2)].copy()
used_source_audit = []
raw_field_mismatches = []
day_audit = []
maps = {"代码":"code", "名称":"name", "连板数":"boards", "最新价":"price", "换手率":"turnover", "首次封板时间":"first_seal", "最后封板时间":"last_seal", "炸板次数":"bomb_count", "成交额":"amount", "流通市值":"float_cap", "封板资金":"sealed_fund"}
for date, group in rows.groupby("date", sort=True):
    source = ROOT / group.source_path.iloc[0]
    meta_path = source.parent / "manifest.json"
    meta = read(meta_path) if meta_path.exists() else {}
    raw = pd.read_csv(source, dtype={"代码": str}).rename(columns=maps)
    raw["code"] = raw.code.str.zfill(6)
    raw = raw[raw.code.str.match(r"^(60|00)\d{4}$") & ~raw.name.str.contains(r"ST|退", regex=True)].set_index("code")
    capture_block = meta.get("datasets", {}).get("limit_up", {})
    actual_sha = sha(source)
    assert group.source_sha256.eq(actual_sha).all()
    assert set(raw.index) == set(group.code), f"candidate coverage mismatch {date}"
    for r in group.itertuples(index=False):
        for field in ["boards", "price", "turnover", "first_seal", "last_seal", "bomb_count", "amount", "float_cap", "sealed_fund"]:
            a = float(getattr(r, field))
            b = float(raw.loc[r.code, field])
            if not np.isclose(a, b, rtol=1e-12, atol=1e-8, equal_nan=True):
                raw_field_mismatches.append({"date":date,"code":r.code,"field":field,"archive":a,"source":b})
    gr = group[group.boards.ge(2)]
    capture = str(group.captured_at.iloc[0])
    aware = bool(re.search(r"(?:Z|[+-]\d\d:\d\d)$", capture))
    # Naive capture timestamps are kept naive. Do not assign a timezone for a strict gate.
    captured_after_signal = capture[:10] >= date if capture != "nan" else None
    before_target = capture[:10] < str(group.target_date.iloc[0]) if capture != "nan" else None
    fact = {"date":date,"rows":len(group),"relay_rows":len(gr),"raw_candidate_set_exact":True,"raw_field_values_exact":True,"source_path":str(source.relative_to(ROOT)),"source_sha256":actual_sha,"original_manifest_sha_present":bool(capture_block.get("sha256")),"manifest_hash_match_if_present":capture_block.get("sha256") in (None,actual_sha),"captured_at":None if capture == "nan" else capture,"capture_has_timezone":aware,"capture_date_on_or_after_signal":captured_after_signal,"capture_date_before_target":before_target,"provider_market_snapshot_exact_time":None,"valid_clock_relay":int(gr.reported_seal_status.eq("PROVIDER_REPORTED_VALID_CLOCKS").sum()),"relay_labels_known":int(gr.promoted_next_day.notna().sum())}
    day_audit.append(fact)
    used_source_audit.append({"path":str(source),"sha256":actual_sha,"manifest_path":str(meta_path),"manifest_sha256":sha(meta_path)})

assert not raw_field_mismatches
independent_first = rows.first_seal.map(clock)
independent_last = rows.last_seal.map(clock)
valid = independent_first.notna() & independent_last.notna() & independent_last.ge(independent_first) & rows.bomb_count.ge(0)
assert valid.eq(rows.reported_seal_status.eq("PROVIDER_REPORTED_VALID_CLOCKS")).all()
assert np.allclose(independent_first.fillna(-1), rows.reported_first_seconds.fillna(-1), rtol=0, atol=0)
assert np.allclose(independent_last.fillna(-1), rows.reported_last_seconds.fillna(-1), rtol=0, atol=0)

prepared_path = ROOT / "research/model_upgrade_validation/relay_context_20261011T151204Z/prepared_source_reconstruction.joblib"
full = joblib.load(prepared_path)["relay"]
keys = set(zip(relay.date, relay.code))
overlap = []
for name, start in [("2025H2","2025-07-01"),("2026JanAug","2026-01-01"),("2026Sep","2026-09-01")]:
    eligible = full[full.label_valid.fillna(False).astype(bool) & full.target.notna() & full.date.lt(start) & full.target_date.lt(start)]
    dates = sorted(eligible.date.unique())[-252:]
    training = eligible[eligible.date.isin(dates)]
    matched = training[["date", "code"]].apply(tuple, axis=1).isin(keys)
    overlap.append({"fold":name,"training_rows":len(training),"training_dates":training.date.nunique(),"key_overlap":int(matched.sum()),"fraction":float(matched.mean())})

placeholder_counts = {}
for f in ["first_seal_minute","last_seal_minute","bomb_count"]:
    values = pd.to_numeric(full[f], errors="coerce")
    placeholder_counts[f] = {"nonmissing":int(values.notna().sum()),"zero":int(values.eq(0).sum()),"positive":int(values.gt(0).sum())}

calendar = report["calendar"]["sessions"]
available_days = set(rows.date)
blocks = []
for date in calendar:
    if date in available_days:
        if blocks and calendar.index(date) == calendar.index(blocks[-1][-1]) + 1:
            blocks[-1].append(date)
        else:
            blocks.append([date])
block_summary = []
for dates in blocks:
    gr = relay[relay.date.isin(dates)]
    block_summary.append({"dates":dates,"relay_rows":len(gr),"known_relay_labels":int(gr.promoted_next_day.notna().sum()),"unknown_relay_labels":int(gr.promoted_next_day.isna().sum()),"independent_unseen_holdout":False})

bank_path = ROOT / "research/model_upgrade_validation/leader_role_sampled_path_20261012T153639Z/actual_sampled_history_CAPTURE_BOUND/sampled_path_evidence_bank.json"
bank = read(bank_path)
sampled = [r["groups"]["sampled_path"]["value"] for r in bank["rows"]]
sam_counts = {"rows":len(sampled),"dates":len(set(r["signal_date"] for r in sampled)),"price_reconciled":sum(r["price_reconciled"] for r in sampled),"full_grid":sum(r["grid_complete"] for r in sampled),"volume_reconciled":sum(r["volume_reconciled"] for r in sampled),"original_capture_hash":sum(r["provider_reported_seal"]["original_capture_manifest_hash_present"] for r in sampled),"report_bomb_positive_no_sampled_close_drop":sum(float(r["provider_reported_seal"]["bomb_count"]) > 0 and r["features"]["sampled_close_drop_after_limit"] == 0 for r in sampled),"sampled_drop_and_return":sum(r["features"]["sampled_later_limit_close_after_drop"] == 1 for r in sampled),"tick_path_verified":sum(r["tick_path_verified"] for r in sampled),"order_book_verified":sum(r["order_book_seal_verified"] for r in sampled),"absorption_verified":sum(r["absorption_verified"] for r in sampled)}
sept_report = read(ROOT / "data/v287/september_five_minute_20261002_v1/report.json")
raw_minute_hash_checks = [{"path":r["raw_path"],"sha256":sha(r["raw_path"]),"match":sha(r["raw_path"]) == r["raw_sha256"]} for r in sept_report["audit"] if r.get("raw_path")]
assert all(r["match"] for r in raw_minute_hash_checks)

# The bounded V49 recovery tree contains source rows, not seal/queue records.
v49cases = []
for p in sorted((ROOT / "data/v49/first_board_minutes_recovery").glob("batch_*/*_*.json")):
    data = read(p)
    if "rows" not in data or "fields" not in data:
        continue
    bars = data["rows"]
    fields = data["fields"]
    date_idx = fields.index("date")
    wrong = any(r[date_idx] != data.get("requested_date") for r in bars)
    v49cases.append({"path":str(p.relative_to(ROOT)),"code":data["code"],"requested_date":data["requested_date"],"rows":len(bars),"date_exact":not wrong,"adjustflag":data.get("adjustflag"),"contains_seal_or_orderbook_fields":any("seal" in f or "fund" in f or "bomb" in f for f in fields),"sha256":sha(p)})

snapshot = {"formed_at":datetime.now(timezone.utc).isoformat(),"scope":"Existing bounded V275,V287,V49 recovery and saved qualifications only; no new sources/fits/outcome-based field selection","rows":len(rows),"dates":rows.date.nunique(),"relay_rows":len(relay),"valid_all_clock_rows":int(valid.sum()),"valid_relay_clock_rows":int(valid[relay.index].sum()),"known_relay_labels":int(relay.promoted_next_day.notna().sum()),"unknown_relay_labels":int(relay.promoted_next_day.isna().sum()),"relay_nonmissing_fields":{f:int(relay[f].notna().sum()) for f in ["first_seal","last_seal","bomb_count","sealed_fund","float_cap","amount"]},"relay_sealed_fund_zero":int(relay.sealed_fund.eq(0).sum()),"relay_sealed_fund_negative":int(relay.sealed_fund.lt(0).sum()),"relay_bomb_zero":int(relay.bomb_count.eq(0).sum()),"relay_bomb_positive":int(relay.bomb_count.gt(0).sum()),"relay_first_last_equal":int(relay.reported_first_seconds.eq(relay.reported_last_seconds).sum()),"relay_auction_clock_proxy":int(relay.auction_sealed_clock_proxy.eq(1).sum()),"relay_original_manifest_hash_rows":sum(r["relay_rows"] for r in day_audit if r["original_manifest_sha_present"]),"relay_naive_capture_rows":sum(r["relay_rows"] for r in day_audit if not r["capture_has_timezone"]),"original_29_training_overlap":overlap,"original_placeholder_counts":placeholder_counts,"available_contiguous_blocks":block_summary,"sampled_price_qualification":sam_counts,"v49_recovery":{"returned_case_files":len(v49cases),"unique_requested_dates":sorted(set(r["requested_date"] for r in v49cases)),"all48":sum(r["rows"]==48 for r in v49cases),"wrong_requested_date":sum(not r["date_exact"] for r in v49cases),"with_seal_fields":sum(r["contains_seal_or_orderbook_fields"] for r in v49cases),"receipt_not_daily_reconciliation":True},"raw_source_field_mismatches":raw_field_mismatches,"no_fit":True,"independent_unseen_event_window_found":False,"source_sha256":{str(archive):sha(archive),str(archive.parent/"report.json"):sha(archive.parent/"report.json"),str(prepared_path):sha(prepared_path),str(bank_path):sha(bank_path),str(Path(__file__).resolve()):sha(__file__)}}
write(OUT / "source_audit_summary.json", snapshot)
write(OUT / "full18_day_capture_audit.json", day_audit)
write(OUT / "raw_source_hash_checks.json", used_source_audit)
write(OUT / "raw_minute_hash_checks.json", raw_minute_hash_checks)
write(OUT / "bounded_v49_case_inventory.json", v49cases)
print(json.dumps(snapshot, ensure_ascii=False, indent=2))
