"""Independently audit captured bytes, code identities and full-cohort mapping."""
from pathlib import Path
import csv
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import pandas as pd
import legacy_industry_snapshot as legacy

OUT=Path(__file__).resolve().parent
table,manifest=legacy.load_snapshot()
assert manifest["strict_original_publication_verified"] is False
assert manifest["first_publication_at"] is None
assert manifest["usable_as_historical_current_industry"] is False
assert manifest["usable_as_immutable_legacy_proxy"] is True
for key in ("stock_basics","knowledge_graph"):
    proof=json.loads((OUT/(key+"_blob_identity.json")).read_text("utf-8"))
    raw=(OUT/proof["decoded_file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==proof["sha256"]
    assert hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()==proof["git_blob_sha1"]
    reply=json.loads((OUT/(key+"_industry_blob.response")).read_text("utf-8"))
    assert reply["sha"]==proof["git_blob_sha1"]
    assert proof["commit_sha"] in reply["html_url"]
    metadata=json.loads((OUT/(key+"_metadata.response")).read_text("utf-8"))
    assert metadata["private"] is False
    assert pd.Timestamp(metadata["pushed_at"])<pd.Timestamp("2023-11-28T00:00:00Z")

with (legacy.ROOT/"data/v11/security_master_intervals.csv").open(encoding="utf-8-sig",newline="") as f:
    project={}
    for row in csv.DictReader(f):
        if row["code"] in project:assert project[row["code"]]==row["listing_date"]
        project[row["code"]]=row["listing_date"]
with (OUT/"stock_basics_industry_original.csv").open(encoding="utf-8-sig",newline="") as f:
    source=list(csv.DictReader(f))
expected={};conflicts=[]
for row in source:
    code=row["symbol"]
    assert code not in expected and len(code)==6 and code.isdigit()
    assert row["ts_code"].split(".")[0]==code
    date=row["list_date"]
    assert date<="20210305" and len(date)==8 and date.isdigit()
    listing=date[:4]+"-"+date[4:6]+"-"+date[6:]
    known=code in project and project[code]==listing
    if code in project and not known:conflicts.append(code)
    expected[code]=(row["industry"].strip() or None) if known else None
assert len(source)==4179
assert sorted(conflicts)==sorted(manifest["identity_conflict_codes"])==["000004","600018","600190"]
normalized=table.set_index("code")
for code,group in expected.items():
    value=normalized.loc[code,"legacy_group_PROXY_ALLOWED"]
    assert (pd.isna(value) and group is None) or value==group

source_file=legacy.ROOT/"research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
keys=pd.read_csv(source_file,usecols=["signal_date","code","stage"],dtype={"code":str})
keys=keys[keys.stage.eq(1)].copy()
assert len(keys)==28503
attached=legacy.attach_legacy_groups(keys)
assert len(attached)==len(keys)
pd.testing.assert_frame_equal(attached[keys.columns],keys)
for row in attached.itertuples():
    desired=expected.get(row.code)
    actual=row.legacy_industry_group_PROXY
    assert (pd.isna(actual) and desired is None) or actual==desired
for cutoff in ("2025-06-30","2025-12-31"):
    prefix=legacy.attach_legacy_groups(keys[keys.signal_date.le(cutoff)])
    pd.testing.assert_frame_equal(prefix,attached[attached.signal_date.le(cutoff)])
attached.to_csv(OUT/"COMPLETE28503_LEGACY_GROUP_KEYS_NO_OUTCOME_COLUMNS.csv",index=False,encoding="utf-8-sig")
assert not any("LABEL_ONLY" in c or "score" in c for c in attached)
assert int(attached.legacy_industry_group_PROXY.notna().sum())==25808
assert int(attached.legacy_industry_group_PROXY.isna().sum())==2695
assert attached.signal_date.min()>"2023-11-28"
assert manifest["acquired_at"].startswith("2026-10-20")

# Existing accepted/rejected model sources and original ledgers stay unchanged.
old=legacy.ROOT/"research/first_board_feasible_rank_20261020_run1/protocol_BEFORE_ANY_FIT.json"
old_protocol=json.loads(old.read_text("utf-8"))
for path,digest in old_protocol["protected"].items():assert legacy.digest(path)==digest
for receipt in (old.parent/"models").glob("*_COMPLETED.json"):
    data=json.loads(receipt.read_text("utf-8"))
    model=receipt.with_name(receipt.name.replace("_COMPLETED.json",".joblib"))
    assert legacy.digest(model)==data["sha256"]
result=dict(status="PASS_LEGACY_PROXY_ONLY",verified_at=pd.Timestamp.now(tz="UTC").isoformat(),
    static_data_blobs=2,repositories=2,primary_source_rows=4179,full_first_board_keys=28503,
    independently_mapped_rows=25808,unknown_rows=2695,identity_conflicts=conflicts,
    independent_git_blob_and_sha256_checks=True,all_source_rows_independently_parsed=True,
    past_prefixes_unchanged=True,original_features_scores_models_ledgers_unchanged=True,
    strict_original_publication_verified=False,historical_current_industry_truth=False,
    legacy_static_proxy_qualification=True,outcome_columns_read_for_qualification=False,
    downloaded_repository_code_executed=False,new_fits=0,new_forward_results=0,actual_trades=0,goal_60_passed=False,
    manifest_sha256=legacy.digest(OUT/"qualified_legacy_snapshot.json"),source_module_sha256=legacy.digest(legacy.__file__),verifier_sha256=legacy.digest(__file__))
(OUT/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(result,ensure_ascii=True,indent=2))
