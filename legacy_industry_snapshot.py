"""Immutable 2021-vintage grouping for retrospective research, not current industry."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
BANK=ROOT/"research/first_board_github_industry_qualification_20261020"
STATUS="QUALIFIED_LEGACY_STATIC_PROXY_NOT_HISTORICAL_CURRENT_INDUSTRY"


def digest(file):return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def validate_metadata(metadata,head,blob_identity):
    # These are separate pieces of evidence. Git author dates are not used as
    # publication times. Host push metadata only supports an old content bound.
    if metadata.get("private") is not False:raise ValueError("Source is not currently public")
    if metadata.get("fork") is not False:raise ValueError("Unexpected source repository identity")
    for field in ("created_at","pushed_at","updated_at"):
        if not metadata.get(field) or pd.Timestamp(metadata[field])>=pd.Timestamp("2023-11-28T00:00:00Z"):
            raise ValueError("Repository host metadata no longer supports the fixed pre-2023 vintage")
    if head["sha"]!=blob_identity["commit_sha"]:raise ValueError("Pinned commit identity mismatch")
    if metadata["full_name"]!=blob_identity["repo"]:raise ValueError("Repository identity mismatch")


def normalize_snapshot(raw,master):
    table=raw[["ts_code","symbol","name","industry","list_date"]].copy()
    if table.symbol.duplicated().any():raise ValueError("Duplicate stock symbol")
    if not table.symbol.str.fullmatch(r"\d{6}").all():raise ValueError("Invalid stock symbol")
    if not table.ts_code.str.fullmatch(r"\d{6}\.(SH|SZ)").all():raise ValueError("Invalid exchange-qualified code")
    if not table.ts_code.str.split(".").str[0].eq(table.symbol).all():raise ValueError("Exchange code/symbol mismatch")
    if not table.list_date.str.fullmatch(r"\d{8}").all():raise ValueError("Invalid source listing date")
    if not table.list_date.le("20210305").all():raise ValueError("Listing date newer than pinned content vintage")
    project=master[["code","listing_date"]].drop_duplicates()
    if project.code.duplicated().any():raise ValueError("Project listing identity is ambiguous")
    table=table.rename(columns={"symbol":"code","name":"vintage_name_SOURCE_ONLY","industry":"legacy_group_RAW"})
    table["source_listing_date"]=pd.to_datetime(table.list_date,format="%Y%m%d").dt.strftime("%Y-%m-%d")
    table=table.merge(project.rename(columns={"listing_date":"project_listing_date"}),on="code",how="left",validate="one_to_one")
    inside=table.project_listing_date.notna()
    same=table.source_listing_date.eq(table.project_listing_date)
    table["identity_status"]="OUTSIDE_PROJECT_SCOPE"
    table.loc[inside&same,"identity_status"]="LISTING_DATE_MATCH"
    table.loc[inside&~same,"identity_status"]="LISTING_DATE_CONFLICT_UNKNOWN"
    group=table.legacy_group_RAW.str.strip().replace("",pd.NA)
    table["legacy_group_PROXY_ALLOWED"]=group.where(inside&same)
    table["snapshot_semantics"]="STATIC_2021_VINTAGE_CATEGORY;NOT_SIGNAL_DATE_CURRENT_INDUSTRY"
    table["strict_original_publication_verified"]=False
    return table


def load_snapshot(bank=BANK):
    bank=Path(bank)
    manifest=json.loads((bank/"qualified_legacy_snapshot.json").read_text("utf-8"))
    if manifest["status"]!=STATUS:raise ValueError("Snapshot is not qualified for this legacy-proxy policy")
    if manifest["usable_as_historical_current_industry"] or manifest["strict_original_publication_verified"]:
        raise ValueError("Legacy classification cannot be promoted into strict current-industry truth")
    for path,expected in manifest["bound_inputs"].items():
        if digest(path)!=expected:raise ValueError("Bound source hash mismatch")
    mapping=bank/manifest["mapping_file"]
    if digest(mapping)!=manifest["mapping_sha256"]:raise ValueError("Normalized snapshot hash mismatch")
    table=pd.read_csv(mapping,dtype=str,encoding="utf-8-sig")
    if table.code.duplicated().any():raise ValueError("Duplicate normalized stock code")
    return table,manifest


def attach_legacy_groups(rows,bank=BANK):
    """Keep every original row. No target/outcome field is read or changed."""
    table,manifest=load_snapshot(bank)
    if not rows.signal_date.ge(manifest["research_minimum_signal_date"]).all():
        raise ValueError("Signal outside this source-qualified research domain")
    out=rows.copy()
    mapping=table.set_index("code").legacy_group_PROXY_ALLOWED
    out["legacy_industry_group_PROXY"]=out.code.map(mapping)
    out["legacy_industry_source_status"]="NOT_IN_2021_SNAPSHOT_OR_UNRESOLVED_IDENTITY"
    out.loc[out.legacy_industry_group_PROXY.notna(),"legacy_industry_source_status"]="STATIC_2021_VINTAGE_PROXY"
    return out


def build_qualification():
    output=BANK/"qualified_legacy_snapshot.json"
    if output.exists():raise FileExistsError("Preserve existing qualification")
    raw_file=BANK/"stock_basics_industry_original.csv"
    identity=json.loads((BANK/"stock_basics_blob_identity.json").read_text("utf-8"))
    raw=raw_file.read_bytes()
    git_hash=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
    if git_hash!=identity["git_blob_sha1"] or digest(raw_file)!=identity["sha256"]:
        raise ValueError("Source data identity mismatch")
    metadata=json.loads((BANK/"stock_basics_metadata.response").read_text("utf-8"))
    head=json.loads((BANK/"stock_basics_head_commit.response").read_text("utf-8"))
    validate_metadata(metadata,head,identity)
    master_file=ROOT/"data/v11/security_master_intervals.csv"
    normalized=normalize_snapshot(pd.read_csv(raw_file,dtype=str),pd.read_csv(master_file,dtype=str))
    normalized["source_commit_sha"]=identity["commit_sha"]
    normalized["source_git_blob_sha1"]=identity["git_blob_sha1"]
    normalized["host_last_push_at"]=metadata["pushed_at"]
    normalized["actually_acquired_at"]=json.loads((BANK/"stock_basics_industry_blob_receipt.json").read_text("utf-8"))["received_at"]
    mapping=BANK/"legacy_2021_industry_mapping_WITH_UNKNOWNS.csv"
    normalized.to_csv(mapping,index=False,encoding="utf-8-sig")
    source=ROOT/"research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
    keys=pd.read_csv(source,usecols=["signal_date","code","stage"],dtype={"code":str})
    keys=keys[keys.stage.eq(1)].copy()
    groups=keys.code.map(normalized.set_index("code").legacy_group_PROXY_ALLOWED)
    coverage={}
    for name,mask in [("all",pd.Series(True,index=keys.index)),("development",keys.signal_date.lt("2025-01-01")),
                      ("validation2025H1",keys.signal_date.between("2025-01-01","2025-06-30")),
                      ("evaluation2025H2",keys.signal_date.between("2025-07-01","2025-12-31")),("evaluation2026",keys.signal_date.ge("2026-01-01"))]:
        coverage[name]=dict(rows=int(mask.sum()),known_legacy_groups=int((mask&groups.notna()).sum()),unknown=int((mask&groups.isna()).sum()),coverage=float(groups[mask].notna().mean()))
    bindings=[raw_file,master_file,source,BANK/"stock_basics_blob_identity.json",BANK/"stock_basics_metadata.response",
        BANK/"stock_basics_head_commit.response",BANK/"stock_basics_industry_blob.response",BANK/"stock_basics_metadata_receipt.json",
        BANK/"stock_basics_head_commit_receipt.json",BANK/"stock_basics_industry_blob_receipt.json",BANK/"qualification_protocol_BEFORE_DATA.json"]
    manifest=dict(status=STATUS,formed_at=pd.Timestamp.now(tz="UTC").isoformat(),source_repo=metadata["full_name"],
        commit_sha=identity["commit_sha"],git_blob_sha1=identity["git_blob_sha1"],source_sha256=identity["sha256"],
        source_rows=len(normalized),source_groups=int(normalized.legacy_group_RAW.nunique()),
        source_currently_public=True,repo_created_at=metadata["created_at"],host_last_push_at=metadata["pushed_at"],host_last_object_update_at=metadata["updated_at"],
        git_author_date_is_not_publication=True,first_publication_at=None,strict_original_publication_verified=False,
        content_vintage_inference="Pinned current tree and host last-push2021 metadata support old content;exact first-public visibility and independent original capture absent",
        qualification_policy="Retrospective fixed legacy-2021 grouping only,based on old host-content evidence;not a certification of historical current industry or strict original acquisition",
        usable_as_immutable_legacy_proxy=True,usable_as_historical_current_industry=False,independent_industry_gold_available=False,
        research_minimum_signal_date="2023-11-28",acquisition_backdated=False,acquired_at=normalized.actually_acquired_at.iloc[0],
        identity_conflict_codes=normalized.loc[normalized.identity_status.eq("LISTING_DATE_CONFLICT_UNKNOWN"),"code"].tolist(),
        unknowns_preserved=True,no_2018_fallback_or_scheme_blending=True,secondary_source_used_for_training=False,
        mapping_file=mapping.name,mapping_sha256=digest(mapping),coverage_without_outcome_columns=coverage,
        bound_inputs={str(p.resolve()):digest(p) for p in bindings},models_fitted=0,production_changes=False,goal_60_passed=False)
    output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return manifest


if __name__=="__main__":
    result=build_qualification()
    print(json.dumps({k:result[k] for k in ["status","source_rows","source_groups","identity_conflict_codes","coverage_without_outcome_columns"]},ensure_ascii=True,indent=2))
