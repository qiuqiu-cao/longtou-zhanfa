"""Independently reconstruct the complete September five-minute archive."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

import archive_september_5min_v287 as engine
import public_minute_capture_v1 as minute
from next_session_expectation_contract_v271 import now, load_registered


def run(directory,cohort_path,daily_path,output):
    directory,target=Path(directory),Path(output)
    if target.exists():raise FileExistsError("refusing to overwrite five-minute audit")
    report=json.loads((directory/"report.json").read_text(encoding="utf-8"))
    expected_sources={str(Path(p).resolve()) for p in (cohort_path,daily_path,engine.__file__,minute.__file__)}
    if set(report["source_hashes"])!=expected_sources:
        raise ValueError("archive source scope changed")
    for path,wanted in report["source_hashes"].items():
        if engine.sha(path)!=wanted:raise ValueError("archive source hash mismatch")
    for path,wanted in report["artifact_hashes"].items():
        resolved=(directory/path).resolve()
        if not resolved.is_relative_to(directory.resolve()) or engine.sha(resolved)!=wanted:
            raise ValueError("archive artifact hash mismatch")
    cohort=pd.read_csv(cohort_path,dtype={"code":str},low_memory=False)
    expected_codes=set(cohort.code)
    audited_codes=[row["code"] for row in report["audit"]]
    if len(audited_codes)!=len(set(audited_codes)) or set(audited_codes)!=expected_codes:
        raise ValueError("source responses omit or duplicate cohort symbols")
    frames=[];raw_count=0
    for i,row in enumerate(report["audit"]):
        if engine.sha(row["raw_path"])!=row["raw_sha256"]:raise ValueError("raw response hash mismatch")
        text=Path(row["raw_path"]).read_text(encoding="utf-8")
        raw=minute._extract_json_array(text)
        stamps=pd.Series([r.get("day") for r in raw],dtype="object")
        if (row.get("status")!="RAW_PARSED" or row.get("http_status")!=200 or
                stamps.duplicated().any() or pd.to_datetime(stamps,errors="coerce").isna().any()):
            raise ValueError("unavailable or invalid raw minute source")
        frame=minute.parse_minute_response(text,row["code"])
        if len(raw)!=row["raw_rows"] or len(frame)!=row["parsed_rows"] or not len(frame):
            raise ValueError("raw and parsed row counts differ")
        frames.append(frame);raw_count+=len(raw)
        if (i+1)%40==0:print(json.dumps(dict(verified_symbols=i+1,total_symbols=len(audited_codes))),flush=True)
    reconstructed=pd.concat(frames,ignore_index=True).sort_values(["code","minute"]).reset_index(drop=True)
    saved=pd.read_csv(directory/"five_minute_bars_UNRECONCILED.csv",dtype={"code":str})
    saved=saved.sort_values(["code","minute"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(saved,reconstructed,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-8)
    if reconstructed.duplicated(["code","minute"]).any():raise ValueError("duplicate archived minute")
    daily=pd.read_csv(daily_path)
    groups={code:rows for code,rows in reconstructed.groupby("code")}
    checks=[engine.check_signal(groups[row.code],daily,row.code,row.date) for row in cohort.itertuples()]
    stored=json.loads((directory/"signal_checks.json").read_text(encoding="utf-8"))
    if stored!=checks:raise ValueError("signal evidence differs from reconstructed raw facts")
    actual=dict(candidate_rows=len(cohort),candidate_dates=int(cohort.date.nunique()),requested_symbols=len(expected_codes),
        available_symbols=len(expected_codes),raw_rows=raw_count,interval_minutes=5,
        price_reconciled_rows=sum(r["price_reconciled"] for r in checks),
        full_grid_rows=sum(r["grid_complete"] for r in checks),
        volume_reconciled_rows=sum(r["volume_reconciled"] for r in checks))
    if any(report.get(key)!=value for key,value in actual.items()):raise ValueError("archive aggregate claims mismatch")
    if any(report.get(key) is not False for key in ("outcome_selection_used","original_forecast_rewritten",
                "accuracy_claim_allowed","trade_authorized")):
        raise ValueError("retrospective data cannot authorize prediction or trade claims")
    coverage=[]
    for day,day_rows in cohort.groupby("date",sort=True):
        detail=[r for r in checks if r["date"]==day]
        coverage.append(dict(date=day,candidate_rows=len(detail),
            price_rows=sum(r["price_reconciled"] for r in detail),
            grid_rows=sum(r["grid_complete"] for r in detail),
            volume_rows=sum(r["volume_reconciled"] for r in detail),
            original_next_session_labels_complete=bool(day_rows.label_valid.all())))
    cohort_counts={"all_rows":len(cohort)}
    for prefix in ("00","60"):
        subset=[r for r in checks if r["code"].startswith(prefix)]
        cohort_counts[prefix]=dict(rows=len(subset),volume_matched=sum(r["volume_reconciled"] for r in subset))
    registry=Path(__file__).resolve().parent/"data/v271/expectation_registry/2026-09-30_2c74dc406eef7dcf.json"
    frozen=load_registered(registry)
    result=dict(version="v287-five-minute-raw-reconstruction-audit-1",recorded_at=now().isoformat(),
        status="RAW_PARSE_SIGNALS_AND_COMPLETE_COHORT_VERIFIED",**actual,coverage_by_date=coverage,
        original_label_complete_dates=sum(r["original_next_session_labels_complete"] for r in coverage),
        volume_coverage_by_prefix=cohort_counts,missing_candidates_dropped=False,
        source_interval_upgraded_to_one_minute=False,accuracy_claim_allowed=False,trade_authorized=False,
        frozen_registration_hash=frozen["sha256"],frozen_registry_byte_hash=engine.sha(registry),
        source_hashes={str(Path(p).resolve()):engine.sha(p) for p in
            (directory/"report.json",cohort_path,daily_path,Path(__file__),Path(engine.__file__),Path(minute.__file__),registry)})
    target.mkdir(parents=True,exist_ok=False)
    (target/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return {key:result[key] for key in ("status","candidate_rows","candidate_dates","raw_rows","price_reconciled_rows",
        "full_grid_rows","volume_reconciled_rows","original_label_complete_dates","volume_coverage_by_prefix")}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("archive","cohort","daily","output"):parser.add_argument("--"+key,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.archive,args.cohort,args.daily,args.output),ensure_ascii=True))
