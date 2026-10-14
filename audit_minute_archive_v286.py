"""Check raw minute payloads before parser deduplication or evidence use."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd
import archive_signal_minutes_v286 as archive
import public_minute_capture_v1 as minute
from next_session_expectation_contract_v271 import load_registered, now


def run(directory, registered_path, output):
    directory, target = Path(directory), Path(output)
    if target.exists():
        raise FileExistsError("refusing to overwrite minute audit")
    report = json.loads((directory/"report.json").read_text(encoding="utf-8"))
    frozen = load_registered(registered_path)
    if report["registration_hash"] != frozen["sha256"]:
        raise ValueError("minute archive refers to another frozen forecast")
    if (report["signal_date"] != frozen["plan"]["signal_date"]
            or report["target_date"] != frozen["plan"]["target_date"]):
        raise ValueError("minute archive signal or target date mismatch")
    expected = {r["code"] for r in frozen["plan"]["candidates"]}
    if report["candidate_denominator"] != len(expected):
        raise ValueError("candidate denominator mismatch")
    for path, wanted in report["source_hashes"].items():
        if archive.sha(path)!=wanted:
            raise ValueError("minute archive source hash changed")
    for path, wanted in report["artifact_hashes"].items():
        if archive.sha(directory/path)!=wanted:
            raise ValueError("minute archive artifact hash changed")
    raw_checks, frames = [], []
    for symbol, source in report["responses"].items():
        if archive.sha(source["raw_path"]) != source["sha256"]:
            raise ValueError("raw provider response hash mismatch")
        text = Path(source["raw_path"]).read_text(encoding="utf-8")
        raw = minute._extract_json_array(text)
        timestamps = pd.Series([r.get("day") for r in raw],dtype="object")
        duplicate_count = int(timestamps.duplicated().sum())
        invalid_count = int(pd.to_datetime(timestamps,errors="coerce").isna().sum())
        raw_checks.append(dict(code=symbol[2:],raw_rows=len(raw),duplicates_before_parser=duplicate_count,
            invalid_raw_timestamps=invalid_count,http_status=source["http_status"],
            raw_source_available_at=source["available_at"]))
        if duplicate_count or invalid_count or source["http_status"] != 200:
            raise ValueError("invalid raw timestamps, duplicates or failed HTTP response")
        frames.append(minute.parse_minute_response(text,symbol[2:]))
    if {r["code"] for r in raw_checks} != expected or report["errors"]:
        raise ValueError("not every frozen candidate has a valid raw response")
    reconstructed = pd.concat(frames,ignore_index=True).sort_values(["code","minute"]).reset_index(drop=True)
    saved = pd.read_csv(directory/"minute_bars_UNRECONCILED.csv",dtype={"code":str}).sort_values(["code","minute"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(reconstructed,saved,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-8)
    if len(saved)!=report["parsed_rows"] or saved.duplicated(["code","minute"]).any():
        raise ValueError("saved minute row count or identity mismatch")
    checks = report["candidate_checks"]
    if len(checks)!=len(expected) or {r["code"] for r in checks}!=expected:
        raise ValueError("candidate checks omit or duplicate a frozen candidate")
    daily_paths = [Path(p) for p in report["source_hashes"] if Path(p).suffix == ".csv"]
    if len(daily_paths) != 1:
        raise ValueError("one hash-bound daily reference required")
    daily = pd.read_csv(daily_paths[0])
    expected_checks = [archive.reconcile(reconstructed[reconstructed.code.eq(row["code"])],
        daily,frozen["plan"]["signal_date"],row["code"]) for row in frozen["plan"]["candidates"]]
    if checks != expected_checks:
        raise ValueError("minute evidence claims differ from reconstructed source facts")
    result=dict(version="v286-raw-minute-integrity-audit-1",recorded_at=now().isoformat(),
        status="RAW_PARSE_AND_DENOMINATOR_VERIFIED",candidate_denominator=len(expected),
        parsed_rows=len(saved),raw_checks=raw_checks,
        price_agreeing_candidates=sum(len(r.get("daily_ohlc_agreement",{}))==4 and
                                      all(r["daily_ohlc_agreement"].values()) for r in checks),
        minute_grid_complete_candidates=sum(r.get("minute_grid_complete") is True for r in checks),
        volume_reconciled_candidates=sum(r.get("volume_agrees_at_inferred_scale_100") is True for r in checks),
        volume_mismatch_codes=[r["code"] for r in checks if r.get("volume_agrees_at_inferred_scale_100") is not True],
        no_missing_bars_imputed=True,original_forecast_rewritten=False,
        accuracy_claim_allowed=False,trade_authorized=False,
        source_hashes={str(p.resolve()):archive.sha(p) for p in
            (directory/"report.json",Path(registered_path),Path(__file__),Path(archive.__file__),Path(minute.__file__))})
    target.mkdir(parents=True,exist_ok=False)
    (target/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return {k:result[k] for k in ("status","candidate_denominator","parsed_rows","minute_grid_complete_candidates",
                                "volume_reconciled_candidates","volume_mismatch_codes","accuracy_claim_allowed")}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("archive","registered","output"):
        parser.add_argument("--"+key,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.archive,args.registered,args.output),ensure_ascii=True))
