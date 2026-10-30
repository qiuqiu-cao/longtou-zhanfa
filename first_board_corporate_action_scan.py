"""Offline provenance audit of XD/DR/XR prefixes in dated first-board sources.

No outcome labels, model fitting, network calls or production data writes.
Prefix absence never certifies absence of a corporate action.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import pandas as pd

DECODER=json.JSONDecoder()
WS=re.compile(r"\s*")

def skip(text,pos):return WS.match(text,pos).end()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()

def member_start(text,key):
    pos=skip(text,0)
    if text[pos]!="{":raise ValueError("Expected source object")
    pos+=1
    while True:
        pos=skip(text,pos)
        if text[pos]=="}":raise KeyError(key)
        candidate,end=DECODER.raw_decode(text,pos)
        pos=skip(text,end)
        if text[pos]!=":":raise ValueError("Expected member colon")
        start=skip(text,pos+1)
        value,end=DECODER.raw_decode(text,start)
        if candidate==key:return start
        pos=skip(text,end)
        if text[pos]=="}":raise KeyError(key)
        if text[pos]!=",":raise ValueError("Expected member separator")
        pos+=1

def array_spans(text,start):
    pos=skip(text,start)
    if text[pos]!="[":raise ValueError("Expected source array")
    pos+=1
    index=0
    while True:
        pos=skip(text,pos)
        if text[pos]=="]":return
        value,end=DECODER.raw_decode(text,pos)
        yield index,pos,end,value
        index+=1
        pos=skip(text,end)
        if text[pos]=="]":return
        if text[pos]!=",":raise ValueError("Expected array separator")
        pos+=1

def run(output):
    output=Path(output)
    protocol=json.loads((output/"JOINT_FROZEN_PROTOCOL.json").read_text("utf-8"))
    if (output/"report.json").exists():raise FileExistsError("No repeated completed audit")
    assert protocol["model_fits_allowed"]==0 and protocol["external_requests_allowed"]==0
    amendment_path=output/"TECHNICAL_CLOCK_AMENDMENT_BEFORE_RESUME.json"
    if amendment_path.exists():
        amendment=json.loads(amendment_path.read_text("utf-8"))
        assert amendment["before_source_sha256"]==protocol["scan_source_sha256"]
        assert amendment["after_source_sha256"]==sha(__file__)
        assert amendment["failure_receipt_sha256"]==sha(output/"SCAN_FAILURE_RECEIPT.json")
        assert amendment["model_fits"]==0 and amendment["source_or_selection_changes"] is False
    else:
        assert sha(__file__)==protocol["scan_source_sha256"]
    receipt_path=Path(protocol["source_receipts"])
    assert sha(receipt_path)==protocol["source_receipts_sha256"]
    receipts=json.loads(receipt_path.read_text("utf-8"))
    assert len(receipts)==668 and len({r["date"] for r in receipts})==668
    all_rows=[]
    date_rows=[]
    try:
        for receipt in sorted(receipts,key=lambda r:r["date"]):
            if datetime.now(timezone.utc)>=datetime.fromisoformat(protocol["deadline_utc"]):
                raise TimeoutError("Fixed audit budget exceeded; no automatic retry")
            path=Path(receipt["source_path"])
            raw=path.read_bytes()
            assert hashlib.sha256(raw).hexdigest()==receipt["sha256"],str(path)
            assert len(raw)==receipt["bytes"]
            text=raw.decode("utf-8")
            payload=json.loads(text)
            assert payload["info"][1]==receipt["date"]==receipt["response_date"]
            assert str(payload["errcode"])=="0"
            assert len(payload["info"][0])==receipt["records"]
            info_start=member_start(text,"info")
            records_start=next(array_spans(text,info_start))[1]
            matched=0
            missing=0
            for index,start,end,record in array_spans(text,records_start):
                assert record==payload["info"][0][index]
                assert len(record)>21 and str(record[15])=="1"
                value_spans=array_spans(text,start)
                next(value_spans)
                _,name_start,name_end,name=next(value_spans)
                assert name==record[1]
                byte_start=len(text[:name_start].encode("utf-8"))
                byte_end=len(text[:name_end].encode("utf-8"))
                assert json.loads(raw[byte_start:byte_end])==name
                normalized=str(name).strip().upper() if name is not None else ""
                match=re.match(r"^(XD|XR|DR)",normalized)
                prefix=match.group(1) if match else ""
                matched+=bool(prefix)
                missing+=not bool(normalized)
                all_rows.append(dict(source_date=receipt["date"],code=str(record[0]).zfill(6),
                    raw_name=name,matched_prefix=prefix,corporate_action_status="CANDIDATE_ONLY" if prefix else "NOT_ESTABLISHED",
                    source_path=str(path),source_sha256=receipt["sha256"],name_pointer=f"/info/0/{index}/1",
                    name_byte_start=byte_start,name_byte_end_exclusive=byte_end,
                    source_received_at=receipt.get("received_at"),
                    source_receive_clock_known=bool(receipt.get("received_at")),official_action_verified=False,
                    absent_prefix_means_no_action=False,model_feature_qualified=False))
            date_rows.append(dict(source_date=receipt["date"],source_rows=len(payload["info"][0]),
                prefix_candidates=matched,empty_names=missing,source_sha256=receipt["sha256"],
                source_receive_clock_known=bool(receipt.get("received_at"))))
        rows=pd.DataFrame(all_rows)
        assert len(rows)==35292
        candidates=rows[rows.matched_prefix.ne("")].copy()
        rows.to_csv(output/"ALL35292_SOURCE_NAMES_AND_PROVENANCE.csv",index=False,encoding="utf-8-sig")
        candidates.to_csv(output/"XD_DR_XR_CANDIDATE_ACTION_LEDGER.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(date_rows).to_csv(output/"COMPLETE668_SOURCE_DATE_COVERAGE.csv",index=False,encoding="utf-8-sig")
        assert sha(receipt_path)==protocol["source_receipts_sha256"]
        report=dict(at=now(),status="COMPLETE668_OFFLINE_PREFIX_QUALIFICATION_NO_FITS",source_dates=668,
            source_rows=len(rows),candidate_rows=len(candidates),candidate_dates=candidates.source_date.nunique(),
            unique_candidate_codes=candidates.code.nunique(),prefix_counts=candidates.matched_prefix.value_counts().to_dict(),
            empty_names=int(rows.raw_name.isna().sum()+rows.raw_name.eq("").sum()),
            duplicate_date_code_rows=int(rows.duplicated(["source_date","code"],keep=False).sum()),
            all_source_hashes_and_date_envelopes=True,all_name_byte_spans_verified=True,
            missing_source_dates=0,source_scope="First-board source only;not all securities or all corporate actions",
            source_dates_without_receive_clock=sum(not bool(r.get("received_at")) for r in receipts),
            source_rows_without_receive_clock=int((~rows.source_receive_clock_known).sum()),
            source_actual_capture_is_late=True,absence_is_unknown=True,official_actions_verified=0,
            target_labels_read=False,model_fits=0,external_requests=0,production_writes=0,
            original_targets_rewritten=0,goal_60_achieved=False,
            artifacts_sha256={name:sha(output/name) for name in ["ALL35292_SOURCE_NAMES_AND_PROVENANCE.csv",
                "XD_DR_XR_CANDIDATE_ACTION_LEDGER.csv","COMPLETE668_SOURCE_DATE_COVERAGE.csv"]})
        with (output/"report.json").open("x",encoding="utf-8") as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2)
        print(json.dumps(report,ensure_ascii=True))
    except Exception as exc:
        failure_path=output/"SCAN_FAILURE_RECEIPT.json"
        if failure_path.exists():
            failure_path=output/datetime.now(timezone.utc).strftime("SCAN_FAILURE_RECEIPT_%Y%m%dT%H%M%S%fZ.json")
        with failure_path.open("x",encoding="utf-8") as stream:
            json.dump(dict(at=now(),error=repr(exc),completed_source_dates=len(date_rows),
                partial_source_rows=len(all_rows),model_fits=0,external_requests=0,no_automatic_retry=True),stream,indent=2)
        raise

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    run(args.output)
