"""A bounded positive-only stage cross-check; no label edits or model fitting."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = OUT / "raw"
FACTS = ROOT / "research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
PREVIOUS = ROOT / "research/forum_factor_design_20261021T025139Z/COMPLETE28503_LABEL_REPORT_CROSSCHECK.csv"
MANIFEST = ROOT / "research/first_board_joint_design_20261020T185417Z/ALL668_SOURCE_RECEIPTS.json"
def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if (OUT / "COLLECTION_PROTOCOL_BEFORE_REQUESTS.json").exists():
        raise FileExistsError("Preserve attempts; no automatic retries")
    calendar = json.loads((ROOT / "data/v279/exchange_calendar_audit_20261002_v1/calendar.json").read_text("utf-8"))["sessions"]
    comparison = pd.read_csv(PREVIOUS, usecols=["label_end", "target_report_present"])
    # Fixed chronological dates, not chosen to obtain favorable model metrics.
    anchors = ["2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01", "2026-08-31"]
    fixed = [next(day for day in calendar if day >= anchor) for anchor in anchors]
    diagnostic = sorted(comparison.loc[comparison.target_report_present.eq(True), "label_end"].unique())
    days = sorted(set(fixed + diagnostic))
    facts = pd.read_csv(FACTS, usecols=["signal_date", "code", "stage", "label_end"], dtype={"code":str})
    relevant = facts[facts.stage.eq(1) & facts.label_end.isin(days)]
    assert len(days) <= 24 and set(days) <= set(calendar)
    protected = {str(p):sha(p) for p in [FACTS, PREVIOUS, MANIFEST]}
    protocol = dict(at=now(), batch_started_at="2026-10-21T06:29:51+00:00", batch_deadline_utc="2026-10-21T06:59:51+00:00",
        budget_minutes=30, purpose="Actual second-board dated envelopes for bounded source/stage consistency audit;not model tuning",
        fixed_chronological_dates=fixed, diagnostic_dates_from_prior36_firstboard_stage_discrepancies=diagnostic,
        diagnostic_selection_is_outcome_seen_not_validation=True, request_dates=days, request_count=len(days),
        original_pool_rows_in_scope=len(relevant), original_pool_dates_in_scope=relevant.signal_date.nunique(),
        request_parameters=dict(a="DailyLimitPerformance", c="HisHomeDingPan", Type=4, PidType=2, st=1000, Index=0),
        parameter_source="Existing archived public-source documentation history section explicitly labels PidType=2 as second board. No secret/device/token copied.",
        attempts_per_date=1, workers=2, spacing_seconds=.6, timeout_seconds=12, no_auth=True,
        stop="No new requests after403/429 or10minutes;record every failure,never bypass. Do not extend dates based on results.",
        interpretation="Date envelope and exact cents can corroborate a positive report;absent report UNKNOWN,not negative. Provider itself not exchange gold truth.",
        model_fits=0, target_rewrites=0, feature_training_authorized=False, protected_hashes=protected)
    RAW.mkdir(exist_ok=True)
    (OUT / "COLLECTION_PROTOCOL_BEFORE_REQUESTS.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")
    lock = threading.Lock()
    halt = threading.Event()
    earliest = [0.]
    began = time.monotonic()
    local = threading.local()
    def fetch(day):
        if halt.is_set() or time.monotonic()-began > 600:
            return dict(date=day,status="NOT_REQUESTED_AFTER_STOP")
        with lock:
            delay = max(0.,earliest[0]-time.monotonic())
            if delay: time.sleep(delay)
            earliest[0] = time.monotonic()+.6
        if halt.is_set():
            return dict(date=day,status="NOT_REQUESTED_AFTER_ACCESS_DENIAL")
        if not hasattr(local,"session"):
            local.session=requests.Session()
            local.session.headers.update({"User-Agent":"LongtouResearch/1.0 (bounded read-only source verification)"})
        item=dict(date=day,requested_at=now(),attempts=1)
        params=dict(Order=0,st=1000,a="DailyLimitPerformance",c="HisHomeDingPan",PhoneOSNew=1,VerSion="5.7.0.12",Index=0,PidType=2,apiv="w31",Type=4,Day=day)
        try:
            response=local.session.get("https://apphis.longhuvip.com/w1/api/index.php",params=params,timeout=12)
            path=RAW / (day+".response")
            with path.open("xb") as f:f.write(response.content)
            item.update(received_at=now(),http_status=response.status_code,source_path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size)
            if response.status_code in (403,429):halt.set()
            if response.status_code == 200:
                payload=response.json()
                info=payload.get("info")
                if isinstance(info,list) and len(info)>1:
                    rows=info[0] if isinstance(info[0],list) else []
                    item.update(response_date=info[1],errcode=payload.get("errcode"),records=len(rows),
                        qualified_date_envelope=info[1]==day and str(payload.get("errcode"))=="0",
                        raw_field15_values=sorted({str(row[15]) for row in rows if isinstance(row,list) and len(row)>=23}))
        except Exception as exc:
            item.update(completed_at=now(),error_type=type(exc).__name__,error=str(exc)[:220])
        return item
    results=[]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures={executor.submit(fetch,day):day for day in days}
        for future in as_completed(futures):
            result=future.result()
            results.append(result)
            with (OUT / "SOURCE_RECEIPTS_APPEND_ONLY.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps(result,ensure_ascii=False)+"\n")
            print(json.dumps(dict(date=result["date"],http=result.get("http_status"),qualified=result.get("qualified_date_envelope",False),rows=result.get("records"),error=result.get("error_type"))),flush=True)
    results.sort(key=lambda x:x["date"])
    with (OUT / "ALL_LIMITED_SECOND_BOARD_RECEIPTS.json").open("x",encoding="utf-8") as f:
        json.dump(results,f,ensure_ascii=False,indent=2)
    assert protected == {str(p):sha(p) for p in [FACTS,PREVIOUS,MANIFEST]}
    with (OUT / "COLLECTION_COMPLETION.json").open("x",encoding="utf-8") as f:
        json.dump(dict(at=now(),requested_dates=len(days),qualified=sum(bool(x.get("qualified_date_envelope")) for x in results),
            failed_or_unrequested=sum(not bool(x.get("qualified_date_envelope")) for x in results),
            rows=sum(x.get("records",0) for x in results if x.get("qualified_date_envelope")),
            model_fits=0,target_rewrites=0,original_protected_hashes_unchanged=True,access_denial_halt=halt.is_set()),f,indent=2)


if __name__ == "__main__":main()
