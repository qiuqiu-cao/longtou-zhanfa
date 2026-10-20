"""Prepare future single-signal-date review packets, without an external API."""
from hashlib import sha256
import copy
import json


FORBIDDEN_KEYS={"target","outcome","outcome_for_display","base_score","b_score","score","model_score","original_direction"}


def single_date_packet(source,signal_date,repeat_id):
    days=[x for x in source["days"] if x["signal_date"]==signal_date]
    if len(days)!=1:raise ValueError("exactly one complete signal date is required")
    day=copy.deepcopy(days[0])
    if day["target_date"]<=signal_date:raise ValueError("future official target required")
    codes=[x["code"] for x in day["candidates"]]
    if len(codes)!=len(set(codes)):raise ValueError("duplicate candidate")
    for candidate in day["candidates"]:
        if any(bar["date"]>signal_date for bar in candidate["daily_bars"]):
            raise ValueError("future daily price in as-of packet")
    def check(value):
        if isinstance(value,dict):
            if FORBIDDEN_KEYS & set(value):raise ValueError("score or outcome inside closed evidence")
            for item in value.values():check(item)
        elif isinstance(value,list):
            for item in value:check(item)
    check(day)
    canonical=copy.deepcopy(day)
    canonical["candidates"].sort(key=lambda x:x["code"])
    digest=sha256(json.dumps(canonical,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode("utf-8")).hexdigest()
    if repeat_id not in (1,2,3):raise ValueError("three predeclared reviewers only")
    day["candidates"].sort(key=lambda x:x["code"],reverse=repeat_id==2)
    if repeat_id==3:
        n=len(day["candidates"])//2;day["candidates"]=day["candidates"][n:]+day["candidates"][:n]
    return dict(protocol="345-v1",repeat_id=repeat_id,canonical_facts_sha256=digest,days=[day],
        event_target=source["event_target"],retrospective=source.get("retrospective",True),
        single_signal_date_only=True,inheritance_exposure_must_be_declared=True,
        production_forecast_requires_original_workbench_gates=True,external_API_required=False)
