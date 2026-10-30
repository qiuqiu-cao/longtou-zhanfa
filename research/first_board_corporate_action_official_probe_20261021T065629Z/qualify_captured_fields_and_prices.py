"""Bind visually inspected official fields and compare only adjacent source prices."""
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
import json,hashlib
import pandas as pd
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
protocol=json.loads((OUT/"JOINT_FROZEN_PROTOCOL.json").read_text("utf-8"))
assert sha(protocol["original_facts"])==protocol["original_facts_sha256"]
sources=json.loads((OUT/"ALL_THREE_CASE_SOURCE_RECEIPTS.json").read_text("utf-8"))
by_code={r["code"]:r for r in sources["cases"]}
field_notes={
 "600361":dict(status="OFFICIAL_CASH_DIFFERENTIAL_FIELDS_CONFIRMED",announcement_number="2024-029",
    publication_date="2024-06-24",record_date="2024-06-27",ex_date="2024-06-28",
    actual_cash_gross_per_participating_share="0.07",actual_bonus_ratio="0",differential_distribution=True,
    published_virtual_cash_for_reference="0.07",published_virtual_share_change_ratio="0",
    total_shares="4336192838",participating_shares="4107435885",excluded_shares="228756953",
    price_band="0.10",reference_basis="Company explicitly states previous close minus0.07; unrounded virtual fraction is separately diagnostic",
    pdf_evidence_pages=[1,2],visual_pages_checked_by_codex=[1,2]),
 "603203":dict(status="NO_IMPLEMENTATION_IN_ONE_BOUNDED_INDEX_UNKNOWN",official_fields_unknown=True,
    ex_date_candidate_only="2026-06-08",no_dividend_not_inferred=True,source_prefix_not_official_action_truth=True),
 "688368":dict(status="OFFICIAL_COMPOUND_DIFFERENTIAL_FIELDS_CONFIRMED",announcement_number="2026-042",
    publication_date="2026-06-03",record_date="2026-06-08",ex_date="2026-06-09",
    actual_cash_gross_per_participating_share="1.00",actual_bonus_ratio="0.40",differential_distribution=True,
    published_virtual_cash_for_reference="0.9937",published_virtual_share_change_ratio="0.3975",
    total_shares="145714693",participating_shares="144801113",excluded_shares="913580",
    price_band="0.20",reference_basis="Company explicitly states(previous close minus0.9937)/1.3975",
    pdf_evidence_pages=[1,2],visual_pages_checked_by_codex=[1,2],allowed_in_main_board_training_pool=False)
}
for code,note in field_notes.items():
    record=by_code[code]
    if "pdf_receipt" in record:
        receipt=record["pdf_receipt"]
        assert sha(receipt["source_path"])==receipt["sha256"]
        note.update(pdf_url=receipt["url"],pdf_sha256=receipt["sha256"],pdf_captured_at=receipt["received_at"])
        text=(OUT/f"{code}_IMPLEMENTATION_TEXT.txt").read_text("utf-8")
        assert code in text and note["announcement_number"] in text
        assert "差异化" in text
        if code=="600361":assert "0.07" in text and "2024/6/28" in text and "4,107,435,885" in text
        if code=="688368":assert "0.9937" in text and "0.3975" in text and "2026/6/9" in text
    note["bounded_index_is_complete"]=record.get("index_has_more") is False
    note["original_signal_time_capture"]=False
write(OUT/"OFFICIAL_FIELDS_BOUND_BEFORE_PRICE_COMPARISON.json",dict(at=datetime.now(timezone.utc).isoformat(),
    cases=field_notes,source_type="Official company PDFs captured late;visual pages1-2 verified",model_fits=0))
calendar=ROOT/"data/v279/exchange_calendar_audit_20261002_v1/calendar.json"
sessions=json.loads(calendar.read_text("utf-8"))["sessions"]
scope=[]
for case in protocol["cases"]:
    ex=case["source_date"]
    previous=sessions[sessions.index(ex)-1]
    scope.append(dict(code=case["code"],ex_date=ex,previous_official_session=previous))
bar_paths=[ROOT/"data/v16/point_in_time_rebuild_cutoff_20260831_v2"/name for name in
    ["point_in_time_current_daily_bars.csv","verified_recovered_daily_bars.csv"]]
write(OUT/"ADJACENT_PRICE_SOURCE_SCOPE_BEFORE_READ.json",dict(at=datetime.now(timezone.utc).isoformat(),
    cases=scope,calendar_sha256=sha(calendar),sources={str(p):sha(p) for p in bar_paths},
    target_labels_not_read=True,only_pre_ex_and_ex_prices=True,model_fits=0))
parts=[]
codes={case["code"] for case in scope}
dates={case[key] for case in scope for key in ["ex_date","previous_official_session"]}
for path in bar_paths:
    for chunk in pd.read_csv(path,usecols=["date","code","open","high","low","close","pre_close","source"],
        dtype={"code":str},float_precision="round_trip",chunksize=200000):
        take=chunk[chunk.code.isin(codes)&chunk.date.isin(dates)].copy()
        take["source_file"]=str(path)
        parts.append(take)
prices=pd.concat(parts,ignore_index=True)
prices.to_csv(OUT/"ADJACENT_PRE_EX_AND_EX_PRICES_NO_TARGETS.csv",index=False,encoding="utf-8-sig")
comparisons=[]
for case in scope:
    code=case["code"]
    note=field_notes[code]
    previous=prices[prices.code.eq(code)&prices.date.eq(case["previous_official_session"])].drop_duplicates(["date","code","close"])
    current=prices[prices.code.eq(code)&prices.date.eq(case["ex_date"])].drop_duplicates(["date","code","close"])
    row=dict(case,official_fields_verified="published_virtual_cash_for_reference" in note,
        previous_price_rows=len(previous),current_price_rows=len(current),label_error_claimed=False)
    if len(previous)==1 and len(current)==1 and row["official_fields_verified"]:
        prior=Decimal(str(previous.iloc[0].close));close=Decimal(str(current.iloc[0].close))
        cash=Decimal(note["published_virtual_cash_for_reference"])
        expansion=Decimal(note["published_virtual_share_change_ratio"])
        band=Decimal(note["price_band"])
        reference=(prior-cash)/(1+expansion)
        cap=(reference*(1+band)).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP)
        nominal_cap=(prior*(1+band)).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP)
        unrounded_cash=Decimal(note["participating_shares"])*Decimal(note["actual_cash_gross_per_participating_share"])/Decimal(note["total_shares"])
        row.update(previous_raw_close=str(prior),ex_raw_close=str(close),
            stored_synthetic_pre_close=str(current.iloc[0].pre_close),
            announcement_derived_reference=str(reference),announcement_formula_cap=str(cap),
            nominal_shifted_close_cap=str(nominal_cap),close_matches_announced_cap=close==cap,
            close_matches_nominal_cap=close==nominal_cap,unrounded_virtual_cash_diagnostic=str(unrounded_cash),
            derived_reference_is_independent_exchange_quote=False,
            asof_price_adjustment_factor=str(reference/prior))
    else:row["price_or_official_field_status"]="UNAVAILABLE_OR_UNVERIFIED_NO_INFERENCE"
    comparisons.append(row)
write(OUT/"THREE_CASE_FIELD_AND_PRICE_QUALIFICATION.json",dict(at=datetime.now(timezone.utc).isoformat(),
    cases=comparisons,pdfs_captured=2,official_field_cases_confirmed=2,case_without_implementation=1,
    differential_distribution_cases=2,ordinary_nondifferential_assumption_rejected=True,
    new_target_corrections=0,model_fits=0,complete_action_database=False,full28503_repaired=False))
assert sha(protocol["original_facts"])==protocol["original_facts_sha256"]
print(json.dumps(comparisons,ensure_ascii=True))
