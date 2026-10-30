"""Verify one company-announced dividend against archived nominal cap and report."""
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
import json
import hashlib
from datetime import datetime,timezone
import pandas as pd
import fitz
import first_board_leader_model as base
OUT=Path(__file__).resolve().parent
PDF=OUT/"raw/603214_2024H1_DIVIDEND_IMPLEMENTATION.pdf"
receipt=json.loads((OUT/"OFFICIAL_DIVIDEND_PDF_RECEIPT.json").read_text("utf-8"))
assert receipt["http_status"]==200 and receipt["sha256"]==hashlib.sha256(PDF.read_bytes()).hexdigest()
doc=fitz.open(PDF)
text=doc[0].get_text()
assert "603214" in text and "2024-053" in text and "0.109" in text
assert "2024/9/30" in text and "2024/10/8" in text
scope=pd.read_csv(OUT/"ALL_ORIGINAL_FIRST_BOARD_ROWS_ON_LIMITED_TARGET_DATES.csv",dtype={"code":str},float_precision="round_trip")
case=scope[scope.code.eq("603214") & scope.signal_date.eq("2024-09-30")]
assert len(case)==1
row=case.iloc[0]
assert row.original_target_LABEL_ONLY==0 and row.target_second_price_equal and row.signal_first_price_equal
prior=Decimal(str(row.original_signal_close));actual=Decimal(str(row.original_target_close));cash=Decimal("0.109")
raw_cap=(prior*Decimal("1.10")).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP)
adjusted_reference=prior-cash
adjusted_cap=(adjusted_reference*Decimal("1.10")).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP)
assert (str(prior),str(actual),str(raw_cap),str(adjusted_cap))==("15.52","16.95","17.07","16.95")
evidence=dict(verified_at=base.now(),status="ONE_CORPORATE_ACTION_TARGET_ERROR_CONFIRMED_SEPARATE_LAYER",
    signal_date="2024-09-30",code="603214",target_date="2024-10-08",original_target_LABEL_ONLY=0,
    corrected_target_LABEL_ONLY=1,signal_close=str(prior),archived_imported_target_pre_close="15.52",
    announced_cash_per_share=str(cash),announced_record_date="2024-09-30",announced_ex_date="2024-10-08",
    announced_differential_dividend=False,derived_ex_reference=str(adjusted_reference),
    raw_previous_close_10pct_cap=str(raw_cap),cash_adjusted_10pct_cap=str(adjusted_cap),actual_target_close=str(actual),
    source_pdf_url=receipt["url"],source_pdf_sha256=receipt["sha256"],source_pdf_page=1,
    source_pdf_captured_at=receipt["received_at"],company_announcement_number="2024-053",announcement_date="2024-09-24",
    index_source_sha256=receipt["source_index_sha256"],
    target_report_source_sha256=row.second_source_sha256,target_report_pointer=row.second_source_pointer,
    target_report_received_at=row.second_source_received_at,
    interpretation="Company original PDF fixes exact cash and ex-date;10% cent formula now matches archived close and dated supplier stage2,while nominal shifted-close formula does not. Derived ex-reference,not an independently captured exchange quote.",
    visual_page1_checked_by_codex=True,original_data_overwrites=0,original_prediction_rewrites=0,model_fits=0,
    strict_original_download_time=False,retrospective_audit_only=True,all_other_labels_verified=False)
with (OUT/"ONE_CONFIRMED_CORPORATE_ACTION_CORRECTION.json").open("x",encoding="utf-8") as f:
    json.dump(base.clean(evidence),f,ensure_ascii=False,indent=2,allow_nan=False)
with (OUT/"CORPORATE_ACTION_CORRECTIONS_SEPARATE_LAYER.jsonl").open("x",encoding="utf-8") as f:
    f.write(json.dumps(base.clean(evidence),ensure_ascii=False,allow_nan=False)+"\n")
base.write_json(OUT/"PDF_CAPTURE_CONSOLE_ERROR_QUALIFICATION.json",dict(at=base.now(),
    network_request_success=True,pdf_hash_and_three_pages_verified=True,page1_rendered_and_visually_checked=True,
    text_extraction_saved_before_error=True,technical_failure="Original capture console print used GBK and could not encode a PDF bullet;network/rawPDF/extraction/render already completed. No re-fetch or source overwrite.",
    previous_receipt_preserved=True,model_fits=0))
print(json.dumps({k:v for k,v in evidence.items() if k in ["status","code","signal_date","target_date","original_target_LABEL_ONLY","corrected_target_LABEL_ONLY","raw_previous_close_10pct_cap","cash_adjusted_10pct_cap","actual_target_close","model_fits","original_data_overwrites"]}))
