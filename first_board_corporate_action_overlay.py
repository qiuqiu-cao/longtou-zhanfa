"""Explicit research-label overlays backed by verified corporate-action evidence.

Never rewrites the original target, infers negatives from provider absence or
feeds target information into features. This is not a complete action database.
"""
from __future__ import annotations
from decimal import Decimal,InvalidOperation,ROUND_HALF_UP
from pathlib import Path
import hashlib
import pandas as pd

ORIGINAL = "promotion_next_LABEL_ONLY"
CORRECTED = "promotion_next_CORRECTED_RESEARCH_LABEL_ONLY"
KEYS = ["signal_date", "code", "label_end"]


def cash_cap(previous_close, gross_cash_per_share):
    try:
        previous = Decimal(str(previous_close));cash = Decimal(str(gross_cash_per_share))
    except InvalidOperation as exc:
        raise ValueError("Invalid decimal price or dividend") from exc
    if not previous.is_finite() or not cash.is_finite() or previous <= 0 or cash < 0 or previous <= cash:
        raise ValueError("Nonpositive reference or invalid gross cash")
    # No rounding of the cash amount to two decimals; the documented0.109 matters.
    reference = previous-cash
    return reference, (reference*Decimal("1.10")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def check_evidence(item, source_root):
    if item.get("status") != "ONE_CORPORATE_ACTION_TARGET_ERROR_CONFIRMED_SEPARATE_LAYER":
        raise ValueError("Correction is not independently verified evidence")
    if item.get("announced_differential_dividend") is not False:
        raise ValueError("Differential dividend requires a separate qualified reference formula")
    if item["target_date"] != item["announced_ex_date"] or item["signal_date"] != item["announced_record_date"]:
        raise ValueError("Date keys do not match the qualified case")
    if item["announcement_date"] > item["signal_date"]:
        raise ValueError("Announcement was not published before the first-close decision")
    path = Path(source_root) / "raw/603214_2024H1_DIVIDEND_IMPLEMENTATION.pdf"
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["source_pdf_sha256"]:
        raise ValueError("Original company evidence hash changed")
    reference, cap = cash_cap(item["signal_close"], item["announced_cash_per_share"])
    if str(reference) != item["derived_ex_reference"] or cap != Decimal(item["actual_target_close"]):
        raise ValueError("Dividend-derived cap does not match observed target close")
    if cap != Decimal(item["cash_adjusted_10pct_cap"]):
        raise ValueError("Stored corrected cap is inconsistent")
    if item["corrected_target_LABEL_ONLY"] != 1 or item["original_target_LABEL_ONLY"] != 0:
        raise ValueError("Only the explicitly audited0->1 case is admitted")
    # The supplier report is a second source, not an exchange queue or execution record.
    target_path = Path(source_root) / "raw" / (item["target_date"] + ".response")
    if hashlib.sha256(target_path.read_bytes()).hexdigest() != item["target_report_source_sha256"]:
        raise ValueError("Original target report hash changed")
    import json
    payload = json.loads(target_path.read_bytes())
    if payload["info"][1] != item["target_date"] or str(payload["errcode"]) != "0":
        raise ValueError("Wrong target report date envelope")
    parts = item["target_report_pointer"].split("/")
    if len(parts) != 4 or parts[1:3] != ["info","0"]:
        raise ValueError("Wrong source pointer")
    record = payload["info"][0][int(parts[3])]
    if str(record[0]).zfill(6) != item["code"] or record[15] != 2 or Decimal(str(record[21])) != cap:
        raise ValueError("Target report identity, stage or close mismatch")


def build_research_overlay(original_rows, corrections, source_root):
    """Return all original rows and fields plus a separately named corrected target."""
    if original_rows.duplicated(KEYS).any():
        raise ValueError("Duplicate original candidate/target key")
    output = original_rows.copy()
    output[CORRECTED] = output[ORIGINAL].copy()
    output["corporate_action_correction_applied"] = False
    output["corporate_action_label_independently_verified"] = False
    seen = set()
    for item in corrections:
        check_evidence(item, source_root)
        key = (item["signal_date"],item["code"],item["target_date"])
        if key in seen:
            raise ValueError("Duplicate correction")
        seen.add(key)
        mask = output.signal_date.eq(key[0]) & output.code.eq(key[1]) & output.label_end.eq(key[2])
        if mask.sum() != 1:
            raise ValueError("Correction does not identify exactly one original row")
        row = output.loc[mask].iloc[0]
        if row[ORIGINAL] != item["original_target_LABEL_ONLY"] or Decimal(str(row.close)) != Decimal(item["signal_close"]):
            raise ValueError("Original target/price no longer matches the audited case")
        output.loc[mask, CORRECTED] = item["corrected_target_LABEL_ONLY"]
        output.loc[mask, "corporate_action_correction_applied"] = True
        output.loc[mask, "corporate_action_label_independently_verified"] = True
    pd.testing.assert_frame_equal(output[original_rows.columns],original_rows)
    return output
