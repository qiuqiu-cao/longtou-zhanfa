from decimal import Decimal
import json
from pathlib import Path
import pandas as pd
import pytest
import first_board_corporate_action_overlay as overlay

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"research/first_board_label_stage_audit_20261021T062951Z"

def evidence():
    return json.loads((SOURCE/"ONE_CONFIRMED_CORPORATE_ACTION_CORRECTION.json").read_text("utf-8"))

def rows():
    return pd.DataFrame(dict(signal_date=["2024-09-30"]*3,code=["603214","000001","000002"],
        label_end=["2024-10-08"]*3,close=[15.52,10.,11.],promotion_next_LABEL_ONLY=[0.,0.,float("nan")]))

def test_actual_gross_cash_preserves_subcent_and_correct_cap():
    reference, cap=overlay.cash_cap("15.52","0.109")
    assert reference==Decimal("15.411") and cap==Decimal("16.95")
    assert overlay.cash_cap("15.52","0")[1]==Decimal("17.07")

def test_full_pool_original_unknown_and_negative_preserved():
    original=rows();result=overlay.build_research_overlay(original,[evidence()],SOURCE)
    pd.testing.assert_frame_equal(result[original.columns],original)
    assert result[overlay.CORRECTED].tolist()[:2]==[1.,0.]
    assert pd.isna(result[overlay.CORRECTED].iloc[2])
    assert result.corporate_action_correction_applied.sum()==1
    assert result.corporate_action_label_independently_verified.tolist()==[True,False,False]

@pytest.mark.parametrize("key,value",[("announced_ex_date","2024-10-09"),("announced_differential_dividend",True),
    ("announcement_date","2024-10-09"),("source_pdf_sha256","0"*64),("target_report_source_sha256","0"*64)])
def test_unsupported_or_unbound_correction_rejected(key,value):
    item=evidence();item[key]=value
    with pytest.raises(ValueError):overlay.build_research_overlay(rows(),[item],SOURCE)

def test_duplicate_and_changed_original_rejected():
    item=evidence()
    with pytest.raises(ValueError,match="Duplicate correction"):
        overlay.build_research_overlay(rows(),[item,item],SOURCE)
    changed=rows();changed.loc[0,overlay.ORIGINAL]=1
    with pytest.raises(ValueError,match="Original target"):
        overlay.build_research_overlay(changed,[item],SOURCE)
