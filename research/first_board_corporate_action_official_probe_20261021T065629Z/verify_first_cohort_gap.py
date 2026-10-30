"""Check one first-board omission without reading its future promotion target."""
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
from datetime import datetime,timezone
import hashlib,json
import pandas as pd
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
facts=ROOT/"research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
before=sha(facts)
original=pd.read_csv(facts,usecols=["signal_date","code","stage"],dtype={"code":str})
assert original[original.code.eq("600361")&original.signal_date.eq("2024-06-28")].empty
ledger=pd.read_csv(ROOT/"research/first_board_corporate_action_qualification_20261021T064545Z/XD_DR_XR_CANDIDATE_ACTION_LEDGER.csv",dtype={"code":str})
candidate=ledger[ledger.code.eq("600361")&ledger.source_date.eq("2024-06-28")]
assert len(candidate)==1
c=candidate.iloc[0]
assert sha(c.source_path)==c.source_sha256
payload=json.loads(Path(c.source_path).read_bytes())
record=payload["info"][0][int(c.name_pointer.split("/")[3])]
assert record[0]=="600361" and record[15]==1 and Decimal(str(record[21]))==Decimal("4.03")
parts=[]
path=ROOT/"data/v16/point_in_time_rebuild_cutoff_20260831_v2/point_in_time_current_daily_bars.csv"
cols=["date","code","open","high","low","close","pre_close","st","new_listing","status_verified"]
for chunk in pd.read_csv(path,usecols=cols,dtype={"code":str},chunksize=200000,float_precision="round_trip"):
    parts.append(chunk[chunk.code.eq("600361")&chunk.date.isin(["2024-06-27","2024-06-28"])])
rows=pd.concat(parts,ignore_index=True).sort_values("date")
assert len(rows)==2
assert not rows.st.any() and not rows.new_listing.any() and rows.status_verified.all()
old,new=rows.iloc[0],rows.iloc[1]
nominal=(Decimal(str(new.pre_close))*Decimal("1.10")).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP)
announced=((Decimal(str(old.close))-Decimal("0.07"))*Decimal("1.10")).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP)
assert str(nominal)=="4.10" and str(announced)=="4.03"
previous_cap=(Decimal(str(old.pre_close))*Decimal("1.10")).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP)
assert Decimal(str(old.close))!=previous_cap
receipt=dict(at=datetime.now(timezone.utc).isoformat(),status="FIRST_COHORT_OMISSION_CORROBORATED_BY_OFFICIAL_REFERENCE_AND_DATED_PROVIDER",
    code="600361",signal_date="2024-06-28",original_stage1_or_stage2_rows=0,
    signal_close="4.03",previous_close="3.73",previous_session_is_not_nominal_limit=True,
    eligible_main_status_verified=True,st=False,new_listing=False,
    old_nominal_cap="4.10",company_formula_cap="4.03",provider_stage=1,
    provider_json_pointer="/".join(c.name_pointer.split("/")[:-1]),
    provider_source_sha256=c.source_sha256,official_pdf_sha256=sha(OUT/"raw/600361_implementation.pdf"),
    future_outcome_or_label_read=False,new_target_label_assigned=False,
    original_rows_added=0,original_facts_sha256=before,old_failures_retrained=False,model_fits=0,
    qualification="Official-announcement-derived cap and supplier first-board report;not an exchange queue gold label or complete cohort repair")
assert sha(facts)==before
with (OUT/"ONE_FIRST_COHORT_OMISSION_EVIDENCE.json").open("x",encoding="utf-8") as stream:
    json.dump(receipt,stream,ensure_ascii=False,indent=2)
print("FIRST_COHORT_OMISSION_600361_VERIFIED_NO_FUTURE_TARGET_READ_OR_ROW_INSERTION")
