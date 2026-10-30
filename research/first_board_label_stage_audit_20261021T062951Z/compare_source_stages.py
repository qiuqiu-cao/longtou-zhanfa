"""Compare only acquired positive stage reports; never infer negatives from absence."""
from collections import Counter
from decimal import Decimal,InvalidOperation
from pathlib import Path
import json
import pandas as pd
import first_board_leader_model as base
import first_second_stage_model as stage

OUT=Path(__file__).resolve().parent
FIRST=base.ROOT/"research/first_board_joint_design_20261020T185417Z/ALL668_SOURCE_RECEIPTS.json"
FACTS=base.ROOT/"research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
def cents(value):
    try:
        x=Decimal(str(value))*100
        return int(x) if x.is_finite() and x>0 and x==x.to_integral_value() else None
    except (InvalidOperation,TypeError,ValueError):return None

def read_reports(manifest,expected_stage):
    result={}
    receipts=json.loads(manifest.read_text("utf-8"))
    for receipt in receipts:
        if not receipt.get("qualified_date_envelope"):continue
        path=Path(receipt["source_path"])
        assert base.sha(path)==receipt["sha256"]
        payload=json.loads(path.read_bytes());info=payload["info"]
        assert str(payload["errcode"])=="0" and info[1]==receipt["date"]
        counts=Counter(str(row[0]).zfill(6) for row in info[0] if isinstance(row,list) and len(row)>=23)
        for i,row in enumerate(info[0]):
            if not isinstance(row,list) or len(row)<23:continue
            code=str(row[0]).zfill(6)
            if not code.isdigit() or len(code)!=6 or counts[code]!=1:continue
            assert row[15]==expected_stage
            result[receipt["date"],code]=dict(close=row[21],reported_stage=row[15],pct=row[22],
                source_sha256=receipt["sha256"],source_path=str(path),pointer=f"/info/0/{i}",received_at=receipt.get("received_at"))
    return result

def main():
    if (OUT/"BOUNDED_STAGE_COMPARISON.json").exists():raise FileExistsError("Preserve comparisons")
    facts_hash=base.sha(FACTS)
    protocol=json.loads((OUT/"COLLECTION_PROTOCOL_BEFORE_REQUESTS.json").read_text("utf-8"))
    days=protocol["request_dates"]
    first=read_reports(FIRST,1)
    second=read_reports(OUT/"ALL_LIMITED_SECOND_BOARD_RECEIPTS.json",2)
    rows=pd.read_csv(FACTS,dtype={"code":str},float_precision="round_trip")
    rows=rows[rows.stage.eq(1)].reset_index(drop=True)
    selected=rows[rows.label_end.isin(days)].copy()
    price=pd.concat([pd.read_csv(base.INPUTS[name],usecols=["date","code","close"],dtype={"code":str},float_precision="round_trip")
        for name in ("bars","recovered")],ignore_index=True)
    price["date"]=price.date.astype(str).str[:10]
    assert not price.duplicated(["date","code"]).any()
    prices=dict(zip(zip(price.date,price.code.str.zfill(6)),price.close))
    records=[]
    for row in selected.itertuples(index=False):
        signal=first.get((row.signal_date,row.code),{})
        same_signal=bool(signal) and cents(row.close)==cents(signal["close"])
        nxt1=first.get((row.label_end,row.code),{})
        nxt2=second.get((row.label_end,row.code),{})
        target_close=prices.get((row.label_end,row.code))
        match1=bool(nxt1) and cents(target_close) is not None and cents(target_close)==cents(nxt1["close"])
        match2=bool(nxt2) and cents(target_close) is not None and cents(target_close)==cents(nxt2["close"])
        y=getattr(row,stage.TARGET)
        if nxt1 and nxt2:category="PROVIDER_STAGES_OVERLAP_PENDING"
        elif nxt2 and not match2:category="SECOND_REPORT_PRICE_CONFLICT_OR_RAW_MISSING_PENDING"
        elif match2 and pd.isna(y):category="ORIGINAL_TARGET_UNKNOWN_SECOND_REPORT_PRESENT_NO_REFILL"
        elif match2 and y==0:category="ORIGINAL_NEGATIVE_VS_SECOND_REPORT_PENDING"
        elif match2 and same_signal:category="ORIGINAL_POSITIVE_BOTH_S_FIRST_AND_NEXT_SECOND_CORROBORATED"
        elif match2:category="ORIGINAL_POSITIVE_NEXT_SECOND_CORROBORATED_SIGNAL_FIRST_PENDING"
        elif match1 and y==1:category="ORIGINAL_POSITIVE_NEXT_PROVIDER_FIRST_STAGE_CONFLICT_PENDING"
        elif match1 and pd.isna(y):category="ORIGINAL_UNKNOWN_NEXT_PROVIDER_FIRST_NO_REFILL"
        elif match1:category="ORIGINAL_NEGATIVE_NEXT_PROVIDER_FIRST_CLOSE_PRESENT_STAGE_PENDING"
        else:category="NO_POSITIVE_MATCHING_STAGE_REPORT_UNKNOWN_NOT_NEGATIVE"
        records.append(dict(signal_date=row.signal_date,code=row.code,label_end=row.label_end,original_target_LABEL_ONLY=y,
            original_signal_close=row.close,original_target_close=target_close,signal_first_present=bool(signal),signal_first_price_equal=same_signal,
            target_first_present=bool(nxt1),target_first_price_equal=match1,target_second_present=bool(nxt2),target_second_price_equal=match2,
            second_source_path=nxt2.get("source_path"),second_source_sha256=nxt2.get("source_sha256"),second_source_pointer=nxt2.get("pointer"),
            second_source_received_at=nxt2.get("received_at"),first_target_source_path=nxt1.get("source_path"),
            first_target_source_sha256=nxt1.get("source_sha256"),category=category,
            sample_group="fixed_chronological" if row.label_end in protocol["fixed_chronological_dates"] else "outcome_seen_stage_diagnostic"))
    result=pd.DataFrame(records)
    result.to_csv(OUT/"ALL_ORIGINAL_FIRST_BOARD_ROWS_ON_LIMITED_TARGET_DATES.csv",index=False,encoding="utf-8-sig")
    found=set(zip(selected.label_end,selected.code))
    outside=[]
    for (day,code),report in second.items():
        if (day,code) not in found:
            outside.append(dict(target_date=day,code=code,provider_close=report["close"],reason="NO_ORIGINAL_FIRST_STAGE_ON_PREVIOUS_OFFICIAL_DAY",**{k:v for k,v in report.items() if k not in ("close",)}))
    pd.DataFrame(outside).to_csv(OUT/"ALL_SECOND_REPORTS_OUTSIDE_ORIGINAL_FIRST_COHORT.csv",index=False,encoding="utf-8-sig")
    assert base.sha(FACTS)==facts_hash
    summary=dict(at=base.now(),status="BOUNDED_STAGE_CROSSCHECK_COMPLETE_NO_LABEL_EDITS",original_scope_rows=len(result),signal_dates=result.signal_date.nunique(),
        target_dates=result.label_end.nunique(),positive_second_report_rows=len(second),second_reports_outside_original_first_cohort=len(outside),
        categories=result.category.value_counts().to_dict(),sample_groups={key:dict(rows=len(part),categories=part.category.value_counts().to_dict(),
            original_positive=int(part.original_target_LABEL_ONLY.eq(1).sum()),original_unknown=int(part.original_target_LABEL_ONLY.isna().sum())) for key,part in result.groupby("sample_group")},
        original_labels_unchanged=True,model_fits=0,target_rewrites=0,confirmed_exchange_corporate_action_errors=0,
        caution="Only21 predeclared target dates;15 include outcome-seen stage discrepancies. Positive supplier reports are not exchange gold labels. Missing means unknown,not negative;no accuracy estimate or cleaned training labels.")
    base.write_json(OUT/"BOUNDED_STAGE_COMPARISON.json",summary)
    print(json.dumps(base.clean(summary),ensure_ascii=True))

if __name__=="__main__":main()
