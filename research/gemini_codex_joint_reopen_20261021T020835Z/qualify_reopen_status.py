"""Qualify a previously unmapped daily path flag; no outcome columns, fitting or requests."""
from collections import Counter
from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import first_board_joint_seal_model as parent

OUT=Path(__file__).resolve().parent
FIELD="provider_reported_any_reopen"
def flag(value):
    if isinstance(value,bool):return np.nan
    if str(value)=="0":return 0.
    if str(value)=="1":return 1.
    return np.nan

def run():
    protocol=dict(at=parent.base.now(),previous_goal_turn="PROGRESS:Gemini andCodexagreedandtrained64/66arms,bothfailed,fullreplayandpeerunsupportedstatisticscorrected",hypothesis="A separately qualified Sdated intraday reopening state provides path information not contained in firstsealtime alone or closing OHLC;do not refit the failed64/66arms",
        new_field=FIELD,raw_pointer="/info/0/{record}/16",preliminary_evidence="61/61 previously archived EM any-bomb flags match raw0/1;now auditbothoriginal61dates without outcomes",fits=0,outcomes_read=False,new_requests=0,
        qualification="Compare all122 raw-boundEMrows across2026Aug28/Aug31,includingpositivebombcounts>1;all available intersections must match binary(anyreportbomb>0),atleast20positivesand20zeros. Original-pool exact-cent/source-date qualification inherited;newfield>=90%coverage development and2025H1. Missing/unknownvalueNaN,explicitreported0 preserved.",
        boundaries="Provider daily quote/path proxy,not exchange truth/actoridentity/actualbuyability. No claim new field is trustworthy merely because array looks binary. Original source clocks/late acquisition andsurvivorship limits retained.",
        proposed_fixed_arms="Discuss original62+flag (63) vs original62+flag+priorqualifiedtime/fund4 (67),samebaseline/params/members andgates. Peeragreement andnewbatchfreeze required beforefit")
    with (OUT/"PROTOCOL_BEFORE_FULL_FIELD_AUDIT.json").open("x",encoding="utf-8") as f:json.dump(protocol,f,ensure_ascii=False,indent=2)
    receipts=json.loads((parent.SOURCES/"ALL668_SOURCE_RECEIPTS.json").read_text("utf-8"))
    by_day={};records=[];hist=Counter()
    for receipt in receipts:
        if not receipt.get("qualified_date_envelope"):continue
        path=Path(receipt["source_path"]);assert parent.base.sha(path)==receipt["sha256"]
        payload=json.loads(path.read_bytes());assert payload["info"][1]==receipt["date"]
        day={}
        for index,row in enumerate(payload["info"][0]):
            if not isinstance(row,list) or len(row)<23:continue
            code=str(row[0]).zfill(6)
            day[code]=row
            hist[(type(row[16]).__name__,str(row[16]))]+=1
            records.append(dict(signal_date=receipt["date"],code=code,reopen_raw_value=row[16],provider_flag=flag(row[16]),raw_record_pointer=f"/info/0/{index}/16",raw_source_path=str(path),raw_source_sha256=receipt["sha256"]))
        by_day[receipt["date"]]=day
    archived=json.loads((ROOT/"research/first_board_seal_source_qualification_20261020/raw_bound_first_board_observations.json").read_text("utf-8"))
    cross=[]
    for original in archived:
        row=by_day.get(original["date"],{}).get(original["code"])
        if row is None:continue
        expected=float(original["bomb_count"]>0)
        observed=flag(row[16])
        cross.append(dict(date=original["date"],code=original["code"],kpl_raw_flag=row[16],kpl_any_reopen=observed,em_reported_bomb_count=original["bomb_count"],em_any_reopen=expected,matches=bool(observed==expected),em_raw_path=original["provider_source_path"],em_raw_sha256=original["provider_source_sha256"],em_raw_line=original["raw_line_number"]))
    assert len(cross)>=100
    pd.DataFrame(cross).to_csv(OUT/"ALL122_CROSS_SOURCE_REOPEN_SEMANTICS_NO_TARGETS.csv",index=False,encoding="utf-8-sig")
    prior=pd.read_csv(parent.QUAL/"COMPLETE28503_NEW_FIELDS_NO_LABELS.csv",dtype={"code":str},float_precision="round_trip")
    raw=pd.DataFrame(records)
    assert not raw.duplicated(parent.KEYS).any()
    joined=prior.merge(raw,on=parent.KEYS,how="left",sort=False,validate="one_to_one")
    # Reuse exact dated stock/cent-price qualification, not future outcomes.
    allowed=joined.exact_cent_close_match & joined.source_time_minutes.notna()
    joined[FIELD]=joined.provider_flag.where(allowed)
    periods={}
    for name,mask in [("development",joined.signal_date.lt("2025-01-01")),("validation2025H1",joined.signal_date.between("2025-01-01","2025-06-30")),("evaluation",joined.signal_date.ge("2025-07-01"))]:
        part=joined[mask]
        periods[name]=dict(rows=len(part),known=int(part[FIELD].notna().sum()),coverage=float(part[FIELD].notna().mean()),reported_zero=int(part[FIELD].eq(0).sum()),reported_reopened=int(part[FIELD].eq(1).sum()))
    cross_ok=all(r["matches"] for r in cross) and sum(r["em_any_reopen"]==1 for r in cross)>=20 and sum(r["em_any_reopen"]==0 for r in cross)>=20
    good=cross_ok and all(periods[p]["coverage"]>=.90 for p in ("development","validation2025H1"))
    assert "promotion_next_LABEL_ONLY" not in joined and "label_end" not in joined
    joined.to_csv(OUT/"COMPLETE28503_REOPEN_AND_PRIOR_FIELDS_NO_TARGETS.csv",index=False,encoding="utf-8-sig")
    report=dict(at=parent.base.now(),qualified=good,cross_source_matches=sum(r["matches"] for r in cross),cross_source_rows=len(cross),cross_source_reported_zeros=sum(r["em_any_reopen"]==0 for r in cross),cross_source_reported_positive=sum(r["em_any_reopen"]==1 for r in cross),cross_source_counts_greater_than1=sum(r["em_reported_bomb_count"]>1 for r in cross),
        raw_value_inventory=[dict(type=k[0],value=k[1],count=v) for k,v in sorted(hist.items())],periods=periods,new_field=FIELD,full_rows=len(joined),original_four_fields_preserved=True,model_fits=0,outcomes_read=False,
        no_exchange_truth_claim=True,no_actor_identity_claim=True,provider_history_retrieved_later=True,survivorship_unchanged=True)
    parent.base.write_json(OUT/"qualification.json",report)
    print(json.dumps(parent.base.clean(report),ensure_ascii=True),flush=True)
    return report

if __name__=="__main__":run()
