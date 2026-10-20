"""Independent histogram/count/quantile audit without loading target columns."""
from pathlib import Path
import bisect
import hashlib
import json
import math
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_input_shift as audit

OUT=Path(__file__).resolve().parent
protocol=json.loads((OUT/"protocol_AFTER_ARRAY_COPY_REPAIR.json").read_text("utf-8"))
original_protocol=json.loads((OUT/"protocol_BEFORE_AUDIT.json").read_text("utf-8"))
assert base.sha(OUT/"protocol_BEFORE_AUDIT.json")==protocol["parent_protocol_sha256"]
assert base.sha(audit.__file__)==protocol["source_sha256"]
assert protocol["inputs"]==original_protocol["inputs"] and protocol["fields"]==original_protocol["fields"]
for path,digest in protocol["inputs"].items():assert base.sha(path)==digest,path
source=base.ROOT/"research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
rows=pd.read_csv(source,usecols=protocol["input_columns"],dtype={"code":str},float_precision="round_trip")
rows=rows[rows.stage.eq(1)].copy()
screen=pd.read_csv(base.ROOT/"research/first_board_capital_risk_20261020_run1/COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv",usecols=["signal_date","code","research_selection_allowed"],dtype={"code":str})
rows=rows.merge(screen,on=["signal_date","code"],validate="one_to_one")
prefix_dates=sorted(pd.read_csv(base.ROOT/"research/first_second_stage_20261020_run1/models/validation_S1_STAGE_CYCLE_GENE_calibration_prefix_members.csv",usecols=["signal_date"]).signal_date.unique())
full_dates=sorted(pd.read_csv(base.ROOT/"research/first_second_stage_20261020_run1/models/validation_S1_STAGE_CYCLE_GENE_full_members.csv",usecols=["signal_date"]).signal_date.unique())
report=json.loads((OUT/"report.json").read_text("utf-8"))
boundaries=json.loads((OUT/"PREFIX_ONLY_FIXED_BOUNDARIES.json").read_text("utf-8"))
metrics=pd.read_csv(OUT/"ALL62_FIXED_BOUNDARY_SHIFT_METRICS_NO_OUTCOMES.csv",float_precision="round_trip")
bins=pd.read_csv(OUT/"ALL_FIXED_BIN_COUNTS_NO_OUTCOMES.csv")
assert len(metrics)==1240
assert not any("LABEL_ONLY" in c or "score" in c for c in rows)
periods={"PREFIX_REFERENCE":rows.signal_date.isin(prefix_dates),"FULL_DEVELOPMENT":rows.signal_date.isin(full_dates),
    "CAL42":rows.signal_date.isin(full_dates[-42:]),"VALIDATION2025H1":rows.signal_date.between("2025-01-01","2025-06-30")}
for month in ["2025-01","2025-02","2025-03","2025-04","2025-05","2025-06"]:periods[month]=rows.signal_date.str.startswith(month)
indexed=metrics.set_index(["universe","period","feature"])
bin_indexed={(u,p,f):g.sort_values("bin_id") for (u,p,f),g in bins.groupby(["universe","period","feature"],sort=False)}
checked=0
def count_values(values,edges):
    counts=[0]*len(edges)
    for value in values:
        if not np.isfinite(value):counts[-1]+=1
        else:counts[bisect.bisect_right(edges[1:-1],value)]+=1
    return np.asarray(counts)
for universe,filter_mask in [("complete_pool",pd.Series(True,index=rows.index)),("fixed_risk_pool",rows.research_selection_allowed)]:
    reference=rows[periods["PREFIX_REFERENCE"]&filter_mask]
    for field in audit.FIELDS:
        rv=reference[field].to_numpy(float)
        observed=rv[np.isfinite(rv)]
        edges=np.array([float(v) for v in boundaries[universe][field]])
        if not len(observed):expected_edges=np.array([-np.inf,np.inf])
        elif np.unique(observed).size==1:
            c=observed[0];expected_edges=np.array([-np.inf,c,np.nextafter(c,np.inf),np.inf])
        else:expected_edges=np.r_[-np.inf,np.unique(pd.Series(observed).quantile(np.arange(.1,1.,.1)).to_numpy()),np.inf]
        np.testing.assert_allclose(edges,expected_edges,atol=1e-12,rtol=1e-12)
        rc=count_values(rv,edges)
        for period,mask in periods.items():
            current=rows.loc[mask&filter_mask,field].to_numpy(float)
            good=current[np.isfinite(current)]
            actual=count_values(current,edges)
            entry=indexed.loc[(universe,period,field)]
            saved=bin_indexed[(universe,period,field)]
            assert actual.sum()==len(current)==entry.rows
            assert rc.sum()==len(reference)
            np.testing.assert_array_equal(actual,saved.observed_count)
            np.testing.assert_array_equal(rc,saved.reference_count)
            assert (saved.observed_total==len(current)).all() and (saved.reference_total==len(reference)).all()
            assert saved.missing_bin.tolist()==[False]*(len(edges)-1)+[True]
            assert entry.finite==len(good) and entry.missing==len(current)-len(good)
            left=(rc+.5)/(rc.sum()+.5*len(rc));right=(actual+.5)/(actual.sum()+.5*len(actual))
            psi=math.fsum(float((b-a)*math.log(b/a)) for a,b in zip(left,right))
            np.testing.assert_allclose(entry.psi_with_missing_bin,psi,atol=1e-12,rtol=1e-12)
            if len(good):
                for q in audit.QUANTILES:
                    name="q"+str(int(q*100)).zfill(2)
                    np.testing.assert_allclose(entry[name],pd.Series(good).quantile(q),atol=1e-10,rtol=1e-10)
            if len(observed):
                assert entry.below_reference_min_count==int((good<observed.min()).sum())
                assert entry.above_reference_max_count==int((good>observed.max()).sum())
            checked+=1
age=pd.read_csv(OUT/"COMPLETE_AGE_CLOCK_DIAGNOSTIC_NO_OUTCOMES.csv",dtype={"code":str})
assert len(age)==len(rows)==28503
assert age.age_equals_source_calendar_position.tolist()==rows.listing_age_sessions.eq(rows.p).tolist()
master_file=base.ROOT/"data/v11/security_master_intervals.csv"
calendar_file=base.INPUTS["calendar"]
master=pd.read_csv(master_file,usecols=["code","listing_date"],dtype={"code":str})
sessions=json.loads(calendar_file.read_text("utf-8"))["sessions"]
listing=master.groupby("code").listing_date.first()
listing_pos={code:int(np.searchsorted(sessions,str(date)[:10])) for code,date in listing.items() if pd.notna(date)}
expected_age=rows.p-rows.code.map(listing_pos)
np.testing.assert_allclose(expected_age,rows.listing_age_sessions,atol=0,rtol=0,equal_nan=True)
for period,mask in periods.items():
    age_metric=report["age_clock"][period]
    assert age_metric["rows"]==int(mask.sum())
    assert age_metric["equals_source_position"]==int((mask&rows.listing_age_sessions.eq(rows.p)).sum())
    assert report["scopes"][period]["risk_allowed"]==int((mask&rows.research_selection_allowed).sum())
result=dict(status="PASS",verified_at=base.now(),full_rows=len(rows),metric_rows_independently_checked=checked,
    fixed_bins_all_counts_and_missing_recomputed=True,quantiles_and_psi_independently_checked=True,
    age_formula_recomputed_from_existing_security_master_and_official_calendar=True,
    exact_legacy_age_semantics="p - searchsorted(source_calendar,listing_date);listedbeforearchive=>p",
    outcome_or_score_columns_read=False,model_fits=0,new_forward_results=0,actual_trades=0,goal_60_passed=False,
    initial_readonlyarray_failure_preserved=True,verifier_sha256=base.sha(__file__),source_sha256=base.sha(audit.__file__),
    metadata_sources={str(master_file.resolve()):base.sha(master_file),str(calendar_file.resolve()):base.sha(calendar_file)})
base.write_json(OUT/"verification.json",result)
print(json.dumps(result,ensure_ascii=True,indent=2))
