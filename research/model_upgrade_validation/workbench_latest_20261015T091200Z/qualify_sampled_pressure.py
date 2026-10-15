"""Qualification and descriptive TP/FP audit; no estimator, ranking or refit."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b
import short_term_thinking_research as old
import short_term_sampled_pressure as p
import public_minute_capture_v1 as parser

ROOT=Path(__file__).resolve().parent
ARCHIVES=[b.ROOT/"data/v289/full_august_five_minute_20261002_v1",b.ROOT/"data/v287/september_five_minute_20261002_v1"]


def raw_replay(archive):
    report=b.read(archive/"report.json")
    b.verify(report["source_hashes"])
    bindings=dict(report["source_hashes"])
    for name,digest in report["artifact_hashes"].items():
        path=archive/name
        if b.sha(path)!=digest:raise ValueError("original minute artifact changed")
        bindings[str(path.resolve())]=digest
    saved=pd.read_csv(archive/"five_minute_bars_UNRECONCILED.csv",dtype={"code":str})
    entries=[x for x in report["audit"] if x.get("kind","minute")=="minute"]
    frames=[];valid=0
    for item in entries:
        if not item.get("raw_path"):continue
        raw=Path(item["raw_path"])
        if b.sha(raw)!=item["raw_sha256"]:raise ValueError("minute original HTTP hash changed")
        bindings[str(raw.resolve())]=item["raw_sha256"]
        if item.get("status")!="RAW_PARSED":continue
        frame=parser.parse_minute_response(raw.read_text("utf-8"),item["code"])
        if frame.empty:raise ValueError("old parsed raw now empty")
        frames.append(frame);valid+=1
    rebuilt=pd.concat(frames,ignore_index=True).sort_values(["code","minute"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt[saved.columns],saved.sort_values(["code","minute"]).reset_index(drop=True),check_dtype=False,atol=1e-9,rtol=1e-12)
    bindings[str((archive/"report.json").resolve())]=b.sha(archive/"report.json")
    return saved,bindings,dict(archive=str(archive.resolve()),raw_symbols_replayed=valid,raw_bars_replayed=len(rebuilt))


def main():
    out=ROOT/"sampled_pressure_evidence"
    if out.exists():raise FileExistsError("immutable sampled-pressure audit")
    out.mkdir()
    prior=b.ROOT/"results/v289/august_september_paths_20261002_v1/features_NO_NEXT_SESSION_LABELS.csv"
    bindings={str(prior.resolve()):b.sha(prior),str(Path(p.__file__).resolve()):b.sha(p.__file__),str(Path(__file__).resolve()):b.sha(__file__)}
    b.write_new(out/"descriptive_protocol_BEFORE_LABEL_JOIN.json",dict(formed_at=b.now(),purpose="Source qualification and descriptive discrimination audit of separate-interval pressure/recovery facts",fields=p.FIELDS,
        old_failed_arms_not_rerun=["FINAL_CAP_PRICE_RUN","CAP_PRICE_BAR_SHARE","SHALLOW_PULLBACK"],new_ranking_arms=0,
        fit_calls=0,threshold_search=False,labels_already_seen_in_past=True,blind=False,
        pressure_semantics="Chronology between non-overlapping five-minute bars only; no intrabar high/low order or true reseal inference",
        missing_policy="Complete grid plus OHLC match required; volume facts additionally need independent daily-volume/amount checks. All9245candidate rows retained.",
        interpretation="Fixed descriptive groups, no winner selection, causal or forecasting claim; original predictions unchanged",source_bindings=bindings))
    cohort=pd.read_csv(prior,dtype={"code":str},usecols=["date","code","boards","price"])
    rows=[];replays=[]
    for archive in ARCHIVES:
        bars,source,receipt=raw_replay(archive);bindings|=source;replays.append(receipt)
        august=archive.parts[-1].startswith("full_august")
        daily_path=archive/"unadjusted_daily_reference.csv" if august else b.ROOT/"data/v278/september_unadjusted_20261002_v3/unadjusted_daily.csv"
        reference=pd.read_csv(daily_path);bindings[str(daily_path.resolve())]=b.sha(daily_path)
        part=cohort[cohort.date.str.startswith("2026-08" if august else "2026-09")]
        groups={code:g for code,g in bars.groupby("code")}
        empty=bars.iloc[:0]
        for stock in part.itertuples():
            row=p.describe(groups.get(stock.code,empty),reference,stock.code,stock.date,stock.price)
            row["source_archive"]=str(archive.resolve())
            rows.append(row)
    observed=pd.DataFrame(rows)
    if len(observed)!=485 or observed.duplicated(["date","code"]).any():raise ValueError("complete original485rows required")
    observed.to_json(out/"qualified_sampled_pressure_OBSERVATIONS_NO_LABELS.json",orient="records",force_ascii=False,indent=2)
    serial=observed.drop(columns=["source_checks"])
    serial.to_csv(out/"qualified_sampled_pressure_BEFORE_LABEL_JOIN.csv",index=False,encoding="utf-8-sig")
    full=joblib.load(b.DEFAULT_RUN/"scope_correction/dataset.joblib")
    # The causal engine has already run without any promotion outcome fields.
    causal=full["features"]
    columns=["date","code","name","boards","daily_one_word","primary_role"]
    combined=causal[columns].merge(serial,on=["date","code"],how="left",validate="one_to_one")
    for key in ["price_path_qualified","volume_path_qualified"]:combined[key]=combined[key].fillna(False).astype(bool)
    combined=p.cohort_relative(combined)
    assert len(combined)==9245
    combined.to_csv(out/"full9245_pressure_context_NO_TARGETS.csv",index=False,encoding="utf-8-sig")
    training=old.merged(full)
    coverage={}
    for fold in old.history.FOLDS:
        subset=old.history.training_rows(training,fold["start"],"relay")
        joined=subset[["date","code"]].merge(combined,on=["date","code"],validate="one_to_one")
        coverage[fold["id"]]=dict(rows=len(joined),price_path_rows=int(joined.price_path_qualified.sum()),volume_path_rows=int(joined.volume_path_qualified.sum()),
            price_coverage=float(joined.price_path_qualified.mean()),volume_coverage=float(joined.volume_path_qualified.mean()))
    # Only now join unchanged saved prediction and target columns for a full
    # descriptive ledger, preserving every false/unknown/abstained candidate.
    pred_path=b.DEFAULT_RUN/"delivery/complete_paired_predictions_VALIDATED.csv"
    pred=pd.read_csv(pred_path,dtype={"code":str},low_memory=False)
    pred=pred[pred.arm.isin(["BASE","ST_DUAL"])&pred.date.between("2026-08-01","2026-09-30")]
    joined=pred.merge(combined.drop(columns=["name","boards","daily_one_word","primary_role"]),on=["date","code"],validate="many_to_one")
    joined["outcome_group"]=["UNKNOWN_LABEL" if not valid or pd.isna(target) else ("TP" if score>.5 else "FN") if target==1 else ("FP" if score>.5 else "TN")
        for target,valid,score in zip(joined.target,joined.label_valid,joined.score)]
    joined.to_csv(out/"complete_485_saved_prediction_pressure_diagnostic.csv",index=False,encoding="utf-8-sig")
    groups=[]
    for (period,arm,outcome),group in joined[joined.daily_one_word.eq(0)].groupby([joined.date.str[:7],"arm","outcome_group"]):
        values={}
        for key in p.FIELDS:
            known=group[key].dropna()
            values[key]=dict(known=len(known),median=float(known.median()) if len(known) else None,mean=float(known.mean()) if len(known) else None)
        groups.append(dict(period=period,arm=arm,outcome_group=outcome,full_rows=len(group),price_qualified=int(group.price_path_qualified.sum()),volume_qualified=int(group.volume_path_qualified.sum()),fields=values))
    feature_cases=joined[joined.arm.eq("ST_DUAL")&joined.daily_one_word.eq(0)][["date","code","name","boards","score","target","label_valid","outcome_group","price_path_qualified","volume_path_qualified"]+p.FIELDS]
    feature_cases.to_csv(out/"all_nonword_dual_cases_NOT_FILTERED.csv",index=False,encoding="utf-8-sig")
    bindings[str(pred_path.resolve())]=b.sha(pred_path)
    b.verify(bindings)
    result=dict(formed_at=b.now(),status="RAW_REPLAY_AND_CAUSAL_PRESSURE_FEATURES_QUALIFIED_NOT_PREDICTIVE_ACCEPTANCE",historical_rows=9245,original_minute_scope_rows=485,
        price_path_qualified=int(observed.price_path_qualified.sum()),volume_path_qualified=int(observed.volume_path_qualified.sum()),raw_replays=replays,
        original_training_windows=coverage,descriptive_groups=groups,
        prior_training_coverage_sufficient_for_95pct_claim=False,strict_original_receipt_training_coverage=0,
        independently_annotated_role_truth_added=0,new_fits=0,new_rank_or_score_comparisons=0,original_scores_changed=False,new_mature_samples=0,
        not_tick_path_or_absorption=True,not_a_new_model_or_deployment=True,source_bindings=bindings)
    b.write_new(out/"qualification_and_discrimination_audit.json",result)
    print({k:v for k,v in result.items() if k not in {"source_bindings","descriptive_groups"}})


if __name__=="__main__":main()
