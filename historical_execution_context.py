"""Recover saved evaluation-day OHLC without changing scores or signal inputs.

Future bars are labels/execution scenarios only. They never enter the causal
feature engine. All sources are reconstructed from the original request bounds
and raw hashes; conflicting observations stay unavailable.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b
import short_term_thinking_research as r
import relay_signal_ohlc_context as archive
import historical_symbolic_context as symbolic


def run(output_dir=symbolic.RUN):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/"execution_context.joblib").exists():raise FileExistsError("immutable evaluation source context exists")
    original=b.DEFAULT_RUN
    saved=original/"delivery/complete_paired_predictions_VALIDATED.csv"
    pred=pd.read_csv(saved,dtype={"code":str},low_memory=False)
    source_bundle=joblib.load(original/"scope_correction/current_extension.joblib")
    sessions=source_bundle["sessions"];index={d:i for i,d in enumerate(sessions)}
    keys=set()
    for row in pred[["date","code"]].drop_duplicates().itertuples(index=False):
        keys.update((d,row.code) for d in sessions[index[row.date]:index[row.date]+4])
    old=source_bundle["bars"]
    old=old[[ (row.date,row.code) in keys for row in old.itertuples()]].copy()
    observations=[];bindings={str(saved.resolve()):b.sha(saved),str(Path(__file__).resolve()):b.sha(__file__)};audits=[]
    for relative,artifact,parser in archive.ARCHIVES:
        frame,metadata,sources,allowed=archive.archive_rows(relative,artifact,parser)
        bindings.update(sources)
        frame["code"]=frame.symbol.str[2:]
        take=frame[[ (x.date,x.code) in keys for x in frame.itertuples()]].copy()
        for x in take.to_dict("records"):
            try:prices=tuple(b.cents(x[k]) for k in ("open","high","low","close"))
            except ValueError:continue
            observations.append(dict(date=x["date"],code=x["code"],prices=prices,volume=x["volume"],archive=relative,
                raw_path=metadata[x["symbol"]]["raw_path"],available_at=metadata[x["symbol"]]["available_at"]))
        audits.append(dict(archive=relative,full_rebuilt_rows=len(frame),evaluation_keys_seen=len(take),raw_sources=len(metadata),
            use="T/B/B+1/B+2 outcome scenario only; not feature or original training authority"))
    grouped={}
    for obs in observations:grouped.setdefault((obs["date"],obs["code"]),[]).append(obs)
    old_index=old.set_index(["date","code"])
    updates=[];conflicts=[];references=[];newly_observed=0
    for (day,code),items in grouped.items():
        price_values={x["prices"] for x in items}
        if len(price_values)!=1:
            conflicts.append(dict(date=day,code=code,reason="SAVED_RAW_ARCHIVES_DISAGREE",raw_paths=[x["raw_path"] for x in items]))
            updates.append(dict(date=day,code=code,open=np.nan,high=np.nan,low=np.nan,close=np.nan,volume=np.nan))
            continue
        price=items[0]["prices"]
        if (day,code) in old_index.index:
            row=old_index.loc[(day,code)]
            try:existing=tuple(b.cents(row[k]) for k in ("open","high","low","close"))
            except ValueError:existing=None
            if existing is not None and existing!=price:
                conflicts.append(dict(date=day,code=code,reason="RAW_VERSUS_PRIOR_NORMALIZED_PRICE_CONFLICT",prior=existing,raw=price))
                updates.append(dict(date=day,code=code,open=np.nan,high=np.nan,low=np.nan,close=np.nan,volume=np.nan))
                continue
        else:newly_observed+=1
        updates.append(dict(date=day,code=code,**dict(zip(("open","high","low","close"),[p/100 for p in price])),volume=items[0]["volume"],
            volume_unit="PROVIDER_NATIVE_SIGN_ONLY_NO_CROSS_SERIES_RATIO"))
        references.append(dict(date=day,code=code,source_count=len(items),raw_paths=[x["raw_path"] for x in items],
            actual_receipts=[x["available_at"] for x in items]))
    enriched=pd.concat([old,pd.DataFrame(updates)],ignore_index=True).drop_duplicates(["date","code"],keep="last")
    pd.DataFrame(references).to_csv(out/"evaluation_daily_source_bindings.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(conflicts).to_csv(out/"evaluation_daily_conflicts_not_selected.csv",index=False,encoding="utf-8-sig")
    context=dict(bars=enriched,sessions=sessions,sources=bindings,not_signal_features=True)
    joblib.dump(context,out/"execution_context.joblib",compress=3)
    comparisons={}
    for fold in r.history.FOLDS+[dict(id="2026Oct",start="2026-10-01",end="2026-10-31")]:
        part=pred[pred.date.between(fold["start"],fold["end"])]
        result={};logs=[]
        for arm,p in part.groupby("arm",sort=True):
            selected=r.top_selection(p)
            before=r.proxy_returns(selected,dict(bars=old,sessions=sessions))
            after=r.proxy_returns(selected,context)
            assert list(zip(before.date,before.code))==list(zip(after.date,after.code))
            after["arm"]=arm;logs.append(after)
            result[arm]=dict(before=r.execution_metrics(before),after=r.execution_metrics(after),selected_codes_unchanged=True)
        if logs:pd.concat(logs,ignore_index=True).to_csv(out/(fold["id"]+"_same_selection_execution_enriched.csv"),index=False,encoding="utf-8-sig")
        comparisons[fold["id"]]=result
    report=dict(formed_at=b.now(),purpose="Outcome/execution source coverage repair, not a refit or new score comparison",
        original_prediction_csv=str(saved.resolve()),original_prediction_sha256=b.sha(saved),original_candidate_scores_changed=False,
        source_audits=audits,needed_daily_keys=len(keys),prior_daily_keys=len(old),raw_qualified_daily_keys=len(references),
        newly_observed_daily_keys=newly_observed,conflicting_keys=len(conflicts),enriched_daily_keys=len(enriched),comparisons=comparisons,
        new_fits=0,new_formal_predictions=0,new_forward_maturity=0,actual_or_paper_fills=0,
        execution_proxy="Original 0..7% non-limit OPEN scenario; costs/slippage, B+1/B+2 and blocked sells unchanged; not live order-flow verification",
        source_bindings=bindings,execution_context_path=str((out/"execution_context.joblib").resolve()),execution_context_sha256=b.sha(out/"execution_context.joblib"))
    b.write_new(out/"execution_source_coverage_report.json",report)
    print(b.clean({k:report[k] for k in ["needed_daily_keys","prior_daily_keys","raw_qualified_daily_keys","newly_observed_daily_keys","conflicting_keys"]}))
    return report


if __name__=="__main__":run()
