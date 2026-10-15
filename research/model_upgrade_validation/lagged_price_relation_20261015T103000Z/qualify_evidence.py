"""Source-only audit, before joining any forward outcomes to new relations."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b
import short_term_thinking_research as old
import historical_price_relation as p

ROOT=Path(__file__).resolve().parent
CONTEXT=ROOT/"causal_context_v2/price_relation_context.joblib"
DATA=b.ROOT/"research/model_upgrade_validation/workbench_latest_20261015T091200Z/attempt_reclassification_correction/corrected_dataset_SCHEMA_COMPLETE_RESEARCH_NOT_FITTED.joblib"
CURRENT=b.ROOT/"results/daily_review_20261015T093600Z"


def main():
    context=joblib.load(CONTEXT);b.verify(context["sources"])
    protected=b.read(ROOT/"execution_start.json")["protected_before"];b.verify(protected)
    engine=joblib.load(context["engine_path"]);rows=context["rows"];pairs=context["pairs"]
    assert len(rows)==9315 and len(pairs)==1465910
    assert rows.duplicated(["date","code"]).sum()==0
    ix={d:i for i,d in enumerate(engine["sessions"])}
    assert all(ix[r.date]-ix[r.estimate_end]>=6 for r in rows.itertuples() if pd.notna(r.estimate_end))
    assert not set(["target","target_date","label_valid","score"]) & set(pairs)
    # Independently build exact adjacent-session returns; no skipped trade dates.
    prices=engine["bars"][["date","code","close","pre_close"]]
    prices=prices[prices.date.isin(ix)]
    close=prices.pivot(index="date",columns="code",values="close").reindex(engine["sessions"])
    previous=close.shift(1)
    reference=prices.pivot(index="date",columns="code",values="pre_close").reindex(engine["sessions"])
    usable=close.gt(0)&previous.gt(0)&~(reference.gt(0)&reference.sub(previous).abs().gt(.011))
    returns=(close/previous-1).where(usable)
    residual=returns.subtract(returns.median(axis=1),axis=0)
    # One deterministic pair per original candidate plus every current15 pair.
    selected=pd.concat([pairs.groupby(["date","code"],sort=False).head(1),pairs[pairs.date.eq("2026-10-15")]]).drop_duplicates(["date","code","peer_code"])
    checks=0
    for r in selected.itertuples():
        window=engine["sessions"][max(0,ix[r.date]-25):max(0,ix[r.date]-5)]
        left=residual[r.code].reindex(window).to_numpy() if r.code in residual else np.full(len(window),np.nan)
        right=residual[r.peer_code].reindex(window).to_numpy() if r.peer_code in residual else np.full(len(window),np.nan)
        valid=np.isfinite(left)&np.isfinite(right)
        assert int(valid.sum())==r.common_sessions
        expected=np.nan
        if valid.sum()>=12 and np.std(left[valid])>1e-9 and np.std(right[valid])>1e-9:
            expected=np.corrcoef(left[valid],right[valid])[0,1]
        np.testing.assert_allclose(r.correlation,expected,atol=1e-11,rtol=0,equal_nan=True)
        checks+=1
    # Entire pair list, not selected positive matches: independently aggregate all masses.
    weights=pairs.assign(weight=np.maximum(pairs.correlation,0)**2)
    complete=rows.scope_complete&rows.price_relation_history_fraction.ge(.6)
    def mass(field,mask):
        known=weights[mask].groupby(["date","code"]).weight.agg(["sum","count"])
        observed=rows.merge(known,left_on=["date","code"],right_index=True,how="left")
        expected=observed["sum"].copy()
        # Empty original subgroups are a known zero; nonempty all-missing groups unknown.
        counts=weights[mask].groupby(["date","code"]).size()
        size=rows.set_index(["date","code"]).index.map(counts).fillna(0).to_numpy()
        expected.loc[size==0]=0
        expected.loc[observed["count"].eq(0)&(size>0)]=np.nan
        expected.loc[~complete]=np.nan
        # Own-only peer exclusion is already represented by the full pair ledger.
        np.testing.assert_allclose(rows[field],expected,atol=1e-11,rtol=0,equal_nan=True)
    mass("price_relation_first_observed_mass",weights.peer_pool.eq("CLOSING")&weights.peer_height.eq(1))
    mass("price_relation_relay_observed_mass",weights.peer_pool.eq("CLOSING")&weights.peer_height.ge(2))
    mass("price_relation_failed_observed_mass",weights.peer_pool.eq("FAILED"))
    historical=old.merged(joblib.load(DATA))
    joined=historical.merge(rows[["date","code"]+p.FIELDS],on=["date","code"],validate="one_to_one")
    windows=[]
    for cutoff in ["2025-07-01","2026-01-01","2026-09-01"]:
        train=old.history.training_rows(joined,cutoff,"relay")
        windows.append(dict(cutoff=cutoff,rows=len(train),dates=train.date.nunique(),
            own_return_history_at_least12=int(train.price_relation_history_fraction.ge(.6).sum()),
            own_return_history_coverage=float(train.price_relation_history_fraction.ge(.6).mean()),
            observed_closing_relation_coverage=float(train.price_relation_closing_pair_fraction.notna().mean()),
            selected_parent_correlation_coverage=float(train.price_relation_selected_parent_corr.notna().mean()),
            every_field_coverage={field:float(train[field].notna().mean()) for field in p.FIELDS}))
    assert all(w["own_return_history_coverage"]>=.95 and w["observed_closing_relation_coverage"]>=.95 for w in windows)
    # Additional already-archived author views are bound separately, never overwrite ours.
    view=b.read(CURRENT/"independent_forward_views_BEFORE_MODEL.json")
    b.verify(view["source_bindings"])
    notices=b.read(CURRENT/"notice_fulltext/manifest.json")
    notice_checks=[]
    for item in notices["rows"]:
        assert b.sha(item["pdf_path"])==item["pdf_sha256"]
        assert b.sha(item["text_path"])==item["text_sha256"]
        assert item["received_at"]<view["formed_at"]
        text=Path(item["text_path"]).read_text(encoding="utf-8-sig")
        assert item["code"] in text
        notice_checks.append(dict(code=item["code"],pdf_sha256=item["pdf_sha256"],text_sha256=item["text_sha256"],
            received_at=item["received_at"],date_semantics="late actual receipt, precise first publication unknown",
            source_scope=item["source_scope"]))
    original_table=b.ROOT/"research/model_upgrade_validation/exact_attempt_response_20261015T100800Z/delivery_final/完整81只对照_含更正炸板输入研究分.csv"
    table=pd.read_csv(original_table,dtype={"code":str},low_memory=False)
    current=rows[rows.date.eq("2026-10-15")]
    current.to_csv(ROOT/"current6_price_relations_NO_NEW_SCORE.csv",index=False,encoding="utf-8-sig")
    view_rows=pd.DataFrame(view["rows"])
    view_rows.to_json(ROOT/"additional_current6_independent_views_BOUND_NOT_REWRITTEN.json",orient="records",force_ascii=False,indent=2)
    result=dict(formed_at=b.now(),status="PASS_SOURCE_ONLY_QUALIFICATION_WITH_EXPLICIT_MISSINGNESS",
        version=p.VERSION,policy=p.POLICY,total_historical_rows=9245,total_with_October=9315,
        all_pool_parent_pairs=len(pairs),independent_pair_checks=checks,all_rows_three_mass_aggregations_checked=len(rows)*3,
        training_windows=windows,original_candidate_failed_source_conflicts=int(rows.candidate_failed_source_conflict.sum()),
        current_full6_observed12=int(current.price_relation_history_fraction.ge(.6).sum()),
        current_full6_missing_relation=int(current.price_relation_history_fraction.lt(.6).sum()),
        current6_full_closing_pair_fraction_range=[float(current.price_relation_closing_pair_fraction.min()),float(current.price_relation_closing_pair_fraction.max())],
        current_missing_cannot_be_filled_by_name_or_future=True,
        preexisting_81_table_unchanged_rows=len(table),additional_author_record=view["formed_at"],
        author_exposure=view["model_exposure"],blinding_independently_verified=False,
        additional_views=[{k:r[k] for k in ["code","name","promotion_judgement","relative_watch_rank","lean"]} for r in view["rows"]],
        notice_verification=notice_checks,protected_files_exact=len(protected),new_fits=0,
        independent_identity_gold=0,active_business_theme_certified=False,no_predictive_gain_claim=True,
        source_bindings=protected|context["sources"]|{str(p.resolve()):b.sha(p) for p in [Path(__file__),CONTEXT,DATA,ROOT/"current6_price_relations_NO_NEW_SCORE.csv"]})
    b.write_new(ROOT/"source_qualification_BEFORE_NEW_FIT.json",result)
    print({k:result[k] for k in ["status","total_historical_rows","all_pool_parent_pairs","independent_pair_checks","new_fits"]})


if __name__=="__main__":main()
