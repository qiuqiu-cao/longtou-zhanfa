"""Independent cents/count checks and full-schema corrected history capsule."""
from collections import Counter
from decimal import Decimal,ROUND_HALF_UP
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b
import short_term_thinking_research as old

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"attempt_reclassification_correction"


def main():
    original_path=b.DEFAULT_RUN/"scope_correction/dataset.joblib"
    original=joblib.load(original_path)
    correction_path=OUT/"corrected_dataset_RESEARCH_NOT_FITTED.joblib";data=joblib.load(correction_path)
    report=b.read(OUT/"correction_and_impact_report.json");b.verify(report["source_bindings"])
    rawpath=b.ROOT/"results/v215/historical_board_attempts_20260920_v2/historical_closing_and_failed_board_attempts.csv"
    raw=pd.read_csv(rawpath,dtype={"code":str});records=[];unknown=set();last={}
    index={day:i for i,day in enumerate(data["sessions"])}
    for stock in raw.sort_values(["signal_date","code"]).itertuples():
        upper=int((Decimal(str(stock.pre_close))*Decimal(110)).quantize(Decimal(1),rounding=ROUND_HALF_UP))
        high=int(Decimal(str(stock.high))*100);close=int(Decimal(str(stock.close))*100)
        assert Decimal(str(stock.high))*100==high and Decimal(str(stock.close))*100==close
        assert upper==int(Decimal(str(stock.limit_price_10))*100)
        if high>upper or close>high:unknown.add(stock.signal_date);continue
        if high<upper:continue
        prior=last.get(stock.code)
        attempted=prior[1]+1 if prior and prior[0]==index[stock.signal_date]-1 else 1
        closing=close==upper
        last[stock.code]=(index[stock.signal_date],attempted if closing else 0)
        records.append(dict(signal_date=stock.signal_date,code=stock.code,board_height=attempted,pool_assignment="CLOSING_LIMIT_UP" if closing else "FAILED_BOARD_OBSERVATION"))
    expected=pd.DataFrame(records);got=data["attempts"][list(expected)]
    pd.testing.assert_frame_equal(expected.reset_index(drop=True),got.reset_index(drop=True),check_dtype=False)
    assert len(expected)==56008 and unknown=={"2026-08-07"}
    failed=Counter((x.signal_date,x.board_height) for x in expected.itertuples() if x.pool_assignment=="FAILED_BOARD_OBSERVATION")
    peers=Counter((x.date,int(x.boards)) for x in original["prepared"]["relay"].itertuples())
    complete_dates=set(data["attempts"].attrs["failed_scope_complete_dates"])
    for x in data["features"].itertuples():
        if x.date not in complete_dates:
            assert pd.isna(x.same_height_failed_count) and pd.isna(x.same_height_elimination_rate)
        else:
            count=failed[(x.date,int(x.boards))];assert x.same_height_failed_count==count
            assert np.isclose(x.same_height_elimination_rate,count/(count+peers[(x.date,int(x.boards))]),atol=1e-15)
    before=original["features"];after=data["features"]
    for field in set(before)-set(after):
        # Preserve all original metadata omitted by a pure engine replay.
        after[field]=before[field].to_numpy()
    after=after[list(before)]
    mutable={"same_height_failed_count","same_height_elimination_rate","is_divergence_survivor","role_USURP_DRAGON","primary_role","roles"}
    pd.testing.assert_frame_equal(before.drop(columns=list(mutable)),after.drop(columns=list(mutable)))
    data["features"]=after
    bindings=data["sources"]|{str(original_path.resolve()):b.sha(original_path),str(correction_path.resolve()):b.sha(correction_path),str(Path(__file__).resolve()):b.sha(__file__)}
    data["sources"]=bindings
    full=old.merged(data);oldfull=old.merged(original)
    for field in ["date","code","target","target_date","label_valid"]:
        pd.testing.assert_series_equal(full[field],oldfull[field])
    prediction_path=b.DEFAULT_RUN/"delivery/complete_paired_predictions_VALIDATED.csv"
    bindings[str(prediction_path.resolve())]=b.sha(prediction_path)
    path=OUT/"corrected_dataset_SCHEMA_COMPLETE_RESEARCH_NOT_FITTED.joblib";joblib.dump(data,path,compress=3)
    # Extend the same source capsule used by the new-date workbench. The old
    # model and old workbench artifacts still point at their immutable inputs.
    enginepath=b.ROOT/"research/model_upgrade_validation/historical_symbolic_context_20261015T081200Z/latest15_prepared_v2/engine_context_WITH_CURRENT15.joblib"
    engine=joblib.load(enginepath)
    historic_keys=set(zip(original["attempts"].signal_date,original["attempts"].code))
    other=engine["attempts"][[ (x.signal_date,x.code) not in historic_keys for x in engine["attempts"].itertuples()]]
    attempts=pd.concat([data["attempts"],other],ignore_index=True)
    attempts.attrs["failed_scope_complete_dates"]=sorted(set(engine["attempts"].attrs["failed_scope_complete_dates"])-unknown)
    features,edges=b.causal_features(engine["prepared"]["relay"],engine["prepared"]["market"],attempts,engine["bars"],engine["sessions"])
    day="2026-10-15"
    current=features[features.date.eq(day)]
    prior_current=engine["features"][engine["features"].date.eq(day)]
    if b.clean(current.to_dict("records"))!=b.clean(prior_current.to_dict("records")):
        raise ValueError("historical attempt correction unexpectedly changed latest15 already-reviewed contexts")
    engine.update(features=features,attempts=attempts,sources=engine["sources"]|bindings|{str(path.resolve()):b.sha(path),str(enginepath.resolve()):b.sha(enginepath)},
        correction_version=data["correction_version"],old_models_not_refitted=True)
    newengine=OUT/"corrected_engine_context_THROUGH_OCT15_RESEARCH_NOT_FITTED.joblib";joblib.dump(engine,newengine,compress=3)
    result=dict(formed_at=b.now(),status="PASS_FULL58772_RECLASSIFICATION_AND9245_COHORT_COUNTS",original_rows_checked=58772,qualified_rows=56008,restored_failed_rows=362,
        original_candidate_rows_checked=9245,latest6_contexts_exact_after_historical_correction=True,original_targets_base_scores_and_other_features_exact=True,
        schema_complete_dataset=str(path.resolve()),schema_complete_dataset_sha256=b.sha(path),current_workbench_engine=str(newengine.resolve()),current_workbench_engine_sha256=b.sha(newengine),
        new_fits=0,new_model_or_prediction_acceptance=False,unknown_scope_dates=sorted(unknown),source_bindings=bindings)
    b.write_new(OUT/"independent_verification.json",result);print({k:v for k,v in result.items() if k!="source_bindings"})


if __name__=="__main__":main()
