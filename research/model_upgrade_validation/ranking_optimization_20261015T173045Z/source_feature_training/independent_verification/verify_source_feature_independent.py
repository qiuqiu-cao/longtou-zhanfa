"""Independent source-feature audit: replay saved models, never fit or retune."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

WORK = Path(__file__).resolve().parent.parent
BATCH = WORK.parent
REPO = BATCH.parents[2]
sys.path.insert(0, str(REPO))
OUT = WORK / "independent_verification"
OLD = REPO / "research/model_upgrade_validation/expectation_345_20261015T163226Z"
ARM = "SOURCE_PATCHED_FEATURE_CLASSIFIER"
FIELD = "causal_price_shape"
PERIODS = [("2025H2", "2025-07-01", "2025-12-31"),
    ("2026JanAug", "2026-01-01", "2026-08-31"),
    ("2026Sep", "2026-09-01", "2026-09-30"),
    ("2026Oct", "2026-10-01", "2026-10-31"),
    ("HISTORICAL_POOLED", "2025-07-01", "2026-09-30")]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def csv(path): return pd.read_csv(path, dtype={"code": str}, low_memory=False)


def equal_cells(a, z, keys, columns):
    assert not a.duplicated(keys).any() and not z.duplicated(keys).any()
    a, z = a.set_index(keys).sort_index(), z.set_index(keys).sort_index()
    assert a.index.equals(z.index)
    for field in columns:
        mask = a[field].eq(z[field]) | (a[field].isna() & z[field].isna())
        assert mask.all(), (field, int((~mask).sum()))
    return len(a) * len(columns)


def frame_digest(f, fields):
    columns = list(dict.fromkeys(["date", "target_date", "target", "label_valid", "code"] + fields))
    return hashlib.sha256(f[columns].to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def classify(f, call_field="evaluation_call"):
    known = f[f.target.notna() & f.label_valid.fillna(False).astype(bool)]
    y = known.target.eq(1)
    positive = known[call_field].eq("PROMOTE")
    negative = known[call_field].eq("FAIL")
    tp, fp, tn, fn = [int(x.sum()) for x in [y & positive, ~y & positive, ~y & negative, y & negative]]
    positives = int(y.sum())
    negatives = len(known)-positives
    missed = positives-tp
    called = int((positive | negative).sum())
    return dict(candidates=len(f), known=len(known), unknown_outcomes=len(f)-len(known), positives=positives,
        positive_calls=int(f[call_field].eq("PROMOTE").sum()), negative_calls=int(f[call_field].eq("FAIL").sum()),
        abstentions=int(f[call_field].eq("UNKNOWN").sum()), tp=tp, fp=fp, fn_called=fn, tn=tn,
        abstained_positives=int((y & ~positive & ~negative).sum()),
        abstained_negatives=int((~y & ~positive & ~negative).sum()), missed_positives=missed,
        Recall=tp/positives if positives else None, Precision=tp/(tp+fp) if tp+fp else None,
        F1=2*tp/(2*tp+fp+missed) if 2*tp+fp+missed else None,
        false_positive_rate=fp/negatives if negatives else None,
        decision_coverage=called/len(known) if len(known) else None,
        full_known_accuracy=(tp+tn)/len(known) if len(known) else None,
        called_accuracy=(tp+tn)/called if called else None,
        utility_TP_minus_1p5FP=tp-1.5*fp)


def full_slot_rank(f, k):
    # Only identity, as-of score and tie code participate in selection.
    eligible = f.loc[f.daily_one_word.eq(0), ["date", "code", "ranking_score"]]
    selected = eligible.sort_values(["date", "ranking_score", "code"], ascending=[True, False, True], na_position="last").groupby("date", sort=True).head(k)
    selected = selected.merge(f[["date", "code", "target", "label_valid", "evaluation_call"]], on=["date", "code"], validate="one_to_one")
    valid = selected.target.notna() & selected.label_valid.fillna(False).astype(bool)
    hits = int((valid & selected.target.eq(1)).sum())
    slots = k*f.date.nunique()
    return dict(k=k, signal_days=f.date.nunique(), full_slots=slots, selected=len(selected),
        empty_slots=slots-len(selected), known_selected=int(valid.sum()), unknown_selected=int((~valid).sum()),
        hits=hits, full_slot_hit_rate=hits/slots if slots else None,
        unknown_direction_selected=int(selected.evaluation_call.eq("UNKNOWN").sum())), selected


def write_new(path, obj):
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, f):
    if Path(path).exists(): raise FileExistsError(path)
    f.to_csv(path, index=False, encoding="utf-8-sig")


def training_audit():
    OUT.mkdir(exist_ok=True)
    result_path = WORK / "execution/fixed_source_feature_result.json"
    result = read(result_path)
    bound = dict(result["source_bindings"])
    for path, wanted in bound.items(): assert sha(path) == wanted, path
    protocol = read(WORK / "protocol_BEFORE_ANY_FIT.json")
    original = joblib.load(OLD / "trainable_corrected_dataset_QUALIFIED.joblib")
    patched = joblib.load(WORK / "prepared.joblib")
    a, z, fields = original["full"], patched["full"], original["fields"]
    assert fields == patched["fields"] and len(fields) == 92 and len(a) == len(z) == 9315
    pd.testing.assert_frame_equal(a.drop(columns=FIELD), z.drop(columns=FIELD), check_exact=True)
    differs = ~(a[FIELD].eq(z[FIELD]) | (a[FIELD].isna() & z[FIELD].isna()))
    assert int(differs.sum()) == 47 and a.loc[differs, FIELD].isna().all()
    assert z.loc[differs, FIELD].isin([0., 1.]).all()
    changed = z.loc[differs, ["date", "code", FIELD]].copy()
    saved_changes = csv(WORK / "all47_single_field_changes_BEFORE_ANY_FIT.csv").rename(columns={"new_value": FIELD})
    equal_cells(changed, saved_changes, ["date", "code"], [FIELD])
    # Late source availability never enters the target or member-selection logic.
    old_protocol = read(OLD / "protocol_BEFORE_ANY_FIT.json")
    original_checks = {c["month"]: c for c in old_protocol["training_checks"]}
    windows, replayed = [], []
    new_fits = reuses = 0
    successor = dict(zip(patched["sessions"][:-1], patched["sessions"][1:]))
    with patch.object(HistGradientBoostingClassifier, "fit", side_effect=AssertionError("independent audit cannot fit")):
        for check in protocol["training_checks"]:
            cutoff, month = check["cutoff"], check["month"]
            train = []
            for full in (a, z):
                known = full[full.target.notna() & full.label_valid.fillna(False).astype(bool) & full.date.lt(cutoff) & full.target_date.lt(cutoff)]
                dates = sorted(known.date.unique())[-252:]
                assert len(dates) == 252
                train.append(known[known.date.isin(dates)].copy())
            old_train, new_train = train
            pd.testing.assert_frame_equal(old_train.drop(columns=FIELD), new_train.drop(columns=FIELD), check_exact=True)
            assert new_train.target_date.lt(cutoff).all() and new_train.target_date.eq(new_train.date.map(successor)).all()
            old_digest, new_digest = frame_digest(old_train, fields), frame_digest(new_train, fields)
            assert old_digest == check["old_digest"] == original_checks[month]["training_digest"]
            assert new_digest == check["new_digest"]
            changed_count = int((~(old_train[FIELD].eq(new_train[FIELD]) | (old_train[FIELD].isna() & new_train[FIELD].isna()))).sum())
            assert changed_count == check["changed_training_cells"]
            reuse = old_digest == new_digest
            assert reuse == check["reuse_original_model"] == (changed_count == 0)
            # Independently reconstruct fixed cost1 weights: inverse day density,
            # equal positive/negative class mass, normalized to252.
            w = 1. / new_train.groupby("date").date.transform("size")
            class_mass = w.groupby(new_train.target).sum()
            w = w / new_train.target.map(class_mass)
            w = w / w.sum() * 252.
            assert abs(w.sum()-252.) < 1e-10
            assert abs(w[new_train.target.eq(0)].sum()-126.) < 1e-10
            assert abs(w[new_train.target.eq(1)].sum()-126.) < 1e-10
            end_path = WORK / f"execution/{month}_COMPLETED.json"
            end = read(end_path)
            start = read(WORK / f"execution/{month}_STARTED.json")
            assert start["attempt"] == 1 and start["plan"] == check
            assert end["new_fit"] == (not reuse) and end["training_digest"] == new_digest
            assert sha(check["original_model_path"]) == check["original_model_sha256"]
            model_path = Path(end["model_path"])
            assert sha(model_path) == end["model_sha256"]
            if reuse:
                assert model_path.resolve() == Path(check["original_model_path"]).resolve()
                assert end["model_sha256"] == check["original_model_sha256"]
                assert not (WORK / f"execution/{month}_model.joblib").exists()
                reuses += 1
            else:
                assert model_path.resolve() == (WORK / f"execution/{month}_model.joblib").resolve()
                new_fits += 1
            model = joblib.load(model_path)
            assert model["features"] == fields and model["cutoff"] == cutoff
            assert model["training_digest"] == new_digest and model["negative_cost"] == 1.
            assert model["estimator"] == protocol["estimator"]
            estimator = model["model"].named_steps["hgb"]
            for key, value in protocol["estimator"].items(): assert estimator.get_params()[key] == value
            test = z[z.date.str[:7].eq(month)]
            with threadpool_limits(limits=2):
                scores = model["model"].predict_proba(test[fields])[:, list(model["model"].classes_).index(1)]
            saved_path = WORK / f"execution/{month}_predictions_BEFORE_EVALUATION.csv"
            assert sha(saved_path) == end["prediction_sha256"]
            saved = csv(saved_path)
            comparable = test[["date", "code"]].copy()
            comparable["replayed_b_score"] = scores
            comparable = comparable.merge(saved[["date", "code", "b_score"]], on=["date", "code"], validate="one_to_one")
            comparable["absolute_difference"] = (comparable.replayed_b_score-comparable.b_score).abs()
            assert len(comparable) == len(test) == end["rows"]
            assert comparable.absolute_difference.le(1e-12).all()
            comparable["month"] = month
            replayed.append(comparable)
            windows.append(dict(month=month, window_days=len(dates), rows=len(new_train),
                latest_signal=new_train.date.max(), latest_mature_target=new_train.target_date.max(),
                changed_training_cells=changed_count, identical_other91_features_members_labels=True,
                old_digest=old_digest, new_digest=new_digest, reused=reuse,
                weight_mass=float(w.sum()), negative_weight_mass=float(w[new_train.target.eq(0)].sum()),
                positive_weight_mass=float(w[new_train.target.eq(1)].sum()), model_sha256=end["model_sha256"],
                original_model_hash_unchanged=True, prediction_rows=len(comparable),
                maximum_absolute_score_difference=float(comparable.absolute_difference.max())))
    assert new_fits == 9 and reuses == 7 and len(windows) == 16
    predictions = csv(WORK / "execution/complete18885_predictions_BEFORE_EVALUATION.csv")
    frozen = csv(WORK / "frozen_source_qualified_references.csv")
    assert len(predictions) == 18885 and predictions.arm.nunique() == 5
    original_qualified_path = BATCH / "source_repair/verified_22662_final/all_predictions_with_isolated_qualification.csv"
    original_qualified = csv(original_qualified_path)
    source = original_qualified[original_qualified.arm.isin(frozen.arm.unique())]
    equal_cells(frozen, source, ["date", "code", "arm"], [c for c in source if c not in ["date", "code", "arm"]])
    old_reference_cells = equal_cells(frozen, predictions[predictions.arm.ne(ARM)], ["date", "code", "arm"], [c for c in frozen if c not in ["date", "code", "arm"]])
    references = frozen[frozen.arm.eq("CORRECTED_CAUSAL")]
    fresh = predictions[predictions.arm.eq(ARM)]
    assert len(fresh) == 3777 and not fresh.duplicated(["date", "code"]).any()
    equal_cells(references, fresh, ["date", "code"], ["target", "target_date", "label_valid", "daily_one_word", "current_price_shape"])
    assert references[references.date.str.startswith("2026-09")].daily_one_word.eq(0).sum() == 143
    assert references[references.date.str.startswith("2026-09")].engine_metadata_daily_one_word.eq(0).sum() == 132
    future = fresh[fresh.date.eq("2026-10-15")]
    assert len(future) == 6 and future.target.isna().all() and not future.label_valid.any()
    # Audit the historical raw route defect without covering up the saved record.
    route = references.merge(fresh, on=["date", "code"], suffixes=("_ref", "_new"), validate="one_to_one")
    keep_reference = ~route.verified_current_price_shape_new.eq(0)
    route_mismatch = keep_reference & route.evaluation_call_ref.ne(route.evaluation_call_new)
    defects = route[route_mismatch].copy()
    outputs = {
        "16训练窗单字段变化与精确复用.csv": pd.DataFrame(windows),
        "全部3777新增头原始分数独立重放.csv": pd.concat(replayed, ignore_index=True),
        "47输入单字段变化独立核对.csv": changed,
        "原执行附件A资格差异_保留原失败.csv": defects,
    }
    for name, frame in outputs.items(): write_csv(OUT/name, frame)
    for path, wanted in bound.items(): assert sha(path) == wanted, path
    receipt = dict(formed_at=datetime.now(timezone.utc).isoformat(), training_and_numeric_invariants_pass=True,
        raw_execution_route_qualification_equivalent=len(defects)==0, raw_execution_route_differences=len(defects),
        raw_execution_route_new_positive_from_unknown=int((defects.evaluation_call_ref.eq("UNKNOWN") & defects.evaluation_call_new.eq("PROMOTE")).sum()),
        raw_execution_route_issue="New verified shape1 A calls were restored whereas comparator kept original UNKNOWN; requires appended common-rule route layer, never refit or overwrite",
        new_fits_by_auditor=0, replayed_months=16, experiment_new_fits=9, exact_original_model_reuses=7,
        changed_feature=FIELD, changed_cells=47, other91_and_all_metadata_labels_members_exactly_unchanged=True,
        replayed_candidates=3777, original_reference_rows_unchanged=15108, original_reference_cells_unchanged=old_reference_cells,
        all16_windows_252_mature_days=True, all_training_class_weights_126_each=True,
        original_September_nonword_denominator=143, latest_engine_September_nonword_denominator_not_used=132,
        future_current15_candidates=6, future_matured=0, calibrated_probabilities=False,
        maximum_absolute_raw_score_difference=max(x["maximum_absolute_score_difference"] for x in windows),
        frozen_sources_verified_before_and_after=len(bound), original_result_screen_pass=result["screen_pass"],
        no_training_or_evaluation_function_from_research_called=True,
        script_sha256=sha(__file__),
        source_bindings={str(p.resolve()):sha(p) for p in [Path(__file__), result_path, WORK/"protocol_BEFORE_ANY_FIT.json", WORK/"prepared.joblib", original_qualified_path, *[OUT/n for n in outputs]]})
    write_new(OUT/"训练与原始分数独立核验收据.json", receipt)
    print(json.dumps({k:receipt[k] for k in ["training_and_numeric_invariants_pass", "experiment_new_fits", "exact_original_model_reuses", "changed_cells", "replayed_candidates", "maximum_absolute_raw_score_difference", "raw_execution_route_differences"]}))


def final_audit(final_path):
    final_path = Path(final_path).resolve()
    final = csv(final_path)
    raw_original = csv(BATCH/"fixed_comparison/all22662_predictions_BEFORE_EVALUATION.csv")
    assert len(final) == 26439 and final.arm.nunique() == 7 and not final.duplicated(["date", "code", "arm"]).any()
    old = final[final.arm.ne(ARM)]
    old_cells = equal_cells(raw_original, old, ["date", "code", "arm"], [c for c in raw_original if c not in ["date", "code", "arm"]])
    qualified = csv(BATCH/"source_repair/verified_22662_final/all_predictions_with_isolated_qualification.csv")
    equal_cells(qualified, old, ["date", "code", "arm"], [c for c in qualified if c not in ["date", "code", "arm"]])
    ref = final[final.arm.eq("CORRECTED_CAUSAL")]
    new = final[final.arm.eq(ARM)]
    assert len(new) == 3777
    # Wrapper may attach its final event fields separately from frozen raw fields.
    event_field = "evaluation_call" if "evaluation_call" in new else "recovered_B_call"
    ref_event = "evaluation_call" if "evaluation_call" in ref and ref.evaluation_call.notna().all() else "recovered_B_call"
    new = new.copy();ref = ref.copy()
    new["evaluation_call"] = new[event_field]
    ref["evaluation_call"] = ref[ref_event]
    routes = ref.merge(new, on=["date", "code"], suffixes=("_ref", "_new"), validate="one_to_one")
    shape = routes.verified_current_price_shape_new
    zero = shape.eq(0)
    expected_b = np.where(routes.b_score_new.gt(.5), "PROMOTE", "FAIL")
    assert routes.loc[zero, "evaluation_call_new"].eq(expected_b[zero]).all()
    assert routes.loc[~zero, "evaluation_call_new"].eq(routes.loc[~zero, "evaluation_call_ref"]).all()
    score_equal = routes.score_new.eq(routes.score_ref) | (routes.score_new.isna() & routes.score_ref.isna())
    assert score_equal[~zero].all()
    raw_new = csv(WORK/"execution/complete18885_predictions_BEFORE_EVALUATION.csv")
    raw_new = raw_new[raw_new.arm.eq(ARM)]
    equal_cells(raw_new, new, ["date", "code"], ["b_score", "ranking_score", "target", "target_date", "label_valid", "daily_one_word"])
    assert new[new.date.eq("2026-10-15")].target.isna().all()
    assert new[new.date.str.startswith("2026-09")].daily_one_word.eq(0).sum() == 143
    records, ranks, selections = [], [], []
    # Only absolute event heads are graded. Pairwise remains ranking-only.
    grade_arms = ["BASE", "ST_DUAL", "ST_DUAL_LAGGED_PRICE_RELATION", "CORRECTED_CAUSAL", ARM]
    for period, first, last in PERIODS:
        for arm in grade_arms:
            f = final[final.arm.eq(arm) & final.date.between(first,last)].copy()
            if arm == ARM: f["evaluation_call"] = new.set_index(["date","code"]).loc[pd.MultiIndex.from_frame(f[["date","code"]]), "evaluation_call"].to_numpy()
            else: f["evaluation_call"] = f.recovered_B_call
            for scope, part in [("FULL_POOL",f),("ORIGINAL_NONWORD",f[f.daily_one_word.eq(0)])]:
                records.append(dict(period=period,arm=arm,scope=scope,**classify(part)))
            for k in (1,2,3):
                summary, picks = full_slot_rank(f,k)
                ranks.append(dict(period=period,arm=arm,**summary))
                picks["period"]=period;picks["arm"]=arm;picks["k"]=k;selections.append(picks)
    stats=pd.DataFrame(records);ranking=pd.DataFrame(ranks)
    for name, frame in [("最终同资格配对指标独立复算.csv",stats),("最终Top1Top2Top3固定全日槽独立复算.csv",ranking),("最终全部预选_先选择后结算.csv",pd.concat(selections,ignore_index=True))]: write_csv(OUT/name,frame)
    receipt=dict(formed_at=datetime.now(timezone.utc).isoformat(),passed=True,new_fits=0,
        final_rows=26439,candidates_per_arm=3777,arms=7,original22662_all_fields_unchanged=True,original_cells_unchanged=old_cells,
        final_new_B_verified_shape0_strict_threshold=True,all_other_routes_inherit_qualified_old_classifier=True,
        original47_input_patch_and16_model_replay_receipt=str((OUT/"训练与原始分数独立核验收据.json").resolve()),
        raw_scores_unchanged_by_route_wrapper=True,original_September_nonword_denominator=143,current15_six_future_unmatured=True,
        classifier_metric_rows=len(stats),ranking_metric_rows=len(ranking),scores_uncalibrated=True,
        script_sha256=sha(__file__),source_bindings={str(p.resolve()):sha(p) for p in [Path(__file__),final_path,OUT/"训练与原始分数独立核验收据.json",OUT/"最终同资格配对指标独立复算.csv",OUT/"最终Top1Top2Top3固定全日槽独立复算.csv",OUT/"最终全部预选_先选择后结算.csv"]})
    write_new(OUT/"最终同资格独立核验收据.json",receipt)
    print(json.dumps({k:receipt[k] for k in ["passed","final_rows","original22662_all_fields_unchanged","classifier_metric_rows","ranking_metric_rows"]}))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["training","final"]);parser.add_argument("--final-predictions")
    args=parser.parse_args()
    training_audit() if args.mode=="training" else final_audit(args.final_predictions)
