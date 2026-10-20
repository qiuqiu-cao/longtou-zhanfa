"""Independent saved-model replay and complete-cohort accounting. Never fits."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

BATCH = Path(__file__).resolve().parent.parent
REPO = BATCH.parents[2]
sys.path.insert(0, str(REPO))
OUT = BATCH / "verification"
ARM = "PAIRWISE_DAY_RANK"
PERIODS = [
    ("2025H2", "2025-07-01", "2025-12-31"),
    ("2026JanAug", "2026-01-01", "2026-08-31"),
    ("2026Sep", "2026-09-01", "2026-09-30"),
    ("2026Oct", "2026-10-01", "2026-10-31"),
    ("HISTORICAL_POOLED", "2025-07-01", "2026-09-30"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    return pd.read_csv(path, dtype={"code": str}, low_memory=False)


def compare_frame(left, right, keys, columns):
    """Exact cell equality including missingness, after deterministic key alignment."""
    assert not left.duplicated(keys).any() and not right.duplicated(keys).any()
    a = left.set_index(keys).sort_index()
    z = right.set_index(keys).sort_index()
    assert a.index.equals(z.index), "different candidate keys"
    checks = []
    for field in columns:
        same = a[field].eq(z[field]) | (a[field].isna() & z[field].isna())
        checks.append({"field": field, "cells": len(a), "different": int((~same).sum())})
    assert not any(row["different"] for row in checks), checks
    return checks


def select_before_truth(frame, k, same_height=False, height_baseline=False):
    """Outcome columns are deliberately unavailable to the selection calculation."""
    scope = frame.loc[frame.daily_one_word.eq(0), ["date", "code", "boards", "ranking_score"]].copy()
    group = ["date", "boards"] if same_height else ["date"]
    sort = group.copy()
    ascending = [True] * len(group)
    if height_baseline and not same_height:
        sort.append("boards")
        ascending.append(False)
    if not height_baseline:
        sort.append("ranking_score")
        ascending.append(False)
    sort.append("code")
    ascending.append(True)
    choices = scope.sort_values(sort, ascending=ascending, na_position="last").groupby(group, sort=True).head(k)
    sizes = scope.groupby(group).size()
    # Join actual labels only after all choices have been made.
    choices = choices.merge(frame[["date", "code", "target", "label_valid", "call"]], on=["date", "code"], validate="one_to_one")
    valid = choices.target.notna() & choices.label_valid.fillna(False).astype(bool)
    hits = int((valid & choices.target.eq(1)).sum())
    count, known = len(choices), int(valid.sum())
    unknown = count - known
    summary = dict(selected=count, known=known, unknown=unknown, hits=hits,
        false_selected=known-hits, hits_per_all_slots=hits/count if count else None,
        hits_per_known_slots=hits/known if known else None,
        unknown_best_case=(hits+unknown)/count if count else None,
        known_coverage=known/count if count else None,
        selected_unknown_direction=int(choices.call.eq("UNKNOWN").sum()),
        selected_missing_score=int(choices.ranking_score.isna().sum()),
        groups_fewer_than_k=int(sizes.lt(k).sum()))
    return summary, choices


def independently_replay(model, group, fields):
    group = group.sort_values("code")
    n = len(group)
    if n == 1:
        return pd.DataFrame([dict(date=group.iloc[0].date, code=group.iloc[0].code,
            ranking_score=np.nan, opponent_count=0, pairwise_antisymmetry_error=0.,
            ranking_status="SINGLETON_NO_COMPARATIVE_SCORE")]), 0
    values = group[fields].to_numpy(dtype=float)
    ordered_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    left = np.stack([values[i] for i, _ in ordered_pairs])
    right = np.stack([values[j] for _, j in ordered_pairs])
    transformed = np.hstack((left, right, left-right))
    with threadpool_limits(limits=2):
        p = model.predict_proba(transformed)[:, list(model.classes_).index(1)]
    matrix = np.full((n, n), np.nan)
    for (i, j), value in zip(ordered_pairs, p):
        matrix[i, j] = value
    preferences = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i != j:
                preferences[i, j] = (matrix[i, j] + 1. - matrix[j, i]) / 2.
    error = float(np.nanmax(np.abs(preferences + preferences.T - 1.)))
    records = []
    for i, row in enumerate(group.itertuples()):
        records.append(dict(date=row.date, code=row.code,
            ranking_score=float(np.nanmean(preferences[i])), opponent_count=n-1,
            pairwise_antisymmetry_error=error,
            ranking_status="UNVALIDATED_COHORT_RELATIVE_PREFERENCE"))
    return pd.DataFrame(records), len(ordered_pairs)


def independently_count_groups(complete, month):
    rows = []
    for day, f in complete.groupby("date", sort=True):
        known = f[f.target.notna() & f.label_valid.fillna(False).astype(bool)]
        pos = int(known.target.eq(1).sum())
        neg = int(known.target.eq(0).sum())
        pairs = 2 * pos * neg
        status = "MIXED_KNOWN_LABELS" if pairs else "NO_KNOWN_ROWS" if not len(known) else "SINGLE_KNOWN_CANDIDATE" if len(known) == 1 else "ALL_KNOWN_POSITIVE" if pos == len(known) else "ALL_KNOWN_NEGATIVE"
        weights = np.full(pairs, 1./pairs) if pairs else np.empty(0)
        mass = float(weights.sum())
        assert not pairs or abs(mass-1.) <= 1e-12
        assert not pairs or abs(weights[:pairs//2].sum()-.5) <= 1e-12
        rows.append(dict(date=day, complete_nonword_candidates=len(f), known=len(known),
            unknown_labels=len(f)-len(known), positives=pos, negatives=neg,
            directed_pairs=pairs, training_group_status=status, month=month,
            independently_computed_day_weight=mass))
    return pd.DataFrame(rows)


def digest(frame, fields):
    columns = list(dict.fromkeys(["date", "target_date", "target", "label_valid", "code"] + fields))
    return hashlib.sha256(frame[columns].to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    receipt_path = OUT / "独立核验收据.json"
    if receipt_path.exists():
        raise FileExistsError("independent verification already completed")
    protocol_path = BATCH / "protocol_BEFORE_ANY_FIT.json"
    protocol = load_json(protocol_path)
    result_path = BATCH / "fixed_comparison/ranking_validation_results.json"
    result = load_json(result_path)
    expected_sources = result["sources"]
    for path, wanted in expected_sources.items():
        assert sha(path) == wanted, ("bound source changed", path)
    original_bound = dict(expected_sources)
    data = joblib.load(BATCH / "fixed_input_before_fit.joblib")
    full, fields = data["full"], data["fields"]
    assert len(fields) == 92 and len(full) == 9315
    pred = read_csv(BATCH / "fixed_comparison/all22662_predictions_BEFORE_EVALUATION.csv")
    references = read_csv(BATCH / "all18885_frozen_reference_predictions.csv")
    source_reference = REPO / "research/model_upgrade_validation/expectation_345_20261015T163226Z/evaluation_scope_alignment/complete18885_predictions_ORIGINAL_FROZEN_COHORT.csv"
    compare_frame(references, read_csv(source_reference), ["date", "code", "arm"], [c for c in references if c not in ["date", "code", "arm"]])
    assert len(pred) == 22662 and not pred.duplicated(["date", "code", "arm"]).any()
    assert len(references) == 18885 and pred.arm.nunique() == 6
    frozen_cell_checks = compare_frame(references, pred[pred.arm.ne(ARM)], ["date", "code", "arm"], [c for c in references if c not in ["date", "code", "arm"]])
    base_keys = references[references.arm.eq("BASE")][["date", "code"]].set_index(["date", "code"]).sort_index().index
    assert len(base_keys) == 3777
    for _, arm_rows in pred.groupby("arm"):
        assert arm_rows.set_index(["date", "code"]).sort_index().index.equals(base_keys)
    compare_frame(pred[pred.arm.eq("BASE")], pred[pred.arm.eq(ARM)], ["date", "code"], ["target", "label_valid", "daily_one_word", "boards", "target_date"])
    assert pred[pred.arm.eq(ARM)].call.eq("UNKNOWN").all() and pred[pred.arm.eq(ARM)].score.isna().all()
    assert pred[pred.date.eq("2026-10-15")].target.isna().all()

    replay_rows, windows, dispositions = [], [], []
    sessions = list(data["sessions"])
    successor = dict(zip(sessions[:-1], sessions[1:]))
    # A process-level guard makes accidental fitting impossible in this audit.
    with patch.object(HistGradientBoostingClassifier, "fit", side_effect=AssertionError("verification must never fit")):
        for check in protocol["training_checks"]:
            month, cutoff = check["month"], check["cutoff"]
            eligible = full[full.target.notna() & full.label_valid.fillna(False).astype(bool) & full.date.lt(cutoff) & full.target_date.lt(cutoff)]
            dates = sorted(eligible.date.unique())[-252:]
            assert len(dates) == 252
            train = eligible[eligible.date.isin(dates)]
            complete = full[full.date.isin(dates) & full.daily_one_word.eq(0)].copy()
            complete.loc[complete.target_date.ge(cutoff), ["target", "label_valid"]] = [np.nan, False]
            known = complete[complete.target.notna() & complete.label_valid.fillna(False).astype(bool)]
            assert known.target_date.lt(cutoff).all() and known.date.lt(cutoff).all()
            assert known.target_date.eq(known.date.map(successor)).all(), "nonadjacent mature label"
            assert digest(complete, fields) == check["input_digest"]
            groups = independently_count_groups(complete, month)
            saved_groups = read_csv(BATCH / f"fixed_comparison/{month}_training_groups.csv")
            compare_frame(groups, saved_groups, ["date"], [c for c in saved_groups if c != "date"])
            dispositions.append(groups)
            mixed = int(groups.directed_pairs.gt(0).sum())
            directed = int(groups.directed_pairs.sum())
            assert len(train) == check["complete_window_rows"] and len(complete) == check["complete_nonword_rows"]
            assert len(known) == check["valid_training_rows"] and mixed == check["mixed_days"]
            assert directed == check["directed_pairs"] and abs(mixed-check["weight_mass"]) <= 1e-12
            prefix = BATCH / "fixed_comparison" / month
            fitted = joblib.load(str(prefix)+"_model.joblib")
            started = load_json(str(prefix)+"_fit_STARTED.json")
            ended = load_json(str(prefix)+"_fit_COMPLETED.json")
            assert started["attempt"] == 1 and started["training_check"] == check == fitted["training_check"]
            assert sha(str(prefix)+"_model.joblib") == ended["model_sha256"]
            assert sha(str(prefix)+"_predictions_BEFORE_EVALUATION.csv") == ended["prediction_sha256"]
            assert fitted["features"] == fields and fitted["cutoff"] == cutoff
            assert fitted["model"].n_features_in_ == 276
            for param, value in protocol["params"].items():
                assert fitted["model"].get_params()[param] == value
            assert ended["directed_pairs"] == directed and ended["effective_mixed_days"] == mixed
            assert abs(ended["supervised_weight_mass"]-mixed) <= 1e-10
            test = full[full.date.str[:7].eq(month) & full.daily_one_word.eq(0)]
            inferred, compared_pairs = [], 0
            for _, group in test.groupby("date", sort=True):
                scores, pairs = independently_replay(fitted["model"], group, fields)
                inferred.append(scores)
                compared_pairs += pairs
            replay = pd.concat(inferred, ignore_index=True)
            saved = pred[pred.arm.eq(ARM) & pred.date.str[:7].eq(month) & pred.daily_one_word.eq(0)]
            merged = replay.merge(saved, on=["date", "code"], validate="one_to_one", suffixes=("_audit", "_stored"))
            assert len(merged) == len(test) == len(saved)
            assert merged.ranking_score_audit.isna().equals(merged.ranking_score_stored.isna())
            delta = (merged.ranking_score_audit-merged.ranking_score_stored).abs()
            assert delta.fillna(0).le(1e-12).all()
            assert merged.opponent_count_audit.eq(merged.opponent_count_stored).all()
            assert merged.ranking_status_audit.eq(merged.ranking_status_stored).all()
            assert merged.pairwise_antisymmetry_error_audit.le(5e-16).all()
            assert (merged.pairwise_antisymmetry_error_audit-merged.pairwise_antisymmetry_error_stored).abs().le(5e-16).all()
            merged["absolute_score_replay_difference"] = delta
            replay_rows.append(merged[["date", "code", "ranking_score_audit", "ranking_score_stored", "absolute_score_replay_difference", "opponent_count_audit", "pairwise_antisymmetry_error_audit", "ranking_status_audit"]])
            windows.append(dict(month=month, window_days=len(dates), first_signal=min(dates), last_signal=max(dates),
                latest_mature_target=known.target_date.max(), valid_full_rows=len(train), complete_nonword=len(complete),
                known_nonword=len(known), effective_mixed_days=mixed, directed_pairs=directed,
                unit_day_weight_sum=float(groups.independently_computed_day_weight.sum()),
                all_known_positive=int(groups.training_group_status.eq("ALL_KNOWN_POSITIVE").sum()),
                all_known_negative=int(groups.training_group_status.eq("ALL_KNOWN_NEGATIVE").sum()),
                singleton_known=int(groups.training_group_status.eq("SINGLE_KNOWN_CANDIDATE").sum()),
                no_known_rows=int(groups.training_group_status.eq("NO_KNOWN_ROWS").sum()),
                unknown_labels=int(groups.unknown_labels.sum()), inference_candidates=len(replay),
                inference_directed_pairs=compared_pairs, maximum_score_difference=float(delta.max()) if delta.notna().any() else 0.,
                maximum_complement_error=float(merged.pairwise_antisymmetry_error_audit.max()),
                model_sha256=sha(str(prefix)+"_model.joblib")))

    table, selected = [], []
    for period, start, end in PERIODS:
        f = pred[pred.date.between(start, end)]
        for arm, a in f.groupby("arm"):
            for k in (1, 2, 3):
                for same in (False, True):
                    for height in (False, True):
                        summary, choices = select_before_truth(a, k, same, height)
                        meta = dict(period=period, arm=arm, k=k, same_height=same, height_baseline=height)
                        table.append(meta | summary)
                        for key, value in meta.items(): choices[key] = value
                        selected.append(choices)
                        if not same and not height:
                            expected = result["summaries"][period]["metrics"][arm][str(k)]
                            for key, value in summary.items():
                                target = expected[key]
                                assert value == target or (isinstance(value, float) and abs(value-target) <= 1e-14), (period, arm, k, key, value, target)
    independent_table = pd.DataFrame(table)
    stored_table = read_csv(BATCH / "fixed_comparison/all_Top1_Top2_Top3_daily_sameheight_height_baselines.csv")
    key = ["period", "arm", "k", "same_height", "height_baseline"]
    a = independent_table.set_index(key).sort_index()
    z = stored_table.set_index(key).sort_index()
    assert a.index.equals(z.index)
    maximum_metric_difference = 0.
    for field in a:
        equal_null = a[field].isna() & z[field].isna()
        difference = (a[field]-z[field]).abs()
        assert (difference.le(1e-14) | equal_null).all(), field
        maximum_metric_difference = max(maximum_metric_difference, float(difference.max()))
    all_groups = pd.concat(dispositions, ignore_index=True)
    frozen_groups = read_csv(BATCH / "all16_complete_training_group_dispositions.csv")
    compare_frame(all_groups, frozen_groups, ["month", "date"], [c for c in frozen_groups if c not in ["month", "date"]])

    outputs = {
        "全部16模型逐股分数重放.csv": pd.concat(replay_rows, ignore_index=True),
        "16训练窗独立重建.csv": pd.DataFrame(windows),
        "全部训练同日组与单位权重.csv": all_groups,
        "全部Top1Top2Top3独立结算.csv": independent_table,
        "全部预选记录_先选择后结算.csv": pd.concat(selected, ignore_index=True),
        "18885旧候选逐字段不变.csv": pd.DataFrame(frozen_cell_checks),
    }
    for filename, frame in outputs.items():
        path = OUT / filename
        if path.exists(): raise FileExistsError(path)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    for path, wanted in original_bound.items():
        assert sha(path) == wanted, ("source changed during verification", path)
    count_replayed = sum(row["inference_candidates"] for row in windows)
    receipt = dict(formed_at=datetime.now(timezone.utc).isoformat(), passed=True,
        independently_implemented=True, research_evaluate_called=False, research_predict_helper_called=False,
        new_fits=0, fit_guard="HistGradientBoostingClassifier.fit patched to raise for entire replay",
        saved_models_replayed=len(windows), base_features=92, pair_input_fields=276,
        paired_rows=22662, candidates_per_arm=3777, arms=6, original_rows_unchanged=18885,
        original_comparison_cells=sum(row["cells"] for row in frozen_cell_checks), original_cell_differences=0,
        replayed_nonword_candidates=count_replayed,
        maximum_score_replay_difference=max(row["maximum_score_difference"] for row in windows),
        maximum_pairwise_complement_error=max(row["maximum_complement_error"] for row in windows),
        independent_metric_rows=len(independent_table), maximum_metric_difference=maximum_metric_difference,
        all_16_windows_252_mature_signal_days=True, all_mixed_days_weight_one=True,
        unknown_preserved_before_topk=True, historical_screen_pass_as_recorded=result["historical_screen_pass"],
        restrictions=["Replay and exact accounting do not establish prediction improvement", "Previously seen historical research, no new blind evaluation", "Relative ranking scores are not event probabilities or absolute PROMOTE calls", "No deployment, orders, realized returns, threshold search or extra training"],
        frozen_source_bindings_verified_before_and_after=len(original_bound),
        frozen_root_code_hashes={str(p): sha(p) for p in [REPO/"relay_pairwise_ranking_research.py", REPO/"relay_context_pairwise_ranker.py"]},
        script_sha256=sha(__file__),
        sources={str(p.resolve()): sha(p) for p in [Path(__file__), protocol_path, result_path, BATCH/"fixed_input_before_fit.joblib", source_reference, *[OUT/name for name in outputs]]})
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ["passed", "saved_models_replayed", "paired_rows", "original_rows_unchanged", "replayed_nonword_candidates", "maximum_score_replay_difference", "maximum_pairwise_complement_error", "independent_metric_rows", "historical_screen_pass_as_recorded"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
