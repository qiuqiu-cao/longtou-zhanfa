"""Finalize only existing predictions, independently checking all denominators. No fit."""
from pathlib import Path
import json
import hashlib
import math
import sys

import joblib
import numpy as np
import pandas as pd

EXPERIMENT = Path(__file__).resolve().parent
ROOT = EXPERIMENT.parents[2]
sys.path.insert(0, str(ROOT))
import relay_curvature_research as research
import relay_context_research as common
from relay_expert_research import subgroup
from relay_recency_walkforward import paired_interval


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_metrics(rows):
    known = [r for r in rows if type(r["truth"]) is bool]
    scored = [r for r in known if type(r["prediction"]) is bool]
    tp = sum(r["truth"] and r["prediction"] for r in scored)
    tn = sum(not r["truth"] and not r["prediction"] for r in scored)
    fp = sum(not r["truth"] and r["prediction"] for r in scored)
    fn = sum(r["truth"] and not r["prediction"] for r in scored)
    positive = sum(r["truth"] for r in known)
    recall = tp / positive if positive else None
    precision = tp / (tp + fp) if tp + fp else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else 0. if precision is not None and recall is not None else None
    return {"registered": len(rows), "known": len(known), "scoreable": len(scored),
            "positive_support": positive, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Accuracy": (tp + tn) / len(scored) if scored else None,
            "Precision": precision, "complete_positive_Recall": recall, "F1": f1,
            "coverage": len(scored) / len(rows) if rows else None,
            "all_negative_Accuracy": (len(known) - positive) / len(known) if known else None,
            "unscored_known": len(known) - len(scored), "unknown_truth": len(rows) - len(known)}


def check_metrics(rows, expected):
    actual = independent_metrics(rows)
    for key in ["Accuracy", "Precision", "complete_positive_Recall", "F1", "coverage"]:
        a, b = actual[key], expected[key]
        if a is None or b is None:
            assert a is b, (key, a, b)
        else:
            assert abs(a-b) < 1e-12, (key, a, b)
    return actual


def independent_interval(base, challenge, sessions, fold):
    grid = [d for d in sessions if fold["start"] <= d <= fold["end"]]
    b = {(r["signal_date"], r["code"]): r for r in base}
    totals = {d: [0, 0] for d in grid}
    for c in challenge:
        r = b[(c["signal_date"], c["code"])]
        if type(c["truth"]) is bool and type(c["prediction"]) is bool and type(r["prediction"]) is bool:
            totals[c["signal_date"]][0] += int(c["prediction"] == c["truth"]) - int(r["prediction"] == r["truth"])
            totals[c["signal_date"]][1] += 1
    values = np.array(list(totals.values()), dtype=float)
    if not values[:, 1].sum():
        return None
    rng = np.random.default_rng(20261007)
    results = []
    for _ in range(2000):
        starts = rng.integers(0, len(grid), size=math.ceil(len(grid) / 5))
        ix = np.concatenate([(s+np.arange(5)) % len(grid) for s in starts])[:len(grid)]
        v = values[ix].sum(axis=0)
        assert v[1] > 0
        results.append(v[0] / v[1])
    return np.quantile(results, [.00625, .99375]).tolist()


def main():
    p = common.read(EXPERIMENT / "protocol_BEFORE_ANY_FIT.json")
    # Exact pre-fit source images remain evidence after the non-predictive fix.
    changed = {}
    for path, wanted in p["sources"].items():
        if sha(path) != wanted:
            before = EXPERIMENT / (Path(path).name + ".before_missing_target_date_fix.txt")
            assert sha(before) == wanted
            changed[path] = {"original": wanted, "preimage": str(before), "current": sha(path)}
    assert len(changed) == 2
    assert p["definition"] == research.definition()
    assert not (EXPERIMENT / "2026Sep_fit_STARTED.json").exists()
    assert len(list(EXPERIMENT.glob('*_fit_STARTED.json'))) == 2
    full = joblib.load(p["prepared_path"])["relay"]
    saved = pd.read_csv(p["baseline_path"], dtype={"code": str}, low_memory=False)
    rows, comparisons, verification, differences = [], {}, {}, []
    for fold in p["definition"]["folds"]:
        test = full[full.date.between(fold["start"], fold["end"])].copy()
        base = saved[saved.arm.eq("BASE") & saved.fold.eq(fold["id"])]
        br = common.project(base, "BASE", p["sessions"])
        file = EXPERIMENT / (fold["id"] + "_predictions_BEFORE_VALIDATION.csv")
        available = file.is_file()
        if available:
            pred = pd.read_csv(file, dtype={"code": str}, low_memory=False)
            assert set(zip(pred.date, pred.code)) == set(zip(test.date, test.code)) and len(pred) == len(test)
            assert pred.raw_score.between(0, 1).all()
            assert np.array_equal(pred.predicted.astype(bool), pred.raw_score.gt(.5))
            model = joblib.load(EXPERIMENT / (fold["id"] + "_research_only.joblib"))
            again = common.history._predict(model, test, common.RELAY_FIELDS, "relay")
            by_code = pred.set_index(["date", "code"])
            err = max(abs(r.raw_score-by_code.loc[(r.date, r.code), "raw_score"]) for r in again.itertuples())
            assert err < 1e-12
            train = common.history.training_rows(full, fold["start"], "relay")
            imputed = model["imputer"].transform(train[common.RELAY_FIELDS])
            original = model["original_scale"].transform(imputed)
            transformed = model["curvature"].transform(original)
            np.testing.assert_array_equal(transformed[:, :original.shape[1]], original)
            assert len(model[-1].coef_[0]) == original.shape[1] + 2
            verification[fold["id"]] = {"saved_model_score_error": err, "original_dimensions_preserved": original.shape[1],
                "square_coefficients": dict(zip(research.CURVE_FIELDS, model[-1].coef_[0][-2:].tolist()))}
        else:
            pred = test.copy()
            pred["status"], pred["predicted"], pred["raw_score"] = "NOT_FITTED_AFTER_FAILURE_STOP", None, np.nan
        cr = common.project(pred, research.ARM, p["sessions"])
        assert research.paired_signature(br) == research.paired_signature(cr)
        bm, cm = common.score(br, p["sessions"], fold), common.score(cr, p["sessions"], fold)
        independent = {"BASE": check_metrics(br, bm), research.ARM: check_metrics(cr, cm)}
        groups = {"BASE": subgroup(br, test), research.ARM: subgroup(cr, test)}
        for arm, group in [("BASE", br), (research.ARM, cr)]:
            for board, expected in (bm if arm == "BASE" else cm)["by_boards"].items():
                check_metrics([r for r in group if r["boards"] == int(board)], expected)
            for identity, value in [("true_nonword", 0.), ("true_oneword", 1.)]:
                check_metrics([r for r in group if r["actual_one_word_daily"] == value], groups[arm][identity])
        delta = cm["Accuracy"] - bm["Accuracy"] if available else None
        interval = independent_interval(br, cr, p["sessions"], fold) if available else None
        if available:
            another = paired_interval(br, cr, p["sessions"], fold, {"seed": 20261007, "official_session_block": 5,
                "draws": 2000, "lower_tail": .00625, "upper_tail": .99375})
            np.testing.assert_allclose(interval, [another["lower"], another["upper"]], atol=1e-15, rtol=0)
            guards = common.guards(bm, cm)
        else:
            guards = {"not_fitted": False}
        comparisons[fold["id"]] = {"status": "SAVED_PREDICTIONS_INDEPENDENTLY_VERIFIED" if available else "NOT_FITTED_PRESERVED_UNKNOWN",
            "BASE": bm, research.ARM: cm, "independent_metrics": independent, "subgroups": groups,
            "Accuracy_delta": delta, "fixed_corrected_block_interval": interval, "guards": guards,
            "passed": available and delta >= .03 and all(guards.values()) and interval[0] > 0}
        mapping = {(r["signal_date"], r["code"]): r for r in br}
        for r in cr:
            b = mapping[(r["signal_date"], r["code"])]
            if type(r["prediction"]) is bool and r["prediction"] != b["prediction"]:
                differences.append({**r, "fold": fold["id"], "BASE_prediction": b["prediction"], "BASE_score": b["raw_score"]})
        rows.extend([{**r, "fold": fold["id"]} for r in br + cr])
    assert len(rows) == 7414
    pd.DataFrame(rows).to_csv(EXPERIMENT / "full_pool_audited_with_unfitted_unknown.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(differences).to_csv(EXPERIMENT / "all_direction_changes.csv", index=False, encoding="utf-8-sig")
    report = {"formed_at": common.now(), "status": "REJECTED_INCOMPLETE_CURVATURE_TRIAL_SAVED_TWO_FOLDS_RECOVERED",
        "actual_predictive_fits": 2, "BASE_refits": 0, "September_fit_count": 0, "fitting_resumed": False,
        "comparisons": comparisons, "verification": verification, "compatibility_preimages": changed,
        "complete_original_pool": 3707, "audited_paired_rows": len(rows), "direction_changes": len(differences),
        "new_mature_forward_samples": 0, "deployment_authorized": False,
        "failure_preserved": "18 paired NaN target dates compared unequal in post-fit check; original attempt and failure retained",
        "source_hashes_after": {name: sha(name) for name in p["sources"]}, "raw_prediction_hashes": {str(f): sha(f) for f in EXPERIMENT.glob('*_predictions_BEFORE_VALIDATION.csv')},
        "original_protocol_sha256": sha(EXPERIMENT / "protocol_BEFORE_ANY_FIT.json"), "auditor_sha256": sha(__file__)}
    common.write_new(EXPERIMENT / "saved_results_audit.json", report)
    print(json.dumps({fold: {"BASE": c["independent_metrics"]["BASE"], "CURVE": c["independent_metrics"][research.ARM],
                            "delta": c["Accuracy_delta"], "interval": c["fixed_corrected_block_interval"]}
                      for fold, c in comparisons.items()}, ensure_ascii=True))


if __name__ == "__main__":
    main()
