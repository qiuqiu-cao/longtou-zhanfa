"""Read-only review of existing frozen inputs and coefficients; never fits."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import model_upgrade_history as history


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    export_path = ROOT / "results/scoring_rules_disclosure_20261014T154718Z/原BASE精确评分参数.json"
    input_path = ROOT / "results/score_update_20261014T1153Z/current29_feature_pack_BEFORE_INFERENCE.json"
    score_path = ROOT / "results/score_update_20261014T1153Z/supplemental_BASE_scores.json"
    manifest_path = ROOT / "research/model_upgrade_validation/fixed_20261007T112017Z/models/relay/BASE/manifest.json"
    protocol_path = ROOT / "research/model_upgrade_validation/recency_walkforward_20261013T134351Z/protocol_BEFORE_ANY_FIT.json"
    exp, pack, scores, manifest, protocol = map(read, [export_path, input_path, score_path, manifest_path, protocol_path])
    model_path = Path(manifest["model_path"])
    assert sha(model_path) == exp["model_sha256"] == manifest["sha256"]
    assert sha(input_path) == scores["feature_source_sha256"]
    rows = pack["relay"]
    saved = {r["code"]: r for r in scores["forecast"]["candidates"]}
    assert {r["code"] for r in rows} == set(saved) and len(rows) == 15
    calculations = []
    for r in rows:
        features = r["features"]
        values = []
        for c in exp["components"]:
            if c["kind"] == "ORIGINAL29":
                x = features[c["field"]]
                if x is None or not math.isfinite(float(x)):
                    x = c["fill_median"]
            else:
                x0 = features[c["field"].removeprefix("missingindicator_")]
                x = float(x0 is None or not math.isfinite(float(x0)))
            values.append((float(x)-c["training_mean"])/c["training_scale"]*c["standardized_logit_coefficient"])
        z = exp["intercept"] + sum(values)
        q = 1/(1+math.exp(-z))
        assert abs(q-saved[r["code"]]["raw_score"]) < 1e-12
        calculations.append({"code": r["code"], "score": q, "saved_score": saved[r["code"]]["raw_score"]})
    same = [r for r in rows if r["boards"] == 4]
    common_fields = exp["original_fields"][:18]
    assert len(same) == 3
    assert all(r["features"]["same_height_rivals"] == 2 for r in same)
    assert all(r["features"][k] == same[0]["features"][k] for r in same for k in common_fields)
    prepared_path = Path(protocol["prepared_path"])
    assert sha(prepared_path) == protocol["prepared_sha256"]
    full = joblib.load(prepared_path)["relay"]
    train = history.training_rows(full, manifest["cutoff"], "relay")
    assert len(train) == 3059 and train.date.nunique() == 252
    assert history._frame_digest(train, exp["original_fields"]) == manifest["training_digest"]
    assert int(train.target.sum()) == 1098
    weights = 1/train.groupby("date").date.transform("size")
    centered_no_missing_z = exp["intercept"] + sum(
        (0-c["training_mean"])/c["training_scale"]*c["standardized_logit_coefficient"]
        for c in exp["components"] if c["kind"] != "ORIGINAL29")
    sources = [export_path, input_path, score_path, manifest_path, protocol_path, prepared_path, model_path,
               ROOT / "september_forecast_holdout_v278.py", ROOT / "next_session_forecast_lab_v271.py", Path(__file__)]
    result = {"formed_at": datetime.now(timezone.utc).isoformat(),
              "status": "READ_ONLY_COMMENT_AUDIT_NOT_PREDICTION_UPGRADE", "fit_count": 0,
              "new_forward_samples": 0, "sources": {str(p): sha(p) for p in sources},
              "all_15_score_reproduction": calculations,
              "max_score_error": max(abs(r["score"]-r["saved_score"]) for r in calculations),
              "four_board_codes": [r["code"] for r in same], "same_height_rivals": 2,
              "common_first_18_features_verified": True,
              "training_rows": len(train), "training_dates": int(train.date.nunique()),
              "positive_count": int(train.target.sum()), "candidate_weighted_positive_rate": float(train.target.mean()),
              "equal_day_weighted_positive_rate": float(np.average(train.target, weights=weights)),
              "sigmoid_intercept": 1/(1+math.exp(-exp["intercept"])),
              "first29_at_training_mean_and_all_missing_flags_zero_score": 1/(1+math.exp(-centered_no_missing_z)),
              "limitations": ["Existing inspected history; no blind test", "Alternative current inputs; formal source gate remains unpassed",
                              "Training rates are specific to the eligible relay pool, not universal A-share priors"]}
    output = Path(__file__).with_name("independent_checks.json")
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ["sources", "all_15_score_reproduction"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
