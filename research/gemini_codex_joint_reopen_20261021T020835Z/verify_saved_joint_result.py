"""Read-only replay of saved joint models and complete validation outcomes; never fit."""
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import first_board_joint_reopen_model as model

out = Path(__file__).resolve().parent
freeze = model.read_freeze()
pending = json.loads((out / "PROTOCOL_PENDING_ACK_SNAPSHOT.json").read_text("utf-8"))
for key in pending:
    if key != "status":
        assert freeze[key] == pending[key], key
assert set(freeze) - set(pending) <= {"agreed_at", "acknowledgment_note"}
rows = model.load_rows()
original = model.prior.load_rows()
pd.testing.assert_frame_equal(rows[original.columns], original)
assert len(rows) == 28503 and rows[model.stage.TARGET].isna().sum() == 126
val = rows[rows.signal_date.between("2025-01-01", "2025-06-30")]
baseline = model.risk.attach_frozen(rows, model.prior.REFERENCE / "validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
assert len(val) == 4724 and val[model.stage.TARGET].isna().sum() == 10
wide = val[model.KEYS + [model.stage.TARGET, "label_end", "one_price", "research_selection_allowed", model.FIELD] + model.prior.TIME + model.prior.FUND].copy()
wide = wide.merge(baseline[model.KEYS + ["model_score", "calibrated_score"]].rename(columns={"model_score":"BASE_raw", "calibrated_score":"BASE_calibrated"}), on=model.KEYS, validate="one_to_one", sort=False)
paired = []
verified = {}
for arm, fields in model.ARMS.items():
    directory = out / arm
    report = json.loads((directory / "report.json").read_text("utf-8"))
    before = json.loads((directory / "protocol_BEFORE_ANY_FIT.json").read_text("utf-8"))
    assert before["freeze_sha256"] == model.base.sha(out / "JOINT_FROZEN_PROTOCOL.json")
    assert before["at"] >= freeze["agreed_at"]
    receipts = list((directory / "models").glob("*_BEFORE_FIT.json"))
    assert len(receipts) == 2
    assert not list((directory / "models").glob("*REPAIR*"))
    bundle = joblib.load(directory / "models" / ("validation_" + arm + "_bundle.joblib"))
    assert bundle["features"] == fields and bundle["stage"] == 1
    for scope in ("full", "calibration_prefix"):
        name = "validation_" + arm + "_" + scope
        completed = json.loads((directory / "models" / (name + "_COMPLETED.json")).read_text("utf-8"))
        assert model.base.sha(directory / "models" / (name + ".joblib")) == completed["sha256"]
        saved = pd.read_csv(directory / "models" / (name + "_members.csv"), dtype={"code": str})
        reference = pd.read_csv(model.prior.REFERENCE / "models" / ("validation_S1_STAGE_CYCLE_GENE_" + scope + "_members.csv"), dtype={"code": str})
        columns = model.KEYS + ["label_end", model.stage.TARGET]
        assert set(map(tuple, saved[columns].to_numpy())) == set(map(tuple, reference[columns].to_numpy()))
        estimator = joblib.load(directory / "models" / (name + ".joblib"))
        actual = estimator.named_steps["hist"].get_params()
        expected = model.base.make_model("hist_gradient").get_params()
        assert actual == expected
    scored = pd.read_csv(directory / "VALIDATION_ALL_PREDICTIONS_BEFORE_METRICS.csv", dtype={"code": str}, float_precision="round_trip")
    assert scored[model.KEYS].equals(val[model.KEYS].reset_index(drop=True))
    np.testing.assert_allclose(scored[fields + [model.stage.TARGET]], val[fields + [model.stage.TARGET]], atol=0, rtol=0, equal_nan=True)
    replay = model.stage.predict(bundle, val)
    for field in ("model_score", "calibrated_score", "train_prior"):
        np.testing.assert_allclose(scored[field], replay[field], atol=1e-14, rtol=1e-14, equal_nan=True)
    for scope, group, reference in (("primary", scored, baseline),
                ("non_one_price", scored[scored.one_price.eq(0)], baseline[baseline.one_price.eq(0)])):
        metric = model.risk.screen_metric(group)
        assert model.base.clean(metric) == report[scope]
        pick = model.base.selected(group[group.research_selection_allowed], "model_score", 1)
        ref = model.base.selected(reference[reference.research_selection_allowed], "model_score", 1)
        pair = pick[["signal_date", "code", model.stage.TARGET]].rename(columns={"code":"challenger_code",model.stage.TARGET:"challenger_outcome"}).merge(
            ref[["signal_date", "code", model.stage.TARGET]].rename(columns={"code":"baseline_code", model.stage.TARGET:"baseline_outcome"}), on="signal_date", validate="one_to_one")
        assert len(pair) == 117
        pair["arm"] = arm
        pair["scope"] = scope
        paired.append(pair)
    assert report["pass_validation"] == bool(report["primary"]["hits"] > 52 and report["non_one_price"]["hits"] >= 48 and report["primary"]["decision_date_coverage"] >= .95)
    wide = wide.merge(scored[model.KEYS + ["model_score", "calibrated_score"]].rename(columns={"model_score":arm + "_raw", "calibrated_score":arm + "_calibrated"}), on=model.KEYS, validate="one_to_one", sort=False)
    verified[arm] = dict(rows=len(scored), all_saved_scores_replayed=True, full_and_prefix_members_match=True,
                         exact_fixed_estimator_params=True, primary=report["primary"], non_one_price=report["non_one_price"],
                         pass_validation=report["pass_validation"], trainer=before["trainer"], core_fits=2, temporal_calibrator_fits=1)
wide.to_csv(out / "COMPLETE4724_BOTH_ARMS_WITH_FROZEN_BASE.csv", index=False, encoding="utf-8-sig")
pairs = pd.concat(paired, ignore_index=True)
pairs.to_csv(out / "COMPLETE117_DAILY_PAIRED_SELECTIONS_ALL_ARMS.csv", index=False, encoding="utf-8-sig")
drift = []
for (arm, scope), group in pairs.groupby(["arm", "scope"]):
    drift.append(dict(arm=arm, scope=scope, dates=len(group), same_code=int(group.challenger_code.eq(group.baseline_code).sum()),
        both_hit=int((group.challenger_outcome.eq(1) & group.baseline_outcome.eq(1)).sum()),
        gained_hits=int((group.challenger_outcome.eq(1) & group.baseline_outcome.eq(0)).sum()),
        lost_hits=int((group.challenger_outcome.eq(0) & group.baseline_outcome.eq(1)).sum()),
        both_failed=int((group.challenger_outcome.eq(0) & group.baseline_outcome.eq(0)).sum()),
        unknown_outcomes=int((group.challenger_outcome.isna() | group.baseline_outcome.isna()).sum())))
receipts = json.loads((model.prior.SOURCES / "ALL668_SOURCE_RECEIPTS.json").read_text("utf-8"))
for receipt in receipts:
    assert model.base.sha(receipt["source_path"]) == receipt["sha256"]
assert len(receipts) == 668
summary = dict(at=model.base.now(), status="PASS_SAVED_VALIDATION_REPLAY", model_fits=0,
    full28503_fields_labels_screen_unchanged=True, full668_original_source_hashes=True,
    unchanged_pending_protocol_except_peer_ack=True, full4724_scores_per_arm=True,
    unknown126_and_validation10_retained=True, arms=verified, drift=drift,
    no_actor_or_causal_interpretation=True, production_changes=False)
with (out / "JOINT_SAVED_RESULT_VERIFICATION.json").open("x", encoding="utf-8") as file:
    json.dump(model.base.clean(summary), file, ensure_ascii=False, indent=2)
print(json.dumps(model.base.clean(summary), ensure_ascii=True))
