"""Independent direct OLS audit and saved-head replay, without fitting."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import first_board_market_residual as new

OUT = Path(__file__).resolve().parent
protocol = json.loads((OUT / "protocol_BEFORE_ANY_FIT.json").read_text("utf-8"))
report = json.loads((OUT / "report.json").read_text("utf-8"))
for file, digest in {**protocol["inputs"], **protocol["protected"]}.items():
    assert base.sha(file) == digest, file
assert base.sha(new.__file__) == protocol["source_sha256"]
rows = pd.read_csv(OUT / "ALL_FIRST_BOARD_MARKET_CONTEXT_UNCHANGED_LABELS.csv", dtype={"code": str}, float_precision="round_trip")
original = pd.read_csv(new.REF / "all_first_and_second_boards_WITH_SEPARATE_LABELS.csv", dtype={"code": str}, float_precision="round_trip")
original = original[original.stage.eq(1)].reset_index(drop=True)
assert list(zip(rows.signal_date, rows.code)) == list(zip(original.signal_date, original.code))
np.testing.assert_allclose(rows[stage.VARIANTS["STAGE_CYCLE_GENE"] + [stage.TARGET]], original[stage.VARIANTS["STAGE_CYCLE_GENE"] + [stage.TARGET]], atol=0, rtol=0, equal_nan=True)
assert rows.label_end.fillna("UNKNOWN").tolist() == original.label_end.fillna("UNKNOWN").tolist()
screen = pd.read_csv(new.SCREEN / "COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv", dtype={"code": str})
assert rows.research_selection_allowed.tolist() == screen.research_selection_allowed.tolist()
assert rows.exclusion_reason.fillna("").tolist() == screen.exclusion_reason.fillna("").tolist()

# Re-read raw price files, independently reconstruct valid adjacent returns.
sessions = json.loads(base.INPUTS["calendar"].read_text("utf-8"))["sessions"]
cols = ["date", "code", "open", "high", "low", "close", "pre_close", "amount", "volume", "status_verified", "st", "new_listing"]
raw = pd.concat([pd.read_csv(base.INPUTS[k], usecols=cols, dtype={"code": str}, low_memory=False) for k in ("bars", "recovered")], ignore_index=True)
raw["code"] = raw.code.str.zfill(6)
raw["date"] = raw.date.astype(str).str[:10]
raw = raw[raw.date.isin(sessions)].drop_duplicates(["date", "code"], keep="last").copy()
numeric = ["open", "high", "low", "close", "pre_close", "amount", "volume"]
raw[numeric] = raw[numeric].apply(pd.to_numeric, errors="coerce")
truth = lambda s: s.astype(str).str.lower().isin(["true", "1", "1.0"])
valid = truth(raw.status_verified) & ~truth(raw.st) & ~truth(raw.new_listing) & raw.code.str.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))
valid &= raw[["open", "high", "low", "close", "pre_close"]].gt(0).all(axis=1)
valid &= raw.high.ge(raw[["open", "close", "low"]].max(axis=1)) & raw.low.le(raw[["open", "close"]].min(axis=1))
valid &= (raw.high * 100).round().le(np.floor(raw.pre_close * 110 + .5 + 1e-8))
valid &= (raw.low * 100).round().ge(np.floor(raw.pre_close * 90 + .5 + 1e-8))
valid &= raw.amount.gt(0) & raw.volume.gt(0)
prices = raw.pivot(index="date", columns="code", values="close").reindex(sessions)
reference = raw.pivot(index="date", columns="code", values="pre_close").reindex_like(prices)
observed = raw.assign(valid=valid).pivot(index="date", columns="code", values="valid").reindex_like(prices).fillna(False).to_numpy(bool)
price = prices.to_numpy(float)
previous = np.roll(price, 1, axis=0)
previous[0] = np.nan
prior_valid = np.roll(observed, 1, axis=0)
prior_valid[0] = False
allowed = observed & prior_valid & (previous > 0) & (np.abs(reference.to_numpy(float) - previous) <= .011)
values = np.where(allowed, price / previous - 1, np.nan)
sums = np.nansum(values, axis=1)
counts = np.isfinite(values).sum(axis=1)
expected = []
for row in rows.itertuples():
    i = prices.index.get_loc(row.signal_date)
    j = prices.columns.get_loc(row.code)
    result = []
    for w in (20, 60):
        calculated = [np.nan] * 4
        if i >= w:
            yy = values[i-w:i, j]
            denominators = counts[i-w:i] - 1
            xx = (sums[i-w:i] - yy) / np.where(denominators > 0, denominators, np.nan)
            if np.isfinite(yy).all() and np.isfinite(xx).all():
                variance_x = np.mean((xx-xx.mean())**2)
                variance_y = np.mean((yy-yy.mean())**2)
                if variance_x > 1e-18:
                    alpha, beta = np.linalg.lstsq(np.column_stack([np.ones(w), xx]), yy, rcond=None)[0]
                    sigma = np.sqrt(np.mean((yy-alpha-beta*xx)**2))
                    corr = np.corrcoef(xx, yy)[0, 1] if variance_x * variance_y > 1e-18 else np.nan
                    z = np.nan
                    if counts[i] > 1 and np.isfinite(values[i, j]) and sigma > 1e-12:
                        current_x = (sums[i]-values[i, j])/(counts[i]-1)
                        z = (values[i, j]-alpha-beta*current_x)/sigma
                    calculated = [float(beta), float(corr), float(sigma), float(z)]
        result.extend(calculated)
    expected.append(result)
np.testing.assert_allclose(expected, rows[new.FIELDS], atol=1e-9, rtol=1e-9, equal_nan=True)

pred = pd.read_csv(OUT / "validation_ALL_BEFORE_METRICS.csv", dtype={"code": str}, float_precision="round_trip")
name = "validation_" + new.VARIANT
bundle = joblib.load(OUT / "models" / (name + "_bundle.joblib"))
inputs = rows.merge(pred[["signal_date", "code"]], on=["signal_date", "code"], validate="one_to_one")
replayed = stage.predict(bundle, inputs)
np.testing.assert_allclose(replayed[["model_score", "calibrated_score"]], pred[["model_score", "calibrated_score"]], atol=1e-12, rtol=1e-12)
cal = pd.read_csv(OUT / "models" / (name + "_calibration_rows.csv"), dtype={"code": str}, float_precision="round_trip")
cal_info = json.loads((OUT / "models" / (name + "_calibration_receipt.json")).read_text("utf-8"))
assert cal_info["inner_max_label_end"] < cal_info["cutoff"]
prefix = joblib.load(OUT / "models" / (name + "_calibration_prefix.joblib"))
cal_x = rows.merge(cal[["signal_date", "code"]], on=["signal_date", "code"], validate="one_to_one")
np.testing.assert_allclose(prefix.predict_proba(cal_x[bundle["features"]])[:, 1], cal.uncalibrated_oof, atol=1e-12, rtol=1e-12)
for suffix in ("_calibration_prefix", "_full"):
    receipt = json.loads((OUT / "models" / (name + suffix + "_COMPLETED.json")).read_text("utf-8"))
    assert base.sha(OUT / "models" / (name + suffix + ".joblib")) == receipt["sha256"]
    members = pd.read_csv(OUT / "models" / (name + suffix + "_members.csv"), dtype={"code": str}, float_precision="round_trip")
    assert members.label_end.max() < "2025-01-01"
    target = members.merge(original[["signal_date", "code", stage.TARGET]], on=["signal_date", "code"], validate="one_to_one", suffixes=("_fit", "_original"))
    np.testing.assert_allclose(target[stage.TARGET + "_fit"], target[stage.TARGET + "_original"], atol=0, rtol=0)

def direct_metric(frame):
    picked = frame[frame.research_selection_allowed].sort_values(["signal_date", "model_score", "code"], ascending=[True, False, True]).groupby("signal_date", sort=False).head(1)
    return picked, dict(selected=len(picked), known=int(picked[stage.TARGET].notna().sum()),
        hits=int(picked[stage.TARGET].eq(1).sum()), false_positives=int(picked[stage.TARGET].eq(0).sum()),
        false_negatives=int(frame[stage.TARGET].eq(1).sum()-picked[stage.TARGET].eq(1).sum()),
        full_positive_pool=int(frame[stage.TARGET].eq(1).sum()))

picked, independent = direct_metric(pred)
for key, value in independent.items():
    assert report["decision"]["primary"][key] == value
frozen = risk.attach_frozen(rows, new.REF / "validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
old_picked, old_metric = direct_metric(frozen)
for key, value in old_metric.items():
    assert report["decision"]["baseline"][key] == value
_, non = direct_metric(pred[pred.one_price.eq(0)])
assert non["hits"] == report["decision"]["non_one_price"]["hits"]
paired = old_picked[["signal_date", "code", "name", stage.TARGET]].merge(picked[["signal_date", "code", "name", stage.TARGET]], on="signal_date", validate="one_to_one", suffixes=("_baseline", "_challenge"))
paired["same_pick"] = paired.code_baseline.eq(paired.code_challenge)
paired.to_csv(OUT / "COMPLETE117_PAIRED_TOP1_OUTCOMES.csv", index=False, encoding="utf-8-sig")
pair_stats = dict(changed=int((~paired.same_pick).sum()), gained=int((paired[stage.TARGET+"_challenge"].eq(1)&paired[stage.TARGET+"_baseline"].eq(0)).sum()),
                  lost=int((paired[stage.TARGET+"_baseline"].eq(1)&paired[stage.TARGET+"_challenge"].eq(0)).sum()))
assert independent["hits"]-old_metric["hits"] == pair_stats["gained"]-pair_stats["lost"]
assert not report["decision"]["pass_validation"] and report["decision"]["selected"] is None
assert len(list((OUT / "models").glob("*_COMPLETED.json"))) == report["core_fit_count"] == 2
assert not list((OUT / "models").glob("2025-*.joblib"))

card_fields = ["signal_date", "code", "name", "turnover", "float_cap", "volume_ratio_5", "one_price"] + new.FIELDS
cards = rows[rows.signal_date.eq("2026-08-28")][card_fields].copy()
cards["actor_identity"] = "UNKNOWN"
cards["interpretation_scope"] = "Retrospective broad-market statistics;not industry/theme/fund identity or new forecast"
assert len(cards) == 61 and not any("LABEL_ONLY" in c or "score" in c for c in cards)
cards.to_csv(OUT / "HISTORICAL_COMPLETE61_MARKET_CARDS_NO_SCORES_NO_TARGETS.csv", index=False, encoding="utf-8-sig")
result = dict(status="PASS", verified_at=base.now(), rows=len(rows), features_independently_recomputed=8,
    method="Raw-source adjacent-session returns and per-stock leave-self-out direct least-squares, not rolling covariance implementation",
    unchanged_original62_features_labels_and_screen=True, core_models_replayed=2, complete_validation_rows=len(pred),
    complete_top1_dates=len(paired), paired_change=pair_stats, historical_case_pool=61,
    monthly_challenge_fits=0, new_fits_during_verification=0, actor_identity_validated=False, goal_60_passed=False,
    source_sha256=base.sha(new.__file__), verifier_sha256=base.sha(__file__))
base.write_json(OUT / "verification.json", result)
print(json.dumps(result, ensure_ascii=True, indent=2))
