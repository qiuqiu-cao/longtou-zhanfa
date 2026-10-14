"""Fixed, outcome-free algebra and coverage audit. This script never fits."""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
sys.path.insert(0, str(WORKSPACE))
import model_upgrade_history as history


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def finite(x):
    return pd.to_numeric(x, errors="coerce").where(lambda y: np.isfinite(y))


def pos(x):
    return finite(x).where(lambda y: y > 0)


def describe(x):
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    return {"n": len(values), "min": float(values.min()) if len(values) else None,
            "p01": float(np.quantile(values, .01)) if len(values) else None,
            "p25": float(np.quantile(values, .25)) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "p75": float(np.quantile(values, .75)) if len(values) else None,
            "p99": float(np.quantile(values, .99)) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
            "mean": float(values.mean()) if len(values) else None,
            "std_population": float(values.std()) if len(values) else None}


def peer_difference(frame, field):
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby(["date", "boards"], sort=False):
        values = group[field].to_numpy(float)
        if len(values) > 1 and np.isfinite(values).all():
            for i, index in enumerate(group.index):
                result.loc[index] = values[i] - np.median(np.delete(values, i))
    return result


def summary(frame, tolerances):
    known = frame.cap.isna() == False
    q = frame.cap_to_amount_turnover_proxy
    comparable = np.isfinite(q)
    logs = frame.loc[comparable, ["log_cap", "log_amount_turnover_proxy", "log_residual"]]
    c, x, r = (logs[k].to_numpy(float) for k in logs)
    if len(c):
        cov = float(np.mean((x-x.mean())*(r-r.mean())))
        variance_identity_error = float(abs(c.var() - x.var() - r.var() - 2*cov))
    else:
        cov = variance_identity_error = None
    return {"full_denominator": len(frame), "cap_known_positive": int(known.sum()),
        "cap_unknown": int((~known).sum()),
        "amount_positive": int(frame.signal_amount.gt(0).sum()),
        "turnover_fraction_positive": int(frame.turnover_fraction.gt(0).sum()),
        "proxy_known_positive": int(frame.amount_turnover_proxy.notna().sum()),
        "cap_and_proxy_comparable": int(comparable.sum()),
        "cap_without_proxy": int((known & ~comparable).sum()),
        "proxy_without_cap": int((frame.amount_turnover_proxy.notna() & ~known).sum()),
        "relative_disagreement_abs_q_minus1": describe(frame.relative_disagreement),
        "cap_ratio_q": describe(q),
        "fixed_tolerance_counts": [
            {"tolerance": t, "comparable_n": int(comparable.sum()),
             "within": int((comparable & frame.relative_disagreement.le(t)).sum()),
             "outside": int((comparable & frame.relative_disagreement.gt(t)).sum()),
             "unknown_full_pool": int((~comparable).sum()),
             "additional_descriptive_not_gate": t == .1} for t in tolerances],
        "strict_identity_rtol_atol_1e10": int(np.isclose(frame.cap, frame.amount_turnover_proxy,
            rtol=1e-10, atol=1e-10, equal_nan=False).sum()),
        "log_cap": describe(frame.log_cap),
        "log_existing_amount_minus_log_existing_turnover": describe(frame.log_amount_turnover_proxy),
        "log_residual": describe(frame.log_residual),
        "log_decomposition_max_absolute_error": float(np.max(np.abs(c-x-r))) if len(c) else None,
        "fixed_identity_log_rmse": float(np.sqrt(np.mean(r*r))) if len(c) else None,
        "log_cap_proxy_pearson_unfitted": float(np.corrcoef(c, x)[0, 1]) if len(c)>1 else None,
        "variance_decomposition": {"var_log_cap": float(c.var()) if len(c) else None,
            "var_log_proxy": float(x.var()) if len(c) else None,
            "var_log_residual": float(r.var()) if len(c) else None,
            "cov_proxy_residual": cov, "max_numerical_identity_error": variance_identity_error,
            "interpretation": "Algebra only. Residual variance is not independent predictive information; covariance prevents treating a variance ratio as added-data share."},
        "peer_log_cap_difference_known": int(frame.peer_log_cap_difference.notna().sum()),
        "peer_log_existing_proxy_difference_known": int(frame.peer_log_existing_proxy_difference.notna().sum()),
        "peer_log_difference_residual": describe(frame.peer_log_difference_residual)}


def main():
    plan = json.loads((HERE / "audit_plan_BEFORE_MEASUREMENT.json").read_text(encoding="utf-8"))
    amend = json.loads((HERE / "root_tolerance_amendment_BEFORE_MEASUREMENT.json").read_text(encoding="utf-8"))
    tolerances = amend["root_tolerances_adopted_before_measurement"] + [.1]
    for path, expected in plan["sources"].items():
        assert sha(path) == expected, path
    assert sha(HERE.parent / "diagnosis_plan_BEFORE_FITS.json") == amend["root_plan_sha256"]
    prepared_path = next(p for p in plan["sources"] if p.endswith("prepared_source_reconstruction.joblib"))
    receipt_path = next(p for p in plan["sources"] if p.endswith("source_reconstruction_BEFORE_FITS.json"))
    prepared = joblib.load(prepared_path)
    original = prepared["relay"]
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    assert sha(prepared_path) == receipt["prepared_sha256"]
    peer_path = next(p for p in plan["sources"] if p.endswith("full9245_peer_context_NO_OUTCOMES.csv"))
    peer = pd.read_csv(peer_path, dtype={"code": str})
    bridge_path = next(p for p in plan["sources"] if p.endswith("September182_cap_source_bridge_NO_TARGETS.csv"))
    bridge = pd.read_csv(bridge_path, dtype={"code": str})
    for frame in (peer, bridge):
        frame["code"] = frame.code.str.zfill(6)
        assert not frame.duplicated(["date", "code"]).any()
    assert len(original) == len(peer) == 9245
    assert list(zip(original.date, original.code)) == list(zip(peer.date, peer.code))
    for field in ["signal_amount", "turnover", "float_cap_log", "amount_float_cap"]:
        np.testing.assert_allclose(finite(original[field]), finite(peer[field]), rtol=1e-12, atol=1e-12, equal_nan=True)
    facts = original[["date", "code", "boards", "price", "signal_amount", "turnover", "signal_float_cap", "float_cap_log", "amount_float_cap"]].copy()
    facts = facts.rename(columns={"turnover": "turnover_fraction", "signal_float_cap": "original_cap"})
    facts["original_cap"] = pos(facts.original_cap)
    facts["cap"] = facts.original_cap
    facts["cap_source"] = np.where(facts.cap.notna(), "ORIGINAL_PREPARED_CAP", "UNKNOWN")
    facts["cap_bridge_appended"] = False
    b = bridge.set_index(["date", "code"])
    keys = list(zip(facts.date, facts.code))
    matched = [k for k in keys if k in b.index]
    assert len(matched) == len(bridge) == 182
    for i, key in enumerate(keys):
        if key not in b.index:
            continue
        row = b.loc[key]
        assert int(row.boards_original) == int(facts.loc[i, "boards"])
        for field, bridge_field in [("signal_amount", "signal_amount"), ("turnover_fraction", "turnover_original"), ("price", "price_original")]:
            assert np.isclose(facts.loc[i, field], row[bridge_field], rtol=1e-12, atol=1e-12)
        assert pd.isna(facts.loc[i, "original_cap"])
        assert np.isfinite(row.float_cap) and row.float_cap > 0
        facts.loc[i, "cap"] = float(row.float_cap)
        facts.loc[i, "cap_source"] = "SEPTEMBER_EXACT_SOURCE_BRIDGE_LATE_APPEND"
        facts.loc[i, "cap_bridge_appended"] = True
    facts["signal_amount"] = pos(facts.signal_amount)
    facts["turnover_fraction"] = pos(facts.turnover_fraction)
    facts["amount_turnover_proxy"] = facts.signal_amount / facts.turnover_fraction
    facts["cap_to_amount_turnover_proxy"] = facts.cap / facts.amount_turnover_proxy
    facts["relative_disagreement"] = abs(facts.cap_to_amount_turnover_proxy - 1)
    facts["log_cap"] = np.log(facts.cap)
    facts["log_amount_turnover_proxy"] = np.log(facts.signal_amount) - np.log(facts.turnover_fraction)
    facts["log_residual"] = np.log(facts.cap_to_amount_turnover_proxy)
    facts["log_decomposition_error"] = facts.log_cap - facts.log_amount_turnover_proxy - facts.log_residual
    facts["peer_log_cap_difference"] = peer_difference(facts, "log_cap")
    facts["peer_log_existing_proxy_difference"] = peer_difference(facts, "log_amount_turnover_proxy")
    facts["peer_log_difference_residual"] = facts.peer_log_cap_difference - facts.peer_log_existing_proxy_difference
    original_log = finite(original.float_cap_log)
    original_ratio = finite(original.amount_float_cap)
    mask = facts.original_cap.notna()
    np.testing.assert_allclose(np.log(facts.loc[mask, "original_cap"]), original_log[mask], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(facts.loc[mask, "signal_amount"] / facts.loc[mask, "original_cap"], original_ratio[mask], rtol=1e-12, atol=1e-12)
    ledger_cols = list(facts.columns)
    assert not set(ledger_cols) & {"target", "target_date", "label_valid", "prediction", "truth", "promoted_next_day", "next_open_gap"}
    # Membership metadata is used without outcome-conditioned grouping or values in the ledger.
    training = []
    for fold in history.FOLDS:
        train = history.training_rows(original, fold["start"], "relay")
        check = next(x for x in receipt["training_checks"] if x["fold"] == fold["id"])
        assert history._frame_digest(train, prepared["relay_fields"]) == check["training_digest"]
        assert len(train) == check["train_n"]
        assert train.date.nunique() == check["training_dates"] == 252
        assert train.target_date.max() == check["training_max_label_date"]
        member = set(zip(train.date, train.code))
        subset = facts[[key in member for key in keys]]
        assert len(subset) == len(train)
        facts["member_train_" + fold["id"]] = [key in member for key in keys]
        training.append({"fold": fold["id"], "cutoff": fold["start"], **check,
                         "summary": summary(subset, tolerances)})
    original_frame = facts[facts.cap_source.eq("ORIGINAL_PREPARED_CAP")]
    bridge_frame = facts[facts.cap_bridge_appended]
    result = {"formed_at": datetime.now(timezone.utc).isoformat(),
        "scope": plan["scope"], "pool_order_preserved": True, "full_pool": summary(facts, tolerances),
        "original_prepared_cap": summary(original_frame, tolerances),
        "late_september_bridge": summary(bridge_frame, tolerances),
        "original_field_checks": {"cap_log_equal": int(mask.sum()), "amount_float_cap_equal": int(mask.sum()),
            "original_cap_unknown": int((~mask).sum())},
        "training_only": training,
        "dimensional_derivation": {"A": "same-day cash turnover amount in CNY", "t": "turnover fraction, not percent",
            "M_proxy": "A/t in CNY", "q": "M_observed*t/A", "same_denominator_conditional_identity":
            "If t=V/N and A=V*VWAP while M=N*close, then q=close/VWAP. Different share denominators add Ncap/Nturnover.",
            "stored_cap_formula_source": "leader_system_v3.py:520-523: outstanding_share * close; historical Sina turnover multiplied by100 before general fraction normalization",
            "source_status": "Source algebra alone does not establish equivalence with independently reported provider free-float cap. Source receipt and units are separately required.",
            "incremental_data_claim": "Not established. Almost-equal values may mainly re-express existing amount and turnover; nonzero residual may contain price-location or denominator/provider differences, not necessarily new predictive information.",
            "feature_expressivity": "A linear logit on original raw turnover cannot exactly represent log(amount)-log(turnover). A cap proxy may change functional representation without adding observed data."},
        "label_conditioned_analysis": False, "candidate_predictions_evaluated": False,
        "new_fits": 0, "new_real_forward_samples": 0, "gain_verified": False,
        "input_bindings": plan["sources"], "script_sha256": sha(__file__),
        "original_labels_only_used_for_fixed_membership": True,
        "original_labels_not_emitted": True, "original_sources_not_modified": True}
    facts.to_csv(HERE / "full9245_cap_algebra_NO_OUTCOMES.csv", index=False)
    result["output_hashes"] = {"full9245_cap_algebra_NO_OUTCOMES.csv": sha(HERE / "full9245_cap_algebra_NO_OUTCOMES.csv")}
    write(HERE / "cap_algebra_results.json", result)
    for path, expected in plan["sources"].items():
        assert sha(path) == expected, path
    print(json.dumps({"full": result["full_pool"], "training": [{"fold":t["fold"], "train_n":t["train_n"], "summary": t["summary"]} for t in training]}, indent=2))


if __name__ == "__main__":
    main()
