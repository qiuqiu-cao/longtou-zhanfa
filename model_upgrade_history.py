"""Fixed V278 recipes and historical comparisons; no fitting at import time.

Only explicit reproduce/evaluate/freeze calls fit models. Historical archives
were previously inspected: none of these outputs is blind or live evidence.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from io import StringIO
import json
from pathlib import Path
import shutil
import warnings
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "results/v278/september_falsification_20261002_v6"
CALENDAR = ROOT / "data/v279/exchange_calendar_audit_20261002_v1/calendar.json"
ARMS = ("BASE", "C1", "C2")
FOLDS = [
    {"id": "2025H2", "start": "2025-07-01", "end": "2025-12-31"},
    {"id": "2026JanAug", "start": "2026-01-01", "end": "2026-08-31"},
    {"id": "2026Sep", "start": "2026-09-01", "end": "2026-09-30"},
]
FREEZE_CUTOFF = "2026-09-01"
MARKET_INTERACTIONS = ("nonpositive_premium_promotion_down", "nonpositive_premium_failure_up")
RELAY_INTERACTIONS = ("amount_rank_competition", "turnover_rank_competition")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def recipe():
    """The exact recipe to bind in BEFORE_EFFECTS.json before any new fit."""
    return {
        "freeze_cutoff": FREEZE_CUTOFF, "folds": FOLDS,
        "arms": list(ARMS), "market_base_fields": 46, "relay_base_fields": 29,
        "market_target": "next emotion_quality_score minus current; >.05 IMPROVE, <-.05 WORSEN, else CONTINUE",
        "relay_target": "original exact-next-session consecutive-board promotion label",
        "training": "signal and target dates strictly before fold start; valid known labels only",
        "market_window": None, "relay_window": 252,
        "relay_window_unit": "last 252 distinct eligible signal dates, not official target grid",
        "market_estimator": {"kind": "forest", "n_estimators": 160, "max_depth": 5,
            "min_samples_leaf": 15, "max_features": .75, "random_state": 274, "n_jobs": 2},
        "relay_estimator": {"kind": "logistic", "C": .25, "max_iter": 2500,
            "random_state": 271, "solver": "lbfgs", "class_weight": None},
        "preprocessing": "training-only median; add_indicator=True; keep_empty_features=True; StandardScaler for relay only",
        "weights": "market ones; relay inverse complete candidate count per signal date",
        "market_C1": "premium<=0 AND promotion_mid below exact official predecessor",
        "market_C2": "C1 plus premium<=0 AND failed_rate above exact official predecessor",
        "relay_C1": "(signal_amount_same_height_percentile-.5)*log1p(same_height_rivals)",
        "relay_C2": "C1 plus (turnover_same_height_percentile-.5)*log1p(same_height_rivals)",
        "interaction_missing": "unknown input or absent official predecessor remains NaN, never false/zero",
        "reproduction_tolerance": 1e-8,
        "reproduction_score": "market stored confidence; relay stored positive-class score",
        "rank_ties": "original V280 score descending/code ascending; V271 analysis top1 also boards descending, separately reported",
        "probabilities": "uncalibrated internal scores; calibrated Brier unavailable",
        "history_status": "PREVIOUSLY_INSPECTED_RETROSPECTIVE_NOT_BLIND_NOT_FORWARD",
        "selection": "fixed V278 choices; no re-selection, thresholds or parameter search",
    }


def _write_json(path, obj):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _protocol(protocol):
    if isinstance(protocol, (str, Path)):
        p = Path(protocol)
        return json.loads(p.read_text(encoding="utf-8")), sha(p)
    return protocol, protocol.get("protocol_sha256", digest(protocol))


def _validate_protocol(protocol, prepared, output=None):
    value, bound = _protocol(protocol)
    if "recipe" in value:
        if value["recipe"] != recipe():
            raise ValueError("protocol recipe differs from the fixed V278 comparison")
    else:
        expected_recipes = {
            "market": {"kind": "forest", "window": None, "fields": 46, "n_estimators": 160,
                "max_depth": 5, "min_samples_leaf": 15, "max_features": .75, "random_state": 274, "n_jobs": 2},
            "relay": {"kind": "logistic", "window": 252, "window_unit": "known-label signal dates",
                "fields": 29, "C": .25, "max_iter": 2500, "random_state": 271,
                "sample_weight": "inverse training candidate rows per signal date"},
        }
        expected_methods = {
            "market_C1": "(yesterday_premium<=0) AND (promotion_mid - exact previous official session <0)",
            "market_C2": "C1 plus (yesterday_premium<=0) AND (failed_rate - exact previous official session >0)",
            "relay_C1": "(signal_amount_same_height_percentile-0.5)*log1p(same_height_rivals)",
            "relay_C2": "C1 plus (turnover_same_height_percentile-0.5)*log1p(same_height_rivals)",
            "market_target": "same formula next official session emotion_quality_score minus signal quality: >0.05 IMPROVE, <-0.05 WORSEN, otherwise CONTINUE; missing UNKNOWN",
            "relay_target": "independent cent-exact unadjusted price and explicit cap reconciled with complete next official closing pool: boards current+1",
            "promotion_mid": "unchanged original previous closing pool 2-4 boards promotion rate",
            "rank_policy": "raw_score descending, code ascending (current V280); negative rank first remains negative",
            "missing": "NaN; original training-only imputation; no future/unknown-to-zero conversion",
        }
        if (value.get("version") != "model-upgrade-fixed-1" or value.get("recipes") != expected_recipes or
                value.get("freeze_cutoff") != FREEZE_CUTOFF or value.get("folds") != FOLDS or
                value.get("method_definitions") != expected_methods or value.get("reproduction_score_tolerance") != 1e-8):
            raise ValueError("protocol fixed recipes/methods/boundaries differ")
        for module in ("market", "relay"):
            if value.get("modules", {}).get(module, {}).get("arms") != {a: a for a in ARMS}:
                raise ValueError("protocol arm set differs")
    provenance = prepared["provenance"]
    if value.get("version") == "model-upgrade-fixed-1":
        receipt_path = Path(value["provenance_path"])
        if sha(receipt_path) != value["provenance_sha256"]:
            raise ValueError("pre-fit provenance receipt changed")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        recorded = receipt.get("provenance", receipt)
        if receipt.get("protocol_sha256", bound) != bound:
            raise ValueError("pre-fit provenance belongs to a different protocol")
        for key in ("source_hashes", "dependency_hashes"):
            if recorded.get(key) != provenance[key]:
                raise ValueError("prepared data differ from pre-fit provenance: " + key)
    bindings = (value.get("input_hashes", {}) | value.get("dependency_hashes", {}) |
                value.get("source_hashes", {}) | value.get("protected_sha256", {}))
    # The immutable implementation snapshot is recorded after coding but before
    # fitting. It expands the already-frozen protocol to the complete hash chain.
    if output is not None:
        implementation = Path(output) / "implementation_BEFORE_FITS.json"
        if implementation.exists():
            snapshot = json.loads(implementation.read_text(encoding="utf-8"))
            for key in ("source_hashes", "dependency_hashes", "code_sha256", "protected_sha256"):
                bindings.update(snapshot.get(key, {}))
    for path, expected in value.get("protected_sha256", {}).items():
        if sha(path) != expected:
            raise ValueError("protected original artifact changed: " + path)
    # Every source and imported local dependency must be explicitly pre-bound.
    for path, expected in provenance["source_hashes"].items() | provenance["dependency_hashes"].items():
        if bindings.get(path) != expected:
            raise ValueError("protocol source/dependency not bound: " + path)
        if sha(path) != expected:
            raise ValueError("source/dependency changed since data preparation: " + path)
    return value, bound


def _dependency_hashes(entry=ROOT / "september_forecast_holdout_v278.py"):
    """Bind the full recursively imported local Python closure, including helpers."""
    pending, seen = [Path(entry), Path(__file__)], set()
    while pending:
        path = pending.pop().resolve()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                candidate = ROOT / (name.replace(".", "/") + ".py")
                if candidate.exists():
                    pending.append(candidate)
    return {str(p): sha(p) for p in sorted(seen)}


def _frame_digest(frame, fields):
    columns = list(dict.fromkeys(["date", "target_date", "target", "label_valid"] +
                                (["code"] if "code" in frame else []) + list(fields)))
    # Stable text includes row order, missingness, labels and exact field order.
    return hashlib.sha256(frame[columns].to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def _successor(sessions):
    if list(sessions) != sorted(set(sessions)):
        raise ValueError("official sessions must be unique and ordered")
    return dict(zip(sessions[:-1], sessions[1:]))


def enforce_exact_targets(frame, sessions):
    """Retain rows, but invalidate labels whose next data row skips a session."""
    f = frame.copy()
    expected = f.date.map(_successor(sessions))
    matched = expected.notna() & f.target_date.eq(expected)
    changed = int((f.label_valid.fillna(False).astype(bool) & ~matched).sum())
    f["label_valid"] = f.label_valid.fillna(False).astype(bool) & matched
    f["official_target_date"] = expected
    return f, changed


def _assert_cached_market_equivalent(combined, saved):
    """Compare with the original CSV roundtrip, including empty-string NA rules."""
    roundtrip = pd.read_csv(StringIO(combined[saved.columns].to_csv(index=False)))
    pd.testing.assert_frame_equal(roundtrip.reset_index(drop=True), saved.reset_index(drop=True),
        check_dtype=False, rtol=1e-10, atol=1e-12)


def prepare_datasets():
    """Read source-bound original archives; rebuild old feature vectors, no fits."""
    from september_forecast_holdout_v278 import (ARCHIVE, DAILY, CUTOFF, attach_ohlc,
        rebuild_market, relay_features)
    from next_session_forecast_lab_v271 import DATA, prepare_market, prepare_relay
    from market_expectation_extensions_v272 import INDEX_PATH, direction
    from historical_ohlc_archive_v278 import verify_archive

    original_report = json.loads((ORIGINAL / "report.json").read_text(encoding="utf-8"))
    sources = {}
    for name, expected in original_report["source_hashes"].items():
        p = ROOT / name.replace("\\", "/")
        if sha(p) != expected:
            raise ValueError("original V278 hash changed: " + name)
        sources[str(p)] = expected
    for p in [ORIGINAL / "report.json", ORIGINAL / "combined_market_input.csv",
              ORIGINAL / "market_predictions_LABEL_ONLY.csv", ORIGINAL / "relay_predictions_LABEL_ONLY.csv", CALENDAR]:
        sources[str(p)] = sha(p)
    official = json.loads(CALENDAR.read_text(encoding="utf-8"))["sessions"]
    _successor(official)
    daily = verify_archive(DAILY.parent)
    bars_report = json.loads((DAILY.parent / "report.json").read_text(encoding="utf-8"))
    for item in bars_report["audit"]:
        if item["status"] == "UNADJUSTED_DAY_AVAILABLE":
            p = DAILY.parent / "raw" / (item["symbol"] + ".json")
            sources[str(p)] = sha(p)
    pools = attach_ohlc(pd.read_csv(ARCHIVE, dtype={"code": str}), daily)
    base = pd.read_csv(DATA / "market_emotion.csv")
    rebuilt, source_audit = rebuild_market(pools, base)
    if source_audit != original_report["source_audit"]:
        raise ValueError("original auxiliary source audit no longer reproduces")
    for name in pools.source_path.dropna().unique():
        p = ROOT / str(name)
        if p.exists():
            sources[str(p)] = sha(p)
        manifest = p.parent / "manifest.json"
        if manifest.exists():
            sources[str(manifest)] = sha(manifest)
    for item in source_audit:
        if item.get("path"):
            p = ROOT / item["path"]
            sources[str(p)] = sha(p)
    combined = pd.concat([base[base.date < CUTOFF], rebuilt[rebuilt.date >= CUTOFF]], ignore_index=True)
    saved = pd.read_csv(ORIGINAL / "combined_market_input.csv")
    _assert_cached_market_equivalent(combined, saved)
    market, market_fields = prepare_market(ORIGINAL / "combined_market_input.csv")
    index = pd.read_csv(INDEX_PATH)[["date", "sh_a_return_pct", "sz_a_return_pct"]]
    bars = daily.sort_values(["symbol", "date"]).copy()
    bars["return_pct"] = bars.groupby("symbol").close.pct_change() * 100
    new_index = bars[bars.symbol.isin(["sh000002", "sz399107"])].pivot(
        index="date", columns="symbol", values="return_pct").reset_index().rename(
        columns={"sh000002": "sh_a_return_pct", "sz399107": "sz_a_return_pct"})
    index = pd.concat([index[index.date < CUTOFF], new_index[new_index.date >= CUTOFF]], ignore_index=True)
    market = market.merge(index, on="date", how="left", validate="one_to_one").sort_values("date").reset_index(drop=True)
    for key in ["sh_a_return_pct", "sz_a_return_pct"]:
        market_fields.append(key)
        for lag in [1, 3]:
            name = key + f"_lag_{lag}"
            market[name] = market[key].shift(lag)
            market_fields.append(name)
    market["target"] = direction(market.emotion_quality_score.shift(-1) - market.emotion_quality_score, .05)
    september = market.date.ge(CUTOFF)
    actual_next = market.date.shift(-1)
    market.loc[september, "target_date"] = market.loc[september, "date"].map(_successor(official))
    market.loc[september, "label_valid"] &= actual_next[september].eq(market.loc[september, "target_date"])
    market, rejected_market = enforce_exact_targets(market, official)
    historical, relay_fields = prepare_relay(DATA / "leader_candidates.csv", base.date.tolist())
    rf = relay_features(pools, rebuilt, historical)
    relay = pd.concat([historical[historical.date < CUTOFF], rf[rf.date >= CUTOFF]], ignore_index=True)
    relay, rejected_relay = enforce_exact_targets(relay, official)
    if len(market_fields) != 46 or len(relay_fields) != 29:
        raise ValueError("original feature-count contract drifted")
    if market.date.duplicated().any() or relay.duplicated(["date", "code"]).any():
        raise ValueError("duplicate signal keys")
    provenance = {"source_hashes": sources, "dependency_hashes": _dependency_hashes(),
        "recipe_hash": digest(recipe()), "sessions": official, "source_audit": source_audit,
        "official_target_invalidated": {"market": rejected_market, "relay": rejected_relay},
        "runtime_versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"]},
        "python_version": sys.version,
        "prepared_at": now(), "training_inputs": {
            "market": _frame_digest(market, market_fields), "relay": _frame_digest(relay, relay_fields)}}
    return {"market": market, "market_fields": market_fields, "relay": relay,
            "relay_fields": relay_fields, "provenance": provenance}


def interactions(frame, module, sessions=None):
    """Append fixed interactions without dropping unknowns or cohort members."""
    f = frame.copy()
    if module == "market":
        if sessions is None:
            raise ValueError("market interactions require official sessions")
        if f.date.duplicated().any():
            raise ValueError("duplicate market date")
        predecessor = dict(zip(sessions[1:], sessions[:-1]))
        _successor(sessions)
        previous_dates = f.date.map(predecessor)
        by_day = f.set_index("date")
        premium = pd.to_numeric(f.yesterday_premium, errors="coerce")
        for field, relation, name in [("promotion_mid", "down", MARKET_INTERACTIONS[0]),
                                       ("failed_rate", "up", MARKET_INTERACTIONS[1])]:
            current = pd.to_numeric(f[field], errors="coerce")
            prior = previous_dates.map(pd.to_numeric(by_day[field], errors="coerce"))
            known = np.isfinite(premium) & np.isfinite(current) & np.isfinite(prior)
            condition = current.lt(prior) if relation == "down" else current.gt(prior)
            f[name] = (premium.le(0) & condition).astype(float).where(known)
    elif module == "relay":
        if f.duplicated(["date", "code"]).any():
            raise ValueError("duplicate relay key")
        size = f.groupby(["date", "boards"])["code"].transform("size")
        rivals = pd.to_numeric(f.same_height_rivals, errors="coerce")
        if (rivals.notna() & rivals.ne(size - 1)).any():
            raise ValueError("incomplete original same-height cohort")
        for field, name in [("signal_amount", RELAY_INTERACTIONS[0]), ("turnover", RELAY_INTERACTIONS[1])]:
            values = pd.to_numeric(f[field], errors="coerce").where(lambda x: np.isfinite(x))
            ranks = values.groupby([f.date, f.boards]).rank(pct=True)
            f[name] = ((ranks - .5) * np.log1p(rivals.where(rivals.ge(0)))).where(np.isfinite(rivals))
    else:
        raise ValueError("unknown module")
    return f


def arm_fields(base_fields, module, arm):
    if arm not in ARMS or module not in ("market", "relay"):
        raise ValueError("unknown module/arm")
    extra = MARKET_INTERACTIONS if module == "market" else RELAY_INTERACTIONS
    return list(base_fields) + list(extra[:ARMS.index(arm)])


def training_rows(frame, cutoff, module):
    valid = frame.label_valid.fillna(False).astype(bool) & frame.target.notna()
    known = frame[valid & frame.date.lt(cutoff) & frame.target_date.lt(cutoff)].copy()
    if module == "relay":
        dates = sorted(known.date.unique())[-252:]
        known = known[known.date.isin(dates)].copy()
    elif module != "market":
        raise ValueError("unknown module")
    return known


def _fit(frame, fields, module):
    from rolling_forecast_challenger_v274 import fit
    from sklearn.exceptions import ConvergenceWarning
    if frame.empty or frame.target.nunique() < 2:
        raise ValueError("insufficient labeled training classes; no unregistered fallback")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            return fit(frame, fields, "forest" if module == "market" else "logistic", module)
    except ConvergenceWarning as error:
        raise ValueError("FIT_NONCONVERGENCE: fixed estimator did not converge; no artifact or fallback") from error


def _predict(model, frame, fields, module):
    result = frame.copy()
    result["status"] = "UNKNOWN_INPUT"
    result["predicted"] = None
    result["confidence"] = np.nan
    result["raw_score"] = np.nan
    classes = list(model.classes_)
    for label in classes:
        result["raw_score_" + str(label)] = np.nan
    eligible = result[fields].notna().any(axis=1)
    if module == "market" and "limit_count" in result:
        eligible &= result.limit_count.notna()
    if eligible.any():
        scores = model.predict_proba(result.loc[eligible, fields])
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite model scores")
        result.loc[eligible, "predicted"] = np.asarray(classes)[scores.argmax(axis=1)]
        result.loc[eligible, "confidence"] = scores.max(axis=1)
        result.loc[eligible, "status"] = "PREDICTED_UNCALIBRATED"
        for j, label in enumerate(classes):
            result.loc[eligible, "raw_score_" + str(label)] = scores[:, j]
        if module == "relay":
            if 1 not in classes:
                raise ValueError("positive relay class missing")
            result.loc[eligible, "raw_score"] = scores[:, classes.index(1)]
            result["probability"] = result.raw_score  # Exact legacy CSV name; still uncalibrated.
    return result


def compare_reproduction(observed, expected, module, tolerance=1e-8):
    keys = ["date"] + (["code"] if module == "relay" else [])
    if observed.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError("duplicate reproduction keys")
    a, b = observed.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("reproduction universe differs from original")
    if not a.predicted.astype(str).equals(b.predicted.astype(str)):
        raise ValueError("original predicted classes do not reproduce")
    col = "confidence" if module == "market" else "probability"
    x, y = a[col].to_numpy(float), b[col].to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("unknown original scores cannot prove reproduction")
    error = float(np.max(np.abs(x - y))) if len(x) else 0.
    if error > tolerance:
        raise ValueError(f"original score drift {error} exceeds {tolerance}")
    return {"rows": len(a), "score_column": col, "max_abs_error": error, "passed": True}


def _majority(train):
    counts = train.target.value_counts()
    return {"class": str(sorted(counts.index, key=lambda x: (-counts[x], str(x)))[0]),
            "counts": {str(k): int(v) for k, v in counts.items()}}


def _save_model(directory, model, fields, train, module, arm, prepared, protocol_hash, cutoff):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model_path = directory / "model.joblib"
    joblib.dump(model, model_path)
    manifest = {"module": module, "arm": arm, "fields": fields, "model_path": str(model_path.resolve()),
        "sha256": sha(model_path), "classes": [str(c) for c in model.classes_],
        "training_max_label_date": str(train.target_date.max()), "formed_at": now(), "cutoff": cutoff,
        "sources": prepared["provenance"]["source_hashes"],
        "code_sha256": prepared["provenance"]["dependency_hashes"],
        "runtime_versions": prepared["provenance"].get("runtime_versions", {}),
        "python_version": prepared["provenance"].get("python_version"),
        "training_digest": _frame_digest(train, fields), "training_rows": len(train),
        "training_signal_dates": int(train.date.nunique()), "training_majority": _majority(train),
        "recipe": recipe(), "protocol_sha256": protocol_hash, "score_status": "UNCALIBRATED_INTERNAL",
        "trade_authorized": False}
    _write_json(directory / "manifest.json", manifest)
    return manifest


def reproduce(output, protocol, prepared=None):
    """Fit only original BASEs, then require exact saved V278 classes/scores."""
    prepared = prepare_datasets() if prepared is None else prepared
    _, protocol_hash = _validate_protocol(protocol, prepared, output)
    directory = Path(output) / "reproduction"
    directory.mkdir(parents=True, exist_ok=False)
    report = {"protocol_sha256": protocol_hash, "formed_at": now(), "modules": {}}
    try:
        for module in ("market", "relay"):
            frame, fields = prepared[module], prepared[module + "_fields"]
            train = training_rows(frame, FREEZE_CUTOFF, module)
            expected = pd.read_csv(ORIGINAL / (module + "_predictions_LABEL_ONLY.csv"), dtype={"code": str})
            keys = ["date"] + (["code"] if module == "relay" else [])
            # Select exactly the original registered output keys, not eligible labels.
            test = frame.merge(expected[keys], on=keys, how="inner", validate="one_to_one")
            model = _fit(train, fields, module)
            predicted = _predict(model, test, fields, module)
            check = compare_reproduction(predicted, expected, module)
            predicted.to_csv(directory / (module + "_predictions.csv"), index=False, encoding="utf-8-sig")
            artifact = _save_model(directory / "BASE_artifacts" / module, model, fields, train,
                module, "BASE", prepared, protocol_hash, FREEZE_CUTOFF)
            report["modules"][module] = check | {"manifest": artifact}
        report["status"] = "REPRODUCED"
        report["both_bases_reproduced"] = True
        _write_json(directory / "report.json", report)
    except Exception as error:
        _write_json(directory / "FAILED.json", report | {"status": "FAILED_NOT_REPRODUCED", "error": str(error)})
        raise
    return report | {"report_path": str(directory / "report.json")}


def _require_reproduction(output, bound):
    path = Path(output) / "reproduction/report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("status") not in ("REPRODUCED", "BOTH_BASES_REPRODUCED") or
            report.get("protocol_sha256") != bound):
        raise ValueError("both original BASEs must reproduce under this protocol first")


def evaluate_module(module, output, protocol, prepared=None):
    """One fixed three-arm, three-fold diagnostic comparison; never select here."""
    prepared = prepare_datasets() if prepared is None else prepared
    _, bound = _validate_protocol(protocol, prepared, output)
    _require_reproduction(output, bound)
    directory = Path(output) / (module + "_history")
    directory.mkdir(parents=True, exist_ok=False)
    frame = interactions(prepared[module], module, prepared["provenance"]["sessions"])
    rows, logs = [], []
    try:
        for fold in FOLDS:
            train = training_rows(frame, fold["start"], module)
            test = frame[frame.date.between(fold["start"], fold["end"])].copy()
            for arm in ARMS:
                fields = arm_fields(prepared[module + "_fields"], module, arm)
                if fold["id"] == "2026Sep" and arm == "BASE":
                    base_directory = Path(output) / "reproduction/BASE_artifacts" / module
                    base_manifest = json.loads((base_directory / "manifest.json").read_text(encoding="utf-8"))
                    if (base_manifest["protocol_sha256"] != bound or base_manifest["fields"] != fields or
                            base_manifest["training_digest"] != _frame_digest(train, fields) or
                            sha(base_directory / "model.joblib") != base_manifest["sha256"]):
                        raise ValueError("reproduced BASE cache binding drift")
                    model = joblib.load(base_directory / "model.joblib")
                else:
                    model = _fit(train, fields, module)
                pred = _predict(model, test, fields, module)
                pred["arm"], pred["fold"] = arm, fold["id"]
                pred["training_max_label_date"] = str(train.target_date.max())
                pred["train_n"] = len(train)
                rows.append(pred)
                logs.append({"fold": fold, "arm": arm, "train_n": len(train),
                    "training_dates": int(train.date.nunique()), "training_max_label_date": str(train.target_date.max()),
                    "training_digest": _frame_digest(train, fields), "fields": fields,
                    "training_majority": _majority(train)})
                if fold["id"] == "2026Sep":
                    _save_model(directory / "september_fit_cache" / arm, model, fields, train,
                        module, arm, prepared, bound, FREEZE_CUTOFF)
        predictions = pd.concat(rows, ignore_index=True)
        predictions.to_csv(directory / "predictions.csv", index=False, encoding="utf-8-sig")
        report = {"status": "FIXED_HISTORY_COMPLETE_NO_UPGRADE_DECISION", "module": module,
            "arms": list(ARMS), "folds": FOLDS, "protocol_sha256": bound, "formed_at": now(),
            "predictions_sha256": sha(directory / "predictions.csv"), "fit_logs": logs,
            "evidence_class": "PREVIOUSLY_INSPECTED_HISTORICAL_DIAGNOSTIC", "trade_authorized": False}
        _write_json(directory / "report.json", report)
    except Exception as error:
        _write_json(directory / "FAILED.json", {"status": "FAILED_NOT_COMPLETE", "error": str(error), "fit_logs": logs})
        raise
    return report | {"predictions": predictions, "report_path": str(directory / "report.json")}


def freeze_module(module, output, protocol, prepared=None):
    """Freeze all fixed arms only after that module's history comparison completes."""
    prepared = prepare_datasets() if prepared is None else prepared
    output = Path(output)
    root_output = output.parent if output.name == "models" else output
    models_output = output if output.name == "models" else output / "models"
    _, bound = _validate_protocol(protocol, prepared, root_output)
    _require_reproduction(root_output, bound)
    history_path = root_output / (module + "_history/report.json")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if (history.get("status") != "FIXED_HISTORY_COMPLETE_NO_UPGRADE_DECISION" or
            history.get("protocol_sha256") != bound or history.get("arms") != list(ARMS) or history.get("folds") != FOLDS):
        raise ValueError("fixed module history must complete before freezing challengers")
    if sha(history_path.parent / "predictions.csv") != history["predictions_sha256"]:
        raise ValueError("history prediction artifact changed")
    frame = interactions(prepared[module], module, prepared["provenance"]["sessions"])
    train = training_rows(frame, FREEZE_CUTOFF, module)
    result = {}
    directory = models_output / module
    directory.mkdir(parents=True, exist_ok=False)
    for arm in ARMS:
        fields = arm_fields(prepared[module + "_fields"], module, arm)
        cached = history_path.parent / "september_fit_cache" / arm
        manifest = json.loads((cached / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["protocol_sha256"] != bound or manifest["module"] != module or manifest["arm"] != arm or
                manifest["fields"] != fields or manifest["cutoff"] != FREEZE_CUTOFF or
                manifest["training_digest"] != _frame_digest(train, fields) or
                sha(cached / "model.joblib") != manifest["sha256"]):
            raise ValueError("Sept model cache binding drift; cannot refit a replacement")
        destination = directory / arm
        destination.mkdir(exist_ok=False)
        shutil.copy2(cached / "model.joblib", destination / "model.joblib")
        manifest = manifest | {"model_path": str((destination / "model.joblib").resolve()),
            "frozen_at": now(), "copied_from_cache": str(cached / "model.joblib"),
            "history_report_sha256": sha(history_path)}
        _write_json(destination / "manifest.json", manifest)
        result[arm] = manifest
    _write_json(directory / "report.json", {"status": "ALL_FIXED_ARMS_FROZEN_NO_UPGRADE_DECISION",
        "module": module, "protocol_sha256": bound, "manifests": result})
    return {"module": module, "manifests": result, "model_directory": str(directory)}
