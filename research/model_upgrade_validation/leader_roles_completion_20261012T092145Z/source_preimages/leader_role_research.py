"""Frozen eleven-role learning, dual-track comparisons and independent gates."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, time, timezone
import importlib.metadata
import json
from pathlib import Path
import warnings
from zoneinfo import ZoneInfo

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import leader_role_engine as e
from stock_judgement_review import metrics as binary_metrics

VERSION = "leader-role-fixed-research-1"
PARAMETERS = {"C": .25, "max_iter": 2500, "random_state": 271, "class_weight": None}
FOLDS = ({"id": "2025H2", "start": "2025-07-01", "end": "2025-12-31"},
         {"id": "2026JanAug", "start": "2026-01-01", "end": "2026-08-31"},
         {"id": "2026Sep", "start": "2026-09-01", "end": "2026-09-30"})
GATE = {"sessions": 120, "min_complete_days_per_role": 60, "min_positive": 20, "min_negative": 20,
        "coverage": .9, "improvement": .03, "block": 5, "draws": 2000,
        "seed": 20261007, "tails": [.00625, .99375], "stability_blocks": [40, 40, 40]}
REGISTRY = Path(__file__).resolve().parent/"data/leader_roles/registry.jsonl"


def masked_packet(dataset):
    """No model outputs, outcome fields or identity calls in reviewer view."""
    return {"version": VERSION, "formed_at": e.now(), "model_outputs_included": False,
        "future_outcomes_included": False, "rows": [{k: r.get(k) for k in
            ("candidate_id", "signal_date", "code", "name", "boards", "decision_at", "scope_id", "groups")}
            for r in dataset["rows"]], "source_bindings": dataset["source_bindings"],
        "blinding_verified": False, "reviewer_attestation_required": True}


def label_template(dataset, packet_path, horizon=0):
    moment = e.now()
    return {"version": VERSION, "status": "PENDING_INDEPENDENT_BLIND_REVIEW", "formed_at": moment,
        "horizon": horizon, "packet_path": str(Path(packet_path).resolve()), "packet_sha256": e.sha(packet_path),
        "source_bindings": e.bind(packet_path), "blinding_verified": False,
        "rows": [{"candidate_id": r["candidate_id"], "role": role, "value": None,
                  "reviews": [], "available_at": moment} for r in dataset["rows"] for role in e.ROLES]}


def load_labels(path, dataset, horizon=0):
    bank = e.read(path)
    if bank.get("version") != VERSION or bank.get("horizon") != horizon:
        raise ValueError("label bank version/horizon mismatch")
    e.verify(bank["source_bindings"] | {bank["packet_path"]: bank["packet_sha256"]})
    if e.stamp(bank["formed_at"]) > e.stamp(e.now()):
        raise ValueError("label bank formed in future")
    packet = e.read(bank["packet_path"])
    if packet.get("model_outputs_included") is not False or packet.get("future_outcomes_included") is not False:
        raise ValueError("identity reviewer packet is not masked")
    expected_packet = masked_packet(dataset)
    if packet["rows"] != expected_packet["rows"] or packet["source_bindings"] != dataset["source_bindings"]:
        raise ValueError("label packet changed candidate facts")
    candidates = {r["candidate_id"]: r for r in dataset["rows"]}
    expected = {(c, role) for c in candidates for role in e.ROLES}
    found = {}
    for cell in bank["rows"]:
        key = (cell["candidate_id"], cell["role"])
        if key in found or key not in expected or (cell.get("value") is not None and type(cell["value"]) is not bool):
            raise ValueError("duplicate, unexpected or invalid role label")
        reviews = cell.get("reviews", [])
        if cell["value"] is not None:
            if len(reviews) != 2 or len({r.get("reviewer_id") for r in reviews}) != 2:
                raise ValueError("two independent reviewers required for known identity")
            for r in reviews:
                if (not r.get("reviewer_id") or r.get("model_masked") is not True or r.get("future_masked") is not True
                        or r.get("independent_of_prediction") is not True or r.get("verdict") is not cell["value"]):
                    raise ValueError("known label needs independent matching blind attestations")
                refs = r.get("evidence_refs", [])
                if not refs or any(p not in dataset["source_bindings"] for p in refs):
                    raise ValueError("identity label lacks bound raw evidence")
                if e.stamp(r["reviewed_at"]) > e.stamp(cell["available_at"]):
                    raise ValueError("label made available before review")
                if e.stamp(r["reviewed_at"]) < e.stamp(packet["formed_at"]):
                    raise ValueError("review precedes masked packet")
        elif reviews and len({r.get("verdict") for r in reviews}) == 1 and reviews[0].get("verdict") is not None:
            # Reviewers may still explicitly abstain pending an unresolved check.
            if not cell.get("unknown_reason"):
                raise ValueError("known matching reviews cannot silently become unknown")
        if e.stamp(cell["available_at"]) > e.stamp(bank["formed_at"]):
            raise ValueError("label after complete bank formation")
        found[key] = cell
    if set(found) != expected:
        raise ValueError("label bank omits candidates or roles, including unknowns")
    return found


def _pipeline():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                         StandardScaler(), LogisticRegression(**PARAMETERS))


def _matrix(rows, fields=e.FEATURES):
    return np.asarray([[np.nan if r["features"].get(k) is None else r["features"][k] for k in fields]
                       for r in rows], dtype=float)


def training_rows(rows, labels, role, cutoff, days, track):
    cutoff_stamp = e.stamp(cutoff)
    cutoff_day = cutoff_stamp.astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat()
    pos = next((i for i, d in enumerate(days) if d >= cutoff_day), len(days))
    allowed = set(days[:max(0, pos-5)])
    selected = []
    for r in rows:
        label = labels[(r["candidate_id"], role)]
        if r["signal_date"] not in allowed or not r["role_eligible"][role] or label["value"] is None:
            continue
        if track == "STRICT" and (e.stamp(r["available_at"]) >= cutoff_stamp or e.stamp(label["available_at"]) >= cutoff_stamp):
            continue
        selected.append(r)
    dates = sorted({r["signal_date"] for r in selected})[-252:]
    return [r for r in selected if r["signal_date"] in dates]


class RoleModel:
    def __init__(self):
        self.models, self.logs, self.training_ids = {}, {}, {}

    def fit(self, rows, labels, cutoff, days, track):
        for role in e.ROLES:
            train = training_rows(rows, labels, role, cutoff, days, track)
            values = [labels[(r["candidate_id"], role)]["value"] for r in train]
            counts = Counter(values)
            log = {"members": len(train), "positive": counts[True], "negative": counts[False],
                   "signal_days": len({r["signal_date"] for r in train}), "attempted": False}
            if counts[True] < 20 or counts[False] < 20:
                log["status"] = "INSUFFICIENT_KNOWN_CLASSES"
            else:
                model = _pipeline()
                sizes = Counter(r["signal_date"] for r in train)
                weights = np.asarray([1/sizes[r["signal_date"]] for r in train])
                log["attempted"] = True
                try:
                    with warnings.catch_warnings(), threadpool_limits(limits=1):
                        warnings.simplefilter("error", ConvergenceWarning)
                        model.fit(_matrix(train), np.asarray(values, dtype=int), logisticregression__sample_weight=weights)
                    self.models[role] = model
                    self.training_ids[role] = [r["candidate_id"] for r in train]
                    log["status"] = "FIT_ONCE"
                except (ValueError, ConvergenceWarning) as err:
                    log.update(status="FIT_FAILED_NO_RETRY", failure=str(err))
            self.logs[role] = log
        self.track, self.cutoff = track, cutoff
        return self

    def predict(self, rows, reject_training_members=False):
        predictions = {}
        for r in rows:
            calls, scores = {}, {}
            for role in e.ROLES:
                if reject_training_members and r["candidate_id"] in self.training_ids.get(role, []):
                    raise ValueError("training member cannot supply downstream role score")
                model = self.models.get(role)
                score = float(model.predict_proba(_matrix([r]))[0, 1]) if model and r["role_eligible"][role] else None
                scores[role], calls[role] = score, score >= .5 if score is not None else None
            predictions[r["candidate_id"]] = {"calls": calls, "scores": scores}
        return predictions


def crossfit(rows, labels, cutoff, days, track):
    """Five chronological date blocks; never predict a head's training member."""
    dates = sorted({r["signal_date"] for r in rows if r["signal_date"] < cutoff[:10]})[-252:]
    output, logs = {}, []
    for block in np.array_split(np.asarray(dates), 5):
        if not len(block):
            continue
        start = str(block[0])+"T00:00:00+08:00"
        model = RoleModel().fit(rows, labels, start, days, track)
        test = [r for r in rows if r["signal_date"] in set(block)]
        output.update(model.predict(test, reject_training_members=True))
        logs.append({"start": start, "dates": block.tolist(), "fits": model.logs})
    return output, logs


def _base(dataset, candidate_id, track):
    v = dataset.get("baseline_promotion", {}).get(candidate_id, {})
    if track == "STRICT" and v.get("origin") != "SOURCE_BOUND_PREAUCTION":
        return {"call": None, "raw_score": None, "truth": None}
    return {k: v.get(k) for k in ("call", "raw_score", "truth")}


def downstream_fit(rows, scores, dataset, cutoff, days, track):
    pos = next((i for i, d in enumerate(days) if d >= cutoff[:10]), len(days))
    allowed = set(days[:max(0, pos-5)])
    eligible = []
    for r in rows:
        score = scores.get(r["candidate_id"])
        base = _base(dataset, r["candidate_id"], track)
        if (r["signal_date"] in allowed and base["truth"] is not None and r["boards"] >= 2
                and score and all(score["scores"][role] is not None for role in e.ROLES)):
            eligible.append(r)
    dates = sorted({r["signal_date"] for r in eligible})[-252:]
    eligible = [r for r in eligible if r["signal_date"] in dates]
    y = [_base(dataset, r["candidate_id"], track)["truth"] for r in eligible]
    log = {"members": len(eligible), "attempted": False}
    if len(set(y)) != 2:
        return None, dict(log, status="NO_ELIGIBLE_CROSSFITTED_ROLE_COHORT")
    x = np.asarray([[r["features"].get(k) for k in e.RELAY_FIELDS]+[scores[r["candidate_id"]]["scores"][role] for role in e.ROLES] for r in eligible], dtype=float)
    sizes = Counter(r["signal_date"] for r in eligible)
    log["attempted"] = True
    model = _pipeline()
    try:
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(x, np.asarray(y, dtype=int), logisticregression__sample_weight=np.asarray([1/sizes[r["signal_date"]] for r in eligible]))
    except (ValueError, ConvergenceWarning) as err:
        return None, dict(log, status="FIT_FAILED_NO_RETRY", failure=str(err))
    return model, dict(log, status="FIT_ONCE")


def _role_metrics(rows, labels, predictions):
    result = {}
    for role in e.ROLES:
        truth = {r["candidate_id"]: labels[(r["candidate_id"], role)]["value"] for r in rows}
        base = {r["candidate_id"]: r["roles"][role]["baseline_call"] for r in rows}
        calls = {r["candidate_id"]: predictions[r["candidate_id"]]["calls"][role] for r in rows}
        result[role] = {"BASE": binary_metrics(base, truth), "C1": binary_metrics(calls, truth)}
    return result


def topk(rows, labels, predictions):
    # Evaluate each role in its exact candidate date/scope group; ties use code.
    groups = defaultdict(list)
    for r in rows:
        groups[(r["signal_date"], r["scope_id"])].append(r)
    result = {}
    for role in e.ROLES:
        slots = []
        for group in groups.values():
            ordered = sorted(group, key=lambda r: (predictions[r["candidate_id"]]["scores"][role] is None,
                -(predictions[r["candidate_id"]]["scores"][role] or 0), r["code"]))
            known = all(labels[(r["candidate_id"], role)]["value"] is not None for r in group)
            selected = [r for r in ordered if predictions[r["candidate_id"]]["scores"][role] is not None]
            slots.append({"known_complete": known, "selected": bool(selected),
                "top1": labels[(selected[0]["candidate_id"], role)]["value"] if known and selected else None,
                "top3": any(labels[(r["candidate_id"], role)]["value"] for r in selected[:3]) if known and selected else None})
        result[role] = {"registered_groups": len(slots), "complete_label_groups": sum(s["known_complete"] for s in slots),
            "selected_groups": sum(s["selected"] for s in slots), **{k+"_hit": sum(s[k] is True for s in slots)/sum(s[k] is not None for s in slots)
                if any(s[k] is not None for s in slots) else None for k in ("top1", "top3")}}
    return result


def audit(dataset_path, labels_path=None, output=None):
    data = e.load_dataset(dataset_path)
    labels = load_labels(labels_path, data) if labels_path else None
    tracks = {track: e.analyse(data, track) for track in ("DIAGNOSTIC", "STRICT")}
    result = {"version": VERSION, "formed_at": e.now(), "status": "SOURCE_AND_LABEL_QUALIFICATION_ONLY",
        "tracks": {track: {"candidate_count": a["candidate_count"], "qualification": a["qualification"]} for track, a in tracks.items()},
        "label_support": {role: dict(Counter("UNKNOWN" if not labels or labels[(r["candidate_id"], role)]["value"] is None else
            "POSITIVE" if labels[(r["candidate_id"], role)]["value"] else "NEGATIVE" for r in data["rows"])) for role in e.ROLES},
        "source_bindings": data["source_bindings"] | e.bind(dataset_path), "fits": 0, "production_changed": False,
        "overall_acceptance": "NOT_PASSED", "new_mature_forward_samples": 0}
    if labels_path:
        result["source_bindings"] |= e.bind(labels_path)
    if output:
        e.write_new(output, result)
    return result


def freeze(dataset_path, labels_path, output, started=None):
    audit(dataset_path, labels_path)
    data = e.load_dataset(dataset_path)
    sources = data["source_bindings"] | e.bind(dataset_path) | e.bind(labels_path)
    label_bank = e.read(labels_path)
    sources |= label_bank["source_bindings"] | e.bind(label_bank["packet_path"])
    for filename in ("leader_role_engine.py", "leader_role_research.py", "leader_role_review.py"):
        sources |= e.bind(Path(__file__).with_name(filename))
    p = {"version": VERSION, "formed_at": e.now(), "batch_started_at": started or e.now(),
        "hypothesis": "Source-bound relations and temporal evidence improve multi-role identification and downstream promotion over existing gates and BASE",
        "dataset_path": str(Path(dataset_path).resolve()), "labels_path": str(Path(labels_path).resolve()),
        "roles": e.ROLES, "role_requirements": e.REQUIRED, "features": e.FEATURES,
        "arms": ["BASE", "C1", "C2"], "parameters": PARAMETERS, "window": 252,
        "folds": FOLDS, "gap_sessions": 5, "crossfit_blocks": 5, "threshold": .5,
        "horizons": [3, 5], "primary_horizon": 5, "gate": GATE,
        "budget_minutes": 30, "source_bindings": sources,
        "runtime_versions": {k: importlib.metadata.version(k) for k in ("scikit-learn", "numpy", "joblib", "pandas")},
        "review_blinding": "MASKED_PACKET_AND_TWO_ATTESTATIONS_NOT_AUTOMATIC_CERTIFICATION",
        "historical_diagnostics_are_forward": False, "production_application_allowed": False}
    e.write_new(output, p)
    return p


def load_protocol(path):
    p = e.read(path)
    if (p.get("version") != VERSION or p["roles"] != list(e.ROLES) or p["features"] != list(e.FEATURES)
            or p["parameters"] != PARAMETERS or p["gate"] != GATE or p["folds"] != list(FOLDS)
            or p["window"] != 252 or p["threshold"] != .5 or p["gap_sessions"] != 5
            or p["crossfit_blocks"] != 5 or p["primary_horizon"] != 5 or p["arms"] != ["BASE", "C1", "C2"]
            or p["role_requirements"] != {k: list(v) for k, v in e.REQUIRED.items()}):
        raise ValueError("frozen role protocol changed")
    e.verify(p["source_bindings"])
    if any(importlib.metadata.version(k) != v for k, v in p["runtime_versions"].items()):
        raise ValueError("frozen runtime changed")
    return p


def backtest(protocol_path, output_dir):
    p = load_protocol(protocol_path)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    commitment = {"protocol_sha256": e.sha(protocol_path)}
    if (out/"attempt.json").exists():
        if e.read(out/"attempt.json") != commitment:
            raise ValueError("backtest output belongs to another protocol")
    else:
        e.write_new(out/"attempt.json", commitment)
    if (out/"report.json").exists():
        raise ValueError("fixed comparison already completed; no rerun")
    started = datetime.now(timezone.utc)
    data = e.load_dataset(p["dataset_path"])
    labels = load_labels(p["labels_path"], data)
    days = e.calendar(data["calendar_path"], data["calendar_sha256"])
    results, all_predictions, fits = {}, [], 0
    for track in ("DIAGNOSTIC", "STRICT"):
        observed = e.analyse(data, track)
        rows = observed["rows"]
        results[track] = {"qualification": observed["qualification"], "folds": {}}
        for fold in FOLDS:
            key = track+"_"+fold["id"]
            done, claim = out/(key+".json"), out/(key+"_STARTED.json")
            if done.exists():
                checkpoint = e.read(done)
                if checkpoint["protocol_sha256"] != e.sha(protocol_path):
                    raise ValueError("checkpoint protocol mismatch")
                results[track]["folds"][fold["id"]] = checkpoint["result"]
                all_predictions.extend(checkpoint["predictions"]); fits += checkpoint["fits"]
                continue
            if claim.exists():
                raise ValueError("interrupted fit attempt needs audit; refusing implicit refit")
            if (datetime.now(timezone.utc)-started).total_seconds() >= 30*60:
                return {"status": "BUDGET_STOP_RESUME_UNSTARTED_CHECKPOINT", "output": str(out.resolve())}
            e.write_new(claim, {"protocol_sha256": e.sha(protocol_path), "formed_at": e.now()})
            cutoff = fold["start"]+"T00:00:00+08:00"
            model = RoleModel().fit(rows, labels, cutoff, days, track)
            test = [r for r in rows if fold["start"] <= r["signal_date"] <= fold["end"]]
            predictions = model.predict(test, reject_training_members=True)
            # Cross-fit only when all eleven heads can be trained. No empty-fit repetition.
            if len(model.models) == len(e.ROLES):
                cfs, cf_log = crossfit(rows, labels, cutoff, days, track)
                c2, c2_log = downstream_fit(rows, cfs, data, cutoff, days, track)
            else:
                cf_log, c2 = [], None
                c2_log = {"attempted": False, "status": "ALL_ROLE_HEADS_REQUIRED_FOR_C2"}
            records = []
            for r in test:
                ident = r["candidate_id"]
                b = _base(data, ident, track)
                complete = all(predictions[ident]["scores"][role] is not None for role in e.ROLES)
                score = None
                if c2 and complete:
                    x = np.asarray([[r["features"].get(k) for k in e.RELAY_FIELDS]+[predictions[ident]["scores"][role] for role in e.ROLES]], dtype=float)
                    score = float(c2.predict_proba(x)[0, 1])
                records.append({"candidate_id": ident, "signal_date": r["signal_date"], "code": r["code"],
                    "boards": r["boards"], "scope_id": r["scope_id"], "track": track, "fold": fold["id"],
                    "baseline_roles": {role: r["roles"][role]["baseline_call"] for role in e.ROLES},
                    "role_truth": {role: labels[(ident, role)]["value"] for role in e.ROLES},
                    "C1": predictions[ident], "BASE_promotion": b["call"],
                    "C2_promotion": score >= .5 if score is not None else b["call"],
                    "C2_score": score if score is not None else b["raw_score"], "promotion_truth": b["truth"],
                    "C2_fallback": None if score is not None else "ROLE_INPUT_OR_MODEL_UNAVAILABLE_USE_ORIGINAL_BASE"})
            role_metrics = _role_metrics(test, labels, predictions)
            relay_records = [r for r in records if r["boards"] >= 2]
            promotion = {arm: binary_metrics({r["candidate_id"]: r[arm+"_promotion"] for r in relay_records},
                {r["candidate_id"]: r["promotion_truth"] for r in relay_records}) for arm in ("BASE", "C2")}
            nfit = sum(l["attempted"] for l in model.logs.values())+int(c2_log["attempted"])+sum(l["attempted"] for c in cf_log for l in c["fits"].values())
            result = {"candidate_count": len(test), "role_metrics": role_metrics, "TopK": topk(test, labels, predictions),
                "promotion": promotion, "C1_fit_logs": model.logs, "crossfit_logs": cf_log, "C2_fit_log": c2_log,
                "role_macro_F1": {arm: float(np.mean([v[arm]["F1"] for v in role_metrics.values()]))
                    if all(v[arm]["F1"] is not None for v in role_metrics.values()) else None for arm in ("BASE", "C1")},
                "historical_screen_passed": False, "label_support_complete": all(v["C1"]["known_labels"] > 0 for v in role_metrics.values())}
            macro = result["role_macro_F1"]
            guards = {role: all(v["C1"][k] is not None and v["BASE"][k] is not None and v["C1"][k] >= v["BASE"][k]
                for k in ("Precision", "complete_positive_Recall", "F1"))
                and v["C1"]["scoring_coverage"] is not None and v["C1"]["scoring_coverage"] >= .9
                and sum(labels[(row["candidate_id"], role)]["value"] is True for row in test) >= 20
                and sum(labels[(row["candidate_id"], role)]["value"] is False for row in test) >= 20
                for role, v in role_metrics.items()}
            result["historical_role_guards"] = guards
            result["historical_screen_passed"] = bool(all(guards.values()) and all(x is not None for x in macro.values())
                and macro["C1"]-macro["BASE"] >= .03)
            a, b = promotion["C2"], promotion["BASE"]
            result["historical_C2_screen_passed"] = bool(c2 and a["Accuracy"] is not None and b["Accuracy"] is not None
                and a["Accuracy"]-b["Accuracy"] >= .03
                and all(a[k] is not None and b[k] is not None and a[k] >= b[k] for k in ("Precision", "complete_positive_Recall", "F1", "scoring_coverage")))
            # Persist only research artifacts. Final forward model eligibility is separate.
            if model.models:
                mpath = out/(key+"_C1.joblib"); joblib.dump(model, mpath)
                result["research_model_path"], result["research_model_sha256"] = str(mpath.resolve()), e.sha(mpath)
            if c2:
                mpath = out/(key+"_C2.joblib"); joblib.dump(c2, mpath)
                result["downstream_model_path"], result["downstream_model_sha256"] = str(mpath.resolve()), e.sha(mpath)
            e.write_new(done, {"protocol_sha256": e.sha(protocol_path), "result": result, "predictions": records, "fits": nfit})
            results[track]["folds"][fold["id"]] = result
            all_predictions.extend(records); fits += nfit
    result = {"version": VERSION, "formed_at": e.now(), "protocol_sha256": e.sha(protocol_path),
        "status": "DUAL_TRACK_COMPLETE_OVERALL_NOT_PASSED", "tracks": results, "fits": fits,
        "baseline_refits": 0, "new_mature_forward_samples": 0, "deployment_allowed": False,
        "overall_acceptance": "NOT_PASSED", "prediction_effect_verified": False,
        "future_role_metrics": {"3": None, "5": None},
        "future_role_reason": "REQUIRES_SEPARATE_MATURE_SOURCE_BOUND_BLIND_ROLE_LABELS",
        "invalidation_latency": None, "invalidation_reason": "REQUIRES_REGISTERED_ROLE_TRANSITIONS_AND_MATURE_TARGET_LABELS"}
    e.write_new(out/"full_candidate_comparison.json", all_predictions)
    result["predictions_sha256"] = e.sha(out/"full_candidate_comparison.json")
    e.write_new(out/"report.json", result)
    strict = results["STRICT"]["folds"]
    if all(v["historical_screen_passed"] and v["historical_C2_screen_passed"] for v in strict.values()):
        last = strict[FOLDS[-1]["id"]]
        e.write_new(out/"forward_model_manifest.json", {"formed_at": e.now(), "protocol_sha256": e.sha(protocol_path),
            "all_role_historical_screen_passed": True, "model_path": last["research_model_path"],
            "model_sha256": last["research_model_sha256"], "C2_model_path": last["downstream_model_path"],
            "C2_model_sha256": last["downstream_model_sha256"], "comparison_path": str((out/"report.json").resolve()),
            "comparison_sha256": e.sha(out/"report.json"), "production_allowed": False})
    return result


def paired_macro_interval(records, sessions):
    """Bootstrap pooled eleven-role F1, preserving the official date grid."""
    if not records or not sessions:
        return None
    counts = np.zeros((len(sessions), len(e.ROLES), 2, 3), dtype=float)
    positions = {day: i for i, day in enumerate(sessions)}
    for r in records:
        if r["signal_date"] not in positions:
            raise ValueError("bootstrap record outside official grid")
        day = positions[r["signal_date"]]
        for j, role in enumerate(e.ROLES):
            y = r["role_truth"][role]
            for arm, call in enumerate((r["baseline_roles"][role], r["C1"]["calls"][role])):
                counts[day, j, arm] += [int(y is True and call is True), int(y is False and call is True), int(y is True)]
    rng = np.random.default_rng(GATE["seed"])
    samples = []
    for _ in range(GATE["draws"]):
        starts = rng.integers(0, len(sessions), size=int(np.ceil(len(sessions)/5)))
        indexes = np.concatenate([(s+np.arange(5)) % len(sessions) for s in starts])[:len(sessions)]
        sampled = counts[indexes].sum(axis=0)
        tp, fp, positives = sampled[..., 0], sampled[..., 1], sampled[..., 2]
        if np.all(positives > 0):
            f1 = 2*tp/(tp+fp+positives)
            samples.append(float(np.mean(f1[:, 1])-np.mean(f1[:, 0])))
    return np.quantile(samples, GATE["tails"]).tolist() if samples else None


def registrations():
    rows = [e.read(r["path"]) for r in _registry_receipts()]
    return rows


def _registry_receipts():
    if not REGISTRY.exists():
        return []
    rows = [json.loads(line) for line in REGISTRY.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        e.verify({r["path"]: r["sha256"]})
    if len({(r["protocol_sha256"], r["signal_date"]) for r in rows}) != len(rows):
        raise ValueError("duplicate role forward registration")
    return rows


def predict(protocol_path, dataset_path, model_path, model_manifest_path, output):
    p = load_protocol(protocol_path)
    manifest = e.read(model_manifest_path)
    if (manifest.get("protocol_sha256") != e.sha(protocol_path) or manifest.get("all_role_historical_screen_passed") is not True
            or manifest.get("model_sha256") != e.sha(model_path)):
        raise ValueError("source-bound passed all-role historical model manifest required")
    e.verify({manifest["comparison_path"]: manifest["comparison_sha256"], manifest["C2_model_path"]: manifest["C2_model_sha256"]})
    comparison = e.read(manifest["comparison_path"])
    if comparison["protocol_sha256"] != e.sha(protocol_path) or not all(
            f["historical_screen_passed"] and f["historical_C2_screen_passed"] for f in comparison["tracks"]["STRICT"]["folds"].values()):
        raise ValueError("all frozen historical screens must pass before forward use")
    d = e.load_dataset(dataset_path); signals = {r["signal_date"] for r in d["rows"]}
    actual = datetime.now(ZoneInfo("Asia/Shanghai"))
    if len(signals) != 1 or next(iter(signals)) != actual.date().isoformat() or actual.time() <= time(15):
        raise ValueError("actual current official signal close required; no retro registration")
    signal = next(iter(signals)); days = e.calendar(d["calendar_path"], d["calendar_sha256"])
    if e.stamp(p["formed_at"]) >= e.stamp(signal+"T15:00:00+08:00"):
        raise ValueError("protocol/model formation must precede eligible signal close")
    if e.stamp(manifest["formed_at"]) >= e.stamp(signal+"T15:00:00+08:00"):
        raise ValueError("estimators frozen too late for this signal")
    previous = [r for r in registrations() if r["protocol_sha256"] == e.sha(protocol_path)]
    if previous and any(r.get("model_manifest_sha256") != e.sha(model_manifest_path) for r in previous):
        raise ValueError("forward estimators cannot change within frozen window")
    if any(r["protocol_sha256"] == e.sha(protocol_path) and r["signal_date"] == signal for r in _registry_receipts()):
        raise ValueError("role signal already registered")
    analysed = e.analyse(d, "STRICT"); model = joblib.load(model_path)
    if not isinstance(model, RoleModel) or set(model.models) != set(e.ROLES):
        raise ValueError("frozen complete role model required")
    calls = model.predict(analysed["rows"], reject_training_members=True)
    record = {"version": VERSION, "status": "INDEPENDENT_ROLE_FORWARD_NOT_PRODUCTION",
        "formed_at": e.now(), "protocol_path": str(Path(protocol_path).resolve()), "protocol_sha256": e.sha(protocol_path),
        "signal_date": signal, "target_dates": {str(h): e.future_day(days, signal, h) for h in (1, 3, 5)},
        "dataset_path": str(Path(dataset_path).resolve()), "source_bindings": d["source_bindings"] | e.bind(dataset_path) | e.bind(model_path) | e.bind(model_manifest_path),
        "rows": analysed["rows"], "predictions": calls, "model_manifest_sha256": e.sha(model_manifest_path), "production_changed": False}
    e.write_new(output, record)
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"path": str(Path(output).resolve()), "sha256": e.sha(output),
            "signal_date": signal, "protocol_sha256": e.sha(protocol_path)}, ensure_ascii=False)+"\n")
    return record


def settle(protocol_path, record_path, target_dataset_path, target_labels_path, horizon, output):
    load_protocol(protocol_path)
    if horizon not in (0, 3, 5):
        raise ValueError("identity or role settlement horizon must be 0, 3 or 5")
    rec = e.read(record_path)
    receipt = [r for r in _registry_receipts() if r["path"] == str(Path(record_path).resolve())]
    if len(receipt) != 1 or rec["protocol_sha256"] != e.sha(protocol_path):
        raise ValueError("original registered role record required")
    e.verify(rec["source_bindings"])
    d = e.load_dataset(target_dataset_path); target = rec["target_dates"][str(horizon)] if horizon else rec["signal_date"]
    if e.stamp(target+"T15:00:00+08:00") >= e.stamp(e.now()) or {r["signal_date"] for r in d["rows"]} != {target}:
        raise ValueError("exact matured target source required")
    labels = load_labels(target_labels_path, d)
    bycode = {r["code"]: r for r in d["rows"]}
    output_rows = []
    for row in rec["rows"]:
        other = bycode.get(row["code"])
        matching = other is not None and other["scope_id"] == row["scope_id"]
        output_rows.append({"candidate_id": row["candidate_id"], "signal_date": rec["signal_date"],
            "code": row["code"], "scope_id": row["scope_id"],
            "baseline_roles": {role: row["roles"][role]["baseline_call"] for role in e.ROLES},
            "role_truth": {role: labels[(other["candidate_id"], role)]["value"] if matching else None for role in e.ROLES},
            "C1": rec["predictions"][row["candidate_id"]],
            "missing_reason": None if matching else "LEFT_SCOPE_OR_SCOPE_CHANGED_NOT_PROOF_OF_FAILURE"})
    result = {"version": VERSION, "formed_at": e.now(), "status": "MATURE_ROLE_HORIZON_SETTLEMENT",
        "protocol_sha256": e.sha(protocol_path), "registration_sha256": e.sha(record_path),
        "signal_date": rec["signal_date"], "target_date": target, "horizon": horizon, "rows": output_rows,
        "source_bindings": rec["source_bindings"] | d["source_bindings"] | e.bind(record_path) | e.bind(target_dataset_path) | e.bind(target_labels_path),
        "new_mature_role_observation_rows": len(output_rows), "original_production_samples_added": 0}
    e.write_new(output, result)
    return result


def review(protocol_path, settlements, calendar_path, output):
    p = load_protocol(protocol_path); days = e.calendar(calendar_path)
    regs = [r for r in registrations() if r["protocol_sha256"] == e.sha(protocol_path)]
    events, seen = [], set()
    for path in settlements:
        s = e.read(path); e.verify(s["source_bindings"])
        key = (s["registration_sha256"], s["horizon"])
        if key in seen or s["protocol_sha256"] != e.sha(protocol_path):
            raise ValueError("duplicate or wrong protocol settlement")
        seen.add(key); events.append(s)
    first = min((r["signal_date"] for r in regs), default=None)
    grid = days[days.index(first):days.index(first)+120] if first else []
    mature = len(grid) == 120 and e.stamp(e.future_day(days, grid[-1], 5)+"T15:00:00+08:00") < e.stamp(e.now())
    records = [r for s in events if s["horizon"] == 0 and s["signal_date"] in grid for r in s["rows"]]
    registry_rows = {r["candidate_id"]: r for reg in regs if reg["signal_date"] in grid for r in reg["rows"]}
    actual = {r["candidate_id"]: r for r in records}
    # Every registered unresolved candidate stays in denominator.
    for key, r in registry_rows.items():
        if key not in actual:
            actual[key] = {"candidate_id": key, "signal_date": r["signal_date"], "code": r["code"],
                "baseline_roles": {role: r["roles"][role]["baseline_call"] for role in e.ROLES},
                "role_truth": {role: None for role in e.ROLES},
                "C1": next(reg["predictions"][key] for reg in regs if key in reg["predictions"])}
    records = list(actual.values()); metrics, checks = {}, {}
    for role in e.ROLES:
        truth = {r["candidate_id"]: r["role_truth"][role] for r in records}
        metrics[role] = {arm: binary_metrics({r["candidate_id"]: r["baseline_roles"][role] if arm == "BASE" else r["C1"]["calls"][role] for r in records}, truth)
                         for arm in ("BASE", "C1")}
        a, b = metrics[role]["C1"], metrics[role]["BASE"]
        complete_days = sum(bool([r for r in records if r["signal_date"] == day]) and
            all(r["role_truth"][role] is not None and r["C1"]["calls"][role] is not None for r in records if r["signal_date"] == day) for day in grid)
        checks[role] = {"complete_days": complete_days, "support": sum(v is True for v in truth.values()) >= 20 and sum(v is False for v in truth.values()) >= 20,
            "coverage": a["scoring_coverage"] is not None and a["scoring_coverage"] >= .9,
            "nondegradation": all(a[k] is not None and b[k] is not None and a[k] >= b[k] for k in ("Precision", "complete_positive_Recall", "F1")),
            "complete_days_pass": complete_days >= 60}
    macro = {arm: float(np.mean([m[arm]["F1"] for m in metrics.values()])) if all(m[arm]["F1"] is not None for m in metrics.values()) else None for arm in ("BASE", "C1")}
    delta = macro["C1"]-macro["BASE"] if all(v is not None for v in macro.values()) else None
    interval = paired_macro_interval(records, grid) if mature and records and delta is not None else None
    future_metrics = {}
    future_passed = True
    for h in (3, 5):
        rows_h = {r["candidate_id"]: r for s in events if s["horizon"] == h and s["signal_date"] in grid for r in s["rows"]}
        filled = [rows_h.get(r["candidate_id"], dict(r, role_truth={role: None for role in e.ROLES})) for r in records]
        byrole = {}
        for role in e.ROLES:
            truth = {r["candidate_id"]: r["role_truth"][role] for r in filled}
            byrole[role] = {arm: binary_metrics({r["candidate_id"]: r["baseline_roles"][role] if arm == "BASE" else r["C1"]["calls"][role] for r in filled}, truth) for arm in ("BASE", "C1")}
        future_metrics[str(h)] = byrole
        if h == 5:
            future_passed = all(m["C1"]["scoring_coverage"] is not None and m["C1"]["scoring_coverage"] >= .9
                and all(m["C1"][k] is not None and m["BASE"][k] is not None and m["C1"][k] >= m["BASE"][k] for k in ("Precision", "complete_positive_Recall", "F1")) for m in byrole.values())
    passed = bool(mature and len({r["signal_date"] for r in regs if r["signal_date"] in grid}) == 120
        and all(all(v for k, v in c.items() if k != "complete_days") for c in checks.values())
        and delta is not None and delta >= .03 and interval and interval[0] > 0)
    passed = passed and future_passed
    stability = []
    for start in (0, 40, 80):
        subrows = [r for r in records if r["signal_date"] in grid[start:start+40]]
        changes = []
        for role in e.ROLES:
            ys = {r["candidate_id"]: r["role_truth"][role] for r in subrows}
            a = binary_metrics({r["candidate_id"]: r["C1"]["calls"][role] for r in subrows}, ys)["F1"]
            b = binary_metrics({r["candidate_id"]: r["baseline_roles"][role] for r in subrows}, ys)["F1"]
            changes.append(a-b if a is not None and b is not None else None)
        stability.append(float(np.mean(changes)) if all(x is not None for x in changes) else None)
    passed = passed and all(x is not None and x >= 0 for x in stability)
    result = {"version": VERSION, "formed_at": e.now(), "status": "ROLE_GATE_PASSED_REQUIRES_SEPARATE_C2_GATE" if passed else "OVERALL_NOT_PASSED",
        "protocol_sha256": e.sha(protocol_path), "grid": grid, "window_mature": mature, "metrics": metrics,
        "checks": checks, "macro_F1": macro, "delta": delta, "interval": interval,
        "role_gate_passed": passed, "C2_original_gate_passed": False, "deployment_allowed": False,
        "future_role_metrics": future_metrics, "primary_future_horizon": 5,
        "three40_macro_F1_deltas": stability,
        "C2_gate_reason": "Original source-bound promotion forward records and original gate receipt required separately",
        "missing_types": [k for k, c in checks.items() if not all(v for name, v in c.items() if name != "complete_days")],
        "source_bindings": e.bind(protocol_path) | e.bind(calendar_path) | {str(Path(f).resolve()): e.sha(f) for f in settlements}}
    e.write_new(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit"); a.add_argument("--dataset", required=True); a.add_argument("--labels"); a.add_argument("--output", required=True)
    a = sub.add_parser("freeze")
    for k in ("dataset", "labels", "output"):
        a.add_argument("--"+k, required=True)
    a = sub.add_parser("backtest"); a.add_argument("--protocol", required=True); a.add_argument("--output-dir", required=True)
    a = sub.add_parser("predict")
    for k in ("protocol", "dataset", "model", "model-manifest", "output"):
        a.add_argument("--"+k, required=True)
    a = sub.add_parser("settle")
    for k in ("protocol", "record", "target-dataset", "target-labels", "output"):
        a.add_argument("--"+k, required=True)
    a.add_argument("--horizon", type=int, choices=(0, 3, 5), required=True)
    a = sub.add_parser("review"); a.add_argument("--protocol", required=True); a.add_argument("--settlements", nargs="*", default=[])
    a.add_argument("--calendar", required=True); a.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "audit": result = audit(args.dataset, args.labels, args.output)
    elif args.command == "freeze": result = freeze(args.dataset, args.labels, args.output)
    elif args.command == "backtest": result = backtest(args.protocol, args.output_dir)
    elif args.command == "predict": result = predict(args.protocol, args.dataset, args.model, args.model_manifest, args.output)
    elif args.command == "settle": result = settle(args.protocol, args.record, args.target_dataset, args.target_labels, args.horizon, args.output)
    else: result = review(args.protocol, args.settlements, args.calendar, args.output)
    print(json.dumps({k: result.get(k) for k in ("status", "fits", "overall_acceptance", "deployment_allowed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
