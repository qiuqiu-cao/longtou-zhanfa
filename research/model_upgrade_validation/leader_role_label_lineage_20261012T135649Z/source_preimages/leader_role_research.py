"""Frozen eleven-role learning, dual-track comparisons and independent gates."""
from __future__ import annotations

import argparse
import copy
import hashlib
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
STRICT_PLAN_VERSION = "leader-role-prospective-date-plan-1"
ROLE_CHECKLIST = {
    "HEIGHT": "完整指定池板位、并列对象与范围；最高板不直接认证总龙",
    "THEME": "已核具体题材成员、去自身扩散、相对强弱与竞争顺序",
    "TURNOVER": "实际完整分歧/回封路径；日放量和首末封间隔不认证承接",
    "CAPACITY": "同题材包含非涨停成员的成交份额、流通规模与范围",
    "TREND": "同题材非涨停成员、连续历史行情与相对趋势",
    "POSITION": "完整同身位竞争对象的先后实际状态；离池不等于失败",
    "SUPPLEMENT": "已核旧核心、具体题材关系、启动先后、旧核心阶段与低位竞争",
    "NODE": "事前市场阶段转折、题材延续与完整同期竞争",
    "CYCLE": "连续市场阶段记录、题材核心延续与反馈",
    "CROSS_CYCLE": "事前压力事件、事件前身份、跨事件存活与重新领先的时序",
    "EMOTION": "个股变化领先生态反馈、市场/共同催化条件控制；联动不证明因果",
}


def _evaluation(truth, call):
    if truth is None:
        return "TRUTH_UNKNOWN"
    if call is None:
        return "POSITIVE_ABSTENTION" if truth else "NEGATIVE_ABSTENTION"
    if truth:
        return "TRUE_POSITIVE" if call else "FALSE_NEGATIVE"
    return "FALSE_POSITIVE" if call else "TRUE_NEGATIVE"


def role_error_diagnosis(records, fit_logs=None):
    """All candidates and roles, without inventing identities from promotion."""
    cases, counts, promotion_counts = [], defaultdict(Counter), Counter()
    for record in records:
        roles = {}
        for role in e.ROLES:
            truth, call = record["role_truth"][role], record["C1"]["calls"][role]
            category = _evaluation(truth, call)
            missing = record.get("missing_role_inputs", {}).get(role)
            eligible = record.get("role_input_eligible", {}).get(role)
            log = (fit_logs or {}).get((record.get("track"), record.get("fold")), {}).get(role, {})
            reasons = []
            if truth is None:
                reasons.append("INDEPENDENT_ROLE_TRUTH_UNAVAILABLE")
            if eligible is False:
                reasons.append("REQUIRED_ROLE_INPUTS_UNAVAILABLE")
            if log and log.get("status") != "FIT_ONCE":
                reasons.append("ROLE_HEAD_NOT_FITTED:"+str(log.get("status")))
            if call is True and eligible is False:
                reasons.append("ROLE_CALL_WITHOUT_REQUIRED_INPUTS")
            if category in ("FALSE_NEGATIVE", "FALSE_POSITIVE"):
                reasons.append("MODEL_CLASSIFICATION_ERROR_REQUIRES_FIXED_TEST")
            elif category == "POSITIVE_ABSTENTION":
                reasons.append("POSITIVE_MISSED_BY_ABSTENTION_COUNTS_IN_COMPLETE_RECALL")
            if missing is None:
                reasons.append("ORIGINAL_COMPARISON_DID_NOT_RETAIN_ROLE_INPUT_GAPS")
            counts[(record.get("track"), record.get("fold"), role)][category] += 1
            roles[role] = {"evaluation": category, "truth": truth, "call": call,
                "internal_score": record["C1"]["scores"].get(role), "input_eligible": eligible,
                "score_explanation": record["C1"].get("explanations", {}).get(role),
                "missing_inputs": missing, "reasons": reasons,
                "next_action": "OBTAIN_INDEPENDENT_ROLE_LABEL" if truth is None else
                    "REBUILD_MISSING_SOURCE_FACTS" if eligible is False else
                    "QUALIFY_FIXED_TRAINING_MEMBERS" if log and log.get("status") != "FIT_ONCE" else
                    "REVIEW_SUPPORT_COUNTEREVIDENCE_AND_PREREGISTER_ONE_HYPOTHESIS" if category in
                        ("FALSE_NEGATIVE", "FALSE_POSITIVE", "POSITIVE_ABSTENTION") else "RETAIN_CASE_AND_CHECK_NEXT_BATCH"}
        promo = {arm: _evaluation(record.get("promotion_truth"), record.get(arm+"_promotion")) for arm in ("BASE", "C2")}
        promotion_counts.update(arm+":"+value for arm, value in promo.items())
        cases.append({k: record.get(k) for k in ("candidate_id", "signal_date", "code", "boards", "scope_id", "track", "fold")}
            | {"roles": roles, "promotion_evaluation": promo, "C2_fallback": record.get("C2_fallback"),
               "promotion_miss_does_not_establish_role_identity": True})
    return {"version": VERSION, "formed_at": e.now(), "status": "COMPLETE_COHORT_ROLE_ERROR_DIAGNOSIS",
        "candidate_records": len(records), "role_cells": len(records)*len(e.ROLES), "cases": cases,
        "summary": [{"track": track, "fold": fold, "role": role, "counts": dict(v)}
                    for (track, fold, role), v in sorted(counts.items())],
        "promotion_summary": dict(promotion_counts), "role_checklist": ROLE_CHECKLIST,
        "causal_explanation_verified": False, "fits": 0, "new_real_forward_samples": 0,
        "prediction_effect_verified": False, "deployment_allowed": False}


def diagnose_saved_comparison(report_path, protocol_path, output):
    report, protocol = e.read(report_path), e.read(protocol_path)
    if report.get("protocol_sha256") != e.sha(protocol_path):
        raise ValueError("saved comparison protocol hash mismatch")
    matrix = Path(report_path).with_name("full_candidate_comparison.json")
    e.verify({str(matrix): report["predictions_sha256"]})
    records = e.read(matrix)
    logs = {(track, fold): result["C1_fit_logs"] for track, value in report["tracks"].items()
            for fold, result in value["folds"].items()}
    diagnosis = role_error_diagnosis(records, logs)
    diagnosis["source_bindings"] = e.bind(report_path) | e.bind(protocol_path) | e.bind(matrix)
    diagnosis["original_source_binding_changes"] = [path for path, expected in protocol["source_bindings"].items()
        if not Path(path).is_file() or e.sha(path) != expected]
    diagnosis["saved_artifact_analysis_only_not_comparison_rerun"] = True
    e.write_new(output, diagnosis)
    return diagnosis


def masked_packet(dataset):
    """Expose rebuilt facts, never links to files containing scores/outcomes."""
    rows = []
    for raw in dataset["rows"]:
        row = {k: copy.deepcopy(raw.get(k)) for k in
               ("candidate_id", "signal_date", "code", "name", "boards", "decision_at", "scope_id", "groups")}
        for group in row["groups"].values():
            # Source verification happens in the operator loader. A raw archive
            # can also contain future labels or model outputs outside this group.
            group.pop("source_path", None)
            group.pop("pointer", None)
        rows.append(row)
    digest = hashlib.sha256(json.dumps(dataset["source_bindings"], sort_keys=True,
                                      ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"version": VERSION, "formed_at": e.now(), "model_outputs_included": False,
        "future_outcomes_included": False, "mask_schema": "FACTS_ONLY_NO_ARCHIVE_LINKS_V2",
        "rows": rows, "source_bindings": {}, "operator_source_bindings_sha256": digest,
        "blinding_verified": False, "reviewer_attestation_required": True}


def label_template(dataset, packet_path, horizon=0):
    moment = e.now()
    return {"version": VERSION, "status": "PENDING_INDEPENDENT_BLIND_REVIEW", "formed_at": moment,
        "horizon": horizon, "packet_path": str(Path(packet_path).resolve()), "packet_sha256": e.sha(packet_path),
        "source_bindings": e.bind(packet_path), "blinding_verified": False,
        "rows": [{"candidate_id": r["candidate_id"], "role": role, "value": None,
                  "reviews": [], "available_at": moment} for r in dataset["rows"] for role in e.ROLES]}


def merge_identity_reviews(dataset_path, packet_path, review_paths, output):
    """Archive actual independent reviews; unmasked consensus stays unknown."""
    if len(review_paths) != 2:
        raise ValueError("exactly two independent review files required")
    data = e.load_dataset(dataset_path)
    bank = label_template(data, packet_path)
    packet, expected = e.read(packet_path), masked_packet(data)
    for key in ("rows", "mask_schema", "source_bindings", "operator_source_bindings_sha256"):
        if packet.get(key) != expected[key]:
            raise ValueError("review packet differs from rebuilt masked facts")
    expected_cells = {(row["candidate_id"], role) for row in data["rows"] for role in e.ROLES}
    review_ids, reviewers = set(), []
    moment = e.now()
    for path in review_paths:
        receipt = e.read(path)
        reviewer_id = receipt.get("reviewer_id")
        if not reviewer_id or reviewer_id in review_ids:
            raise ValueError("two different actual reviewers required")
        review_ids.add(reviewer_id)
        if (not receipt.get("packet_path") or str(Path(receipt["packet_path"]).resolve()) != bank["packet_path"]
                or receipt.get("packet_sha256") != bank["packet_sha256"]):
            raise ValueError("reviewer packet hash/path mismatch")
        if not (e.stamp(packet["formed_at"]) <= e.stamp(receipt["review_started_at"])
                <= e.stamp(receipt["reviewed_at"]) <= e.stamp(moment)):
            raise ValueError("actual review time inconsistent with packet")
        cells = {}
        for original_cell in receipt["rows"]:
            cell = copy.deepcopy(original_cell)
            cell["evidence_refs"] = [str(Path(p).resolve()) for p in cell.get("evidence_refs", [])]
            key = (cell["candidate_id"], cell["role"])
            if (key not in expected_cells or key in cells or not cell.get("basis")
                    or cell.get("verdict") is not None and type(cell["verdict"]) is not bool
                    or cell.get("evidence_refs") != [bank["packet_path"]]):
                raise ValueError("invalid or incomplete actual review cell")
            cells[key] = cell
        if set(cells) != expected_cells:
            raise ValueError("review omits candidates or roles")
        eligible = (receipt.get("model_masked") is True and receipt.get("future_masked") is True
                    and receipt.get("independent_of_prediction") is True
                    and receipt.get("formal_blind_label_eligible") is not False
                    and receipt.get("inherited_AGENTS_contains_mature_outcome_summary") is not True)
        reviewers.append((receipt, cells, eligible))
        bank["source_bindings"] |= e.bind(path)
        if receipt.get("supersedes_path"):
            e.verify({receipt["supersedes_path"]: receipt["supersedes_sha256"]})
            bank["source_bindings"] |= e.bind(receipt["supersedes_path"])
    for cell in bank["rows"]:
        key = (cell["candidate_id"], cell["role"])
        for receipt, cells, eligible in reviewers:
            cell["reviews"].append(cells[key] | {k: receipt.get(k) for k in
                ("reviewer_id", "reviewed_at", "model_masked", "future_masked", "independent_of_prediction")})
        verdicts = [cells[key]["verdict"] for _, cells, _ in reviewers]
        if all(eligible for _, _, eligible in reviewers) and verdicts[0] is not None and verdicts[0] is verdicts[1]:
            cell["value"] = verdicts[0]
        else:
            cell["unknown_reason"] = ("REVIEW_CONTEXT_NOT_BLINDED" if not all(x[2] for x in reviewers)
                else "REVIEW_DISAGREEMENT" if verdicts[0] is not verdicts[1] else "INSUFFICIENT_ROLE_EVIDENCE")
        cell["available_at"] = moment
    bank.update(formed_at=moment, status="ACTUAL_REVIEWS_IMPORTED_UNQUALIFIED_CELLS_UNKNOWN",
                reviewer_ids=sorted(review_ids), blinding_verified=False)
    e.write_new(output, bank)
    load_labels(output, data)
    return bank


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
    if (packet.get("mask_schema") != expected_packet["mask_schema"]
            or packet["rows"] != expected_packet["rows"] or packet["source_bindings"] != {}
            or packet.get("operator_source_bindings_sha256") != expected_packet["operator_source_bindings_sha256"]):
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
                if not refs or any(p != bank["packet_path"] for p in refs):
                    raise ValueError("identity label lacks bound masked fact evidence")
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


def metric_not_lower(candidate, baseline, name):
    a, b = candidate[name], baseline[name]
    # Keep the null reported precision; this is only a gate reference for a
    # baseline making no positive calls, not an invented measured precision.
    if name == "Precision" and b is None and baseline["known_labels"] > 0 and baseline["TP"]+baseline["FP"] == 0:
        b = 0.
    return a is not None and b is not None and a >= b-1e-12


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
            calls, scores, explanations = {}, {}, {}
            for role in e.ROLES:
                if reject_training_members and r["candidate_id"] in self.training_ids.get(role, []):
                    raise ValueError("training member cannot supply downstream role score")
                model = self.models.get(role)
                score = float(model.predict_proba(_matrix([r]))[0, 1]) if model and r["role_eligible"][role] else None
                scores[role], calls[role] = score, score >= .5 if score is not None else None
                explanations[role] = self.explain(r, role) if score is not None else None
            predictions[r["candidate_id"]] = {"calls": calls, "scores": scores, "explanations": explanations}
        return predictions

    def explain(self, row, role):
        """Exact fixed-model logit contributions, not market causal effects."""
        model = self.models.get(role)
        if model is None or not row["role_eligible"][role]:
            return None
        x = _matrix([row]); transformed = model[:-1].transform(x)[0]
        classifier = model.steps[-1][1]
        names = model.steps[0][1].get_feature_names_out(list(e.FEATURES))
        parts = transformed*classifier.coef_[0]
        intercept = float(classifier.intercept_[0])
        logit = float(model.decision_function(x)[0])
        if not np.isclose(intercept+parts.sum(), logit, rtol=1e-10, atol=1e-10):
            raise ValueError("role score explanation does not reproduce frozen logit")
        strongest = sorted(range(len(parts)), key=lambda i: (-abs(parts[i]), str(names[i])))[:5]
        return {"intercept": intercept, "logit": logit, "sum_feature_contributions": float(parts.sum()),
            "judgement_logit_threshold": 0., "top_factors": [{"feature": str(names[i]),
                "transformed_value": float(transformed[i]), "logit_contribution": float(parts[i])} for i in strongest],
            "missing_original_features": [key for key in e.FEATURES if row["features"].get(key) is None],
            "not_causal_effect": True, "not_calibrated_probability": True}


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
    trusted = dataset.get("adapter", {}).get("kind") in ("SAVED_HISTORY", "AUGMENTED", "MULTI_ARCHIVE")
    row = next((r for r in dataset["rows"] if r["candidate_id"] == candidate_id), None)
    if not trusted or (track == "STRICT" and (v.get("origin") != "SOURCE_BOUND_PREAUCTION" or not (dataset.get("baseline_binding") or dataset.get("adapter", {}).get("kind") == "MULTI_ARCHIVE")
            or not row or e.stamp(v["available_at"]) > e.stamp(row["decision_at"]))):
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
            if track == "STRICT":
                cell = dataset["baseline_promotion"][r["candidate_id"]]
                if not cell.get("truth_available_at") or e.stamp(cell["truth_available_at"]) >= e.stamp(cutoff):
                    continue
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
    annual = {}
    for year in sorted({row["signal_date"][:4] for row in data["rows"]}):
        raw = [row for row in data["rows"] if row["signal_date"].startswith(year)]
        dates = sorted({row["signal_date"] for row in raw})
        annual[year] = {"candidate_count": len(raw), "archived_dates": len(dates),
            "first_date": dates[0], "last_date": dates[-1],
            "scope_counts": dict(Counter(row["scope_id"] for row in raw)),
            "saved_BASE_prediction_count": sum(row["candidate_id"] in data.get("baseline_promotion", {}) for row in raw),
            "label_support": {role: dict(Counter("UNKNOWN" if not labels or
                labels[(row["candidate_id"], role)]["value"] is None else "POSITIVE" if
                labels[(row["candidate_id"], role)]["value"] else "NEGATIVE" for row in raw)) for role in e.ROLES},
            "tracks": {}}
        for track, observed in tracks.items():
            rows = [row for row in observed["rows"] if row["signal_date"].startswith(year)]
            annual[year]["tracks"][track] = {
                "role_input_eligible": {role: sum(row["role_eligible"][role] for row in rows) for role in e.ROLES},
                "height_temporal_available": {field: sum(row["features"][field] is not None for row in rows)
                                               for field in e.HEIGHT_TEMPORAL_FIELDS}}
    result["annual_summary"] = annual
    result["height_temporal_records"] = [{k: row[k] for k in ("candidate_id", "signal_date", "code", "scope_id")}
        | {"track": track, "facts": {field: row["features"][field] for field in
            ("scope_max_boards", "scope_top_count", *e.HEIGHT_TEMPORAL_FIELDS)}}
        for track, observed in tracks.items() for row in observed["rows"]]
    result["height_temporal_facts_are_independent_identity_labels"] = False
    result["source_bindings"] |= e.bind(Path(e.__file__)) | e.bind(Path(__file__))
    if labels_path:
        result["source_bindings"] |= e.bind(labels_path)
    if output:
        e.write_new(output, result)
    return result


def _validate_future_folds(folds, days, formed_at):
    if not isinstance(folds, list) or len(folds) != 3:
        raise ValueError("three preregistered strict development folds required")
    actual_day = e.stamp(formed_at).astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat()
    ids, last_end = set(), None
    for fold in folds:
        if set(fold) != {"id", "start", "end"} or not isinstance(fold["id"], str) or not fold["id"]:
            raise ValueError("strict fold schema invalid")
        if fold["id"] in ids or fold["id"] in {f["id"] for f in FOLDS}:
            raise ValueError("strict development fold ids must be distinct from historical folds")
        ids.add(fold["id"])
        if fold["start"] not in days or fold["end"] not in days or fold["start"] > fold["end"]:
            raise ValueError("strict development dates require verified official calendar")
        if fold["start"] <= actual_day:
            raise ValueError("strict development plan must precede every evaluation date")
        if last_end is not None and days.index(fold["start"])-days.index(last_end)-1 < 5:
            raise ValueError("strict development folds require five official-session embargo")
        e.future_day(days, fold["end"], 5)
        last_end = fold["end"]


def plan_strict(dataset_path, labels_path, fold_spec_path, output=None, formed_at=None):
    """Commit future development dates only after actual warmup qualification."""
    data = e.load_dataset(dataset_path); labels = load_labels(labels_path, data)
    spec = e.read(fold_spec_path)
    if set(spec) != {"folds"}:
        raise ValueError("date specification cannot change parameters or gates")
    moment = formed_at or e.now(); days = e.calendar(data["calendar_path"], data["calendar_sha256"])
    bank = e.read(labels_path)
    if e.stamp(data["formed_at"]) > e.stamp(moment) or e.stamp(bank["formed_at"]) > e.stamp(moment):
        raise ValueError("warmup archive did not exist at strict date preregistration")
    _validate_future_folds(spec["folds"], days, moment)
    rows = e.analyse(data, "STRICT")["rows"]
    cutoff = spec["folds"][0]["start"]+"T00:00:00+08:00"
    counts = {}
    for role in e.ROLES:
        train = training_rows(rows, labels, role, cutoff, days, "STRICT")
        values = Counter(labels[(row["candidate_id"], role)]["value"] for row in train)
        counts[role] = {"positive": values[True], "negative": values[False], "members": len(train),
                        "qualified": values[True] >= 20 and values[False] >= 20}
    # This is source/label readiness, never a substitute for actual cross-fitting.
    pos = days.index(spec["folds"][0]["start"]); allowed = set(days[:max(0, pos-5)])
    promotion = Counter()
    for row in rows:
        if row["signal_date"] not in allowed or row["boards"] < 2:
            continue
        base = _base(data, row["candidate_id"], "STRICT")
        cell = data.get("baseline_promotion", {}).get(row["candidate_id"], {})
        if (base["truth"] is not None and cell.get("truth_available_at")
                and e.stamp(cell["truth_available_at"]) <= e.stamp(moment)):
            promotion[base["truth"]] += 1
    ready = all(c["qualified"] for c in counts.values()) and promotion[True] > 0 and promotion[False] > 0
    bank = e.read(labels_path)
    sources = data["source_bindings"] | e.bind(dataset_path) | e.bind(labels_path) | bank["source_bindings"] | e.bind(bank["packet_path"])
    sources |= e.bind(fold_spec_path) | e.bind(data["calendar_path"])
    for name in ("leader_role_engine.py", "leader_role_research.py", "leader_role_review.py"):
        sources |= e.bind(Path(__file__).with_name(name))
    result = {"version": STRICT_PLAN_VERSION, "formed_at": moment,
        "status": "FROZEN_FUTURE_STRICT_DEVELOPMENT" if ready else "NOT_FREEZABLE_TRAINING_QUALIFICATION_MISSING",
        "folds": spec["folds"], "fold_spec_path": str(Path(fold_spec_path).resolve()),
        "calendar_path": data["calendar_path"], "calendar_sha256": data["calendar_sha256"],
        "warmup_dataset_path": str(Path(dataset_path).resolve()), "warmup_labels_path": str(Path(labels_path).resolve()),
        "training_support": counts, "promotion_label_support": {"positive": promotion[True], "negative": promotion[False]},
        "parameters": PARAMETERS, "features": e.FEATURES, "roles": e.ROLES, "gate": GATE,
        "window": 252, "threshold": .5, "gap_sessions": 5, "crossfit_blocks": 5,
        "candidate_policy": "COMPLETE_SOURCE_BOUND_DAILY_ROSTERS_KEEP_FAILED_REAR_UNKNOWN",
        "source_bindings": sources, "fits": 0, "deployment_allowed": False,
        "real_forward_samples": 0, "actual_crossfit_still_required": True}
    if output:
        e.write_new(output, result)
    return result


def load_strict_plan(path):
    plan = e.read(path)
    if plan.get("version") != STRICT_PLAN_VERSION or plan.get("status") != "FROZEN_FUTURE_STRICT_DEVELOPMENT":
        raise ValueError("qualified frozen future strict date plan required")
    if (plan["parameters"] != PARAMETERS or plan["gate"] != GATE or plan["features"] != list(e.FEATURES)
            or plan["roles"] != list(e.ROLES) or plan["window"] != 252 or plan["threshold"] != .5
            or plan["gap_sessions"] != 5 or plan["crossfit_blocks"] != 5):
        raise ValueError("strict development plan changed fixed parameters or acceptance")
    e.verify(plan["source_bindings"])
    _validate_future_folds(plan["folds"], e.calendar(plan["calendar_path"], plan["calendar_sha256"]), plan["formed_at"])
    if e.read(plan["fold_spec_path"]) != {"folds": plan["folds"]}:
        raise ValueError("strict development date commitment changed")
    rebuilt = plan_strict(plan["warmup_dataset_path"], plan["warmup_labels_path"], plan["fold_spec_path"],
                          formed_at=plan["formed_at"])
    if (rebuilt["status"] != plan["status"] or rebuilt["training_support"] != plan["training_support"]
            or rebuilt["promotion_label_support"] != plan["promotion_label_support"]
            or rebuilt["source_bindings"] != plan["source_bindings"]):
        raise ValueError("strict plan warmup qualification cannot be certified by flags alone")
    return plan


def validation_folds(protocol, track):
    return protocol.get("strict_folds", list(FOLDS)) if track == "STRICT" else list(FOLDS)


def strict_comparison_passed(protocol, comparison):
    folds = comparison.get("tracks", {}).get("STRICT", {}).get("folds", {})
    expected = {fold["id"] for fold in validation_folds(protocol, "STRICT")}
    return (set(folds) == expected and bool(expected) and all(
        value.get("historical_screen_passed") is True and value.get("historical_C2_screen_passed") is True
        for value in folds.values()))


def freeze(dataset_path, labels_path, output, started=None, strict_plan_path=None):
    audit(dataset_path, labels_path)
    data = e.load_dataset(dataset_path)
    sources = data["source_bindings"] | e.bind(dataset_path) | e.bind(labels_path)
    label_bank = e.read(labels_path)
    sources |= label_bank["source_bindings"] | e.bind(label_bank["packet_path"])
    for filename in ("leader_role_engine.py", "leader_role_research.py", "leader_role_review.py"):
        sources |= e.bind(Path(__file__).with_name(filename))
    for filename in ("model_upgrade_metrics.py", "relay_context_shadow.py", "relay_context_research.py", "model_upgrade_compat.py"):
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
    if strict_plan_path:
        plan = load_strict_plan(strict_plan_path)
        if (data["calendar_path"], data["calendar_sha256"]) != (plan["calendar_path"], plan["calendar_sha256"]):
            raise ValueError("strict plan and comparison calendar differ")
        warmup = e.load_dataset(plan["warmup_dataset_path"])
        indexed = {row["candidate_id"]: row for row in data["rows"]}
        if any(indexed.get(row["candidate_id"]) != row for row in warmup["rows"]):
            raise ValueError("strict comparison changed frozen warmup candidates or inputs")
        p.update(strict_plan_path=str(Path(strict_plan_path).resolve()), strict_folds=plan["folds"],
                 strict_route="PREREGISTERED_FUTURE_DATES_SOURCE_TIME_VALIDATION_NOT_REAL_FORWARD")
        p["source_bindings"] |= plan["source_bindings"] | e.bind(strict_plan_path)
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
    if p.get("strict_plan_path"):
        plan = load_strict_plan(p["strict_plan_path"])
        if p.get("strict_folds") != plan["folds"]:
            raise ValueError("strict protocol dates differ from preregistration")
    elif p.get("strict_folds"):
        raise ValueError("strict dates cannot change without a qualified preregistration")
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
        for fold in validation_folds(p, track):
            if (p.get("strict_plan_path") and track == "STRICT"
                    and datetime.now(ZoneInfo("Asia/Shanghai")) <= datetime.combine(
                        datetime.fromisoformat(e.future_day(days, fold["end"], 5)).date(), time(15), ZoneInfo("Asia/Shanghai"))):
                return {"status": "WAIT_PREREGISTERED_STRICT_FOLD_LABEL_WINDOW", "output": str(out.resolve()), "fits": fits}
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
                    "role_input_eligible": r["role_eligible"],
                    "missing_role_inputs": {role: r["roles"][role]["missing"] for role in e.ROLES},
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
            guards = {role: all(metric_not_lower(v["C1"], v["BASE"], k)
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
            diagnosis = role_error_diagnosis(records, {(track, fold["id"]): model.logs})
            result["role_error_summary"] = diagnosis["summary"]
            result["promotion_error_summary"] = diagnosis["promotion_summary"]
            e.write_new(done, {"protocol_sha256": e.sha(protocol_path), "result": result, "predictions": records, "fits": nfit})
            results[track]["folds"][fold["id"]] = result
            all_predictions.extend(records); fits += nfit
    result = {"version": VERSION, "formed_at": e.now(), "protocol_sha256": e.sha(protocol_path),
        "status": "DUAL_TRACK_COMPLETE_OVERALL_NOT_PASSED", "tracks": results, "fits": fits,
        "baseline_refits": 0, "new_mature_forward_samples": 0, "deployment_allowed": False,
        "overall_acceptance": "NOT_PASSED", "prediction_effect_verified": False,
        "future_role_metrics": {"3": None, "5": None},
        "future_role_reason": "REQUIRES_SEPARATE_MATURE_SOURCE_BOUND_BLIND_ROLE_LABELS",
        "invalidation_latency": None, "invalidation_reason": "REQUIRES_REGISTERED_ROLE_TRANSITIONS_AND_MATURE_TARGET_LABELS",
        "validation_folds": {track: validation_folds(p, track) for track in ("DIAGNOSTIC", "STRICT")},
        "strict_plan_path": p.get("strict_plan_path"), "strict_route": p.get("strict_route", "ORIGINAL_HISTORICAL_SOURCE_TIME_DIAGNOSTIC")}
    e.write_new(out/"full_candidate_comparison.json", all_predictions)
    result["predictions_sha256"] = e.sha(out/"full_candidate_comparison.json")
    logs = {(track, fold): value["C1_fit_logs"] for track, track_result in results.items()
            for fold, value in track_result["folds"].items()}
    e.write_new(out/"role_error_diagnostics.json", role_error_diagnosis(all_predictions, logs))
    result["role_error_diagnostics_sha256"] = e.sha(out/"role_error_diagnostics.json")
    e.write_new(out/"report.json", result)
    strict = results["STRICT"]["folds"]
    if strict_comparison_passed(p, result):
        last = strict[validation_folds(p, "STRICT")[-1]["id"]]
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
    if comparison["protocol_sha256"] != e.sha(protocol_path) or not strict_comparison_passed(p, comparison):
        raise ValueError("all frozen historical screens must pass before forward use")
    d = e.load_dataset(dataset_path); signals = {r["signal_date"] for r in d["rows"]}
    actual = datetime.now(ZoneInfo("Asia/Shanghai"))
    if len(signals) != 1 or next(iter(signals)) != actual.date().isoformat() or actual.time() <= time(15):
        raise ValueError("actual current official signal close required; no retro registration")
    signal = next(iter(signals)); days = e.calendar(d["calendar_path"], d["calendar_sha256"])
    if actual >= datetime.combine(datetime.fromisoformat(e.future_day(days, signal, 1)).date(), time(9, 15), ZoneInfo("Asia/Shanghai")):
        raise ValueError("forward role prediction must precede next target auction")
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
    if not d.get("baseline_binding"):
        raise ValueError("current source-bound BASE/feature/compatibility binding required")
    c2 = joblib.load(manifest["C2_model_path"])
    promotion = promotion_predictions(d, analysed["rows"], calls, c2)
    record = {"version": VERSION, "status": "INDEPENDENT_ROLE_FORWARD_NOT_PRODUCTION",
        "formed_at": e.now(), "protocol_path": str(Path(protocol_path).resolve()), "protocol_sha256": e.sha(protocol_path),
        "signal_date": signal, "target_dates": {str(h): e.future_day(days, signal, h) for h in (1, 3, 5)},
        "dataset_path": str(Path(dataset_path).resolve()), "source_bindings": d["source_bindings"] | e.bind(dataset_path) | e.bind(model_path) | e.bind(model_manifest_path)
            | e.bind(manifest["comparison_path"]) | e.bind(manifest["C2_model_path"]),
        "baseline_binding": d["baseline_binding"], "promotion_predictions": promotion,
        "rows": analysed["rows"], "predictions": calls, "model_path": str(Path(model_path).resolve()),
        "model_manifest_sha256": e.sha(model_manifest_path), "production_changed": False}
    record["role_observations"] = role_observations(analysed["rows"], calls)
    e.write_new(output, record)
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"path": str(Path(output).resolve()), "sha256": e.sha(output),
            "signal_date": signal, "protocol_sha256": e.sha(protocol_path)}, ensure_ascii=False)+"\n")
    return record


def role_observations(rows, predictions):
    result = []
    for row in rows:
        p = predictions[row["candidate_id"]]
        positive = [role for role in e.ROLES if p["calls"][role] is True]
        positive.sort(key=lambda role: (-p["scores"][role], e.ROLES.index(role)))
        result.append({"candidate_id": row["candidate_id"], "primary_role": positive[0] if positive else None,
            "auxiliary_roles": positive[1:], "state": "CANDIDATE" if positive else "COMPETING" if any(v is False for v in p["calls"].values()) else "UNKNOWN",
            "roles": {role: {**row["roles"][role], "call": p["calls"][role], "score": p["scores"][role],
                "state": "UNKNOWN" if p["calls"][role] is None else "CANDIDATE" if p["calls"][role] else "COMPETING"} for role in e.ROLES},
            "knowledge_cutoff": row["decision_at"], "available_at": row["available_at"], "scope_id": row["scope_id"],
            "identity_certified": False, "probability_calibrated": False, "entry_authorized": False})
    return result


def promotion_predictions(dataset, rows, role_predictions, c2):
    binding = dataset.get("baseline_binding")
    if not binding:
        raise ValueError("original BASE roster required")
    expected = set(binding["original_codes"])
    indexed = {r["code"]: r for r in rows if r["boards"] >= 2}
    if set(indexed) != expected:
        raise ValueError("promotion must retain complete original >=2-board cohort")
    result = {"BASE": [], "ROLE_C2": []}
    for code, row in sorted(indexed.items()):
        base = _base(dataset, row["candidate_id"], "STRICT")
        if base["call"] is None or base["raw_score"] is None:
            raise ValueError("source-bound original BASE unavailable before role decision")
        role = role_predictions[row["candidate_id"]]
        complete = all(role["scores"].get(k) is not None for k in e.ROLES)
        score, fallback = base["raw_score"], "ROLE_INPUT_OR_MODEL_UNAVAILABLE_USE_ORIGINAL_BASE"
        if c2 is not None and complete:
            expected_fields = len(e.RELAY_FIELDS)+len(e.ROLES)
            if getattr(c2, "n_features_in_", expected_fields) != expected_fields:
                raise ValueError("C2 feature order/width mismatch")
            vector = np.asarray([[row["features"].get(k) for k in e.RELAY_FIELDS]+[role["scores"][k] for k in e.ROLES]], dtype=float)
            score = float(c2.predict_proba(vector)[0, 1]); fallback = None
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("C2 invalid internal score")
        facts = {"code": code, "boards": row["boards"], "actual_one_word_daily": row["actual_one_word_daily"]}
        result["BASE"].append({**facts, "prediction": base["call"], "raw_score": base["raw_score"]})
        result["ROLE_C2"].append({**facts, "prediction": score >= .5 if fallback is None else base["call"],
            "raw_score": score, "fallback": fallback})
    for arm in result:
        result[arm].sort(key=lambda r: (-r["raw_score"], r["code"]))
    return result


def reconstruct_promotion_outcomes(record, archive):
    from model_upgrade_compat import runtime
    binding = record["baseline_binding"]
    e.verify(record["source_bindings"])
    days = e.calendar(e.load_dataset(record["dataset_path"])["calendar_path"])
    target = e.future_day(days, record["signal_date"], 1)
    if target != binding["target_date"] or target != record["target_dates"]["1"]:
        raise ValueError("promotion target is not exact official successor")
    inputs, _ = runtime(binding["compatibility_path"])
    outcome = inputs.build_outcomes({"feature_archive": binding["feature_archive"],
        "knowledge_cutoff": record["formed_at"], "signal_date": record["signal_date"], "target_date": target}, archive)
    if set(outcome["candidates"]) != set(binding["original_codes"]):
        raise ValueError("outcome adapter lost original promotion candidates")
    return outcome


def _registered_record(record_path, protocol_path):
    record = e.read(record_path)
    receipts = [r for r in _registry_receipts() if r["path"] == str(Path(record_path).resolve())]
    if len(receipts) != 1 or record["protocol_sha256"] != e.sha(protocol_path):
        raise ValueError("registered original role record required")
    e.verify(record["source_bindings"])
    return record


def settle_promotion(protocol_path, record_path, archive, output):
    load_protocol(protocol_path)
    rec = _registered_record(record_path, protocol_path)
    outcome = reconstruct_promotion_outcomes(rec, archive)
    result = {"version": VERSION, "status": "MATURE_SOURCE_BOUND_PROMOTION_SETTLEMENT", "formed_at": e.now(),
        "protocol_sha256": e.sha(protocol_path), "registration_sha256": e.sha(record_path), "record_path": str(Path(record_path).resolve()),
        "signal_date": rec["signal_date"], "target_date": rec["target_dates"]["1"], "horizon": 1,
        "archive_path": str(Path(archive).resolve()), "outcomes": outcome,
        "source_bindings": rec["source_bindings"] | outcome["sources"] | e.bind(record_path),
        "original_production_samples_added": 0}
    e.write_new(output, result)
    return result


def assess_promotion(records, outcomes, grid, days):
    """Reuse unchanged original and expanded acceptance in a separate namespace."""
    import relay_context_shadow as shadow
    import relay_context_research as context
    import model_upgrade_metrics as original_metrics
    from model_upgrade_compat import clone_module
    if not grid or not records:
        return {"passed": False, "status": "NO_FIXED_FORWARD_WINDOW", "original_gate": None, "expanded_gate": None}
    context_view = clone_module(context, {"ARMS": {"ROLE_C2": list(e.RELAY_FIELDS)+list(e.ROLES)}})
    shadow_view = clone_module(shadow, {"r": context_view})
    old_rows = [{"signal_date": r["signal_date"], "target_date": r["target_dates"]["1"],
        "original_codes": r["baseline_binding"]["original_codes"], "predictions": r["promotion_predictions"]} for r in records]
    arms, reasons = shadow_view.paired_day_rows(old_rows, outcomes, grid)
    expanded = shadow_view.assess(arms, reasons, grid, days)
    official = e.read(Path(__file__).parent/"research/model_upgrade_validation/fixed_20261007T112017Z/protocol_BEFORE_EFFECTS.json")
    official["forward_start_date"] = grid[0]
    # Only the actual clock may certify closure; calendar length alone is insufficient.
    actual = datetime.now(ZoneInfo("Asia/Shanghai"))
    closed = [d for d in days if d < actual.date().isoformat() or d == actual.date().isoformat() and actual.time() > time(15)]
    official["closed_through_date"] = closed[-1] if closed else None
    mapped = []
    for rec in records:
        b, c = rec["promotion_predictions"]["BASE"], rec["promotion_predictions"]["ROLE_C2"]
        one = lambda group: {"status": "KNOWN", "candidates": [{**v, "predicted_positive": v["prediction"]} for v in group]}
        nonword = [v for v in b if v["actual_one_word_daily"] == 0]
        highest = sorted(nonword, key=lambda v: (-v["boards"], v["code"]))[0]["code"] if nonword else None
        mapped.append({"module": "relay", "signal_date": rec["signal_date"], "target_date": rec["target_dates"]["1"],
            "forward": True, "original_codes": rec["baseline_binding"]["original_codes"],
            "arms": {"BASE": one(b), "C1": one(c), "C2": {"status": "UNKNOWN", "candidates": []}},
            "outcomes": {"candidates": outcomes.get(rec["signal_date"], {})}, "naive": {"highest_nonword": highest}})
    prior = original_metrics.acceptance("relay", mapped, official, days)
    # Old internal-score squared loss is kept as a diagnostic, not probability Brier.
    def rename(value):
        if isinstance(value, dict):
            return {k.replace("raw_score_Brier", "internal_squared_loss_diagnostic").replace("Brier_definition", "squared_loss_definition"): rename(v) for k, v in value.items()}
        if isinstance(value, str):
            return value.replace("Brier", "uncalibrated internal squared loss")
        return [rename(v) for v in value] if isinstance(value, list) else value
    mature = len(grid) == 120 and e.stamp(e.future_day(days, grid[-1], 1)+"T15:00:00+08:00") < e.stamp(e.now())
    passed = bool(mature and prior["gates"]["C1"]["passed"] and expanded["challenges"]["ROLE_C2"]["passed"])
    return {"passed": passed, "window_mature": mature, "original_gate": rename(prior), "expanded_gate": expanded,
        "arm_mapping": "Only this independent ROLE_C2 is mapped to C1 slot for pure original metric evaluation; original experiment is not modified",
        "probability_Brier": None, "internal_score_squared_loss_is_calibration": False}


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
    target_facts = e.analyse(d, "STRICT")
    target_model = joblib.load(rec["model_path"])
    target_calls = target_model.predict(target_facts["rows"], reject_training_members=True)
    output_rows = []
    for row in rec["rows"]:
        other = bycode.get(row["code"])
        matching = other is not None and other["scope_id"] == row["scope_id"]
        output_rows.append({"candidate_id": row["candidate_id"], "signal_date": rec["signal_date"],
            "code": row["code"], "scope_id": row["scope_id"],
            "baseline_roles": {role: row["roles"][role]["baseline_call"] for role in e.ROLES},
            "role_truth": {role: labels[(other["candidate_id"], role)]["value"] if matching else None for role in e.ROLES},
            "C1": rec["predictions"][row["candidate_id"]],
            "target_role_states": {role: "UNKNOWN" if not matching or target_calls[other["candidate_id"]]["calls"][role] is None else
                "INVALIDATED" if target_calls[other["candidate_id"]]["calls"][role] is False and rec["predictions"][row["candidate_id"]]["calls"][role] is True else
                "COMPETING" if target_calls[other["candidate_id"]]["calls"][role] is False else "CANDIDATE" for role in e.ROLES},
            "missing_reason": None if matching else "LEFT_SCOPE_OR_SCOPE_CHANGED_NOT_PROOF_OF_FAILURE"})
    result = {"version": VERSION, "formed_at": e.now(), "status": "MATURE_ROLE_HORIZON_SETTLEMENT",
        "protocol_sha256": e.sha(protocol_path), "registration_sha256": e.sha(record_path),
        "record_path": str(Path(record_path).resolve()), "target_dataset_path": str(Path(target_dataset_path).resolve()),
        "target_labels_path": str(Path(target_labels_path).resolve()),
        "signal_date": rec["signal_date"], "target_date": target, "horizon": horizon, "rows": output_rows,
        "source_bindings": rec["source_bindings"] | d["source_bindings"] | e.bind(record_path) | e.bind(target_dataset_path) | e.bind(target_labels_path),
        "new_mature_role_observation_rows": len(output_rows), "original_production_samples_added": 0}
    e.write_new(output, result)
    return result


def invalidation_latency(records, settlements, days):
    """First observed confirmed loss, with right censoring and unknowns retained.

    Observations at h3/h5 bound detection time; they do not invent the exact
    intraday or between-observation moment of identity loss.
    """
    bank = {(s["registration_sha256"], s["horizon"]): s for s in settlements if s["horizon"] in (0, 3, 5)}
    output = []
    for record_hash, rec in records.items():
        by_h = {h: {r["candidate_id"]: r for r in bank.get((record_hash, h), {}).get("rows", [])} for h in (0, 3, 5)}
        for original in rec["rows"]:
            key = original["candidate_id"]
            for role in e.ROLES:
                initial = by_h[0].get(key, {}).get("role_truth", {}).get(role)
                first_loss, observed_detection, previous_true = None, None, 0
                for h in (3, 5):
                    y = by_h[h].get(key, {}).get("role_truth", {}).get(role)
                    if y is True:
                        previous_true = h
                    if first_loss is None and y is False and initial is True:
                        first_loss = h
                if first_loss is not None:
                    for h in (3, 5):
                        state = by_h[h].get(key, {}).get("target_role_states", {}).get(role)
                        y = by_h[h].get(key, {}).get("role_truth", {}).get(role)
                        if h >= first_loss and y is False and state in ("INVALIDATED", "COMPETING"):
                            observed_detection = h
                            break
                status = "INITIAL_IDENTITY_UNKNOWN" if initial is None else "NOT_INITIAL_POSITIVE" if initial is False else "LOSS_OBSERVED" if first_loss else "RIGHT_CENSORED_OR_UNKNOWN"
                output.append({"candidate_id": key, "role": role, "status": status,
                    "first_observed_loss_horizon": first_loss, "last_observed_positive_horizon": previous_true,
                    "detected_loss_horizon": observed_detection,
                    "delay_after_observed_loss_sessions": observed_detection-first_loss if observed_detection is not None else None,
                    "exact_loss_time_verified": False})
    measured = [r["delay_after_observed_loss_sessions"] for r in output if r["delay_after_observed_loss_sessions"] is not None]
    return {"rows": output, "observed_losses": sum(r["first_observed_loss_horizon"] is not None for r in output),
        "measured_delays": len(measured), "mean_observed_delay_sessions": float(np.mean(measured)) if measured else None,
        "unknown_or_censored": sum(r["status"] in ("INITIAL_IDENTITY_UNKNOWN", "RIGHT_CENSORED_OR_UNKNOWN") for r in output),
        "exact_intraday_loss_time_or_latency": None}


def verify_settlement(s, protocol_path, byhash):
    key = s.get("registration_sha256")
    if key not in byhash or s.get("protocol_sha256") != e.sha(protocol_path):
        raise ValueError("settlement does not belong to actual registered original")
    rec = byhash[key]
    e.verify(s["source_bindings"])
    if s["signal_date"] != rec["signal_date"]:
        raise ValueError("settlement signal changed")
    h = s["horizon"]
    expected_target = rec["signal_date"] if h == 0 else rec["target_dates"].get(str(h))
    if s["target_date"] != expected_target or e.stamp(expected_target+"T15:00:00+08:00") >= e.stamp(e.now()):
        raise ValueError("unmatured or wrong exact settlement target")
    if h == 1:
        rebuilt = reconstruct_promotion_outcomes(rec, s["archive_path"])
        if rebuilt["candidates"] != s["outcomes"]["candidates"] or rebuilt["sources"] != s["outcomes"]["sources"]:
            raise ValueError("promotion truths do not reproduce original raw archive")
    elif h in (0, 3, 5):
        d = e.load_dataset(s["target_dataset_path"])
        if {r["signal_date"] for r in d["rows"]} != {expected_target}:
            raise ValueError("role target source changed date")
        labels = load_labels(s["target_labels_path"], d)
        indexed = {r["code"]: r for r in d["rows"]}
        original = {r["candidate_id"]: r for r in rec["rows"]}
        observed = {r["candidate_id"]: r for r in s["rows"]}
        if set(original) != set(observed) or len(observed) != len(s["rows"]):
            raise ValueError("role settlement drops original candidates")
        target_facts = e.analyse(d, "STRICT")
        target_calls = joblib.load(rec["model_path"]).predict(target_facts["rows"], reject_training_members=True)
        for ident, row in original.items():
            other = indexed.get(row["code"])
            match = other is not None and other["scope_id"] == row["scope_id"]
            truth = {role: labels[(other["candidate_id"], role)]["value"] if match else None for role in e.ROLES}
            if observed[ident]["role_truth"] != truth or observed[ident]["C1"] != rec["predictions"][ident]:
                raise ValueError("role truth or original call altered")
            states = {role: "UNKNOWN" if not match or target_calls[other["candidate_id"]]["calls"][role] is None else
                "INVALIDATED" if target_calls[other["candidate_id"]]["calls"][role] is False and rec["predictions"][ident]["calls"][role] is True else
                "COMPETING" if target_calls[other["candidate_id"]]["calls"][role] is False else "CANDIDATE" for role in e.ROLES}
            if observed[ident].get("target_role_states") != states:
                raise ValueError("target role transitions changed after original frozen model")
    else:
        raise ValueError("unsupported settlement horizon")


def review(protocol_path, settlements, calendar_path, output):
    p = load_protocol(protocol_path); days = e.calendar(calendar_path)
    regs = [r for r in registrations() if r["protocol_sha256"] == e.sha(protocol_path)]
    receipts = [r for r in _registry_receipts() if r["protocol_sha256"] == e.sha(protocol_path)]
    byhash = {r["sha256"]: e.read(r["path"]) for r in receipts}
    events, seen = [], set()
    for path in settlements:
        s = e.read(path); e.verify(s["source_bindings"])
        verify_settlement(s, protocol_path, byhash)
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
            "nondegradation": all(metric_not_lower(a, b, k) for k in ("Precision", "complete_positive_Recall", "F1")),
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
                and all(metric_not_lower(m["C1"], m["BASE"], k) for k in ("Precision", "complete_positive_Recall", "F1")) for m in byrole.values())
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
    promotion_regs = [reg for reg in regs if reg["signal_date"] in grid and reg.get("promotion_predictions")]
    outcomes = {s["signal_date"]: s["outcomes"]["candidates"] for s in events if s["horizon"] == 1}
    promotion_gate = assess_promotion(promotion_regs, outcomes, grid, days)
    both = bool(passed and promotion_gate["passed"])
    result = {"version": VERSION, "formed_at": e.now(), "status": "ALL_ROLE_AND_PROMOTION_GATES_PASSED" if both else "OVERALL_NOT_PASSED",
        "protocol_sha256": e.sha(protocol_path), "grid": grid, "window_mature": mature, "metrics": metrics,
        "checks": checks, "macro_F1": macro, "delta": delta, "interval": interval,
        "role_gate_passed": passed, "C2_original_gate_passed": promotion_gate["passed"], "deployment_allowed": False,
        "eligible_for_new_date_application": both, "C2_acceptance": promotion_gate,
        "invalidation_latency": invalidation_latency(byhash, events, days),
        "future_role_metrics": future_metrics, "primary_future_horizon": 5,
        "three40_macro_F1_deltas": stability,
        "precision_guard_reference": "Reported undefined BASE precision remains null; baseline with known truth and zero positive calls has gate reference zero only",
        "C2_gate_reason": "Both original non-oneword rank-first and expanded complete-cohort gates evaluated from original registered predictions and rebuilt truths",
        "missing_types": [k for k, c in checks.items() if not all(v for name, v in c.items() if name != "complete_days")],
        "source_bindings": e.bind(protocol_path) | e.bind(calendar_path) | {str(Path(f).resolve()): e.sha(f) for f in settlements}}
    e.write_new(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit"); a.add_argument("--dataset", required=True); a.add_argument("--labels"); a.add_argument("--output", required=True)
    a = sub.add_parser("diagnose")
    for k in ("report", "protocol", "output"):
        a.add_argument("--"+k, required=True)
    a = sub.add_parser("import-reviews")
    for k in ("dataset", "packet", "output"):
        a.add_argument("--"+k, required=True)
    a.add_argument("--reviews", nargs=2, required=True)
    a = sub.add_parser("freeze")
    for k in ("dataset", "labels", "output"):
        a.add_argument("--"+k, required=True)
    a.add_argument("--strict-plan")
    a = sub.add_parser("plan-strict")
    for k in ("dataset", "labels", "fold-spec", "output"):
        a.add_argument("--"+k, required=True)
    a = sub.add_parser("backtest"); a.add_argument("--protocol", required=True); a.add_argument("--output-dir", required=True)
    a = sub.add_parser("predict")
    for k in ("protocol", "dataset", "model", "model-manifest", "output"):
        a.add_argument("--"+k, required=True)
    a = sub.add_parser("settle")
    for k in ("protocol", "record", "output"):
        a.add_argument("--"+k, required=True)
    a.add_argument("--target-dataset"); a.add_argument("--target-labels"); a.add_argument("--outcome-archive")
    a.add_argument("--horizon", type=int, choices=(0, 1, 3, 5), required=True)
    a = sub.add_parser("review"); a.add_argument("--protocol", required=True); a.add_argument("--settlements", nargs="*", default=[])
    a.add_argument("--calendar", required=True); a.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "audit": result = audit(args.dataset, args.labels, args.output)
    elif args.command == "diagnose": result = diagnose_saved_comparison(args.report, args.protocol, args.output)
    elif args.command == "import-reviews": result = merge_identity_reviews(args.dataset, args.packet, args.reviews, args.output)
    elif args.command == "freeze": result = freeze(args.dataset, args.labels, args.output, strict_plan_path=args.strict_plan)
    elif args.command == "plan-strict": result = plan_strict(args.dataset, args.labels, args.fold_spec, args.output)
    elif args.command == "backtest": result = backtest(args.protocol, args.output_dir)
    elif args.command == "predict": result = predict(args.protocol, args.dataset, args.model, args.model_manifest, args.output)
    elif args.command == "settle":
        if args.horizon == 1:
            if not args.outcome_archive: p.error("horizon1 needs --outcome-archive")
            result = settle_promotion(args.protocol, args.record, args.outcome_archive, args.output)
        else:
            if not args.target_dataset or not args.target_labels: p.error("role settlement needs --target-dataset and --target-labels")
            result = settle(args.protocol, args.record, args.target_dataset, args.target_labels, args.horizon, args.output)
    else: result = review(args.protocol, args.settlements, args.calendar, args.output)
    print(json.dumps({k: result.get(k) for k in ("status", "fits", "overall_acceptance", "deployment_allowed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
