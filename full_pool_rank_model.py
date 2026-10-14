"""Train a development-only full-pool opportunity ranking model.

All candidates from a session receive equal total weight, preventing a large
limit-up day from dominating training. Model selection is based on validation
daily-top-one hit rate, reciprocal rank, and Brier score. Blind rows are neither
materialized into a feature matrix nor evaluated here.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from full_pool_rank_dataset import TARGET_DEFINITION, SELECTION_SCOPE


def _matrix(rows, candidate_ids, features):
    matrix, labels, dates = [], [], []
    for candidate_id in candidate_ids:
        row = rows[candidate_id]
        if (row.get("label_available") is not True
                or not isinstance(row.get("opportunity_proxy_label"), bool)
                or row.get("issues")):
            raise ValueError("invalid development rank row: " + candidate_id)
        values = row.get("selected_features", {})
        matrix.append([float(values.get(feature, np.nan)) for feature in features])
        labels.append(int(row["opportunity_proxy_label"]))
        dates.append(row["signal_date"])
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("development feature matrix contains missing or nonfinite values")
    labels = np.asarray(labels, dtype=int)
    if np.unique(labels).size != 2:
        raise ValueError("development split requires both proxy classes")
    counts = defaultdict(int)
    for day in dates:
        counts[day] += 1
    weights = np.asarray([1.0 / counts[day] for day in dates], dtype=float)
    return matrix, labels, dates, weights


def _models():
    return {
        "regularized_logistic": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=2000, random_state=31)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=120,
                                                      l2_regularization=.1, random_state=31)),
        ]),
    }


def _logit(probability):
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def _daily_metrics(candidate_ids, rows, probability):
    groups = defaultdict(list)
    for candidate_id, score in zip(candidate_ids, probability):
        groups[rows[candidate_id]["signal_date"]].append((candidate_id, float(score)))
    top_hits, reciprocal, top_rows = [], [], []
    for day, items in sorted(groups.items()):
        ordered = sorted(items, key=lambda item: (-item[1], item[0]))
        winner = rows[ordered[0][0]]
        top_hits.append(int(winner["opportunity_proxy_label"]))
        positive_ranks = [rank for rank, (candidate_id, _) in enumerate(ordered, start=1)
                          if rows[candidate_id]["opportunity_proxy_label"]]
        reciprocal.append(1 / min(positive_ranks) if positive_ranks else 0.0)
        top_rows.append({"signal_date": day, "candidate_id": ordered[0][0],
                         "score": ordered[0][1], "hit": bool(top_hits[-1]),
                         "boards": winner["boards"], "board_group": winner["board_group"]})
    return {"session_count": len(groups),
            "daily_top1_proxy_hit_rate": float(np.mean(top_hits)) if top_hits else None,
            "mean_reciprocal_rank": float(np.mean(reciprocal)) if reciprocal else None,
            "top1_first_board_count": sum(row["board_group"] == "FIRST_BOARD" for row in top_rows),
            "top1_consecutive_board_count": sum(row["board_group"] == "CONSECUTIVE_BOARD" for row in top_rows)}, top_rows


def _candidate_metrics(labels, probability, weights):
    prevalence = float(np.average(labels, weights=weights))
    brier = float(np.average((labels - probability) ** 2, weights=weights))
    baseline = float(np.average((labels - prevalence) ** 2, weights=weights))
    return {"candidate_count": int(len(labels)), "day_weighted_prevalence": prevalence,
            "candidate_auc": float(roc_auc_score(labels, probability, sample_weight=weights)),
            "day_weighted_brier": brier,
            "brier_skill": 1 - brier / baseline if baseline > 0 else None,
            "day_weighted_log_loss": float(log_loss(labels, probability, labels=[0, 1], sample_weight=weights))}


def _height_baseline(candidate_ids, rows):
    groups = defaultdict(list)
    for candidate_id in candidate_ids:
        groups[rows[candidate_id]["signal_date"]].append(candidate_id)
    winners = []
    for day, ids in sorted(groups.items()):
        chosen = sorted(ids, key=lambda candidate_id: (-rows[candidate_id]["boards"], candidate_id))[0]
        winners.append({"signal_date": day, "candidate_id": chosen,
                        "hit": bool(rows[chosen]["opportunity_proxy_label"]),
                        "boards": rows[chosen]["boards"], "board_group": rows[chosen]["board_group"]})
    return {"name": "highest_board_then_candidate_id", "frozen_rule": True,
            "session_count": len(winners),
            "daily_top1_proxy_hit_rate": float(np.mean([int(row["hit"]) for row in winners])) if winners else None,
            "top_rows": winners}


def train(dataset, split):
    if (dataset.get("training_ready") is not True or dataset.get("full_pool_scope_established") is not True
            or dataset.get("target_definition") != TARGET_DEFINITION
            or dataset.get("selection_scope") != SELECTION_SCOPE
            or dataset.get("blind_rows_removed") is not True
            or dataset.get("blind_label_fields_present") is not False):
        raise ValueError("upstream full-pool rank dataset is not ready")
    if (split.get("status") != "READY_FOR_FULL_POOL_RANK_DEVELOPMENT"
            or split.get("blind_labels_not_summarized") is not True
            or split.get("same_signal_date_kept_together") is not True
            or split.get("blind_access_control") != "SEPARATE_FEATURE_AND_LABEL_VAULT"
            or split.get("source_dataset_with_blind_labels_not_required_for_training") is not True):
        raise ValueError("purged full-pool development split is not ready")
    partitions = split["partitions"]
    development = set(partitions["train"]) | set(partitions["calibration"]) | set(partitions["validation"])
    if "blind" in partitions or sum(len(partitions[name]) for name in ("train", "calibration", "validation")) != len(development):
        raise ValueError("sealed development split must not expose blind identities")
    rows = {row["candidate_id"]: row for row in dataset["rows"]}
    if development != set(rows):
        raise ValueError("development view must contain exactly the development candidates")
    features = dataset.get("feature_allowlist")
    if not isinstance(features, list) or not features or len(features) != len(set(features)):
        raise ValueError("frozen feature allowlist required")
    x_train, y_train, _, w_train = _matrix(rows, partitions["train"], features)
    x_cal, y_cal, _, w_cal = _matrix(rows, partitions["calibration"], features)
    x_val, y_val, _, w_val = _matrix(rows, partitions["validation"], features)
    fitted, evaluations, val_probabilities = {}, {}, {}
    for name, model in _models().items():
        fit_params = {"model__sample_weight": w_train}
        model.fit(x_train, y_train, **fit_params)
        raw_cal = model.predict_proba(x_cal)[:, 1]
        calibrator = LogisticRegression(C=1000, max_iter=2000, random_state=37)
        calibrator.fit(_logit(raw_cal), y_cal, sample_weight=w_cal)
        raw_val = model.predict_proba(x_val)[:, 1]
        probability = calibrator.predict_proba(_logit(raw_val))[:, 1]
        candidate_metrics = _candidate_metrics(y_val, probability, w_val)
        daily_metrics, top_rows = _daily_metrics(partitions["validation"], rows, probability)
        evaluations[name] = {**candidate_metrics, **daily_metrics, "top_rows": top_rows}
        val_probabilities[name] = probability
        fitted[name] = {"base": model, "calibrator": calibrator}
    chosen = min(evaluations, key=lambda name: (-evaluations[name]["daily_top1_proxy_hit_rate"],
                                                 -evaluations[name]["mean_reciprocal_rank"],
                                                 evaluations[name]["day_weighted_brier"], name))
    probability = val_probabilities[chosen]
    validation_predictions = [{"candidate_id": candidate_id,
                               "signal_date": rows[candidate_id]["signal_date"],
                               "boards": rows[candidate_id]["boards"],
                               "board_group": rows[candidate_id]["board_group"],
                               "rank_score": float(score),
                               "opportunity_proxy_label": int(rows[candidate_id]["opportunity_proxy_label"])}
                              for candidate_id, score in zip(partitions["validation"], probability)]
    baseline = _height_baseline(partitions["validation"], rows)
    bundle = {"feature_allowlist": features, "selection_scope": SELECTION_SCOPE,
              "target_definition": TARGET_DEFINITION, "chosen_model": chosen,
              "model": fitted[chosen], "trained_candidate_ids": sorted(development),
              "blind_feature_matrix_built": False, "source_payload_including_blind_rows_parsed": False,
              "blind_candidate_count": split.get("blind_count"),
              "blind_candidate_ids_sha256": split.get("blind_candidate_ids_sha256"),
              "blind_labels_sha256_commitment": split.get("blind_labels_sha256_commitment")}
    card = {"status": "FULL_POOL_RANK_DEVELOPMENT_MODEL_ONLY", "deployment_allowed": False,
            "selection_scope": SELECTION_SCOPE, "target_definition": TARGET_DEFINITION,
            "actual_trade_target": False, "full_pool_training_scope": True,
            "rank_score_named_probability": False, "feature_allowlist": features,
            "partition_counts": {**{name: len(partitions[name]) for name in partitions},
                                 "blind": split.get("blind_count")},
            "chosen_model": chosen, "candidate_models": evaluations,
            "validation_predictions": validation_predictions, "validation_height_baseline": baseline,
            "blind_metrics": None, "blind_feature_matrix_built": False,
            "blind_label_fields_accessed": False, "source_payload_including_blind_rows_parsed": False,
            "blind_label_access_control": "SEPARATE_FEATURE_AND_LABEL_VAULT",
            "blind_candidate_ids_sha256": split.get("blind_candidate_ids_sha256"),
            "blind_labels_sha256_commitment": split.get("blind_labels_sha256_commitment"),
            "limitations": ["validation proxy performance is not complete-trade profitability",
                            "model choice uses validation daily-top-one hit rate and therefore still requires untouched blind evaluation",
                            "training inputs exclude blind rows; organizationally independent custody is still required",
                            "daily OHLC does not establish actual fillability"]}
    return card, bundle


def run(dataset_path, split_path, output_dir):
    dataset_path, split_path, output_dir = map(Path, (dataset_path, split_path, output_dir))
    if output_dir.exists():
        raise FileExistsError("choose a new immutable rank model output directory")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    card, bundle = train(dataset, split)
    card["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact = output_dir / "full_pool_rank_model.joblib"
    joblib.dump(bundle, artifact, compress=3)
    card["source_hashes"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (dataset_path, split_path, Path(__file__))}
    card["model_artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (output_dir / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2,
                                                           allow_nan=False), encoding="utf-8")
    return card


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.split, args.output), ensure_ascii=False, indent=2))
