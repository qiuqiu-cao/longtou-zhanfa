"""Assemble, hash, and audit the frozen two-stage leader model stack."""
import argparse
import hashlib
import json
from pathlib import Path

from full_pool_rank_dataset import TARGET_DEFINITION as RANK_TARGET, SELECTION_SCOPE
from complete_trade_dataset import TARGET_DEFINITION as TRADE_TARGET
from full_pool_rank_blind_vault import _canonical_hash
from leader_model_stack_audit import audit
from leader_target_screen_v17 import timestamp


def _calibrated(block):
    return (isinstance(block, dict) and block.get("chosen") in (block.get("candidates") or {})
            and (block["candidates"][block["chosen"]].get("n") or 0) > 0)


def assemble(rank_dataset, rank_split, rank_model, rank_blind,
             trade_dataset, trade_split, trade_model, threshold, trade_blind,
             component_hashes, frozen_at):
    lock_time = timestamp(frozen_at)
    required_hashes = {"rank_model_artifact", "trade_model_artifact", "threshold_report", "feature_schema"}
    if set(component_hashes) != required_hashes or any(not isinstance(value, str) or len(value) != 64
                                                        for value in component_hashes.values()):
        raise ValueError("exact SHA-256 component hash set required")
    observed_times = []
    for block, key in ((rank_model, "generated_at"), (rank_blind, "evaluated_at"),
                       (trade_model, "generated_at"), (threshold, "validated_at"),
                       (trade_blind, "evaluated_at")):
        try:
            observed_times.append(timestamp(block[key]))
        except (KeyError, TypeError, ValueError):
            raise ValueError("missing immutable component time: " + key)
    if max(observed_times) > lock_time:
        raise ValueError("stack cannot be frozen before its components")
    policy_ids = {trade_dataset.get("policy_id"), trade_model.get("policy_id"),
                  threshold.get("selection_policy_id"), trade_blind.get("policy_id")}
    if len(policy_ids) != 1 or not next(iter(policy_ids)):
        raise ValueError("trade components use different or missing policy ids")
    policy_id = next(iter(policy_ids))
    schema = {"rank_features": rank_dataset.get("feature_allowlist"),
              "trade_features": trade_dataset.get("feature_allowlist"),
              "trade_support": trade_model.get("feature_support_envelope")}
    if _canonical_hash(schema) != component_hashes["feature_schema"]:
        raise ValueError("feature schema hash mismatch")
    if (rank_model.get("model_artifact_sha256") != component_hashes["rank_model_artifact"]
            or trade_model.get("model_artifact_sha256") != component_hashes["trade_model_artifact"]):
        raise ValueError("model artifact hash mismatch")
    trade_metrics = trade_blind.get("metrics", {})
    evidence = {
        "full_pool_ranker": {
            "dataset_ready": rank_dataset.get("status") == "FULL_POOL_RANK_DATASET_READY",
            "selection_scope": rank_dataset.get("selection_scope"),
            "target_definition": rank_dataset.get("target_definition"),
            "actual_trade_target": rank_dataset.get("actual_trade_target"),
            "purged_split_ready": rank_split.get("status") == "READY_FOR_FULL_POOL_RANK_DEVELOPMENT",
            "blind_labels_untouched_until_evaluation": rank_blind.get("blind_labels_untouched_until_evaluation"),
            "blind_label_access_control": rank_blind.get("blind_label_access_control"),
            "blind_evaluation_status": rank_blind.get("status"),
            "blind_session_count": rank_blind.get("blind_session_count"),
            "blind_top1_proxy_rate": rank_blind.get("blind_top1_proxy_rate"),
            "blind_baseline_proxy_rate": rank_blind.get("blind_baseline_proxy_rate"),
            "baseline_frozen_before_blind": rank_blind.get("baseline_frozen_before_blind"),
            "first_and_relay_reported": rank_blind.get("first_and_relay_reported"),
            "frozen_at": rank_model.get("generated_at"),
        },
        "policy_conditional_trade_models": {
            "complete_trade_dataset_ready": trade_dataset.get("status") == "POLICY_CONDITIONAL_DATASET_READY",
            "purged_split_ready": trade_split.get("status") == "READY_FOR_MODEL_DEVELOPMENT",
            "selection_scope": trade_dataset.get("selection_scope"),
            "target_definition": trade_dataset.get("target_definition"),
            "execution_basis_verified": trade_blind.get("execution_basis_verified"),
            "generalization_scope": trade_dataset.get("generalization_scope"),
            "profit_model_calibrated": _calibrated(trade_model.get("profit_model")),
            "large_loss_model_calibrated": _calibrated(trade_model.get("risk_model")),
            "validation_complete_trades": threshold.get("validation_observations"),
            "out_of_distribution_gate_frozen": trade_model.get("out_of_distribution_gate_frozen"),
            "blind_label_access_control": trade_blind.get("blind_label_access_control"),
            "blind_labels_untouched_until_evaluation": trade_blind.get("blind_labels_untouched_until_evaluation"),
            "blind_evaluation_status": trade_blind.get("status"),
            "blind_completed_trades": trade_metrics.get("completed_trades"),
            "blind_win_rate": trade_metrics.get("win_rate"),
            "blind_large_loss_rate": trade_metrics.get("big_loss_rate"),
            "blind_mean_net_return": trade_metrics.get("mean_net_return"),
            "blind_profit_factor": trade_metrics.get("profit_factor"),
            "blind_outperforms_baseline": trade_blind.get("outperforms_baseline"),
            "selection_policy_id": policy_id,
            "frozen_at": trade_model.get("generated_at"),
        },
        "frozen_threshold": dict(threshold),
        "stack_lock": {
            "selection_policy_id": policy_id, "frozen_at": lock_time.isoformat(),
            "rank_model_sha256": component_hashes["rank_model_artifact"],
            "profit_model_sha256": component_hashes["trade_model_artifact"],
            "large_loss_model_sha256": component_hashes["trade_model_artifact"],
            "threshold_sha256": component_hashes["threshold_report"],
            "feature_schema_sha256": component_hashes["feature_schema"],
            "rank_score_named_probability": rank_model.get("rank_score_named_probability"),
            "complete_trade_probability_from_rank_proxy": False,
        },
    }
    result = audit(evidence, lock_time.isoformat())
    return {"status": result["status"], "frozen_at": lock_time.isoformat(),
            "selection_scope": SELECTION_SCOPE, "rank_target": RANK_TARGET, "trade_target": TRADE_TARGET,
            "evidence": evidence, "stack_audit": result, "component_hashes": component_hashes,
            "deployment_allowed": False,
            "limitations": ["a ready stack only permits locked blind paper use; system acceptance and forward evidence remain separate"]}


def run(paths, output_path, frozen_at):
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("choose a new immutable model stack lock")
    loaded, source_hashes = {}, {}
    for name, raw in paths.items():
        path = Path(raw)
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
        source_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    rank_model_hash = loaded["rank_model"].get("model_artifact_sha256")
    trade_model_hash = loaded["trade_model"].get("model_artifact_sha256")
    threshold_hash = source_hashes[str(Path(paths["threshold"]).resolve())]
    schema = {"rank_features": loaded["rank_dataset"].get("feature_allowlist"),
              "trade_features": loaded["trade_dataset"].get("feature_allowlist"),
              "trade_support": loaded["trade_model"].get("feature_support_envelope")}
    components = {"rank_model_artifact": rank_model_hash, "trade_model_artifact": trade_model_hash,
                  "threshold_report": threshold_hash, "feature_schema": _canonical_hash(schema)}
    result = assemble(loaded["rank_dataset"], loaded["rank_split"], loaded["rank_model"], loaded["rank_blind"],
                      loaded["trade_dataset"], loaded["trade_split"], loaded["trade_model"], loaded["threshold"],
                      loaded["trade_blind"], components, frozen_at)
    result["source_hashes"] = source_hashes
    result["source_hashes"][str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rank_dataset", "rank_split", "rank_model", "rank_blind", "trade_dataset",
                 "trade_split", "trade_model", "threshold", "trade_blind"):
        parser.add_argument("--" + name.replace("_", "-"), dest=name, required=True)
    parser.add_argument("--frozen-at", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in ("rank_dataset", "rank_split", "rank_model", "rank_blind",
             "trade_dataset", "trade_split", "trade_model", "threshold", "trade_blind")}
    print(json.dumps(run(paths, args.output, args.frozen_at), ensure_ascii=False, indent=2))
