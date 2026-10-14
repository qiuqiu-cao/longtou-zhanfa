"""Audit how much legacy reconstructed data can support the new rank contract."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from leader_target_screen_v17 import timestamp


FEATURES = [
    "boards_for_comparison", "height_gap_to_known_top", "same_height_known_count",
    "sole_known_top_proxy", "open_gap_pct", "amplitude_pct", "amount_to_float_cap",
    "one_price_day_proxy", "market_limit_count", "market_consecutive_count",
    "market_failed_rate", "market_seal_rate", "market_yesterday_premium",
    "market_previous_limit_tail_loss", "market_promotion_overall", "market_promotion_mid",
    "market_space_feedback", "market_middle_density", "market_ladder_gap_count",
]


def _bool_series(series):
    return series.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})


def audit(features, outcomes, inventory_manifest, inventory=None):
    required_features = {"code", "signal_date", "boards", "feature_available_not_before",
                         "feature_status", "pool_scope", *FEATURES}
    required_outcomes = {"code", "signal_date", "exit_date_LABEL_ONLY", "price_path_state",
                         "net_close_mark_LABEL_ONLY", "entry_one_price_LABEL_ONLY"}
    missing = {"features": sorted(required_features - set(features)),
               "outcomes": sorted(required_outcomes - set(outcomes))}
    if any(missing.values()):
        raise ValueError("missing legacy fields: " + json.dumps(missing, ensure_ascii=False))
    features = features.copy()
    outcomes = outcomes.copy()
    board_group_source = "supplied"
    if "board_group" not in features:
        numeric_boards = pd.to_numeric(features["boards"], errors="coerce")
        features["board_group"] = numeric_boards.map(
            lambda value: "FIRST_BOARD" if value == 1 else "CONSECUTIVE_BOARD"
            if pd.notna(value) and value >= 2 and value % 1 == 0 else None)
        board_group_source = "derived_from_integer_boards"
    for frame in (features, outcomes):
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        if frame.duplicated(["code", "signal_date"]).any():
            raise ValueError("duplicate legacy candidate/date key")
    merged = features.merge(outcomes, on=["code", "signal_date"], how="outer", indicator=True,
                            validate="one_to_one")
    matched = merged[merged["_merge"] == "both"].copy()
    boards = pd.to_numeric(matched["boards"], errors="coerce")
    valid_board = boards.notna() & boards.ge(1) & boards.mod(1).eq(0)
    expected_group = boards.map(lambda value: "FIRST_BOARD" if value == 1 else "CONSECUTIVE_BOARD")
    group_valid = valid_board & matched["board_group"].eq(expected_group)
    feature_times_valid = []
    for _, row in matched.iterrows():
        try:
            available = timestamp(row["feature_available_not_before"])
            feature_times_valid.append(available <= timestamp(row["signal_date"] + "T15:05:00+08:00")
                                       and available.strftime("%Y-%m-%d") == row["signal_date"])
        except (TypeError, ValueError):
            feature_times_valid.append(False)
    feature_times_valid = pd.Series(feature_times_valid, index=matched.index)
    feature_coverage = {}
    for feature in FEATURES:
        numeric = pd.to_numeric(matched[feature], errors="coerce")
        feature_coverage[feature] = {"observed": int(numeric.notna().sum()), "missing": int(numeric.isna().sum()),
                                     "coverage": float(numeric.notna().mean()) if len(numeric) else None}
    state_valid = matched["price_path_state"].eq("PRICE_OPPORTUNITY_PROXY_ONLY")
    one_price = _bool_series(matched["entry_one_price_LABEL_ONLY"])
    mark = pd.to_numeric(matched["net_close_mark_LABEL_ONLY"], errors="coerce")
    # In the legacy generator mark is present only when next-session limit was touched.
    touched = mark.notna()
    resolved = state_valid & one_price.notna()
    target = touched & ~one_price.fillna(True).astype(bool) & mark.gt(0)
    target_counts = {"resolved": int(resolved.sum()), "unresolved": int((~resolved).sum()),
                     "positive": int((resolved & target).sum()), "negative": int((resolved & ~target).sum())}
    reconstruction_status = sorted(matched["feature_status"].dropna().astype(str).unique())
    scopes = sorted(matched["pool_scope"].dropna().astype(str).unique())
    board_counts = Counter(str(int(value)) for value in boards[valid_board])
    inventory_reconciliation = None
    if inventory is not None:
        inventory = inventory.copy()
        inventory["code"] = inventory["code"].astype(str).str.zfill(6)
        if inventory.duplicated(["code", "signal_date"]).any():
            raise ValueError("duplicate inventory candidate/date key")
        reconciled = features[["code", "signal_date", "boards"]].merge(
            inventory[["code", "signal_date", "boards", "board_group"]],
            on=["code", "signal_date"], how="left", suffixes=("_feature", "_inventory"), indicator=True,
            validate="one_to_one")
        feature_board = pd.to_numeric(reconciled["boards_feature"], errors="coerce")
        inventory_board = pd.to_numeric(reconciled["boards_inventory"], errors="coerce")
        inventory_reconciliation = {
            "inventory_candidate_rows": len(inventory),
            "feature_rows_found_in_inventory": int(reconciled["_merge"].eq("both").sum()),
            "feature_rows_missing_from_inventory": int(reconciled["_merge"].ne("both").sum()),
            "board_height_exact_matches": int(feature_board.eq(inventory_board).sum()),
            "inventory_left_censored_but_feature_height_assigned": int((inventory_board.isna() & feature_board.notna()).sum()),
            "feature_coverage_of_inventory": len(features) / len(inventory) if len(inventory) else None,
            "outcome_matched_coverage_of_inventory": len(matched) / len(inventory) if len(inventory) else None,
        }
    blockers = []
    if inventory_manifest.get("historical_reconstruction_only") is True:
        blockers.append("HISTORICAL_RECONSTRUCTION_ONLY")
    if inventory_manifest.get("original_point_in_time_capture_available") is not True:
        blockers.append("ORIGINAL_POINT_IN_TIME_CAPTURE_UNAVAILABLE")
    if inventory_manifest.get("status") != "FULL_POINT_IN_TIME_UNIVERSE_VERIFIED":
        blockers.append("FULL_HISTORICAL_UNIVERSE_NOT_VERIFIED")
    if any(value != "verified_forward_archive" for value in reconstruction_status):
        blockers.append("FEATURE_AVAILABILITY_IS_RETROSPECTIVE")
    if any("not_complete_market" in value for value in scopes):
        blockers.append("POOL_SCOPE_NOT_COMPLETE_MARKET")
    if board_counts.get("1", 0) == 0:
        blockers.append("FIRST_BOARD_RANK_FEATURES_ABSENT")
    if inventory_reconciliation and inventory_reconciliation["inventory_left_censored_but_feature_height_assigned"]:
        blockers.append("LEFT_CENSORED_IDENTITY_ASSIGNED_IN_LEGACY_FEATURES")
    blockers.extend(["POINT_IN_TIME_SECURITY_STATUS_NOT_REAUDITED",
                     "ROUNDED_LIMIT_AND_BOARD_IDENTITY_ONLY",
                     "FAILED_BOARD_DAILY_MANIFESTS_UNAVAILABLE",
                     "ACTUAL_QUEUE_AND_FILL_EVIDENCE_UNAVAILABLE"])
    return {"status": "DIAGNOSTIC_ONLY", "legacy_feature_rows": len(features),
            "legacy_outcome_rows": len(outcomes), "matched_rows": len(matched),
            "feature_only_rows": int((merged["_merge"] == "left_only").sum()),
            "outcome_only_rows": int((merged["_merge"] == "right_only").sum()),
            "valid_board_and_group_rows": int(group_valid.sum()),
            "board_group_source": board_group_source,
            "feature_time_at_or_before_close_rows": int(feature_times_valid.sum()),
            "board_counts": dict(sorted(board_counts.items(), key=lambda item: int(item[0]))),
            "first_and_relay_feature_coverage": board_counts.get("1", 0) > 0 and sum(
                count for board, count in board_counts.items() if int(board) >= 2) > 0,
            "inventory_reconciliation": inventory_reconciliation,
            "target_proxy_counts": target_counts, "feature_coverage": feature_coverage,
            "feature_status_values": reconstruction_status, "pool_scope_values": scopes,
            "usable_for_exploratory_proxy_research": bool(len(matched) and target_counts["resolved"] > 0),
            "usable_for_locked_model_stack": False, "complete_trade_evidence": False,
            "blocking_gaps": sorted(set(blockers)),
            "next_data_actions": [
                "archive each new closing limit pool and failed-board pool with candidate IDs and board counts",
                "attach dated exchange/security-master status to every candidate",
                "store features with observed_at and available_at at collection time",
                "continue separate auction/minute/order/fill and complete-trade ledgers",
            ],
            "note": "legacy opportunity labels are not actual trades and cannot satisfy the 70% complete-trade gate"}


def run(features_path, outcomes_path, inventory_manifest_path, inventory_observations_path, output_path):
    paths = list(map(Path, (features_path, outcomes_path, inventory_manifest_path, inventory_observations_path)))
    target = Path(output_path)
    if target.exists():
        raise FileExistsError("choose a new immutable legacy-gap audit path")
    result = audit(pd.read_csv(paths[0], dtype={"code": str}), pd.read_csv(paths[1], dtype={"code": str}),
                   json.loads(paths[2].read_text(encoding="utf-8")), pd.read_csv(paths[3], dtype={"code": str}))
    result["source_hashes"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in paths + [Path(__file__)]}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--inventory-manifest", required=True)
    parser.add_argument("--inventory-observations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.outcomes, args.inventory_manifest, args.inventory_observations, args.output),
                     ensure_ascii=False, indent=2))
