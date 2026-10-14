"""Build a leakage-audited full closing-limit pool ranking dataset.

The target is deliberately an opportunity proxy.  Daily OHLC can describe a
next-session non-one-price limit touch and a later close mark, but cannot prove
queue position, order acceptance, fill price, or executable profit.  Actual
complete-trade calibration belongs to ``complete_trade_dataset.py``.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from leader_target_screen_v17 import finite, timestamp


TARGET_DEFINITION = "next_session_non_one_price_limit_touch_and_b2_positive_close_mark_proxy"
SELECTION_SCOPE = "all_mainboard_nonst_closing_limitups_first_and_consecutive"
GROUPS = {"FIRST_BOARD", "CONSECUTIVE_BOARD"}


def _valid_feature_bundle(row, allowlist, decision_at):
    values = row.get("features", {})
    evidence = row.get("feature_evidence", {})
    selected, issues = {}, []
    for name in allowlist:
        value = finite(values.get(name))
        meta = evidence.get(name, {})
        try:
            observed = timestamp(meta["observed_at"])
            available = timestamp(meta["available_at"])
            valid = (value is not None and observed <= available <= decision_at
                     and bool(meta.get("source")) and bool(meta.get("definition_id"))
                     and meta.get("data_status") in {"verified", "inferred"})
        except (KeyError, TypeError, ValueError):
            valid = False
        if valid:
            selected[name] = value
        else:
            issues.append("FEATURE_UNAVAILABLE_OR_FUTURE_" + name)
    return selected, sorted(set(values) - set(allowlist)), issues


def _target(row, decision_at, cutoff):
    outcome = row.get("opportunity_outcome", {})
    if outcome.get("status") == "UNAVAILABLE":
        return None, None, None, ["OPPORTUNITY_OUTCOME_UNAVAILABLE"]
    issues = []
    try:
        available = timestamp(outcome["available_at"])
        touched = outcome["next_session_touched_limit"]
        one_price = outcome["next_session_one_price_limit"]
        mark = finite(outcome["b2_close_mark_net_return_proxy"])
        label = outcome["target_label"]
        valid_types = (isinstance(touched, bool) and isinstance(one_price, bool)
                       and isinstance(label, bool) and (mark is not None or not touched))
        expected = bool(touched and not one_price and mark > 0)
        if (not valid_types or available <= decision_at or available > cutoff
                or not outcome.get("source") or outcome.get("execution_verified") is not False
                or label != expected):
            issues.append("INVALID_OR_INCONSISTENT_OPPORTUNITY_TARGET")
            return None, mark, available.isoformat(), issues
        return label, mark, available.isoformat(), issues
    except (KeyError, TypeError, ValueError):
        return None, None, None, ["INVALID_OR_INCONSISTENT_OPPORTUNITY_TARGET"]


def build(observations, session_manifests, policy, asof):
    cutoff = timestamp(asof)
    try:
        frozen = timestamp(policy["frozen_at"])
        available = timestamp(policy["available_at"])
        allowlist = policy["feature_allowlist"]
        valid_policy = (frozen <= available <= cutoff and bool(policy.get("policy_id"))
                        and policy.get("target_definition") == TARGET_DEFINITION
                        and policy.get("selection_scope") == SELECTION_SCOPE
                        and isinstance(allowlist, list) and bool(allowlist)
                        and len(allowlist) == len(set(allowlist))
                        and all(isinstance(name, str) and name for name in allowlist))
    except (KeyError, TypeError, ValueError):
        valid_policy, allowlist = False, []
    if not valid_policy:
        raise ValueError("invalid or unavailable frozen ranking-dataset policy")

    keys = []
    for row in observations:
        key = (str(row.get("signal_date", "")), str(row.get("code", "")).zfill(6))
        if not key[0] or not key[1].isdigit() or len(key[1]) != 6:
            raise ValueError("invalid candidate key")
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate candidate/date observation")

    manifest_by_date = {}
    global_issues = []
    for manifest in session_manifests:
        day = manifest.get("session_date")
        if not day or day in manifest_by_date:
            raise ValueError("missing or duplicate session manifest")
        manifest_by_date[day] = manifest
    observed_dates = sorted({key[0] for key in keys})
    if set(observed_dates) != set(manifest_by_date):
        global_issues.append("SESSION_MANIFEST_DATE_COVERAGE_MISMATCH")

    rows = []
    for observation, (signal_date, code) in zip(observations, keys):
        issues = []
        try:
            decision_at = timestamp(observation["decision_at"])
            observed = timestamp(observation["observed_at"])
            known = timestamp(observation["available_at"])
            if (not frozen <= observed <= known <= decision_at <= cutoff
                    or decision_at.strftime("%Y-%m-%d") != signal_date):
                issues.append("INVALID_CANDIDATE_TIME_SEQUENCE")
        except (KeyError, TypeError, ValueError):
            decision_at = cutoff
            issues.append("INVALID_CANDIDATE_TIME_SEQUENCE")
        boards = finite(observation.get("boards"))
        expected_group = ("FIRST_BOARD" if boards == 1 else "CONSECUTIVE_BOARD"
                          if boards is not None and boards >= 2 and boards.is_integer() else None)
        if (expected_group is None or observation.get("board_group") != expected_group
                or observation.get("pool_assignment") != "CLOSING_LIMIT_UP"
                or observation.get("is_mainboard") is not True or observation.get("is_st") is not False
                or observation.get("security_status") != "verified"
                or observation.get("limit_up_status") != "verified"
                or observation.get("identity_status") != "verified"
                or not observation.get("source")):
            issues.append("INVALID_OBSERVATION_SCOPE_OR_SOURCE")
        features, extras, feature_issues = _valid_feature_bundle(observation, allowlist, decision_at)
        issues.extend(feature_issues)
        label, mark, label_available_at, target_issues = _target(observation, decision_at, cutoff)
        issues.extend(target_issues)
        rows.append({
            "candidate_id": observation.get("candidate_id") or signal_date + ":" + code,
            "code": code,
            "signal_date": signal_date,
            "decision_at": decision_at.isoformat(),
            "boards": int(boards) if boards is not None and boards.is_integer() else None,
            "board_group": expected_group,
            "selected_features": features,
            "ignored_unregistered_features": extras,
            "opportunity_proxy_label": label,
            "b2_close_mark_net_return_proxy": mark,
            "label_available_at": label_available_at,
            "label_available": label is not None,
            "issues": sorted(set(issues)),
        })

    row_ids_by_date = {}
    for row in rows:
        row_ids_by_date.setdefault(row["signal_date"], set()).add(row["candidate_id"])
    manifest_audit = []
    for day, manifest in sorted(manifest_by_date.items()):
        declared_ids = manifest.get("candidate_ids")
        declared_counts = manifest.get("board_level_counts")
        actual = [row for row in rows if row["signal_date"] == day]
        actual_counts = Counter(str(row["boards"]) for row in actual if row["boards"] is not None)
        valid_ids = (isinstance(declared_ids, list) and len(declared_ids) == len(set(declared_ids))
                     and set(declared_ids) == row_ids_by_date.get(day, set()))
        valid_counts = (isinstance(declared_counts, dict)
                        and {str(k): int(v) for k, v in declared_counts.items()} == dict(actual_counts)
                        and manifest.get("closing_limit_count") == len(actual))
        complete = (manifest.get("universe_complete") is True
                    and manifest.get("point_in_time_status_complete") is True
                    and manifest.get("all_observed_board_levels_reported") is True
                    and manifest.get("failed_board_pool_mixed_with_main") is False
                    and valid_ids and valid_counts and bool(manifest.get("source")))
        manifest_audit.append({"session_date": day, "complete": complete,
                               "candidate_count": len(actual), "ids_match": valid_ids,
                               "board_counts_match": valid_counts})
    if not manifest_audit or not all(item["complete"] for item in manifest_audit):
        global_issues.append("INCOMPLETE_FULL_POOL_SESSION_MANIFEST")

    labeled = [row for row in rows if row["label_available"] and not row["issues"]]
    group_counts = {group: sum(row["board_group"] == group for row in labeled) for group in GROUPS}
    board_counts = Counter(str(row["boards"]) for row in labeled)
    day_count = len(manifest_audit)
    issue_count = sum(bool(row["issues"]) for row in rows)
    training_ready = (not global_issues and issue_count == 0 and day_count >= 250 and len(labeled) >= 1000
                      and all(group_counts[group] > 0 for group in GROUPS)
                      and all(board_counts[str(board)] >= 100 for board in range(1, 6)))
    positive = sum(row["opportunity_proxy_label"] is True for row in labeled)
    return {
        "status": "FULL_POOL_RANK_DATASET_READY" if training_ready else "DIAGNOSTIC_ONLY",
        "policy_id": policy["policy_id"],
        "target_definition": TARGET_DEFINITION,
        "selection_scope": SELECTION_SCOPE,
        "rows": rows,
        "session_manifest_audit": manifest_audit,
        "all_candidate_count": len(rows),
        "labeled_proxy_count": len(labeled),
        "unresolved_count": sum(not row["label_available"] for row in rows),
        "opportunity_proxy_rate": positive / len(labeled) if labeled else None,
        "group_labeled_counts": group_counts,
        "board_labeled_counts": dict(sorted(board_counts.items(), key=lambda item: int(item[0]))),
        "session_count": day_count,
        "row_issue_count": issue_count,
        "global_issues": sorted(set(global_issues)),
        "training_ready": training_ready,
        "full_pool_scope_established": training_ready,
        "actual_trade_target": False,
        "execution_verified": False,
        "can_claim_complete_trade_win_probability": False,
        "nonselected_candidates_are_included": True,
        "unresolved_candidates_are_negative_labels": False,
        "limitations": [
            "daily OHLC cannot prove order timing, queue position, fill price, or sellability",
            "the opportunity proxy ranks the full pool but cannot estimate actual complete-trade win rate",
            "actual profitability and large-loss calibration require the separate policy-conditional dataset",
            "a ranking model still requires purged chronological development and an untouched blind period",
        ],
    }


def run(input_path, output_path):
    source, target = Path(input_path), Path(output_path)
    if target.exists():
        raise FileExistsError("choose a new immutable full-pool dataset report")
    payload = json.loads(source.read_text(encoding="utf-8"))
    result = build(**payload)
    result["source_hashes"] = {
        str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest(),
        str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), ensure_ascii=False, indent=2))
