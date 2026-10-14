"""Exact sampled price comparisons for every declared peer, without identity scores.

The source adapter has already replayed the original prices. This layer retains
each peer and missing path, distinguishing a price-mean advantage from beating
every opponent. Shared sampled timestamps are not synchronized tick events.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import leader_role_engine as e

VERSION = "relay-pairwise-comparison-1"
SCOPES = ("current", "prior")


def _changes(path, day, code):
    """Return the observed price ratios, retaining partial grids as partial."""
    if not path or path.get("signal_date") != day or path.get("code") != code:
        return {}, False, "MISSING_OR_DIFFERENT_STOCK_DAY"
    previous = e.number(path.get("previous_close"))
    bars = path.get("bars", [])
    if previous is None or previous <= 0 or path.get("price_reconciled") is not True:
        return {}, False, "UNRECONCILED_PRICE_OR_REFERENCE"
    values = {}
    for bar in bars:
        minute, close = bar.get("minute"), e.number(bar.get("close"))
        if (not isinstance(minute, str) or minute[:10] != day
                or minute in values or close is None or close <= 0):
            return {}, False, "INVALID_OR_DUPLICATED_OBSERVATION"
        values[minute] = Fraction(str(close))/Fraction(str(previous))*100-100
    complete = path.get("grid_complete") is True and bool(values)
    return values, complete, None if complete else "INCOMPLETE_SAMPLED_GRID"


def _pair(own, other, complete, reason, code):
    minutes = sorted(set(own) & set(other))
    differences = [own[m]-other[m] for m in minutes]
    facts = {"peer_code": code, "state": "UNKNOWN", "complete_pair": complete,
             "reason": reason, "matched_sample_count": len(minutes),
             "first_observed_at": minutes[0] if minutes else None,
             "last_observed_at": minutes[-1] if minutes else None,
             "relative_min_pp": float(min(differences)) if differences else None,
             "relative_max_pp": float(max(differences)) if differences else None,
             "relative_last_pp": float(differences[-1]) if differences else None,
             "own_ahead_samples": sum(x > 0 for x in differences),
             "equal_samples": sum(x == 0 for x in differences),
             "peer_ahead_samples": sum(x < 0 for x in differences),
             "own_ahead_at_every_sample": None,
             "peer_ahead_at_some_sample": None,
             "last_own_change_pct": float(own[minutes[-1]]) if minutes else None,
             "last_peer_change_pct": float(other[minutes[-1]]) if minutes else None,
             "role_displacement_confirmed": False,
             "shared_bar_end_is_not_synchronous_tick_order": True,
             "percentage_gap_can_reflect_limit_price_rounding": True}
    if not complete or not differences:
        return facts
    facts.update(own_ahead_at_every_sample=all(x > 0 for x in differences),
                 peer_ahead_at_some_sample=any(x < 0 for x in differences))
    if all(x > 0 for x in differences):
        facts["state"] = "OWN_PRICE_AHEAD_AT_EVERY_SAMPLE"
    elif all(x == 0 for x in differences):
        facts["state"] = "EQUAL_PRICE_CHANGE_AT_EVERY_SAMPLE"
    elif all(x < 0 for x in differences):
        facts["state"] = "PEER_PRICE_AHEAD_AT_EVERY_SAMPLE"
    else:
        facts["state"] = "MIXED_OR_TIED_PRICE_COMPARISON"
    return facts


def describe(row, value=None, paths=None):
    """Pure description; source_projection supplies qualified original paths."""
    paths = paths or {}
    day, code = row["signal_date"], row["code"]
    own_path = paths.get(day+":"+code)
    own, own_complete, own_reason = _changes(own_path, day, code)
    scopes = {}
    for scope in SCOPES:
        qualified = {g for role in row.get("roles", {}).values() for g in role.get("supporting_groups", [])}
        competition = row.get("source_groups", {}).get("competition", {}).get("value", {}) if "competition" in qualified else {}
        fallback_peers = row.get("peer_codes", []) if scope == "current" and "pool" in qualified else competition.get("peer_codes", []) if scope == "prior" else []
        data = value[scope] if value else {"peer_codes": fallback_peers,
            "scope_complete": "pool" in qualified if scope == "current" else bool(competition)}
        expected = data.get("peer_codes", [])
        pairs, peer_values = [], {}
        for peer in expected:
            peer_path = paths.get(day+":"+peer)
            other, other_complete, other_reason = _changes(peer_path, day, peer)
            same_previous_session = (own_path and peer_path
                and own_path.get("previous_session") == peer_path.get("previous_session")
                and own_path.get("previous_session") is not None)
            complete = bool(own_complete and other_complete and same_previous_session
                            and set(own) == set(other))
            reason = None if complete else own_reason or other_reason or "REFERENCE_SESSION_OR_SAMPLE_GRID_DIFFERENCE"
            pairs.append(_pair(own, other, complete, reason, peer))
            if complete:
                peer_values[peer] = other
        all_complete = bool(expected) and data.get("scope_complete") is True and all(p["complete_pair"] for p in pairs)
        mean_min = data.get("features", {}).get("relative_min_pp")
        original_mean_ahead = e.number(mean_min) is not None and mean_min > 0 and data.get("complete_price_comparison") is True
        exact_mean_differences = [own[m]-sum(v[m] for v in peer_values.values())/len(peer_values)
                                  for m in sorted(own)] if all_complete else []
        mean_ahead = all(x > 0 for x in exact_mean_differences) if all_complete else None
        every_ahead = all(p["own_ahead_at_every_sample"] for p in pairs) if all_complete else None
        any_peer_ahead = any(p["peer_ahead_at_some_sample"] for p in pairs) if all_complete else None
        ranks = [1+sum(v[minute] > own[minute] for v in peer_values.values())
                 for minute in sorted(own)] if all_complete else []
        scopes[scope] = {"comparison_cohort": "CURRENT_CLOSING_SURVIVORS_ONLY" if scope == "current" else "PRIOR_DECLARED_RIVALS_WITH_CURRENT_OBSERVATIONS",
            "declared_peer_codes": expected, "declared_scope_complete": data.get("scope_complete") is True,
            "complete_scope_comparison": all_complete,
            "missing_or_incomplete_peer_codes": [p["peer_code"] for p in pairs if not p["complete_pair"]],
            "pairs": pairs, "peer_mean_ahead_at_every_sample_original": original_mean_ahead if value else None,
            "peer_mean_ahead_at_every_sample_exact_ratios": mean_ahead,
            "original_float_and_exact_ratio_mean_sign_agree": original_mean_ahead == mean_ahead if all_complete else None,
            "exact_ratio_peer_mean_relative_min_pp": float(min(exact_mean_differences)) if exact_mean_differences else None,
            "own_ahead_of_every_peer_at_every_sample": every_ahead,
            "a_peer_ahead_at_some_sample": any_peer_ahead,
            "mean_advantage_does_not_hold_against_every_peer": mean_ahead and not every_ahead if all_complete else None,
            "mean_advantage_with_peer_strictly_ahead": mean_ahead and any_peer_ahead if all_complete else None,
            "price_change_rank_first": ranks[0] if ranks else None,
            "price_change_rank_last": ranks[-1] if ranks else None,
            "price_change_rank_best": min(ranks) if ranks else None,
            "price_change_rank_worst": max(ranks) if ranks else None,
            "rank_is_only_unadjusted_price_change_not_core_position": True,
            "comparison_is_not_full_market_competitor_coverage": True,
                "missing_price_path_does_not_remove_known_peer_membership": True,
            "unknown_is_negative_evidence": False}
    return {"version": VERSION, "candidate_id": row["candidate_id"], "code": code,
            "signal_date": day, "scopes": scopes, "source_qualified": value is not None,
            "not_a_score_or_promotion_call": True, "automatic_role_identity": False,
            "trade_entry_or_absorption_confirmed": False}


def source_projection(row):
    if e.number(row.get("features", {}).get("sampled_current_peer_expected")) is None:
        return describe(row)
    cell = row.get("source_groups", {}).get("sampled_peers", {})
    if not cell.get("source_path") or not cell.get("source_sha256"):
        raise ValueError("qualified peer projection requires exact source binding")
    e.verify({cell["source_path"]: cell["source_sha256"]})
    bundle = e._rich_source_json(str(Path(cell["source_path"]).resolve()), cell["source_sha256"])
    e.verify(bundle["source_bindings"])
    if bundle.get("derivation_kind") != "HISTORICAL_PEER_PRICE_PATH":
        raise ValueError("unsupported pairwise original price source")
    original = bundle.get("groups", {}).get(row["candidate_id"], {}).get("sampled_peers")
    if original is None or original != cell.get("value"):
        raise ValueError("peer values differ from replayed original prices")
    result = describe(row, original, bundle["paths"])
    result["source"] = {key: cell[key] for key in ("source_path", "source_sha256", "pointer", "observed_at", "available_at", "reconstructed") if key in cell}
    return result


def validate_assessments(context, draft):
    """Keep author interpretation explicit, including stronger peers and three paths."""
    expected = {r["code"]: r["pairwise_price_competition"] for r in context["rows"]}
    actual = {r["code"]: r for r in draft["rows"]}
    if len(actual) != len(draft["rows"]) or set(actual) != set(expected):
        raise ValueError("pairwise review must preserve complete original candidates")
    for code, row in actual.items():
        assessment = row.get("pairwise_competition_assessment", {})
        if (set(assessment.get("scopes", {})) != set(SCOPES)
                or assessment.get("price_rank_is_not_core_identity") is not True
                or assessment.get("role_causality_or_entry_confirmed") is not False):
            raise ValueError("separate pairwise scope review and no automatic identity required")
        for scope, review in assessment["scopes"].items():
            facts = expected[code]["scopes"][scope]
            state = review.get("state")
            if state not in {"REVIEWED", "PARTIAL_REVIEWED", "UNKNOWN"} or not isinstance(review.get("reason"), str) or not review["reason"].strip():
                raise ValueError("pairwise review requires explicit state and interpretation")
            if state == "REVIEWED" and not facts["complete_scope_comparison"]:
                raise ValueError("incomplete pair cohort cannot be reviewed as complete")
            if state == "PARTIAL_REVIEWED" and (facts["complete_scope_comparison"] or not any(p["complete_pair"] for p in facts["pairs"])):
                raise ValueError("partial pair review needs actual available pair facts")
            stronger = sorted(p["peer_code"] for p in facts["pairs"] if p["peer_ahead_at_some_sample"] is True)
            for key, value in (("comparison_peer_codes", facts["declared_peer_codes"]),
                               ("incomplete_peer_codes", facts["missing_or_incomplete_peer_codes"]),
                               ("stronger_observed_peer_codes", stronger)):
                if review.get(key) != value:
                    raise ValueError("pairwise review omitted or changed " + key)
            paths = review.get("next_session_paths", {})
            if set(paths) != {"joint_promotion", "joint_failure", "divergence"}:
                raise ValueError("joint promotion, joint failure and divergence paths required")
            for path in paths.values():
                if path.get("state") not in {"POSSIBLE", "CONDITIONAL", "UNKNOWN"} or not isinstance(path.get("reason"), str) or not path["reason"].strip():
                    raise ValueError("each competing next-session path needs an explicit reason")


@e.verified_session
def build(dataset_path, output, track):
    if track not in {"STRICT", "DIAGNOSTIC"}:
        raise ValueError("explicit availability track required")
    codes = e.bind(__file__) | e.bind(e.__file__)
    e.verify(codes)
    dataset = e.load_dataset(dataset_path)
    days = e.calendar(dataset["calendar_path"], dataset["calendar_sha256"])
    ids = [r["candidate_id"] for r in dataset["rows"]]
    if len(ids) != len(set(ids)):
        raise ValueError("original candidate identities are duplicated")
    selected = [r for r in dataset["rows"] if "sampled_peers" in r["groups"]]
    if any(r["signal_date"] not in days or r["candidate_id"] != r["signal_date"]+":"+r["code"]
           or e.stamp(r["available_at"]) > e.stamp(e.now())
           or e.stamp(r["decision_at"]) > e.stamp(e.now()) for r in selected):
        raise ValueError("pairwise candidate calendar, identity or availability mismatch")
    rows = [source_projection(e.evidence_row(r, track, days)) for r in selected]
    e.verify(codes)
    result = {"version": VERSION, "formed_at": e.now(), "track": track,
        "all_original_ids": ids, "all_original_count": len(ids), "selected_case_count": len(rows),
        "rows": rows, "complete_scope_counts": {scope: sum(r["scopes"][scope]["complete_scope_comparison"] for r in rows) for scope in SCOPES},
        "mean_advantage_hidden_peer_counts": {scope: sum(r["scopes"][scope]["mean_advantage_with_peer_strictly_ahead"] is True for r in rows) for scope in SCOPES},
        "source_bindings": dataset["source_bindings"] | e.bind(dataset_path) | codes,
        "dataset_path": str(Path(dataset_path).resolve()), "numeric_fits": 0,
        "prediction_registered": False, "new_mature_forward_samples": 0}
    e.write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--track", choices=("STRICT", "DIAGNOSTIC"), required=True)
    args = parser.parse_args()
    report = build(args.dataset, args.output, args.track)
    print(json.dumps({k: report[k] for k in ("selected_case_count", "complete_scope_counts", "mean_advantage_hidden_peer_counts", "prediction_registered")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
