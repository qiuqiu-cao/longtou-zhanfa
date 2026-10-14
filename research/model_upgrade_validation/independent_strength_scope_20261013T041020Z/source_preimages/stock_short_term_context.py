"""Rebuild short-term lifecycle facts without scores, fitting or identity claims.

The input roles remain source-qualified evidence. A theme membership change,
market phase change and a new published narrative are three different facts.
Missing daily links never establish a launch, switch or independent core.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import leader_role_engine as e
import stock_catalyst_context as catalyst
import stock_judgement_review as original
import relay_aesthetic_context as aesthetic


VERSION = "stock-short-term-context-1"
CHECKS = ("launch_and_continuation", "theme_transition", "independent_strength",
          "absorption_and_competition")
STATES = {"SUPPORTED", "CONTRADICTED", "UNKNOWN"}


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _qualified_group(row, name):
    supported = {group for role in row["roles"].values()
                 for group in role["supporting_groups"]}
    if name not in supported:
        return None
    value = row.get("source_groups", {}).get(name, {}).get("value")
    # A valid historical cycle sequence is not necessarily the current cycle.
    if name == "cycle" and value and value.get("records", [{}])[-1].get("date") != row["signal_date"]:
        return None
    return value


def _proof(row, group):
    cell = row.get("source_groups", {}).get(group, {})
    return {key: cell.get(key) for key in ("source_path", "source_sha256", "pointer",
            "observed_at", "available_at", "reconstructed", "adapter") if key in cell}


def _check(gates, facts, support=(), counter=()):
    return {"facts": facts, "gates": gates,
            "supported_allowed": bool(gates) and all(gates.values()),
            "missing_gates": [key for key, value in gates.items() if not value],
            "supporting_refs": list(support), "counterevidence_refs": list(counter)}


def project_rows(role_rows, catalyst_rows, days, signal, track):
    """Pure projection of already-qualified rows; build() supplies checked inputs."""
    if track not in {"STRICT", "DIAGNOSTIC"} or signal not in days:
        raise ValueError("explicit role track and official signal session required")
    candidates = original.unique_rows(catalyst_rows)
    current = {r["code"]: r for r in role_rows if r["signal_date"] == signal}
    if not set(candidates) <= set(current):
        raise ValueError("short-term facts lost original judgement candidates")
    result = []
    for code, cat in sorted(candidates.items()):
        row = current[code]
        if row["boards"] != cat["boards"]:
            raise ValueError("catalyst and role board facts differ")
        history = sorted((r for r in role_rows if r["code"] == code
                          and r["signal_date"] <= signal and r["scope_id"] == row["scope_id"]),
                         key=lambda r: r["signal_date"])
        dates = [r["signal_date"] for r in history]
        if len(dates) != len(set(dates)):
            raise ValueError("duplicate same-stock daily short-term observations")
        expected = days[days.index(dates[0]):days.index(signal)+1]
        missing_days = [day for day in expected if day not in dates]
        consecutive = len(history) >= 2 and not missing_days
        dependency = _qualified_group(row, "dependency")
        path = _qualified_group(row, "path")
        pressure = _qualified_group(row, "pressure")
        competition = _qualified_group(row, "competition")
        theme = _qualified_group(row, "theme")
        cycle = _qualified_group(row, "cycle")
        starts = [b["signal_date"] for a, b in zip(history, history[1:])
                  if days.index(b["signal_date"]) == days.index(a["signal_date"])+1
                  and a["boards"] == 0 and b["boards"] == 1]
        own_start = dependency.get("own_start") if dependency else (starts[-1] if starts else None)
        own_start_known = own_start in days and own_start <= signal if own_start else False
        narratives = [{key: claim.get(key) for key in ("source_id", "event_id", "theme_id", "excerpt", "url",
                       "published_upper", "available_at", "usable_at_original_cutoff")}
                      for claim in cat.get("evidence", []) if claim.get("kind") == "NARRATIVE"]
        # A stock mention or a board-count summary has no topic identifier.
        # Only a reviewed claim explicitly linked to the qualified specific
        # theme can support topic participation; neither proves a cause.
        topic_narratives = [n for n in narratives if theme and _text(n.get("theme_id"))
                            and n["theme_id"] == theme["theme_id"] and _text(n.get("excerpt"))]
        theme_history = []
        for old in history:
            group = _qualified_group(old, "theme")
            theme_history.append({"date": old["signal_date"],
                "theme_id": group.get("theme_id") if group else None,
                "comparison_scope_id": group.get("comparison_scope_id") if group else None,
                "relative_return_ex_self_pp": old["features"].get("theme_return_ex_self_pp"),
                "source": _proof(old, "theme") if group else None})
        themes_complete = (consecutive and all(item["theme_id"] and item["comparison_scope_id"] for item in theme_history)
                           and len({item["comparison_scope_id"] for item in theme_history}) == 1)
        comparable_theme = (themes_complete and all(item["comparison_scope_id"] for item in theme_history)
                            and len({(item["theme_id"], item["comparison_scope_id"])
                                     for item in theme_history}) == 1)
        membership_changes = [{"last_observed_previous_date": a["date"], "first_observed_new_date": b["date"],
                               "from_theme_id": a["theme_id"], "to_theme_id": b["theme_id"],
                               "exact_intraday_switch_known": False}
                              for a, b in zip(theme_history, theme_history[1:])
                              if themes_complete and a["theme_id"] != b["theme_id"]]
        cycle_records = cycle.get("records", []) if cycle else []
        stage_changes = [{"last_observed_previous_date": a["date"], "first_observed_new_date": b["date"],
                          "from_stage": a["state"], "to_stage": b["state"],
                          "is_market_theme_switch": False, "exact_intraday_switch_known": False}
                         for a, b in zip(cycle_records, cycle_records[1:]) if a["state"] != b["state"]]
        own_return, others_return = None, None
        if theme:
            members = theme["members"]
            own_return = e.number(members[code].get("return_pct"))
            peer_returns = [e.number(member.get("return_pct")) for other, member in members.items()
                            if other != code]
            if peer_returns and all(value is not None for value in peer_returns):
                others_return = sum(peer_returns)/len(peer_returns)
        relative_sequence = [item["relative_return_ex_self_pp"] for item in theme_history]
        repeated_relative = (comparable_theme and all(e.number(value) is not None for value in relative_sequence)
                             and all(value > 0 for value in relative_sequence[-2:]))
        surviving_pressure = (pressure is not None and row["features"].get("pressure_pre_event_core") == 1.0
                              and row["features"].get("pressure_survived") == 1.0
                              and row["features"].get("pressure_role_retained") == 1.0)
        theme_pressure = (own_return is not None and others_return is not None
                          and own_return > 0 and others_return <= 0)
        resealed = path is not None and row["features"].get("actual_path_resealed") == 1.0
        board_growth = consecutive and history[-1]["boards"] > history[-2]["boards"]
        board_reset = consecutive and history[-1]["boards"] <= history[-2]["boards"]
        timeline = [{"signal_date": old["signal_date"], "boards": old["boards"],
                     "available_at": old["available_at"], "decision_at": old["decision_at"],
                     "actual_one_word_daily": old.get("actual_one_word_daily"),
                     "source_pool": _proof(old, "pool")} for old in history]
        checks = {
            "launch_and_continuation": _check({
                "source_bound_actual_start": own_start_known,
                "consecutive_same_scope_stock_observations": consecutive,
                "board_continuation_observed": board_growth,
                "checked_specific_topic_narrative_present": bool(topic_narratives)}, {
                    "own_start": own_start if own_start_known else None,
                    "first_observed_pool_date": dates[0], "first_pool_observation_is_launch": False,
                    "anchor_code": dependency.get("anchor_code") if dependency else None,
                    "anchor_stage": dependency.get("anchor_stage") if dependency else None,
                    "launch_or_continuation_cause_confirmed": False,
                    "narrative_evidence": narratives, "specific_topic_narratives": topic_narratives,
                    "stock_mention_or_board_count_is_launch_cause": False},
                support=[key for key, known in (("source_start", own_start_known), ("daily_timeline", consecutive),
                                               ("catalyst_topic_narratives", bool(topic_narratives))) if known],
                counter=["observed_board_sequence_not_continuing"] if board_reset else []),
            "theme_transition": _check({
                "consecutive_verified_specific_theme_memberships": themes_complete,
                "observed_theme_membership_change": bool(membership_changes)}, {
                    "market_stage_history": cycle_records, "market_stage_changes": stage_changes,
                    "market_theme_switch": {"state": "UNKNOWN", "reason": "NO_BOUND_DAILY_MARKET_MAINLINE_HISTORY"},
                    "stock_theme_membership_history": theme_history,
                    "stock_theme_membership_changes": membership_changes,
                    "stock_narrative_switch": {"state": "UNKNOWN", "reason": "NARRATIVE_OCCURRENCE_DOES_NOT_VERIFY_PREVIOUS_NARRATIVE_EXIT"},
                    "new_narrative_presence_is_switch": False,
                    "support_scope": "Observed specific-theme membership change only; market and narrative switches remain separate"},
                support=["daily_theme_memberships"] if themes_complete else [],
                counter=["same_specific_theme_membership_in_observed_window"] if themes_complete and not membership_changes else []),
            "independent_strength": _check({
                "consecutive_comparable_theme_relative_strength": repeated_relative,
                "theme_weakness_excluding_self_with_own_strength": theme_pressure,
                "actual_pressure_before_after_core_retained": surviving_pressure,
                "complete_post_event_competition": pressure is not None}, {
                    "own_return_pct": own_return, "theme_peer_mean_return_pct_ex_self": others_return,
                    "relative_return_ex_self_pp": row["features"].get("theme_return_ex_self_pp"),
                    "pressure_observations": {key: pressure.get(key) for key in (
                        "event_id", "started_at", "ended_at", "pre_cycle_id", "post_cycle_id",
                        "pre_event_observation", "post_event_observation", "post_competition")} if pressure else None,
                    "meaning": "Independent-strength candidate under observed pressure, never verified autonomous funding or core identity",
                    "identity_confirmed": False, "fund_causality_confirmed": False},
                support=[key for key, known in (("daily_theme_memberships", comparable_theme),
                                               ("theme_ex_self_returns", own_return is not None and others_return is not None),
                                               ("pressure_before_after", pressure is not None)) if known],
                counter=(["pressure_after_invalidated"] if pressure and pressure["post_event_observation"]["state"] == "INVALIDATED" else [])
                    + (["not_outperforming_same_theme_in_current_observation"]
                       if comparable_theme and own_return is not None and others_return is not None and own_return <= others_return else [])),
            "absorption_and_competition": _check({
                "actual_complete_open_and_reseal_path": resealed,
                "complete_same_observation_competition": competition is not None}, {
                    "actual_events": path.get("events") if path else None,
                    "actual_open_minutes": row["features"].get("actual_path_open_minutes"),
                    "actual_resealed": row["features"].get("actual_path_resealed"),
                    "competition": competition,
                    "same_height_peers": row.get("peer_codes", []),
                    "joint_promotion_joint_failure_and_position_competition_all_possible": True,
                    "single_volume_increase_confirms_absorption": False,
                    "causal_absorption_or_tradability_confirmed": False,
                    "support_scope": "Observed open/reseal and competition facts only; order-book causality and fills remain unknown"},
                support=[key for key, known in (("actual_price_path", path is not None),
                                               ("complete_competition", competition is not None)) if known],
                counter=["complete_path_ends_open_without_reseal"]
                    if path and path["events"][-1]["state"] == "OPEN" else []),
        }
        counter_details = {
            "observed_board_sequence_not_continuing": {
                "target_claim": "BOARD_CONTINUATION_IN_OBSERVED_WINDOW",
                "window_start": dates[-2] if consecutive else None, "window_end": signal,
                "source_refs": ["daily_timeline"], "does_not_disprove": ["START_DATE", "START_CAUSE"]},
            "same_specific_theme_membership_in_observed_window": {
                "target_claim": "SPECIFIC_THEME_MEMBERSHIP_CHANGED_IN_OBSERVED_WINDOW",
                "window_start": dates[0], "window_end": signal, "source_refs": ["daily_theme_memberships"],
                "does_not_disprove": ["MARKET_MAINLINE_CHANGE", "STOCK_NARRATIVE_CHANGE"]},
            "not_outperforming_same_theme_in_current_observation": {
                "target_claim": "CURRENT_THEME_RELATIVE_OUTPERFORMANCE",
                "window_start": signal, "window_end": signal, "source_refs": ["theme_ex_self_returns"],
                "does_not_disprove": ["HISTORICAL_INDEPENDENT_STRENGTH", "FUNDING_CAUSE"]},
            "pressure_after_invalidated": {
                "target_claim": "CORE_RETAINED_THROUGH_RECORDED_PRESSURE",
                "window_start": pressure.get("started_at") if pressure else None,
                "window_end": pressure.get("post_event_observation", {}).get("observed_at") if pressure else None,
                "source_refs": ["pressure_before_after"], "does_not_disprove": ["OTHER_CYCLE_IDENTITY"]},
            "complete_path_ends_open_without_reseal": {
                "target_claim": "SEALED_AT_LAST_RECORDED_PATH_EVENT",
                "window_start": path["events"][0].get("observed_at") if path else None,
                "window_end": path["events"][-1].get("observed_at") if path else None,
                "source_refs": ["actual_price_path"], "does_not_disprove": ["LATER_RESEAL", "FINAL_CLOSE_WITHOUT_CLOSE_OBSERVATION"]},
        }
        for check in checks.values():
            check["counterevidence_details"] = {ref: counter_details[ref] for ref in check["counterevidence_refs"]}
        result.append({"code": code, "name": cat.get("name"), "boards": row["boards"],
                       "signal_date": signal, "scope_id": row["scope_id"], "track": track,
                       "daily_timeline": timeline, "missing_official_sessions": missing_days,
                       "multi_day_connection_complete": consecutive, "checks": checks,
                       "aesthetic_price_comparison": aesthetic.describe(row),
                       "role_identity_confirmed": False, "fund_causality_confirmed": False,
                       "model_score_call_or_rank_included": False})
    return result


def _market_history(paths, days, signal, moment):
    records, bindings = [], {}
    for path in paths:
        report = e.read(path)
        if not report.get("source_hashes") or not report.get("policy_version"):
            raise ValueError("source-bound mainline report required")
        e.verify(report["source_hashes"])
        day = report["signal_date"]
        formed = e.stamp(report["generated_at"])
        if day not in days or day > signal or formed > moment or not report.get("scope"):
            raise ValueError("mainline date, scope or availability differs")
        if e.stamp(report["knowledge_cutoff"]) > formed:
            raise ValueError("mainline cutoff follows report formation")
        records.append({"date": day, "mainline": report["mainline"], "status": report["status"],
                        "policy_version": report["policy_version"], "available_at": report["generated_at"],
                        "scope": report["scope"], "source_path": str(Path(path).resolve()),
                        "source_sha256": e.sha(path)})
        bindings |= report["source_hashes"] | e.bind(path)
    records.sort(key=lambda r: r["date"])
    if len({r["scope"] for r in records}) > 1:
        raise ValueError("market mainline histories require the same market scope")
    if len({r["date"] for r in records}) != len(records):
        raise ValueError("duplicate market mainline date")
    changes = [{"last_observed_previous_date": a["date"], "first_observed_new_date": b["date"],
                "from_mainline": a["mainline"], "to_mainline": b["mainline"],
                "exact_intraday_switch_known": False, "fund_flow_cause_confirmed": False}
               for a, b in zip(records, records[1:])
               if days.index(b["date"]) == days.index(a["date"])+1
               and a["policy_version"] == b["policy_version"]
               and a["status"] == b["status"] == "INFERRED_SHORT_TERM_MAINLINE"
               and a["mainline"] and b["mainline"] and a["mainline"] != b["mainline"]]
    return {"state": "OBSERVED_MAINLINE_HYPOTHESIS_CHANGE" if changes else "UNKNOWN",
            "records": records, "changes": changes,
            "reason": None if changes else "NO_ADJACENT_SAME_SCOPE_KNOWN_MAINLINE_CHANGE",
            "stock_narrative_or_market_causality_confirmed": False}, bindings


def _build(dataset_path, catalyst_path, track, formed_at, history_paths=(), mainline_paths=()):
    dataset = e.load_dataset(dataset_path)
    context = e.read(catalyst_path)
    if context.get("version") != catalyst.VERSION:
        raise ValueError("original catalyst context schema required")
    e.verify(context["source_bindings"])
    moment = e.stamp(formed_at)
    if e.stamp(context["formed_at"]) > moment or e.stamp(context["knowledge_cutoff"]) > moment:
        raise ValueError("catalyst context follows short-term cutoff")
    if moment > e.stamp(e.now()):
        raise ValueError("short-term context cutoff is in future")
    days = e.calendar(dataset["calendar_path"], dataset["calendar_sha256"])
    signal = context["signal_date"]
    if context["target_date"] != e.future_day(days, signal, 1):
        raise ValueError("catalyst target is not exact next official session")
    rows = e.analyse(dataset, track)["rows"]
    history_bindings, history_records = {}, []
    seen = {r["candidate_id"] for r in rows}
    for path in history_paths:
        historical = e.load_dataset(path)
        if (historical["calendar_path"], historical["calendar_sha256"]) != (dataset["calendar_path"], dataset["calendar_sha256"]):
            raise ValueError("short-term history requires identical official calendar binding")
        if e.stamp(historical["formed_at"]) > moment:
            raise ValueError("historical archive formed after short-term cutoff")
        previous = [r for r in e.analyse(historical, track)["rows"] if r["signal_date"] < signal]
        if seen & {r["candidate_id"] for r in previous}:
            raise ValueError("duplicate historical date/code observations")
        seen.update(r["candidate_id"] for r in previous)
        rows.extend(previous)
        history_bindings |= historical["source_bindings"] | e.bind(path)
        history_records.append({"path": str(Path(path).resolve()), "sha256": e.sha(path),
                                "prior_rows_used": len(previous)})
    # A future date in a multi-date archive must not enter this signal's facts.
    usable = [row for row in rows if row["signal_date"] <= signal]
    if any(e.stamp(row["available_at"]) > moment for row in usable):
        raise ValueError("daily role facts follow short-term cutoff")
    projected = project_rows(usable, context["rows"], days, signal, track)
    market, market_bindings = _market_history(mainline_paths, days, signal, moment)
    for row in projected:
        row["checks"]["theme_transition"]["facts"]["market_theme_switch"] = market
    proofs = dataset["source_bindings"] | context["source_bindings"] | history_bindings | market_bindings | e.bind(dataset_path) | e.bind(catalyst_path) | e.bind(__file__) | e.bind(e.__file__) | e.bind(aesthetic.__file__)
    return {"version": VERSION, "formed_at": formed_at, "knowledge_cutoff": formed_at,
            "signal_date": signal, "target_date": context["target_date"], "track": track,
            "status": "STRICT_SOURCE_FACTS" if track == "STRICT" else "HISTORICAL_DIAGNOSTIC_FUNCTION_PROJECTION",
            "dataset_path": str(Path(dataset_path).resolve()), "dataset_sha256": e.sha(dataset_path),
            "catalyst_context_path": str(Path(catalyst_path).resolve()), "catalyst_context_sha256": e.sha(catalyst_path),
            "original_catalyst_cutoff": context.get("original_feature_cutoff"),
            "source_bindings": proofs, "pool_size": len(projected), "rows": projected,
            "current_dataset_candidate_count": len(dataset["rows"]), "history_datasets": history_records,
            "mainline_report_paths": [str(Path(p).resolve()) for p in mainline_paths],
            "original_time_forecast_or_training_record": False,
            "prediction_registered": False, "new_mature_forward_samples": 0,
            "numeric_scores_or_parameters_changed": False, "automatic_identity_or_entry_allowed": False,
            "track_rule": "STRICT retains original role source availability gates; DIAGNOSTIC may describe late/reconstructed history and cannot certify an original-time forecast.",
            "interpretation": "Short-term hypotheses require author review; narrative occurrence, relative strength, role identity and executable entry remain separate."}


def build(dataset_path, catalyst_path, output, track="STRICT", history_paths=(), mainline_paths=()):
    result = _build(dataset_path, catalyst_path, track, e.now(), history_paths, mainline_paths)
    e.write_new(output, result)
    return result


def checked_projection(facts, required=False):
    path = facts.get("short_term_context_path")
    if path is None:
        if required:
            raise ValueError("new independent judgement requires short-term preparation")
        return None
    e.verify({path: facts["short_term_context_sha256"]})
    saved = e.read(path)
    if saved.get("version") != VERSION:
        raise ValueError("unsupported short-term context schema")
    e.verify(saved["source_bindings"])
    rebuilt = _build(saved["dataset_path"], saved["catalyst_context_path"], saved["track"], saved["formed_at"],
                     [r["path"] for r in saved.get("history_datasets", [])], saved.get("mainline_report_paths", []))
    if saved != rebuilt:
        raise ValueError("short-term projection differs from source reconstruction")
    if saved["signal_date"] != facts["signal_date"] or saved["target_date"] != facts["target_date"]:
        raise ValueError("short-term facts signal/target mismatch")
    if saved["dataset_path"] != facts["role_dataset_path"] or saved["dataset_sha256"] != facts["role_dataset_sha256"]:
        raise ValueError("short-term context differs from bound role dataset")
    if saved["catalyst_context_path"] != facts["catalyst_context_path"] or saved["catalyst_context_sha256"] != facts["catalyst_context_sha256"]:
        raise ValueError("short-term context differs from bound catalyst context")
    if e.stamp(saved["formed_at"]) > e.stamp(facts["prepared_at"]):
        raise ValueError("short-term facts were formed after complete preparation")
    indexed = original.unique_rows(saved["rows"])
    supplied = original.unique_rows(facts["rows"])
    if set(indexed) != set(supplied) or any(row.get("short_term_context") != indexed[code] for code, row in supplied.items()):
        raise ValueError("displayed short-term facts changed or lost original candidates")
    return saved


def validate_assessments(context, draft):
    expected, supplied = original.unique_rows(context["rows"]), original.unique_rows(draft["rows"])
    if set(expected) != set(supplied):
        raise ValueError("short-term review requires complete original judgement pool")
    for code, row in supplied.items():
        assessment = row.get("short_term_assessment", {})
        checks = assessment.get("checks", {})
        if set(checks) != set(CHECKS):
            raise ValueError("four separate short-term checks required: " + code)
        for key, value in checks.items():
            state = value.get("state")
            if state not in STATES or not _text(value.get("reason")):
                raise ValueError("explicit short-term state and reason required")
            facts = expected[code]["checks"][key]
            refs = value.get("evidence_refs", [])
            if not isinstance(refs, list) or any(not _text(ref) for ref in refs):
                raise ValueError("short-term evidence references must be explicit")
            if state == "SUPPORTED" and (not facts["supported_allowed"] or not refs or not set(refs) <= set(facts["supporting_refs"])):
                raise ValueError("supported short-term hypothesis lacks required source gates: " + key)
            if state == "CONTRADICTED" and (not refs or not set(refs) <= set(facts["counterevidence_refs"])):
                raise ValueError("short-term contradiction requires actual bound counterevidence")
            if state == "CONTRADICTED":
                claims = value.get("counter_claims", [])
                required = {facts["counterevidence_details"][ref]["target_claim"] for ref in refs}
                if not isinstance(claims, list) or set(claims) != required:
                    raise ValueError("counterevidence must target its declared narrow claim")
        for key in ("alternative_explanations", "counterevidence", "cancel_conditions"):
            value = assessment.get(key)
            if not isinstance(value, list) or not value or not all(_text(item) for item in value):
                raise ValueError("explicit short-term " + key + " required")
        plan = assessment.get("confirmation_plan")
        if not isinstance(plan, list) or not plan:
            raise ValueError("short-term confirmation timing and requirements required")
        for item in plan:
            when = e.stamp(item.get("planned_at"))
            if (when.astimezone(original.SHANGHAI).date().isoformat() != context["target_date"]
                    or when <= e.stamp(context["formed_at"])
                    or item.get("status") != "PENDING" or not _text(item.get("requirement"))):
                raise ValueError("next-session confirmation must remain explicitly pending")
        if assessment.get("role_identity_confirmed") is not False or assessment.get("fund_causality_confirmed") is not False:
            raise ValueError("short-term checks cannot certify identity or funding cause")
    return supplied


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("build")
    for key in ("dataset", "catalyst-context", "output"):
        a.add_argument("--"+key, required=True)
    a.add_argument("--track", choices=("STRICT", "DIAGNOSTIC"), required=True)
    a.add_argument("--history-dataset", action="append", default=[])
    a.add_argument("--mainline-report", action="append", default=[])
    args = p.parse_args()
    result = build(args.dataset, args.catalyst_context, args.output, args.track, args.history_dataset, args.mainline_report)
    print(json.dumps({key: result[key] for key in ("status", "signal_date", "target_date", "pool_size", "prediction_registered")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
