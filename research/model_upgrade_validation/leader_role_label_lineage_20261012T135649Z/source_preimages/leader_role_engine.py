"""Source-bound multi-role evidence; role hypotheses never authorize trades.

Rich groups must point to exact saved JSON values, not caller-made verified
booleans. Historical reconstruction and strict availability remain separate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from current_model_diagnostics import RELAY_FIELDS

VERSION = "leader-roles-1"
ROLES = ("HEIGHT", "THEME", "TURNOVER", "CAPACITY", "TREND", "POSITION",
         "SUPPLEMENT", "NODE", "CYCLE", "CROSS_CYCLE", "EMOTION")
REQUIRED = {
    "HEIGHT": ("pool",), "THEME": ("pool", "theme"),
    "TURNOVER": ("pool", "path"), "CAPACITY": ("theme", "capacity"),
    "TREND": ("theme", "capacity", "history"), "POSITION": ("competition",),
    "SUPPLEMENT": ("theme", "dependency"), "NODE": ("theme", "cycle", "competition"),
    "CYCLE": ("theme", "cycle"), "CROSS_CYCLE": ("cycle", "pressure", "history"),
    "EMOTION": ("cycle", "emotion"),
}
RELATION_FIELDS = ("theme_peer_count", "theme_amount_share", "theme_return_ex_self_pp",
    "theme_first_count", "theme_middle_count", "theme_height_gap", "same_height_peer_count",
    "same_height_amount_percentile", "same_height_turnover_percentile", "actual_path_open_minutes",
    "actual_path_resealed", "price_history_return_5", "price_history_return_20",
    "capacity_amount_percentile", "dependency_height_gap", "cycle_pressure_count",
    "competition_failed_share", "emotion_response_count", "capacity_float_cap", "dependency_start_lag_sessions",
    "dependency_anchor_stable", "cycle_is_start", "cycle_is_repair", "cycle_is_acceleration",
    "cycle_is_consensus", "cycle_is_divergence", "cycle_is_retreat", "cycle_transition_count",
    "pressure_elapsed_sessions", "pressure_post_competition", "emotion_direction_agreement",
    "dependency_anchor_is_divergence", "dependency_anchor_is_retreat", "dependency_start_after_anchor_break_sessions",
    "pressure_pre_event_core", "pressure_survived", "pressure_role_retained", "pressure_rank_improvement",
    "pressure_price_change_pct", "pressure_cycle_changed", "scope_max_boards", "scope_top_count")
HEIGHT_TEMPORAL_FIELDS = tuple(
    f"{field}_{lag}" for field in ("scope_height_delta", "own_boards_delta",
                                  "height_lead_retained", "height_expansion_steps")
    for lag in (1, 3, 5))
FEATURES = tuple(RELAY_FIELDS) + RELATION_FIELDS + tuple(
    f"{field}_delta_{lag}" for field in ("theme_amount_share", "theme_return_ex_self_pp", "price", "turnover")
    for lag in (1, 3, 5)) + HEIGHT_TEMPORAL_FIELDS
GROUPS = set(g for values in REQUIRED.values() for g in values) | {"price"}


def now():
    return datetime.now(timezone.utc).isoformat()


def stamp(value):
    d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise ValueError("timezone-aware timestamp required")
    return d


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return {str(Path(path).resolve()): sha(path)}


def verify(sources):
    for path, expected in sources.items():
        if not Path(path).is_file() or sha(path) != expected:
            raise ValueError("source hash mismatch: " + path)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def write_new(path, payload):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as f:
        json.dump(clean(payload), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def number(value):
    if isinstance(value, bool):
        return None
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def calendar(path, expected=None):
    if expected and sha(path) != expected:
        raise ValueError("calendar hash changed")
    d = read(path)
    days = d.get("sessions", [])
    if d.get("status") != "verified" or not d.get("source") or not days or days != sorted(set(days)):
        raise ValueError("verified complete ordered official calendar required")
    if any(datetime.fromisoformat(s).date().isoformat() != s for s in days):
        raise ValueError("invalid session date")
    if d.get("source_manifest"):
        verify({d["source_manifest"]: d["source_manifest_sha256"]})
    return days


def future_day(days, signal, horizon):
    if signal not in days or days.index(signal) + horizon >= len(days):
        raise ValueError("official calendar cannot assign exact horizon")
    return days[days.index(signal) + horizon]


def _value(source, pointer):
    if not isinstance(pointer, list) or not pointer:
        raise ValueError("exact JSON source pointer required")
    for key in pointer:
        if str(key).startswith(("target", "future", "outcome", "label", "exit", "truth")):
            raise ValueError("future/label field forbidden in identity facts")
        source = source[key]
    return source


def _group(row, kind, track):
    cell = row.get("groups", {}).get(kind)
    if not cell:
        return None, "MISSING_" + kind.upper()
    if cell.get("status") not in ("VERIFIED", "INFERRED"):
        return None, "UNVERIFIED_" + kind.upper()
    if track == "STRICT" and stamp(row["available_at"]) > stamp(row["decision_at"]):
        return None, "COMPLETE_DATASET_LATE_" + kind.upper()
    observed, available, decision = map(stamp, (cell["observed_at"], cell["available_at"], row["decision_at"]))
    if observed > available or observed.date().isoformat() > row["signal_date"]:
        raise ValueError("group observation is inconsistent or after signal")
    if available > stamp(now()):
        raise ValueError("group captured in the future")
    if track == "STRICT" and available > decision:
        return None, "LATE_" + kind.upper()
    if track == "STRICT" and cell.get("reconstructed") is True:
        return None, "RECONSTRUCTED_" + kind.upper()
    # The adapter groups are rebuilt from immutable originals on load.
    if cell.get("adapter") not in ("SAVED_BASE_ALLOWLIST", "SOURCE_POOL_ALLOWLIST"):
        path = cell["source_path"]
        verify({path: cell["source_sha256"]})
        source = read(path)
        # Normalized fact bundles retain their original raw source bindings.
        # A stable bundle hash alone must not hide changed upstream evidence.
        if source.get("source_bindings"):
            verify(source["source_bindings"])
        formed = next((source[k] for k in ("formed_at", "generated_at", "recorded_at", "available_at") if source.get(k)), None)
        if formed is None or stamp(formed) > available:
            raise ValueError("rich fact source formation is not bound")
        if _value(source, cell["pointer"]) != cell["value"]:
            raise ValueError("rich group differs from source JSON")
    value = cell["value"]
    if not isinstance(value, dict) or value.get("signal_date", row["signal_date"]) != row["signal_date"]:
        raise ValueError("group signal date mismatch")
    if value.get("code", row["code"]) != row["code"]:
        raise ValueError("group stock mismatch")
    def check_nested(obj):
        if isinstance(obj, dict):
            for key, item in obj.items():
                if key in ("anchor_start", "own_start", "anchor_break_day") and isinstance(item, str):
                    day = datetime.fromisoformat(item).date().isoformat()
                    if day != item or day > row["signal_date"]:
                        raise ValueError("nested dependency date follows signal or is not an ISO date")
                elif key.endswith("_at") and isinstance(item, str):
                    from zoneinfo import ZoneInfo
                    when = stamp(item)
                    if when > available or when.astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat() > row["signal_date"]:
                        raise ValueError("nested event follows observation availability/signal")
                else:
                    check_nested(item)
        elif isinstance(obj, list):
            for item in obj:
                check_nested(item)
    check_nested(value)
    return value, None


def _valid_group(kind, v, row):
    if kind == "pool":
        codes = v.get("codes", [])
        boards = v.get("boards", {})
        return (v.get("complete_within_scope") is True and codes == sorted(set(codes))
                and row["code"] in codes and set(boards) == set(codes)
                and all(type(x) is int and x >= 0 for x in boards.values())
                and boards.get(row["code"]) == row["boards"])
    if kind == "theme":
        codes = v.get("codes", [])
        return (v.get("membership_verified") is True and v.get("complete_within_scope") is True
                and v.get("granularity") == "specific" and bool(v.get("theme_id"))
                and row["code"] in codes and len(codes) == len(set(codes)) and len(codes) >= 2
                and set(v.get("members", {})) == set(codes))
    if kind == "path":
        events = v.get("events", [])
        return (v.get("event_sequence_complete") is True and len(events) >= 2
                and all(e.get("state") in ("OPEN", "SEALED") and e.get("observed_at") for e in events)
                and all(stamp(e["observed_at"]).date().isoformat() == row["signal_date"] for e in events)
                and all(stamp(a["observed_at"]) < stamp(b["observed_at"]) for a, b in zip(events, events[1:])))
    if kind == "capacity":
        return (v.get("includes_non_limit_members") is True and v.get("complete_within_scope") is True
                and row["code"] in v.get("amounts", {}) and number(v.get("float_cap")) is not None
                and all(number(x) is not None and number(x) >= 0 for x in v["amounts"].values()))
    if kind == "history":
        bars = v.get("bars", [])
        dates = [b.get("date") for b in bars]
        return (v.get("unadjusted") is True and len(bars) >= 21 and dates == sorted(set(dates))
                and dates[-1] == row["signal_date"]
                and all(number(b.get("close")) is not None and number(b["close"]) > 0 for b in bars))
    if kind == "competition":
        peers = v.get("peer_codes", [])
        return (v.get("complete_within_scope") is True and peers == sorted(set(peers))
                and row["code"] not in peers and len(peers) > 0
                and set(v.get("prior", {})) == set(peers) | {row["code"]}
                and set(v.get("current", {})) == set(v["prior"])
                and bool(v.get("prior_at")) and bool(v.get("current_at"))
                and stamp(v["prior_at"]) < stamp(v["current_at"]) <= stamp(row["decision_at"])
                and all(s in ("STRONG", "FAILED", "COMPETING") for s in v["current"].values()))
    if kind == "dependency":
        return (bool(v.get("anchor_code")) and v["anchor_code"] != row["code"]
                and v.get("same_specific_theme") is True and v.get("anchor_identity_source")
                and v.get("anchor_stable") in (True, False)
                and v.get("own_start") and v.get("anchor_start") and v["anchor_start"] < v["own_start"])
    if kind == "cycle":
        records = v.get("records", [])
        return (len(records) >= 3 and len({r.get("date") for r in records}) == len(records)
                and [r["date"] for r in records] == sorted(r["date"] for r in records)
                and records[-1]["date"] <= row["signal_date"]
                and all(r.get("state") in ("START", "REPAIR", "ACCELERATION", "CONSENSUS", "DIVERGENCE", "RETREAT") for r in records))
    if kind == "pressure":
        before, after = v.get("pre_event_observation", {}), v.get("post_event_observation", {})
        return (bool(v.get("event_id")) and v.get("pre_event_role") and v.get("started_at") and v.get("ended_at")
                and bool(v.get("pre_cycle_id")) and bool(v.get("post_cycle_id"))
                and all(obs.get("code") == row["code"] and obs.get("observed_at")
                    and obs.get("state") in ("CORE", "COMPETING", "INVALIDATED")
                    and type(obs.get("relative_rank")) is int and obs["relative_rank"] >= 1
                    and number(obs.get("price")) is not None and number(obs["price"]) > 0 for obs in (before, after))
                and stamp(before["observed_at"]) <= stamp(v["started_at"]) < stamp(v["ended_at"])
                <= stamp(after["observed_at"]) <= stamp(row["decision_at"])
                and v.get("post_event_competition_verified") is True
                and _valid_group("competition", v.get("post_competition", {}), row)
                and stamp(v["post_competition"]["prior_at"]) == stamp(before["observed_at"])
                and stamp(v["post_competition"]["current_at"]) == stamp(after["observed_at"]))
    if kind == "emotion":
        return (v.get("market_controlled") is True and v.get("common_catalyst_reviewed") is True
                and len(v.get("responses", [])) >= 2
                and all(stamp(r["leader_at"]) < stamp(r["market_at"]) <= stamp(row["decision_at"]) for r in v["responses"]))
    return True


def evidence_row(row, track, days):
    values, missing = {}, {}
    for group in sorted(GROUPS):
        value, reason = _group(row, group, track)
        if value is not None and not _valid_group(group, value, row):
            reason, value = "INCOMPLETE_" + group.upper(), None
        # Histories must use consecutive official sessions, not sparse pool dates.
        if group == "history" and value:
            expected = days[max(0, days.index(row["signal_date"])-len(value["bars"])+1):days.index(row["signal_date"])+1]
            if [b["date"] for b in value["bars"]] != expected:
                reason, value = "NONCONSECUTIVE_HISTORY", None
        values[group], missing[group] = value, reason
    features = {k: None for k in FEATURES}
    price = values.get("price") or {}
    features.update({k: number(price.get(k)) for k in RELAY_FIELDS})
    pool, theme, path = (values[k] for k in ("pool", "theme", "path"))
    peers = []
    relations = []
    if pool:
        features["scope_max_boards"] = max(pool["boards"].values())
        features["scope_top_count"] = sum(b == features["scope_max_boards"] and b > 0
                                          for b in pool["boards"].values())
    if pool and row["boards"] > 0:
        b = pool["boards"][row["code"]]
        peers = sorted(c for c in pool["codes"] if c != row["code"] and pool["boards"][c] == b)
        features["same_height_peer_count"] = len(peers)
        for field, out in (("amounts", "same_height_amount_percentile"), ("turnovers", "same_height_turnover_percentile")):
            vals = pool.get(field, {})
            if all(number(vals.get(c)) is not None for c in [row["code"], *peers]):
                own = number(vals[row["code"]])
                features[out] = sum(number(vals[c]) <= own for c in [row["code"], *peers])/(len(peers)+1)
        relations.extend({"kind": "SAME_HEIGHT", "to": c} for c in peers)
    if theme:
        members = theme["members"]
        others = [c for c in theme["codes"] if c != row["code"]]
        features["theme_peer_count"] = len(others)
        amounts = [number(members[c].get("amount")) for c in theme["codes"]]
        returns = [number(members[c].get("return_pct")) for c in others]
        own = number(members[row["code"]].get("return_pct"))
        if all(x is not None for x in amounts) and sum(amounts) > 0:
            features["theme_amount_share"] = number(members[row["code"]]["amount"])/sum(amounts)
        if returns and all(x is not None for x in returns) and own is not None:
            features["theme_return_ex_self_pp"] = own-sum(returns)/len(returns)
        heights = [number(members[c].get("boards")) for c in others]
        if all(x is not None for x in heights):
            features["theme_first_count"] = sum(h == 1 for h in heights)
            features["theme_middle_count"] = sum(h >= 2 for h in heights)
            features["theme_height_gap"] = max([number(row["boards"]), *heights])-number(row["boards"])
        relations.append({"kind": "THEME_MEMBER", "to": theme["theme_id"], "peer_codes": sorted(others)})
    if path:
        minutes, opened, resealed = 0., None, False
        for event in path["events"]:
            when = stamp(event["observed_at"])
            if event["state"] == "OPEN" and opened is None:
                opened = when
            elif event["state"] == "SEALED" and opened is not None:
                # Exclude the lunch break, never use first-to-last seal span.
                span = (when-opened).total_seconds()/60
                from zoneinfo import ZoneInfo
                a, b = opened.astimezone(ZoneInfo("Asia/Shanghai")), when.astimezone(ZoneInfo("Asia/Shanghai"))
                lunch_a, lunch_b = a.replace(hour=11, minute=30, second=0), a.replace(hour=13, minute=0, second=0)
                overlap = max(0., (min(b, lunch_b)-max(a, lunch_a)).total_seconds()/60)
                minutes += max(0., span-overlap)
                opened, resealed = None, True
        features["actual_path_open_minutes"] = minutes if opened is None else None
        features["actual_path_resealed"] = float(resealed and opened is None)
    if values["history"]:
        bars = values["history"]["bars"]
        for lag in (5, 20):
            features[f"price_history_return_{lag}"] = (bars[-1]["close"]/bars[-1-lag]["close"]-1)*100
    if values["capacity"]:
        amounts = values["capacity"]["amounts"]
        own = amounts[row["code"]]
        features["capacity_amount_percentile"] = sum(a <= own for a in amounts.values())/len(amounts)
        features["capacity_float_cap"] = values["capacity"]["float_cap"]
    if values["competition"]:
        v = values["competition"]
        features["competition_failed_share"] = sum(v["current"][c] == "FAILED" for c in v["peer_codes"])/len(v["peer_codes"])
        relations.extend({"kind": "COMPETITOR", "to": c, "state": v["current"][c]} for c in v["peer_codes"])
    if values["dependency"]:
        v = values["dependency"]
        relations.append({"kind": "DEPENDENCY", "to": v["anchor_code"]})
        if number(v.get("anchor_boards")) is not None:
            features["dependency_height_gap"] = v["anchor_boards"]-row["boards"]
        if v["own_start"] in days and v["anchor_start"] in days:
            features["dependency_start_lag_sessions"] = days.index(v["own_start"])-days.index(v["anchor_start"])
        features["dependency_anchor_stable"] = float(v["anchor_stable"])
        if v.get("anchor_stage") in ("START", "REPAIR", "ACCELERATION", "CONSENSUS", "DIVERGENCE", "RETREAT"):
            features["dependency_anchor_is_divergence"] = float(v["anchor_stage"] == "DIVERGENCE")
            features["dependency_anchor_is_retreat"] = float(v["anchor_stage"] == "RETREAT")
        if v.get("anchor_break_day") in days and v["own_start"] in days:
            features["dependency_start_after_anchor_break_sessions"] = days.index(v["own_start"])-days.index(v["anchor_break_day"])
    if values["cycle"]:
        records = values["cycle"]["records"]
        for state in ("START", "REPAIR", "ACCELERATION", "CONSENSUS", "DIVERGENCE", "RETREAT"):
            features["cycle_is_"+state.lower()] = float(records[-1]["state"] == state)
        features["cycle_transition_count"] = sum(a["state"] != b["state"] for a, b in zip(records, records[1:]))
    if values["pressure"]:
        features["cycle_pressure_count"] = 1
        relations.append({"kind": "PRESSURE_EVENT", "to": values["pressure"]["event_id"]})
        v = values["pressure"]
        before, after = v["pre_event_observation"], v["post_event_observation"]
        features["pressure_pre_event_core"] = float(before["state"] == "CORE")
        features["pressure_survived"] = float(after["state"] != "INVALIDATED")
        features["pressure_role_retained"] = float(after["state"] == "CORE")
        features["pressure_rank_improvement"] = before["relative_rank"]-after["relative_rank"]
        features["pressure_price_change_pct"] = (number(after["price"])/number(before["price"])-1)*100
        features["pressure_cycle_changed"] = float(v["pre_cycle_id"] != v["post_cycle_id"])
        from zoneinfo import ZoneInfo
        start = stamp(values["pressure"]["started_at"]).astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat()
        features["pressure_elapsed_sessions"] = sum(start <= d <= row["signal_date"] for d in days)
        features["pressure_post_competition"] = float(values["pressure"]["post_event_competition_verified"])
    if values["emotion"]:
        features["emotion_response_count"] = len(values["emotion"]["responses"])
        pairs = [(number(r.get("leader_change")), number(r.get("market_change"))) for r in values["emotion"]["responses"]]
        if all(a is not None and b is not None for a, b in pairs):
            features["emotion_direction_agreement"] = sum(a*b > 0 for a, b in pairs)/len(pairs)
    eligible, roles = {}, {}
    for role in ROLES:
        gaps = [missing[g] for g in REQUIRED[role] if missing[g]]
        eligible[role] = not gaps
        call = None
        if not gaps and role == "HEIGHT":
            call = row["boards"] > 0 and pool["boards"][row["code"]] == max(pool["boards"].values())
        # Other identities are hypotheses, never certified from one feature.
        roles[role] = {"state": "EVIDENCE_SUPPORTED" if call is True else "COMPETING" if call is False else "UNKNOWN" if gaps else "CANDIDATE",
            "baseline_call": call, "score": None, "missing": gaps,
            "supporting_groups": [g for g in REQUIRED[role] if values[g] is not None],
            "counterevidence": [], "invalidation": "Recheck required evidence, peer competition and hard risks at next actual observation"}
    return {"candidate_id": row["candidate_id"], "signal_date": row["signal_date"], "code": row["code"],
        "name": row.get("name"), "boards": row["boards"], "decision_at": row["decision_at"],
        "available_at": row["available_at"], "scope_id": row["scope_id"],
        "pool_kind": row.get("pool_kind", "limit_up"),
        "actual_one_word_daily": row.get("actual_one_word_daily"),
        "low_turnover_proxy": price.get("signal_one_word"),
        "features": features, "role_eligible": eligible, "roles": roles, "relations": relations,
        "peer_codes": peers, "evidence_grade": "SOURCE_BOUND_NOT_EXTERNAL_TRUTH_CERTIFICATION",
        "source_groups": row.get("groups", {}), "role_identity_automatically_certified": False}


def _height_temporal_features(rows, days, track):
    """Past official sessions in the same declared pool, including missing rivals.

    Scope height can be compared across changing daily stock membership. Own
    continuity remains unknown if the stock or any intervening pool is absent.
    Pool departure is never an identity failure label.
    """
    pools = {}
    for row in rows:
        if not row["role_eligible"]["HEIGHT"]:
            continue
        cell = row["source_groups"]["pool"]
        key = (row["signal_date"], row["scope_id"])
        boards = cell["value"]["boards"]
        available = max(stamp(row["available_at"]), stamp(cell["available_at"]))
        if key in pools:
            if pools[key]["boards"] != boards:
                raise ValueError("same date/scope has conflicting height pools")
            pools[key]["available_at"] = max(pools[key]["available_at"], available)
        else:
            pools[key] = {"boards": boards, "available_at": available}
    positions = {day: i for i, day in enumerate(days)}
    for row in rows:
        if not row["role_eligible"]["HEIGHT"]:
            continue
        pos = positions[row["signal_date"]]
        current = pools[(row["signal_date"], row["scope_id"])]["boards"]
        for lag in (1, 3, 5):
            if pos < lag:
                continue
            sequence = [pools.get((day, row["scope_id"])) for day in days[pos-lag:pos+1]]
            prior = sequence[0]
            if not prior or (track == "STRICT" and prior["available_at"] > stamp(row["decision_at"])):
                continue
            row["features"][f"scope_height_delta_{lag}"] = max(current.values())-max(prior["boards"].values())
            code = row["code"]
            if code in prior["boards"]:
                row["features"][f"own_boards_delta_{lag}"] = current[code]-prior["boards"][code]
            if (any(pool is None for pool in sequence)
                    or track == "STRICT" and any(pool["available_at"] > stamp(row["decision_at"]) for pool in sequence)):
                continue
            maxima = [max(pool["boards"].values()) for pool in sequence]
            row["features"][f"height_expansion_steps_{lag}"] = sum(b > a for a, b in zip(maxima, maxima[1:]))
            if all(code in pool["boards"] for pool in sequence):
                row["features"][f"height_lead_retained_{lag}"] = float(all(
                    pool["boards"][code] > 0 and pool["boards"][code] == maximum
                    for pool, maximum in zip(sequence, maxima)))


def analyse(dataset, track):
    if track not in ("DIAGNOSTIC", "STRICT") or dataset.get("version") != VERSION:
        raise ValueError("unsupported dataset or track")
    verify(dataset["source_bindings"])
    days = calendar(dataset["calendar_path"], dataset["calendar_sha256"])
    rows = dataset["rows"]
    ids = [r["candidate_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate date/code candidate")
    for r in rows:
        if r["signal_date"] not in days or r["candidate_id"] != r["signal_date"]+":"+r["code"]:
            raise ValueError("candidate date/id mismatch")
        if len(r["code"]) != 6 or not r["code"].isdigit() or type(r["boards"]) is not int or r["boards"] < 0:
            raise ValueError("invalid stock or board count")
        if stamp(r["available_at"]) > stamp(now()):
            raise ValueError("future dataset availability")
        if stamp(r["decision_at"]) > stamp(now()):
            raise ValueError("future decision cutoff")
    output = [evidence_row(r, track, days) for r in rows]
    _height_temporal_features(output, days, track)
    index = {(r["signal_date"], r["code"]): r for r in output}
    for r in output:
        pos = days.index(r["signal_date"])
        for field in ("theme_amount_share", "theme_return_ex_self_pp", "price", "turnover"):
            for lag in (1, 3, 5):
                prior = index.get((days[pos-lag], r["code"])) if pos >= lag else None
                a, b = r["features"][field], prior["features"][field] if prior else None
                theme_comparable = True
                if field.startswith("theme_"):
                    current_theme = r["source_groups"].get("theme", {}).get("value", {})
                    prior_theme = prior["source_groups"].get("theme", {}).get("value", {}) if prior else {}
                    theme_comparable = (bool(current_theme.get("comparison_scope_id"))
                        and current_theme.get("theme_id") == prior_theme.get("theme_id")
                        and current_theme["comparison_scope_id"] == prior_theme.get("comparison_scope_id"))
                if (prior and prior["scope_id"] == r["scope_id"] and theme_comparable and a is not None and b is not None
                        and (track == "DIAGNOSTIC" or stamp(prior["available_at"]) <= stamp(r["decision_at"]))):
                    r["features"][f"{field}_delta_{lag}"] = a-b
    return {"version": VERSION, "formed_at": now(), "track": track, "rows": output,
        "candidate_count": len(rows), "official_full_market": False,
        "qualification": {role: {"eligible": sum(r["role_eligible"][role] for r in output),
            "unknown": sum(not r["role_eligible"][role] for r in output),
            "missing_reasons": dict(Counter(g for r in output for g in r["roles"][role]["missing"]))} for role in ROLES},
        "calibrated_probability": False, "production_changed": False, "trade_authorized": False}


def from_archive(features_dir, calendar_path, theme_path=None, output=None):
    """Rebuild fixed signal allowlist and scoped pool; no target fields copied."""
    fpath, mpath = Path(features_dir)/"features.json", Path(features_dir)/"manifest.json"
    data, manifest = read(fpath), read(mpath)
    if sha(fpath) != manifest["features_sha256"] or data["signal_date"] != manifest["signal_date"]:
        raise ValueError("feature archive hash/date mismatch")
    verify(manifest["source_hashes"])
    days = calendar(calendar_path)
    signal = data["signal_date"]
    if data["target_date"] != future_day(days, signal, 1):
        raise ValueError("archive target is not next official session")
    roster = data["original_pool"]["rows"]
    codes = sorted(r["code"] for r in roster)
    if len(codes) != len(set(codes)):
        raise ValueError("duplicate original roster")
    price = {r["code"]: r for r in data["relay"]}
    # The original relay model intentionally has no first-board predictions.
    # Role identification must still retain their already archived quote facts.
    projected = closing_price_facts(roster, price)
    sources = manifest["source_hashes"] | bind(fpath) | bind(mpath) | bind(calendar_path)
    if theme_path:
        t = read(theme_path)
        verify(t["source_hashes"])
        if t["signal_date"] != signal or set(c["code"] for c in t["all_candidates"] if c["pool_kind"] == "limit_up") != set(codes):
            raise ValueError("theme lost original closing pool")
        sources |= bind(theme_path) | t["source_hashes"]
    cutoff = manifest["knowledge_cutoff"]
    pool = {"signal_date": signal, "complete_within_scope": True, "codes": codes,
        "boards": {r["code"]: int(r["boards"]) for r in roster},
        "amounts": {c: projected[c]["signal_amount"] for c in codes},
        "turnovers": {c: projected[c]["turnover"] for c in codes}}
    rows = []
    observed = signal+"T15:00:00+08:00"
    for r in roster:
        groups = {}
        for kind, value in (("pool", pool), ("price", projected[r["code"]])):
            groups[kind] = {"value": value, "status": "VERIFIED", "observed_at": observed,
                "available_at": manifest["formed_at"], "adapter": "SAVED_BASE_ALLOWLIST", "reconstructed": False}
        rows.append({"candidate_id": signal+":"+r["code"], "signal_date": signal,
            "code": r["code"], "name": r.get("name"), "boards": int(r["boards"]),
            "scope_id": data["scope"], "decision_at": cutoff, "available_at": manifest["formed_at"],
            "actual_one_word_daily": price.get(r["code"], {}).get("actual_one_word_daily"), "groups": groups})
    if theme_path:
        # Failed-board observations stay in the identity denominator. Their
        # attempted board count is not a confirmed closing board height.
        for r in t["all_candidates"]:
            if r["pool_kind"] != "failed_board":
                continue
            if r["code"] in codes:
                raise ValueError("stock appears in both closing and failed pools")
            rows.append({"candidate_id": signal+":"+r["code"], "signal_date": signal,
                "code": r["code"], "name": r.get("name"), "boards": 0, "pool_kind": "failed_board",
                "scope_id": data["scope"], "decision_at": cutoff, "available_at": t["generated_at"],
                "actual_one_word_daily": None, "groups": {}})
    result = {"version": VERSION, "formed_at": now(), "rows": rows,
        "calendar_path": str(Path(calendar_path).resolve()), "calendar_sha256": sha(calendar_path),
        "source_bindings": sources, "adapter": {"kind": "SAVED_ARCHIVE", "features": str(Path(features_dir).resolve()),
            "theme": str(Path(theme_path).resolve()) if theme_path else None},
        "original_pool_counts": {"closing": len(roster), "relay": len(price), "failed": len(rows)-len(roster)},
        "official_full_market": False}
    if output:
        write_new(output, result)
    return result


def closing_price_facts(roster, original_relay):
    """Pure signal fields only; leave all original >=2-board features untouched."""
    if not roster or len({r["code"] for r in roster}) != len(roster):
        raise ValueError("complete nonduplicated closing quote roster required")
    heights = [r["boards"] for r in roster]
    if any(type(h) is not int or h < 1 for h in heights):
        raise ValueError("closing board count must be a positive integer")
    top, counts = max(heights), Counter(heights)
    result = {}
    for r in roster:
        code = r["code"]
        if code in original_relay:
            result[code] = {k: original_relay[code]["features"].get(k) for k in RELAY_FIELDS}
            continue
        fields = {k: None for k in RELAY_FIELDS}
        cents, amount, turnover = [number(r.get(k)) for k in ("price_cents", "amount", "turnover_pct")]
        fields.update(boards=r["boards"], height_gap=top-r["boards"], same_height_rivals=counts[r["boards"]]-1,
            unique_space=float(r["boards"] == top and counts[r["boards"]] == 1), height_share=r["boards"]/top,
            is_space_leader=float(r["boards"] == top), price=cents/100 if cents is not None and cents > 0 else None,
            signal_amount=amount if amount is not None and amount >= 0 else None,
            turnover=turnover/100 if turnover is not None and turnover >= 0 else None,
            log_signal_amount=math.log1p(amount) if amount is not None and amount >= 0 else None)
        result[code] = fields
    for r in roster:
        if r["code"] in original_relay:
            continue
        group = [other["code"] for other in roster if other["boards"] == r["boards"]]
        for field in ("price", "signal_amount", "turnover"):
            own = result[r["code"]][field]
            vals = [result[c][field] for c in group]
            if own is not None and all(v is not None for v in vals):
                # Same fixed average-tie percentile convention as original pandas rank.
                below, equal = sum(v < own for v in vals), sum(v == own for v in vals)
                result[r["code"]][field+"_same_height_percentile"] = (below+(equal+1)/2)/len(vals)
    return result


def from_source_roster(manifest_path, calendar_path, output=None, formed_at=None):
    """Reuse v75's reviewed rosters and saved full pool; diagnostic snapshots."""
    import pandas as pd
    from build_source_bound_theme_roles_v75 import build
    manifest = read(manifest_path)
    if manifest.get("version") != "v75-source-bound-theme-roles-1":
        raise ValueError("supported source-bound v75 manifest required")
    pool_path, evidence_path = manifest["source_pool"], manifest["evidence_file"]
    roles_path = Path(manifest_path).with_name("candidate_theme_roles.csv")
    sources = {pool_path: manifest["source_pool_sha256"], evidence_path: manifest["evidence_sha256"],
               str(roles_path.resolve()): manifest["output_sha256"]}
    verify(sources)
    sources |= bind(manifest_path) | bind(calendar_path) | bind(Path(__file__).with_name("build_source_bound_theme_roles_v75.py"))
    days = calendar(calendar_path); signal = manifest["signal_date"]
    if signal not in days:
        raise ValueError("source roster signal outside official calendar")
    frame = pd.read_csv(pool_path, dtype={"code": str}, keep_default_na=False)
    if (set(frame.signal_date) != {signal} or frame.code.duplicated().any()
            or not set(frame.pool_assignment).issubset({"CLOSING_LIMIT_UP", "FAILED_BOARD_OBSERVATION"})):
        raise ValueError("source pool has wrong dates, duplicated codes or unknown pool kind")
    evidence = read(evidence_path)
    if evidence.get("signal_date") != signal:
        raise ValueError("theme evidence signal mismatch")
    # Preserve the original adapter's numeric clock parsing when reproducing
    # its receipt; identity tags themselves are never imported as truth.
    rebuilt = build(pd.read_csv(pool_path, dtype={"code": str}), evidence, manifest["asof"])
    archived = pd.read_csv(roles_path, dtype={"code": str}, keep_default_na=False)
    pd.testing.assert_frame_equal(archived.reset_index(drop=True), rebuilt.reset_index(drop=True), check_dtype=False)
    moment = formed_at or now()
    if stamp(moment) > stamp(now()) or stamp(manifest["asof"]) > stamp(moment):
        raise ValueError("source roster formation time inconsistent")
    bycode, closing, prices = {}, [], {}
    for raw in frame.to_dict("records"):
        code = raw["code"]
        if len(code) != 6 or not code.isdigit() or raw["snapshot_kind"] != "closing":
            raise ValueError("source code or snapshot invalid")
        observed, available = stamp(raw["observed_at"]), stamp(raw["available_at"])
        if observed.date().isoformat() != signal or observed > available or available > stamp(moment):
            raise ValueError("source pool observation/acquisition time invalid")
        is_closed = raw["pool_assignment"] == "CLOSING_LIMIT_UP"
        board = number(raw["boards"])
        if board is None or not board.is_integer() or board < 0 or is_closed and board < 1:
            raise ValueError("source closing board number invalid")
        if not is_closed and (board != 0 or raw["limit_up_status"] != "FAILED_TO_CLOSE_AT_LIMIT"):
            raise ValueError("failed-board attempt height cannot replace closing height")
        if is_closed and raw["limit_up_status"] != "VERIFIED_TWO_SOURCE_CLOSE":
            raise ValueError("closing identity not independently verified in source pool")
        raw["boards"] = int(board); bycode[code] = raw
        if is_closed:
            closing.append({"code": code, "boards": int(board), "price_cents": number(raw["last_price"])*100
                if number(raw["last_price"]) is not None else None, "amount": raw["amount"], "turnover_pct": raw["turnover"]})
    prices = closing_price_facts(closing, {}) if closing else {}
    for code, raw in bycode.items():
        if code not in prices:
            value = {key: None for key in RELAY_FIELDS}
            price, amount, turnover = (number(raw.get(k)) for k in ("last_price", "amount", "turnover"))
            value.update(boards=0, price=price if price is not None and price > 0 else None,
                signal_amount=amount if amount is not None and amount >= 0 else None,
                turnover=turnover/100 if turnover is not None and turnover >= 0 else None)
            prices[code] = value
    codes = sorted(bycode)
    pool = {"complete_within_scope": True, "codes": codes, "boards": {c: bycode[c]["boards"] for c in codes},
            "amounts": {c: prices[c]["signal_amount"] for c in codes}, "turnovers": {c: prices[c]["turnover"] for c in codes}}
    themes = {}
    for group in evidence["groups"]:
        members = {}
        for code in sorted(group["codes"]):
            raw = bycode.get(code, {})
            members[code] = {"amount": number(raw.get("amount")), "return_pct": number(raw.get("change_pct")),
                             "boards": raw.get("boards"), "pool_kind": raw.get("pool_assignment", "NOT_IN_ARCHIVED_POOL_UNKNOWN")}
        value = {"membership_verified": True, "complete_within_scope": True, "granularity": "specific",
            "theme_id": group["theme_group_id"], "codes": sorted(members), "members": members,
            "scope": "SAVED_EXPLICIT_NEWS_ROSTER_NOT_COMPLETE_THEME_OR_MARKET", "full_theme_complete": False,
            "source_url": group["url"], "source_published_at": group["published_at"],
            "comparison_scope_id": group["theme_group_id"]+":"+",".join(sorted(members))}
        for code in members:
            if code in bycode:
                themes[code] = value
    rows = []
    cutoff = signal+"T16:30:00+08:00"
    for code, raw in sorted(bycode.items()):
        values = {"pool": pool, "price": prices[code]}
        if code in themes:
            values["theme"] = themes[code]
        groups = {kind: {"value": value, "status": "INFERRED", "observed_at": raw["observed_at"],
            "available_at": moment, "original_source_available_at": raw["available_at"],
            "reconstructed": True, "adapter": "SOURCE_POOL_ALLOWLIST"} for kind, value in values.items()}
        rows.append({"candidate_id": signal+":"+code, "signal_date": signal, "code": code,
            "name": raw["name"], "boards": raw["boards"], "pool_kind": "limit_up" if raw["pool_assignment"] == "CLOSING_LIMIT_UP" else "failed_board",
            "scope_id": "COMPLETE_SAVED_CLOSING_AND_FAILED_ROSTER_NOT_FULL_MARKET", "decision_at": cutoff,
            "available_at": moment, "actual_one_word_daily": None, "groups": groups})
    result = {"version": VERSION, "formed_at": moment, "rows": rows, "source_bindings": sources,
        "calendar_path": str(Path(calendar_path).resolve()), "calendar_sha256": sha(calendar_path),
        "adapter": {"kind": "SOURCE_ROSTER", "manifest": str(Path(manifest_path).resolve())},
        "baseline_promotion": {}, "official_full_market": False, "historical_diagnostic_only": True}
    if output:
        write_new(output, result)
    return result


def from_history(receipt_path, base_root, calendar_path, output=None):
    """Import existing prepared history and saved BASE, without refitting it."""
    import joblib
    import pandas as pd
    receipt = read(receipt_path)
    sources = receipt["sources"] | {receipt["prepared_path"]: receipt["prepared_sha256"]}
    verify(sources)
    prepared = joblib.load(receipt["prepared_path"])
    original = prepared["relay"]
    bpath = Path(base_root)/"relay_history/predictions.csv"
    base = pd.read_csv(bpath, dtype={"code": str}, low_memory=False)
    if "arm" not in base:
        raise ValueError("saved BASE arm identity required")
    base = base[base.arm.eq("BASE")].copy()
    sources |= bind(receipt_path) | bind(bpath) | bind(calendar_path)
    days = calendar(calendar_path)
    captured = receipt.get("formed_at", receipt.get("recorded_at", now()))
    rows = []
    for signal, cohort in original.groupby("date", sort=True):
        signal = str(signal)
        if signal not in days:
            raise ValueError("history date outside official calendar")
        codes = sorted(cohort.code.astype(str))
        if len(codes) != len(set(codes)):
            raise ValueError("history duplicates candidate")
        pool = {"complete_within_scope": True, "codes": codes,
            "boards": {str(r.code): int(r.boards) for r in cohort.itertuples()},
            "amounts": {str(r.code): clean(r.signal_amount) for r in cohort.itertuples()},
            "turnovers": {str(r.code): clean(r.turnover) for r in cohort.itertuples()}}
        for _, r in cohort.iterrows():
            code = str(r.code).zfill(6)
            row = {"candidate_id": signal+":"+code, "signal_date": signal, "code": code,
                "name": r.get("name"), "boards": int(r.boards), "decision_at": signal+"T16:30:00+08:00",
                "available_at": captured, "scope_id": "ORIGINAL_ARCHIVED_RELAY_COHORT_NOT_FULL_MARKET",
                "actual_one_word_daily": clean(r.get("actual_one_word_daily")), "groups": {}}
            for kind, value in (("pool", pool), ("price", {k: clean(r.get(k)) for k in RELAY_FIELDS})):
                row["groups"][kind] = {"value": value, "observed_at": signal+"T15:00:00+08:00",
                    "available_at": captured, "status": "INFERRED", "reconstructed": True,
                    "adapter": "SAVED_BASE_ALLOWLIST"}
            rows.append(row)
    saved = {}
    for _, r in base.iterrows():
        day = str(r.get("signal_date", r.get("date")))
        code = str(r.code).zfill(6)
        key = day+":"+code
        if key in saved:
            raise ValueError("duplicate saved BASE prediction")
        call = r.get("predicted", r.get("prediction"))
        call = bool(int(call)) if number(call) in (0., 1.) else None
        truth = r.get("target")
        saved[key] = {"call": call, "raw_score": clean(r.get("raw_score")),
            "truth": bool(int(truth)) if number(truth) in (0., 1.) else None,
            "origin": "HISTORICAL_DIAGNOSTIC", "available_at": captured}
    result = {"version": VERSION, "formed_at": now(), "rows": rows,
        "baseline_promotion": saved, "calendar_path": str(Path(calendar_path).resolve()),
        "calendar_sha256": sha(calendar_path), "source_bindings": sources,
        "adapter": {"kind": "SAVED_HISTORY", "receipt": str(Path(receipt_path).resolve()),
                    "base_root": str(Path(base_root).resolve())}, "official_full_market": False}
    if output:
        write_new(output, result)
    return result


def load_dataset(path):
    d = read(path)
    verify(d["source_bindings"])
    if d.get("outcome_parent"):
        parent = d["outcome_parent"]
        verify({parent["path"]: parent["sha256"]})
        p = load_dataset(parent["path"])
        if any(d.get(k) != p.get(k) for k in ("rows", "adapter", "formed_at", "baseline_binding", "calendar_path", "calendar_sha256")):
            raise ValueError("outcome attachment altered original identity input")
        for ident, cell in p["baseline_promotion"].items():
            if any(d["baseline_promotion"][ident].get(k) != v for k, v in cell.items() if k != "truth"):
                raise ValueError("outcome attachment altered original BASE prediction")
        expected = p["source_bindings"] | d["promotion_outcome_binding"]["source_bindings"] | bind(parent["path"])
        if d["source_bindings"] != expected:
            raise ValueError("outcome attachment source scope changed")
        _check_promotion_truth(d)
        return d
    if d.get("adapter", {}).get("kind") == "MULTI_ARCHIVE":
        rebuilt = combine(d["adapter"]["datasets"])
        if any(d.get(k) != rebuilt.get(k) for k in ("rows", "source_bindings", "baseline_promotion", "calendar_path", "calendar_sha256")):
            raise ValueError("combined archive projection changed")
        return d
    if d.get("adapter", {}).get("kind") == "AUGMENTED":
        a = d["adapter"]
        verify({a["parent_path"]: a["parent_sha256"]})
        rebuilt = _augment(load_dataset(a["parent_path"]), a.get("evidence_path"), a.get("baseline_path"),
                           a.get("compatibility_path"), d["formed_at"], a.get("decision_at"))
        rebuilt["source_bindings"] |= bind(a["parent_path"])
        if any(rebuilt[k] != d[k] for k in ("rows", "source_bindings", "baseline_promotion", "baseline_binding")):
            raise ValueError("augmented evidence/baseline projection changed")
    elif d.get("adapter", {}).get("kind") == "SAVED_ARCHIVE":
        a = d["adapter"]
        rebuilt = from_archive(a["features"], d["calendar_path"], a.get("theme"))
        if rebuilt["rows"] != d["rows"] or rebuilt["source_bindings"] != d["source_bindings"]:
            raise ValueError("saved adapter projection changed")
    elif d.get("adapter", {}).get("kind") == "SAVED_HISTORY":
        a = d["adapter"]
        rebuilt = from_history(a["receipt"], a["base_root"], d["calendar_path"])
        if any(rebuilt[k] != d[k] for k in ("rows", "source_bindings", "baseline_promotion")):
            raise ValueError("history adapter projection changed")
    elif d.get("adapter", {}).get("kind") == "SOURCE_ROSTER":
        rebuilt = from_source_roster(d["adapter"]["manifest"], d["calendar_path"], formed_at=d["formed_at"])
        if any(rebuilt[k] != d[k] for k in ("rows", "source_bindings", "baseline_promotion", "calendar_path", "calendar_sha256")):
            raise ValueError("saved source roster projection changed")
    elif any(c.get("adapter") in ("SAVED_BASE_ALLOWLIST", "SOURCE_POOL_ALLOWLIST") for r in d["rows"] for c in r.get("groups", {}).values()):
        raise ValueError("caller cannot claim the trusted saved adapter")
    if d.get("promotion_outcome_binding"):
        _check_promotion_truth(d)
    return d


def combine(dataset_paths, output=None):
    data = [load_dataset(path) for path in dataset_paths]
    if not data or any((d["calendar_path"], d["calendar_sha256"]) != (data[0]["calendar_path"], data[0]["calendar_sha256"]) for d in data):
        raise ValueError("combined archives require same official calendar binding")
    if any(d.get("adapter", {}).get("kind") not in ("SAVED_ARCHIVE", "SAVED_HISTORY", "AUGMENTED", "SOURCE_ROSTER") for d in data):
        raise ValueError("combine accepts individually source-rebuilt archives only")
    rows = [r for d in data for r in d["rows"]]
    if len({r["candidate_id"] for r in rows}) != len(rows):
        raise ValueError("combined archives duplicate candidate/date")
    sources, base = {}, {}
    for path, d in zip(dataset_paths, data):
        sources |= d["source_bindings"] | bind(path)
        if set(base) & set(d.get("baseline_promotion", {})):
            raise ValueError("combined archives duplicate BASE predictions")
        base.update(d.get("baseline_promotion", {}))
    result = {"version": VERSION, "formed_at": now(), "rows": sorted(rows, key=lambda r: r["candidate_id"]),
        "source_bindings": sources, "baseline_promotion": base, "calendar_path": data[0]["calendar_path"],
        "calendar_sha256": data[0]["calendar_sha256"], "adapter": {"kind": "MULTI_ARCHIVE", "datasets": [str(Path(p).resolve()) for p in dataset_paths]},
        "official_full_market": False}
    if output: write_new(output, result)
    return result


def _check_promotion_truth(data):
    binding = data["promotion_outcome_binding"]
    from model_upgrade_compat import runtime
    base = data["baseline_binding"]
    inputs, _ = runtime(base["compatibility_path"])
    e_manifest = read(Path(binding["archive_path"])/"manifest.json")
    verify({str(Path(binding["archive_path"])/"manifest.json"): binding["manifest_sha256"]})
    outcome = inputs.build_outcomes({"feature_archive": base["feature_archive"], "knowledge_cutoff": data["formed_at"],
        "signal_date": base["signal_date"], "target_date": base["target_date"]}, binding["archive_path"])
    if outcome["sources"] != binding["source_bindings"]:
        raise ValueError("promotion training truth source changed")
    for code, truth in outcome["candidates"].items():
        cell = data["baseline_promotion"][base["signal_date"]+":"+code]
        if cell.get("truth") != truth or cell.get("truth_available_at") != e_manifest["recorded_at"]:
            raise ValueError("promotion training truth/time changed")


def attach_promotion_truth(dataset_path, archive, output):
    data = load_dataset(dataset_path)
    if not data.get("baseline_binding"):
        raise ValueError("source-bound baseline required before outcome attachment")
    from model_upgrade_compat import runtime
    base = data["baseline_binding"]
    inputs, _ = runtime(base["compatibility_path"])
    outcome = inputs.build_outcomes({"feature_archive": base["feature_archive"], "knowledge_cutoff": data["formed_at"],
        "signal_date": base["signal_date"], "target_date": base["target_date"]}, archive)
    manifest = read(Path(archive)/"manifest.json")
    if stamp(manifest["recorded_at"]) <= stamp(base["target_date"]+"T15:00:00+08:00"):
        raise ValueError("promotion training label must follow exact target close")
    for code, truth in outcome["candidates"].items():
        data["baseline_promotion"][base["signal_date"]+":"+code].update(truth=truth, truth_available_at=manifest["recorded_at"])
    data["promotion_outcome_binding"] = {"archive_path": str(Path(archive).resolve()),
        "manifest_sha256": sha(Path(archive)/"manifest.json"), "source_bindings": outcome["sources"]}
    # Retain original creation time/adapter projection; the new outcome availability is separate.
    data["outcomes_attached_at"] = now()
    data["source_bindings"] |= outcome["sources"] | bind(dataset_path)
    data["outcome_parent"] = {"path": str(Path(dataset_path).resolve()), "sha256": sha(dataset_path)}
    write_new(output, data)
    return data


def original_archive(dataset):
    a = dataset.get("adapter", {})
    if a.get("kind") == "AUGMENTED":
        verify({a["parent_path"]: a["parent_sha256"]})
        return original_archive(load_dataset(a["parent_path"]))
    if a.get("kind") != "SAVED_ARCHIVE":
        raise ValueError("source-bound original current feature archive required")
    return a["features"]


def _augment(parent, evidence_path, baseline_path, compatibility_path, formed, decision_at=None):
    import copy
    result = copy.deepcopy(parent)
    sources = result["source_bindings"].copy()
    index = {r["candidate_id"]: r for r in result["rows"]}
    if evidence_path:
        bank = read(evidence_path)
        if bank.get("version") != VERSION or stamp(bank["formed_at"]) > stamp(formed):
            raise ValueError("rich evidence bank version/formation mismatch")
        seen = set()
        for row in bank["rows"]:
            key = row["candidate_id"]
            if key not in index or key in seen or set(row["groups"]) - (GROUPS - {"pool", "price"}):
                raise ValueError("rich evidence changes original roster or protected price/pool")
            seen.add(key)
            for kind, cell in row["groups"].items():
                if cell.get("adapter"):
                    raise ValueError("rich evidence cannot claim a trusted adapter")
                if kind in index[key]["groups"]:
                    raise ValueError("append-only evidence cannot replace existing group")
                sources.update({cell["source_path"]: cell["source_sha256"]})
            index[key]["groups"].update(row["groups"])
        sources.update(bind(evidence_path))
    for row in result["rows"]:
        # Original knowledge cutoff stays in its own field; this is a new observation.
        row.setdefault("original_decision_at", row["decision_at"])
        row["decision_at"] = decision_at or formed
        row["available_at"] = max([formed, row["available_at"], *[c["available_at"] for c in row["groups"].values()]], key=stamp)
        if stamp(row["decision_at"]) > stamp(now()):
            raise ValueError("augmented decision cutoff is in future")
    result["formed_at"] = formed
    result["source_bindings"] = sources
    result["baseline_promotion"] = copy.deepcopy(parent.get("baseline_promotion", {}))
    result["baseline_binding"] = None
    if baseline_path:
        if not compatibility_path:
            raise ValueError("original compatibility policy required for current BASE")
        from model_upgrade_compat import runtime
        import stock_judgement_review as original
        inputs, _ = runtime(compatibility_path)
        archive = original_archive(parent)
        data = inputs.load_feature_archive(archive, formed)
        b = read(baseline_path)
        if any(b[k] != data[k] for k in ("signal_date", "target_date")) or b["feature_sha256"] != sha(Path(archive)/"features.json"):
            raise ValueError("BASE changed date or original feature source")
        verify(b["source_hashes"])
        original.pre_auction(b["signal_date"], b["target_date"], data["calendar"]["sessions"], stamp(b["generated_at"]))
        if stamp(b["knowledge_cutoff"]) > stamp(b["generated_at"]) or stamp(b["generated_at"]) > stamp(formed):
            raise ValueError("BASE availability is inconsistent")
        expected = {r["code"] for r in data["relay"]}
        bs = original.unique_rows(b["relay"]["candidates"])
        if set(bs) != expected:
            raise ValueError("BASE omits original relay candidates")
        base_rows = {}
        for code, r in bs.items():
            if type(r["predicted_positive"]) is not bool or number(r.get("raw_score")) is None or not 0 <= r["raw_score"] <= 1:
                raise ValueError("original BASE boolean and raw score required")
            base_rows[b["signal_date"]+":"+code] = {"call": r["predicted_positive"], "raw_score": r["raw_score"],
                "truth": None, "origin": "SOURCE_BOUND_PREAUCTION", "available_at": b["generated_at"]}
        result["baseline_promotion"] = base_rows
        result["baseline_binding"] = {"baseline_path": str(Path(baseline_path).resolve()), "baseline_sha256": sha(baseline_path),
            "feature_archive": archive, "compatibility_path": str(Path(compatibility_path).resolve()),
            "signal_date": data["signal_date"], "target_date": data["target_date"], "original_codes": sorted(expected)}
        result["source_bindings"] |= b["source_hashes"] | bind(baseline_path) | bind(compatibility_path)
    verify(result["source_bindings"])
    # Validate all rich pointers/dates now, retaining incomplete groups as unknown.
    for row in result["rows"]:
        for kind in row["groups"]:
            _group(row, kind, "DIAGNOSTIC")
    return result


def augment(parent_path, output, evidence_path=None, baseline_path=None, compatibility_path=None, decision_at=None):
    parent = load_dataset(parent_path)
    formed = now()
    result = _augment(parent, evidence_path, baseline_path, compatibility_path, formed, decision_at)
    result["source_bindings"] |= bind(parent_path)
    result["adapter"] = {"kind": "AUGMENTED", "parent_path": str(Path(parent_path).resolve()), "parent_sha256": sha(parent_path),
        "evidence_path": str(Path(evidence_path).resolve()) if evidence_path else None,
        "baseline_path": str(Path(baseline_path).resolve()) if baseline_path else None,
        "compatibility_path": str(Path(compatibility_path).resolve()) if compatibility_path else None, "decision_at": decision_at}
    write_new(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("build")
    for key in ("features", "calendar", "output"):
        a.add_argument("--"+key, required=True)
    a.add_argument("--theme-report")
    a = sub.add_parser("source-roster")
    for key in ("manifest", "calendar", "output"):
        a.add_argument("--"+key, required=True)
    a = sub.add_parser("augment")
    a.add_argument("--dataset", required=True); a.add_argument("--output", required=True)
    a.add_argument("--evidence"); a.add_argument("--baseline"); a.add_argument("--compatibility"); a.add_argument("--decision-at")
    a = sub.add_parser("attach-outcomes")
    for k in ("dataset", "outcome-archive", "output"): a.add_argument("--"+k, required=True)
    a = sub.add_parser("combine"); a.add_argument("--datasets", nargs="+", required=True); a.add_argument("--output", required=True)
    a = sub.add_parser("analyse")
    a.add_argument("--dataset", required=True); a.add_argument("--track", choices=("DIAGNOSTIC", "STRICT"), required=True)
    a.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "build":
        r = from_archive(args.features, args.calendar, args.theme_report, args.output)
    elif args.command == "source-roster":
        r = from_source_roster(args.manifest, args.calendar, args.output)
    elif args.command == "augment":
        r = augment(args.dataset, args.output, args.evidence, args.baseline, args.compatibility, args.decision_at)
    elif args.command == "attach-outcomes":
        r = attach_promotion_truth(args.dataset, args.outcome_archive, args.output)
    elif args.command == "combine":
        r = combine(args.datasets, args.output)
    else:
        r = analyse(load_dataset(args.dataset), args.track); write_new(args.output, r)
    print(json.dumps({"version": VERSION, "rows": len(r["rows"]), "production_changed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
