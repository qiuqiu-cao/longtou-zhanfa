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
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from functools import wraps
from contextvars import ContextVar
from pathlib import Path

from current_model_diagnostics import RELAY_FIELDS

VERSION = "leader-roles-1"
_verification_scope = ContextVar("leader_role_verification_scope", default=None)
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
SAMPLED_PATH_FIELDS = ("sampled_limit_touch", "sampled_close_drop_after_limit",
    "sampled_later_limit_close_after_drop", "sampled_after_touch_drawdown_pct",
    "sampled_grid_complete", "sampled_observed_bars", "sampled_volume_reconciled")
PEER_PATH_FIELDS = tuple("sampled_"+scope+"_"+field for scope in ("current", "prior")
    for field in ("peer_expected", "peer_available", "peer_coverage", "relative_min_pp",
                  "relative_last_pp", "positive_bar_share", "amount_share"))
MARKET_RAW_FIELDS = ("limit_count", "consecutive_count", "max_board", "failed_count", "down_limit_count",
    "seal_rate", "failed_rate", "middle_density", "ladder_gap_count", "unique_space", "yesterday_premium",
    "previous_limit_tail_loss", "failed_tail_loss", "promotion_overall", "promotion_1to2", "promotion_2to3",
    "promotion_3to4", "promotion_4to5", "promotion_mid", "space_feedback", "previous_limit_feedback_count",
    "previous_limit_feedback_coverage")
MARKET_MODEL_FIELDS = ("limit_count", "max_board", "failed_rate", "down_limit_count", "promotion_mid", "yesterday_premium")
MARKET_CONTEXT_FIELDS = tuple("market_"+field+suffix for field in MARKET_MODEL_FIELDS
    for suffix in ("", "_delta_1", "_delta_3", "_delta_5")) + ("market_limit_count_ex_self", "market_consecutive_count_ex_self",
    "market_height_ex_self", "cycle_is_warming")
PHASE_MAP = {"启动": "START", "升温": "WARMING", "修复": "REPAIR", "加速": "ACCELERATION",
    "一致": "CONSENSUS", "大分歧": "DIVERGENCE", "退潮": "RETREAT"}
FEATURES = tuple(RELAY_FIELDS) + RELATION_FIELDS + tuple(
    f"{field}_delta_{lag}" for field in ("theme_amount_share", "theme_return_ex_self_pp", "price", "turnover")
    for lag in (1, 3, 5)) + HEIGHT_TEMPORAL_FIELDS + SAMPLED_PATH_FIELDS + PEER_PATH_FIELDS + MARKET_CONTEXT_FIELDS
GROUPS = set(g for values in REQUIRED.values() for g in values) | {"price", "sampled_path", "sampled_peers", "market_context"}
ARCHIVE_THEME_SCOPE = "SAVED_EXPLICIT_NEWS_ROSTER_NOT_COMPLETE_THEME_OR_MARKET"
# These original V75 topic definitions mix branches or business/event scopes.
# Preserve their posted associations without certifying comparable core rivals.
BROAD_ARCHIVE_THEME_IDS = frozenset(("consumer", "ai_hardware", "automotive", "medicine",
    "ai_apps", "power_grid", "semiconductor_materials", "military", "network_security",
    "domestic_chip", "robot", "robotics"))
DEFINED_ARCHIVE_THEME_IDS = frozenset(("transport.shipping", "energy.coal", "smart_grid",
    "glass_fiber", "pcb", "passive_components", "cloud_dc"))


def theme_granularity(value):
    if value.get("scope") == ARCHIVE_THEME_SCOPE:
        if value.get("theme_id") in BROAD_ARCHIVE_THEME_IDS:
            return "BROAD_PARENT_OR_MIXED"
        if value.get("theme_id") not in DEFINED_ARCHIVE_THEME_IDS:
            return "UNKNOWN_SOURCE_TOPIC_GRANULARITY"
    return value.get("granularity", "UNKNOWN")


def theme_source_definitions(sources):
    """Recover original topic wording and confidence without changing old rows."""
    definitions = {}
    for path in sources:
        if Path(path).name != "manifest.json":
            continue
        manifest = read(path)
        if manifest.get("version") != "v75-source-bound-theme-roles-1":
            continue
        raw_path, expected = manifest["evidence_file"], manifest["evidence_sha256"]
        if sources.get(raw_path) != expected:
            raise ValueError("original theme definition is not in source bindings")
        verify({raw_path: expected})
        source = read(raw_path)
        if source["signal_date"] != manifest["signal_date"]:
            raise ValueError("original theme definition date mismatch")
        for group in source["groups"]:
            key = (source["signal_date"], group["theme_group_id"])
            definition = {"raw_topic": group["theme"], "source_tier": group.get("source_tier"),
                "source_url": group["url"], "published_at": group["published_at"],
                "codes": sorted(group["codes"]), "source_path": raw_path, "source_sha256": expected}
            if key in definitions and definitions[key] != definition:
                raise ValueError("conflicting original theme definitions")
            definitions[key] = definition
    return definitions


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
    snapshot = _verification_scope.get()
    for path, expected in sources.items():
        # Repeated evidence cells use the same bound path spelling. Resolving
        # every duplicate on Windows would repeat filesystem work thousands of
        # times; alternate spellings are independently hashed and rechecked.
        key = str(path)
        if snapshot is not None and key in snapshot:
            if snapshot[key] != expected:
                raise ValueError("inconsistent source hashes within one operation: " + path)
            continue
        if not Path(path).is_file() or sha(path) != expected:
            raise ValueError("source hash mismatch: " + path)
        if snapshot is not None: snapshot[key] = expected


def verified_session(function):
    """Deduplicate immutable source checks within one operation, then rehash all.

    The final independent pass detects changes even when file timestamps/size
    are restored. Nothing is cached across CLI operations or later predictions.
    """
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _verification_scope.get() is not None:
            return function(*args, **kwargs)
        bindings = {}; token = _verification_scope.set(bindings)
        try:
            return function(*args, **kwargs)
        finally:
            _verification_scope.reset(token)
            for path, expected in bindings.items():
                if not Path(path).is_file() or sha(path) != expected:
                    raise ValueError("source changed during verified operation: " + path)
    return wrapped


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


@lru_cache(maxsize=8)
def _rich_source_json(path, expected_hash):
    """Private read-only source JSON; caller rechecks file and upstream hashes."""
    verify({path: expected_hash})
    source = read(path)
    kind = source.get("derivation_kind")
    specifications = {
        "HISTORICAL_RELATION_FACTS": (_replay_relation_bundle,("groups","return_diagnostics","source_bindings")),
        "HISTORICAL_SAMPLED_PRICE_PATH": (_replay_sampled_bundle,("groups","source_bindings","coverage")),
        "HISTORICAL_PEER_PRICE_PATH": (_replay_peer_bundle,("groups","paths","source_bindings","coverage")),
        "HISTORICAL_MARKET_RAW_CONTEXT": (_replay_market_bundle,("groups","market_records","source_bindings","coverage"))}
    if kind in specifications:
        if kind=="HISTORICAL_SAMPLED_PRICE_PATH" and source.get("capture_metadata_bound") is not True:
            raise ValueError("sampled capture metadata binding required")
        replay,fields=specifications[kind];rebuilt=replay(path,expected_hash)
        if any(source.get(k)!=rebuilt.get(k) for k in fields):
            phrase="original source replay" if kind=="HISTORICAL_RELATION_FACTS" else "original response replay"
            raise ValueError("derived facts differ from "+phrase+": "+kind)
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
        source = _rich_source_json(str(Path(path).resolve()), cell["source_sha256"])
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
                and theme_granularity(v) == "specific" and bool(v.get("theme_id"))
                and row["code"] in codes and len(codes) == len(set(codes)) and len(codes) >= 2
                and set(v.get("members", {})) == set(codes))
    if kind == "path":
        events = v.get("events", [])
        return (v.get("event_sequence_complete") is True and len(events) >= 2
                and all(e.get("state") in ("OPEN", "SEALED") and e.get("observed_at") for e in events)
                and all(stamp(e["observed_at"]).date().isoformat() == row["signal_date"] for e in events)
                and all(stamp(a["observed_at"]) < stamp(b["observed_at"]) for a, b in zip(events, events[1:])))
    if kind == "sampled_path":
        return (v.get("price_reconciled") is True and v.get("interval_minutes") == 5
                and v.get("tick_path_verified") is False and v.get("auction_verified") is False
                and v.get("order_book_seal_verified") is False and v.get("absorption_verified") is False)
    if kind == "sampled_peers":
        return (v.get("signal_date") == row["signal_date"] and v.get("code") == row["code"]
                and v.get("scope") == "DECLARED_ARCHIVE_ONLY_NOT_ALL_MARKET_COMPETITORS"
                and v.get("identities_certified") is False)
    if kind == "market_context":
        records = v.get("records", [])
        return (len(records) >= 1 and records[-1].get("date") == row["signal_date"]
                and v.get("scope") == "ORIGINAL_ARCHIVE_NOT_EXCHANGE_FULL_MARKET_CERTIFIED"
                and all(set(r.get("raw", {})) == set(MARKET_RAW_FIELDS) for r in records))
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
                and all(r.get("state") in tuple(PHASE_MAP.values()) for r in records))
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


def evidence_row(row, track, days, theme_definitions=None):
    values, missing = {}, {}
    observed_theme = None
    for group in sorted(GROUPS):
        value, reason = _group(row, group, track)
        if group == "theme":
            observed_theme = value
        if value is not None and not _valid_group(group, value, row):
            granularity = theme_granularity(value) if group == "theme" else None
            reason = ("BROAD_THEME_COMPARISON_SCOPE" if granularity == "BROAD_PARENT_OR_MIXED" else
                      "UNKNOWN_SOURCE_TOPIC_GRANULARITY" if granularity == "UNKNOWN_SOURCE_TOPIC_GRANULARITY" else
                      "INCOMPLETE_" + group.upper())
            value = None
        # Histories must use consecutive official sessions, not sparse pool dates.
        if group == "history" and value:
            expected = days[max(0, days.index(row["signal_date"])-len(value["bars"])+1):days.index(row["signal_date"])+1]
            if [b["date"] for b in value["bars"]] != expected:
                reason, value = "NONCONSECUTIVE_HISTORY", None
        if group == "cycle" and value:
            pos = days.index(row["signal_date"])
            expected = days[max(0, pos-len(value["records"])+1):pos+1]
            if [r["date"] for r in value["records"]] != expected:
                reason, value = "NONCONSECUTIVE_OR_STALE_CYCLE", None
        values[group], missing[group] = value, reason
    features = {k: None for k in FEATURES}
    price = values.get("price") or {}
    features.update({k: number(price.get(k)) for k in RELAY_FIELDS})
    pool, theme, path = (values[k] for k in ("pool", "theme", "path"))
    peers = []
    relations = []
    theme_scope = None
    if observed_theme:
        definition = (theme_definitions or {}).get((row["signal_date"], observed_theme.get("theme_id")))
        if definition and (definition["source_url"] != observed_theme.get("source_url") or
                           definition["published_at"] != observed_theme.get("source_published_at") or
                           definition["codes"] != sorted(observed_theme["codes"])):
            raise ValueError("theme facts differ from original definition")
        theme_scope = {"effective_granularity": theme_granularity(observed_theme),
            "original_claimed_granularity": observed_theme.get("granularity"),
            "original_definition": definition, "scope": observed_theme.get("scope"),
            "full_theme_complete": observed_theme.get("full_theme_complete", False),
            "observed_posted_association_only": observed_theme.get("scope") == ARCHIVE_THEME_SCOPE,
            "business_membership_certified": False, "fund_driver_certified": False}
        if not theme:
            relations.append({"kind": "OBSERVED_THEME_ASSOCIATION", "to": observed_theme["theme_id"],
                "peer_codes": sorted(c for c in observed_theme["codes"] if c != row["code"]),
                "comparable_core_competitors_certified": False})
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
    if values["sampled_path"]:
        features.update({k: number(values["sampled_path"]["features"].get(k)) for k in SAMPLED_PATH_FIELDS})
    if values["sampled_peers"]:
        for scope in ("current", "prior"):
            features.update({"sampled_"+scope+"_"+k: number(v)
                for k, v in values["sampled_peers"][scope]["features"].items()})
    if values["market_context"]:
        features.update({k: number(v) for k, v in values["market_context"]["features"].items()})
        features.update(market_ex_self(values["market_context"]["records"][-1], pool, row["code"])["features"])
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
        features["cycle_is_warming"] = float(records[-1]["state"] == "WARMING")
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
            "counterevidence": (["BROAD_THEME_COMPARISON_SCOPE"] if "theme" in REQUIRED[role] and
                observed_theme and theme_granularity(observed_theme) == "BROAD_PARENT_OR_MIXED" else []),
            "invalidation": "Recheck required evidence, peer competition and hard risks at next actual observation"}
    return {"candidate_id": row["candidate_id"], "signal_date": row["signal_date"], "code": row["code"],
        "name": row.get("name"), "boards": row["boards"], "decision_at": row["decision_at"],
        "available_at": row["available_at"], "scope_id": row["scope_id"],
        "pool_kind": row.get("pool_kind", "limit_up"),
        "actual_one_word_daily": row.get("actual_one_word_daily"),
        "low_turnover_proxy": price.get("signal_one_word"),
        "features": features, "role_eligible": eligible, "roles": roles, "relations": relations,
        "theme_scope_assessment": theme_scope,
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
    definitions = theme_source_definitions(dataset["source_bindings"])
    output = [evidence_row(r, track, days, definitions) for r in rows]
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
    if d.get("adapter", {}).get("kind") == "COMPLETE_DATE_SLICE":
        a = d["adapter"]
        verify({a["parent_path"]: a["parent_sha256"]})
        rebuilt = slice_dates(a["parent_path"], a["start"], a["end"])
        if any(d.get(k) != rebuilt.get(k) for k in
               ("rows", "source_bindings", "baseline_promotion", "calendar_path", "calendar_sha256", "adapter")):
            raise ValueError("complete-date slice changed original candidates or facts")
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


def slice_dates(parent_path, start, end, output=None):
    """Select complete archived dates, never filter stocks by identity or outcome."""
    parent = load_dataset(parent_path)
    days = calendar(parent["calendar_path"], parent["calendar_sha256"])
    if start not in days or end not in days or start > end:
        raise ValueError("complete-date slice needs ordered official endpoints")
    rows = [row for row in parent["rows"] if start <= row["signal_date"] <= end]
    if not rows:
        raise ValueError("complete-date slice is empty")
    ids = {row["candidate_id"] for row in rows}
    result = {"version": VERSION, "formed_at": now(), "rows": rows,
        "source_bindings": parent["source_bindings"] | bind(parent_path),
        "baseline_promotion": {key: value for key, value in parent.get("baseline_promotion", {}).items() if key in ids},
        "calendar_path": parent["calendar_path"], "calendar_sha256": parent["calendar_sha256"],
        "adapter": {"kind": "COMPLETE_DATE_SLICE", "parent_path": str(Path(parent_path).resolve()),
                    "parent_sha256": sha(parent_path), "start": start, "end": end}, "official_full_market": False}
    if output:
        write_new(output, result)
    return result


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


def unadjusted_quote_change(code, signal, quote, bars, days):
    """Exact cent-matched close versus previous official-session raw close."""
    result = {"code": code, "signal_date": signal, "value": None, "status": "UNKNOWN"}
    if signal not in days or days.index(signal) == 0:
        return result | {"reason": "MISSING_PREVIOUS_OFFICIAL_SESSION"}
    previous = days[days.index(signal)-1]
    matching = [bar for bar in bars if bar["date"] in (previous, signal)]
    selected = {bar["date"]: bar for bar in matching}
    if len(selected) != len(matching):
        return result | {"reason": "DUPLICATE_CURRENT_OR_PREVIOUS_BAR"}
    if len(selected) != 2:
        return result | {"reason": "MISSING_EXACT_CURRENT_OR_PREVIOUS_BAR"}
    try:
        values = [Decimal(str(value)) for value in (quote, selected[signal]["close"], selected[previous]["close"])]
    except (InvalidOperation, ValueError):
        return result | {"reason": "UNAVAILABLE_QUOTE"}
    if any(not value.is_finite() or value <= 0 or value*100 != (value*100).to_integral_value() for value in values):
        return result | {"reason": "INVALID_OR_NON_CENT_QUOTE"}
    if values[0] != values[1]:
        return result | {"reason": "SIGNAL_CLOSE_MISMATCH"}
    return result | {"value": float((values[1]/values[2]-1)*100), "status": "INFERRED",
        "previous_session": previous, "reason": "EXACT_UNADJUSTED_CLOSE_CHANGE_NOT_ADJUSTMENT_VERIFIED"}


def _build_relation_bundle(parent_path, relation_path, formed):
    import copy
    parent, relation = load_dataset(parent_path), load_dataset(relation_path)
    if parent["calendar_sha256"] != relation["calendar_sha256"]:
        raise ValueError("historical relation calendar differs")
    days = calendar(parent["calendar_path"], parent["calendar_sha256"])
    sources = parent["source_bindings"] | relation["source_bindings"] | bind(parent_path) | bind(relation_path)
    latest = max(row["signal_date"] for row in relation["rows"])
    candles = {}
    for raw_path in parent["source_bindings"]:
        p = Path(raw_path)
        if p.parent.name != "raw" or p.parent.parent.name != "september_unadjusted_20261002_v3":
            continue
        symbol = p.stem
        response = read(p)
        if response.get("code") != 0 or "day" not in response.get("data", {}).get(symbol, {}):
            raise ValueError("saved unadjusted raw symbol/field mismatch")
        bars = [{"date": row[0], "close": row[2]} for row in response["data"][symbol]["day"] if row[0] <= latest]
        dates = [bar["date"] for bar in bars]
        if dates != sorted(set(dates)) or any(day not in days for day in dates):
            raise ValueError("saved unadjusted raw dates duplicate or invalid")
        candles[symbol[-6:]] = bars
    indexed = {row["candidate_id"]: row for row in relation["rows"]}
    groups, diagnostics = {}, []
    for original in parent["rows"]:
        matched = indexed.get(original["candidate_id"])
        if not matched:
            continue
        added = {}
        for kind in ("theme", "competition"):
            if kind not in matched["groups"]:
                continue
            if kind in original["groups"]:
                raise ValueError("historical relation cannot overwrite existing evidence")
            value = copy.deepcopy(matched["groups"][kind]["value"])
            if kind == "theme":
                for code, member in value["members"].items():
                    member_row = indexed.get(original["signal_date"]+":"+code)
                    quote = member_row.get("groups", {}).get("price", {}).get("value", {}).get("price") if member_row else None
                    change = unadjusted_quote_change(code, original["signal_date"], quote, candles.get(code, []), days)
                    diagnostics.append(change | {"candidate_id": original["candidate_id"], "theme_id": value["theme_id"]})
                    if member.get("return_pct") is None and change["value"] is not None:
                        member["return_pct"] = change["value"]
                        member["return_basis"] = "UNADJUSTED_CLOSE_VS_PREVIOUS_OFFICIAL_SESSION_NOT_ADJUSTMENT_VERIFIED"
            added[kind] = value
        if added:
            groups[original["candidate_id"]] = added
    return {"version": VERSION, "formed_at": formed, "derivation_kind": "HISTORICAL_RELATION_FACTS",
        "parent_path": str(Path(parent_path).resolve()), "relation_path": str(Path(relation_path).resolve()),
        "source_bindings": sources, "groups": groups, "return_diagnostics": diagnostics,
        "not_full_market_or_verified_fund_causality": True, "historical_diagnostic_only": True}


@lru_cache(maxsize=4)
def _replay_relation_bundle(path, expected_hash):
    verify({path: expected_hash})
    source = read(path)
    return _build_relation_bundle(source["parent_path"], source["relation_path"], source["formed_at"])


def relate_history(parent_path, relation_path, output_dir):
    root = Path(output_dir)
    if root.exists():
        raise ValueError("historical relation output directory must be new")
    formed = now()
    bundle = _build_relation_bundle(parent_path, relation_path, formed)
    root.mkdir(parents=True)
    source_path = root/"source_rebuilt_relation_facts.json"
    write_new(source_path, bundle)
    evidence_path = root/"relation_evidence_bank.json"
    rows = [{"candidate_id": ident, "groups": {kind: {"value": value, "status": "INFERRED",
        "source_path": str(source_path.resolve()), "source_sha256": sha(source_path), "pointer": ["groups", ident, kind],
        "observed_at": ident[:10]+"T15:00:00+08:00", "available_at": formed, "reconstructed": True}
        for kind, value in groups.items()}} for ident, groups in bundle["groups"].items()]
    write_new(evidence_path, {"version": VERSION, "formed_at": formed, "rows": rows})
    result = augment(parent_path, root/"history_with_relations.json", evidence_path=evidence_path)
    return result


def _cent_price(value):
    try:
        p = Decimal(str(value))
        if not p.is_finite() or p <= 0 or p * 100 != (p * 100).to_integral_value():
            raise ValueError("positive exact-cent price required")
        return p
    except InvalidOperation as exc:
        raise ValueError("invalid cent price") from exc


def sampled_price_path(bars, check, limit_price):
    """OHLC facts at observed five-minute timestamps, never queue/seal events.

    A three-bar close sequence proves successive observed traded prices only.
    High/low order within a bar is unknown. Missing bars cannot prove absence.
    """
    quote = _cent_price(limit_price)
    ordered = sorted(bars, key=lambda b: b["minute"])
    times = [b["minute"] for b in ordered]
    if not ordered or times != sorted(set(times)):
        raise ValueError("unique sampled bars required")
    if check.get("price_reconciled") is not True or check.get("interval_minutes") != 5:
        raise ValueError("reconciled five-minute prices required")
    day = check["date"]
    if any(t[:10] != day for t in times) or len(ordered) != check["observed_bars"]:
        raise ValueError("sampled signal date/count mismatch")
    values = [{k: _cent_price(b[k]) for k in ("open", "high", "low", "close")} for b in ordered]
    if any(v["high"] > quote or v["low"] > min(v["open"], v["close"])
           or v["high"] < max(v["open"], v["close"]) for v in values):
        raise ValueError("sampled prices conflict with closing limit-up price")
    touches = [i for i, v in enumerate(values) if v["high"] == quote]
    limit_closes = [i for i, v in enumerate(values) if v["close"] == quote]
    drops = [i for i, v in enumerate(values) if v["close"] < quote and any(j < i for j in limit_closes)]
    recoveries = [i for i in limit_closes if any(j < i for j in drops)]
    complete = check["grid_complete"] is True
    observed_flag = lambda found: 1. if found else 0. if complete else None
    after_touch = values[touches[0]+1:] if touches else []
    drawdown = float((min(v["low"] for v in after_touch)/quote-1)*100) if after_touch else None
    features = dict(zip(SAMPLED_PATH_FIELDS, (
        observed_flag(touches), observed_flag(drops), observed_flag(recoveries), drawdown,
        float(complete), float(len(ordered)), float(check["volume_reconciled"]))))
    volume_ok = check["volume_reconciled"] is True and check.get("amount_volume_unit_consistent") is True
    volume = sum(number(b["volume_shares"]) or 0 for b in ordered) if volume_ok else None
    below_volume = sum(number(b["volume_shares"]) or 0 for b, v in zip(ordered, values)
                       if v["close"] < quote) if volume_ok else None
    return {"signal_date": day, "code": check["code"], "interval_minutes": 5,
        "price_reconciled": True, "grid_complete": complete, "missing_times": check["missing_times"],
        "features": features, "first_observed_touch_bar": times[touches[0]] if touches else None,
        "close_below_after_limit_bar": times[drops[0]] if drops else None,
        "later_limit_close_bar": times[recoveries[0]] if recoveries else None,
        "actual_one_word_observed_OHLC": bool(all(v[k] == quote for v in values for k in v)) if complete else None,
        "volume_reconciled": check["volume_reconciled"], "amount_volume_unit_consistent": check.get("amount_volume_unit_consistent"),
        "observed_volume_shares": volume, "volume_in_bars_closing_below_limit": below_volume,
        "volume_by_bar_close_is_not_volume_traded_at_that_price": True,
        "tick_path_verified": False, "auction_verified": False, "order_book_seal_verified": False,
        "absorption_verified": False, "actual_open_duration_minutes": None, "actual_reseal_verified": None,
        "within_bar_high_low_order": "UNKNOWN", "price_sequence_is_not_order_book_seal_sequence": True}


def validate_reported_capture(metadata, day, raw_hash, row_count):
    date = str(metadata.get("date", "")).replace("-", "")
    block = metadata.get("datasets", {}).get("limit_up", {})
    if date != day.replace("-", ""):
        raise ValueError("reported seal capture date mismatch")
    if (block.get("sha256") and block["sha256"] != raw_hash) or block.get("rows") != row_count:
        raise ValueError("reported seal manifest hash/count mismatch")
    # Older capture manifests lack a hash. The immutable v275 reconstruction
    # still binds the raw bytes, but cannot establish the original capture hash.
    return bool(block.get("sha256"))


def _build_sampled_bundle(parent_path, report_path, seal_path, formed):
    import csv
    import pandas as pd
    import public_minute_capture_v1 as parser
    import archive_september_5min_v287 as archive
    parent = load_dataset(parent_path)
    report = read(report_path)
    if (report.get("version") != "v287-full-september-five-minute-1" or report.get("interval_minutes") != 5
            or report.get("outcome_selection_used") is not False or report.get("no_missing_rows_imputed") is not True):
        raise ValueError("original complete, unselected saved five-minute archive required")
    sources = bind(parent_path) | bind(report_path) | report["source_hashes"]
    directory = Path(report_path).resolve().parent
    for name, digest in report["artifact_hashes"].items():
        target = (directory/name).resolve()
        if target.parent != directory:
            raise ValueError("sampled artifact leaves source directory")
        sources[str(target)] = digest
    audits = {a["code"]: a for a in report["audit"]}
    if len(audits) != len(report["audit"]):
        raise ValueError("duplicate raw sampled symbol")
    for audit in audits.values():
        if (audit.get("status") != "RAW_PARSED" or audit.get("interval_minutes") != 5
                or audit.get("duplicates_before_parser") != 0 or audit.get("invalid_raw_times") != 0):
            raise ValueError("unverified raw sampled response")
        if Path(audit["raw_path"]).resolve().parent != directory:
            raise ValueError("raw sampled response leaves source directory")
        if stamp(audit["available_at"]) > stamp(report["recorded_at"]) or stamp(report["recorded_at"]) > stamp(formed):
            raise ValueError("sampled archive formation mismatch")
        sources[audit["raw_path"]] = audit["raw_sha256"]
    verify(sources)
    cohort_path = next(p for p in report["source_hashes"] if Path(p).name == "september_relay_paths_LABEL_ONLY.csv")
    daily_path = next(p for p in report["source_hashes"] if Path(p).name == "unadjusted_daily.csv")
    # Only date/code are read from the label-bearing cohort. Neither model nor outcome is projected.
    with open(cohort_path, encoding="utf-8-sig", newline="") as f:
        cohort = [(r["date"], r["code"].zfill(6)) for r in csv.DictReader(f)]
    if len(cohort) != len(set(cohort)) or len(cohort) != report["candidate_rows"]:
        raise ValueError("complete sampled cohort changed")
    index = {r["candidate_id"]: r for r in parent["rows"]}
    expected = {r["candidate_id"] for r in parent["rows"] if r["signal_date"].startswith("2026-09-")}
    if {d+":"+c for d, c in cohort} != expected or set(audits) != {c for _, c in cohort}:
        raise ValueError("sampled cohort differs from complete original September candidates")
    reference = pd.read_csv(daily_path)
    checks_list = read(directory/"signal_checks.json")
    checks = {(v["date"], v["code"]): v for v in checks_list}
    if len(checks) != len(checks_list) or set(checks) != set(cohort):
        raise ValueError("sampled checks omit or duplicate original candidates")
    raw = {}
    for code, audit in audits.items():
        response = Path(audit["raw_path"]).read_text(encoding="utf-8")
        rows = parser._extract_json_array(response)
        if len(rows) != audit["raw_rows"] or len({r["day"] for r in rows}) != len(rows):
            raise ValueError("raw sampled duplicate/count mismatch")
        raw[code] = parser.parse_minute_response(response, code)
        if len(raw[code]) != audit["parsed_rows"]:
            raise ValueError("raw sampled parser count mismatch")
    seals = {}
    raw_seals = {}
    if seal_path:
        sources |= bind(seal_path)
        with open(seal_path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                key = r["date"]+":"+r["code"].zfill(6)
                if key not in expected:
                    continue
                if key in seals:
                    raise ValueError("duplicate reported seal candidate")
                verify({r["source_path"]: r["source_sha256"]})
                sources[r["source_path"]] = r["source_sha256"]
                if r["source_path"] not in raw_seals:
                    with open(r["source_path"], encoding="utf-8-sig", newline="") as original:
                        records = list(csv.DictReader(original))
                    codes = [v["代码"].zfill(6) for v in records]
                    if len(codes) != len(set(codes)):
                        raise ValueError("duplicate raw reported seal stock")
                    metadata_path = Path(r["source_path"]).parent/"manifest.json"
                    metadata = read(metadata_path)
                    validate_reported_capture(metadata, r["date"], r["source_sha256"], len(records))
                    sources |= bind(metadata_path)
                    raw_seals[r["source_path"]] = dict(zip(codes, records))
                metadata = read(Path(r["source_path"]).parent/"manifest.json")
                original_hash_present = validate_reported_capture(metadata, r["date"], r["source_sha256"], len(raw_seals[r["source_path"]]))
                if str(metadata.get("collected_at") or "") != r["captured_at"]:
                    raise ValueError("reported capture time differs from original manifest")
                original = raw_seals[r["source_path"]][r["code"].zfill(6)]
                for field, raw_field in (("price", "最新价"), ("first_seal", "首次封板时间"),
                                         ("last_seal", "最后封板时间"), ("bomb_count", "炸板次数")):
                    if Decimal(r[field]) != Decimal(original[raw_field]):
                        raise ValueError("reported seal differs from saved raw quote")
                seals[key] = dict(r, original_capture_manifest_hash_present=original_hash_present)
    groups = {}
    for day, code in cohort:
        check = archive.check_signal(raw[code], reference, code, day)
        if clean(check) != checks[(day, code)]:
            raise ValueError("sampled checks differ from original raw response reconstruction")
        own = index[day+":"+code]
        quote = own["groups"]["price"]["value"].get("price")
        if _cent_price(quote) != _cent_price(check["observed_ohlc"]["close"]):
            raise ValueError("sampled price differs by a cent from original candidate")
        bars = raw[code][raw[code].minute.str[:10].eq(day)].to_dict("records")
        fact = sampled_price_path(bars, check, quote)
        if day+":"+code in seals:
            s = seals[day+":"+code]
            if _cent_price(s["price"]) != _cent_price(quote):
                raise ValueError("reported seal closing quote differs from sampled limit")
            fact["provider_reported_seal"] = {k: s[k] for k in ("first_seal", "last_seal", "bomb_count", "first_last_clock_gap_minutes")}
            fact["provider_reported_seal"]["first_last_gap_is_not_actual_open_duration"] = True
            fact["provider_reported_seal"]["original_capture_manifest_hash_present"] = s["original_capture_manifest_hash_present"]
            fact["provider_reported_seal"]["raw_hash_binding"] = ("ORIGINAL_MANIFEST_AND_V275_ARCHIVE" if s["original_capture_manifest_hash_present"] else "V275_ARCHIVE_HASH_ORIGINAL_CAPTURE_HASH_UNKNOWN")
        groups[day+":"+code] = {"sampled_path": fact}
    return {"version": VERSION, "formed_at": formed, "derivation_kind": "HISTORICAL_SAMPLED_PRICE_PATH",
        "parent_path": str(Path(parent_path).resolve()), "report_path": str(Path(report_path).resolve()),
        "seal_path": str(Path(seal_path).resolve()) if seal_path else None, "capture_metadata_bound": True,
        "source_bindings": sources, "groups": groups, "historical_diagnostic_only": True,
        "coverage": {"candidates": len(groups), "dates": len({d for d, _ in cohort}),
                     "complete_grid": sum(v["sampled_path"]["grid_complete"] for v in groups.values()),
                     "volume_reconciled": sum(v["sampled_path"]["volume_reconciled"] for v in groups.values()),
                     "reported_seal_rows": len(seals)}, "role_labels_created": False}


@lru_cache(maxsize=2)
def _replay_sampled_bundle(path, expected_hash):
    verify({path: expected_hash})
    source = read(path)
    return _build_sampled_bundle(source["parent_path"], source["report_path"], source.get("seal_path"), source["formed_at"])


def sampled_history(parent_path, report_path, output_dir, seal_path=None):
    root = Path(output_dir)
    if root.exists():
        raise ValueError("sampled history output directory must be new")
    formed = now()
    bundle = _build_sampled_bundle(parent_path, report_path, seal_path, formed)
    root.mkdir(parents=True)
    source = root/"source_rebuilt_sampled_price_facts.json"
    write_new(source, bundle)
    bank = root/"sampled_path_evidence_bank.json"
    write_new(bank, {"version": VERSION, "formed_at": formed, "rows": [
        {"candidate_id": ident, "groups": {"sampled_path": {"value": group["sampled_path"], "status": "INFERRED",
         "source_path": str(source.resolve()), "source_sha256": sha(source), "pointer": ["groups", ident, "sampled_path"],
         "observed_at": ident[:10]+"T15:00:00+08:00", "available_at": formed,
         "original_source_available_at": read(report_path)["recorded_at"], "reconstructed": True}}}
        for ident, group in bundle["groups"].items()]})
    return augment(parent_path, root/"history_with_sampled_paths.json", evidence_path=bank)


def relative_sampled_prices(own_code, expected_peers, paths, scope_complete):
    """Complete declared peers only; missing peers never become failed rivals."""
    peers = sorted(expected_peers)
    if peers != sorted(set(peers)) or own_code in peers:
        raise ValueError("unique peer codes excluding self required")
    available = [c for c in peers if c in paths and paths[c].get("price_reconciled") is True]
    features = {"peer_expected": len(peers), "peer_available": len(available),
        "peer_coverage": len(available)/len(peers) if peers else None,
        "relative_min_pp": None, "relative_last_pp": None, "positive_bar_share": None, "amount_share": None}
    result = {"peer_codes": peers, "missing_peer_codes": sorted(set(peers)-set(available)),
        "scope_complete": scope_complete is True, "features": features, "relative_series": [],
        "complete_price_comparison": False, "not_fund_causality_or_absorption": True}
    if not peers or scope_complete is not True or len(available) != len(peers) or own_code not in paths:
        return result
    selected = [paths[c] for c in [own_code, *peers]]
    if any(p.get("price_reconciled") is not True or p.get("grid_complete") is not True
           or number(p.get("previous_close")) is None or p["previous_close"] <= 0 for p in selected):
        return result
    times = [[b["minute"] for b in p["bars"]] for p in selected]
    if not times[0] or any(t != times[0] or t != sorted(set(t)) for t in times):
        return result
    dates = {t[:10] for t in times[0]}
    if len(dates) != 1 or any(p.get("signal_date") not in dates for p in selected):
        raise ValueError("peer bars cross signal dates")
    previous = [_cent_price(p["previous_close"]) for p in selected]
    series = []
    for i, minute in enumerate(times[0]):
        changes = [float((_cent_price(p["bars"][i]["close"])/prev-1)*100) for p, prev in zip(selected, previous)]
        series.append({"minute": minute, "own_unadjusted_change_pct": changes[0],
            "peer_mean_ex_self_pct": sum(changes[1:])/len(peers),
            "relative_pp": changes[0]-sum(changes[1:])/len(peers)})
    features.update(relative_min_pp=min(v["relative_pp"] for v in series),
        relative_last_pp=series[-1]["relative_pp"], positive_bar_share=sum(v["relative_pp"] > 0 for v in series)/len(series))
    if all(p.get("volume_reconciled") is True and p.get("amount_volume_unit_consistent") is True for p in selected):
        amounts = [sum(number(b.get("amount")) or 0 for b in p["bars"]) for p in selected]
        if sum(amounts) > 0: features["amount_share"] = amounts[0]/sum(amounts)
    result.update(complete_price_comparison=True, relative_series=series)
    return result


def _build_peer_bundle(parent_path, formed):
    import pandas as pd
    import public_minute_capture_v1 as parser
    import archive_september_5min_v287 as archive
    parent = load_dataset(parent_path)
    selected = [r for r in parent["rows"] if "sampled_path" in r["groups"]]
    source_paths = {r["groups"]["sampled_path"]["source_path"] for r in selected}
    if len(source_paths) != 1:
        raise ValueError("one complete source-bound sampled archive required")
    sampled_path = next(iter(source_paths)); sampled = read(sampled_path)
    report_path = sampled["report_path"]; report = read(report_path)
    if sampled.get("capture_metadata_bound") is not True or len(selected) != report["candidate_rows"]:
        raise ValueError("complete capture-bound sampled cohort required")
    sources = bind(parent_path) | bind(sampled_path) | bind(report_path) | sampled["source_bindings"]
    verify(sources)
    daily_path = next(p for p in report["source_hashes"] if Path(p).name == "unadjusted_daily.csv")
    reference = pd.read_csv(daily_path)
    days = calendar(parent["calendar_path"], parent["calendar_sha256"])
    audits = {r["code"]: r for r in report["audit"]}
    raw = {code: parser.parse_minute_response(Path(a["raw_path"]).read_text(encoding="utf-8"), code)
           for code, a in audits.items()}
    paths = {}
    def get_path(code, day):
        key = day+":"+code
        if key in paths: return paths[key]
        if code not in raw: return None
        check = archive.check_signal(raw[code], reference, code, day)
        bars = raw[code][raw[code].minute.str[:10].eq(day)].to_dict("records")
        if not bars: return None
        pos = days.index(day)
        previous_day = days[pos-1] if pos else None
        symbol = ("sz" if code.startswith("00") else "sh")+code
        previous = reference[reference.symbol.eq(symbol)&reference.date.eq(previous_day)]
        previous_close = float(previous.close.iloc[0]) if len(previous) == 1 else None
        if previous_close is not None: _cent_price(previous_close)
        paths[key] = {"signal_date": day, "code": code, "previous_session": previous_day,
            "previous_close": previous_close, "bars": clean(bars),
            **{k: check.get(k) for k in ("price_reconciled", "grid_complete", "missing_times",
                "volume_reconciled", "amount_volume_unit_consistent")}, "unadjusted_price_change_not_trading_profit": True}
        return paths[key]
    groups = {}
    for row in selected:
        day, code = row["signal_date"], row["code"]
        pool = row["groups"]["pool"]["value"]
        current = sorted(c for c in pool["codes"] if c != code and pool["boards"][c] == row["boards"])
        competition = row["groups"].get("competition", {}).get("value", {})
        prior = competition.get("peer_codes", [])
        local = {c: get_path(c, day) for c in [code, *sorted(set(current+prior))]}
        local = {c: p for c, p in local.items() if p is not None}
        own = local.get(code)
        if not own or own["price_reconciled"] is not True:
            raise ValueError("original sampled own prices failed replay")
        value = {"signal_date": day, "code": code, "scope": "DECLARED_ARCHIVE_ONLY_NOT_ALL_MARKET_COMPETITORS",
            "current": relative_sampled_prices(code, current, local, pool.get("complete_within_scope")),
            "prior": relative_sampled_prices(code, prior, local, competition.get("complete_within_scope")),
            "current_scope": "CURRENT_SAME_HEIGHT_CLOSING_COHORT_SURVIVORS_ONLY",
            "prior_scope": "PREVIOUS_SESSION_DECLARED_COMPETITORS_WITH_ACTUAL_CURRENT_STATES_OR_UNKNOWN",
            "prior_current_states": competition.get("current", {}),
            "path_refs": {c: day+":"+c for c in local}, "identities_certified": False,
            "absorption_verified": False, "actual_opponent_elimination_not_inferred_from_pool_departure": True}
        groups[row["candidate_id"]] = {"sampled_peers": value}
    coverage = {"candidates": len(groups), "unique_stock_day_paths": len(paths),
        "current_complete_price_comparisons": sum(g["sampled_peers"]["current"]["complete_price_comparison"] for g in groups.values()),
        "prior_complete_price_comparisons": sum(g["sampled_peers"]["prior"]["complete_price_comparison"] for g in groups.values()),
        "prior_declared_competition_cases": sum(bool(g["sampled_peers"]["prior"]["peer_codes"]) for g in groups.values())}
    return {"version": VERSION, "formed_at": formed, "derivation_kind": "HISTORICAL_PEER_PRICE_PATH",
        "parent_path": str(Path(parent_path).resolve()), "source_bindings": sources,
        "paths": paths, "groups": groups, "coverage": coverage, "role_labels_created": False,
        "historical_diagnostic_only": True}


@lru_cache(maxsize=2)
def _replay_peer_bundle(path, expected_hash):
    verify({path: expected_hash})
    bundle = read(path)
    return _build_peer_bundle(bundle["parent_path"], bundle["formed_at"])


@verified_session
def peer_history(parent_path, output_dir):
    root = Path(output_dir)
    if root.exists(): raise ValueError("peer path output directory must be new")
    formed = now(); bundle = _build_peer_bundle(parent_path, formed)
    root.mkdir(parents=True)
    source = root/"source_rebuilt_peer_price_facts.json"; write_new(source, bundle)
    bank = root/"peer_path_evidence_bank.json"
    write_new(bank, {"version": VERSION, "formed_at": formed, "rows": [
        {"candidate_id": ident, "groups": {"sampled_peers": {"value": group["sampled_peers"], "status": "INFERRED",
         "source_path": str(source.resolve()), "source_sha256": sha(source), "pointer": ["groups", ident, "sampled_peers"],
         "observed_at": ident[:10]+"T15:00:00+08:00", "available_at": formed, "reconstructed": True}}}
        for ident, group in bundle["groups"].items()]})
    return augment(parent_path, root/"history_with_peer_paths.json", evidence_path=bank)


def market_ex_self(record, pool, own_code):
    """Only subtract a stock when archived ladder counts and height agree."""
    features = {k: None for k in ("market_limit_count_ex_self", "market_consecutive_count_ex_self", "market_height_ex_self")}
    raw = record["raw"]
    if (not pool or own_code not in pool["boards"] or pool["boards"][own_code] < 2
            or number(raw.get("limit_count")) is None or number(raw.get("consecutive_count")) != len(pool["codes"])
            or number(raw.get("max_board")) != max(pool["boards"].values())
            or raw["limit_count"] < raw["consecutive_count"]):
        return {"features": features, "status": "COHORT_COUNTS_OR_HEIGHT_UNMATCHED_UNKNOWN"}
    peers = [b for c,b in pool["boards"].items() if c != own_code]
    first_count = raw["limit_count"]-raw["consecutive_count"]
    features.update(market_limit_count_ex_self=raw["limit_count"]-1,
        market_consecutive_count_ex_self=raw["consecutive_count"]-1,
        market_height_ex_self=max([*peers, *([1] if first_count > 0 else [])], default=0))
    return {"features": features, "status": "SOURCE_COHORT_COUNT_AND_HEIGHT_MATCH_NOT_FULL_MARKET_CERTIFICATION"}


def market_context_window(records, signal, days):
    """Past official sessions only. Missing observations stay in the window."""
    if signal not in days: raise ValueError("market context signal outside official calendar")
    indexed = {}
    for r in records:
        if r["date"] in indexed or r["date"] not in days:
            raise ValueError("duplicate or unofficial market date")
        if set(r.get("raw", {})) != set(MARKET_RAW_FIELDS):
            raise ValueError("market record contains unapproved forecast/label columns")
        indexed[r["date"]] = r
    pos = days.index(signal); selected_days = days[max(0,pos-5):pos+1]
    empty = lambda day: {"date":day,"raw":{k:None for k in MARKET_RAW_FIELDS},"data_status":"MISSING_OFFICIAL_SESSION","rule_phase":None}
    window = [indexed.get(day,empty(day)) for day in selected_days]
    current = window[-1];features = {}
    for field in MARKET_MODEL_FIELDS:
        own = number(current["raw"].get(field));features["market_"+field] = own
        for lag in (1,3,5):
            prior = indexed.get(days[pos-lag]) if pos>=lag else None
            before = number(prior["raw"].get(field)) if prior else None
            features["market_"+field+"_delta_"+str(lag)] = own-before if own is not None and before is not None else None
    context = {"signal_date":signal,"scope":"ORIGINAL_ARCHIVE_NOT_EXCHANGE_FULL_MARKET_CERTIFIED",
        "records":window,"features":features,"market_phase_is_not_stock_role_truth":True,
        "market_count_or_daily_comovement_is_not_lead_lag_fund_causality":True}
    # Existing stage rules are context features only, never teacher identities.
    recent = window[-3:];essential=("limit_count","max_board","failed_count","down_limit_count")
    phase_ok = len(recent)==3 and all(r["data_status"]=="verified" and r.get("rule_phase") in PHASE_MAP.values()
        and all(number(r["raw"].get(k)) is not None for k in essential) for r in recent)
    cycle = {"records":[{"date":r["date"],"state":r["rule_phase"]} for r in recent],
        "stage_origin":"EXISTING_SAVED_RULE_STAGE_NOT_INDEPENDENT_ROLE_IDENTITY",
        "rule_uses_legacy_low_turnover_proxy_not_actual_one_word_identity":True,
        "market_phase_is_not_stock_role_truth":True} if phase_ok else None
    return context,cycle


def _build_market_bundle(parent_path, receipt_path, formed):
    import pandas as pd
    import joblib
    parent = load_dataset(parent_path);receipt = read(receipt_path)
    bindings = receipt["sources"] | {receipt["prepared_path"]:receipt["prepared_sha256"]} | bind(receipt_path) | bind(parent_path)
    verify(bindings)
    if stamp(receipt["formed_at"])>stamp(formed): raise ValueError("market receipt formed after context")
    prepared = joblib.load(receipt["prepared_path"])
    frame = prepared["market"]
    csv_path = next(p for p in receipt["sources"] if Path(p).name=="combined_market_input.csv")
    columns = ["date","data_status","emotion_state",*MARKET_RAW_FIELDS]
    source = pd.read_csv(csv_path,usecols=columns)
    if frame.date.duplicated().any() or source.date.duplicated().any() or set(frame.date)!=set(source.date):
        raise ValueError("complete original market dates differ")
    raw_index = {r["date"]:r for r in source.to_dict("records")}
    days = calendar(parent["calendar_path"],parent["calendar_sha256"])
    records=[]
    for r in frame.to_dict("records"):
        original = raw_index[r["date"]]
        for field in MARKET_RAW_FIELDS:
            a,b = number(r.get(field)),number(original.get(field))
            if (a is None)!=(b is None) or a is not None and abs(a-b)>1e-10*max(1.,abs(b)):
                raise ValueError("prepared market differs from original raw column: "+field)
        phase = r.get("emotion_state") if isinstance(r.get("emotion_state"),str) else None
        original_phase = original.get("emotion_state") if isinstance(original.get("emotion_state"),str) else None
        if phase!=original_phase or r["data_status"]!=original["data_status"]:
            raise ValueError("original saved rule phase/status changed")
        records.append({"date":r["date"],"raw":{k:number(original.get(k)) for k in MARKET_RAW_FIELDS},
            "data_status":r["data_status"],"rule_phase":PHASE_MAP.get(phase),
            "rule_phase_not_independent_role_or_actual_capital_flow_truth":True})
    groups = {}
    for day in sorted({r["signal_date"] for r in parent["rows"]}):
        context,cycle = market_context_window(records,day,days)
        groups[day]={"market_context":context}
        if cycle:groups[day]["cycle"]=cycle
    coverage = {"market_records":len(records),"candidate_dates":len(groups),"cycle_context_dates":sum("cycle" in g for g in groups.values()),
        "market_status_counts":dict(Counter(r["data_status"] for r in records)),
        "candidates_with_context":len(parent["rows"]),"candidates_with_cycle":sum("cycle" in groups[r["signal_date"]] for r in parent["rows"])}
    return {"version":VERSION,"formed_at":formed,"derivation_kind":"HISTORICAL_MARKET_RAW_CONTEXT",
        "parent_path":str(Path(parent_path).resolve()),"receipt_path":str(Path(receipt_path).resolve()),
        "source_bindings":bindings,"market_records":records,"groups":groups,"coverage":coverage,
        "raw_field_allowlist":list(MARKET_RAW_FIELDS),"forecast_probabilities_or_future_targets_projected":False,
        "historical_diagnostic_only":True,"original_acquisition_times_not_backfilled":True}


@lru_cache(maxsize=2)
def _replay_market_bundle(path,expected_hash):
    verify({path:expected_hash});source=read(path)
    return _build_market_bundle(source["parent_path"],source["receipt_path"],source["formed_at"])


@verified_session
def market_history(parent_path,receipt_path,output_dir):
    root=Path(output_dir)
    if root.exists():raise ValueError("market context output directory must be new")
    started=now();bundle=_build_market_bundle(parent_path,receipt_path,started)
    formed=now();bundle["formed_at"]=formed
    root.mkdir(parents=True);source=root/"source_rebuilt_market_raw_facts.json";write_new(source,bundle)
    source_digest=sha(source)
    parent=load_dataset(parent_path);bank=root/"market_context_evidence_bank.json"
    rows=[{"candidate_id":r["candidate_id"],"groups":{kind:{"value":value,"status":"INFERRED",
        "source_path":str(source.resolve()),"source_sha256":source_digest,"pointer":["groups",r["signal_date"],kind],
        "observed_at":r["signal_date"]+"T15:00:00+08:00","available_at":formed,"reconstructed":True}
        for kind,value in bundle["groups"][r["signal_date"]].items()}} for r in parent["rows"]]
    write_new(bank,{"version":VERSION,"formed_at":formed,"rows":rows})
    return augment(parent_path,root/"history_with_market_context.json",evidence_path=bank)


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
    a = sub.add_parser("slice-dates")
    for key in ("dataset", "start", "end", "output"): a.add_argument("--"+key, required=True)
    a = sub.add_parser("relate-history")
    for key in ("dataset", "relations", "output-dir"): a.add_argument("--"+key, required=True)
    a = sub.add_parser("sampled-history")
    for key in ("dataset", "report", "output-dir"): a.add_argument("--"+key, required=True)
    a.add_argument("--reported-seals")
    a = sub.add_parser("peer-history")
    for key in ("dataset", "output-dir"): a.add_argument("--"+key, required=True)
    a = sub.add_parser("market-history")
    for key in ("dataset","receipt","output-dir"):a.add_argument("--"+key,required=True)
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
    elif args.command == "slice-dates":
        r = slice_dates(args.dataset, args.start, args.end, args.output)
    elif args.command == "relate-history":
        r = relate_history(args.dataset, args.relations, args.output_dir)
    elif args.command == "sampled-history":
        r = sampled_history(args.dataset, args.report, args.output_dir, args.reported_seals)
    elif args.command == "peer-history":
        r = peer_history(args.dataset, args.output_dir)
    elif args.command == "market-history":
        r = market_history(args.dataset,args.receipt,args.output_dir)
    else:
        r = analyse(load_dataset(args.dataset), args.track); write_new(args.output, r)
    print(json.dumps({"version": VERSION, "rows": len(r["rows"]), "production_changed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
