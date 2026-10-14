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
    "competition_failed_share", "emotion_response_count")
FEATURES = tuple(RELAY_FIELDS) + RELATION_FIELDS + tuple(
    f"{field}_delta_{lag}" for field in ("theme_amount_share", "theme_return_ex_self_pp", "price", "turnover")
    for lag in (1, 3, 5))
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
    if cell.get("adapter") != "SAVED_BASE_ALLOWLIST":
        path = cell["source_path"]
        verify({path: cell["source_sha256"]})
        source = read(path)
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
                if key.endswith("_at") and isinstance(item, str):
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
                and all(type(x) is int and x >= 0 for x in boards.values()))
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
        return (bool(v.get("event_id")) and v.get("pre_event_role") and v.get("ended_at")
                and stamp(v["started_at"]) < stamp(v["ended_at"]) <= stamp(row["decision_at"])
                and v.get("post_event_competition_verified") is True)
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
    if values["competition"]:
        v = values["competition"]
        features["competition_failed_share"] = sum(v["current"][c] == "FAILED" for c in v["peer_codes"])/len(v["peer_codes"])
        relations.extend({"kind": "COMPETITOR", "to": c, "state": v["current"][c]} for c in v["peer_codes"])
    if values["dependency"]:
        v = values["dependency"]
        relations.append({"kind": "DEPENDENCY", "to": v["anchor_code"]})
        if number(v.get("anchor_boards")) is not None:
            features["dependency_height_gap"] = v["anchor_boards"]-row["boards"]
    if values["pressure"]:
        features["cycle_pressure_count"] = 1
        relations.append({"kind": "PRESSURE_EVENT", "to": values["pressure"]["event_id"]})
    if values["emotion"]:
        features["emotion_response_count"] = len(values["emotion"]["responses"])
    eligible, roles = {}, {}
    for role in ROLES:
        gaps = [missing[g] for g in REQUIRED[role] if missing[g]]
        eligible[role] = not gaps
        call = None
        if not gaps and role == "HEIGHT":
            call = pool["boards"][row["code"]] == max(pool["boards"].values())
        # Other identities are hypotheses, never certified from one feature.
        roles[role] = {"state": "EVIDENCE_SUPPORTED" if call is True else "COMPETING" if call is False else "UNKNOWN" if gaps else "CANDIDATE",
            "baseline_call": call, "score": None, "missing": gaps,
            "supporting_groups": [g for g in REQUIRED[role] if values[g] is not None],
            "counterevidence": [], "invalidation": "Recheck required evidence, peer competition and hard risks at next actual observation"}
    return {"candidate_id": row["candidate_id"], "signal_date": row["signal_date"], "code": row["code"],
        "name": row.get("name"), "boards": row["boards"], "decision_at": row["decision_at"],
        "available_at": row["available_at"], "scope_id": row["scope_id"],
        "actual_one_word_daily": row.get("actual_one_word_daily"),
        "low_turnover_proxy": price.get("signal_one_word"),
        "features": features, "role_eligible": eligible, "roles": roles, "relations": relations,
        "peer_codes": peers, "evidence_grade": "SOURCE_BOUND_NOT_EXTERNAL_TRUTH_CERTIFICATION",
        "source_groups": row.get("groups", {}), "role_identity_automatically_certified": False}


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
    index = {(r["signal_date"], r["code"]): r for r in output}
    for r in output:
        pos = days.index(r["signal_date"])
        for field in ("theme_amount_share", "theme_return_ex_self_pp", "price", "turnover"):
            for lag in (1, 3, 5):
                prior = index.get((days[pos-lag], r["code"])) if pos >= lag else None
                a, b = r["features"][field], prior["features"][field] if prior else None
                if prior and prior["scope_id"] == r["scope_id"] and a is not None and b is not None and stamp(prior["available_at"]) <= stamp(r["decision_at"]):
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
        "amounts": {c: price.get(c, {}).get("features", {}).get("signal_amount") for c in codes},
        "turnovers": {c: price.get(c, {}).get("features", {}).get("turnover") for c in codes}}
    rows = []
    observed = signal+"T15:00:00+08:00"
    for r in roster:
        groups = {}
        for kind, value in (("pool", pool), ("price", {k: price.get(r["code"], {}).get("features", {}).get(k) for k in RELAY_FIELDS})):
            groups[kind] = {"value": value, "status": "VERIFIED", "observed_at": observed,
                "available_at": manifest["formed_at"], "adapter": "SAVED_BASE_ALLOWLIST", "reconstructed": False}
        rows.append({"candidate_id": signal+":"+r["code"], "signal_date": signal,
            "code": r["code"], "name": r.get("name"), "boards": int(r["boards"]),
            "scope_id": data["scope"], "decision_at": cutoff, "available_at": manifest["formed_at"],
            "actual_one_word_daily": price.get(r["code"], {}).get("actual_one_word_daily"), "groups": groups})
    result = {"version": VERSION, "formed_at": now(), "rows": rows,
        "calendar_path": str(Path(calendar_path).resolve()), "calendar_sha256": sha(calendar_path),
        "source_bindings": sources, "adapter": {"kind": "SAVED_ARCHIVE", "features": str(Path(features_dir).resolve()),
            "theme": str(Path(theme_path).resolve()) if theme_path else None},
        "original_pool_counts": {"closing": len(roster), "relay": len(price)},
        "official_full_market": False}
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
    if d.get("adapter", {}).get("kind") == "SAVED_ARCHIVE":
        a = d["adapter"]
        rebuilt = from_archive(a["features"], d["calendar_path"], a.get("theme"))
        if rebuilt["rows"] != d["rows"] or rebuilt["source_bindings"] != d["source_bindings"]:
            raise ValueError("saved adapter projection changed")
    elif d.get("adapter", {}).get("kind") == "SAVED_HISTORY":
        a = d["adapter"]
        rebuilt = from_history(a["receipt"], a["base_root"], d["calendar_path"])
        if any(rebuilt[k] != d[k] for k in ("rows", "source_bindings", "baseline_promotion")):
            raise ValueError("history adapter projection changed")
    elif any(c.get("adapter") == "SAVED_BASE_ALLOWLIST" for r in d["rows"] for c in r.get("groups", {}).values()):
        raise ValueError("caller cannot claim the trusted saved adapter")
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("build")
    for key in ("features", "calendar", "output"):
        a.add_argument("--"+key, required=True)
    a.add_argument("--theme-report")
    a = sub.add_parser("analyse")
    a.add_argument("--dataset", required=True); a.add_argument("--track", choices=("DIAGNOSTIC", "STRICT"), required=True)
    a.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "build":
        r = from_archive(args.features, args.calendar, args.theme_report, args.output)
    else:
        r = analyse(load_dataset(args.dataset), args.track); write_new(args.output, r)
    print(json.dumps({"version": VERSION, "rows": len(r["rows"]), "production_changed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
