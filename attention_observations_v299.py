"""Archive current provider attention observations; never certify forum consensus.

Five finite, read-only data queries per original candidate. Historical rows
become available at this capture, not at the calculation dates they contain.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import argparse
import hashlib
import json
from pathlib import Path
import re
import requests

from daily_research_contract_v292 import stamp
from next_session_expectation_contract_v271 import digest
from point_context_v295 import APP, scope_from_registered, scope_from_pool, symbol

ROOT = Path(__file__).resolve().parent
ZONE = timezone(timedelta(hours=8))
VERSION = "v299-provider-attention-observations-1"
ENDPOINTS = {"latest": "getCurrentLatest", "intraday": "getCurrentList", "profiles": "getHisProfileList",
    "keywords": "getHotStockRankList", "related": "getFollowStockRankList"}
DISCOVERY = ROOT / "data/v299/attention_source_probe_20261002_v1/discovery_source.response"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(payload)


def query(code, kind):
    payload = {"appId": APP["appId"], "globalId": APP["globalId"], "srcSecurityCode": symbol(code)}
    if kind in {"latest", "intraday", "profiles"}:
        payload["marketType"] = ""
    if kind == "profiles":
        payload["yearType"] = "5"
    return "https://emappdata.eastmoney.com/stockrank/" + ENDPOINTS[kind], payload


def fetch(task, directory):
    code, kind = task
    url, payload = query(code, kind)
    record = {"code": code, "kind": kind, "method": "POST", "url": url, "query": payload, "requested_at": now()}
    try:
        response = requests.post(url, json=payload, timeout=(5, 8))
        path = directory / "raw" / (code + "_" + kind + ".response")
        with path.open("xb") as handle:
            handle.write(response.content)
        record.update(available_at=now(), http_status=response.status_code, final_url=response.url,
            raw_path=str(path.resolve()), raw_sha256=sha(path))
        response.raise_for_status()
        if response.url != url:
            raise ValueError("unexpected provider redirect")
        response.json()
        record["status"] = "RAW_JSON_RECEIVED"
    except Exception as exc:
        record.update(status="UNAVAILABLE", available_at=now(), error=str(exc)[:600])
    return record


def integer(value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("nonnegative integer required")
    return value


def percent(value, bounded=True):
    if not isinstance(value, str) or not re.fullmatch(r"[-+]?\d+(?:\.\d+)?%", value):
        raise ValueError("explicit percentage string required")
    number = Decimal(value[:-1])
    if not number.is_finite() or (bounded and not Decimal(0) <= number <= Decimal(100)):
        raise ValueError("percentage outside required domain")
    return number


def calculated(value, available_at):
    if not isinstance(value, str):
        raise ValueError("full calculation timestamp required")
    result = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZONE)
    if result.strftime("%Y-%m-%d %H:%M:%S") != value or result > stamp(available_at):
        raise ValueError("invalid or future calculation timestamp")
    return result


def data(value):
    if not isinstance(value, dict) or value.get("status") != 0 or value.get("code") != 0 or value.get("message") != "OK":
        raise ValueError("provider response unsuccessful")
    return value.get("data")


def rows_data(value):
    result = data(value)
    if not isinstance(result, list) or not result:
        raise ValueError("empty or missing rows do not certify no activity")
    return result


def identity(row, code):
    if row.get("srcSecurityCode") != symbol(code):
        raise ValueError("explicit source security identity mismatch")
    # innerCode is opaque; gs-prefixed IDs must not be rejected as tickers.


def freshness(time, available_at, seconds):
    age = (stamp(available_at) - time).total_seconds()
    return {"age_seconds_at_capture": age, "fresh_at_capture": age <= seconds}


def parse_latest(value, code, available_at):
    row = data(value)
    if not isinstance(row, dict):
        raise ValueError("latest rank object required")
    identity(row, code)
    time = calculated(row.get("calcTime"), available_at)
    rank, total = integer(row.get("rank"), 1), integer(row.get("marketAllCount"), 1)
    if rank > total:
        raise ValueError("rank outside provider universe")
    return {"status": "PROVIDER_EXPLICIT_CODE_RANK", "calculated_at": time.isoformat(), "rank": rank,
        "provider_universe_count": total, **freshness(time, available_at, 1200),
        "rank_changes_provider": {k: row.get(k) for k in ["rankChange", "hisRankChange"]}, "provider_change_semantics_verified": False}


def parse_intraday(value, code, available_at, latest):
    rows = []; seen = set()
    for row in rows_data(value):
        if "srcSecurityCode" in row:
            identity(row, code)
        time = calculated(row.get("calcTime"), available_at)
        if time in seen:
            raise ValueError("duplicate intraday calculation time")
        seen.add(time); rows.append({"calculated_at": time.isoformat(), "rank": integer(row.get("rank"), 1)})
    if [r["calculated_at"] for r in rows] != sorted(r["calculated_at"] for r in rows):
        raise ValueError("intraday rows are not ordered")
    if len({stamp(r["calculated_at"]).astimezone(ZONE).date() for r in rows}) != 1:
        raise ValueError("intraday response mixes calculation dates")
    same_clock = next((r for r in rows if latest and r["calculated_at"] == latest["calculated_at"]), None)
    if same_clock and same_clock["rank"] != latest["rank"]:
        raise ValueError("same-clock rank endpoints disagree")
    times = [stamp(r["calculated_at"]).astimezone(ZONE) for r in rows]
    if any(t.minute % 10 or t.second for t in times):
        raise ValueError("intraday times not on documented ten-minute grid")
    midnight = times[0].replace(hour=0, minute=0, second=0, microsecond=0)
    expected = [midnight + timedelta(minutes=10*i) for i in range(int((times[-1]-midnight).total_seconds()/600)+1)]
    missing = [t.isoformat() for t in expected if t not in seen]
    return {"status": "REQUEST_BOUND_INTRADAY_PROVIDER_RANK", "code": code, "rows": rows,
        "security_code_returned_in_each_row": all("srcSecurityCode" in r for r in rows_data(value)),
        "identity_limitation": "request parameter only when the response omits the security code; same-provider latest is not independent certification",
        "same_clock_latest_checked": same_clock is not None, "missing_grid_times": missing,
        "first_rank": rows[0]["rank"], "last_rank": rows[-1]["rank"],
        "rank_improvement_first_minus_last": rows[0]["rank"]-rows[-1]["rank"],
        "window_seconds": (times[-1]-times[0]).total_seconds(), **freshness(times[-1], available_at, 1200),
        "bullishness": None, "discussion_count": None}


def parse_profiles(value, code, available_at):
    rows = []; seen = set()
    for row in rows_data(value):
        identity(row, code); time = calculated(row.get("calcTime"), available_at)
        if time in seen:
            raise ValueError("duplicate profile calculation time")
        seen.add(time)
        new, old = percent(row.get("newUidRate")), percent(row.get("oldUidRate"))
        if abs(new + old - Decimal(100)) > Decimal("0.02"):
            raise ValueError("new and established profile percentages inconsistent")
        provider_count = row.get("uidCount")
        if provider_count is not None:
            integer(provider_count)
        rows.append({"calculated_at": time.isoformat(), "new_user_share": float(new/100), "established_user_share": float(old/100),
            "uid_count_provider_unverified": provider_count, "verified_user_count": None,
            "zero_count_with_nonzero_shares": provider_count == 0 and (new > 0 or old > 0),
            "provider_changes_unverified": {k: row.get(k) for k in ["newUidChangeRank", "oldUidChangeRank"]}})
    if [r["calculated_at"] for r in rows] != sorted(r["calculated_at"] for r in rows):
        raise ValueError("profile rows are not ordered")
    latest = rows[-1]
    return {"status": "PROVIDER_EXPLICIT_CODE_PROFILE_HISTORY_CAPTURE", "rows": rows, "latest": latest,
        **freshness(stamp(latest["calculated_at"]), available_at, 5400), "historical_original_availability_verified": False,
        "interpretation": "provider viewing/interaction user composition, not bull/bear votes, trader positioning or capital flows",
        "historical_rows_are_live_features": False}


def parse_keywords(value, code, available_at):
    rows = []; seen = set()
    for row in rows_data(value):
        identity(row, code); time = calculated(row.get("calcTime"), available_at)
        key = (time, row.get("conceptId"))
        name = row.get("conceptName")
        if not isinstance(key[1], str) or not re.fullmatch(r"BK\d+", key[1]) or key in seen or not isinstance(name, str) or not name.strip():
            raise ValueError("invalid or duplicate provider keyword identity")
        seen.add(key)
        rows.append({"calculated_at": time.isoformat(), "concept_id": key[1], "concept_name_provider": name,
            "hit_count_provider": integer(row.get("hitCount")), **freshness(time, available_at, 5400)})
    return {"status": "PROVIDER_EXPLICIT_CODE_KEYWORD_FIELDS", "rows": rows,
        "hit_count_denominator_and_unique_users_verified": False, "discussion_count": None,
        "active_trading_theme_verified": False, "interpretation": "provider keyword hitCount only; no conversion to total forum discussions"}


def parse_related(value, code, available_at):
    rows = []; seen = set()
    for row in rows_data(value):
        identity(row, code); time = calculated(row.get("calcTime"), available_at)
        other = row.get("followSrcSecurityCode")
        if not isinstance(other, str) or not re.fullmatch(r"(?:SH|SZ)\d{6}|HK\d{5}", other) or (time, other) in seen:
            raise ValueError("invalid or duplicate related security identity")
        seen.add((time, other))
        rate = percent(row.get("rate"), bounded=False)
        rows.append({"calculated_at": time.isoformat(), "related_security_provider": other,
            "provider_rate_literal": row["rate"], "provider_rate_percent_unverified_semantics": float(rate), **freshness(time, available_at, 5400)})
    return {"status": "PROVIDER_EXPLICIT_CODE_RELATED_STOCK_FIELDS", "rows": rows,
        "relationship_causality_verified": False, "rate_is_verified_stock_return": False,
        "same_theme_or_controller_verified": False, "related_rows_expand_relay_denominator": False}


def normalize(scope, entries, recorded_at):
    index = {(e["code"], e["kind"]): e for e in entries}
    if len(index) != len(entries) or set(index) != {(c, k) for c in scope["codes"] for k in ENDPOINTS}:
        raise ValueError("all original codes and five requests must be preserved")
    observations = []
    for code in scope["codes"]:
        observation = {"code": code, "errors": {}, "forum_heat": None, "discussion_change": None,
            "market_agreement": None, "bull_bear_extremes": None}
        def read(kind):
            e = index[code, kind]
            if stamp(e["requested_at"]) > stamp(e["available_at"]) or stamp(e["available_at"]) > stamp(recorded_at):
                raise ValueError("request availability clocks inconsistent")
            if e["url"] != query(code, kind)[0] or e["query"] != query(code, kind)[1] or e.get("method") != "POST":
                raise ValueError("request code or endpoint scope differs")
            if e["status"] != "RAW_JSON_RECEIVED" or e.get("http_status") != 200:
                raise ValueError(e.get("error", "raw provider response unavailable"))
            if sha(e["raw_path"]) != e["raw_sha256"]:
                raise ValueError("raw response changed")
            return json.loads(Path(e["raw_path"]).read_bytes()), e
        for kind in ENDPOINTS:
            observation[kind] = None
            try:
                value, entry = read(kind)
                if kind == "intraday":
                    item = parse_intraday(value, code, entry["available_at"], observation["latest"])
                else:
                    parser = {"latest": parse_latest, "profiles": parse_profiles, "keywords": parse_keywords, "related": parse_related}[kind]
                    item = parser(value, code, entry["available_at"])
                item.update(available_at=entry["available_at"], raw_path=entry["raw_path"], raw_sha256=entry["raw_sha256"])
                observation[kind] = item
            except (ValueError, KeyError, TypeError, OSError) as exc:
                observation["errors"][kind] = str(exc)[:600]
        observations.append(observation)
    return {"version": VERSION, "recorded_at": recorded_at, "scope": scope, "observations": observations,
        "forum_consensus_verified": False, "historical_features_backfilled": False, "independent_provider_confirmation": False,
        "leader_confirmation_authorized": False, "probability_adjustment_authorized": False, "trade_authorized": False}


def reconstruct(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION:
        raise ValueError("unsupported attention archive")
    for name, wanted in manifest["source_hashes"].items():
        if sha(name) != wanted:
            raise ValueError("attention source program or discovery evidence changed")
    scope = manifest["scope"]
    if scope["kind"] == "ALL_ORIGINAL_REGISTERED_RELAY_CANDIDATES":
        rebuilt_scope = scope_from_registered(scope["source_path"])
    else:
        rebuilt_scope = scope_from_pool(scope["source_path"], scope["price_signal_date"])
    if digest(scope) != digest(rebuilt_scope):
        raise ValueError("original attention denominator source changed")
    for entry in manifest["entries"]:
        if "raw_path" in entry and sha(entry["raw_path"]) != entry["raw_sha256"]:
            raise ValueError("raw attention response missing or changed")
    expected = normalize(scope, manifest["entries"], manifest["recorded_at"])
    saved = json.loads((directory / "observations.json").read_text(encoding="utf-8"))
    if sha(directory / "observations.json") != manifest["observations_sha256"] or digest(expected) != digest(saved):
        raise ValueError("attention normalized fields differ from raw reconstruction")
    return expected


def collect(scope, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False); (output / "raw").mkdir()
    tasks = [(c, k) for c in scope["codes"] for k in ENDPOINTS]
    entries = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for entry in executor.map(lambda task: fetch(task, output), tasks):
            entries.append(entry)
            if len(entries) % 5 == 0:
                print(json.dumps({"codes_received": len(entries)//5, "all_codes": len(scope["codes"]), "last_code": entry["code"]}), flush=True)
    recorded = now(); value = normalize(scope, entries, recorded)
    write(output / "observations.json", value)
    manifest = {"version": VERSION, "recorded_at": recorded, "scope": scope, "entries": entries,
        "observations_sha256": sha(output / "observations.json"),
        "source_hashes": {str(p.resolve()): sha(p) for p in [Path(__file__), ROOT / "point_context_v295.py", ROOT / "public_outcome_sources_v293.py", DISCOVERY]}}
    write(output / "manifest.json", manifest)
    reconstruct(output)
    return {"status": "RAW_ATTENTION_OBSERVATIONS_RECONSTRUCTED", "codes": len(scope["codes"]),
        "requests": len(entries), "valid_fields": {k: sum(r[k] is not None for r in value["observations"]) for k in ENDPOINTS},
        "forum_consensus_verified": False, "original_forecast_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["collect", "verify"])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--registered"); group.add_argument("--pool-archive")
    parser.add_argument("--pool-date"); parser.add_argument("--output"); parser.add_argument("--archive")
    args = parser.parse_args()
    if args.command == "verify":
        if not args.archive: parser.error("--archive required")
        value = reconstruct(args.archive)
        result = {"status": "RAW_ATTENTION_RECONSTRUCTION_PASSED", "codes": len(value["observations"]), "forum_consensus_verified": False}
    else:
        if not args.output or not (args.registered or args.pool_archive): parser.error("source scope and --output required")
        if args.pool_archive and not args.pool_date: parser.error("--pool-date required")
        scope = scope_from_registered(args.registered) if args.registered else scope_from_pool(args.pool_archive, args.pool_date)
        result = collect(scope, args.output)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
