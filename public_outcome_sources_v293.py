"""Immutable raw public receipts for exact signal/target closing research.

Eastmoney pools retain qdate/tc before parsing. Tencent raw unadjusted day
lists and their quote snapshot are archived separately. Positional quote
fields are explicitly inferred and require cross-reconciliation, not treated
as exchange-certified limits or Level-2 information.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path

import requests

from daily_research_runner_v292 import sha, write_new
from next_session_expectation_contract_v271 import digest, finite, stamp, next_session

VERSION = "v293-public-outcome-receipts-1"
EM_BASE = "https://push2ex.eastmoney.com/"
DAY_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
ENDPOINTS = {"limit_up": "getTopicZTPool", "failed_board": "getTopicZBPool",
    "limit_down": "getTopicDTPool", "previous_limit": "getYesterdayZTPool"}
MAINBOARD = ("000", "001", "002", "003", "600", "601", "603", "605")


def now():
    return datetime.now(timezone.utc).isoformat()


def cent(value):
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("invalid price") from exc
    if not number.is_finite() or number <= 0 or number*100 != (number*100).to_integral_value():
        raise ValueError("price must be positive and exactly on the cent grid")
    return int(number*100)


def parse_pool(payload, date, kind):
    data = payload.get("data")
    if not isinstance(data, dict) or str(data.get("qdate")) != date.replace("-", ""):
        raise ValueError("pool returned date differs from the requested exact trading day")
    raw = data.get("pool")
    if not isinstance(raw, list) or type(data.get("tc")) is not int or data["tc"] != len(raw):
        raise ValueError("pool total count disagrees with raw response")
    seen, rows = set(), []
    for item in raw:
        code, name = str(item.get("c", "")), str(item.get("n", ""))
        if len(code) != 6 or not code.isdigit() or not name or code in seen:
            raise ValueError("invalid or duplicate code in raw pool")
        seen.add(code)
        price = cent(Decimal(str(item["p"]))/1000)
        if not code.startswith(MAINBOARD) or "ST" in name.upper() or "退" in name:
            continue
        row = {"code": code, "name": name, "price_cents": price,
            "amount": item.get("amount"), "turnover_pct": item.get("hs"),
            "pct": item.get("zdp"), "industry": item.get("hybk"),
            "seal_first": item.get("fbt"), "seal_last": item.get("lbt"),
            "reported_open_count": item.get("zbc"), "reported_seal_order": item.get("fund")}
        if kind == "limit_up":
            if type(item.get("lbc")) is not int or item["lbc"] < 1:
                raise ValueError("invalid consecutive-board count")
            row["boards"] = item["lbc"]
        if "ztp" in item:
            row["upper_cents_explicit"] = cent(Decimal(str(item["ztp"]))/1000)
        rows.append(row)
    return {"date": date, "kind": kind, "all_market_raw_count": len(raw), "rows": rows,
        "capture_complete_within_provider_response": True,
        "exchange_security_status_verified": False, "exchange_wide_completeness_verified": False}


def parse_daily(payload, code, start, end):
    symbol = ("sh" if code.startswith("6") else "sz")+code
    block = (payload.get("data") or {}).get(symbol)
    if not isinstance(block, dict) or not isinstance(block.get("day"), list) or not block["day"]:
        raise ValueError("raw unadjusted day list missing; adjusted list cannot substitute")
    days, previous = {}, None
    for row in block["day"]:
        if not isinstance(row, list) or len(row) < 6:
            raise ValueError("invalid unadjusted day row")
        date = str(row[0])
        datetime.strptime(date, "%Y-%m-%d")
        if previous is not None and date <= previous:
            raise ValueError("duplicate or unordered daily dates")
        previous = date
        if not start <= date <= end:
            continue
        o, c, h, l = [cent(x) for x in row[1:5]]
        volume = float(row[5])
        if not l <= o <= h or not l <= c <= h or not finite(volume) or volume < 0:
            raise ValueError("invalid daily price bounds or volume")
        days[date] = {"open": o, "close": c, "high": h, "low": l, "volume_native": volume}
    if not days:
        raise ValueError("requested daily date range missing")
    quote = (block.get("qt") or {}).get(symbol)
    snapshot = {"status": "UNAVAILABLE", "reason": "no usable raw quote snapshot"}
    if isinstance(quote, list) and len(quote) >= 49 and str(quote[2]) == code:
        try:
            time = datetime.strptime(str(quote[30]), "%Y%m%d%H%M%S").replace(tzinfo=stamp("2000-01-01T00:00:00+08:00").tzinfo)
            values = {k: cent(quote[index]) for k, index in {"close": 3, "previous_close": 4, "open": 5,
                "high": 33, "low": 34, "upper_candidate": 47, "lower_candidate": 48}.items()}
            if not values["low"] <= values["close"] <= values["high"] or not values["lower_candidate"] < values["upper_candidate"]:
                raise ValueError("raw quote price bounds inconsistent")
            snapshot = {"status": "POSITIONAL_FIELDS_INFERRED", "observed_at": time.isoformat(), "values": values,
                "semantic_status": "INFERENCE_REQUIRES_EXPLICIT_PRICE_LIMIT_RECONCILIATION"}
        except (ValueError, TypeError, KeyError) as exc:
            snapshot = {"status": "UNAVAILABLE", "reason": str(exc)}
    return {"code": code, "days": days, "quote": snapshot, "adjusted": False,
        "auction_verified": False, "reseal_verified": False}


def fetch(url, params, path):
    entry = {"url": url, "parameters": params, "requested_at": now()}
    try:
        response = requests.get(url, params=params, timeout=(5, 8), headers={"User-Agent": "Mozilla/5.0"})
        path.write_bytes(response.content)
        entry.update(http_status=response.status_code, available_at=now(), raw_path=str(path.resolve()), raw_sha256=sha(path))
        response.raise_for_status()
        json.loads(response.content)
        entry["status"] = "RAW_JSON_RECEIVED"
    except Exception as exc:
        entry.update(status="UNAVAILABLE", error=str(exc)[:350])
    return entry


def reconstruct(directory):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION:
        raise ValueError("unsupported raw receipt schema")
    if manifest["target_date"] != next_session(manifest["calendar"], manifest["signal_date"]):
        raise ValueError("archive target is not the exact next trading session")
    if any(not Path(path).is_file() or sha(path) != wanted for path, wanted in manifest["source_hashes"].items()):
        raise ValueError("raw archive source/code hash mismatch")
    pools, daily, unavailable = {}, {}, []
    seen = set()
    for entry in manifest["requests"]:
        key = (entry["kind"], entry.get("date"), entry.get("code"))
        if key in seen:
            raise ValueError("duplicate receipt request")
        seen.add(key)
        path = Path(entry["raw_path"]) if entry.get("raw_path") else None
        if path and (not path.is_file() or sha(path) != entry["raw_sha256"]):
            raise ValueError("raw receipt bytes missing or changed")
        if entry.get("status") != "RAW_JSON_RECEIVED":
            unavailable.append({"key": key, "reason": entry.get("error", "unavailable")})
            continue
        if stamp(entry["requested_at"]) > stamp(entry["available_at"]) or stamp(entry["available_at"]) > stamp(manifest["recorded_at"]):
            raise ValueError("invalid receipt clock")
        try:
            payload = json.loads(path.read_bytes())
            if entry["kind"] == "daily":
                expected = {"param": f"{('sh' if entry['code'].startswith('6') else 'sz')+entry['code']},day,{manifest['daily_start']},{manifest['target_date']},40,"}
                if entry["url"] != DAY_URL or entry["parameters"] != expected:
                    raise ValueError("daily request metadata changed")
                daily[entry["code"]] = {**parse_daily(payload, entry["code"], manifest["daily_start"], manifest["target_date"]),
                    "available_at": entry["available_at"], "raw_path": entry["raw_path"]}
            else:
                date, kind = entry["date"], entry["kind"]
                if kind not in ENDPOINTS or entry["url"] != EM_BASE+ENDPOINTS[kind] or entry["parameters"].get("date") != date.replace("-", ""):
                    raise ValueError("pool request metadata changed")
                if stamp(entry["requested_at"]) < stamp(date+"T15:00:00+08:00"):
                    raise ValueError("an intraday capture cannot become a closing result")
                pools[(date, kind)] = {**parse_pool(payload, date, kind), "available_at": entry["available_at"], "raw_path": entry["raw_path"]}
        except ValueError as exc:
            unavailable.append({"key": key, "reason": str(exc)})
    expected_pools = {(k, d, None) for d in [manifest["signal_date"], manifest["target_date"]] for k in ENDPOINTS}
    if not expected_pools.issubset(seen):
        raise ValueError("raw receipt archive omits a requested pool")
    expected_daily = {("daily", None, code) for code in manifest["requested_codes"]}
    if seen != expected_pools | expected_daily:
        raise ValueError("raw receipt archive changes the code denominator")
    normalized = {"pools": {d+"/"+k: value for (d,k), value in pools.items()}, "daily": daily, "unavailable": unavailable}
    saved = json.loads((directory/"normalized.json").read_text(encoding="utf-8"))
    if digest(saved) != digest(normalized):
        raise ValueError("normalized observations differ from complete raw reconstruction")
    return manifest, pools, daily, unavailable


def collect(signal_date, target_date, calendar, output, extra_codes=()):
    target = Path(output)
    if target.exists():
        raise FileExistsError("immutable source archive already exists")
    if target_date != next_session(calendar, signal_date) or stamp(now()) < stamp(target_date+"T15:00:00+08:00"):
        raise ValueError("only an exact completed target session may be collected")
    target.mkdir(parents=True)
    (target/"raw").mkdir()
    entries, pools, daily, unavailable = [], {}, {}, []
    for date in [signal_date, target_date]:
        for kind, endpoint in ENDPOINTS.items():
            params = {"ut": "7eea3edcaed734bea9cbfc24409ed989", "dpt": "wz.ztzt", "Pageindex": "0", "pagesize": "10000",
                "sort": "fbt:asc" if kind in {"limit_up", "failed_board"} else "fund:asc" if kind == "limit_down" else "zs:desc", "date": date.replace("-", "")}
            entry = fetch(EM_BASE+endpoint, params, target/"raw"/(date+"_"+kind+".json"))
            entry.update(date=date, kind=kind)
            entries.append(entry)
            try:
                if entry["status"] != "RAW_JSON_RECEIVED":
                    raise ValueError(entry.get("error", "unavailable"))
                pools[(date, kind)] = {**parse_pool(json.loads(Path(entry["raw_path"]).read_bytes()), date, kind),
                    "available_at": entry["available_at"], "raw_path": entry["raw_path"]}
            except ValueError as exc:
                unavailable.append({"key": (kind, date, None), "reason": str(exc)})
    codes = set(extra_codes)
    for pool in pools.values():
        codes.update(row["code"] for row in pool["rows"])
    codes = sorted(codes)
    if any(not isinstance(c, str) or len(c) != 6 or not c.isdigit() or not c.startswith(MAINBOARD) for c in codes):
        raise ValueError("invalid original candidate code")
    start = (datetime.strptime(signal_date, "%Y-%m-%d")-timedelta(days=25)).strftime("%Y-%m-%d")
    def one(code):
        symbol = ("sh" if code.startswith("6") else "sz")+code
        entry = fetch(DAY_URL, {"param": f"{symbol},day,{start},{target_date},40,"}, target/"raw"/(code+"_daily.json"))
        entry.update(kind="daily", code=code)
        return entry
    with ThreadPoolExecutor(max_workers=5) as executor:
        for entry in executor.map(one, codes):
            entries.append(entry)
            try:
                if entry["status"] != "RAW_JSON_RECEIVED":
                    raise ValueError(entry.get("error", "unavailable"))
                daily[entry["code"]] = {**parse_daily(json.loads(Path(entry["raw_path"]).read_bytes()), entry["code"], start, target_date),
                    "available_at": entry["available_at"], "raw_path": entry["raw_path"]}
            except ValueError as exc:
                unavailable.append({"key": ("daily", None, entry["code"]), "reason": str(exc)})
    normalized = {"pools": {d+"/"+k: value for (d,k), value in pools.items()}, "daily": daily, "unavailable": unavailable}
    write_new(target/"normalized.json", normalized)
    manifest = {"version": VERSION, "signal_date": signal_date, "target_date": target_date, "calendar": calendar,
        "daily_start": start, "requested_codes": codes, "recorded_at": now(), "requests": entries,
        "source_hashes": {str(Path(__file__).resolve()): sha(__file__)}, "original_pool_fields_not_rewritten": True,
        "historical_capture": stamp(now()).date().isoformat() > target_date,
        "raw_qdate_total_count_retained": True, "unavailable_entries": len(unavailable)}
    write_new(target/"manifest.json", manifest)
    reconstruct(target)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--calendar", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--registered", help="include all original candidates even if no returned pool contains them")
    args = parser.parse_args()
    calendar = json.loads(Path(args.calendar).read_text(encoding="utf-8"))
    codes = []
    if args.registered:
        from next_session_expectation_contract_v271 import load_registered
        registered = load_registered(args.registered)
        if (registered["plan"]["signal_date"], registered["plan"]["target_date"]) != (args.signal, args.target):
            raise ValueError("original registration belongs to a different exact target")
        codes = [c["code"] for c in registered["plan"]["candidates"]]
    result = collect(args.signal, args.target, calendar, args.output, codes)
    print(json.dumps({"requested_codes": len(result["requested_codes"]), "unavailable": result["unavailable_entries"], "status": "RAW_RECONSTRUCTION_PASSED"}))
