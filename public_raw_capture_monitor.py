"""Add raw-response receipts to the original six-point public monitor.

The original monitor, quote/minute parsers and decision rules remain unchanged.
This entry records public evidence only; no orders, inferred fills or training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from threading import Lock
from uuid import uuid4

import requests

import public_live_capture_v1 as quotes
import public_minute_capture_v1 as minutes
import public_live_monitor_v1 as monitor


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def field_references(content, kind, params):
    """Pointers refer to original byte spans and provider field positions."""
    result = []
    if kind == "quote":
        pattern = rb'var\s+hq_str_([a-z]{2}\d{6})="([^"]*)";'
        fields = {"provider_name": 0, "open": 1, "previous_close": 2,
                  "last": 3, "high": 4, "low": 5, "bid": 6, "ask": 7,
                  "volume_shares": 8, "amount": 9, "date": 30, "time": 31}
        for match in re.finditer(pattern, content):
            values = match.group(2).split(b",")
            if len(values) < 33:
                continue
            refs = {}
            for name, index in fields.items():
                offset = match.start(2) + sum(len(v)+1 for v in values[:index])
                refs[name] = {"byte_offset": offset, "byte_length": len(values[index]),
                              "provider_field_index": index,
                              "value": values[index].decode("gb18030", errors="replace")}
            result.append({"code": match.group(1).decode("ascii")[-6:],
                           "symbol": match.group(1).decode("ascii"), "fields": refs,
                           "observed_at_native": refs["date"]["value"]+" "+refs["time"]["value"]})
    elif kind == "minute":
        match = re.search(rb"var\s+_data=\((\[.*?\])\);", content, flags=re.S)
        if match:
            payload = json.loads(match.group(1).decode("utf-8"))
            if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
                raise ValueError("native minute array must contain objects")
            symbol = (params or {}).get("symbol", "")
            for index, item in enumerate(payload):
                result.append({"code": symbol[-6:], "symbol": symbol,
                    "observed_at_native": item.get("day"),
                    "array_byte_offset": match.start(1), "array_byte_length": len(match.group(1)),
                    "fields": {field: {"array_json_pointer": f"/{index}/{field}", "value": value}
                               for field, value in item.items()}})
    else:
        raise ValueError("quote or minute response kind required")
    return result


class RecordingGet:
    """Record every actual request, including HTTP and transport failures."""
    def __init__(self, directory, kind, get=requests.get):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.kind, self.get = kind, get
        self.entries, self.lock = [], Lock()

    def __call__(self, url, **kwargs):
        entry = {"request_id": uuid4().hex, "url": url,
                 "parameters": kwargs.get("params"), "requested_at": now(),
                 "timeout": kwargs.get("timeout"), "kind": self.kind}
        try:
            response = self.get(url, **kwargs)
            entry["received_at"] = now()
            content = response.content
            raw = self.directory/(entry["request_id"]+".bin")
            with raw.open("xb") as handle:
                handle.write(content)
            entry.update(raw_path=str(raw.resolve()), raw_sha256=sha(raw),
                         http_status=getattr(response, "status_code", None))
            try:
                entry["field_references"] = field_references(content, self.kind, kwargs.get("params"))
            except (ValueError, TypeError) as exc:
                entry.update(field_references=[], pointer_error=str(exc))
            entry["status"] = "RESPONSE_BYTES_SAVED_NOT_MARKET_CERTIFIED"
            return response
        except Exception as exc:
            entry.update(received_at=now(), status="TRANSPORT_OR_ARCHIVE_FAILURE", error=str(exc)[:500])
            raise
        finally:
            receipt = self.directory/(entry["request_id"]+".json")
            write_new(receipt, entry)
            with self.lock:
                self.entries.append({"receipt_path": str(receipt.resolve()), "receipt_sha256": sha(receipt)})


def verify_receipts(entries):
    for ref in entries:
        path = Path(ref["receipt_path"])
        if sha(path) != ref["receipt_sha256"]:
            raise ValueError("raw request receipt hash mismatch")
        item = json.loads(path.read_text(encoding="utf-8"))
        if not datetime.fromisoformat(item["requested_at"]) <= datetime.fromisoformat(item["received_at"]) <= datetime.now(timezone.utc):
            raise ValueError("raw request and receive clocks reversed or future")
        if item.get("raw_path"):
            raw = Path(item["raw_path"])
            if sha(raw) != item["raw_sha256"]:
                raise ValueError("raw response bytes changed")
            try:
                expected = field_references(raw.read_bytes(), item["kind"], item.get("parameters"))
            except (ValueError, TypeError) as exc:
                if item.get("pointer_error") != str(exc) or item.get("field_references") != []:
                    raise ValueError("malformed raw response must retain its pointer failure") from exc
                continue
            if expected != item.get("field_references"):
                raise ValueError("field pointers differ from original response bytes")
    return True


def verify_capture_link(link):
    """The native parsed capture must bind the same saved response bodies."""
    verify_receipts(link["request_receipts"])
    if not link.get("native_manifest"):
        return
    path = Path(link["native_manifest"])
    if sha(path) != link["native_sha256"]:
        raise ValueError("native capture manifest changed")
    native = json.loads(path.read_text(encoding="utf-8"))
    expected = native["provider_response_hashes"]
    expected = set(expected.values()) if isinstance(expected, dict) else set(expected)
    saved = set()
    for ref in link["request_receipts"]:
        item = json.loads(Path(ref["receipt_path"]).read_text(encoding="utf-8"))
        if item.get("raw_path"):
            saved.add(item["raw_sha256"])
    if not expected.issubset(saved):
        raise ValueError("native parsed capture has no matching original response bytes")


def validate_pool(candidate_path, calendar_path, trade_date):
    calendar = json.loads(Path(calendar_path).read_text(encoding="utf-8"))
    sessions = calendar["sessions"]
    if calendar.get("status") != "verified" or not calendar.get("source") or sessions != sorted(set(sessions)):
        raise ValueError("verified unique sorted exchange calendar required")
    if trade_date not in sessions or sessions.index(trade_date) == 0:
        raise ValueError("target session and its official predecessor required")
    pool = quotes.load_candidates(Path(candidate_path))
    if pool.empty or set(pool["signal_date"]) != {sessions[sessions.index(trade_date)-1]}:
        raise ValueError("current complete pool must belong to the exact official predecessor")
    # The native reader filters by date and security scope. Reject a file that
    # would silently drop rows instead of pretending the reduced pool is full.
    import pandas as pd
    raw = quotes._rename_candidate_columns(pd.read_csv(candidate_path, dtype={"code": "string", "代码": "string"}))
    if len(raw) != len(pool) or set(raw["code"].map(quotes.code6)) != set(pool["code"]):
        raise ValueError("capture cannot silently reduce the supplied candidate denominator")
    return pool, calendar


def run(candidate_path, calendar_path, trade_date, output):
    pool, calendar = validate_pool(candidate_path, calendar_path, trade_date)
    if datetime.now(quotes.SHANGHAI).date().isoformat() != trade_date:
        raise ValueError("live capture is only for the actual current exchange date")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out/"inputs").mkdir()
    calendar_copy = out/"inputs/calendar.json"
    with calendar_copy.open("xb") as handle:
        handle.write(Path(calendar_path).read_bytes())
    indexes = []
    capture_links = []

    def finish(recorder, result, kind, point):
        link = {"kind": kind, "asof": point, "request_receipts": recorder.entries,
                "native_manifest": result.get("manifest_path") if result else None,
                "native_sha256": sha(result["manifest_path"]) if result else None}
        indexes.extend(recorder.entries)
        capture_links.append(link)
        verify_capture_link(link)

    def capture_quote(candidates, date, point, ignored_output, ignored_universe):
        recorder = RecordingGet(out/"raw"/("quote_"+point.replace(":", "")), "quote")
        result = None
        try:
            result = quotes.capture(candidates, date, point, out/"quote", None,
                quote_fetcher=lambda codes: quotes.fetch_sina_quotes(codes, get=recorder, retries=1))
            return result
        finally:
            finish(recorder, result, "quote", point)

    def capture_minute(candidates, date, point, ignored_output):
        recorder = RecordingGet(out/"raw"/("minute_"+point.replace(":", "")), "minute")
        result = None
        try:
            result = minutes.capture(candidates, date, point, out/"minute",
                fetcher=lambda codes: minutes.fetch_minute_bars(codes, get=recorder, retries=1))
            return result
        finally:
            finish(recorder, result, "minute", point)

    result = monitor.run_monitor(Path(candidate_path), trade_date, out/"monitor", None,
        capture_runner=capture_quote, minute_runner=capture_minute,
        absorption_registry_root=None)
    verify_receipts(indexes)
    native_check = monitor.verify_run(Path(result["manifest_path"]))
    if native_check["status"] != "PASS":
        raise ValueError("original monitor verification failed: "+str(native_check["errors"]))
    value = {"version": "public-raw-capture-adapter-1", "formed_at": now(),
        "status": result["status"], "signal_date": str(pool["signal_date"].iloc[0]),
        "trade_date": trade_date, "candidate_count": len(pool),
        "registered_points": result["registered_points"],
        "completed_captures": result["completed_captures"], "skipped_captures": result["skipped_captures"],
        "monitor_manifest": result["manifest_path"], "monitor_sha256": sha(result["manifest_path"]),
        "raw_request_receipts": indexes, "raw_request_count": len(indexes),
        "capture_links": capture_links,
        "source_bindings": {str(Path(p).resolve()): sha(p) for p in
            (calendar_copy, result["candidate_path"], __file__, quotes.__file__, minutes.__file__, monitor.__file__)},
        "training_or_role_truth_authorized": False, "direct_entry_authorized": False,
        "new_orders_or_fills": 0, "source_authenticity_independently_certified": False,
        "limits": ["Public Level-1 and minute bars are not tick order, Level-2 queue or execution evidence.",
                   "Local clocks, hashes and saved bodies are source bindings, not independent vendor authentication.",
                   "A skipped/failed window stays unknown; observation has not been converted into a paper fill."]}
    write_new(out/"raw_capture_manifest.json", value)
    return value


def verify(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("version") != "public-raw-capture-adapter-1" or value.get("direct_entry_authorized") is not False:
        raise ValueError("raw monitor schema or execution authority changed")
    for p, digest in value["source_bindings"].items():
        if sha(p) != digest:
            raise ValueError("raw adapter source binding changed")
    if sha(value["monitor_manifest"]) != value["monitor_sha256"]:
        raise ValueError("original monitor manifest changed")
    if monitor.verify_run(Path(value["monitor_manifest"]))["status"] != "PASS":
        raise ValueError("original monitor event chain changed")
    native = json.loads(Path(value["monitor_manifest"]).read_text(encoding="utf-8"))
    for field in ("registered_points", "completed_captures", "skipped_captures", "trade_date"):
        if value[field] != native[field]:
            raise ValueError("raw adapter cannot change original monitor counts or session")
    pool = quotes.load_candidates(Path(native["candidate_path"]))
    if value["candidate_count"] != len(pool) or value["raw_request_count"] != len(value["raw_request_receipts"]):
        raise ValueError("raw adapter cannot change candidate or request denominator")
    verify_receipts(value["raw_request_receipts"])
    for link in value["capture_links"]:
        verify_capture_link(link)
    return {"status": "PASS", "raw_requests": value["raw_request_count"],
            "candidate_count": value["candidate_count"], "completed_captures": value["completed_captures"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify")
    parser.add_argument("--candidates")
    parser.add_argument("--calendar")
    parser.add_argument("--trade-date")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify)
    else:
        if not all((args.candidates, args.calendar, args.trade_date, args.output)):
            parser.error("--candidates, --calendar, --trade-date and --output are required")
        result = run(args.candidates, args.calendar, args.trade_date, args.output)
        result = {k: result[k] for k in ("status", "candidate_count", "completed_captures", "skipped_captures", "raw_request_count")}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
