"""Prospective signal-day research inputs, separate from target-day settlement.

Only the actual signal session, after close and by the fixed 16:30 Shanghai
cutoff, may start collection. No target-day request, model, label, or security
verification is supplied by this entry. Raw failures and the prior roster stay
visible. Existing V293 settlement and pinned input adapters remain unchanged.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

import prepare_theme_competition_inputs as preparation
import public_outcome_sources_v293 as sources

SHANGHAI = timezone(timedelta(hours=8))
QUOTE_URL = "https://push2.eastmoney.com/api/qt/stock/get"
QUOTE_FIELDS = "f43,f44,f45,f46,f47,f48,f51,f52,f57,f58,f59,f60,f86"


def now():
    return datetime.now(timezone.utc).isoformat()


def stamp(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid timestamp") from exc
    if result.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return result


def validate_window(signal_date, calendar, clock):
    try:
        signal = datetime.strptime(signal_date, "%Y-%m-%d").date()
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid exact signal date") from exc
    sessions = calendar.get("sessions", [])
    scope = calendar.get("scope", [])
    if (calendar.get("status") != "verified" or not calendar.get("source")
            or len(scope) != 2 or not scope[0] <= signal_date <= scope[1]
            or sessions != sorted(set(sessions)) or signal_date not in sessions):
        raise ValueError("verified exact-date calendar required")
    next_date = sources.next_session(calendar, signal_date)
    if not signal_date < next_date <= scope[1]:
        raise ValueError("next official session outside verified scope")
    local = stamp(clock).astimezone(SHANGHAI)
    close = datetime.combine(signal, datetime.min.time(), SHANGHAI) + timedelta(hours=15)
    cutoff = close + timedelta(hours=1, minutes=30)
    if local.date() != signal or not close <= local <= cutoff:
        raise ValueError("actual signal-day close-to-16:30 window required; no backdating")
    return cutoff.isoformat()


def parse_quote(payload, code, signal_date, available_at):
    data = payload.get("data") if isinstance(payload, dict) else None
    if (not isinstance(data, dict) or data.get("f57") != code
            or not isinstance(data.get("f58"), str) or not data["f58"].strip()):
        raise ValueError("quote code/name mismatch or missing")
    if data.get("f59") != 2:
        raise ValueError("expected two-decimal native CNY quote precision")
    try:
        epoch = data["f86"]
        if type(epoch) is not int or epoch <= 0:
            raise ValueError("invalid quote timestamp")
        observed = datetime.fromtimestamp(epoch, SHANGHAI)
        available = stamp(available_at)
        if (observed.date().isoformat() != signal_date
                or observed < stamp(signal_date + "T15:00:00+08:00")
                or observed > available):
            raise ValueError("quote is not an already available signal-day close")
        # fltt=2 returns native decimal CNY, unlike the scaled pool fields.
        values = {k: sources.cent(data[field]) for k, field in
                  {"open": "f46", "close": "f43", "high": "f44",
                   "low": "f45", "previous_close": "f60"}.items()}
        upper, lower = sources.cent(data["f51"]), sources.cent(data["f52"])
    except (KeyError, TypeError, OverflowError, OSError) as exc:
        raise ValueError("missing or invalid quote field") from exc
    if (not lower < upper or not lower <= values["low"] <= values["high"] <= upper
            or not values["low"] <= values["open"] <= values["high"]
            or not values["low"] <= values["close"] <= values["high"]):
        raise ValueError("quote OHLC/explicit price limits inconsistent")
    return {"code": code, "name": data["f58"], "observed_at": observed.isoformat(),
            "available_at": available.isoformat(), "values_cents": values,
            "upper_cents": upper, "lower_cents": lower,
            "price_units": "native_CNY_fltt_2", "security_status_verified": False}


def reconcile_quote_daily(quote, daily, signal_date):
    if daily.get("code") != quote["code"] or daily.get("adjusted") is not False:
        raise ValueError("independent exact-code unadjusted daily prices required")
    day = daily.get("days", {}).get(signal_date)
    if not day:
        raise ValueError("signal-day unadjusted daily row missing")
    for field in ("open", "close", "high", "low"):
        if day[field] != quote["values_cents"][field]:
            raise ValueError("independent daily/quote price conflict: " + field)
    snapshot = daily.get("quote", {})
    positional_match = None
    snapshot_time_usable = False
    if snapshot.get("status") == "POSITIONAL_FIELDS_INFERRED" and daily.get("available_at"):
        snapshot_time = stamp(snapshot["observed_at"]).astimezone(SHANGHAI)
        snapshot_time_usable = (snapshot_time.date().isoformat() == signal_date
            and stamp(signal_date + "T15:00:00+08:00") <= snapshot_time <= stamp(daily["available_at"]))
    if snapshot_time_usable:
        inferred = snapshot["values"]
        for field in ("open", "close", "high", "low", "previous_close"):
            if inferred[field] != quote["values_cents"][field]:
                raise ValueError("Tencent snapshot/explicit quote conflict: " + field)
        if inferred["upper_candidate"] != quote["upper_cents"] or inferred["lower_candidate"] != quote["lower_cents"]:
            raise ValueError("inferred and explicit price limits conflict")
        positional_match = True
    return {"code": quote["code"], "independent_unadjusted_ohlc_match": True,
            "positional_limit_match": positional_match,
            "positional_snapshot_time_usable": snapshot_time_usable,
            "descriptive_close_observation": ("AT_EXPLICIT_REPORTED_UPPER" if
                quote["values_cents"]["close"] == quote["upper_cents"] else "BELOW_EXPLICIT_REPORTED_UPPER"),
            "exchange_limit_price_verified": False, "promotion_label": None}


def read_prior_roster(path, expected_sha256, calendar, signal_date, clock):
    if not path:
        if expected_sha256:
            raise ValueError("prior roster path/hash must be paired")
        return [], {}
    path = Path(path).resolve()
    if not expected_sha256 or sources.sha(path) != expected_sha256:
        raise ValueError("prior roster source hash mismatch")
    if datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) > stamp(clock):
        raise ValueError("prior roster actual file formation follows capture")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    previous = [d for d in calendar["sessions"] if d < signal_date]
    if not rows or not previous:
        raise ValueError("nonempty prior-session roster required")
    seen = set()
    for row in rows:
        code = row.get("code", "")
        if (len(code) != 6 or not code.isdigit() or not code.startswith(sources.MAINBOARD)
                or code in seen or row.get("original_pool_date") != previous[-1]
                or row.get("target_date") != signal_date
                or not row.get("name") or int(row.get("original_boards", 0)) < 1):
            raise ValueError("invalid/duplicate code or wrong original session in prior roster")
        if not row.get("roster_derived_at") or stamp(row["roster_derived_at"]) > stamp(clock):
            raise ValueError("prior roster actual derived time missing or follows capture")
        if not row.get("original_available_at") or stamp(row["original_available_at"]) > stamp(clock):
            raise ValueError("original prior observation not yet available")
        seen.add(code)
    return rows, {str(path): expected_sha256}


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def verify_archive(directory, require_by_cutoff=False):
    """Verify receipts before research use; this never grants training eligibility."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    formation = json.loads((directory/"formation_receipt.json").read_text(encoding="utf-8"))
    if (manifest.get("artifact_kind") != "SIGNAL_DAY_RESEARCH_CAPTURE"
            or formation.get("manifest_sha256") != sources.sha(directory/"manifest.json")
            or formation.get("prediction_use_allowed") is not False
            or manifest.get("v151_ready") is not False):
        raise ValueError("research-only archive/formation binding invalid")
    for path, expected in manifest["source_hashes"].items():
        if sources.sha(path) != expected:
            raise ValueError("source or program changed: " + path)
    for name, expected in manifest["artifact_hashes"].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or sources.sha(path) != expected:
            raise ValueError("bound artifact changed or escaped archive: " + name)
    requested = manifest["requested_codes"]
    if requested != sorted(set(requested)):
        raise ValueError("duplicate or unordered requested codes")
    expected_keys = {(kind, None) for kind in sources.ENDPOINTS}
    expected_keys.update((kind, code) for kind in ("daily", "quote") for code in requested)
    keys = [(entry["kind"], entry.get("code")) for entry in manifest["requests"]]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ValueError("original request scope or failures missing")
    for entry in manifest["requests"]:
        if entry.get("raw_path"):
            path = Path(entry["raw_path"]).resolve()
            if not path.is_relative_to(directory/"raw") or sources.sha(path) != entry["raw_sha256"]:
                raise ValueError("raw receipt hash/path mismatch")
    normalized = json.loads((directory/"normalized.json").read_text(encoding="utf-8"))
    if sorted(normalized["observations"]) != requested:
        raise ValueError("unknown or failed requested observation removed")
    prior_path = directory/"prior_roster.source.csv"
    prior = []
    if prior_path.exists():
        with prior_path.open(encoding="utf-8-sig", newline="") as handle:
            prior = list(csv.DictReader(handle))
        with (directory/"prior_roster_observations.csv").open(encoding="utf-8-sig", newline="") as handle:
            observed = list(csv.DictReader(handle))
        if ([r["code"] for r in prior] != [r["code"] for r in observed]
                or len(prior) != manifest["prior_roster_rows"]):
            raise ValueError("full original roster was not retained")
    cutoff = stamp(manifest["knowledge_cutoff"])
    actual_last_file = max(datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
                           for p in directory.rglob("*") if p.is_file())
    by_cutoff = (manifest["prepared_by_cutoff"] and formation["prepared_by_cutoff"]
                 and stamp(formation["recorded_at"]) <= cutoff and actual_last_file <= cutoff)
    if require_by_cutoff and not by_cutoff:
        raise ValueError("complete archive was not actually formed by the fixed cutoff")
    return {"status": "BOUND_RESEARCH_ARCHIVE_VERIFIED", "prepared_by_cutoff": bool(by_cutoff),
            "prior_roster_rows": len(prior), "requested_codes": len(requested), "v151_ready": False}


def collect(signal_date, calendar_path, protocol_path, output, prior_roster=None,
            prior_roster_sha256=None, theme_evidence=None, theme_bindings=None):
    started = now()  # CLI deliberately has no clock override.
    calendar_path, protocol_path, output = map(Path, (calendar_path, protocol_path, output))
    calendar = json.loads(calendar_path.read_text(encoding="utf-8"))
    cutoff = stamp(validate_window(signal_date, calendar, started))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (protocol.get("prospective_information_mode") != "B_FIXED_EVENING"
            or protocol.get("prospective_knowledge_cutoff_local") != "16:30:00 Asia/Shanghai"
            or stamp(protocol["registered_at"]) >= stamp(signal_date + "T09:15:00+08:00")):
        raise ValueError("fixed evening protocol must be registered before signal session")
    if bool(theme_evidence) != bool(theme_bindings):
        raise ValueError("reviewed theme evidence/bindings must be paired")
    previous, bindings = read_prior_roster(prior_roster, prior_roster_sha256, calendar, signal_date, started)
    bindings.update({str(p.resolve()): sources.sha(p) for p in (calendar_path, protocol_path)})
    evidence = None
    if theme_evidence:
        evidence, theme_hashes = preparation.validate_theme_sources(
            theme_evidence, theme_bindings, signal_date, cutoff, stamp(now()))
        bindings.update(theme_hashes)
    if output.exists():
        raise FileExistsError("refusing to overwrite research capture archive")
    validate_window(signal_date, calendar, now())
    output.mkdir(parents=True)
    (output/"raw").mkdir()
    if prior_roster:
        (output/"prior_roster.source.csv").write_bytes(Path(prior_roster).read_bytes())
        if sources.sha(output/"prior_roster.source.csv") != prior_roster_sha256:
            raise ValueError("prior roster changed after preflight; raw output retained")
    entries, pools, failures = [], {}, []
    forbidden_hosts, host_lock = set(), Lock()

    def fetch(url, parameters, path, **identity):
        if stamp(now()) > cutoff:
            return {"url": url, "parameters": parameters, "requested_at": None,
                    "status": "NOT_REQUESTED_CUTOFF_PASSED", **identity}
        host = urlsplit(url).netloc
        with host_lock:
            if host in forbidden_hosts:
                return {"url": url, "parameters": parameters, "requested_at": None,
                        "status": "NOT_REQUESTED_SOURCE_FORBIDDEN", **identity}
        entry = sources.fetch(url, parameters, path)
        if entry.get("http_status") == 403:
            with host_lock:
                forbidden_hosts.add(host)
        return {**entry, **identity}

    def payload(entry):
        if entry["status"] != "RAW_JSON_RECEIVED":
            raise ValueError(entry.get("error", entry["status"]))
        if stamp(entry["available_at"]) > cutoff:
            raise ValueError("response arrived after fixed cutoff")
        raw = Path(entry["raw_path"])
        if sources.sha(raw) != entry["raw_sha256"]:
            raise ValueError("raw receipt hash changed")
        def invalid_constant(value):
            raise ValueError("nonfinite JSON constant: " + value)
        return json.loads(raw.read_bytes(), parse_constant=invalid_constant)

    for kind, endpoint in sources.ENDPOINTS.items():
        parameters = {"ut": "7eea3edcaed734bea9cbfc24409ed989", "dpt": "wz.ztzt", "Pageindex": "0", "pagesize": "10000",
                      "sort": "fbt:asc" if kind in {"limit_up", "failed_board"} else "fund:asc" if kind == "limit_down" else "zs:desc",
                      "date": signal_date.replace("-", "")}
        entry = fetch(sources.EM_BASE+endpoint, parameters, output/"raw"/(kind+".json"), kind=kind, date=signal_date)
        entries.append(entry)
        try:
            pools[kind] = {**sources.parse_pool(payload(entry), signal_date, kind), "available_at": entry["available_at"]}
        except (ValueError, KeyError, TypeError) as exc:
            failures.append({"kind": kind, "code": None, "reason": str(exc)})
    codes = {row["code"] for row in previous}
    codes.update(row["code"] for pool in pools.values() for row in pool["rows"])
    codes = sorted(codes)
    start = (datetime.strptime(signal_date, "%Y-%m-%d")-timedelta(days=25)).strftime("%Y-%m-%d")

    def one(code):
        symbol = ("sh" if code.startswith("6") else "sz")+code
        day_entry = fetch(sources.DAY_URL, {"param": f"{symbol},day,{start},{signal_date},40,"},
                          output/"raw"/(code+"_daily.json"), kind="daily", code=code)
        quote_entry = fetch(QUOTE_URL, {"secid": ("1." if code.startswith("6") else "0.")+code,
                         "fields": QUOTE_FIELDS, "fltt": "2", "invt": "2"},
                         output/"raw"/(code+"_quote.json"), kind="quote", code=code)
        observations, errors = {"code": code, "status": "UNKNOWN", "promotion_label": None}, []
        daily = quote = None
        for kind, entry in (("daily", day_entry), ("quote", quote_entry)):
            try:
                raw = payload(entry)
                if kind == "daily":
                    block = (raw.get("data") or {}).get(symbol, {})
                    if any(str(row[0]) > signal_date for row in block.get("day", [])):
                        raise ValueError("future daily row in signal-only response")
                    daily = {**sources.parse_daily(raw, code, start, signal_date),
                             "available_at": entry["available_at"]}
                else:
                    quote = parse_quote(raw, code, signal_date, entry["available_at"])
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                errors.append({"kind": kind, "code": code, "reason": str(exc)})
        if daily is not None and quote is not None:
            try:
                observations.update(reconcile_quote_daily(quote, daily, signal_date))
                observations.update(status="DESCRIPTIVE_TWO_PROVIDER_PRICE_MATCH", quote=quote,
                                    available_at=max(day_entry["available_at"], quote_entry["available_at"]))
            except (ValueError, KeyError, TypeError) as exc:
                errors.append({"kind": "price_reconciliation", "code": code, "reason": str(exc)})
        return [day_entry, quote_entry], observations, errors

    observations = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for receipts, observed, errors in executor.map(one, codes):
            entries.extend(receipts)
            observations[observed["code"]] = observed
            failures.extend(errors)
    closing = pools.get("limit_up")
    closing_codes = {row["code"] for row in closing["rows"]} if closing else set()
    prior_observations = []
    for row in previous:
        observation = observations[row["code"]]
        quote = observation.get("quote", {})
        prior_observations.append({**row, "observation_signal_date": signal_date,
            "current_provider_closing_pool_presence": row["code"] in closing_codes if closing else None,
            "price_observation_status": observation["status"],
            "close_cents": quote.get("values_cents", {}).get("close"), "upper_cents": quote.get("upper_cents"),
            "descriptive_close_observation": observation.get("descriptive_close_observation"),
            "current_promotion_label": None, "current_theme_status": "UNKNOWN_UNLESS_SEPARATELY_REVIEWED",
            "absence_from_pool_is_failure": False, "observation_available_at": observation.get("available_at")})
    if prior_observations:
        with (output/"prior_roster_observations.csv").open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prior_observations[0]))
            writer.writeheader()
            writer.writerows(prior_observations)
    prepared = now()
    if closing:
        try:
            values = preparation.frames(closing, signal_date, prepared, cutoff, evidence=evidence)
            pool_frame, pm, roles, tm, intake, competition, cm, factors = values
            for name, frame in [("closing_pool.csv", pool_frame), ("candidate_theme_roles.csv", roles),
                    ("candidate_theme_intake.csv", intake), ("candidate_competition_snapshot.csv", competition),
                    ("candidate_theme_competition.csv", factors)]:
                frame.to_csv(output/name, index=False, encoding="utf-8-sig")
            for name, value in (("closing_pool_manifest.json", pm), ("theme_manifest.json", tm),
                                ("competition_manifest.json", cm)):
                dump(output/name, value)
        except (ValueError, KeyError, TypeError) as exc:
            failures.append({"kind": "input_preparation", "code": None, "reason": str(exc)})
    dump(output/"normalized.json", {"pools": pools, "observations": observations, "failures": failures})
    dump(output/"specific_theme_evidence.draft.json", {"schema_version": preparation.roles_v150.VERSION,
         "signal_date": signal_date, "groups": [], "review_status": "UNREVIEWED"})
    dump(output/"theme_source_bindings.draft.json", {"signal_date": signal_date, "recorded_at": None,
         "reviewed_at": None, "review_status": "UNREVIEWED", "reviewer": None,
         "review_notes": None, "evidence_sha256": None, "groups": {}})
    bindings.update({str(Path(module.__file__).resolve()): sources.sha(module.__file__)
                     for module in (sources, preparation, preparation.roles_v150,
                                    preparation.competition_v114, preparation.archive_v151)})
    bindings[str(Path(sources.next_session.__code__.co_filename).resolve())] = sources.sha(sources.next_session.__code__.co_filename)
    bindings[str(Path(__file__).resolve())] = sources.sha(__file__)
    artifact_hashes = {str(p.relative_to(output)): sources.sha(p) for p in output.rglob("*") if p.is_file()}
    completed = now()
    result = {"artifact_kind": "SIGNAL_DAY_RESEARCH_CAPTURE", "capture_started_at": started,
        "recorded_at": completed, "available_at": completed, "signal_date": signal_date,
        "target_date": sources.next_session(calendar, signal_date), "calendar": calendar,
        "knowledge_cutoff": cutoff.isoformat(), "prepared_by_cutoff": stamp(completed) <= cutoff,
        "status": "PREPARED_WITH_GAPS" if stamp(completed) <= cutoff else "LATE_ARCHIVE_NOT_PIT_ELIGIBLE",
        "requests": entries, "requested_codes": codes, "prior_roster_rows": len(previous),
        "prior_roster_retained_rows": len(prior_observations), "prior_roster_retention": 1.0 if previous else None,
        "provider_closing_rows": len(closing_codes) if closing else None,
        "descriptive_price_matched_rows": sum(r["status"] != "UNKNOWN" for r in observations.values()),
        "failures": failures, "source_hashes": bindings,
        "artifact_hashes": artifact_hashes,
        "official_full_market": False, "v151_ready": False, "entry_authorized": False,
        "new_settled_forward_samples": 0, "performance_experiment_started": False,
        "model_upgraded": False, "old_prediction_modified": False,
        "blockers": ["exchange security/risk status unverified", "exchange-wide completeness unverified",
                     "explicit cap remains vendor-reported; Tencent limit fields inferred",
                     "reviewed PIT themes and baseline scores require separate binding"],
        "not_a_v293_settlement_archive": True,
        "formation_receipt_required": True}
    dump(output/"manifest.json", result)
    # This separate final receipt is authoritative when hashing/writing crosses
    # the cutoff. A consumer must also verify actual mtimes of every bound file.
    formed = now()
    actual_last_file = max(datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
                           for p in output.rglob("*") if p.is_file())
    by_cutoff = stamp(formed) <= cutoff and actual_last_file <= cutoff
    formation = {"recorded_at": formed, "knowledge_cutoff": cutoff.isoformat(),
        "manifest_sha256": sources.sha(output/"manifest.json"),
        "prepared_by_cutoff": by_cutoff, "prediction_use_allowed": False,
        "status": "RESEARCH_ONLY_WITH_GAPS" if by_cutoff else "LATE_ARCHIVE_NOT_PIT_ELIGIBLE",
        "requires_all_actual_file_mtimes_by_cutoff": True}
    dump(output/"formation_receipt.json", formation)
    result = {**result, "prepared_by_cutoff": by_cutoff, "status": formation["status"],
              "formation_receipt_sha256": sources.sha(output/"formation_receipt.json")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("signal", "calendar", "protocol", "output"):
        parser.add_argument("--"+name, required=True)
    for name in ("prior-roster", "prior-roster-sha256", "theme-evidence", "theme-bindings"):
        parser.add_argument("--"+name)
    args = parser.parse_args()
    result = collect(args.signal, args.calendar, args.protocol, args.output, args.prior_roster,
                     args.prior_roster_sha256, args.theme_evidence, args.theme_bindings)
    print(json.dumps({k: result[k] for k in ("status", "prepared_by_cutoff", "prior_roster_rows",
          "prior_roster_retained_rows", "provider_closing_rows", "descriptive_price_matched_rows", "v151_ready")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
