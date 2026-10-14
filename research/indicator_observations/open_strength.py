"""Frozen original-group 09:35 comparison, independent research observations only.

The arithmetic function is pure: provenance/units/times must be explicit. File
adapters re-read bound bytes; a declared contract is not exchange certification.
Sina's existing endpoint does not establish start/end or adjustment semantics, so
its raw adapter deliberately keeps these unknown and cannot unlock the metric.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

from next_session_expectation_contract_v271 import next_session

SHANGHAI = ZoneInfo("Asia/Shanghai")
SCHEMA = "original-group-open-strength-research"


def stamp(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone-aware timestamp required")
    return parsed.astimezone(SHANGHAI)


def now():
    return datetime.now(timezone.utc).astimezone(SHANGHAI).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code6(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{6}", value):
        raise ValueError("literal six-digit code required; no implicit normalization")
    return value


def _positive(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def freeze_groups(intake_rows, signal_date, target_date, calendar, *, frozen_at,
                  original_roster_path, original_roster_sha256, candidate_codes=None):
    """Pure group preparation. Freeze before target-day midnight; never infer peers.

    A late signal-day intake may support a NEW target-day independent observation;
    it does not alter the old forecast or create signal-day point-in-time evidence.
    """
    if target_date != next_session(calendar, signal_date):
        raise ValueError("target must be the exact next official trading session")
    frozen = stamp(frozen_at)
    if not stamp(signal_date + "T15:00:00+08:00") <= frozen < stamp(target_date + "T00:00:00+08:00"):
        raise ValueError("original group must be frozen after signal close and before target day")
    if not re.fullmatch(r"[0-9a-f]{64}", original_roster_sha256 or ""):
        raise ValueError("original roster SHA256 required")
    rows = list(intake_rows)
    by_code = {}
    for row in rows:
        code = code6(row.get("code"))
        if code in by_code or row.get("signal_date") != signal_date:
            raise ValueError("duplicate code or wrong signal date in original intake")
        boards = row.get("boards")
        if isinstance(boards, str) and re.fullmatch(r"[1-9]\d*", boards):
            boards = int(boards)
        if type(boards) is not int or boards < 1:
            raise ValueError("positive integer original board height required")
        peers = row.get("provider_same_height_codes")
        if isinstance(peers, str):
            peers = json.loads(peers)
        if not isinstance(peers, list) or len(peers) != len(set(peers)):
            raise ValueError("unique explicit original peer list required")
        peers = sorted(code6(peer) for peer in peers if peer != code)
        by_code[code] = dict(code=code, name=row.get("name"), boards=boards,
            original_peer_codes=peers)
    if not by_code:
        raise ValueError("nonempty original intake required")
    for row in by_code.values():
        expected = sorted(code for code, other in by_code.items()
            if other["boards"] == row["boards"] and code != row["code"])
        if row["original_peer_codes"] != expected:
            raise ValueError("original peer list does not preserve the whole original same-height group")
    candidates = (sorted(code for code, row in by_code.items() if row["boards"] >= 2)
                  if candidate_codes is None else list(candidate_codes))
    if not candidates or len(candidates) != len(set(candidates)) or not set(candidates) <= set(by_code):
        raise ValueError("unique original candidate codes inside intake required")
    body = dict(schema=SCHEMA, signal_date=signal_date, target_date=target_date,
        frozen_at=frozen.isoformat(), original_roster_path=str(original_roster_path),
        original_roster_sha256=original_roster_sha256,
        original_roster_count=len(by_code), original_roster_codes=sorted(by_code),
        groups=[by_code[code] for code in candidates], calendar_sha256=digest(calendar),
        scope="ORIGINAL_PROVIDER_SAME_HEIGHT_ROSTER_ONLY",
        original_forecast_changed=False, production_score_changed=False,
        late_signal_freeze=frozen.date().isoformat() != signal_date,
        entry_authorized=False, forward_prediction_samples_added=0)
    body["original_group_sha256"] = digest(body)
    return body


def freeze_intake(intake_path, signal_date, target_date, calendar, candidate_codes=None):
    """Real-clock file entry. Hashes and preserves the original CSV without editing it."""
    path = Path(intake_path).resolve()
    frozen_at = now()
    if datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) > stamp(frozen_at):
        raise ValueError("original intake was not available at freeze time")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return freeze_groups(rows, signal_date, target_date, calendar, frozen_at=frozen_at,
        original_roster_path=str(path), original_roster_sha256=file_sha(path),
        candidate_codes=candidate_codes)


def validate_frozen(frozen, calendar):
    body = {k: v for k, v in frozen.items() if k != "original_group_sha256"}
    if (frozen.get("schema") != SCHEMA or digest(body) != frozen.get("original_group_sha256")
            or frozen.get("calendar_sha256") != digest(calendar)
            or frozen.get("target_date") != next_session(calendar, frozen.get("signal_date"))):
        raise ValueError("frozen group/calendar binding changed")
    if not stamp(frozen["signal_date"] + "T15:00:00+08:00") <= stamp(frozen["frozen_at"]) < stamp(frozen["target_date"] + "T00:00:00+08:00"):
        raise ValueError("frozen group time is ineligible")


def _indexed(rows):
    result, duplicates = {}, set()
    for row in rows:
        code = row.get("code")
        if code in result:
            duplicates.add(code)
        result[code] = row
    return result, duplicates


def _value(row, kind, signal, target, cutoff):
    if row is None:
        return None, "SOURCE_MISSING"
    try:
        code6(row.get("code"))
        if row.get("adjustment") != "none" or row.get("price_unit") != "yuan":
            return None, "PRICE_UNIT_OR_ADJUSTMENT_UNKNOWN"
        if not row.get("source") or not row.get("raw_path") or not re.fullmatch(r"[0-9a-f]{64}", row.get("raw_sha256", "")):
            return None, "SOURCE_BINDING_MISSING"
        if row.get("binding_status") != "RAW_FIELDS_REBUILT":
            return None, "RAW_FIELDS_NOT_REBUILT"
        available = stamp(row["available_at"])
        if available > cutoff:
            return None, "SOURCE_NOT_AVAILABLE_BY_CUTOFF"
        if kind == "daily":
            if row.get("date") != signal or row.get("independent") is not True:
                return None, "INDEPENDENT_EXACT_SIGNAL_CLOSE_REQUIRED"
            observation = stamp(signal + "T15:00:00+08:00")
        else:
            observation = stamp(row["bar_end"])
            if observation != stamp(target + "T09:35:00+08:00"):
                return None, "EXACT_TARGET_0935_BAR_REQUIRED"
            if row.get("timestamp_side") != "end" or row.get("interval_seconds") != 60:
                return None, "MINUTE_TIMESTAMP_SEMANTICS_UNKNOWN"
        if available < observation:
            return None, "SOURCE_TIME_PRECEDES_OBSERVATION"
        value = _positive(row.get("close"))
        return (value, None) if value is not None else (None, "INVALID_PRICE")
    except (ValueError, TypeError, KeyError):
        return None, "INVALID_SOURCE_OR_TIME"


def derive_open_strength(frozen, previous_closes, minute_rows, calendar,
                         knowledge_cutoff, derived_at=None):
    """Pure arithmetic given rebuilt source rows; never changes forecasts or scores.

    `derived_at` is explicit for deterministic replay; normal callers omit it and
    receive real time. Replays remain independent research, never forward samples.
    """
    validate_frozen(frozen, calendar)
    derived = stamp(derived_at or now())
    cutoff = stamp(knowledge_cutoff)
    if cutoff > derived or derived < stamp(frozen["frozen_at"]):
        raise ValueError("derivation cannot precede cutoff or frozen original group")
    signal, target = frozen["signal_date"], frozen["target_date"]
    observed = stamp(target + "T09:35:00+08:00")
    closes, close_duplicates = _indexed(previous_closes)
    minutes, minute_duplicates = _indexed(minute_rows)
    results = []
    for group in frozen["groups"]:
        peers = group["original_peer_codes"]
        required = [group["code"], *peers]
        returns, reasons, evidence = {}, {}, {}
        for code in required:
            if cutoff < observed:
                reasons[code] = "WAIT_TARGET_OBSERVATION"
                continue
            if code in close_duplicates or code in minute_duplicates:
                reasons[code] = "DUPLICATE_CODE_OBSERVATIONS"
                continue
            close, close_error = _value(closes.get(code), "daily", signal, target, cutoff)
            price, price_error = _value(minutes.get(code), "minute", signal, target, cutoff)
            reason = close_error or price_error
            if not reason and closes[code]["source"] == minutes[code]["source"]:
                reason = "INDEPENDENT_DAILY_SOURCE_REQUIRED"
            if reason:
                reasons[code] = reason
            else:
                value = 100.0 * (price / close - 1.0)
                if math.isfinite(value):
                    returns[code] = value
                else:
                    reasons[code] = "NONFINITE_DERIVED_RETURN"
            evidence[code] = dict(previous_close=closes.get(code), minute=minutes.get(code))
        missing = [code for code in peers if code not in returns]
        own = returns.get(group["code"])
        partial = median([returns[code] for code in peers if code in returns]) if len(missing) < len(peers) else None
        complete = bool(peers) and not missing and own is not None
        full_median = partial if bool(peers) and not missing else None
        status = ("WAIT_TARGET_OBSERVATION" if cutoff < observed else
                  "OBSERVED_RESEARCH_ONLY" if complete else "EVIDENCE_UNKNOWN")
        if not peers:
            reasons["original_peer_group"] = "NO_ORIGINAL_PEERS"
        source_times = [stamp(row["available_at"]) for code in returns
            for row in (closes[code], minutes[code])]
        results.append(dict(signal_date=signal, target_date=target, code=group["code"],
            name=group.get("name"), boards=group["boards"],
            own_return_0935_pct=own,
            peer_return_0935_median_pct=full_median,
            relative_strength_0935_pp=own-full_median if complete else None,
            own_return_unit="percent", peer_return_unit="percent", relative_strength_unit="percentage_points",
            original_peer_codes=list(peers), original_peer_count=len(peers),
            observed_peer_count=len(peers)-len(missing), missing_peer_codes=missing,
            peer_coverage=(len(peers)-len(missing))/len(peers) if peers else None,
            partial_peer_median_pct_for_gap_description=partial,
            observed_at=observed.isoformat(),
            available_at=max([derived, *source_times]).isoformat() if cutoff >= observed else None,
            derived_at=derived.isoformat(), knowledge_cutoff=cutoff.isoformat(),
            derived_within_knowledge_cutoff=derived <= cutoff,
            frozen_at=frozen["frozen_at"], original_group_sha256=frozen["original_group_sha256"],
            original_roster_path=frozen["original_roster_path"], original_roster_sha256=frozen["original_roster_sha256"],
            scope=frozen["scope"], status=status, missing_reasons=reasons, source_evidence=evidence,
            source_verification="BOUND_DECLARED_RESEARCH_CONTRACT_NOT_EXCHANGE_CERTIFICATION",
            mode="POSTCLOSE_REPLAY" if derived >= stamp(target+"T15:00:00+08:00") else "INDEPENDENT_TARGET_OBSERVATION",
            original_forecast_changed=False, production_score_changed=False,
            forward_prediction_samples_added=0, entry_authorized=False))
    return results


def load_bound_observations(bundle_path):
    """Offline adapter for explicit normalized raw records with byte-bound fields.

    Bundle: {records:[{code,...,raw_path,raw_sha256,raw_record_index,available_at}]};
    each bound raw file must be a JSON list. All semantic fields are rebuilt from
    that original list, so editing a normalized price/date/unit cannot pass.
    `independent` still describes the supplied data contract, not official status.
    """
    bundle_path = Path(bundle_path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8-sig"))
    records, failures = [], []
    for ref in bundle.get("records", []):
        try:
            path = Path(ref["raw_path"])
            if not path.is_absolute() or file_sha(path) != ref["raw_sha256"]:
                raise ValueError("raw bytes/hash mismatch")
            available = stamp(ref["available_at"])
            if datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) > available:
                raise ValueError("raw file formed after declared availability")
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, list):
                raise ValueError("explicit normalized JSON raw record list required")
            index = ref["raw_record_index"]
            if type(index) is not int or not 0 <= index < len(payload):
                raise ValueError("raw record locator invalid")
            raw = payload[index]
            if raw.get("code") != ref.get("code"):
                raise ValueError("raw code differs from requested binding")
            for key in ("close", "date", "bar_end", "price_unit", "adjustment", "timestamp_side", "interval_seconds", "independent", "source"):
                if key in ref and ref[key] != raw.get(key):
                    raise ValueError("normalized field differs from raw record: " + key)
            records.append(dict(raw, raw_path=str(path.resolve()), raw_sha256=ref["raw_sha256"],
                raw_record_index=index, available_at=available.isoformat(), binding_status="RAW_FIELDS_REBUILT"))
        except (ValueError, TypeError, KeyError, IndexError, OSError) as exc:
            failures.append(dict(code=ref.get("code"), reason=str(exc), original_binding=ref))
    return {"records": records, "failures": failures, "bundle_path": str(bundle_path.resolve()),
            "bundle_sha256": file_sha(bundle_path)}


def collect_target_minutes(frozen, calendar, output, *, get=None):
    """One request per original code, real target day after 09:35, immutable raw.

    Existing Sina contract has unknown bar-side/adjustment semantics. These are
    retained explicitly; reception alone cannot create a valid 09:35 comparison.
    There are no clock/date overrides, retry loops or 403 bypasses.
    """
    import requests
    import public_minute_capture_v1 as minute

    validate_frozen(frozen, calendar)
    clock = stamp(now())
    target = frozen["target_date"]
    if clock.date().isoformat() != target or clock < stamp(target + "T09:35:00+08:00"):
        raise ValueError("collector only runs on the actual target official session after 09:35")
    source_path = Path(frozen["original_roster_path"])
    if not source_path.is_file() or file_sha(source_path) != frozen["original_roster_sha256"]:
        raise ValueError("original roster bytes changed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    get = get or requests.get
    codes = sorted({code for group in frozen["groups"] for code in [group["code"], *group["original_peer_codes"]]})
    def fetch_one(code):
        observation = None
        params = {"symbol": ("sh" if code.startswith("6") else "sz") + code,
                  "scale": "1", "ma": "no", "datalen": "240"}
        entry = dict(code=code, parameters=params, requested_at=now(),
            url=minute.SINA_MINUTE_URL, retry_count=0, timeout_seconds=[5, 8])
        try:
            response = get(minute.SINA_MINUTE_URL, params=params,
                headers=minute.SINA_HEADERS, timeout=(5, 8))
            raw_path = output / (code + ".response")
            raw_path.write_bytes(response.content)
            entry.update(raw_path=str(raw_path.resolve()), raw_sha256=file_sha(raw_path),
                available_at=now(), http_status=response.status_code)
            response.raise_for_status()
            # The old parser drops duplicate timestamps; inspect raw rows first.
            payload = minute._extract_json_array(response.content.decode("utf-8", errors="replace"))
            returned = sorted({str(row.get("day", ""))[:10] for row in payload})
            exact = [row for row in payload if row.get("day") == target + " 09:35:00"]
            entry.update(returned_dates=returned, returned_row_count=len(payload), exact_0935_count=len(exact))
            if (not returned or returned[-1] != target or any(not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) for day in returned)
                    or len(exact) != 1):
                raise ValueError("latest response date must be exact target with unique 09:35 raw row")
            entry["status"] = "RAW_RECEIVED_TIMESTAMP_AND_ADJUSTMENT_SEMANTICS_UNKNOWN"
            observation = dict(code=code, raw_timestamp=exact[0]["day"], close=exact[0].get("close"),
                raw_path=entry["raw_path"], raw_sha256=entry["raw_sha256"],
                available_at=entry["available_at"], source="Sina_public_1min",
                timestamp_side="unknown", adjustment="unknown", price_unit="unknown",
                binding_status="RAW_FIELDS_REBUILT", status="EVIDENCE_UNKNOWN")
        except Exception as exc:
            entry.update(status="UNAVAILABLE", error=str(exc)[:350], failure_at=now())
        return entry, observation
    # Each job is one bounded request. executor.map retains original code order.
    with ThreadPoolExecutor(max_workers=min(6, len(codes))) as executor:
        fetched = list(executor.map(fetch_one, codes))
    entries = [entry for entry, _ in fetched]
    observations = [observation for _, observation in fetched if observation is not None]
    manifest = dict(schema=SCHEMA, status="INDEPENDENT_RESEARCH_RAW_CAPTURE_ONLY",
        signal_date=frozen["signal_date"], target_date=target,
        started_at=clock.isoformat(), formed_at=now(), requested_codes=codes,
        original_group_sha256=frozen["original_group_sha256"],
        original_roster_sha256=frozen["original_roster_sha256"], requests=entries,
        observations=observations, original_forecast_changed=False,
        production_score_changed=False, forward_prediction_samples_added=0,
        entry_authorized=False, timestamp_and_adjustment_verified=False,
        request_workers=min(6, len(codes)))
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return manifest
