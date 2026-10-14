"""Source-bound seal observations for research, without labels or model changes.

``compute_seal_quality`` is pure: source bytes travel with their references and
are hashed and re-read before values are accepted. The offline V275 adapter
reads original pool records, not its promotion labels or reconstructed clocks.
Unknown/later archives may be displayed only with explicit retrospective mode.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, time, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

SHANGHAI = ZoneInfo("Asia/Shanghai")
METHOD = "research-seal-quality"
CLOCK_SEMANTICS = "EASTMONEY_FIRST_LAST_SEAL"


def _stamp(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else None
    except ValueError:
        return None


def _code(value):
    return isinstance(value, str) and re.fullmatch(r"\d{6}", value) is not None


def _integer(value):
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        number = float(value)
        return int(number) if number.is_integer() and number >= 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def _source_date(value):
    if isinstance(value, str) and re.fullmatch(r"\d{8}", value):
        return datetime.strptime(value, "%Y%m%d").date().isoformat()
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return date.fromisoformat(value).isoformat()
    raise ValueError("SOURCE_DATE_FORMAT_UNKNOWN")


def seal_clock_seconds(value):
    """09:25 auction or continuous-session clock; zero placeholders are invalid."""
    if isinstance(value, bool) or value in (None, ""):
        return None
    text = str(value).strip()
    if ":" in text:
        if not re.fullmatch(r"\d{2}:\d{2}:\d{2}", text):
            return None
        text = text.replace(":", "")
    else:
        if not re.fullmatch(r"\d{1,6}(?:\.0+)?", text):
            return None
        text = text.split(".")[0].zfill(6)
    hour, minute, second = int(text[:2]), int(text[2:4]), int(text[4:])
    if hour > 23 or minute > 59 or second > 59:
        return None
    seconds = hour * 3600 + minute * 60 + second
    if seconds == 33900 or 34200 <= seconds <= 41400 or 46800 <= seconds <= 54000:
        return seconds
    return None


def _verified_bytes(ref):
    raw = ref.get("source_bytes")
    path, digest = ref.get("source_path"), ref.get("source_sha256")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("SOURCE_PATH_NOT_ABSOLUTE")
    if not isinstance(raw, bytes) or not isinstance(digest, str):
        raise ValueError("SOURCE_BYTES_OR_HASH_MISSING")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("SOURCE_HASH_MISMATCH")
    return raw


def _public_ref(ref):
    return {k: v for k, v in ref.items() if not isinstance(v, bytes)}


def _csv_rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def _source_record(observation):
    ref = observation["provenance"]
    raw = _verified_bytes(ref)
    date, code = observation["signal_date"], observation["code"]
    fmt = ref.get("source_format")
    if fmt == "EASTMONEY_POOL_JSON":
        payload = json.loads(raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            raise ValueError("SOURCE_POOL_PAYLOAD_INVALID")
        data = payload["data"]
        if str(data.get("qdate")) != date.replace("-", ""):
            raise ValueError("SOURCE_DATE_MISMATCH")
        pool = data.get("pool")
        if not isinstance(pool, list) or not all(isinstance(r, dict) for r in pool) or _integer(data.get("tc")) != len(pool):
            raise ValueError("SOURCE_POOL_INCOMPLETE")
        matches = [r for r in pool if str(r.get("c")) == code]
        if len(matches) != 1:
            raise ValueError("SOURCE_CODE_MISSING_OR_DUPLICATE")
        r = matches[0]
        return dict(code=code, signal_date=date, boards=r.get("lbc"),
                    first_seal=r.get("fbt"), last_seal=r.get("lbt"), bomb_count=r.get("zbc"))
    if fmt == "PROVIDER_POOL_CSV":
        manifest_bytes = ref.get("manifest_bytes")
        if not isinstance(manifest_bytes, bytes) or hashlib.sha256(manifest_bytes).hexdigest() != ref.get("manifest_sha256"):
            raise ValueError("SOURCE_MANIFEST_HASH_MISMATCH")
        meta = json.loads(manifest_bytes)
        if not isinstance(meta, dict):
            raise ValueError("SOURCE_MANIFEST_PAYLOAD_INVALID")
        block = meta.get("datasets", {}).get("limit_up", {})
        rows = _csv_rows(raw)
        if _source_date(meta.get("date")) != date or block.get("sha256") != ref["source_sha256"]:
            raise ValueError("SOURCE_MANIFEST_DATE_OR_POOL_HASH_MISMATCH")
        if _integer(block.get("rows")) != len(rows):
            raise ValueError("SOURCE_POOL_INCOMPLETE")
        # Unknown legacy timezone is never invented by this reader.
        if ref.get("captured_at") != meta.get("collected_at"):
            raise ValueError("CAPTURE_TIME_NOT_MANIFEST_BOUND")
        matches = [r for r in rows if r.get("代码") == code]
        if len(matches) != 1:
            raise ValueError("SOURCE_CODE_MISSING_OR_DUPLICATE")
        r = matches[0]
        return dict(code=code, signal_date=date, boards=r.get("连板数"),
                    first_seal=r.get("首次封板时间"), last_seal=r.get("最后封板时间"), bomb_count=r.get("炸板次数"))
    raise ValueError("UNSUPPORTED_SOURCE_FORMAT")


def _timeline(ref, cutoff, derived_at, signal_date):
    times = [_stamp(ref.get(k)) for k in ("captured_at", "available_at", "normalized_at", "archive_formed_at")]
    # A missing old capture time does not excuse derivation before known reads.
    known_normalized, known_formed = times[2:]
    if ((known_normalized is not None and known_normalized > derived_at)
            or (known_formed is not None and known_formed > derived_at)
            or (known_normalized is not None and known_formed is not None and known_normalized > known_formed)):
        return ["SOURCE_TIMING_ORDER_INVALID"]
    known_mtime = _stamp(ref.get("source_actual_file_mtime"))
    if known_mtime is not None and known_mtime > derived_at:
        return ["SOURCE_FILE_MTIME_AFTER_DERIVATION"]
    if any(t is None for t in times):
        return ["SOURCE_TIMING_UNKNOWN"]
    captured, available, normalized, formed = times
    if not captured <= available <= normalized <= formed <= derived_at:
        return ["SOURCE_TIMING_ORDER_INVALID"]
    actual_file_times = [_stamp(ref.get("source_actual_file_mtime"))]
    if ref.get("source_format") == "PROVIDER_POOL_CSV":
        actual_file_times.append(_stamp(ref.get("manifest_actual_file_mtime")))
    if ref.get("receipt_format") == "SIGNAL_CAPTURE_MANIFEST":
        actual_file_times.extend(_stamp(ref.get(k)) for k in
                                 ("receipt_actual_file_mtime", "original_archive_actual_last_file_mtime"))
    if any(t is None for t in actual_file_times):
        return ["SOURCE_ACTUAL_FILE_TIMING_UNKNOWN"]
    if any(t > derived_at for t in actual_file_times):
        return ["SOURCE_FILE_MTIME_AFTER_DERIVATION"]
    if ref.get("source_format") != "PROVIDER_POOL_CSV":
        raw = ref.get("receipt_bytes")
        if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != ref.get("receipt_sha256"):
            return ["SOURCE_TIMING_NOT_SOURCE_BOUND"]
        try:
            receipt = json.loads(raw)
            if ref.get("receipt_format") == "SIGNAL_CAPTURE_MANIFEST":
                matches = [r for r in receipt.get("requests", []) if r.get("kind") == "limit_up"
                           and r.get("raw_path") == ref.get("source_path") and r.get("raw_sha256") == ref.get("source_sha256")]
                if (len(matches) != 1 or matches[0].get("status") != "RAW_JSON_RECEIVED"
                        or matches[0].get("available_at") != ref.get("available_at")
                        or ref.get("captured_at") != ref.get("available_at")
                        or receipt.get("signal_date") != signal_date):
                    return ["SOURCE_CAPTURE_RECEIPT_MISMATCH"]
            elif any(receipt.get(k) != ref.get(k) for k in ("source_path", "source_sha256", "captured_at", "available_at")):
                return ["SOURCE_CAPTURE_RECEIPT_MISMATCH"]
        except (ValueError, UnicodeDecodeError):
            return ["SOURCE_CAPTURE_RECEIPT_INVALID"]
    if captured < datetime.combine(date.fromisoformat(signal_date), time(15), SHANGHAI):
        return ["SIGNAL_CLOSING_FACTS_NOT_YET_OBSERVED"]
    if (formed > cutoff or derived_at > cutoff or any(t > cutoff for t in actual_file_times)
            or ref.get("original_archive_prepared_by_cutoff") is False):
        return ["SOURCE_OR_DERIVATION_AFTER_CUTOFF"]
    return []


def _group_errors(group, signal_date, own_code):
    errors = []
    codes = group.get("codes")
    if not isinstance(codes, list) or not all(_code(c) for c in codes) or len(codes) != len(set(codes)):
        return ["FROZEN_GROUP_CODES_INVALID"]
    if own_code not in codes:
        errors.append("OWN_CODE_NOT_IN_FROZEN_GROUP")
    if group.get("signal_date") != signal_date or not _integer(group.get("boards")):
        errors.append("FROZEN_GROUP_DATE_OR_BOARDS_INVALID")
    try:
        raw = _verified_bytes(group)
        if group.get("source_format") == "FROZEN_GROUP_JSON":
            payload = json.loads(raw)
            if any(payload.get(k) != group.get(k) for k in ("signal_date", "boards", "codes", "frozen_at")):
                errors.append("FROZEN_GROUP_CONTENT_MISMATCH")
        elif group.get("source_format") == "INTAKE_CSV":
            rows = _csv_rows(raw)
            matches = [r for r in rows if r.get("signal_date", r.get("date")) == signal_date and _integer(r.get("boards")) == _integer(group["boards"])]
            if len(matches) != len({r.get("code") for r in matches}) or set(codes) != {r.get("code") for r in matches}:
                errors.append("FROZEN_GROUP_MEMBERSHIP_MISMATCH")
            for row in matches:
                if "provider_same_height_codes" in row and set(json.loads(row["provider_same_height_codes"])) != set(codes) - {row["code"]}:
                    errors.append("FROZEN_GROUP_PEER_COLUMN_MISMATCH")
        else:
            errors.append("FROZEN_GROUP_SOURCE_FORMAT_UNSUPPORTED")
    except (ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
        errors.append(str(exc))
    return list(dict.fromkeys(errors))


def _trading_seconds(start, end):
    return sum(max(0, min(end, b) - max(start, a)) for a, b in ((34200, 41400), (46800, 54000)))


def _event_durations(events, *, code, signal_date, cutoff, derived_at, allow_retrospective):
    """Require a source-bound complete 09:30-to-close state-change record."""
    if not isinstance(events, dict):
        return None, None, ["COMPLETE_EVENT_SEQUENCE_UNAVAILABLE"], False
    try:
        ref = events["provenance"]
        payload = json.loads(_verified_bytes(ref))
        if not isinstance(payload, dict):
            raise ValueError("EVENT_PAYLOAD_INVALID")
        if any(payload.get(k) != events.get(k) for k in ("code", "signal_date", "semantics", "complete", "coverage_start", "coverage_end", "closing_state", "states")):
            raise ValueError("EVENT_CONTENT_MISMATCH")
        if events.get("code") != code or events.get("signal_date") != signal_date:
            raise ValueError("EVENT_CODE_OR_DATE_MISMATCH")
        if events.get("semantics") != "COMPLETE_CONTINUOUS_SESSION_SEAL_STATE_TRANSITIONS" or events.get("complete") is not True:
            raise ValueError("EVENT_SEQUENCE_NOT_COMPLETE")
        if events.get("coverage_start") != "09:30:00" or events.get("coverage_end") != "15:00:00":
            raise ValueError("EVENT_SESSION_COVERAGE_INCOMPLETE")
        if not isinstance(events.get("states"), list) or not all(isinstance(r, dict) for r in events["states"]):
            raise ValueError("EVENT_STATE_LIST_INVALID")
        points = [(seal_clock_seconds(r.get("time")), r.get("state")) for r in events["states"]]
        if not points or points[0][0] != 34200 or any(t is None or t < 34200 or s not in {"OPEN", "SEALED"} for t, s in points):
            raise ValueError("EVENT_STATE_OR_TIME_INVALID")
        if any(b[0] <= a[0] or b[1] == a[1] for a, b in zip(points, points[1:])):
            raise ValueError("EVENT_TRANSITIONS_NOT_STRICT")
        if events.get("closing_state") != points[-1][1]:
            raise ValueError("EVENT_CLOSING_STATE_MISMATCH")
        timeline = _timeline(ref, cutoff, derived_at, signal_date)
        # Invalid orders are corrupt evidence; only missing/late times permit diagnosis.
        hard_errors = {"SOURCE_TIMING_ORDER_INVALID", "SOURCE_FILE_MTIME_AFTER_DERIVATION", "SOURCE_CAPTURE_RECEIPT_MISMATCH", "SOURCE_CAPTURE_RECEIPT_INVALID", "SIGNAL_CLOSING_FACTS_NOT_YET_OBSERVED"}
        if hard_errors.intersection(timeline) or (timeline and not allow_retrospective):
            return None, None, timeline, False
        open_seconds = sum(_trading_seconds(t, points[i + 1][0] if i + 1 < len(points) else 54000)
                           for i, (t, state) in enumerate(points) if state == "OPEN")
        stable_seconds = _trading_seconds(points[-1][0], 54000) if points[-1][1] == "SEALED" else 0
        return open_seconds / 60, stable_seconds / 60, timeline, not timeline
    except (ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
        return None, None, [str(exc)], False


def compute_seal_quality(own_code, frozen_group, observations, *, signal_date,
                         target_date, knowledge_cutoff, derived_at,
                         allow_retrospective=False):
    """Compute original-group research fields; values never imply a promotion.

    ``frozen_group`` includes source bytes/hash/path, signal_date/boards/codes,
    frozen_at/archive_formed_at, and source_format INTAKE_CSV or FROZEN_GROUP_JSON.
    ``observations`` is keyed by six-digit code. Each row includes original
    first_seal/last_seal/bomb_count, count_origin, clock_semantics and provenance.
    All returned objects are JSON serializable; bytes are excluded from output.
    """
    cutoff, derived = _stamp(knowledge_cutoff), _stamp(derived_at)
    if cutoff is None or derived is None:
        raise ValueError("knowledge_cutoff and derived_at require timezone-aware ISO timestamps")
    try:
        signal, target = date.fromisoformat(signal_date), date.fromisoformat(target_date)
    except (TypeError, ValueError) as exc:
        raise ValueError("signal and target require ISO dates") from exc
    if not _code(own_code) or target <= signal:
        raise ValueError("invalid own code or target date")
    if cutoff.astimezone(SHANGHAI).date() != signal or cutoff.astimezone(SHANGHAI).time() > time(16, 30) or cutoff.astimezone(SHANGHAI).time() < time(15):
        raise ValueError("signal closing knowledge cutoff must be 15:00 through 16:30 Asia/Shanghai")
    group_errors = _group_errors(frozen_group, signal_date, own_code)
    original_codes = list(frozen_group.get("codes") or [])
    peers = [c for c in original_codes if c != own_code]
    group_times = [_stamp(frozen_group.get(k)) for k in ("frozen_at", "archive_formed_at")]
    group_timing = []
    if any(t is None for t in group_times):
        group_timing = ["FROZEN_GROUP_TIMING_UNKNOWN"]
    elif group_times[1] > group_times[0] or group_times[0] > derived:
        group_errors.append("FROZEN_GROUP_TIMING_ORDER_INVALID")
    elif any(t > cutoff for t in group_times):
        group_timing = ["FROZEN_GROUP_AFTER_CUTOFF"]
    row_info = {}
    first_times = {}
    for code in original_codes:
        row = observations.get(code)
        errors, timing = [], []
        original = None
        if row is None:
            errors = ["SOURCE_ROW_MISSING"]
        else:
            try:
                if row.get("code") != code or row.get("signal_date") != signal_date or _integer(row.get("boards")) != _integer(frozen_group.get("boards")):
                    raise ValueError("ROW_CODE_DATE_OR_BOARDS_MISMATCH")
                original = _source_record(row)
                if _integer(original["boards"]) != _integer(row["boards"]):
                    raise ValueError("SOURCE_BOARDS_MISMATCH")
                if row.get("clock_semantics") != CLOCK_SEMANTICS:
                    raise ValueError("CLOCK_SEMANTICS_UNKNOWN")
                for field in ("first_seal", "last_seal"):
                    if seal_clock_seconds(row.get(field)) != seal_clock_seconds(original.get(field)):
                        raise ValueError("SOURCE_FIELD_MISMATCH:" + field)
                if _integer(row.get("bomb_count")) != _integer(original.get("bomb_count")):
                    raise ValueError("SOURCE_FIELD_MISMATCH:bomb_count")
                timing = _timeline(row["provenance"], cutoff, derived, signal_date)
                hard_errors = {"SOURCE_TIMING_ORDER_INVALID", "SOURCE_FILE_MTIME_AFTER_DERIVATION", "SOURCE_CAPTURE_RECEIPT_MISMATCH", "SOURCE_CAPTURE_RECEIPT_INVALID", "SIGNAL_CLOSING_FACTS_NOT_YET_OBSERVED"}
                if hard_errors.intersection(timing):
                    errors.extend(timing)
            except (ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
                errors.append(str(exc))
        valid = not group_errors and not errors and (allow_retrospective or not (timing or group_timing))
        first = seal_clock_seconds(original.get("first_seal")) if original else None
        last = seal_clock_seconds(original.get("last_seal")) if original else None
        if first is None:
            errors.append("FIRST_SEAL_UNKNOWN_OR_INVALID")
        if valid and first is not None:
            first_times[code] = first
        row_info[code] = dict(valid=valid, errors=list(dict.fromkeys(errors)), timing=timing,
                              source=_public_ref(row["provenance"]) if row and "provenance" in row else None,
                              first_seconds=first, last_seconds=last, original=original)
    own = row_info.get(own_code, {})
    missing = [c for c in peers if c not in first_times]
    percentile = None
    if peers and not missing and own_code in first_times:
        percentile = (sum(first_times[c] < first_times[own_code] for c in peers)
                      + .5 * sum(first_times[c] == first_times[own_code] for c in peers)) / len(peers)
    percentile_reasons = list(group_errors + group_timing)
    if not peers:
        percentile_reasons.append("NO_ORIGINAL_PEERS")
    if missing:
        percentile_reasons.append("ORIGINAL_PEER_FIRST_SEAL_INCOMPLETE")
    if own_code not in first_times:
        percentile_reasons.append("OWN_FIRST_SEAL_UNAVAILABLE")
    original = own.get("original") or {}
    row = observations.get(own_code) or {}
    count = _integer(original.get("bomb_count")) if own.get("valid") and row.get("count_origin") == "ORIGINAL_PROVIDER_REPORT" else None
    last = own.get("last_seconds") if own.get("valid") else None
    last_reasons = []
    if last is not None and own.get("first_seconds") is not None and last < own["first_seconds"]:
        last, last_reasons = None, ["LAST_SEAL_PRECEDES_FIRST"]
    open_minutes, stable_minutes, event_reasons, events_pit = _event_durations(
        row.get("events"), code=own_code, signal_date=signal_date, cutoff=cutoff,
        derived_at=derived, allow_retrospective=allow_retrospective)
    if group_errors:
        open_minutes = stable_minutes = None
        event_reasons = group_errors + event_reasons
    all_timing = group_timing + [e for r in row_info.values() for e in r["timing"]]
    pit = not group_errors and not all_timing and not any(r["errors"] for r in row_info.values())
    return dict(method=METHOD, signal_date=signal_date, target_date=target_date, code=own_code,
                boards=frozen_group.get("boards"), knowledge_cutoff=knowledge_cutoff,
                derived_at=derived_at, original_group_codes=original_codes, original_peer_codes=peers,
                original_peer_count=len(peers), observed_peer_count=len(peers) - len(missing),
                missing_peer_codes=missing, peer_coverage=(len(peers) - len(missing)) / len(peers) if peers else None,
                first_seal_percentile_same_height_ex_self=percentile,
                first_seal_percentile_unit="dimensionless; higher means later",
                failed_seal_count_reported=count,
                last_seal_time_reported=(time(last // 3600, (last % 3600) // 60, last % 60).isoformat() if last is not None else None),
                open_duration_trading_minutes=open_minutes, seal_stable_duration_trading_minutes=stable_minutes,
                reasons={"first_seal_percentile":list(dict.fromkeys(percentile_reasons)),
                         "failed_seal_count":[] if count is not None else ["ORIGINAL_REPORTED_COUNT_UNAVAILABLE"] + own.get("errors", []),
                         "last_seal":last_reasons + ([] if last is not None else ["LAST_SEAL_UNAVAILABLE"]) + own.get("errors", []),
                         "durations":event_reasons},
                group_source=_public_ref(frozen_group),
                observations=[dict(code=c, **{k: v for k, v in row_info[c].items() if k not in {"original"}}) for c in original_codes],
                point_in_time_eligible=pit, duration_point_in_time_eligible=events_pit and not group_errors and not group_timing,
                status="OBSERVED_RESEARCH_ONLY" if pit else ("RETROSPECTIVE_DIAGNOSTIC_ONLY" if allow_retrospective else "EVIDENCE_UNKNOWN"),
                new_forward_samples=0, model_changed=False, calibrated_probability=False,
                supplier_scope_only=True, trade_authorized=False)


def load_v275_observations(csv_path, report_path=None, *, project_root, signal_date):
    """Rebuild original raw facts; preserve absent/naive capture times as unknown.

    Does not read promoted_next_day, promotion_label_status or target labels.
    A hash/date mismatch is retained as row error rather than dropping the code.
    """
    archive = Path(csv_path).resolve()
    report = Path(report_path).resolve() if report_path else archive.parent / "report.json"
    root = Path(project_root).resolve()
    archive_rows = _csv_rows(archive.read_bytes())
    receipt = json.loads(report.read_bytes())
    archive_formed = datetime.fromtimestamp(max(archive.stat().st_mtime, report.stat().st_mtime), timezone.utc).isoformat()
    output = {}
    for archive_row in archive_rows:
        if archive_row.get("date") != signal_date:
            continue
        code = archive_row.get("code")
        if not _code(code) or code in output:
            raise ValueError("V275 archive duplicate or invalid stock code")
        source = Path(archive_row.get("source_path", ""))
        source = source.resolve() if source.is_absolute() else (root / source).resolve()
        normalized_at = datetime.now(timezone.utc).isoformat()
        row = dict(code=code, signal_date=signal_date, boards=_integer(archive_row.get("boards")),
                   clock_semantics=CLOCK_SEMANTICS, count_origin="ORIGINAL_PROVIDER_REPORT", events=None)
        ref = dict(source_path=str(source), source_sha256=archive_row.get("source_sha256"),
                   source_bytes=b"", captured_at=None, available_at=None,
                   normalized_at=normalized_at, archive_formed_at=normalized_at,
                   prior_archive_formed_at=archive_formed,
                   adapter_archive_path=str(archive), adapter_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                   adapter_report_path=str(report), adapter_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
        row["provenance"] = ref
        try:
            source.relative_to(root)
            ref.update(source_bytes=source.read_bytes(),
                       source_actual_file_mtime=datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat())
        except (OSError, ValueError) as exc:
            row.update(first_seal=None, last_seal=None, bomb_count=None, adapter_error="ORIGINAL_SOURCE_UNAVAILABLE:" + str(exc))
            output[code] = row
            continue
        relative = archive_row.get("source_path")
        if receipt.get("source_hashes", {}).get(relative) != ref["source_sha256"]:
            ref["source_bytes"] = b""  # Fail the existing hash gate; retain original code.
            row["adapter_error"] = "V275_REPORT_SOURCE_HASH_MISMATCH"
        if source.suffix.lower() == ".json":
            ref["source_format"] = "EASTMONEY_POOL_JSON"
        else:
            manifest = source.parent / "manifest.json"
            try:
                manifest_bytes = manifest.read_bytes()
                meta = json.loads(manifest_bytes)
            except (OSError, ValueError) as exc:
                row.update(first_seal=None, last_seal=None, bomb_count=None, adapter_error="ORIGINAL_MANIFEST_UNAVAILABLE:" + str(exc))
                output[code] = row
                continue
            ref.update(source_format="PROVIDER_POOL_CSV", manifest_bytes=manifest_bytes,
                       manifest_path=str(manifest), manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                       manifest_actual_file_mtime=datetime.fromtimestamp(manifest.stat().st_mtime, timezone.utc).isoformat(),
                       captured_at=meta.get("collected_at"), available_at=meta.get("collected_at"))
        try:
            original = _source_record(row)
            row.update({k: original[k] for k in ("first_seal", "last_seal", "bomb_count")})
        except (ValueError, KeyError, TypeError) as exc:
            row.update(first_seal=None, last_seal=None, bomb_count=None, adapter_error=str(exc))
        finished_at = datetime.now(timezone.utc).isoformat()
        ref.update(normalized_at=finished_at, archive_formed_at=finished_at)
        output[code] = row
    return output


def load_signal_capture_observations(directory, *, signal_date):
    """Read the existing signal-day capture with its original formation gate.

    Whole-pool request failure/date conflict returns no invented rows; the caller
    retains original group codes and will mark each missing member unknown.
    The verify_archive contract re-checks bound program, raw and artifact bytes.
    No requests, labels, training or capture-time overrides are introduced here.
    """
    from capture_signal_theme_inputs import verify_archive

    directory = Path(directory).resolve()
    check = verify_archive(directory, require_by_cutoff=False)
    manifest_path = directory / "manifest.json"
    formation_path = directory / "formation_receipt.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    formation_bytes = formation_path.read_bytes()
    formation = json.loads(formation_bytes)
    if manifest.get("signal_date") != signal_date:
        raise ValueError("SIGNAL_CAPTURE_DATE_MISMATCH")
    entries = [r for r in manifest["requests"] if r.get("kind") == "limit_up"]
    if len(entries) != 1:
        raise ValueError("SIGNAL_CAPTURE_LIMIT_POOL_REQUEST_SCOPE_INVALID")
    entry = entries[0]
    if entry.get("status") != "RAW_JSON_RECEIVED":
        return {}
    path = Path(entry["raw_path"]).resolve()
    raw = path.read_bytes()
    data = json.loads(raw).get("data") or {}
    if str(data.get("qdate")) != signal_date.replace("-", ""):
        raise ValueError("SOURCE_DATE_MISMATCH")
    pool = data.get("pool")
    if not isinstance(pool, list) or _integer(data.get("tc")) != len(pool):
        raise ValueError("SOURCE_POOL_INCOMPLETE")
    actual_last_file = max(datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
                           for p in directory.rglob("*") if p.is_file())
    source_ref = dict(source_path=str(path), source_sha256=entry["raw_sha256"],
                      source_bytes=raw, source_format="EASTMONEY_POOL_JSON",
                      captured_at=entry["available_at"], available_at=entry["available_at"],
                      source_actual_file_mtime=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                      receipt_format="SIGNAL_CAPTURE_MANIFEST", receipt_bytes=manifest_bytes,
                      receipt_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                      receipt_path=str(manifest_path), receipt_actual_file_mtime=datetime.fromtimestamp(manifest_path.stat().st_mtime, timezone.utc).isoformat(),
                      original_formation_receipt_path=str(formation_path), original_formation_receipt_sha256=hashlib.sha256(formation_bytes).hexdigest(),
                      original_formation_recorded_at=formation["recorded_at"],
                      original_archive_actual_last_file_mtime=actual_last_file.isoformat(),
                      original_archive_prepared_by_cutoff=check["prepared_by_cutoff"])
    output = {}
    for original in pool:
        code = original.get("c")
        if not _code(code) or code in output:
            raise ValueError("SOURCE_CODE_MISSING_OR_DUPLICATE")
        row = dict(code=code, signal_date=signal_date, boards=_integer(original.get("lbc")),
                   first_seal=original.get("fbt"), last_seal=original.get("lbt"), bomb_count=original.get("zbc"),
                   clock_semantics=CLOCK_SEMANTICS, count_origin="ORIGINAL_PROVIDER_REPORT", events=None,
                   provenance=dict(source_ref))
        _source_record(row)  # Bind original date/code/values again independently.
        finished = datetime.now(timezone.utc)
        row["provenance"].update(normalized_at=finished.isoformat(),
                                  archive_formed_at=max(finished, actual_last_file).isoformat())
        output[code] = row
    return output
