"""Synthetic source/timing/group checks, without real outcomes or fitting."""
import copy
import hashlib
import json

import pytest

from research.indicator_observations.seal_quality import (
    CLOCK_SEMANTICS, compute_seal_quality, load_v275_observations, seal_clock_seconds,
    load_signal_capture_observations,
)


SIGNAL = "2026-09-30"
TARGET = "2026-10-08"
CUTOFF = "2026-09-30T16:30:00+08:00"
DERIVED = "2026-09-30T16:10:00+08:00"


def binding(path, raw, fmt):
    ref = dict(source_path=str(path.resolve()), source_bytes=raw,
               source_sha256=hashlib.sha256(raw).hexdigest(), source_format=fmt,
               captured_at="2026-09-30T15:10:00+08:00", available_at="2026-09-30T15:11:00+08:00",
               source_actual_file_mtime="2026-09-30T15:10:00+08:00",
               normalized_at="2026-09-30T15:20:00+08:00", archive_formed_at="2026-09-30T15:21:00+08:00")
    receipt = json.dumps({k: ref[k] for k in ("source_path", "source_sha256", "captured_at", "available_at")}).encode()
    ref.update(receipt_bytes=receipt, receipt_sha256=hashlib.sha256(receipt).hexdigest())
    return ref


def fixture(tmp_path):
    codes = ["000001", "000002", "000003"]
    pool = [dict(c=c, lbc=2, fbt=t, lbt=140000, zbc=0) for c, t in zip(codes, [100000, 94500, 100000])]
    raw = json.dumps(dict(data=dict(qdate="20260930", tc=3, pool=pool))).encode()
    ref = binding(tmp_path / "pool.json", raw, "EASTMONEY_POOL_JSON")
    rows = {r["c"]:dict(code=r["c"], signal_date=SIGNAL, boards=2,
                        first_seal=r["fbt"], last_seal=r["lbt"], bomb_count=r["zbc"],
                        clock_semantics=CLOCK_SEMANTICS, count_origin="ORIGINAL_PROVIDER_REPORT",
                        provenance=copy.deepcopy(ref), events=None) for r in pool}
    payload = dict(signal_date=SIGNAL, boards=2, codes=codes, frozen_at="2026-09-30T16:00:00+08:00")
    groupraw = json.dumps(payload).encode()
    group = dict(**payload, source_path=str((tmp_path / "group.json").resolve()), source_bytes=groupraw,
                 source_sha256=hashlib.sha256(groupraw).hexdigest(), source_format="FROZEN_GROUP_JSON",
                 archive_formed_at="2026-09-30T15:59:00+08:00")
    return group, rows


def compute(group, rows, **kwargs):
    return compute_seal_quality("000001", group, rows, signal_date=SIGNAL, target_date=TARGET,
                                knowledge_cutoff=CUTOFF, derived_at=kwargs.pop("derived_at", DERIVED), **kwargs)


def events(tmp_path, **changes):
    payload = dict(code="000001", signal_date=SIGNAL,
                   semantics="COMPLETE_CONTINUOUS_SESSION_SEAL_STATE_TRANSITIONS", complete=True,
                   coverage_start="09:30:00", coverage_end="15:00:00", closing_state="SEALED",
                   states=[dict(time="09:30:00", state="SEALED"), dict(time="11:20:00", state="OPEN"),
                           dict(time="13:10:00", state="SEALED")])
    payload.update(changes)
    raw = json.dumps(payload).encode()
    return dict(**payload, provenance=binding(tmp_path / "events.json", raw, "SEAL_STATE_TRANSITIONS_JSON"))


def test_full_group_tie_self_exclusion_and_literal_reported_zero(tmp_path):
    group, rows = fixture(tmp_path)
    result = compute(group, rows)
    assert result["first_seal_percentile_same_height_ex_self"] == .75
    assert result["original_peer_codes"] == ["000002", "000003"]
    assert result["peer_coverage"] == 1
    assert result["failed_seal_count_reported"] == 0
    assert result["last_seal_time_reported"] == "14:00:00"
    assert result["point_in_time_eligible"] is True
    assert result["open_duration_trading_minutes"] is None
    assert result["seal_stable_duration_trading_minutes"] is None
    assert result["new_forward_samples"] == 0 and result["model_changed"] is False
    json.dumps(result, allow_nan=False)


def test_missing_original_peer_cannot_create_survivor_percentile(tmp_path):
    group, rows = fixture(tmp_path)
    del rows["000002"]
    result = compute(group, rows)
    assert result["first_seal_percentile_same_height_ex_self"] is None
    assert result["original_peer_count"] == 2
    assert result["missing_peer_codes"] == ["000002"]
    assert result["peer_coverage"] == .5


@pytest.mark.parametrize("field,value", [("first_seal", 0), ("first_seal", 120000), ("bomb_count", 9), ("boards", 3), ("code", "000004")])
def test_raw_field_or_identity_mutation_is_rejected(tmp_path, field, value):
    group, rows = fixture(tmp_path)
    rows["000002"][field] = value
    result = compute(group, rows, allow_retrospective=True)
    assert result["first_seal_percentile_same_height_ex_self"] is None
    assert "000002" in result["missing_peer_codes"]


def test_changed_raw_bytes_hash_and_group_membership_fail(tmp_path):
    group, rows = fixture(tmp_path)
    rows["000002"]["provenance"]["source_bytes"] += b" "
    assert compute(group, rows)["first_seal_percentile_same_height_ex_self"] is None
    group, rows = fixture(tmp_path)
    group["codes"].remove("000003")
    assert compute(group, rows)["first_seal_percentile_same_height_ex_self"] is None


def test_reconstructed_zero_count_is_not_original_zero(tmp_path):
    group, rows = fixture(tmp_path)
    rows["000001"]["count_origin"] = "RECONSTRUCTED_ZERO_PLACEHOLDER"
    assert compute(group, rows)["failed_seal_count_reported"] is None


def test_source_and_derivation_later_than_cutoff_are_diagnostic_only(tmp_path):
    group, rows = fixture(tmp_path)
    late = "2026-10-07T16:00:00+08:00"
    result = compute(group, rows, derived_at=late)
    assert result["first_seal_percentile_same_height_ex_self"] is None
    retrospective = compute(group, rows, derived_at=late, allow_retrospective=True)
    assert retrospective["first_seal_percentile_same_height_ex_self"] == .75
    assert retrospective["status"] == "RETROSPECTIVE_DIAGNOSTIC_ONLY"
    assert retrospective["point_in_time_eligible"] is False
    assert retrospective["new_forward_samples"] == 0


def test_unbound_or_naive_capture_is_never_point_in_time(tmp_path):
    group, rows = fixture(tmp_path)
    rows["000002"]["provenance"].pop("receipt_bytes")
    assert compute(group, rows)["first_seal_percentile_same_height_ex_self"] is None
    assert compute(group, rows, allow_retrospective=True)["point_in_time_eligible"] is False
    group, rows = fixture(tmp_path)
    rows["000002"]["provenance"]["captured_at"] = "2026-09-30T15:10:00"
    assert compute(group, rows)["first_seal_percentile_same_height_ex_self"] is None


def test_invalid_receipt_or_timing_order_is_not_salvaged_by_diagnosis(tmp_path):
    group, rows = fixture(tmp_path)
    rows["000002"]["provenance"]["captured_at"] = "2026-09-30T15:12:00+08:00"
    assert compute(group, rows, allow_retrospective=True)["first_seal_percentile_same_height_ex_self"] is None


def test_complete_events_compute_trading_minutes_excluding_lunch(tmp_path):
    group, rows = fixture(tmp_path)
    rows["000001"]["events"] = events(tmp_path)
    result = compute(group, rows)
    assert result["open_duration_trading_minutes"] == 20
    assert result["seal_stable_duration_trading_minutes"] == 110
    assert result["duration_point_in_time_eligible"] is True


@pytest.mark.parametrize("change", [dict(complete=False), dict(coverage_start="10:00:00"),
                                    dict(closing_state="OPEN"), dict(code="000003"),
                                    dict(states=[dict(time="09:30:00", state="OPEN"), dict(time="12:00:00", state="SEALED")])])
def test_incomplete_or_invalid_sequence_never_infers_duration(tmp_path, change):
    group, rows = fixture(tmp_path)
    rows["000001"]["events"] = events(tmp_path, **change)
    result = compute(group, rows)
    assert result["open_duration_trading_minutes"] is None
    assert result["seal_stable_duration_trading_minutes"] is None


def test_first_last_gap_is_not_used_as_duration(tmp_path):
    group, rows = fixture(tmp_path)
    rows["000001"]["first_last_clock_gap_minutes"] = 240
    result = compute(group, rows)
    assert result["open_duration_trading_minutes"] is None


def test_clock_session_boundaries(tmp_path):
    assert seal_clock_seconds("09:25:00") == 33900
    assert seal_clock_seconds("113000.0") == 41400
    assert seal_clock_seconds("15:00:00") == 54000
    assert seal_clock_seconds(0) is None
    assert seal_clock_seconds("09:25:01") is None
    assert seal_clock_seconds("12:00:00") is None
    assert seal_clock_seconds("15:01:00") is None


def test_v275_adapter_rebuilds_raw_source_without_labels(tmp_path):
    original = tmp_path / "raw" / "zt_pool_20260930.json"
    original.parent.mkdir()
    raw = json.dumps(dict(data=dict(qdate="20260930", tc=1,
                                   pool=[dict(c="000001", lbc=2, fbt=100000, lbt=140000, zbc=0)]))).encode()
    original.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    archive = tmp_path / "reported_seal_rows.csv"
    archive.write_text("date,code,boards,source_path,source_sha256,first_seal,promoted_next_day\n"
                       f"2026-09-30,000001,2,raw/zt_pool_20260930.json,{digest},0,999\n", encoding="utf-8")
    (tmp_path / "report.json").write_text(json.dumps(dict(source_hashes={"raw/zt_pool_20260930.json":digest})))
    rows = load_v275_observations(archive, project_root=tmp_path, signal_date=SIGNAL)
    assert rows["000001"]["first_seal"] == 100000
    assert rows["000001"]["bomb_count"] == 0
    assert "promoted_next_day" not in rows["000001"]
    assert rows["000001"]["provenance"]["captured_at"] is None


def test_cutoff_requires_signal_date_timezone_and_original_1630_limit(tmp_path):
    group, rows = fixture(tmp_path)
    for cutoff in ("2026-09-30T16:30:00", "2026-10-07T16:30:00+08:00", "2026-09-30T16:31:00+08:00"):
        with pytest.raises(ValueError):
            compute_seal_quality("000001", group, rows, signal_date=SIGNAL, target_date=TARGET,
                                 knowledge_cutoff=cutoff, derived_at=DERIVED)


def test_real_filetime_after_cutoff_cannot_count_as_point_in_time(tmp_path):
    group, rows = fixture(tmp_path)
    rows["000002"]["provenance"]["source_actual_file_mtime"] = "2026-10-07T15:00:00+08:00"
    result = compute(group, rows, derived_at="2026-10-07T16:00:00+08:00", allow_retrospective=True)
    assert result["first_seal_percentile_same_height_ex_self"] == .75
    assert result["point_in_time_eligible"] is False
    assert compute(group, rows, allow_retrospective=True)["first_seal_percentile_same_height_ex_self"] is None


def test_adapter_missing_original_source_keeps_stock_row_unknown(tmp_path):
    archive = tmp_path / "reported_seal_rows.csv"
    digest = "0" * 64
    archive.write_text("date,code,boards,source_path,source_sha256\n"
                       f"2026-09-30,000001,2,missing.json,{digest}\n", encoding="utf-8")
    (tmp_path / "report.json").write_text(json.dumps(dict(source_hashes={"missing.json":digest})))
    rows = load_v275_observations(archive, project_root=tmp_path, signal_date=SIGNAL)
    assert list(rows) == ["000001"]
    assert rows["000001"]["first_seal"] is None
    assert "ORIGINAL_SOURCE_UNAVAILABLE" in rows["000001"]["adapter_error"]


def test_unknown_old_capture_does_not_allow_derivation_before_actual_normalization(tmp_path):
    group, rows = fixture(tmp_path)
    rows["000002"]["provenance"]["captured_at"] = None
    rows["000002"]["provenance"]["normalized_at"] = "2026-10-07T16:00:01+08:00"
    rows["000002"]["provenance"]["archive_formed_at"] = "2026-10-07T16:00:01+08:00"
    result = compute(group, rows, derived_at="2026-10-07T16:00:00+08:00", allow_retrospective=True)
    assert result["first_seal_percentile_same_height_ex_self"] is None


@pytest.mark.parametrize("manifest_date", ["20260930", "2026-09-30"])
def test_original_csv_manifest_accepts_only_two_explicit_date_formats(tmp_path, manifest_date):
    group, rows = fixture(tmp_path)
    raw = ("代码,连板数,首次封板时间,最后封板时间,炸板次数\n"
           "000001,2,100000,140000,0\n000002,2,94500,140000,0\n000003,2,100000,140000,0\n").encode("utf-8-sig")
    digest = hashlib.sha256(raw).hexdigest()
    meta = dict(date=manifest_date, collected_at="2026-09-30T15:10:00+08:00",
                datasets=dict(limit_up=dict(sha256=digest, rows=3)))
    meta_bytes = json.dumps(meta).encode()
    for row in rows.values():
        row["provenance"].update(source_format="PROVIDER_POOL_CSV", source_bytes=raw,
                                 source_sha256=digest, manifest_bytes=meta_bytes,
                                 manifest_sha256=hashlib.sha256(meta_bytes).hexdigest(),
                                 manifest_actual_file_mtime="2026-09-30T15:10:00+08:00")
    assert compute(group, rows)["first_seal_percentile_same_height_ex_self"] == .75


def test_later_next_morning_archive_is_diagnostic_but_prior_day_evening_is_rejected(tmp_path):
    for captured, accepted in [("2026-10-01T10:00:00+08:00", True), ("2026-09-29T18:00:00+08:00", False)]:
        group, rows = fixture(tmp_path)
        ref = rows["000002"]["provenance"]
        ref.update(captured_at=captured, available_at=captured,
                   normalized_at="2026-10-01T10:01:00+08:00", archive_formed_at="2026-10-01T10:02:00+08:00")
        receipt = json.dumps({k:ref[k] for k in ("source_path", "source_sha256", "captured_at", "available_at")}).encode()
        ref.update(receipt_bytes=receipt, receipt_sha256=hashlib.sha256(receipt).hexdigest())
        result = compute(group, rows, derived_at="2026-10-07T16:00:00+08:00", allow_retrospective=True)
        assert (result["first_seal_percentile_same_height_ex_self"] is not None) is accepted
        assert result["point_in_time_eligible"] is False


def test_late_group_never_has_duration_point_in_time_eligibility(tmp_path):
    group, rows = fixture(tmp_path)
    rows["000001"]["events"] = events(tmp_path)
    group["frozen_at"] = "2026-10-07T15:00:00+08:00"
    payload = {k:group[k] for k in ("signal_date", "boards", "codes", "frozen_at")}
    group["source_bytes"] = json.dumps(payload).encode()
    group["source_sha256"] = hashlib.sha256(group["source_bytes"]).hexdigest()
    result = compute(group, rows, derived_at="2026-10-07T16:00:00+08:00", allow_retrospective=True)
    assert result["open_duration_trading_minutes"] == 20
    assert result["duration_point_in_time_eligible"] is False


def test_signal_capture_adapter_reuses_original_archive_validation(tmp_path, monkeypatch):
    import capture_signal_theme_inputs
    calls = []
    monkeypatch.setattr(capture_signal_theme_inputs, "verify_archive",
                        lambda directory, require_by_cutoff: calls.append((directory, require_by_cutoff)) or {"prepared_by_cutoff":False})
    directory = tmp_path / "capture"
    (directory / "raw").mkdir(parents=True)
    path = (directory / "raw" / "limit_up.json").resolve()
    raw = json.dumps(dict(data=dict(qdate="20260930", tc=1,
                                   pool=[dict(c="000001", lbc=2, fbt=100000, lbt=140000, zbc=0)]))).encode()
    path.write_bytes(raw)
    entry = dict(kind="limit_up", status="RAW_JSON_RECEIVED", raw_path=str(path),
                 raw_sha256=hashlib.sha256(raw).hexdigest(), available_at="2026-09-30T19:18:00+08:00")
    manifest = dict(signal_date=SIGNAL, requests=[entry])
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "formation_receipt.json").write_text(json.dumps(dict(recorded_at="2026-09-30T19:20:00+08:00")))
    result = load_signal_capture_observations(directory, signal_date=SIGNAL)
    assert calls == [(directory.resolve(), False)]
    assert result["000001"]["bomb_count"] == 0
    assert result["000001"]["provenance"]["available_at"] == entry["available_at"]
    assert result["000001"]["provenance"]["receipt_format"] == "SIGNAL_CAPTURE_MANIFEST"
    assert result["000001"]["provenance"]["original_archive_prepared_by_cutoff"] is False
