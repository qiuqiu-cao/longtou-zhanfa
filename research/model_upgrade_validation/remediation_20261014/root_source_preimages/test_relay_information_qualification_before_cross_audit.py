import copy
import json
from pathlib import Path

import pandas as pd
import pytest

import relay_information_qualification as q


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def provider(date="20260930", first=92500, bomb=0):
    return {"rc": 0, "data": {"qdate": date, "tc": 1, "pool": [
        {"c": "002242", "n": "anchor", "lbc": 3, "fbt": first, "lbt": first,
         "zbc": bomb, "fund": 12, "amount": 100, "ltsz": 1000}]}}


def pool(tmp_path):
    return save(tmp_path / "pool.json", {"signal_date": "2026-09-30", "target_date": "2026-10-08",
        "rows": [{"code": "002242", "name": "anchor", "boards": 3},
                 {"code": "002696", "name": "child", "boards": 2}]})


def test_original_zero_bomb_is_real_provider_field(tmp_path):
    path = save(tmp_path / "2026-09-30_limit_up.json", provider())
    rows, evidence = q.source_rows(path, {"2026-09-30"})
    assert evidence["status"] == "SOURCE_BOUND_PROVIDER_VALUES_DIAGNOSTIC"
    assert rows[0]["bomb_count_is_reported_zero"] is True
    assert rows[0]["reported_clock_contract_valid"] is True
    assert rows[0]["closing_order_book_verified"] is False


def test_qdate_does_not_follow_filename_by_assumption(tmp_path):
    path = save(tmp_path / "2026-09-29_limit_up.json", provider())
    rows, evidence = q.source_rows(path, {"2026-09-29", "2026-09-30"})
    assert not rows
    assert evidence["status"] == "SOURCE_DATE_MISMATCH"


def test_non_session_rejected(tmp_path):
    path = save(tmp_path / "2026-09-30_limit_up.json", provider())
    assert q.source_rows(path, {"2026-09-29"})[1]["status"] == "NON_SESSION_OR_UNBOUND_DATE"


def test_adjacent_clock_retains_unknown_semantics(tmp_path):
    path = save(tmp_path / "2026-09-30_limit_up.json", provider(first=92501))
    rows, _ = q.source_rows(path, {"2026-09-30"})
    assert rows[0]["first_clock_context"]["structural_seconds"] == 33901
    assert rows[0]["reported_clock_contract_valid"] is False


def test_reconstructed_csv_never_turns_bomb_zero_into_report(tmp_path):
    path = tmp_path / "zt_pool_20260930.csv"
    pd.DataFrame([{"代码": "002242", "名称": "anchor", "连板数": 3,
                   "首次封板时间": 999999, "炸板次数": 0, "data_source": "lianban.net JSON-LD"}]).to_csv(path, index=False)
    assert q.source_rows(path, {"2026-09-30"})[1]["status"] == "RECONSTRUCTED_PLACEHOLDER_NOT_REPORTED_EVENT"


def test_original_receipt_hash_guard(tmp_path):
    raw = save(tmp_path / "raw" / "2026-09-30_limit_up.json", provider())
    save(tmp_path / "manifest.json", {"requests": [{"raw_path": str(raw), "raw_sha256": "changed", "date": "2026-09-30"}]})
    assert q.source_rows(raw, {"2026-09-30"})[1]["status"] == "CAPTURE_HASH_MISMATCH"


def test_preclose_receive_is_not_closing_snapshot(tmp_path):
    raw = save(tmp_path / "raw" / "2026-09-30_limit_up.json", provider())
    save(tmp_path / "manifest.json", {"requests": [{"raw_path": str(raw), "raw_sha256": q.sha(raw),
        "date": "2026-09-30", "available_at": "2026-09-30T04:29:35+00:00"}]})
    events, _ = q.source_rows(raw, {"2026-09-30"})
    assert events[0]["capture_scope"] == "PRE_CLOSE_RECEIVE_TIME_NOT_SIGNAL_CLOSE_PROOF"
    frame = pd.DataFrame([{"date": "2026-09-30", "code": "002242", "name": "anchor", "boards": 3}])
    result = q.complete_projection(frame, events)
    assert result.loc[0, "reported_event_state"] == "UNKNOWN_NO_QUALIFIED_LOCAL_REPORT"


def test_full_pool_unknown_and_conflicting_snapshots_survive(tmp_path):
    p = q.read(pool(tmp_path))
    frame = pd.DataFrame([{"date": p["signal_date"], **r} for r in p["rows"]])
    raw = save(tmp_path / "2026-09-30_limit_up.json", provider())
    events, _ = q.source_rows(raw, {"2026-09-30"})
    alternative = copy.deepcopy(events[0]); alternative["sealed_fund"] = 99
    result = q.complete_projection(frame, events + [alternative])
    assert len(result) == 2
    assert list(result.reported_event_state) == ["CONFLICTING_SNAPSHOTS_UNKNOWN", "UNKNOWN_NO_QUALIFIED_LOCAL_REPORT"]
    assert result.sealed_fund.isna().all()
    assert result.parent_connection_observed.isna().all()


def test_capture_scaffold_preserves_full_pool(tmp_path):
    path = pool(tmp_path)
    contract = q.future_capture_contract(path)
    assert q.validate_capture_contract(contract, path)["pool_rows"] == 2
    contract["rows"].pop()
    with pytest.raises(ValueError, match="full candidate"):
        q.validate_capture_contract(contract, path)


def test_unknown_has_no_zero_placeholder(tmp_path):
    path = pool(tmp_path); contract = q.future_capture_contract(path)
    contract["rows"][0]["reported_event_fields"]["bomb_count"]["value"] = 0
    with pytest.raises(ValueError, match="placeholder"):
        q.validate_capture_contract(contract, path)


def test_target_selected_input_rejected(tmp_path):
    path = pool(tmp_path); contract = q.future_capture_contract(path)
    contract["rows"][0]["promotion_outcome_used_for_extraction"] = True
    with pytest.raises(ValueError, match="target outcomes"):
        q.validate_capture_contract(contract, path)


def test_capture_field_exact_pointer_and_semantics(tmp_path):
    path = pool(tmp_path); contract = q.future_capture_contract(path)
    raw = save(tmp_path / "capture.json", provider())
    source = {"source_id": "one", "url": "https://provider.test/pool", "requested_date": "2026-09-30",
              "requested_at": "2026-09-30T07:00:01+00:00", "received_at": "2026-09-30T07:00:02+00:00",
              "raw_path": str(raw), "raw_sha256": q.sha(raw), "http_status": 200,
              "market_snapshot_at": None, "field_definitions": "Original provider reported count; snapshot time unknown"}
    contract["sources"] = [source]
    item = {"value": 0, "status": "PROVIDER_REPORTED", "source_id": "one",
            "original_provider_report": True, "raw_json_pointer": "/data/pool/0/zbc"}
    contract["rows"][0]["reported_event_fields"]["bomb_count"] = item
    assert q.validate_capture_contract(contract, path)["nonunknown_fields"] == 1
    item["value"] = 1
    with pytest.raises(ValueError, match="differs"):
        q.validate_capture_contract(contract, path)
