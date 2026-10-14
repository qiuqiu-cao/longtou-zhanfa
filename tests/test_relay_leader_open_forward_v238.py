import json

from relay_leader_open_forward_v238 import freeze, record, verify


def _files(tmp_path):
    script = tmp_path / "confirm.py"
    script.write_text("RULE=1\n", encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "version": "v237-relay-leader-open-confirmation-1",
        "signal_date": "2026-09-21", "decision_date": "2026-09-22",
        "decision": "DATA_UNAVAILABLE", "entry_authorized": False,
    }), encoding="utf-8")
    return script, report


def test_freeze_record_and_verify(tmp_path):
    script, report = _files(tmp_path)
    registry = tmp_path / "registry"
    freeze(str(registry), str(script), str(report))
    event = record(str(registry), str(report))
    assert event["decision"] == "DATA_UNAVAILABLE"
    assert verify(str(registry))["status"] == "PASS"


def test_confirmation_tampering_is_detected(tmp_path):
    script, report = _files(tmp_path)
    registry = tmp_path / "registry"
    frozen = freeze(str(registry), str(script), str(report))
    record(str(registry), str(report))
    path = registry / "confirmations" / frozen["freeze_id"] / "2026-09-21_for_2026-09-22.json"
    path.write_text("{}", encoding="utf-8")
    assert verify(str(registry))["status"] == "FAIL"
