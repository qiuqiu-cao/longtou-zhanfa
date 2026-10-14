"""Recover exact audit preimage from the single recorded additive patch."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
current = (ROOT / "relay_information_qualification.py").read_text(encoding="utf-8")
initial = current.replace("from zoneinfo import ZoneInfo\n", "")
initial = initial.replace('    received = aware_stamp(evidence["available_at"])\n    if received and received.astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat() == day:\n        local = received.astimezone(ZoneInfo("Asia/Shanghai"))\n        if (local.hour, local.minute, local.second) < (15, 0, 0):\n            evidence["capture_scope"] = "PRE_CLOSE_RECEIVE_TIME_NOT_SIGNAL_CLOSE_PROOF"\n', '')
initial = initial.replace('        comparable = [x for x in candidates if x["boards"] == float(original["boards"])\n                      and x["capture_scope"] not in {"INTRADAY", "PRE_CLOSE_RECEIVE_TIME_NOT_SIGNAL_CLOSE_PROOF"}]', '        comparable = [x for x in candidates if x["boards"] == float(original["boards"]) and x["capture_scope"] != "INTRADAY"]')
start = initial.index("    unqualified = []\n")
end = initial.index('    projected.to_csv(output / "complete9245_information_qualification.csv"', start)
initial = initial[:start] + initial[end:]
initial = initial.replace('                        "unqualified_report_style_csv_keys": int(training[["date", "code"]].apply(tuple, axis=1).isin(provisional_keys).sum()),\n', '')
initial = initial.replace('              "unqualified_report_style_csv_relay_rows": len(unqualified),\n              "unqualified_report_style_unique_relay_keys": len(provisional_keys),\n              "unqualified_report_style_full_historical_key_overlap": int(projected.unqualified_report_style_csv_present.sum()),\n', '')
report = json.loads((OUT / "actual_local_qualification/report.json").read_text(encoding="utf-8"))
expected = report["sources"][str(ROOT / "relay_information_qualification.py")]
actual = hashlib.sha256(initial.encode("utf-8")).hexdigest()
assert actual == expected, (actual, expected)
(OUT / "relay_information_qualification.initial_source.py").write_text(initial, encoding="utf-8", newline="\n")
receipt = {"initial_source_sha256": actual, "current_source_sha256": hashlib.sha256(current.encode("utf-8")).hexdigest(),
    "initial_report_preserved": True, "initial_source_recovered_exact_sha_from_recorded_patch": True,
    "repair": "Exclude same-day pre-close receive timestamps from close comparison; distinguish5placeholderCSV from15unqualifiedreport-styleCSV and preserve184rows", "fit_count": 0}
(OUT / "initial_audit_and_additive_repair_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(receipt, ensure_ascii=True))
