"""Independent read-only receipt and full-pool recomputation, without model fits."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
DELIVERY = OUT / "actual_local_qualification_v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


report = read(DELIVERY / "report.json")
roster = read(DELIVERY / "inventory_BEFORE_AUDIT.json")["files"]
assert len(roster) == len({r["path"] for r in roster}) == 75
assert all(sha(r["path"]) == r["sha256"] for r in roster)
assert all(sha(path) == digest for path, digest in report["sources"].items())
prepared = joblib.load(ROOT / "research/model_upgrade_validation/relay_context_20261011T151204Z/prepared_source_reconstruction.joblib")["relay"]
qualified = pd.read_csv(DELIVERY / "complete9245_information_qualification.csv", dtype={"code": str})
assert len(qualified) == len(prepared) == 9245
assert set(zip(qualified.date, qualified.code)) == set(zip(prepared.date, prepared.code))
assert not qualified.duplicated(["date", "code"]).any()
assert not qualified.training_or_prediction_changed.any()
assert qualified[["parent_connection_observed", "child_continues_parent_break", "child_restarts_parent_break"]].isna().all().all()
assert qualified.parent_information_state.eq("UNKNOWN_NO_COMPLETE_HISTORICAL_RELATION_INPUT").all()
old = pd.read_csv(ROOT / "data/v275/reported_seal_archive_20261002_v2/reported_seal_rows.csv", dtype={"code": str})
old = old[old.boards.ge(2)].set_index(["date", "code"])
joined = qualified[qualified.reported_event_state.eq("SOURCE_BOUND_REPORTS_AGREE")].set_index(["date", "code"])
assert set(joined.index) == set(old.index)
for field in ["first_seal", "last_seal", "bomb_count", "sealed_fund", "amount", "float_cap"]:
    assert np.allclose(joined[field], old.loc[joined.index, field], atol=1e-8, rtol=1e-12, equal_nan=True)
assert joined.reported_clock_contract_valid.sum() == 191
provisional = read(DELIVERY / "all_unqualified_report_style_csv_relay_rows.json")
assert len(provisional) == 184
for row in provisional:
    path = Path(row["source_path"])
    assert sha(path) == row["source_sha256"]
    frame = pd.read_csv(path, dtype={"代码": str})
    source = frame[frame["代码"].str.zfill(6).eq(row["code"])].iloc[0]
    for native, field in [("首次封板时间", "first_seal"), ("最后封板时间", "last_seal"), ("炸板次数", "bomb_count"),
                          ("封板资金", "sealed_fund"), ("成交额", "amount"), ("流通市值", "float_cap")]:
        assert float(source[native]) == float(row["reported_style_values"][field])
    assert row["original_time_eligible"] is False and row["currently_qualified_for_fixed_challenge"] is False
sources = read(DELIVERY / "all_source_qualification.json")
assert dict(Counter(r["status"] for r in sources)) == report["source_status_counts"]
for row in sources:
    if row["status"] == "SOURCE_DATE_MISMATCH":
        assert row["requested_date"] != row["response_date"]
    if row["original_receipt_hash_present"]:
        assert row["original_receipt_hash_match"] is True
banks = []
for path in sorted((ROOT / "research/model_upgrade_validation").rglob("*evidence_bank.json")):
    bank = read(path)
    if not isinstance(bank, dict) or not isinstance(bank.get("rows"), list):
        continue
    rows = bank["rows"]
    groups = Counter(k for row in rows for k in row.get("groups", {}))
    anchor_mentions = json.dumps(bank, ensure_ascii=False).count('"anchor_code"')
    banks.append({"path": str(path), "sha256": sha(path), "rows": len(rows), "group_counts": dict(groups),
                  "explicit_dependency_groups": groups.get("dependency", 0),
                  "anchor_code_fields": anchor_mentions,
                  "source_existence_is_not_confirmed_dependency_or_negative_evidence": True})
assert sum(b["explicit_dependency_groups"] for b in banks) == 0
assert sum(b["anchor_code_fields"] for b in banks) == 0
contract = read(DELIVERY / "full15_future_information_capture_PENDING.json")
pool = read(contract["pool_path"])
assert len(contract["rows"]) == len(pool["rows"]) == 15
assert {r["code"] for r in contract["rows"]} == {r["code"] for r in pool["rows"]}
assert all(i["status"] == "UNKNOWN" and i["value"] is None for row in contract["rows"] for i in row["reported_event_fields"].values())
verification = {"status": "INDEPENDENT_COMPLETE_SOURCE_AND_POOL_CHECKS_PASS_NOT_ACCURACY_GAIN",
                "fit_count": 0, "source_files_hash_checked": len(roster), "full_pool_rows": 9245,
                "old217_all6_reported_fields_match": True, "valid_old_clock_rows_unchanged": 191,
                "provisional184_all_raw_fields_match": True, "historical_role_banks_examined": len(banks),
                "historical_dependency_groups": 0, "historical_parent_anchor_fields": 0,
                "current_full15_unknown_capture_scaffold_validated": True,
                "prediction_accuracy_improvement_proved": False,
                "source": {str(Path(__file__).resolve()): sha(__file__), str(DELIVERY / "report.json"): sha(DELIVERY / "report.json")}}
(OUT / "historical_relation_bank_inventory.json").write_text(json.dumps(banks, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "independent_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(verification, ensure_ascii=True))
