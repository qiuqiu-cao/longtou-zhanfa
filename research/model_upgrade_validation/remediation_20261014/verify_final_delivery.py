"""Reproduce the current additive remediation against preserved local sources."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import stock_judgement_review as old
import relay_information_qualification as data
import relay_review_workbench as workbench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    protected = old.read(ROOT / "research/model_upgrade_validation/project_audit_20261014/verification.json")["source_bindings"]
    old.check_sources(protected)
    ledgers = [ROOT / "data/v293/settlement_ledger.jsonl", ROOT / "data/independent_stock_judgement/registry.jsonl",
        ROOT / "data/independent_stock_judgement/settlement_ledger.jsonl"]
    before = {str(p): old.sha(p) for p in ledgers}
    q = workbench.qualify_information(out / "data")
    roster = data.read(out / "data/inventory_BEFORE_AUDIT.json")["files"]
    assert len(roster) == 75 and all(old.sha(r["path"]) == r["sha256"] for r in roster)
    old.check_sources(q["sources"])
    receipt = data.read(ROOT / "research/model_upgrade_validation/relay_context_20261011T151204Z/source_reconstruction_BEFORE_FITS.json")
    full = joblib.load(receipt["prepared_path"])["relay"]
    actual = pd.read_csv(out / "data/complete9245_information_qualification.csv", dtype={"code": str})
    assert len(actual) == len(full) == 9245
    assert set(zip(actual.date, actual.code)) == set(zip(full.date, full.code))
    assert not actual.duplicated(["date", "code"]).any()
    assert actual[["parent_connection_observed", "child_continues_parent_break", "child_restarts_parent_break"]].isna().all().all()
    reference = pd.read_csv(ROOT / "data/v275/reported_seal_archive_20261002_v2/reported_seal_rows.csv", dtype={"code": str})
    reference = reference[reference.boards.ge(2)].set_index(["date", "code"])
    qualified = actual[actual.reported_event_state.eq("SOURCE_BOUND_REPORTS_AGREE")].set_index(["date", "code"])
    assert set(qualified.index) == set(reference.index) and len(qualified) == 217
    for field in data.FIELDS:
        assert np.allclose(qualified[field], reference.loc[qualified.index, field], equal_nan=True, atol=1e-8, rtol=1e-12)
    assert qualified.reported_clock_contract_valid.sum() == 191
    pending = data.read(out / "data/full15_future_information_capture_PENDING.json")
    assert data.validate_capture_contract(pending, pending["pool_path"])["pool_rows"] == 15
    report_style = data.read(out / "data/all_unqualified_report_style_csv_relay_rows.json")
    assert len(report_style) == 184
    frames = {}
    for row in report_style:
        path = row["source_path"]
        assert old.sha(path) == row["source_sha256"]
        if path not in frames:
            frames[path] = pd.read_csv(path, dtype={"代码": str}).rename(columns=data.MAP)
        source = frames[path][frames[path].code.str.zfill(6).eq(row["code"])].iloc[0]
        for field in data.FIELDS:
            assert float(source[field]) == float(row["reported_style_values"][field])
        assert not row["original_time_eligible"] and not row["currently_qualified_for_fixed_challenge"]
    sources = data.read(out / "data/all_source_qualification.json")
    assert dict(Counter(r["status"] for r in sources)) == q["source_status_counts"]
    script = ROOT / "research/model_upgrade_validation/remediation_20261014/workbench/verify_actual_workbench.py"
    spec = importlib.util.spec_from_file_location("saved_workbench_verifier", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.main(["--output-dir", str(out / "workbench")])
    s = workbench.validation_status(out / "validation")
    formal = s["formal_promotion"]["reverified_promotion_metrics"]
    assert s["formal_promotion"]["latest_settled_signal_days"] == 3
    assert (formal["registered"], formal["known_labels"], formal["TP"] + formal["TN"]) == (39, 38, 28)
    assert not s["formal_promotion"]["aggregate_withheld"]
    assert s["independent_comparison"]["status"] == "SEPARATE_INDEPENDENT_LEDGER_REVERIFIED"
    assert s["forum_pending"]["new_forecast_abstentions"] == s["new_formal_mature_samples"] == 0
    base = ROOT / "research/model_upgrade_validation/forum_reasoning_20261014/actual_full15"
    e = workbench.execution(base / "facts_with_forum_reasoning.json", base / "author_argument_completed.json",
        ROOT / "results/daily_review_20261014T1118Z/15日完整15只条件判断.json",
        ROOT / "data/v298/exchange_calendar_extension_20261003_v2/calendar.json", out / "execution",
        asof=datetime.now(timezone.utc).isoformat())
    assert e["pool_size"] == 15 and e["actual_execution_records"] == e["paper_execution_records"] == 0
    assert all(r["actual_complete_trade_net_return"] is None for r in e["rows"])
    assert e["funnel"]["cash_day_rate"] is None
    old.check_sources(e["source_bindings"])
    tests = ["tests/test_relay_review_workbench.py", "tests/test_relay_information_qualification.py",
        "tests/test_forum_execution_review.py", "tests/test_relay_validation_status.py", "tests/test_forum_reasoning_review.py",
        "tests/test_forum_reasoning_outcome_context.py", "tests/test_supplement_relation_review.py", "tests/test_supplement_role_review.py"]
    tested = subprocess.run([sys.executable, "-m", "pytest", *tests, "-q"], cwd=ROOT, capture_output=True, text=True)
    (out / "focused_tests.log").write_text(tested.stdout + tested.stderr, encoding="utf-8")
    assert tested.returncode == 0
    old.check_sources(protected); old.check_sources(before)
    files = [ROOT / p for p in ["relay_review_workbench.py", "relay_information_qualification.py",
        "forum_execution_review.py", "relay_validation_status.py", *tests]]
    files += [Path(__file__).resolve(), script, out / "focused_tests.log", out / "data/report.json",
        out / "workbench/actual_verification.json", out / "validation/report.json", out / "execution/execution_review.json"]
    result = {"status": "CURRENT_IMPLEMENTATION_COMPLETE_LOCAL_VERIFICATION_PASS", "verified_at": datetime.now(timezone.utc).isoformat(),
        "protected_source_bindings": protected, "protected_ledgers": before,
        "source_bindings": {str(p): old.sha(p) for p in files}, "focused_tests_exit_code": tested.returncode,
        "full_historical_pool": 9245, "sources_checked": 75, "qualified_report_keys": 217, "provisional_report_style_rows": 184,
        "complete_current_pool": 15, "formal_signal_days": 3, "formal_registered": 39, "formal_known": 38, "formal_correct": 28,
        "new_fits": 0, "new_formal_forecasts": 0, "new_mature_samples": 0, "actual_orders_or_fills": 0,
        "prediction_accuracy_improvement_proved": False, "all_predictive_and_profit_deficits_resolved": False,
        "source_preimages": {str(p): old.sha(p) for p in (ROOT / "research/model_upgrade_validation/remediation_20261014/root_source_preimages").glob("*.py")}}
    old.write_new(out / "verification.json", result)
    print(json.dumps({k: result[k] for k in ("status", "full_historical_pool", "complete_current_pool", "new_fits", "new_mature_samples")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
