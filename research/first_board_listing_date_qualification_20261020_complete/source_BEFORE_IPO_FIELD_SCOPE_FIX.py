"""One independent first-close trial of actual IPO tenure; Gemini 4 reviews afterwards."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import shutil
import time

import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import first_board_matured_event_model as history
import first_board_drop_clock as old_ablation
import numpy as np
import pandas as pd

QUAL = base.ROOT / "research/first_board_listing_date_qualification_20261020_complete"
REFERENCE = risk.REFERENCE
SOURCE = REFERENCE / "all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
SCREEN = history.SCREEN / "COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv"
MASTER = base.INPUTS["security_intervals"]
RAW_LISTS = [
    base.ROOT / "data/v4/security_evidence/szse_mainboard_list_20260831.csv",
    base.ROOT / "data/v5/security_evidence/sse_current/sse_current_mainboard.csv",
    base.ROOT / "data/v7/security_evidence/sse_breadth_gap/sse_current_mainboard.csv",
    base.ROOT / "data/v49/security_sse_asof_20260904/sse_current_mainboard.csv",
    base.ROOT / "data/v4/historical_universe/official_delisted_mainboard.csv",
]
OLD = "listing_age_sessions"
NEW = "listing_calendar_age_days"
ORIGINAL_FIELDS = stage.VARIANTS["STAGE_CYCLE_GENE"].copy()
FIELDS = [NEW if field == OLD else field for field in ORIGINAL_FIELDS]
VARIANT = "ACTUAL_IPO_CALENDAR_TENURE_62"
KEYS = ["signal_date", "code"]


def derive_age(keys, identities):
    if keys.duplicated(KEYS).any() or identities.code.duplicated().any():
        raise ValueError("Duplicate identity or complete-pool key")
    out = keys.merge(identities, on="code", how="left", validate="many_to_one", sort=False)
    listed = pd.to_datetime(out.listing_date, format="%Y-%m-%d", errors="coerce")
    signal = pd.to_datetime(out.signal_date, format="%Y-%m-%d", errors="raise")
    valid = out.identity_qualified.fillna(False).astype(bool) & listed.notna() & listed.le(signal)
    out[NEW] = (signal - listed).dt.days.astype(float).where(valid)
    # This is calendar tenure. No pre-2023 official-session count is invented.
    return out


def qualify():
    if QUAL.exists():
        raise FileExistsError("Preserve the fixed source qualification")
    QUAL.mkdir(parents=True)
    paths = [MASTER, SOURCE, *RAW_LISTS]
    protocol = dict(at=base.now(), task="Outcome-blind qualification of stable actual listing dates for one replacement feature",
                    hypothesis="A real pre-archive listing date supplies cross-stock tenure absent from the left-censored archive clock; this is not another deletion/cap of the old field",
                    field=NEW, definition="Signal calendar date minus actual IPO listing calendar date, in days; no log, cap, bins or optimized cutoffs",
                    source_policy="Existing official-list CSVs must reproduce the master date; missing/conflicting/future listing dates remain NaN. Historical source retrieval clocks are not backdated.",
                    no_outcome_columns_read=True, no_new_network_requests=True, coverage_gate="At least95% actual-age coverage and100 observed rows in development and2025H1 validation",
                    bindings={str(p.resolve()): base.sha(p) for p in paths}, source_sha256=base.sha(__file__), fits=0)
    base.write_json(QUAL / "protocol_BEFORE_DATA_NO_OUTCOMES.json", protocol)
    master = pd.read_csv(MASTER, dtype=str)
    identity = []
    for code, part in master.groupby("code", sort=True):
        dates = part.listing_date.dropna().unique()
        official = part.source_type.isin(["szse_official", "sse_official"]).all()
        declared = part.source_verified.str.lower().isin(["true", "1"]).all()
        identity.append(dict(code=code, listing_date=dates[0] if len(dates) == 1 else None,
                             master_date_count=len(dates), master_declared_official=bool(official and declared),
                             master_row_pointers="|".join(str(i + 2) for i in part.index),
                             source_clock_original="|".join(sorted(part.source_timestamp.dropna().unique()))))
    identities = pd.DataFrame(identity)
    raw = []
    for path in RAW_LISTS:
        frame = pd.read_csv(path, dtype=str)
        if "A_STOCK_CODE" in frame:
            codes = frame.A_STOCK_CODE.str.zfill(6)
            dates = pd.to_datetime(frame.LIST_DATE, format="%Y%m%d", errors="raise").dt.strftime("%Y-%m-%d")
        else:
            codes, dates = frame.code.str.zfill(6), frame.listing_date
        for idx, code, listed in zip(frame.index, codes, dates):
            raw.append(dict(code=code, raw_listing_date=listed, pointer=f"{path.relative_to(base.ROOT).as_posix()}:csv_record={idx + 2}"))
    raw_frame = pd.DataFrame(raw)
    by_code = {code: part for code, part in raw_frame.groupby("code", sort=False)}
    match, conflicts, pointers = [], [], []
    for row in identities.itertuples():
        records = by_code.get(row.code, raw_frame.iloc[:0])
        dates = set(records.raw_listing_date.dropna())
        good = bool(row.master_date_count == 1 and row.master_declared_official and len(dates) == 1 and row.listing_date in dates)
        match.append(good)
        conflicts.append(bool(dates and (len(dates) != 1 or row.listing_date not in dates)))
        pointers.append("|".join(records.pointer))
    identities["identity_qualified"] = match
    identities["raw_date_conflict"] = conflicts
    identities["raw_row_pointers"] = pointers
    identities.to_csv(QUAL / "LISTING_DATE_IDENTITIES_WITH_RAW_POINTERS.csv", index=False, encoding="utf-8-sig")
    keys = pd.read_csv(SOURCE, dtype={"code": str}, usecols=KEYS + ["stage", OLD], float_precision="round_trip")
    keys = keys[keys.stage.eq(1)].reset_index(drop=True)
    output = derive_age(keys, identities[["code", "listing_date", "identity_qualified"]])
    for row in output[output[NEW].notna()].itertuples():
        assert getattr(row, NEW) == (date.fromisoformat(row.signal_date) - date.fromisoformat(row.listing_date)).days
    checks = []
    for cutoff in ("2025-06-30", "2025-12-31"):
        prefix = derive_age(keys[keys.signal_date.le(cutoff)], identities[["code", "listing_date", "identity_qualified"]])
        np.testing.assert_allclose(prefix[NEW], output.loc[output.signal_date.le(cutoff), NEW], equal_nan=True, rtol=0, atol=0)
        checks.append(dict(cutoff=cutoff, rows=len(prefix), exact_equal=True))
    output.to_csv(QUAL / "COMPLETE_FIRST_BOARD_ACTUAL_AGE_NO_OUTCOMES.csv", index=False, encoding="utf-8-sig")
    periods = {}
    for name, mask in [("development", output.signal_date.lt("2025-01-01")), ("validation2025H1", output.signal_date.between("2025-01-01", "2025-06-30")), ("evaluation", output.signal_date.ge("2025-07-01"))]:
        part = output[mask]
        groups = part.groupby(["signal_date", OLD])[NEW].nunique()
        periods[name] = dict(rows=len(part), known=int(part[NEW].notna().sum()), unknown=int(part[NEW].isna().sum()),
                             coverage=float(part[NEW].notna().mean()), archive_age_groups_with_multiple_actual_tenures=int(groups.gt(1).sum()),
                             distinct_actual_age_values=int(part[NEW].nunique()))
    good = all(periods[p]["coverage"] >= .95 and periods[p]["known"] >= 100 for p in ("development", "validation2025H1"))
    report = dict(at=base.now(), qualified=good, rows=len(output), periods=periods, prefix_checks=checks,
                  independent_calendar_arithmetic_all_rows=True, raw_conflict_codes=identities.loc[identities.raw_date_conflict, "code"].tolist(),
                  current_universe_survivorship_unchanged=True, acquisition_late_not_original_pit_capture=True,
                  publication_clock="Listing event date is a stable historical fact; original pre-signal HTTP acquisition is not certified",
                  field_is_calendar_days_not_official_sessions=True, no_labels_read=True, fits=0)
    base.write_json(QUAL / "qualification.json", report)
    print(json.dumps(base.clean(report), ensure_ascii=True), flush=True)
    return report


def attach_qualified_age(rows, extra):
    columns = KEYS + [NEW]
    if rows.duplicated(KEYS).any() or extra.duplicated(KEYS).any():
        raise ValueError("Duplicate age feature keys")
    if set(map(tuple, rows[KEYS].to_numpy())) != set(map(tuple, extra[KEYS].to_numpy())):
        raise ValueError("Incomplete unchanged first-board denominator")
    out = rows.merge(extra[columns], on=KEYS, validate="one_to_one", sort=False)
    np.testing.assert_allclose(rows[ORIGINAL_FIELDS + [stage.TARGET]], out[ORIGINAL_FIELDS + [stage.TARGET]], equal_nan=True, rtol=0, atol=0)
    if out[NEW].dropna().lt(0).any():
        raise ValueError("Future listing date")
    return out


def run(output):
    began = time.monotonic()
    out = Path(output)
    if out.exists():
        raise FileExistsError("One fixed comparison, no overwrite or effect-driven retry")
    assert len(FIELDS) == 62 and len(set(FIELDS)) == 62 and OLD not in FIELDS
    qualification = json.loads((QUAL / "qualification.json").read_text("utf-8"))
    assert qualification["qualified"]
    inputs = [SOURCE, SCREEN, QUAL / "qualification.json", QUAL / "COMPLETE_FIRST_BOARD_ACTUAL_AGE_NO_OUTCOMES.csv", QUAL / "LISTING_DATE_IDENTITIES_WITH_RAW_POINTERS.csv", MASTER, *RAW_LISTS]
    protected = [Path(base.__file__), Path(stage.__file__), Path(risk.__file__), Path(old_ablation.__file__),
                 REFERENCE / "report.json", REFERENCE / "S1_ALL_PREDICTIONS.csv", REFERENCE / "S2_ALL_PREDICTIONS.csv"]
    protocol = dict(at=base.now(), budget_minutes=30, trainer="Codex only", external_reviewer="Gemini4 only after independent training and saved verification",
                    hypothesis="Replace a left-censored archive-position field with actual historical IPO calendar tenure; supply cross-stock listing information absent from original age clock",
                    old_failed_drop_clock="Preserved failed61-field deletion; this new62-field comparison uses separately qualified actualIPOdate information, not capping/deleting/refitting the same failed arm",
                    fields=FIELDS, replacement={"old": OLD, "new": NEW, "definition": "Exact calendar days since qualified listing date; no caps/logs/bins/threshold fitting"},
                    baseline="Exact saved62-field baseline plus unchanged capital/thin-turnover screen. No baseline or second-stage fit",
                    training="Identical mature members, equal-day weights, originalHGB160/15/leaf60/lr.06/l2=8/balanced/seed42; unchanged last42-day temporal PlattC1",
                    validation="Original2025H1 full4724rows/117days; require>52screenedTop1hits,>=48non-one-pricehits,>=95%decision-datecoverage. Baseline wins ties. Save all predictions first.",
                    evaluation="Onlyifpass:14monthly original252-official-position windows with identical mature members. Preserve unknowns/full pools; same-stock second-stage saved scores reused only after actual promotion.",
                    acceptance="DailyTop1 next-official-close1to2>=60%,>=200known,>=95%outcome coverage,no abstention dates;2025H2 and2026 each>=55%",
                    stop="One fixed arm; finish comparison or30-minute checkpoint. No age transformation/cutoff/other feature changes after results",
                    inputs={str(p.resolve()): base.sha(p) for p in inputs}, protected={str(p.resolve()): base.sha(p) for p in protected},
                    raw_sources={str(p.resolve()): base.sha(p) for p in base.INPUTS.values()}, source_sha256=base.sha(__file__),
                    new_forward_results=0, actual_trades=0, production_changes=False, already_studied_history_not_new_blind=True)
    out.mkdir(parents=True)
    base.write_json(out / "protocol_BEFORE_ANY_FIT.json", protocol)
    shutil.copy2(__file__, out / "source_before_fit.py")
    with (QUAL / "SINGLE_FIXED_COMPARISON_ONCE.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(at=base.now(), output=str(out.resolve()), protocol_sha256=base.sha(out / "protocol_BEFORE_ANY_FIT.json")), stream, indent=2)
    rows = pd.read_csv(SOURCE, dtype={"code": str}, float_precision="round_trip")
    rows = rows[rows.stage.eq(1)].reset_index(drop=True)
    extra = pd.read_csv(QUAL / "COMPLETE_FIRST_BOARD_ACTUAL_AGE_NO_OUTCOMES.csv", dtype={"code": str}, float_precision="round_trip")
    rows = attach_qualified_age(rows, extra)
    screen = pd.read_csv(SCREEN, dtype={"code": str}, float_precision="round_trip")
    rows = rows.merge(screen[KEYS + ["research_selection_allowed", "exclusion_reason", "threeboard_evidence_status"]], on=KEYS, validate="one_to_one", sort=False)
    rows.to_csv(out / "COMPLETE_FIRST_BOARD_UNCHANGED_FACTS_LABELS_AND_ACTUAL_AGE.csv", index=False, encoding="utf-8-sig")
    train = old_ablation.train_members(rows, "2025-01-01")
    original_members = pd.read_csv(REFERENCE / "models/validation_S1_STAGE_CYCLE_GENE_full_members.csv", dtype={"code": str}, float_precision="round_trip")
    assert set(map(tuple, train[KEYS + ["label_end", stage.TARGET]].to_numpy())) == set(map(tuple, original_members[KEYS + ["label_end", stage.TARGET]].to_numpy()))
    val = rows[rows.signal_date.between("2025-01-01", "2025-06-30")]
    assert len(val) == 4724 and val.signal_date.nunique() == 117
    frozen = risk.attach_frozen(rows, REFERENCE / "validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
    baseline = risk.screen_metric(frozen)
    baseline_non = risk.screen_metric(frozen[frozen.one_price.eq(0)])
    assert baseline["hits"] == 52 and baseline_non["hits"] == 48
    stage.VARIANTS[VARIANT] = FIELDS
    head = stage.fit_head(train, VARIANT, out, "validation_actual_listing_age")
    scores = stage.predict(head, val)
    scores.to_csv(out / "VALIDATION_ALL_PREDICTIONS_BEFORE_METRICS.csv", index=False, encoding="utf-8-sig")
    primary, non = risk.screen_metric(scores), risk.screen_metric(scores[scores.one_price.eq(0)])
    passed = bool(primary["hits"] > 52 and non["hits"] >= 48 and primary["decision_date_coverage"] >= .95)
    decision = dict(at=base.now(), baseline=baseline, baseline_non_one_price=baseline_non, challenger=primary,
                    challenger_non_one_price=non, unfiltered_challenger=stage.metric(scores), pass_validation=passed)
    base.write_json(out / "selection_BEFORE_EVALUATION.json", decision)
    print(json.dumps(base.clean(decision), ensure_ascii=True), flush=True)
    report = dict(at=None, decision=decision, goal_60_passed=False, trainer="Codex", reviewer="Gemini4_PENDING_REVIEW", production_deployment_allowed=False, actual_trades=0, new_forward_results=0)
    if passed:
        sessions = json.loads(base.INPUTS["calendar"].read_text("utf-8"))["sessions"]
        parts = []
        for month, part in rows[rows.signal_date.ge("2025-07-01")].groupby(rows.signal_date.str[:7], sort=True):
            if time.monotonic() - began >= 29 * 60:
                base.write_json(out / "CHECKPOINT_30MIN.json", dict(at=base.now(), completed_months=len(parts), next_month=month, status="FIXED_PROTOCOL_INCOMPLETE_NO_RETUNING"))
                return None
            boundary = month + "-01"
            position = int(np.searchsorted(sessions, boundary))
            members = old_ablation.train_members(rows, boundary, position - 252)
            old = pd.read_csv(REFERENCE / "models" / (month + "_S1_STAGE_CYCLE_GENE_full_members.csv"), dtype={"code": str}, float_precision="round_trip")
            assert set(map(tuple, members[KEYS + ["label_end", stage.TARGET]].to_numpy())) == set(map(tuple, old[KEYS + ["label_end", stage.TARGET]].to_numpy()))
            bundle = stage.fit_head(members, VARIANT, out, month + "_actual_listing_age")
            assert bundle["train_max_label_end"] < part.signal_date.min()
            predictions = stage.predict(bundle, part)
            predictions["model_month"] = month
            predictions.to_csv(out / (month + "_ALL_PREDICTIONS_BEFORE_METRICS.csv"), index=False, encoding="utf-8-sig")
            parts.append(predictions)
            print(month + " saved", flush=True)
        assert len(parts) == 14
        evaluation = pd.concat(parts, ignore_index=True)
        evaluation.to_csv(out / "EVALUATION_ALL_PREDICTIONS.csv", index=False, encoding="utf-8-sig")
        reference = risk.attach_frozen(rows, REFERENCE / "S1_ALL_PREDICTIONS.csv")
        report["evaluation"] = history.period_metrics(evaluation, reference)
        allowed = evaluation[evaluation.research_selection_allowed]
        second = pd.read_csv(REFERENCE / "S2_ALL_PREDICTIONS.csv", dtype={"code": str}, float_precision="round_trip")
        chain, report["same_stock_second_recheck"] = stage.chain_recheck(allowed, second)
        chain.to_csv(out / "SAME_STOCK_SECOND_RECHECK.csv", index=False, encoding="utf-8-sig")
        report["calibration"] = stage.calibration_metrics(allowed)
        m = report["evaluation"]["all"]["new_screened"]
        report["goal_60_passed"] = bool(m["hit_rate"] >= .60 and m["known"] >= 200 and m["outcome_coverage"] >= .95 and m["abstention_dates"] == 0
              and all(report["evaluation"][period]["new_screened"]["hit_rate"] >= .55 for period in ("2025H2", "2026")))
    for path, digest in {**protocol["inputs"], **protocol["protected"], **protocol["raw_sources"]}.items():
        assert base.sha(path) == digest
    report.update(at=base.now(), status="FIXED_MONTHLY_COMPARISON_FINISHED" if passed else "REJECTED_VALIDATION_NO_MONTHLY_FITS",
                  core_fit_count=len(list((out / "models").glob("*_COMPLETED.json"))), baseline_refits=0, second_stage_refits=0)
    base.write_json(out / "report.json", report)
    print(json.dumps(base.clean({key: report[key] for key in ("status", "goal_60_passed", "core_fit_count")}), ensure_ascii=True), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["qualify", "train"])
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.command == "qualify":
        qualify()
    else:
        if not args.output:
            parser.error("train requires --output")
        run(args.output)
