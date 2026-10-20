"""First-close capital and conservative thin-turnover screening experiment.

Risk proxies are not fund identities. All scores, exclusions, lost positives and
unknowns remain in the complete archived cohort. No later-board observation is
allowed to revise a first-board prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from urllib.parse import urlparse

import first_board_leader_model as base
import first_second_stage_model as stage
import historical_symbolic_context as symbolic
import joblib
import numpy as np
import pandas as pd

REFERENCE = base.ROOT / "research/first_second_stage_20261020_run1"
SYMBOLIC = base.ROOT / "research/first_board_symbolic_qualification_20261020"
SOURCES = base.ROOT / "research/first_board_capital_risk_20261020_sources"
CAP_FIELDS = ["cap_rank_in_first_pool", "log_cap_to_first_pool_median",
              "log_cap_excess_development80", "microcap_low_turnover_proxy",
              "microcap_one_price_proxy", "microcap_volume_compression_proxy"]
VARIANT = "FIRST_CLOSE_CAPITAL_CONTEXT"
FIELDS = stage.VARIANTS["STAGE_CYCLE_GENE"] + CAP_FIELDS + symbolic.FIELDS
RISK_POLICY = dict(microcap_yuan=3_000_000_000, low_turnover_percent=5.,
                   volume_ratio_max=1., upper_cap_quantile=.8,
                   upper_cap_training="Pre2025 calibration-prefix signals only, before the final42 development sessions",
                   provisional_exclusion="Microcap AND turnover<5% AND (current one-price OR current volume<previous5-session mean)",
                   identity="A conservative daily-bar risk proxy, NOT verified three-board-fund identity",
                   evidence_clock="Source publication must be no later than signal15:00 Asia/Shanghai; unknown is not false",
                   restructuring_exception="Only a dated verified pre-close restructuring disclosure; no selected retrospective case bank used for training")


def qualified_preclose_evidence(rows, records, strict_acquisition=False):
    """Bind author-reviewed original evidence, rejecting post-close revelations."""
    keys = set(zip(rows.signal_date, rows.code))
    accepted = {}
    for record in records:
        key = (record["signal_date"], record["code"])
        if key not in keys:
            raise ValueError("Evidence is outside the complete signal cohort")
        cutoff = pd.Timestamp(key[0] + "T15:00:00+08:00")
        published = pd.Timestamp(record["published_at"])
        acquired = pd.Timestamp(record["acquired_at"])
        if published.tzinfo is None or acquired.tzinfo is None:
            raise ValueError("Evidence timestamps require explicit timezones")
        if published > cutoff or (strict_acquisition and acquired > cutoff):
            raise ValueError("Evidence was not available at the signal close")
        if acquired < published:
            raise ValueError("Acquisition cannot precede publication")
        if record.get("review_status") != "AUTHOR_VERIFIED" or not record.get("rationale"):
            raise ValueError("Missing independent source review")
        raw = Path(record["raw_path"])
        if not raw.is_file() or base.sha(raw) != record["raw_sha256"]:
            raise ValueError("Raw evidence hash mismatch")
        kind = record["kind"]
        if kind == "RESTRUCTURING_DISCLOSURE":
            domain = (urlparse(record["source_url"]).hostname or "").lower()
            if not any(domain == d or domain.endswith("." + d) for d in ("cninfo.com.cn", "sse.com.cn", "szse.cn")):
                raise ValueError("Restructuring exception needs an official disclosure")
            field = "verified_restructuring_preclose"
        elif kind == "SUSPECTED_THREEBOARD_STRUCTURE":
            channels = set(record.get("independently_verified_channels", []))
            if len(channels & {"minute_abrupt_pull", "coordinated_seats", "repeated_order_cancellations", "verified_weak_sector"}) < 2:
                raise ValueError("Single-seat or daily-bar pattern cannot establish suspected fund structure")
            field = "verified_threeboard_risk_preclose"
        else:
            raise ValueError("Unsupported evidence kind")
        accepted.setdefault(key, {})[field] = 1.
    return accepted


def capital_context(rows, upper_cap, evidence=None, strict_acquisition=False):
    """Only current/past columns and same-date complete cohort enter features."""
    out = rows.copy()
    cap = pd.to_numeric(out.float_cap, errors="coerce").where(lambda x: x.gt(0))
    out["cap_rank_in_first_pool"] = cap.groupby(out.signal_date).rank(pct=True)
    median = cap.groupby(out.signal_date).transform("median")
    out["log_cap_to_first_pool_median"] = np.log(cap / median)
    out["log_cap_excess_development80"] = np.log(cap / upper_cap).clip(lower=0)
    small = cap.le(RISK_POLICY["microcap_yuan"]).astype(float).where(cap.notna())
    low = out.turnover.lt(RISK_POLICY["low_turnover_percent"]).astype(float).where(out.turnover.notna())
    compressed = out.volume_ratio_5.lt(RISK_POLICY["volume_ratio_max"]).astype(float).where(out.volume_ratio_5.notna())
    out["microcap_low_turnover_proxy"] = small * low
    out["microcap_one_price_proxy"] = small * out.one_price
    out["microcap_volume_compression_proxy"] = small * compressed
    # These unavailable evidence fields remain unknown, never zero-filled.
    for f in ("verified_restructuring_preclose", "verified_threeboard_risk_preclose",
              "abrupt_first_pull_verified", "coordinated_seats_verified", "weak_sector_verified"):
        if f in out and out[f].notna().any():
            raise ValueError("Evidence flags must come from dated reviewed raw evidence, not feature-file assertions")
        out[f] = np.nan
    accepted = qualified_preclose_evidence(out, evidence or [], strict_acquisition)
    for i, key in zip(out.index, zip(out.signal_date, out.code)):
        for field, value in accepted.get(key, {}).items():
            out.loc[i, field] = value
    out["threeboard_evidence_status"] = np.where(
        out.verified_threeboard_risk_preclose.eq(1), "SUSPECTED_WITH_DATED_EVIDENCE", "INSUFFICIENT_EVIDENCE")
    # Proxy-only exclusions are separately named and never relabelled as fund identities.
    thin = out.microcap_low_turnover_proxy.eq(1) & (out.one_price.eq(1) | compressed.eq(1))
    large = cap.gt(upper_cap) & ~out.verified_restructuring_preclose.eq(1)
    confirmed_risk = out.verified_threeboard_risk_preclose.eq(1)
    out["capital_missing_review"] = cap.isna()
    out["capital_oversize_review"] = large
    out["thin_turnover_proxy_excluded"] = thin
    out["threeboard_evidence_excluded"] = confirmed_risk
    out["research_selection_allowed"] = ~(cap.isna() | large | thin | confirmed_risk)
    out["exclusion_reason"] = ["|".join(label for flag, label in zip(flags, (
        "CAP_MISSING_REVIEW", "ABOVE_DEVELOPMENT_CAP_NO_VERIFIED_EXCEPTION",
        "CONSERVATIVE_MICROCAP_THIN_TURNOVER_PROXY", "DATED_THREEBOARD_RISK")) if flag)
        for flags in zip(cap.isna(), large, thin, confirmed_risk)]
    return out


def screen_metric(rows, score="model_score"):
    allowed = rows[rows.research_selection_allowed]
    m = stage.metric(allowed, score=score)
    # Recall uses the original complete positive pool, including excluded winners.
    total_positive = int(rows[stage.TARGET].eq(1).sum())
    m.update(full_positive_pool=total_positive,
             full_positive_recall=m["hits"] / total_positive if total_positive else None,
             false_negatives=total_positive - m["hits"],
             full_signal_dates=int(rows.signal_date.nunique()),
             abstention_dates=int(rows.signal_date.nunique() - allowed.signal_date.nunique()),
             decision_date_coverage=float(allowed.signal_date.nunique() / rows.signal_date.nunique()),
             successful_predictions_per_all_pool_dates=float(m["hits"] / rows.signal_date.nunique()),
             excluded_rows=int((~rows.research_selection_allowed).sum()),
             excluded_known_promoters=int((~rows.research_selection_allowed & rows[stage.TARGET].eq(1)).sum()),
             excluded_unknown=int((~rows.research_selection_allowed & rows[stage.TARGET].isna()).sum()))
    return m


def prefix_checks(rows, upper_cap):
    checks = []
    expected = capital_context(rows, upper_cap)
    for cutoff in ("2025-06-30", "2025-12-31"):
        p = capital_context(rows[rows.signal_date.le(cutoff)].drop(columns=[stage.TARGET, "label_end"]), upper_cap)
        old = expected[expected.signal_date.le(cutoff)]
        np.testing.assert_allclose(p[CAP_FIELDS], old[CAP_FIELDS], equal_nan=True, rtol=1e-12, atol=1e-12)
        assert p.research_selection_allowed.tolist() == old.research_selection_allowed.tolist()
        checks.append(dict(cutoff=cutoff, rows=len(p), all_six_fields_and_screen_equal=True, targets_absent=True))
    return checks


def load_rows():
    path = REFERENCE / "all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
    rows = pd.read_csv(path, dtype={"code": str}, float_precision="round_trip")
    rows = rows[rows.stage.eq(1)].copy()
    names = pd.read_csv(SYMBOLIC / "complete28503_first_board_dated_symbolic_features.csv", dtype={"code": str}, float_precision="round_trip")
    names = names.rename(columns={"date": "signal_date"})
    assert set(zip(rows.signal_date, rows.code)) == set(zip(names.signal_date, names.code))
    rows = rows.merge(names[["signal_date", "code"] + symbolic.FIELDS], on=["signal_date", "code"], validate="one_to_one")
    development = rows[rows.signal_date.lt("2025-01-01") & rows.label_end.lt("2025-01-01") & rows[stage.TARGET].notna()]
    cutoff = sorted(development.signal_date.unique())[-42]
    scale_train = development[development.label_end.lt(cutoff)]
    upper_cap = float(scale_train.float_cap.where(scale_train.float_cap.gt(0)).quantile(RISK_POLICY["upper_cap_quantile"]))
    return rows, upper_cap, dict(max_scale_signal=scale_train.signal_date.max(), max_scale_label_end=scale_train.label_end.max(),
                                 calibration_start=cutoff, rows=len(scale_train), upper_float_cap_yuan=upper_cap)


def attach_frozen(rows, file):
    ref = pd.read_csv(file, dtype={"code": str}, float_precision="round_trip")
    part = rows[rows.signal_date.isin(ref.signal_date.unique())].copy()
    assert set(zip(part.signal_date, part.code)) == set(zip(ref.signal_date, ref.code))
    old = part.set_index(["signal_date", "code"])[stage.TARGET].sort_index()
    new = ref.set_index(["signal_date", "code"])[stage.TARGET].sort_index()
    np.testing.assert_allclose(old, new, equal_nan=True, atol=0, rtol=0)
    return part.merge(ref[["signal_date", "code", "model_score", "calibrated_score", "train_prior"]],
                      on=["signal_date", "code"], validate="one_to_one")


def cap_distribution(rows, out):
    # Descriptive bins are fixed before any outcome inspection; not new filter candidates.
    binned = rows.assign(float_cap_bin=pd.cut(rows.float_cap / 1e8, [0, 20, 30, 50, 100, 200, 500, np.inf],
                    labels=["<20", "20-30", "30-50", "50-100", "100-200", "200-500", ">=500"]))
    binned["float_cap_bin"] = binned.float_cap_bin.astype(object).fillna("UNKNOWN")
    binned["period"] = np.select([binned.signal_date.lt("2025-01-01"), binned.signal_date.lt("2025-07-01"), binned.signal_date.lt("2026-01-01")],
                                  ["development", "validation2025H1", "evaluation2025H2"], default="evaluation2026")
    result = binned.groupby(["period", "float_cap_bin"], sort=True).agg(rows=("code", "size"), known=(stage.TARGET, "count"), hits=(stage.TARGET, "sum"))
    result["hit_rate"] = result.hits / result.known
    result.reset_index().to_csv(out / "fixed_cap_bins_DESCRIPTIVE_NOT_THRESHOLD_SEARCH.csv", index=False)
    return result.reset_index().to_dict("records")


def run(output):
    out = Path(output)
    if out.exists():
        raise FileExistsError("Refuse rerun or overwrite; preserve failures and all prior fits")
    out.mkdir(parents=True)
    rows, upper_cap, scale = load_rows()
    protected = {str(p.resolve()): base.sha(p) for p in [Path(base.__file__), Path(stage.__file__), Path(symbolic.__file__),
                 REFERENCE / "report.json", REFERENCE / "S1_ALL_PREDICTIONS.csv", REFERENCE / "S2_ALL_PREDICTIONS.csv"]}
    inputs = {str(p.resolve()): base.sha(p) for p in [REFERENCE / "all_first_and_second_boards_WITH_SEPARATE_LABELS.csv",
              SYMBOLIC / "complete28503_first_board_dated_symbolic_features.csv", SYMBOLIC / "qualification.json",
              SOURCES / "source_adoption.json"]}
    protocol = dict(recorded_at=base.now(), budget_minutes=30, user_constraints=["First-board close only", "First-to-second Top1 target60%", "Review the same stock again at second-board close", "Avoid very large capital absent verified restructuring and suspected three-board-fund patterns"],
        hypothesis="A fixed capital feasibility screen plus conservative observable thin-turnover avoidance and dated competing-support proxies improve first-close next-board ranking.",
        baseline="Exact saved62-field STAGE_CYCLE_GENE scores; no baseline or stage2 refit",
        arms={"SCREEN_FROZEN": "The saved baseline with the new fixed screen; zero fits", "CONTEXT_SCREEN": "Same screen plus one fixed62+6+17=85-field model; no parameter/threshold search"},
        fields=FIELDS, risk_policy=RISK_POLICY, scale=scale,
        validation="2025H1, original complete117-date pool. Only an arm with strictly more hits than frozen baseline,>=95% decision-date coverage, no less non-one-price Top1 hits, proceeds. Baseline wins ties. Among passing arms choose most hits then SCREEN_FROZEN tie.",
        evaluation="Only selected arm across original2025Jul-2026Aug complete pools. If CONTEXT_SCREEN wins,14monthly windows of original252 official positions, mature labels strictly before each boundary; one prefix and one full model each, original parameters/equal-day weights/calibration. Save all predictions before metrics.",
        second_stage="Unmodified saved second-stage scores, exact same-code next-session rechecks",
        goal_60="Known selected Top1 rate>=60%,>=200known,>=95% outcome coverage, no abstention days and both2025H2/2026>=55%; full-date success, non-one-price, FP/FN, discarded positives, calibration reported",
        interpretation="Historical research already studied, not pristine blind or verified identity recognition; current-universe survivorship, source-specific float vs total capital, missing original effective-share raw data remain. No selected forum winner list is a label or blacklist.",
        end_condition="After this one fixed comparison finish, no test-driven new threshold/variant. Rejected pairwise and old relay-symbolic challenges never rerun.",
        inputs=inputs, protected=protected, original_raw_hashes={str(p.resolve()): base.sha(p) for p in base.INPUTS.values()},
        source_sha256=base.sha(__file__), actual_orders=0, production_changes=False)
    base.write_json(out / "protocol_BEFORE_ANY_FIT.json", protocol)
    shutil.copy2(__file__, out / "source_before_first_fit.py")
    rows = capital_context(rows, upper_cap)
    rows.to_csv(out / "COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv", index=False, encoding="utf-8-sig")
    base.write_json(out / "prefix_causality_BEFORE_FITS.json", prefix_checks(rows, upper_cap))
    stage.VARIANTS[VARIANT] = FIELDS
    known = rows[rows[stage.TARGET].notna()]
    train = known[known.signal_date.lt("2025-01-01") & known.label_end.lt("2025-01-01")]
    val = rows[rows.signal_date.between("2025-01-01", "2025-06-30")]
    reference = attach_frozen(rows, REFERENCE / "validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
    bundle = stage.fit_head(train, VARIANT, out, "validation_context")
    scored = stage.predict(bundle, val)
    scored.to_csv(out / "validation_CONTEXT_ALL_PREDICTIONS_BEFORE_METRICS.csv", index=False)
    baseline = stage.metric(reference)
    checks = {"SCREEN_FROZEN": screen_metric(reference), "CONTEXT_SCREEN": screen_metric(scored)}
    reference_non = stage.metric(reference[reference.one_price.eq(0)])
    sources = {"SCREEN_FROZEN": reference, "CONTEXT_SCREEN": scored}
    passing = [key for key, m in checks.items() if m["hits"] > baseline["hits"] and m["decision_date_coverage"] >= .95
               and screen_metric(sources[key][sources[key].one_price.eq(0)])["hits"] >= reference_non["hits"]]
    winner = max(passing, key=lambda key: checks[key]["hits"]) if passing else None
    decision = dict(formed_at=base.now(), winner=winner, validation=checks, frozen_baseline=baseline,
                    validation_non_one_price={key: screen_metric(value[value.one_price.eq(0)]) for key, value in sources.items()},
                    baseline_non_one_price=reference_non, continue_to_evaluation=winner is not None)
    base.write_json(out / "selection_BEFORE_EVALUATION.json", decision)
    print(json.dumps(base.clean(dict(validation=checks, baseline=baseline, winner=winner)), ensure_ascii=False), flush=True)
    report = dict(completed_at=None, protocol=str((out / "protocol_BEFORE_ANY_FIT.json").resolve()), decision=decision,
                  scale=scale, goal_60_passed=False, identity_recognition_validated=False,
                  full_historical_restructuring_coverage=False, real_forward_results=0, actual_trades=0,
                  source_sha256=base.sha(__file__), protected_unchanged=True)
    if winner:
        parts = []
        sessions = json.loads(base.INPUTS["calendar"].read_text("utf-8"))["sessions"]
        frozen = attach_frozen(rows, REFERENCE / "S1_ALL_PREDICTIONS.csv")
        for month, part in frozen.groupby(frozen.signal_date.str[:7], sort=True):
            if winner == "CONTEXT_SCREEN":
                boundary = month + "-01"
                position = int(np.searchsorted(sessions, boundary))
                tr = known[known.p.ge(position-252) & known.label_end.lt(boundary)]
                head = stage.fit_head(tr, VARIANT, out, month + "_context")
                assert head["train_max_label_end"] < part.signal_date.min()
                part = stage.predict(head, part)
            part["model_month"] = month
            part.to_csv(out / (month + "_FULL_PREDICTIONS_BEFORE_METRICS.csv"), index=False)
            parts.append(part)
            print(month + " saved", flush=True)
        pred = pd.concat(parts, ignore_index=True)
        pred.to_csv(out / "EVALUATION_FULL_PREDICTIONS.csv", index=False, encoding="utf-8-sig")
        report["evaluation"] = {}
        for period, mask in [("all", pd.Series(True, index=pred.index)), ("2025H2", pred.signal_date.lt("2026-01-01")),
                             ("2026", pred.signal_date.ge("2026-01-01")), ("non_one_price", pred.one_price.eq(0))]:
            report["evaluation"][period] = dict(screened=screen_metric(pred[mask]),
                original_frozen=stage.metric(frozen[frozen.signal_date.isin(pred[mask].signal_date.unique()) & (frozen.one_price.eq(0) if period=="non_one_price" else True)]))
        allowed = pred[pred.research_selection_allowed]
        second = pd.read_csv(REFERENCE / "S2_ALL_PREDICTIONS.csv", dtype={"code": str}, float_precision="round_trip")
        chain, report["same_stock_second_recheck"] = stage.chain_recheck(allowed, second)
        chain.to_csv(out / "FIRST_SELECTION_TO_SAME_STOCK_SECOND_RECHECK.csv", index=False, encoding="utf-8-sig")
        report["calibration_on_allowed_pool"] = stage.calibration_metrics(allowed)
        m = report["evaluation"]["all"]["screened"]
        report["goal_60_passed"] = bool(m["hit_rate"] >= .60 and m["known"] >= 200 and m["outcome_coverage"] >= .95 and m["abstention_dates"] == 0
              and all(report["evaluation"][p]["screened"]["hit_rate"] >= .55 for p in ("2025H2", "2026")))
        picks = base.selected(allowed, "model_score", 1)
        picks.to_csv(out / "EVALUATION_SELECTED_TOP1.csv", index=False, encoding="utf-8-sig")
        pred[~pred.research_selection_allowed].to_csv(out / "ALL_EXCLUSIONS_INCLUDING_PROMOTERS.csv", index=False, encoding="utf-8-sig")
    report["fixed_cap_bins"] = cap_distribution(rows, out)
    report["core_fit_count"] = len(list((out / "models").glob("*_COMPLETED.json")))
    report["completed_at"] = base.now()
    report["status"] = "FIXED_COMPARISON_FINISHED" if winner else "BOTH_ARMS_REJECTED_VALIDATION_NO_TEST_REFIT"
    for path, digest in protected.items():
        assert base.sha(path) == digest
    base.write_json(out / "report.json", report)
    print(json.dumps(base.clean({key: report.get(key) for key in ("status", "core_fit_count", "evaluation", "same_stock_second_recheck", "goal_60_passed")}), ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
