"""First-close dependence on the archived market, without inferred fund labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import first_board_matured_event_model as prior
import numpy as np
import pandas as pd

REF = risk.REFERENCE
SCREEN = prior.SCREEN
QUAL = base.ROOT / "research/first_board_industry_qualification_20261020"
VARIANT = "FIRST_CLOSE_MARKET_RESIDUAL"
FIELDS = [f"market_{name}{w}" for w in (20, 60) for name in ("beta", "corr", "residual_vol", "signal_residual_z")]
MODEL_FIELDS = stage.VARIANTS["STAGE_CYCLE_GENE"] + FIELDS
POLICY = dict(windows=[20, 60], estimation_end="Previous official session, excluding the first-board signal day",
    return_definition="Unadjusted close/exact adjacent official previous close-1; require eligible valid regular bars, positive amount/volume and provider previous-close agreement within0.011yuan",
    market="Same-day equal-weight mean of eligible archived observed returns, excluding this stock itself; not an exchange-complete index or verified industry",
    missing="All window positions must contain valid stock/market pairs; no forward/back fill or skipping missing official dates",
    regression="Fixed rolling OLS intercept and beta; Pearson correlation and residual population volatility; signal-day residual divided by prior residual volatility",
    identity="Statistical market dependence, not absorption/control/program-fund or industry/theme identity",
    source_fallback="Bounded dated industry API/PDF attempts failed. No current industry classification used. Existing full daily archive supports a different, explicit market-relative test before any fitting.")


def return_grid(bars, sessions):
    cols = ["date", "code", "p", "close", "pre_close", "eligible", "valid_price", "regular_price", "amount", "volume"]
    b = bars[cols].copy()
    if b.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate daily prices")
    if list(sessions) != sorted(set(sessions)):
        raise ValueError("Official calendar is not sorted and unique")
    close = b.pivot(index="date", columns="code", values="close").reindex(sessions)
    reference = b.pivot(index="date", columns="code", values="pre_close").reindex_like(close)
    valid = b.eligible & b.valid_price & b.regular_price & b.amount.gt(0) & b.volume.gt(0)
    allowed = b.assign(allowed=valid).pivot(index="date", columns="code", values="allowed").reindex_like(close).fillna(False).astype(bool)
    previous = close.shift(1)
    usable = allowed & allowed.shift(1, fill_value=False) & previous.gt(0) & reference.sub(previous).abs().le(.011)
    returns = (close / previous - 1).where(usable)
    count = returns.count(axis=1)
    mean_without_self = returns.rsub(returns.sum(axis=1), axis=0).div(count.sub(1).where(count.gt(1)), axis=0)
    return returns, mean_without_self.where(returns.notna())


def build_features(bars, sessions, keys):
    y, x = return_grid(bars, sessions)
    rows = keys[["signal_date", "code"]].copy()
    if rows.duplicated(["signal_date", "code"]).any():
        raise ValueError("Duplicate signal keys")
    row_positions = y.index.get_indexer(rows.signal_date)
    col_positions = y.columns.get_indexer(rows.code)
    if (row_positions < 0).any() or (col_positions < 0).any():
        raise ValueError("Signal outside source grid")
    for w in POLICY["windows"]:
        # Grid rows are official dates. Every rolling estimate ends at S-1.
        def mean(z):
            return z.rolling(w, min_periods=w).mean().shift(1)
        mx, my = mean(x), mean(y)
        vx = (mean(x * x) - mx * mx).clip(lower=0)
        vy = (mean(y * y) - my * my).clip(lower=0)
        covariance = mean(x * y) - mx * my
        beta = covariance / vx.where(vx.gt(1e-18))
        corr = (covariance / np.sqrt(vx * vy).where((vx * vy).gt(1e-18))).clip(-1, 1)
        residual_vol = np.sqrt((vy - covariance * beta).clip(lower=0))
        residual_z = (y - (my - beta * mx) - beta * x) / residual_vol.where(residual_vol.gt(1e-12))
        for name, value in [("beta", beta), ("corr", corr), ("residual_vol", residual_vol), ("signal_residual_z", residual_z)]:
            rows[f"market_{name}{w}"] = value.to_numpy()[row_positions, col_positions]
    rows[FIELDS] = rows[FIELDS].replace([np.inf, -np.inf], np.nan)
    return rows


def run(output):
    out = Path(output)
    if out.exists():
        raise FileExistsError("One fixed comparison; refuse overwrite/repeat")
    out.mkdir(parents=True)
    source = REF / "all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
    screen_source = SCREEN / "COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv"
    protected = [Path(base.__file__), Path(stage.__file__), Path(risk.__file__), Path(prior.__file__),
                 REF / "report.json", REF / "S1_ALL_PREDICTIONS.csv", REF / "S2_ALL_PREDICTIONS.csv", SCREEN / "report.json"]
    protocol = dict(formed_at=base.now(), batch_started_at="2026-10-20T13:33:19+00:00", batch_deadline="2026-10-20T14:03:19+00:00",
        hypothesis="Prior stock-specific market dependence plus its first-close residual adds ranking information beyond the original62price/cycle/gene fields.",
        previous_work="Own-price volume structure failed. Industry source qualification attempted once without usable full mapping; no industry fits, current memberships or mechanism labels.",
        policy=POLICY, variants={VARIANT: MODEL_FIELDS}, baseline="Saved62-field baseline with unchanged prior capital/thin-turnover screen; unscreened diagnostic retained",
        validation="2025H1 full original4724rows/117days; strictly moreTop1hits than frozen screened baseline,>=95%datecoverage and at least baseline non-one-price Top1hits; baseline wins ties",
        evaluation="Only if validation passes: selected arm once in14original monthly252official-position windows, original parameters/weights/temporal42datecalibration; original second-head reused for same-code recheck",
        target60="Top1>=60%,>=200known,>=95%outcomecoverage,no abstention dates,2025H2and2026each>=55%; report uncertainty",
        stop="One arm, once. No adjusted windows/thresholds/weights/parameters after seeing results. Stop fixed comparison or budget; checkpoint before optional evaluation if budget reached.",
        historical_replay_not_pristine_blind=True, actual_trades=0, new_forward_results=0, production_changes=False,
        inputs={str(p.resolve()): base.sha(p) for p in [source, screen_source] + list(base.INPUTS.values())},
        protected={str(p.resolve()): base.sha(p) for p in protected}, source_sha256=base.sha(__file__))
    base.write_json(out / "protocol_BEFORE_ANY_FIT.json", protocol)
    shutil.copy2(__file__, out / "source_before_first_fit.py")
    original = pd.read_csv(source, dtype={"code": str}, float_precision="round_trip")
    first = original[original.stage.eq(1)].copy()
    _, bars, sessions, audit = base.build_dataset()
    extra = build_features(bars, sessions, first)
    rows = first.merge(extra, on=["signal_date", "code"], validate="one_to_one")
    screen = pd.read_csv(screen_source, dtype={"code": str}, float_precision="round_trip")
    rows = rows.merge(screen[["signal_date", "code", "research_selection_allowed", "exclusion_reason", "threeboard_evidence_status"]], on=["signal_date", "code"], validate="one_to_one")
    rows.to_csv(out / "ALL_FIRST_BOARD_MARKET_CONTEXT_UNCHANGED_LABELS.csv", index=False, encoding="utf-8-sig")
    checks = []
    for cutoff in ("2025-06-30", "2025-12-31"):
        selected = first[first.signal_date.le(cutoff)]
        prefix = build_features(bars[bars.date.le(cutoff)], [d for d in sessions if d <= cutoff], selected)
        old = rows[rows.signal_date.le(cutoff)]
        np.testing.assert_allclose(prefix[FIELDS], old[FIELDS], atol=1e-11, rtol=1e-11, equal_nan=True)
        checks.append(dict(cutoff=cutoff, rows=len(prefix), all8fields_equal=True))
    base.write_json(out / "causality_and_coverage_BEFORE_FIT.json", dict(prefix_checks=checks,
        coverage=rows[FIELDS].notna().mean().to_dict(), audit=audit))
    del bars, extra
    known = rows[rows[stage.TARGET].notna()]
    train = known[known.signal_date.lt("2025-01-01") & known.label_end.lt("2025-01-01")]
    val = rows[rows.signal_date.between("2025-01-01", "2025-06-30")]
    frozen = risk.attach_frozen(rows, REF / "validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
    baseline = risk.screen_metric(frozen)
    baseline_non = risk.screen_metric(frozen[frozen.one_price.eq(0)])
    stage.VARIANTS[VARIANT] = MODEL_FIELDS
    bundle = stage.fit_head(train, VARIANT, out, "validation_" + VARIANT)
    scored = stage.predict(bundle, val)
    scored.to_csv(out / "validation_ALL_BEFORE_METRICS.csv", index=False)
    metric = risk.screen_metric(scored)
    non = risk.screen_metric(scored[scored.one_price.eq(0)])
    passed = bool(metric["hits"] > baseline["hits"] and metric["decision_date_coverage"] >= .95 and non["hits"] >= baseline_non["hits"])
    decision = dict(formed_at=base.now(), selected=VARIANT if passed else None, baseline=baseline,
                    baseline_non_one_price=baseline_non, primary=metric, non_one_price=non,
                    unfiltered_diagnostic=stage.metric(scored), pass_validation=passed)
    base.write_json(out / "selection_BEFORE_EVALUATION.json", decision)
    report = dict(formed_at=base.now(), decision=decision, goal_60_passed=False, actor_identity_validated=False,
                  new_forward_results=0, actual_trades=0, core_fit_count=len(list((out / "models").glob("*_COMPLETED.json"))),
                  status="VALIDATION_PASSED_EVALUATION_PENDING_BUDGET_CHECK" if passed else "REJECTED_VALIDATION_NO_MONTHLY_FITS")
    # Optional monthly fitting is deliberately a separate resumable checkpoint.
    # The already-fixed protocol, variant and scores must be reused if it passed.
    for path, digest in protocol["protected"].items():
        assert base.sha(path) == digest
    base.write_json(out / "report.json", report)
    print(json.dumps(base.clean(report), ensure_ascii=True, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
