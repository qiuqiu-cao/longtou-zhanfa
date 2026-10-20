"""Stage-specific stock memory built only from previously matured episodes.

Historical first-board outcomes can describe past stock behaviour. They do not
identify the funds behind a current first board or supply its future outcome.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import joblib
import numpy as np
import pandas as pd

REF = risk.REFERENCE
SCREEN = base.ROOT / "research/first_board_capital_risk_20261020_run1"
LONG = base.ROOT / "research/first_board_leader_20261020T114600Z/all_first_boards_features_and_LABEL_ONLY.csv"
QUALIFICATION = base.ROOT / "research/first_board_matured_history_qualification_20261020"
FIRST = ["event_first_known252", "event_first_smoothed_rate252", "event_first_rate_vs_market252",
         "event_previous_first_outcome", "event_previous_first_age", "event_first_known20"]
PATH = ["event_complete_paths252", "event_second_to3_smoothed252", "event_third_to4_smoothed252",
        "event_first_to4_smoothed252", "event_stop_at2_count252", "event_stop_at3_count252",
        "event_max_path_height252", "event_last_complete_height"]
VARIANTS = {"FIRST_EVENT_MEMORY": stage.VARIANTS["STAGE_CYCLE_GENE"] + FIRST,
            "FULL_PATH_MEMORY": stage.VARIANTS["STAGE_CYCLE_GENE"] + FIRST + PATH}
POLICY = dict(window_official_sessions=252, short_window=20, prior_strength=5.,
    first_maturity="Original separate next-official-close label_end<=signal close; event itself strictly earlier",
    path_maturity="Original uniform5-session path label_end<=signal close; incomplete paths remain unknown",
    historical_rate_prior="Pooled fully matured events in the same preceding252 official signal positions; no later outcome",
    current_fund_identity="UNKNOWN; past stops at2/3 are observed past paths, not fund labels",
    risk_screen="Unchanged exact yuan threshold and proxy definition from the previous protocol; no new cap or turnover threshold selected.")


def smooth(wins, known, prior):
    return (wins + POLICY["prior_strength"] * prior) / (known + POLICY["prior_strength"]) if np.isfinite(prior) else np.nan


def build_features(first, long, sessions):
    rows = first.copy()
    if rows.duplicated(["signal_date", "code"]).any() or long.duplicated(["signal_date", "code"]).any():
        raise ValueError("duplicate episode keys")
    positions = {d:i for i,d in enumerate(sessions)}
    events = first[["signal_date", "code", "p", "label_end", stage.TARGET]].copy()
    events["mature_p"] = events.label_end.map(positions)
    paths = long[["signal_date", "code", "p", "label_end", "reach2_LABEL_ONLY", "reach3_LABEL_ONLY", "reach4_LABEL_ONLY", "max_height_5_LABEL_ONLY"]].copy()
    paths["mature_p"] = paths.label_end.map(positions)
    known = events[events[stage.TARGET].notna() & events.mature_p.notna()].copy()
    complete = paths[paths.max_height_5_LABEL_ONLY.notna() & paths.mature_p.notna()].copy()
    if not set(zip(first.signal_date, first.code)) == set(zip(paths.signal_date, paths.code)):
        raise ValueError("path and first-stage episode cohorts differ")
    by_code = {c:g.sort_values("p") for c,g in known.groupby("code", sort=False)}
    by_path = {c:g.sort_values("p") for c,g in complete.groupby("code", sort=False)}
    produced = []
    for p, part in rows.groupby("p", sort=True):
        start = p - POLICY["window_official_sessions"]
        pool = known[known.p.ge(start) & known.p.lt(p) & known.mature_p.le(p)]
        prior_first = pool[stage.TARGET].mean()
        long_pool = complete[complete.p.ge(start) & complete.p.lt(p) & complete.mature_p.le(p)]
        seconds = long_pool[long_pool.reach2_LABEL_ONLY.eq(1)]
        thirds = long_pool[long_pool.reach3_LABEL_ONLY.eq(1)]
        prior_second = seconds.reach3_LABEL_ONLY.mean()
        prior_third = thirds.reach4_LABEL_ONLY.mean()
        prior_four = long_pool.reach4_LABEL_ONLY.mean()
        for row in part.itertuples():
            own = by_code.get(row.code, known.iloc[:0])
            own = own[own.p.ge(start) & own.p.lt(p) & own.mature_p.le(p)]
            past = by_path.get(row.code, complete.iloc[:0])
            past = past[past.p.ge(start) & past.p.lt(p) & past.mature_p.le(p)]
            own2 = past[past.reach2_LABEL_ONLY.eq(1)]
            own3 = past[past.reach3_LABEL_ONLY.eq(1)]
            rate = smooth(own[stage.TARGET].sum(), len(own), prior_first)
            values = [len(own), rate, rate-prior_first,
                own[stage.TARGET].iloc[-1] if len(own) else np.nan,
                p-own.p.iloc[-1] if len(own) else np.nan,
                int(own.p.ge(p-POLICY["short_window"]).sum()),
                len(past), smooth(own2.reach3_LABEL_ONLY.sum(), len(own2), prior_second),
                smooth(own3.reach4_LABEL_ONLY.sum(), len(own3), prior_third),
                smooth(past.reach4_LABEL_ONLY.sum(), len(past), prior_four),
                int(past.max_height_5_LABEL_ONLY.eq(2).sum()), int(past.max_height_5_LABEL_ONLY.eq(3).sum()),
                past.max_height_5_LABEL_ONLY.max() if len(past) else np.nan,
                past.max_height_5_LABEL_ONLY.iloc[-1] if len(past) else np.nan]
            produced.append(dict(signal_date=row.signal_date, code=row.code, **dict(zip(FIRST+PATH, values))))
    extra = pd.DataFrame(produced)
    result = rows.merge(extra, on=["signal_date", "code"], validate="one_to_one")
    return result


def prefix_check(first, paths, sessions, features):
    result = []
    for cutoff in ("2025-06-30", "2025-12-31"):
        # Outcomes formed after this cutoff must be removed even for older signals.
        old = first[first.signal_date.le(cutoff)].copy()
        old.loc[old.label_end.gt(cutoff), stage.TARGET] = np.nan
        long = paths[paths.signal_date.le(cutoff)].copy()
        long.loc[long.label_end.gt(cutoff), ["reach2_LABEL_ONLY", "reach3_LABEL_ONLY", "reach4_LABEL_ONLY", "max_height_5_LABEL_ONLY"]] = np.nan
        current = build_features(old, long, sessions)
        expected = features[features.signal_date.le(cutoff)]
        np.testing.assert_allclose(current[FIRST+PATH], expected[FIRST+PATH], atol=1e-12, rtol=1e-12, equal_nan=True)
        result.append(dict(cutoff=cutoff, rows=len(current), all14_equal=True, future_formed_outcomes_removed=True))
    return result


def period_metrics(pred, frozen):
    metrics = {}
    for period, mask in [("all", pd.Series(True,index=pred.index)), ("2025H2", pred.signal_date.lt("2026-01-01")),
                         ("2026", pred.signal_date.ge("2026-01-01")), ("non_one_price", pred.one_price.eq(0))]:
        sub = pred[mask]
        ref = frozen[frozen.signal_date.isin(sub.signal_date.unique())]
        if period == "non_one_price":
            ref = ref[ref.one_price.eq(0)]
        metrics[period] = dict(new_screened=risk.screen_metric(sub), frozen_screened=risk.screen_metric(ref),
                               original_full=stage.metric(ref), new_unfiltered_diagnostic=stage.metric(sub))
    return metrics


def run(output):
    out = Path(output)
    if out.exists():
        raise FileExistsError("Refuse to rerun this fixed comparison")
    out.mkdir(parents=True)
    data_path = REF / "all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
    original = pd.read_csv(data_path,dtype={"code":str},float_precision="round_trip")
    first = original[original.stage.eq(1)].copy()
    paths = pd.read_csv(LONG,dtype={"code":str},float_precision="round_trip")
    sessions = json.loads(base.INPUTS["calendar"].read_text("utf-8"))["sessions"]
    protected_paths = [Path(stage.__file__),Path(base.__file__),Path(risk.__file__), REF/"S1_ALL_PREDICTIONS.csv", REF/"S2_ALL_PREDICTIONS.csv",
                       REF/"report.json", SCREEN/"report.json", SCREEN/"protocol_BEFORE_ANY_FIT.json"]
    inputs = [data_path,LONG,SCREEN/"COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv",QUALIFICATION/"minute_qualification.json",QUALIFICATION/"extended_minute_qualification.json"]
    protocol = dict(formed_at=base.now(), budget_minutes=30, previous_goal_turn="PROGRESS: new screened experiment ruled out; no same-threshold retry. Existing minute intersection now audited.",
        primary="First-close dailyTop1 next-official-close1to2 target60%; same stock second-close re-evaluation remains separate",
        hypothesis="Previously matured stage-specific stock first-board and low-height path histories add information beyond aggregate60-session limit-up genes.",
        expected_gain="At least5pp is the research expectation, not a guarantee; target60 gate unchanged",
        policy=POLICY, variants=VARIANTS, baseline="Exact saved62-field score plus previous unchanged risk screen. Also retain original unscreened results.",
        validation="Original2025H1 complete117days. Challenger needs strictly more Top1 hits than frozen screened baseline,>=95% decision-date coverage, no fewer non-one-price Top1 hits. If both pass choose most hits; FIRST_EVENT_MEMORY wins ties.",
        test="Only chosen variant, original14monthly previous252 official-position windows and strictly earlier mature labels. Same model/calibration settings and equal-day weights, no baseline or stage2 refits.",
        no_current_target_leakage="Historical outcome may enter stock memory only after its explicit label_end; unknowns never negative. Current first-board outcome is never a feature.",
        fund_identity_unknown=True, source_retrieved_later_not_backdated=True, historical_research_previously_seen=True,
        target60="Known Top1>=.60,>=200known,>=95% outcome coverage, no abstention days, both2025H2/2026>=.55. Report all candidates/excluded positives/unknowns/calibration/same-stock stage2.",
        stop="One fixed2-arm comparison; preserve failures and no retune. No retry of pairwise, capital/name-context or previous relay experiments.",
        protected={str(p.resolve()):base.sha(p) for p in protected_paths}, inputs={str(p.resolve()):base.sha(p) for p in inputs},
        source_sha256=base.sha(__file__), original_raw_hashes={str(p.resolve()):base.sha(p) for p in base.INPUTS.values()},
        production_changes=False, actual_orders=0)
    base.write_json(out/"protocol_BEFORE_ANY_FIT.json",protocol)
    shutil.copy2(__file__,out/"source_before_first_fit.py")
    rows = build_features(first,paths,sessions)
    screen = pd.read_csv(SCREEN/"COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv",dtype={"code":str},float_precision="round_trip")
    extra = ["research_selection_allowed","exclusion_reason","threeboard_evidence_status"]
    rows = rows.merge(screen[["signal_date","code"]+extra],on=["signal_date","code"],validate="one_to_one")
    rows.to_csv(out/"ALL_FIRST_BOARDS_WITH_MATURED_MEMORY_AND_UNCHANGED_SCREEN.csv",index=False,encoding="utf-8-sig")
    causality = prefix_check(first,paths,sessions,rows)
    coverage = []
    for period,mask in [("development",rows.signal_date.lt("2025-01-01")),("validation",rows.signal_date.between("2025-01-01","2025-06-30")),("evaluation",rows.signal_date.ge("2025-07-01"))]:
        sub=rows[mask]
        coverage.append(dict(period=period,rows=len(sub),previous_first_known=int(sub.event_first_known252.gt(0).sum()),
                             previous_complete_path_known=int(sub.event_complete_paths252.gt(0).sum()),all_field_coverage=sub[FIRST+PATH].notna().mean().to_dict()))
    base.write_json(out/"causality_and_new_event_coverage_BEFORE_FITS.json",dict(prefix=causality,coverage=coverage))
    known=rows[rows[stage.TARGET].notna()]
    train=known[known.signal_date.lt("2025-01-01")&known.label_end.lt("2025-01-01")]
    val=rows[rows.signal_date.between("2025-01-01","2025-06-30")]
    frozen=risk.attach_frozen(rows, REF/"validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
    baseline=risk.screen_metric(frozen)
    baseline_non=risk.screen_metric(frozen[frozen.one_price.eq(0)])
    decisions={}
    for variant,fields in VARIANTS.items():
        stage.VARIANTS[variant]=fields
        bundle=stage.fit_head(train,variant,out,"validation_"+variant)
        pred=stage.predict(bundle,val)
        pred.to_csv(out/("validation_"+variant+"_ALL_BEFORE_METRICS.csv"),index=False)
        m=risk.screen_metric(pred);n=risk.screen_metric(pred[pred.one_price.eq(0)])
        decisions[variant]=dict(primary=m,non_one_price=n,unfiltered_diagnostic=stage.metric(pred),
            pass_validation=bool(m["hits"]>baseline["hits"] and m["decision_date_coverage"]>=.95 and n["hits"]>=baseline_non["hits"]))
        print(json.dumps(base.clean(dict(variant=variant,metrics=decisions[variant])),ensure_ascii=True),flush=True)
    passing=[v for v in VARIANTS if decisions[v]["pass_validation"]]
    selected=max(passing,key=lambda v:decisions[v]["primary"]["hits"]) if passing else None
    decision=dict(formed_at=base.now(),selected=selected,baseline=baseline,baseline_non_one_price=baseline_non,variants=decisions)
    base.write_json(out/"selection_BEFORE_EVALUATION.json",decision)
    report=dict(formed_at=None,decision=decision,goal_60_passed=False,full_fund_identity_validated=False,new_forward_results=0,actual_trades=0,
                protected_unchanged=True,source_sha256=base.sha(__file__))
    if selected:
        test=rows[rows.signal_date.ge("2025-07-01")];parts=[]
        for month,part in test.groupby(test.signal_date.str[:7],sort=True):
            boundary=month+"-01";p=int(np.searchsorted(sessions,boundary))
            tr=known[known.p.ge(p-252)&known.label_end.lt(boundary)]
            bundle=stage.fit_head(tr,selected,out,month+"_"+selected)
            pred=stage.predict(bundle,part);pred["model_month"]=month
            pred.to_csv(out/(month+"_ALL_BEFORE_METRICS.csv"),index=False);parts.append(pred)
            print(month+" saved",flush=True)
        predictions=pd.concat(parts,ignore_index=True)
        predictions.to_csv(out/"EVALUATION_ALL_PREDICTIONS.csv",index=False,encoding="utf-8-sig")
        frozen=risk.attach_frozen(rows,REF/"S1_ALL_PREDICTIONS.csv")
        report["evaluation"]=period_metrics(predictions,frozen)
        allowed=predictions[predictions.research_selection_allowed]
        second=pd.read_csv(REF/"S2_ALL_PREDICTIONS.csv",dtype={"code":str},float_precision="round_trip")
        chain,report["same_stock_second_recheck"]=stage.chain_recheck(allowed,second)
        chain.to_csv(out/"SAME_STOCK_SECOND_RECHECK.csv",index=False,encoding="utf-8-sig")
        report["historical_calibration_allowed_pool"]=stage.calibration_metrics(allowed)
        m=report["evaluation"]["all"]["new_screened"]
        report["goal_60_passed"]=bool(m["hit_rate"]>=.60 and m["known"]>=200 and m["outcome_coverage"]>=.95 and m["abstention_dates"]==0
            and all(report["evaluation"][x]["new_screened"]["hit_rate"]>=.55 for x in ("2025H2","2026")))
        base.selected(allowed,"model_score",1).to_csv(out/"ALL_SELECTED_TOP1.csv",index=False,encoding="utf-8-sig")
        predictions[~predictions.research_selection_allowed].to_csv(out/"ALL_EXCLUDED_INCLUDING_PROMOTERS.csv",index=False,encoding="utf-8-sig")
    report.update(formed_at=base.now(),core_fit_count=len(list((out/"models").glob("*_COMPLETED.json"))),
                  status="FIXED_EVALUATION_FINISHED" if selected else "BOTH_ARMS_REJECTED_VALIDATION_NO_TEST_FITS")
    for path,digest in protocol["protected"].items():
        assert base.sha(path)==digest
    base.write_json(out/"report.json",report)
    print(json.dumps(base.clean(report),ensure_ascii=True,indent=2))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    run(parser.parse_args().output)
