"""Compare fixed existing heads and the submitted overlay before/after a semantic bug fix.

No fitting, threshold search, unseen-test claims, auction data, or production promotion.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from first_board_comprehensive_engine import EngineConfig, ComprehensiveFirstBoardEngine
from legacy_industry_snapshot import attach_legacy_groups

HERE = Path(__file__).resolve().parent
KEYS = ["signal_date", "code"]
LABELS = ["promotion_next_LABEL_ONLY", "reach4_LABEL_ONLY"]
SCREEN = ROOT / "research/first_board_leader_emotion_20261021T1706Z/ALL28503_PATHS_AND_ORIGINAL_SCREEN.csv"
HEIGHT = ROOT / "research/first_board_leader_20261020T114600Z/evaluation_ALL_PREDICTIONS.csv"
S1 = ROOT / "research/first_second_stage_20261020_run1/S1_ALL_PREDICTIONS.csv"
SNAPSHOT = HERE / "source_snapshot/first_board_comprehensive_engine.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(path, dtype={"code": str}, float_precision="round_trip", low_memory=False)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def metrics(picks, dates, k, label):
    known = int(picks[label].notna().sum())
    hits = int(picks[label].eq(1).sum())
    return {"dates_in_scope": len(dates), "possible_slots": len(dates)*k,
            "selected": len(picks), "selected_dates": int(picks.signal_date.nunique()),
            "abstained_dates": len(dates)-int(picks.signal_date.nunique()),
            "unfilled_slots": len(dates)*k-len(picks), "known": known,
            "unknown_selected": len(picks)-known, "hits": hits,
            "precision_known": hits/known if known else None,
            "hits_per_full_date_slots": hits/(len(dates)*k)}


def select(scored, k, apply_defense=False):
    eligible = ~scored.sneak_veto & ~scored.pattern_veto
    if apply_defense:
        eligible &= ~scored.is_defense_day
    return scored[eligible].sort_values(["signal_date", "final_rank", "code"]).groupby("signal_date").head(k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--simulate", action="store_true", help="Replay the two submitted/fixed candidates under the existing daily-open scenario")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    old = load_module(SNAPSHOT, "submitted_overnight_engine_snapshot")
    pair = load_module(ROOT / "research/overnight_score_review_20261029/review_saved_scores.py", "existing_paired_interval")
    methods = {}
    for name, w in [("HEIGHT",0.),("S1",1.),("BLEND_50_50",.5),("ACTUAL_60_40_REFERENCE",.6),("BLEND_70_30",.7)]:
        methods[name] = ComprehensiveFirstBoardEngine(EngineConfig(fusion_weight_s1=w,
            enable_sector_resonance=False, enable_seal_divergence=False, enable_emotion_defense=False))
    methods["SUBMITTED_ENGINE_70_30"] = old.ComprehensiveFirstBoardEngine(old.EngineConfig())
    methods["FIXED_ENGINE_70_30"] = ComprehensiveFirstBoardEngine(EngineConfig.overnight_research_candidate())
    inputs = [SCREEN, HEIGHT, S1, SNAPSHOT, ROOT / "first_board_comprehensive_engine.py", Path(__file__)]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    protocol = {"formed_at": datetime.now(timezone.utc).isoformat(), "source_hashes": hashes,
        "scope": "Previously inspected 286 days; descriptive engineering correction, not independent validation",
        "methods": {name: asdict(e.config) for name,e in methods.items()},
        "fixed_changes": ["Correct close/MA-1 inequality", "Restore independently switchable chart/small-cap overlays", "Apply pattern veto consistently in selection and simulation"],
        "forbidden": ["No new fit", "No parameter search", "No auction in overnight ranking", "No automatic production replacement"],
        "candidate_count": 2, "candidate_semantics": "Submitted implementation and corrected implementation only; other methods are saved-score controls"}
    (out / "PROTOCOL_BEFORE_COMPARISON.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")
    screen, hp, sp = map(read, [SCREEN, HEIGHT, S1])
    qualified = screen[screen.signal_date.between("2025-07-01","2026-08-31") & screen.allowed.eq(True)].copy()
    merged = qualified.merge(hp[KEYS+["model_score"]].rename(columns={"model_score":"height_score"}),on=KEYS,validate="one_to_one")
    merged = merged.merge(sp[KEYS+["model_score",LABELS[0]]].rename(columns={"model_score":"s1_score",LABELS[0]:"s1_label_check"}),on=KEYS,validate="one_to_one")
    assert len(merged)==7254 and merged.signal_date.nunique()==286
    assert merged[LABELS[0]].fillna(-1).equals(merged.s1_label_check.fillna(-1))
    merged = attach_legacy_groups(merged)
    # Explicitly remove all outcomes before calling either engine.
    fields = KEYS + ["name","height_score","s1_score","amount","turnover","float_cap","one_price",
        "position_60","close_ma5","close_ma20","close_ma60","breakout_20","market_down_limits",
        "market_failed_rate","previous_first_promotion","market_positive_fraction",
        "legacy_industry_group_PROXY","legacy_industry_source_status"]
    truth = merged[KEYS+LABELS]
    features = merged[fields].copy()
    output_metrics, selections, scored_all, intervals, picks_index = [], [], {}, [], {}
    scopes = {"all":features,"2025H2":features[features.signal_date.lt("2026-01-01")],
              "2026":features[features.signal_date.ge("2026-01-01")],"non_one_price":features[features.one_price.eq(0)]}
    for scope, rows in scopes.items():
        dates = sorted(rows.signal_date.unique())
        for name, engine in methods.items():
            groups = [engine.process_daily_candidates(date,g,g.iloc[0]) for date,g in rows.groupby("signal_date",sort=True)]
            scored = pd.concat(groups,ignore_index=True).merge(truth,on=KEYS,validate="one_to_one")
            if scope=="all":
                scored_all[name] = scored
                scored.to_csv(out / f"{name}_FULL_SCORED.csv",index=False,encoding="utf-8-sig")
            for defense in (False,True):
                for k in (1,3):
                    take=select(scored,k,defense)
                    picks_index[(scope,name,defense,k)]=take
                    for label in LABELS:
                        output_metrics.append(dict(scope=scope,method=name,apply_defense=defense,k=k,label=label,**metrics(take,dates,k,label)))
                    if scope=="all":
                        selections.append(take[KEYS+LABELS+["final_score","final_rank"]].assign(method=name,apply_defense=defense,k=k))
        for control in ("HEIGHT","S1","BLEND_70_30","SUBMITTED_ENGINE_70_30"):
            for k in (1,3):
                for label in LABELS:
                    a=picks_index[(scope,"FIXED_ENGINE_70_30",False,k)]
                    b=picks_index[(scope,control,False,k)]
                    intervals.append(dict(scope=scope,control=control,k=k,label=label,**pair.paired(a,b,dates,label)))
    pd.DataFrame(output_metrics).to_csv(out/"METRICS_ALL_DATES_AND_ABSTENTIONS.csv",index=False,encoding="utf-8-sig")
    pd.concat(selections,ignore_index=True).to_csv(out/"ALL_SELECTIONS.csv",index=False,encoding="utf-8-sig")
    old_bonus=scored_all["SUBMITTED_ENGINE_70_30"].pattern_adjustment.gt(0)
    new_bonus=scored_all["FIXED_ENGINE_70_30"].pattern_adjustment.gt(0)
    report={"formed_at":datetime.now(timezone.utc).isoformat(),"source_hashes":hashes,
        "rows":7254,"dates":286,"new_fits":0,"new_parameter_searches":0,"independent_forward_validation":False,
        "metrics":output_metrics,"descriptive_paired_intervals":intervals,
        "submitted_wrong_ma_bonus_rows":int(old_bonus.sum()),"fixed_ma_bonus_rows":int(new_bonus.sum()),
        "lhb_implemented":False,"production_replaced":False,"scenario_reports":{}}
    if args.simulate:
        from run_comprehensive_benchmark import load_all_bars, CALENDAR_PATH
        sessions=json.loads(CALENDAR_PATH.read_text("utf-8"))["sessions"]
        print("Loading daily bars for the two fixed execution comparisons...",flush=True)
        bars,lookup=load_all_bars(sessions)
        for name in ("SUBMITTED_ENGINE_70_30","FIXED_ENGINE_70_30"):
            scored,events,rep=methods[name].run_simulation(features,lookup,sessions)
            expected=scored_all[name].drop(columns=LABELS)
            pd.testing.assert_frame_equal(scored.reset_index(drop=True),expected.reset_index(drop=True))
            report["scenario_reports"][name]=rep
            events.to_csv(out/f"{name}_DAILY_OPEN_SCENARIO_EVENTS.csv",index=False,encoding="utf-8-sig")
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    (out/"COMPARISON.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    brief=[m for m in output_metrics if m["scope"]=="all" and not m["apply_defense"]]
    print(pd.DataFrame(brief)[["method","k","label","hits","known","precision_known"]].to_string(index=False))


if __name__=="__main__":
    main()
