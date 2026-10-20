"""Read-only diagnosis of complete saved first-board prediction cohorts.

Oracle fields are retrospective outcomes, never model inputs or live selectors.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import first_board_matured_event_model as previous
import numpy as np
import pandas as pd

TARGET = stage.TARGET
REF = risk.REFERENCE
SCREEN = previous.SCREEN


def ranked(frame):
    out = frame.sort_values(["signal_date", "model_score", "code"], ascending=[True, False, True]).copy()
    out["saved_score_rank"] = out.groupby("signal_date", sort=False).cumcount() + 1
    return out


def daily_audit(full, allowed_only):
    candidates = full[full.research_selection_allowed] if allowed_only else full
    ordered = ranked(candidates)
    order_groups = {d: g for d, g in ordered.groupby("signal_date", sort=False)}
    daily = []
    for date, complete in full.groupby("signal_date", sort=True):
        rows = order_groups.get(date, ordered.iloc[:0])
        positives = rows[rows[TARGET].eq(1)]
        row = dict(signal_date=date, complete_candidates=len(complete), complete_known=int(complete[TARGET].notna().sum()),
            complete_unknown=int(complete[TARGET].isna().sum()), complete_positives=int(complete[TARGET].eq(1).sum()),
            admissible_candidates=len(rows), admissible_known=int(rows[TARGET].notna().sum()),
            admissible_unknown=int(rows[TARGET].isna().sum()), admissible_positives=int(rows[TARGET].eq(1).sum()),
            oracle_has_known_winner_LABEL_ONLY=bool(len(positives)),
            oracle_best_known_winner_rank_LABEL_ONLY=int(positives.saved_score_rank.min()) if len(positives) else np.nan,
            filter_removes_all_known_winners_LABEL_ONLY=bool(complete[TARGET].eq(1).any() and not len(positives)),
            no_known_winner_with_unknown_LABEL_ONLY=bool(not len(positives) and rows[TARGET].isna().any()),
            all_admissible_known_negative_LABEL_ONLY=bool(len(rows) and rows[TARGET].notna().all() and not len(positives)),
            no_admissible_candidate=not len(rows))
        if len(rows):
            first = rows.iloc[0]
            row.update(top1_code=first.code, top1_name=first["name"], top1_score=float(first.model_score),
                top1_outcome_LABEL_ONLY=first[TARGET], top1_exclusion_reason=first.exclusion_reason,
                top1_one_price=int(first.one_price), top1_float_cap=float(first.float_cap) if pd.notna(first.float_cap) else np.nan)
        else:
            row.update(top1_code=None, top1_name=None, top1_score=np.nan, top1_outcome_LABEL_ONLY=np.nan,
                top1_exclusion_reason=None, top1_one_price=np.nan, top1_float_cap=np.nan)
        for k in (1, 3, 5):
            top = rows.head(k)
            row[f"top{k}_selected"] = len(top)
            row[f"top{k}_known"] = int(top[TARGET].notna().sum())
            row[f"top{k}_unknown"] = int(top[TARGET].isna().sum())
            row[f"top{k}_positive_LABEL_ONLY"] = int(top[TARGET].eq(1).sum())
            row[f"top{k}_contains_known_winner_LABEL_ONLY"] = bool(top[TARGET].eq(1).any())
        # Mutually exclusive miss categories retain uncertain candidate pools.
        if row["no_admissible_candidate"]:
            category = "NO_ADMISSIBLE_CANDIDATE"
        elif pd.isna(row["top1_outcome_LABEL_ONLY"]):
            category = "TOP1_OUTCOME_UNKNOWN"
        elif row["top1_outcome_LABEL_ONLY"] == 1:
            category = "TOP1_HIT"
        elif len(positives):
            best = int(positives.saved_score_rank.min())
            category = "TOP1_MISS_WINNER_RANK2TO3" if best <= 3 else ("TOP1_MISS_WINNER_RANK4TO5" if best <= 5 else "TOP1_MISS_WINNER_BEYOND5")
        elif row["admissible_unknown"]:
            category = "TOP1_MISS_NO_KNOWN_WINNER_BUT_POOL_UNKNOWN"
        else:
            category = "TOP1_MISS_ALL_ADMISSIBLE_KNOWN_NEGATIVE"
        row["retrospective_error_category_LABEL_ONLY"] = category
        daily.append(row)
    return ordered, pd.DataFrame(daily)


def summarize(full, ordered, daily):
    metric = dict(rows=len(full), dates=len(daily), complete_known=int(full[TARGET].notna().sum()),
        complete_unknown=int(full[TARGET].isna().sum()), complete_positive_pool=int(full[TARGET].eq(1).sum()),
        admissible_rows=len(ordered), admissible_unknown=int(ordered[TARGET].isna().sum()),
        admissible_positive_pool=int(ordered[TARGET].eq(1).sum()),
        known_winner_dates_LABEL_ONLY=int(daily.oracle_has_known_winner_LABEL_ONLY.sum()),
        oracle_known_winner_date_fraction_LABEL_ONLY=float(daily.oracle_has_known_winner_LABEL_ONLY.mean()),
        filter_removes_all_known_winner_dates_LABEL_ONLY=int(daily.filter_removes_all_known_winners_LABEL_ONLY.sum()),
        no_known_winner_dates_with_unknown_LABEL_ONLY=int(daily.no_known_winner_with_unknown_LABEL_ONLY.sum()),
        all_admissible_known_negative_dates_LABEL_ONLY=int(daily.all_admissible_known_negative_LABEL_ONLY.sum()),
        abstention_dates=int(daily.no_admissible_candidate.sum()),
        error_categories_LABEL_ONLY=daily.retrospective_error_category_LABEL_ONLY.value_counts().to_dict(),
        top={})
    for k in (1, 3, 5):
        count = int(daily[f"top{k}_selected"].sum())
        known = int(daily[f"top{k}_known"].sum())
        hits = int(daily[f"top{k}_positive_LABEL_ONLY"].sum())
        metric["top"][str(k)] = dict(selected=count, known=known, unknown=count-known, positive_stocks=hits,
            stock_precision=hits/known if known else None,
            full_positive_recall=hits/metric["complete_positive_pool"] if metric["complete_positive_pool"] else None,
            dates_containing_known_winner=int(daily[f"top{k}_contains_known_winner_LABEL_ONLY"].sum()),
            date_any_winner_fraction=float(daily[f"top{k}_contains_known_winner_LABEL_ONLY"].mean()),
            semantics="Top1stockprecision is primary. Top3/5any-winner fractions are diagnostic,not Top1success or a tradable oracle.")
    return metric


def run(output):
    out = Path(output)
    if out.exists():
        raise FileExistsError("Refuse overwrite")
    out.mkdir(parents=True)
    dataset = REF / "all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
    screen_file = SCREEN / "COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv"
    files = [dataset, screen_file, REF / "validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv", REF / "S1_ALL_PREDICTIONS.csv",
             REF / "report.json", SCREEN / "report.json", Path(base.__file__), Path(stage.__file__), Path(risk.__file__)]
    protocol = dict(formed_at=base.now(), budget_minutes=30,
        previous_goal_turn="PROGRESS:new market-relative model rejected after2cores;independent28503x8fieldrecomputation and saved-model replay completed;source qualification produced no eligible industry mapping",
        task="Diagnose whether saved first-close errors reflect unavailable admissible winners,ranking ambiguity or fixed-screen opportunity loss",
        metrics="Complete cohort known/unknown/positive counts;daily known-winner availability before/after screen;Top1precision and Top3/5stockprecision plus separate date-any-winner fractions;complete mutually exclusive error categories and all excluded promoters",
        boundaries="Only original saved validation2025H1 and evaluation2025Jul-2026Aug scores;exact unchanged stage labels/screen;no model fits or new provider requests",
        expected_decision="Identify one evidence-based next hypothesis;do not select a new threshold/algorithm or call multi-pick any-hit a60%Top1success",
        stop="Complete fixed read-only comparison and verification,or30minutes;no parameter search",
        oracle_fields_are_future_labels_and_prohibited_as_inputs=True, historical_period_already_researched_not_pristine_blind=True,
        inputs={str(p.resolve()):base.sha(p) for p in files}, calendar_sha256=base.sha(base.INPUTS["calendar"]),
        model_fits=0, new_forward_results=0, actual_trades=0, production_changes=False, source_sha256=base.sha(__file__))
    base.write_json(out / "protocol_BEFORE_AUDIT.json", protocol)
    shutil.copy2(__file__, out / "source_before_audit.py")
    original = pd.read_csv(dataset, dtype={"code": str}, float_precision="round_trip")
    original = original[original.stage.eq(1)].copy()
    screen = pd.read_csv(screen_file, dtype={"code": str}, float_precision="round_trip")
    keys = ["signal_date", "code"]
    assert set(map(tuple,original[keys].to_numpy())) == set(map(tuple,screen[keys].to_numpy()))
    selected_fields = ["research_selection_allowed", "exclusion_reason", "threeboard_evidence_status"]
    original = original.merge(screen[keys+selected_fields], on=keys, validate="one_to_one")
    parts = []
    for name, file in [("validation", files[2]), ("evaluation", files[3])]:
        frozen = risk.attach_frozen(original, file)
        frozen["source_period"] = name
        parts.append(frozen)
    full = pd.concat(parts, ignore_index=True)
    assert not full.duplicated(keys).any()
    full.to_csv(out / "COMPLETE_SAVED_POOLS_UNCHANGED_LABELS_SCORES_AND_SCREEN.csv", index=False, encoding="utf-8-sig")
    report = dict(formed_at=None, periods={}, goal_60_passed=False, new_fits=0, new_forward_results=0, actual_trades=0)
    for period, mask in [("validation2025H1", full.source_period.eq("validation")),
                         ("evaluation_all", full.source_period.eq("evaluation")),
                         ("2025H2", full.signal_date.between("2025-07-01", "2025-12-31")),
                         ("2026", full.signal_date.ge("2026-01-01"))]:
        part = full[mask]
        report["periods"][period] = {}
        for universe, frame in [("all", part), ("non_one_price", part[part.one_price.eq(0)])]:
            for screen_name, apply in [("unfiltered", False), ("fixed_screen", True)]:
                ordered, daily = daily_audit(frame, apply)
                key = universe + "_" + screen_name
                report["periods"][period][key] = summarize(frame, ordered, daily)
                daily.to_csv(out / f"{period}_{key}_DAILY_ORACLE_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
                positives = ordered[ordered[TARGET].eq(1)]
                positives.to_csv(out / f"{period}_{key}_ALL_KNOWN_WINNER_RANKS_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
    for reason, mask in [("all_excluded", ~full.research_selection_allowed),
                         ("above_cap", full.exclusion_reason.fillna("").str.contains("ABOVE_DEVELOPMENT_CAP")),
                         ("thin_turnover_proxy", full.exclusion_reason.fillna("").str.contains("MICROCAP_THIN")),
                         ("cap_missing", full.exclusion_reason.fillna("").str.contains("CAP_MISSING"))]:
        rows = full[mask]
        report.setdefault("exclusions", {})[reason] = dict(rows=len(rows), known=int(rows[TARGET].notna().sum()), unknown=int(rows[TARGET].isna().sum()),
            promoters=int(rows[TARGET].eq(1).sum()), dates=int(rows.signal_date.nunique()))
    full[(~full.research_selection_allowed)&full[TARGET].eq(1)].to_csv(out / "ALL_EXCLUDED_PROMOTERS_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
    for file, digest in protocol["inputs"].items():
        assert base.sha(file) == digest
    report.update(formed_at=base.now(), status="READ_ONLY_FIXED_AUDIT_FINISHED")
    base.write_json(out / "report.json", report)
    print({period:{key:{"top1":m["top"]["1"],"oracle_days":m["known_winner_dates_LABEL_ONLY"],"filter_all_winners_lost":m["filter_removes_all_known_winner_dates_LABEL_ONLY"]} for key,m in groups.items() if key.startswith("all_")} for period,groups in report["periods"].items()})
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
