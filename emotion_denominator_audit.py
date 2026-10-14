"""Reconstruct previous-limit feedback with explicit observed/missing denominators.

Only yesterday's roster and today's prices enter today's statistics.
No entry, leader designation or learned emotion regime is produced.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from full_pool_outcomes import usable
from premium_opportunity_research import rounded_limit


def broken_feedback(codes, daily, previous, current):
    """Codes were confirmed broken at previous close; never select on current loss."""
    values = []
    if codes is not None:
        for code in codes:
            old_key, new_key = (code, previous), (code, current)
            old = daily.loc[old_key].to_dict() if old_key in daily.index else None
            new = daily.loc[new_key].to_dict() if new_key in daily.index else None
            if not usable(old) or not usable(new):
                continue
            if abs(float(new["pre_close"])-float(old["close"])) > .011:
                continue
            if float(new["high"]) > rounded_limit(old["close"], "1.10")+.011:
                continue
            values.append(float(new["close"])/float(new["pre_close"])-1)
    size = len(codes) if codes is not None else None
    series = pd.Series(values, dtype=float)
    return dict(previous_broken_roster_known=codes is not None,
                previous_broken_denominator=size,
                previous_broken_observed=len(values) if codes is not None else None,
                previous_broken_missing=size-len(values) if size is not None else None,
                previous_broken_coverage=len(values)/size if size else None,
                previous_broken_return_median=series.median() if len(values) else None,
                previous_broken_return_q10=series.quantile(.1) if len(values) else None,
                previous_broken_negative_share=float(series.lt(0).mean()) if len(values) else None,
                previous_broken_roster_codes=";".join(sorted(codes)) if codes is not None else None,
                previous_broken_metrics_basis="observed_subset_not_complete_if_missing")


def build(observations, daily, calendar, end):
    if calendar != sorted(set(calendar)) or not daily.index.is_unique:
        raise ValueError("invalid calendar or price keys")
    if observations.duplicated(["code", "signal_date"]).any():
        raise ValueError("duplicate observation")
    rosters = {day: group for day, group in observations.groupby("signal_date")}
    results = []
    broken_by_date = {}
    for previous, current in zip(calendar[:-1], calendar[1:]):
        if current >= end or previous not in rosters:
            continue
        records = []
        for item in rosters[previous].to_dict("records"):
            prior_key, key = (item["code"], previous), (item["code"], current)
            prior = daily.loc[prior_key].to_dict() if prior_key in daily.index else None
            bar = daily.loc[key].to_dict() if key in daily.index else None
            valid = usable(prior) and usable(bar)
            if valid:
                ceiling = rounded_limit(prior["close"], "1.10")
                valid = abs(float(bar["pre_close"])-float(prior["close"])) <= .011 and float(bar["high"]) <= ceiling+.011
            records.append(dict(code=item["code"], boards=item["boards"],
                                observed=valid,
                                promoted=(abs(float(bar["close"])-ceiling) < .00001) if valid else None,
                                feedback=(float(bar["close"])/float(bar["pre_close"])-1) if valid else None))
        roster = pd.DataFrame(records)
        output = dict(date=current, previous_session=previous, scope="provided_historical_limit_up_roster",
                      status="DIAGNOSTIC_RECONSTRUCTION", unavailable_is_failure=False)
        groups = {"all": roster, "first": roster[roster.boards.eq(1)],
                  "mid_2to4": roster[roster.boards.between(2,4)],
                  "high_5plus": roster[roster.boards.ge(5)]}
        for name, cohort in groups.items():
            available = cohort[cohort.observed]
            count, found = len(cohort), len(available)
            wins = int(available.promoted.eq(True).sum())
            output.update({name+"_denominator": count, name+"_observed": found,
                           name+"_missing": count-found,
                           name+"_coverage": found/count if count else None,
                           name+"_promotion_observed": wins/found if found else None,
                           name+"_promotion_complete": wins/count if count and found==count else None,
                           name+"_promotion_lower_bound": wins/count if count else None,
                           name+"_promotion_upper_bound": (wins+count-found)/count if count else None})
        feedback = roster.loc[roster.observed, "feedback"].astype(float)
        broken = roster.loc[roster.observed & roster.promoted.eq(False), "feedback"].astype(float)
        broken_codes = roster.loc[roster.observed & roster.promoted.eq(False), "code"].tolist()
        broken_by_date[current] = broken_codes
        output.update(broken_feedback(broken_by_date.get(previous), daily, previous, current))
        # An observed roster may itself be incomplete if yesterday's limit cohort
        # had missing feedback at the time of identifying broken names.
        output["previous_broken_roster_scope"] = "previously_confirmed_breaks_only_unknown_breaks_not_inferred"
        output.update(previous_limit_return_median=feedback.median() if len(feedback) else None,
                      previous_limit_return_q10=feedback.quantile(.1) if len(feedback) else None,
                      broken_today_count=len(broken),
                      broken_today_return_q10=broken.quantile(.1) if len(broken) else None,
                      unknown_prior_height_count=int(roster.boards.isna().sum()))
        results.append(output)
    return pd.DataFrame(results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("observations", "daily", "calendar", "old-market", "output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--end-exclusive",required=True)
    args=parser.parse_args()
    daily=pd.read_csv(args.daily,dtype={"code":str})
    result=build(pd.read_csv(args.observations,dtype={"code":str}),
                 daily[daily.date<args.end_exclusive].set_index(["code","date"]),
                 pd.read_csv(args.calendar).date.tolist(),args.end_exclusive)
    args.output.mkdir(parents=True,exist_ok=False)
    result.to_csv(args.output/"feedback_denominators.csv",index=False,encoding="utf-8-sig")
    old=pd.read_csv(args.old_market)[["date","promotion_overall","promotion_1to2","promotion_mid"]]
    comparison=result.merge(old,on="date",how="left",validate="one_to_one")
    comparison.to_csv(args.output/"legacy_comparison.csv",index=False,encoding="utf-8-sig")
    report=dict(days=len(result), days_with_missing_previous_limit_feedback=int(result.all_missing.gt(0).sum()),
                incomplete_first_board_days=int(result.first_missing.gt(0).sum()),
                total_missing_feedback=int(result.all_missing.sum()),
                days_with_previous_broken_roster=int(result.previous_broken_roster_known.sum()),
                days_with_missing_previous_broken_feedback=int(result.previous_broken_missing.gt(0).sum()),
                previous_broken_observations=int(result.previous_broken_observed.sum()),
                scope="available current-universe history, not complete market",
                no_model_trained=True, no_live_probability=True,
                source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (args.observations,args.daily,args.calendar,args.old_market,Path(__file__))})
    (args.output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
