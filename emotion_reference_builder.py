"""Build causal percentile references for the five-part emotion evidence pack."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from emotion_evidence_pack import METRICS
from leader_target_screen_v17 import timestamp, finite


METRIC_DEFINITIONS = {metric:unit for definitions in METRICS.values() for metric,(_,unit) in definitions.items()}
REQUIRED = {"date","metric","value","unit","denominator","observed_count","missing_count","scope_id",
            "definition_id","cohort_id","source","observed_at","available_at","data_status"}


def _valid_record(row, cutoff=None):
    try:
        metric = row["metric"]
        value = finite(row["value"])
        denominator, observed, missing = [int(row[key]) for key in ("denominator","observed_count","missing_count")]
        observed_at, available_at = timestamp(row["observed_at"]), timestamp(row["available_at"])
        return (metric in METRIC_DEFINITIONS and row["unit"] == METRIC_DEFINITIONS[metric]
                and value is not None and (row["unit"] != "ratio" or 0 <= value <= 1)
                and denominator >= 1 and observed >= 1 and missing >= 0 and observed+missing == denominator
                and observed_at <= available_at and (cutoff is None or available_at <= cutoff)
                and observed_at.strftime("%Y-%m-%d") == str(row["date"])
                and bool(row["scope_id"]) and bool(row["definition_id"]) and bool(row["cohort_id"])
                and bool(row["source"]) and row["data_status"] in {"verified","delayed","inferred"})
    except (KeyError, TypeError, ValueError):
        return False


def build(frame, asof, sessions, window=120, minimum=60):
    missing = REQUIRED-set(frame.columns)
    if missing:
        raise ValueError("missing columns: "+", ".join(sorted(missing)))
    if window < minimum or minimum < 60:
        raise ValueError("reference window must contain at least 60 prior observations")
    cutoff = timestamp(asof)
    if sessions != sorted(set(sessions)) or cutoff.strftime("%Y-%m-%d") not in sessions:
        raise ValueError("sorted unique trading calendar including current session required")
    rows = frame.to_dict("records")
    usable = [row for row in rows if row.get("date") in sessions and _valid_record(row, cutoff)]
    current_date = cutoff.strftime("%Y-%m-%d")
    current_rows = [row for row in usable if row["date"] == current_date]
    current_scopes = {row["scope_id"] for row in current_rows}
    if len(current_scopes) > 1:
        raise ValueError("current emotion metrics use multiple market scopes")
    by_metric = {}
    for row in current_rows:
        by_metric.setdefault(row["metric"], []).append(row)
    metrics = {}
    audit = {}
    for metric in METRIC_DEFINITIONS:
        options = sorted(by_metric.get(metric, []), key=lambda row:(timestamp(row["observed_at"]),timestamp(row["available_at"])))
        if not options:
            audit[metric] = dict(status="CURRENT_MISSING",reference_count=0)
            continue
        current = options[-1]
        earlier = [row for row in usable if row["metric"] == metric and row["date"] < current_date
                   and row["scope_id"] == current["scope_id"] and row["definition_id"] == current["definition_id"]
                   and row["cohort_id"] == current["cohort_id"] and row["unit"] == current["unit"]
                   and timestamp(row["available_at"]) <= timestamp(current["available_at"])]
        earlier.sort(key=lambda row:(row["date"],timestamp(row["observed_at"])))
        # One latest observation per date avoids overweighting days with more snapshots.
        daily = {}
        for row in earlier:
            daily[row["date"]] = row
        reference_rows = [daily[day] for day in sorted(daily)][-window:]
        if len(reference_rows) < minimum:
            audit[metric] = dict(status="REFERENCE_INSUFFICIENT",reference_count=len(reference_rows),current_options=len(options))
            continue
        series = pd.Series([float(row["value"]) for row in reference_rows],dtype=float)
        first, last = reference_rows[0], reference_rows[-1]
        metrics[metric] = {
            **{key:current[key] for key in ("metric","value","unit","denominator","observed_count","missing_count",
                                             "scope_id","definition_id","cohort_id","source","observed_at",
                                             "available_at","data_status")},
            "max_age_seconds": current.get("max_age_seconds",300),
            "reference": dict(p25=float(series.quantile(.25)),median=float(series.median()),p75=float(series.quantile(.75)),
                              sample_count=len(reference_rows),definition_id=current["definition_id"],
                              cohort_id=current["cohort_id"],source="causal archive rows: "+str(first["source"]),
                              window_start=timestamp(first["observed_at"]).isoformat(),
                              window_end=timestamp(last["observed_at"]).isoformat(),
                              available_at=max(timestamp(row["available_at"]) for row in reference_rows).isoformat()),
        }
        audit[metric] = dict(status="REFERENCE_READY",reference_count=len(reference_rows),current_options=len(options),
                             excluded_incompatible_or_unavailable=len(rows)-len([row for row in rows if row in reference_rows or row is current]))
    return dict(asof=cutoff.isoformat(),scope_id=(next(iter(metrics.values()))["scope_id"] if metrics else None),
                metrics=metrics,reference_window=window,minimum_reference_observations=minimum,
                metric_audit=audit,all_metrics_ready=len(metrics)==len(METRIC_DEFINITIONS),
                invalid_or_non_session_rows=len(rows)-len(usable),
                limitations=["source text summarizes archive provenance; source file and calendar hashes are recorded by CLI",
                             "percentiles are descriptive and are not optimized trading thresholds"])


def run(input_path, calendar_path, output_path, asof, window=120, minimum=60):
    source,calendar_path,target=Path(input_path),Path(calendar_path),Path(output_path)
    if target.exists():
        raise FileExistsError("choose a new immutable reference snapshot path")
    frame=pd.read_csv(source)
    if calendar_path.suffix.lower()==".json":
        sessions=json.loads(calendar_path.read_text(encoding="utf-8"))
    else:
        sessions=pd.read_csv(calendar_path)["date"].astype(str).tolist()
    result=build(frame,asof,sessions,window,minimum)
    result["source_hashes"]={str(source.resolve()):hashlib.sha256(source.read_bytes()).hexdigest(),
                             str(calendar_path.resolve()):hashlib.sha256(calendar_path.read_bytes()).hexdigest(),
                             str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x",encoding="utf-8") as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True)
    parser.add_argument("--calendar",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--asof",required=True)
    parser.add_argument("--window",type=int,default=120)
    parser.add_argument("--minimum",type=int,default=60)
    args=parser.parse_args()
    print(json.dumps(run(args.input,args.calendar,args.output,args.asof,args.window,args.minimum),ensure_ascii=False,indent=2))
