"""Create purged chronological partitions while keeping blind labels sealed."""
import argparse
import hashlib
import json
from pathlib import Path

from leader_target_screen_v17 import timestamp


SEGMENTS=("train","calibration","validation","blind")
MINIMUM_COUNTS={"train":100,"calibration":30,"validation":30,"blind":100}


def audit(dataset,sessions,split,asof):
    cutoff=timestamp(asof)
    if sessions!=sorted(set(sessions)) or not sessions:
        raise ValueError("sorted unique trading calendar required")
    index={day:i for i,day in enumerate(sessions)}
    embargo=split.get("embargo_sessions")
    if isinstance(embargo,bool) or not isinstance(embargo,int) or embargo<1:
        raise ValueError("at least one embargo session required")
    bounds={}
    for segment in SEGMENTS:
        start,end=split.get(segment+"_start"),split.get(segment+"_end")
        if start not in index or end not in index or index[start]>index[end]:
            raise ValueError("invalid "+segment+" bounds")
        bounds[segment]=(start,end)
    for left,right in zip(SEGMENTS[:-1],SEGMENTS[1:]):
        if index[bounds[right][0]]-index[bounds[left][1]]-1<embargo:
            raise ValueError("insufficient embargo between "+left+" and "+right)
    if not all(index[bounds[a][1]]<index[bounds[b][0]] for a,b in zip(SEGMENTS[:-1],SEGMENTS[1:])):
        raise ValueError("overlapping chronological partitions")
    if timestamp(bounds["blind"][1]+"T15:00:00+08:00")>cutoff:
        raise ValueError("blind period is not fully observable at cutoff")
    rows=dataset.get("rows",[])
    if dataset.get("generalization_scope")!="FROZEN_SELECTION_POLICY_CONDITIONAL_ONLY":
        raise ValueError("dataset scope is incompatible")
    assignments={segment:[] for segment in SEGMENTS}
    embargo_rows=[]
    outside=[]
    issues=[]
    upstream_ready=dataset.get("training_ready") is True
    if not upstream_ready:
        issues.append("UPSTREAM_COMPLETE_TRADE_DATASET_NOT_READY")
    for row in rows:
        if not row.get("training_eligible"):
            continue
        day=row.get("signal_date")
        matches=[segment for segment,(start,end) in bounds.items() if index.get(start,-1)<=index.get(day,-2)<=index.get(end,-1)]
        if len(matches)==1:
            assignments[matches[0]].append(row)
        elif day in index and index[bounds["train"][0]]<=index[day]<=index[bounds["blind"][1]]:
            embargo_rows.append(row)
        else:
            outside.append(row)
    purged={segment:[] for segment in SEGMENTS}
    kept={segment:[] for segment in SEGMENTS}
    next_start={"train":bounds["calibration"][0],"calibration":bounds["validation"][0],
                "validation":bounds["blind"][0],"blind":None}
    for segment,items in assignments.items():
        for row in items:
            try:
                closed=timestamp(row["closed_at"])
                if closed>cutoff:
                    raise ValueError("future close")
                boundary=next_start[segment]
                crosses=boundary is not None and closed>=timestamp(boundary+"T00:00:00+08:00")
                if segment=="blind" and closed>timestamp(bounds["blind"][1]+"T23:59:59+08:00"):
                    crosses=True
                (purged if crosses else kept)[segment].append(row)
            except (KeyError,TypeError,ValueError):
                purged[segment].append(row)
                issues.append("INVALID_CLOSE_TIME_"+str(row.get("decision_id")))
    counts={segment:len(kept[segment]) for segment in SEGMENTS}
    class_ready={}
    for segment in ("train","calibration","validation"):
        profits={row.get("complete_trade_profit_label") for row in kept[segment]}
        losses={row.get("large_loss_label") for row in kept[segment]}
        class_ready[segment]=(profits=={False,True} and losses=={False,True})
    size_ready=all(counts[segment]>=MINIMUM_COUNTS[segment] for segment in SEGMENTS)
    development_ready=(upstream_ready and not issues and size_ready and all(class_ready.values()))
    return dict(status="READY_FOR_MODEL_DEVELOPMENT" if development_ready else "DIAGNOSTIC_ONLY",
                split=split,kept_counts=counts,purged_counts={segment:len(purged[segment]) for segment in SEGMENTS},
                embargo_row_count=len(embargo_rows),outside_window_count=len(outside),
                development_class_ready=class_ready,minimum_counts=MINIMUM_COUNTS,size_ready=size_ready,
                partitions={segment:[row["decision_id"] for row in kept[segment]] for segment in SEGMENTS},
                purged={segment:[row.get("decision_id") for row in purged[segment]] for segment in SEGMENTS},
                blind_label_metrics=None,blind_labels_not_read_by_audit=True,
                blind_label_access_control="NOT_ENFORCED_SOURCE_DATASET_CONTAINS_LABELS",
                blind_count=counts["blind"],issues=issues,
                generalization_scope=dataset["generalization_scope"],
                limitations=["blind labels are present in the source dataset but are not summarized by this audit; access is not technically sealed",
                             "this split supports only the frozen selection-policy population, not unselected counterfactuals"])


def run(dataset_path,calendar_path,split_path,output_path,asof):
    dataset_path,calendar_path,split_path,target=map(Path,(dataset_path,calendar_path,split_path,output_path))
    if target.exists():
        raise FileExistsError("choose a new immutable split audit")
    dataset=json.loads(dataset_path.read_text(encoding="utf-8"))
    split=json.loads(split_path.read_text(encoding="utf-8"))
    sessions=(json.loads(calendar_path.read_text(encoding="utf-8")) if calendar_path.suffix.lower()==".json"
              else [row.strip() for row in calendar_path.read_text(encoding="utf-8-sig").splitlines()[1:] if row.strip()])
    result=audit(dataset,sessions,split,asof)
    result["source_hashes"]={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (dataset_path,calendar_path,split_path,Path(__file__))}
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x",encoding="utf-8") as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--calendar",required=True)
    parser.add_argument("--split",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--asof",required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.dataset,args.calendar,args.split,args.output,args.asof),ensure_ascii=False,indent=2))
