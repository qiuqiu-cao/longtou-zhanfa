"""Create purged date-grouped partitions for the full-pool ranking target.

Every candidate from one signal session stays in the same partition.  A row is
purged when its B+2 proxy label became available on or after the next segment,
so target windows cannot cross a development boundary.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from full_pool_rank_dataset import TARGET_DEFINITION, SELECTION_SCOPE
from leader_target_screen_v17 import timestamp


SEGMENTS = ("train", "calibration", "validation", "blind")
MINIMUM_SESSION_COUNTS = {"train": 250, "calibration": 40, "validation": 60, "blind": 100}
MINIMUM_ROW_COUNTS = {"train": 1000, "calibration": 150, "validation": 250, "blind": 500}


def audit(dataset, sessions, split, asof):
    cutoff = timestamp(asof)
    if sessions != sorted(set(sessions)) or not sessions:
        raise ValueError("sorted unique sourced trading calendar required")
    index = {day: i for i, day in enumerate(sessions)}
    embargo = split.get("embargo_sessions")
    if isinstance(embargo, bool) or not isinstance(embargo, int) or embargo < 2:
        raise ValueError("B+2 proxy requires at least two embargo sessions")
    bounds = {}
    for segment in SEGMENTS:
        start, end = split.get(segment + "_start"), split.get(segment + "_end")
        if start not in index or end not in index or index[start] > index[end]:
            raise ValueError("invalid " + segment + " bounds")
        bounds[segment] = (start, end)
    for left, right in zip(SEGMENTS[:-1], SEGMENTS[1:]):
        if index[bounds[right][0]] - index[bounds[left][1]] - 1 < embargo:
            raise ValueError("insufficient embargo between " + left + " and " + right)
    if timestamp(bounds["blind"][1] + "T23:59:59+08:00") > cutoff:
        raise ValueError("blind window is not fully in the past")
    if (dataset.get("target_definition") != TARGET_DEFINITION
            or dataset.get("selection_scope") != SELECTION_SCOPE):
        raise ValueError("incompatible full-pool ranking dataset")

    issues = []
    if dataset.get("training_ready") is not True or dataset.get("full_pool_scope_established") is not True:
        issues.append("UPSTREAM_FULL_POOL_RANK_DATASET_NOT_READY")
    rows = dataset.get("rows", [])
    row_by_id = {}
    day_to_ids = {}
    for row in rows:
        candidate_id = row.get("candidate_id")
        if not candidate_id or candidate_id in row_by_id:
            raise ValueError("missing or duplicate candidate id")
        row_by_id[candidate_id] = row
        day_to_ids.setdefault(row.get("signal_date"), []).append(candidate_id)

    assignments = {segment: [] for segment in SEGMENTS}
    embargo_ids, outside_ids = [], []
    segment_by_day = {}
    for day in day_to_ids:
        if day not in index:
            outside_ids.extend(day_to_ids[day])
            continue
        matches = [segment for segment, (start, end) in bounds.items()
                   if index[start] <= index[day] <= index[end]]
        if len(matches) == 1:
            segment_by_day[day] = matches[0]
            assignments[matches[0]].extend(day_to_ids[day])
        elif index[bounds["train"][0]] <= index[day] <= index[bounds["blind"][1]]:
            embargo_ids.extend(day_to_ids[day])
        else:
            outside_ids.extend(day_to_ids[day])

    # This is both a check and an explicit invariant in the report.
    if any(len({segment_by_day.get(row_by_id[candidate_id].get("signal_date"))
                for candidate_id in ids}) > 1 for ids in day_to_ids.values()):
        raise ValueError("same-day candidates split across partitions")

    next_start = {"train": bounds["calibration"][0], "calibration": bounds["validation"][0],
                  "validation": bounds["blind"][0], "blind": None}
    kept, purged = {segment: [] for segment in SEGMENTS}, {segment: [] for segment in SEGMENTS}
    for segment, candidate_ids in assignments.items():
        for candidate_id in candidate_ids:
            row = row_by_id[candidate_id]
            invalid = (row.get("label_available") is not True
                       or not isinstance(row.get("opportunity_proxy_label"), bool)
                       or bool(row.get("issues")))
            try:
                label_time = timestamp(row["label_available_at"])
                if label_time > cutoff:
                    invalid = True
                boundary = next_start[segment]
                crosses = boundary is not None and label_time >= timestamp(boundary + "T00:00:00+08:00")
                if segment == "blind" and label_time > timestamp(bounds["blind"][1] + "T23:59:59+08:00"):
                    crosses = True
            except (KeyError, TypeError, ValueError):
                invalid, crosses = True, True
            if invalid:
                purged[segment].append(candidate_id)
                issues.append("INVALID_RANK_ROW_" + str(candidate_id))
            elif crosses:
                purged[segment].append(candidate_id)
            else:
                kept[segment].append(candidate_id)

    row_counts = {segment: len(kept[segment]) for segment in SEGMENTS}
    session_counts = {segment: len({row_by_id[candidate_id]["signal_date"] for candidate_id in kept[segment]})
                      for segment in SEGMENTS}
    class_ready, group_ready = {}, {}
    for segment in ("train", "calibration", "validation"):
        labels = {row_by_id[candidate_id]["opportunity_proxy_label"] for candidate_id in kept[segment]}
        groups = {row_by_id[candidate_id]["board_group"] for candidate_id in kept[segment]}
        class_ready[segment] = labels == {False, True}
        group_ready[segment] = groups == {"FIRST_BOARD", "CONSECUTIVE_BOARD"}
    size_ready = (all(row_counts[name] >= MINIMUM_ROW_COUNTS[name] for name in SEGMENTS)
                  and all(session_counts[name] >= MINIMUM_SESSION_COUNTS[name] for name in SEGMENTS))
    development_ready = (not issues and dataset.get("training_ready") is True and size_ready
                         and all(class_ready.values()) and all(group_ready.values()))
    board_counts = {segment: dict(Counter(str(row_by_id[candidate_id]["boards"])
                                          for candidate_id in kept[segment])) for segment in SEGMENTS}
    return {
        "status": "READY_FOR_FULL_POOL_RANK_DEVELOPMENT" if development_ready else "DIAGNOSTIC_ONLY",
        "split": split,
        "partitions": kept,
        "purged": purged,
        "row_counts": row_counts,
        "session_counts": session_counts,
        "board_counts": board_counts,
        "minimum_row_counts": MINIMUM_ROW_COUNTS,
        "minimum_session_counts": MINIMUM_SESSION_COUNTS,
        "embargo_candidate_count": len(embargo_ids),
        "outside_candidate_count": len(outside_ids),
        "development_class_ready": class_ready,
        "development_group_ready": group_ready,
        "same_signal_date_kept_together": True,
        "size_ready": size_ready,
        "issues": sorted(set(issues)),
        "target_definition": TARGET_DEFINITION,
        "selection_scope": SELECTION_SCOPE,
        "blind_proxy_metrics": None,
        "blind_labels_not_summarized": True,
        "blind_label_access_control": "NOT_ENFORCED_SOURCE_DATASET_CONTAINS_LABELS",
        "limitations": [
            "blind labels remain present in the source JSON and require an independent evaluator for real access control",
            "the target remains a price-opportunity proxy and cannot establish complete-trade profitability",
        ],
    }


def run(dataset_path, calendar_path, split_path, output_path, asof):
    dataset_path, calendar_path, split_path, target = map(Path, (dataset_path, calendar_path, split_path, output_path))
    if target.exists():
        raise FileExistsError("choose a new immutable full-pool split report")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    sessions = (json.loads(calendar_path.read_text(encoding="utf-8")) if calendar_path.suffix.lower() == ".json"
                else [line.strip().split(",")[0] for line in calendar_path.read_text(encoding="utf-8-sig").splitlines()[1:]
                      if line.strip()])
    result = audit(dataset, sessions, split, asof)
    result["source_hashes"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (dataset_path, calendar_path, split_path, Path(__file__))}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--calendar", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asof", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.calendar, args.split, args.output, args.asof), ensure_ascii=False, indent=2))
