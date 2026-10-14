"""Independent recount of the saved, single-attempt R3 comparison; no fitting."""
import csv
import math
from datetime import datetime
from pathlib import Path

import joblib
import pytest

from model_upgrade_research import read, sha

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT/"research/model_upgrade_validation/relay_phase_20261009T095400Z"


def saved_rows():
    with (PHASE/"full_candidate_comparison.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("truth", "prediction"):
            assert row[key] in ("True", "False", "")
            row[key] = {"True": True, "False": False, "": None}[row[key]]
        row["boards"] = int(row["boards"])
        for key in ("raw_score", "actual_one_word_daily"):
            row[key] = float(row[key]) if row[key] else None
    return rows


def recount(rows):
    scored = [r for r in rows if r["truth"] is not None and r["prediction"] is not None]
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for row in scored:
        counts[("TP" if row["truth"] else "FP") if row["prediction"] else ("FN" if row["truth"] else "TN")] += 1
    tp, tn, fp = counts["TP"], counts["TN"], counts["FP"]
    positives = sum(r["truth"] is True for r in rows)
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/positives if positives else None
    f1 = None if precision is None or recall is None else (2*precision*recall/(precision+recall) if precision+recall else 0.)
    return {"original_registered_count": len(rows), "scoreable_count": len(scored),
            "coverage": len(scored)/len(rows) if rows else None,
            "Accuracy": (tp+tn)/len(scored) if scored else None,
            "Precision": precision, "complete_positive_Recall": recall, "F1": f1,
            "confusion": counts, "positive_support": positives,
            "complete_positive_misses": positives-tp,
            "signal_days": len({r["signal_date"] for r in rows})}


def assert_metrics_equal(actual, expected):
    for key, value in actual.items():
        if isinstance(value, float):
            assert expected[key] == pytest.approx(value, rel=0, abs=1e-12), key
        else:
            assert expected[key] == value, key


def rank_first(rows, days):
    daily = []
    for day in days:
        group = [r for r in rows if r["signal_date"] == day]
        status, chosen = "NO_SAVED_CANDIDATES", None
        if group:
            if any(r["raw_score"] is None for r in group):
                status = "UNKNOWN_SCORE"
            elif any(r["actual_one_word_daily"] not in (0, 1) for r in group):
                status = "UNKNOWN_OHLC_IDENTITY"
            else:
                eligible = [r for r in group if r["actual_one_word_daily"] == 0]
                chosen = min(eligible, key=lambda r: (-r["raw_score"], r["code"])) if eligible else None
                status = "NO_NONWORD" if chosen is None else "MISSING_LABEL" if chosen["truth"] is None else "SCORED"
        daily.append({"signal_date": day, "status": status,
                      "selected_code": chosen["code"] if chosen else None,
                      "original_prediction": chosen["prediction"] if chosen else None,
                      "truth": chosen["truth"] if chosen else None})
    scored = [r for r in daily if r["status"] == "SCORED"]
    return {"official_days": len(days), "scoreable_days": len(scored),
            "promotions": sum(r["truth"] is True for r in scored),
            "rate": sum(r["truth"] is True for r in scored)/len(scored) if scored else None,
            "daily_rows": daily}


def verification():
    protocol = read(PHASE/"protocol_BEFORE_EFFECTS.json")
    result = read(PHASE/"report.json")
    receipt = read(protocol["cached_receipt"])
    assert sha(PHASE/"full_candidate_comparison.csv") == result["predictions_sha256"]
    assert sha(PHASE/"protocol_BEFORE_EFFECTS.json") == result["protocol_sha256"]
    assert sha(receipt["prepared_path"]) == receipt["prepared_sha256"]
    raw = joblib.load(receipt["prepared_path"])
    sessions = receipt["sessions"]
    predecessor = dict(zip(sessions[1:], sessions[:-1]))
    successor = dict(zip(sessions[:-1], sessions[1:]))
    raw_index = {(r.date, r.code): r for r in raw.itertuples(index=False)}
    assert len(raw_index) == len(raw)
    transitions, phase_known = set(), set()
    for key, row in raw_index.items():
        previous = raw_index.get((predecessor.get(row.date), row.code))
        if (previous is not None and previous.signal_one_word in (0, 1)
                and previous.boards+1 == row.boards and row.signal_one_word in (0, 1)):
            phase_known.add(key)
            if previous.signal_one_word == 1 and row.signal_one_word == 0:
                transitions.add(key)
    rows = saved_rows(); assert len(rows) == 7414
    with (Path(protocol["base_root"])/"relay_history/predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        baseline = {(r["fold"], r["date"], r["code"]): r for r in csv.DictReader(handle) if r["arm"] == "BASE"}
    summary = {}
    for fold in protocol["folds"]:
        expected_roster = {key for key in raw_index if fold["start"] <= key[0] <= fold["end"]}
        arm_rows = {arm: [r for r in rows if r["fold"] == fold["id"] and r["arm"] == arm]
                    for arm in ("BASE", "R3")}
        evidence = result["comparisons"][fold["id"]]
        recomputed = {}
        for arm, cohort in arm_rows.items():
            assert len(cohort) == len(expected_roster)
            assert {(r["signal_date"], r["code"]) for r in cohort} == expected_roster
            for row in cohort:
                original = raw_index[(row["signal_date"], row["code"])]
                original_target = original.target_date if isinstance(original.target_date, str) else ""
                assert row["target_date"] == original_target
                truth = (bool(original.target) if original.label_valid and not math.isnan(original.target)
                         and original_target == successor[row["signal_date"]] else None)
                assert row["truth"] is truth
                assert row["boards"] == int(original.boards)
                if arm == "BASE":
                    saved = baseline[(fold["id"], row["signal_date"], row["code"])]
                    assert row["raw_score"] == pytest.approx(float(saved["raw_score"]), rel=0, abs=1e-12)
                    saved_call = bool(int(float(saved["predicted"]))) if saved["status"] == "PREDICTED_UNCALIBRATED" else None
                    assert row["prediction"] is saved_call
            measured = recount(cohort)
            assert_metrics_equal(measured, evidence[arm])
            for boards in {r["boards"] for r in cohort}:
                assert_metrics_equal(recount([r for r in cohort if r["boards"] == boards]), evidence[arm]["by_boards"][str(boards)])
            top = rank_first(cohort, [d for d in sessions if fold["start"] <= d <= fold["end"]])
            assert_metrics_equal(top, evidence[arm]["true_nonword_rank_first"])
            subgroup = recount([r for r in cohort if (r["signal_date"], r["code"]) in transitions])
            expected_subgroup = next(g["metrics"] for g in evidence["transition_subgroup_diagnostic_only"] if g["arm"] == arm)
            assert_metrics_equal(subgroup, expected_subgroup)
            recomputed[arm] = {**measured, "true_nonword_rank_first": {k: v for k,v in top.items() if k != "daily_rows"},
                               "transition_subgroup": subgroup}
        delta = recomputed["R3"]["Accuracy"]-recomputed["BASE"]["Accuracy"]
        assert evidence["Accuracy_delta"] == pytest.approx(delta, abs=1e-12, rel=0)
        assert evidence["phase_known_test_rows"] == len(expected_roster & phase_known)
        assert evidence["test_transition_rows"] == len(expected_roster & transitions)
        summary[fold["id"]] = {**recomputed, "Accuracy_delta": delta,
                               "phase_known_rows": len(expected_roster & phase_known),
                               "transition_rows": len(expected_roster & transitions)}
    assert result["status"] == "REJECTED_FIXED_PHASE_CHALLENGE"
    assert result["production_changed"] is False
    assert result["new_mature_forward_samples"] == 0
    assert len(result["fit_logs"]) == 3
    assert all(log["fitted_once"] and not log["deployment_weights_saved"] for log in result["fit_logs"])
    stamps = [read(PHASE/name)["formed_at"] for name in (
        "protocol_BEFORE_EFFECTS.json", "implementation_BEFORE_FITS.json",
        "training_checks_BEFORE_FITS.json", "attempt_STARTED.json", "report.json")]
    assert list(map(datetime.fromisoformat, stamps)) == sorted(map(datetime.fromisoformat, stamps))
    return summary


def test_full_rosters_unknowns_original_scores_and_independent_metrics_match():
    verification()
