import unittest
from datetime import date, timedelta

from full_pool_rank_dataset import TARGET_DEFINITION, SELECTION_SCOPE
from full_pool_rank_split_audit import audit


def weekdays(n=470):
    result, day = [], date(2024, 1, 2)
    while len(result) < n:
        if day.weekday() < 5:
            result.append(day.isoformat())
        day += timedelta(days=1)
    return result


def dataset(sessions, ready=True):
    rows = []
    for i, day in enumerate(sessions):
        for j, boards in enumerate((1, 2, 3, 4, 5)):
            candidate_id = f"{day}:{j:06d}"
            outcome_day = sessions[min(i + 2, len(sessions) - 1)]
            rows.append({"candidate_id": candidate_id, "signal_date": day, "boards": boards,
                         "board_group": "FIRST_BOARD" if boards == 1 else "CONSECUTIVE_BOARD",
                         "label_available": True, "label_available_at": outcome_day + "T15:05:00+08:00",
                         "opportunity_proxy_label": (i + j) % 2 == 0, "issues": []})
    return {"training_ready": ready, "full_pool_scope_established": ready,
            "target_definition": TARGET_DEFINITION, "selection_scope": SELECTION_SCOPE, "rows": rows}


def split(s):
    return {"embargo_sessions": 2,
            "train_start": s[0], "train_end": s[249],
            "calibration_start": s[252], "calibration_end": s[291],
            "validation_start": s[294], "validation_end": s[353],
            "blind_start": s[356], "blind_end": s[457]}


class FullPoolRankSplitAuditTests(unittest.TestCase):
    def test_ready_split_keeps_whole_days_together(self):
        sessions = weekdays()
        result = audit(dataset(sessions), sessions, split(sessions), sessions[-1] + "T23:59:59+08:00")
        self.assertEqual(result["status"], "READY_FOR_FULL_POOL_RANK_DEVELOPMENT")
        self.assertTrue(result["same_signal_date_kept_together"])
        self.assertIsNone(result["blind_proxy_metrics"])
        self.assertEqual(result["session_counts"]["blind"], 100)

    def test_b2_boundary_rows_are_purged(self):
        sessions = weekdays()
        data = dataset(sessions)
        # Last training signal receives its label exactly at calibration start.
        target_day = sessions[249]
        for row in data["rows"]:
            if row["signal_date"] == target_day:
                row["label_available_at"] = sessions[252] + "T09:00:00+08:00"
        result = audit(data, sessions, split(sessions), sessions[-1] + "T23:59:59+08:00")
        self.assertEqual(len(result["purged"]["train"]), 5)

    def test_two_session_embargo_is_mandatory(self):
        sessions = weekdays()
        spec = split(sessions)
        spec["embargo_sessions"] = 1
        with self.assertRaises(ValueError):
            audit(dataset(sessions), sessions, spec, sessions[-1] + "T23:59:59+08:00")

    def test_incomplete_upstream_stays_diagnostic(self):
        sessions = weekdays()
        result = audit(dataset(sessions, False), sessions, split(sessions), sessions[-1] + "T23:59:59+08:00")
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertIn("UPSTREAM_FULL_POOL_RANK_DATASET_NOT_READY", result["issues"])

    def test_invalid_row_is_not_silently_kept(self):
        sessions = weekdays()
        data = dataset(sessions)
        data["rows"][0]["issues"] = ["bad"]
        result = audit(data, sessions, split(sessions), sessions[-1] + "T23:59:59+08:00")
        self.assertIn(data["rows"][0]["candidate_id"], result["purged"]["train"])
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")

    def test_blind_end_must_be_historical(self):
        sessions = weekdays()
        with self.assertRaises(ValueError):
            audit(dataset(sessions), sessions, split(sessions), sessions[400] + "T12:00:00+08:00")

    def test_dataset_target_must_match(self):
        sessions = weekdays()
        data = dataset(sessions)
        data["target_definition"] = "profit"
        with self.assertRaises(ValueError):
            audit(data, sessions, split(sessions), sessions[-1] + "T23:59:59+08:00")
