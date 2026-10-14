from severe_abnormal_label_queue_v208 import build


def test_unconfirmed_scan_is_not_negative_or_probability():
    library = {"events": [
        {"code": "1", "event_date": "2026-01-05", "target_pct": 100, "scanned": True,
         "official_confirmed": False, "training_label_ready": False, "evidence_available_at": None},
        {"code": "2", "event_date": "2026-01-06", "target_pct": 100, "scanned": True,
         "official_confirmed": True, "training_label_ready": False,
         "evidence_available_at": "2026-01-07T18:00:00+08:00"},
    ]}
    queue, report = build(library, {"failed_gates": ["labels"],
                                     "calibrated_hit_probability_allowed": False})
    assert report["training_ready_negative_events"] == 0
    assert report["calibrated_probability_emitted"] is False
    assert queue.set_index("code").loc["000001", "label_status"] == "UNVERIFIED_THRESHOLD_HYPOTHESIS"
