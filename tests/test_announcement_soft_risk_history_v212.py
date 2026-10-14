import json
from pathlib import Path

import pandas as pd

from announcement_soft_risk_history_v212 import build


def test_zero_requires_exact_successful_query_scope(tmp_path: Path):
    pool = tmp_path / "results" / "current_full_pool_20260105_v1"
    pool.mkdir(parents=True)
    pd.DataFrame([
        {"signal_date": "2026-01-05", "code": "1", "pool_assignment": "CLOSING_LIMIT_UP"},
        {"signal_date": "2026-01-05", "code": "2", "pool_assignment": "CLOSING_LIMIT_UP"},
    ]).to_csv(pool / "all_closing_and_failed_candidates.csv", index=False)
    context = tmp_path / "results" / "context"
    context.mkdir(parents=True)
    notice = context / "2026-01-05_announcements.csv"
    pd.DataFrame([{"code": "1", "risk_type": "监管背景_普通异常波动", "risk_hard": 0}]).to_csv(notice, index=False)
    manifest = {"signal_date": "2026-01-05", "collected_at": "2026-01-05T20:00:00",
                "candidate_count": 2,
                "announcements": {"candidate_codes": 2, "failed_codes": 1,
                                  "errors": {"000002": "failed"}, "rows": 1},
                "files": {"announcements": {"path": str(notice)}}}
    (context / "2026-01-05_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    candidates = pd.DataFrame([{"signal_date": "2026-01-05", "code": "1", "name": "A"},
                               {"signal_date": "2026-01-05", "code": "2", "name": "B"}])
    rows, report, _ = build(tmp_path, candidates)
    values = rows.set_index("code")
    assert values.loc["000001", "announcement_soft_risk"] == .3
    assert pd.isna(values.loc["000002", "announcement_soft_risk"])
    assert report["verified_soft_risk_rows"] == 1
