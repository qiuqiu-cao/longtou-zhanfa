import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import pytest

from collect_current_abnormal_inputs_v191 import collect


class Response:
    url = "https://example.test/history"
    content = json.dumps({"data": {"klines": [
        "2026-09-14,10,10,10,10,100,1000",
        "2026-09-15,10,10.5,10.5,10,200,2100",
        "2026-09-16,10.5,11.55,11.55,10.5,300,3465",
    ]}}).encode()

    def raise_for_status(self):
        return None

    def json(self):
        return json.loads(self.content)


def _files(root: Path):
    pool = root / "pool.csv"
    manifest = root / "pool.json"
    pd.DataFrame([{
        "signal_date": "2026-09-16", "code": "600001", "name": "样本",
        "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True,
    }]).to_csv(pool, index=False)
    manifest.write_text(json.dumps({
        "signal_date": "2026-09-16", "mainboard_closing_limit_rows": 1,
        "point_in_time_security_scope_complete": False,
    }), encoding="utf-8")
    return pool, manifest


def test_collect_archives_every_bound_candidate_with_provenance():
    with TemporaryDirectory() as temp, patch("requests.get", return_value=Response()) as get:
        root = Path(temp)
        pool, manifest = _files(root)
        output = root / "out"
        report = collect(pool, manifest, output, "2026-09-01", "2026-09-16", 1)
        frame = pd.read_csv(output / "candidate_daily_history.csv", dtype={"code": str})
        assert report["candidate_count"] == report["provider_observed_count"] == 1
        assert report["official_exchange_reconciled"] is False
        assert frame["code"].iloc[0] == "600001"
        assert frame["price_basis"].eq("provider_unadjusted").all()
        assert "timeout" in get.call_args.kwargs
        with pytest.raises(FileExistsError):
            collect(pool, manifest, output, "2026-09-01", "2026-09-16", 1)


def test_collect_rejects_pool_count_mismatch():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        pool, manifest = _files(root)
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["mainboard_closing_limit_rows"] = 2
        manifest.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="manifest-bound"):
            collect(pool, manifest, root / "out", "2026-09-01", "2026-09-16", 1)
