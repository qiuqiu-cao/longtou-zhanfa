"""Bounded recent-minute feasibility probe; never backfill a frozen forecast."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import requests
import public_minute_capture_v1 as minute


def run(output):
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    responses = {}
    def get(url, **kwargs):
        code = kwargs["params"]["symbol"]
        started = datetime.now().astimezone().isoformat()
        response = requests.get(url, **kwargs)
        path = target / (code + ".response")
        path.write_bytes(response.content)
        responses[code] = dict(url=response.url, http_status=response.status_code,
            requested_at=started, received_at=datetime.now().astimezone().isoformat(),
            raw_path=str(path.resolve()), sha256=hashlib.sha256(response.content).hexdigest())
        return response
    result = minute.fetch_minute_bars(["000678", "600825"], get=get, workers=2,
                                     timeout=8, retries=0, datalen=240)
    dates = []
    if not result.bars.empty:
        result.bars.to_csv(target / "bars_UNRECONCILED.csv", index=False)
        dates = sorted(result.bars.minute.str[:10].unique().tolist())
    manifest = dict(status="MINUTE_SOURCE_FEASIBILITY_ONLY", recorded_at=datetime.now().astimezone().isoformat(),
        source="Sina public minute endpoint already used by project", requested_codes=["000678", "600825"],
        requested_bar_limit=240, errors=result.errors, parsed_rows=len(result.bars),
        returned_dates=dates, responses=responses,
        old_forecast_rewrite_allowed=False, trade_authorized=False, accuracy_claim_allowed=False,
        raw_minute_prices_units_and_daily_reconciliation_required=True,
        source_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(__file__),Path(minute.__file__))})
    (target / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    print(json.dumps(run(parser.parse_args().output),ensure_ascii=True))
