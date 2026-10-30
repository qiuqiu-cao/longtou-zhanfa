"""Replay loss-tail claims descriptively; never select a strategy using future losses."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TRADES = ROOT / "research/first_board_leader_20261020T114600Z/model_score_SCENARIO_TRADES.csv"
CALENDAR = ROOT / "data/v279/exchange_calendar_audit_20261002_v1/calendar.json"
BAR_DIR = ROOT / "data/v16/point_in_time_rebuild_cutoff_20260831_v2"
BAR_FILES = [BAR_DIR / "point_in_time_current_daily_bars.csv", BAR_DIR / "verified_recovered_daily_bars.csv"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    result_path = OUT / "LOSS_TAIL_CLAIM_REVIEW.json"
    if result_path.exists():
        raise FileExistsError("Keep prior diagnostic evidence")
    sources = [TRADES, CALENDAR, *BAR_FILES, Path(__file__)]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    all_trades = pd.read_csv(TRADES, dtype={"code": str})
    closed = all_trades[all_trades.status.eq("SCENARIO_CLOSED")].copy()
    assert not closed.duplicated(["signal_date", "code"]).any()
    sessions = json.loads(CALENDAR.read_text("utf-8"))["sessions"]
    pos = {d: i for i, d in enumerate(sessions)}
    closed["b1_date"] = closed.entry_date.map(lambda d: sessions[pos[d]+1] if pos[d]+1 < len(sessions) else None)
    desired = pd.concat([
        closed[["entry_date", "code"]].rename(columns={"entry_date": "date"}),
        closed[["b1_date", "code"]].rename(columns={"b1_date": "date"}),
    ], ignore_index=True).dropna().drop_duplicates()
    cols = ["date", "code", "name", "open", "high", "low", "close", "pre_close", "volume"]
    parts = []
    for file in BAR_FILES:
        for chunk in pd.read_csv(file, usecols=cols, dtype={"code": str}, chunksize=250000):
            chunk["code"] = chunk.code.str.zfill(6)
            matched = chunk.merge(desired, on=["date", "code"], how="inner", validate="many_to_one")
            if not matched.empty:
                matched["bar_source"] = str(file.relative_to(ROOT))
                parts.append(matched)
    bars = pd.concat(parts, ignore_index=True)
    prices = ["open", "high", "low", "close", "pre_close", "volume"]
    conflicts = bars.groupby(["date", "code"])[prices].nunique(dropna=False).gt(1).any(axis=1)
    if conflicts.any():
        raise ValueError("Conflicting daily sources; do not choose convenient prices")
    bars = bars.drop_duplicates(["date", "code"])
    closed = closed.merge(bars.rename(columns={c: "b_"+c for c in bars if c != "code"}),
                          left_on=["entry_date", "code"], right_on=["b_date", "code"], how="left", validate="one_to_one")
    closed = closed.merge(bars.rename(columns={c: "b1_"+c for c in bars if c != "code"}),
                          on=["b1_date", "code"], how="left", validate="one_to_one")
    assert len(closed) == 214
    closed["entry_bar_known"] = closed[["b_open", "b_high", "b_low", "b_close", "b_pre_close"]].gt(0).all(axis=1)
    closed["b1_bar_known"] = closed[["b1_open", "b1_high", "b1_low", "b1_close", "b1_pre_close"]].gt(0).all(axis=1)
    closed["entry_gap_return"] = closed.b_open / closed.b_pre_close - 1
    closed["entry_close_vs_open_return"] = closed.b_close / closed.b_open - 1
    closed["b1_open_vs_entry_return"] = closed.b1_open / closed.entry_price - 1
    cap_cents = np.floor(closed.b_pre_close * 110 + .5 + 1e-8)
    closed["entry_day_touched_cap_PROXY"] = (np.rint(closed.b_high*100) >= cap_cents).where(closed.entry_bar_known)
    closed["entry_day_closed_at_cap_PROXY"] = (np.rint(closed.b_close*100) >= cap_cents).where(closed.entry_bar_known)
    closed["ex_post_loss_gt10pct_LABEL_ONLY"] = closed.net_return.lt(-.10)
    closed["auction_evidence_status"] = "NOT_AVAILABLE_IN_THIS_DAILY_BAR_AUDIT"
    closed["board_fill_status"] = "UNKNOWN_DAILY_HIGH_IS_NOT_QUEUE_OR_FILL_EVIDENCE"
    closed["monotonic_intraday_path_status"] = "UNKNOWN_DAILY_OHLC_CANNOT_PROVE_PATH_ORDER"
    tail = closed[closed.ex_post_loss_gt10pct_LABEL_ONLY]
    others = closed[~closed.ex_post_loss_gt10pct_LABEL_ONLY]

    def summary(rows):
        return {
            "rows": len(rows), "net_pnl_cny": float(rows.net_pnl.sum()),
            "mean_net_return": float(rows.net_return.mean()),
            "entry_bar_known": int(rows.entry_bar_known.sum()),
            "b1_bar_known": int(rows.b1_bar_known.sum()),
            "mean_entry_gap": float(rows.entry_gap_return.mean()),
            "positive_entry_gap": int(rows.entry_gap_return.gt(0).sum()),
            "nonpositive_entry_gap": int(rows.entry_gap_return.le(0).sum()),
            "mean_entry_close_vs_open": float(rows.entry_close_vs_open_return.mean()),
            "entry_close_ge_open": int(rows.entry_close_vs_open_return.ge(0).sum()),
            "entry_day_touched_cap_proxy": int(rows.entry_day_touched_cap_PROXY.eq(True).sum()),
            "entry_day_closed_at_cap_proxy": int(rows.entry_day_closed_at_cap_PROXY.eq(True).sum()),
            "b1_open_already_below_entry_by_more_than_3pct": int(rows.b1_open_vs_entry_return.lt(-.03).sum()),
            "blocked_exit_trade_count": int(rows.blocked_days.gt(0).sum()),
        }

    receipt = {
        "formed_at": datetime.now(timezone.utc).isoformat(),
        "status": "RETROSPECTIVE_LOSS_DIAGNOSIS_NOT_A_VALIDATED_FILTER",
        "source_hashes": hashes,
        "original_scenario_status_counts": all_trades.status.value_counts().to_dict(),
        "all_closed": summary(closed), "ex_post_loss_gt10pct": summary(tail),
        "all_other_closed": summary(others),
        "oracle_removed_loss_sum_NOT_A_REPLAYED_STRATEGY_cny": float(others.net_pnl.sum()),
        "scope": "Original unscreened height Top1 daily-open/B+2-close scenario; not the screened or comprehensive strategy",
        "limitations": [
            "33 trades were identified from future realized losses, not an ex-ante observable filter.",
            "Removing their original PnL does not replay the changed cash, positions, fees or false rejections.",
            "Daily OHLC cannot certify a monotonic path, 09:25 queue, intraday reseal or a 09:45 execution.",
            "Observed opening gaps show why a stop trigger cannot guarantee a maximum loss.",
            "High reaching the upper limit is a price proxy, not proof of order fill.",
        ],
        "new_fits": 0, "new_strategy_return": None, "auction_rejection_recall": None,
        "current_objective": "Improve first-board S-close overnight selection; relay structure supplies context rather than replacing that objective",
        "production_changed": False,
    }
    assert all(digest(ROOT / p) == expected for p, expected in hashes.items())
    closed.to_csv(OUT / "ALL214_CLOSED_SCENARIOS_WITH_DAILY_DIAGNOSTICS.csv", index=False, encoding="utf-8-sig")
    tail.to_csv(OUT / "EX_POST33_LOSS_TAIL_NOT_AN_EX_ANTE_FILTER.csv", index=False, encoding="utf-8-sig")
    result_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ["all_closed", "ex_post_loss_gt10pct", "all_other_closed"]}, indent=2))


if __name__ == "__main__":
    main()
