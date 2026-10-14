"""V221 first-board next-session auction and 09:35 confirmation.

This module only evaluates a candidate already frozen by V220.  It never
substitutes another stock after cancellation and never turns research data
into an authorised order.  A decision made with 09:35 information may only
use the 09:36 bar as its first causal price opportunity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


VERSION = "v221-first-board-open-confirmation-1"
TZ = ZoneInfo("Asia/Shanghai")
VERIFIED_THEME_LANES = {"VERIFIED_REAL_THEME"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dt(value: str) -> datetime:
    stamp = datetime.fromisoformat(str(value))
    if stamp.tzinfo is None:
        raise ValueError("timestamps must include timezone")
    return stamp.astimezone(TZ)


def _f(obj: dict, key: str) -> float:
    value = obj.get(key)
    if value in (None, ""):
        raise ValueError(f"missing numeric field: {key}")
    return float(value)


def evaluate(snapshot: dict, auction: dict | None, bars: list[dict] | None,
             theme_sync: dict | None, decision_date: str) -> dict:
    candidate = snapshot.get("candidate")
    base = {
        "version": VERSION,
        "signal_date": snapshot.get("signal_date"),
        "decision_date": decision_date,
        "candidate": candidate,
        "replacement_after_cancel": False,
        "entry_authorized": False,
        "level2_used": False,
        "decision": "DATA_UNAVAILABLE",
        "reasons": [],
    }
    if not candidate:
        base["decision"] = "NO_FROZEN_CANDIDATE"
        return base
    code = str(candidate["code"]).zfill(6)
    if not auction or not bars or not theme_sync:
        base["reasons"].append("竞价、分钟线或题材同步证据缺失")
        return base
    if str(auction.get("code", "")).zfill(6) != code:
        raise ValueError("auction code differs from frozen candidate")
    if str(theme_sync.get("code", "")).zfill(6) != code:
        raise ValueError("theme-sync code differs from frozen candidate")
    if auction.get("trade_date") != decision_date or theme_sync.get("trade_date") != decision_date:
        raise ValueError("next-session evidence date mismatch")

    cutoff_0925 = _dt(f"{decision_date}T09:25:00+08:00")
    if _dt(auction["available_at"]) > cutoff_0925:
        raise ValueError("auction evidence was unavailable at 09:25")
    gap = _f(auction, "auction_gap_pct")
    amount_ratio = _f(auction, "auction_amount_ratio_pct")
    decay = _f(auction, "match_amount_decay_proxy")
    if not 0 <= decay <= 1:
        raise ValueError("match_amount_decay_proxy must be within [0,1]")
    base["auction_0925"] = {
        "gap_pct": gap,
        "amount_ratio_pct": amount_ratio,
        "match_amount_decay_proxy": decay,
        "source_status": auction.get("source_status", "UNSPECIFIED"),
        "is_level2_cancel_ratio": False,
    }
    auction_cancel = (
        (gap > 5 and decay > .45)
        or (gap > 5 and amount_ratio < 2)
        or gap < -3
    )
    if auction_cancel:
        base["decision"] = "CANCEL_AT_0925"
        base["reasons"].append("竞价过热且承接不足、匹配额衰减过大，或低开超过弱转强边界")
        return base

    required = [f"09:{minute:02d}" for minute in range(31, 37)]
    by_minute = {str(row.get("minute")): row for row in bars}
    missing = [minute for minute in required if minute not in by_minute]
    if missing:
        base["reasons"].append(f"缺少因果分钟线: {','.join(missing)}")
        return base
    for minute in required:
        row = by_minute[minute]
        if str(row.get("code", "")).zfill(6) != code:
            raise ValueError("minute-bar code differs from frozen candidate")
        if _dt(row["available_at"]) > _dt(f"{decision_date}T{minute}:59+08:00"):
            raise ValueError(f"{minute} minute bar has a future availability timestamp")
        if row.get("volume_unit") != "share" or row.get("amount_unit") != "CNY":
            raise ValueError("minute bars require volume_unit=share and amount_unit=CNY")
        for field in ("open", "high", "low", "close", "volume", "amount"):
            _f(row, field)

    first5 = [by_minute[m] for m in required[:5]]
    cumulative_amount = 0.0
    cumulative_volume = 0.0
    vwaps = []
    below = []
    failed_seals = 0
    reseal_minute = None
    for row in first5:
        cumulative_amount += _f(row, "amount")
        cumulative_volume += _f(row, "volume")
        vwap = cumulative_amount / cumulative_volume if cumulative_volume > 0 else float("nan")
        vwaps.append(vwap)
        close = _f(row, "close")
        below.append(close < vwap)
        limit_price = float(row["limit_price"]) if row.get("limit_price") not in (None, "") else None
        if limit_price is not None and _f(row, "high") >= limit_price and close < limit_price:
            failed_seals += 1
        if limit_price is not None and close >= limit_price:
            reseal_minute = row["minute"]
    close_935 = _f(first5[-1], "close")
    close_vs_vwap = close_935 / vwaps[-1] - 1 if vwaps[-1] else float("nan")
    slope_positive = vwaps[-1] > vwaps[0]
    below_expanding = bool(below[-1] and below[-2] and _f(first5[-1], "volume") > _f(first5[-2], "volume"))

    sync = _f(theme_sync, "sync_ratio")
    coverage = _f(theme_sync, "coverage_ratio")
    peers = int(_f(theme_sync, "peer_count"))
    lane = str(theme_sync.get("evidence_lane", "UNAVAILABLE"))
    if _dt(theme_sync["available_at"]) > _dt(f"{decision_date}T09:35:59+08:00"):
        raise ValueError("theme sync was unavailable at 09:35")
    base["confirmation_0935"] = {
        "vwap_slope_positive": slope_positive,
        "close_vs_vwap": close_vs_vwap,
        "below_vwap_expanding": below_expanding,
        "failed_seal_episodes": failed_seals,
        "reseal_minute": reseal_minute,
        "theme_sync_ratio": sync,
        "theme_coverage_ratio": coverage,
        "theme_peer_count": peers,
        "theme_evidence_lane": lane,
    }
    if failed_seals >= 2 or below_expanding:
        base["decision"] = "CANCEL_AT_0935"
        base["reasons"].append("两次炸板或连续跌破VWAP且成交扩张")
        return base
    min_sync = .8 if gap > 7 else (.7 if gap > 5 or gap <= 0 else .6)
    confirmed = (
        amount_ratio >= 3
        and slope_positive
        and close_vs_vwap >= 0
        and peers >= 2
        and coverage >= .7
        and sync >= min_sync
    )
    if gap > 7:
        confirmed = confirmed and decay <= .2 and reseal_minute in {"09:31", "09:32", "09:33", "09:34", "09:35"}
    if not confirmed:
        base["decision"] = "WATCH_NO_CONFIRMATION"
        base["reasons"].append("09:35承接、题材同步或覆盖度未达到预设阈值")
        return base

    bar936 = by_minute["09:36"]
    causal_price = _f(bar936, "amount") / _f(bar936, "volume") if _f(bar936, "volume") > 0 else _f(bar936, "open")
    base["first_causal_price_opportunity"] = {
        "minute": "09:36", "price_proxy": causal_price,
        "meaning": "研究用机会价格，不代表实际成交",
    }
    if lane not in VERIFIED_THEME_LANES:
        base["decision"] = "PAPER_RESEARCH_CONFIRM_UNVERIFIED_THEME"
        base["reasons"].append("盘面确认通过，但题材催化尚未达到VERIFIED_REAL_THEME")
    else:
        base["decision"] = "PAPER_CONFIRM_AFTER_0935"
        base["reasons"].append("仅形成纸面确认；实盘授权仍为False")
    return base


def run(snapshot_path: str, auction_path: str | None, bars_path: str | None,
        theme_sync_path: str | None, decision_date: str, output: str) -> dict:
    sources = [Path(snapshot_path)]
    snapshot = json.loads(sources[0].read_text(encoding="utf-8-sig"))
    values = []
    for path in (auction_path, bars_path, theme_sync_path):
        if path:
            p = Path(path); sources.append(p)
            values.append(json.loads(p.read_text(encoding="utf-8-sig")))
        else:
            values.append(None)
    report = evaluate(snapshot, values[0], values[1], values[2], decision_date)
    report["source_hashes"] = {str(p.resolve()): _sha(p) for p in sources}
    target = Path(output)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V221 report")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--auction")
    ap.add_argument("--bars")
    ap.add_argument("--theme-sync")
    ap.add_argument("--decision-date", required=True)
    ap.add_argument("--output", required=True)
    ns = ap.parse_args()
    print(json.dumps(run(ns.snapshot, ns.auction, ns.bars, ns.theme_sync,
                         ns.decision_date, ns.output), ensure_ascii=False, indent=2))
