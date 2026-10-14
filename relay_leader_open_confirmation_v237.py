"""Next-session confirmation for the frozen V236 relay challenger.

The module never replaces the frozen close challenger.  It distinguishes a
healthy hand-off from a retreat contagion when the height anchor weakens, and
keeps verified business-theme synchrony separate from a verified symbolic
name-roster.  It uses only timestamped Level-1/minute proxies and never claims
Level-2 queue, cancel-sequence or actual-fill evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


VERSION = "v237-relay-leader-open-confirmation-1"
TZ = ZoneInfo("Asia/Shanghai")
REAL_THEME_LANES = {"VERIFIED_REAL_THEME"}
SYMBOLIC_LANES = {"VERIFIED_SYMBOLIC_ROSTER"}
MINUTES = [f"09:{minute:02d}" for minute in range(31, 37)]


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


def _role(snapshot: dict, key: str) -> dict | None:
    found = [item.get("candidate") for item in snapshot.get("role_candidates", [])
             if item.get("role") == key]
    return found[0] if len(found) == 1 else None


def _bar_features(rows: list[dict], code: str, decision_date: str) -> dict:
    selected = [row for row in rows if str(row.get("code", "")).zfill(6) == code]
    by_minute = {str(row.get("minute")): row for row in selected}
    missing = [minute for minute in MINUTES if minute not in by_minute]
    if missing:
        raise ValueError(f"{code} missing causal minute bars: {','.join(missing)}")
    cumulative_amount = cumulative_volume = 0.0
    vwaps, below, failed_seals = [], [], 0
    reseal_minute = None
    previous_close = None
    for minute in MINUTES:
        row = by_minute[minute]
        if _dt(row["available_at"]) > _dt(f"{decision_date}T{minute}:59+08:00"):
            raise ValueError(f"{code} {minute} bar has future availability")
        if row.get("volume_unit") != "share" or row.get("amount_unit") != "CNY":
            raise ValueError("minute bars require volume_unit=share and amount_unit=CNY")
        for field in ("open", "high", "low", "close", "volume", "amount", "previous_close"):
            _f(row, field)
        previous_close = _f(row, "previous_close")
        cumulative_amount += _f(row, "amount")
        cumulative_volume += _f(row, "volume")
        vwap = cumulative_amount / cumulative_volume if cumulative_volume else float("nan")
        vwaps.append(vwap)
        close = _f(row, "close")
        below.append(close < vwap)
        limit_price = (None if row.get("limit_price") in (None, "")
                       else float(row["limit_price"]))
        if limit_price is not None and _f(row, "high") >= limit_price and close < limit_price:
            failed_seals += 1
        if limit_price is not None and close >= limit_price:
            reseal_minute = minute
    first5 = [by_minute[minute] for minute in MINUTES[:5]]
    close_935 = _f(first5[-1], "close")
    vwap_935 = vwaps[4]
    causal = by_minute["09:36"]
    causal_price = (_f(causal, "amount") / _f(causal, "volume")
                    if _f(causal, "volume") > 0 else _f(causal, "open"))
    return {
        "vwap_slope_positive": vwaps[4] > vwaps[0],
        "close_vs_vwap": close_935 / vwap_935 - 1 if vwap_935 else float("nan"),
        "return_at_0935_pct": (close_935 / previous_close - 1) * 100,
        "below_vwap_expanding": bool(
            below[4] and below[3] and _f(first5[4], "volume") > _f(first5[3], "volume")),
        "failed_seal_episodes": failed_seals,
        "reseal_minute": reseal_minute,
        "causal_price_0936": causal_price,
    }


def _sync_lane(sync: dict) -> dict:
    business_lane = str(sync.get("business_evidence_lane", "UNAVAILABLE"))
    symbolic_lane = str(sync.get("symbolic_evidence_lane", "UNAVAILABLE"))
    business = {
        "lane": business_lane,
        "sync": float(sync.get("business_sync_ratio") or 0),
        "coverage": float(sync.get("business_coverage_ratio") or 0),
        "peers": int(float(sync.get("business_peer_count") or 0)),
    }
    symbolic = {
        "lane": symbolic_lane,
        "sync": float(sync.get("symbolic_sync_ratio") or 0),
        "coverage": float(sync.get("symbolic_coverage_ratio") or 0),
        "peers": int(float(sync.get("symbolic_peer_count") or 0)),
    }
    real_ok = (business_lane in REAL_THEME_LANES and business["coverage"] >= .7
               and business["peers"] >= 2)
    symbolic_ok = (symbolic_lane in SYMBOLIC_LANES and symbolic["coverage"] >= .7
                   and symbolic["peers"] >= 2
                   and bool(str(sync.get("symbolic_evidence_id", "")).strip()))
    if real_ok:
        return {"kind": "VERIFIED_REAL_THEME", **business}
    if symbolic_ok:
        return {"kind": "VERIFIED_SYMBOLIC_ROSTER_NOT_REAL_THEME", **symbolic}
    return {"kind": "UNVERIFIED_OR_INCOMPLETE_LADDER", "lane": "UNAVAILABLE",
            "sync": max(business["sync"], symbolic["sync"]),
            "coverage": max(business["coverage"], symbolic["coverage"]),
            "peers": max(business["peers"], symbolic["peers"])}


def evaluate(snapshot: dict, auction: dict | None, bars: list[dict] | None,
             ladder_sync: dict | None, decision_date: str) -> dict:
    anchor = _role(snapshot, "height_emotion_anchor")
    candidate = _role(snapshot, "close_relay_challenger")
    base = {
        "version": VERSION,
        "signal_date": snapshot.get("signal_date"),
        "decision_date": decision_date,
        "height_emotion_anchor": anchor,
        "frozen_close_relay_challenger": candidate,
        "replacement_after_failure": False,
        "entry_authorized": False,
        "level2_queue_used": False,
        "decision": "DATA_UNAVAILABLE",
        "reasons": [],
    }
    if not candidate:
        base["decision"] = "NO_FROZEN_CLOSE_RELAY_CHALLENGER"
        return base
    if not anchor or not auction or not bars or not ladder_sync:
        base["reasons"].append("竞价、候选/高标分钟线或梯队同步证据缺失")
        return base
    code, anchor_code = str(candidate["code"]).zfill(6), str(anchor["code"]).zfill(6)
    if str(auction.get("code", "")).zfill(6) != code:
        raise ValueError("auction code differs from frozen relay challenger")
    if str(ladder_sync.get("code", "")).zfill(6) != code:
        raise ValueError("ladder-sync code differs from frozen relay challenger")
    if auction.get("trade_date") != decision_date or ladder_sync.get("trade_date") != decision_date:
        raise ValueError("next-session evidence date mismatch")
    if _dt(auction["available_at"]) > _dt(f"{decision_date}T09:25:00+08:00"):
        raise ValueError("auction evidence was unavailable at 09:25")
    if _dt(ladder_sync["available_at"]) > _dt(f"{decision_date}T09:35:59+08:00"):
        raise ValueError("ladder sync was unavailable at 09:35")

    gap = _f(auction, "auction_gap_pct")
    amount_ratio = _f(auction, "auction_amount_ratio_pct")
    decay = _f(auction, "match_amount_decay_proxy")
    if not 0 <= decay <= 1:
        raise ValueError("match_amount_decay_proxy must be within [0,1]")
    base["auction_0925"] = {
        "gap_pct": gap, "amount_ratio_pct": amount_ratio,
        "match_amount_decay_proxy": decay,
        "is_level2_cancel_ratio": False,
        "source_status": auction.get("source_status", "UNSPECIFIED"),
    }
    if gap < -3 or (gap > 5 and (decay > .45 or amount_ratio < 2)):
        base["decision"] = "CANCEL_AT_0925"
        base["reasons"].append("低开超过弱转强边界，或高开时匹配额衰减/金额承接不足")
        return base

    candidate_open = _bar_features(bars, code, decision_date)
    anchor_open = _bar_features(bars, anchor_code, decision_date)
    lane = _sync_lane(ladder_sync)
    base["confirmation_0935"] = {
        "candidate": candidate_open, "height_anchor": anchor_open,
        "ladder": lane,
    }
    anchor_severe = (
        anchor_open["return_at_0935_pct"] <= -7
        or (anchor_open["return_at_0935_pct"] <= -5
            and anchor_open["below_vwap_expanding"])
        or anchor_open["failed_seal_episodes"] >= 2
    )
    min_sync = .8 if gap > 7 else (.7 if gap > 5 or gap <= 0 else .6)
    candidate_strong = (
        amount_ratio >= 3
        and candidate_open["vwap_slope_positive"]
        and candidate_open["close_vs_vwap"] >= 0
        and not candidate_open["below_vwap_expanding"]
        and candidate_open["failed_seal_episodes"] < 2
        and lane["coverage"] >= .7 and lane["peers"] >= 2 and lane["sync"] >= min_sync
    )
    if gap > 7:
        candidate_strong = (
            candidate_strong and decay <= .2
            and candidate_open["reseal_minute"] in MINUTES[:5]
        )
    if anchor_severe and not (candidate_strong and lane["sync"] >= .8):
        base["decision"] = "CANCEL_RETREAT_CONTAGION"
        base["reasons"].append("高标严重负反馈，挑战者和梯队不足以证明良性切换")
        return base
    if not candidate_strong:
        base["decision"] = "WATCH_NO_CONFIRMATION"
        base["reasons"].append("候选承接、VWAP、封板路径或梯队同步未通过")
        return base

    base["first_causal_price_opportunity"] = {
        "minute": "09:36", "price_proxy": candidate_open["causal_price_0936"],
        "meaning": "研究用机会价格，不代表实际成交",
    }
    if lane["kind"] == "UNVERIFIED_OR_INCOMPLETE_LADDER":
        base["decision"] = "PAPER_RESEARCH_CONFIRM_UNVERIFIED_LADDER"
        base["reasons"].append("盘面条件通过，但题材/符号梯队证据未完整核验")
    elif anchor_severe:
        base["decision"] = "PAPER_CORE_SWITCH_AFTER_0935"
        base["reasons"].append("高标负反馈下，挑战者与梯队同时强化，仅形成纸面核心切换")
    elif lane["kind"] == "VERIFIED_SYMBOLIC_ROSTER_NOT_REAL_THEME":
        base["decision"] = "PAPER_CONFIRM_SYMBOLIC_LADDER_ONLY"
        base["reasons"].append("符号梯队同步已核验，但不是真实产业题材")
    else:
        base["decision"] = "PAPER_CONFIRM_AFTER_0935"
        base["reasons"].append("候选承接、真实题材梯队与高标反馈通过纸面确认")
    return base


def run(snapshot_path: str, decision_date: str, output: str,
        auction_path: str | None = None, bars_path: str | None = None,
        ladder_sync_path: str | None = None) -> dict:
    source_paths = [Path(snapshot_path)]
    snapshot = json.loads(source_paths[0].read_text(encoding="utf-8-sig"))
    if snapshot.get("version") != "v236-relay-leader-role-forward-1":
        raise ValueError("V236 snapshot version mismatch")
    values = []
    for item in (auction_path, bars_path, ladder_sync_path):
        if item:
            path = Path(item); source_paths.append(path)
            values.append(json.loads(path.read_text(encoding="utf-8-sig")))
        else:
            values.append(None)
    report = evaluate(snapshot, values[0], values[1], values[2], decision_date)
    report["source_hashes"] = {str(path.resolve()): _sha(path) for path in source_paths}
    target = Path(output)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V237 report")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                      encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--decision-date", required=True)
    ap.add_argument("--auction")
    ap.add_argument("--bars")
    ap.add_argument("--ladder-sync")
    ap.add_argument("--output", required=True)
    ns = ap.parse_args()
    print(json.dumps(run(ns.snapshot, ns.decision_date, ns.output, ns.auction,
                         ns.bars, ns.ladder_sync), ensure_ascii=False, indent=2))
