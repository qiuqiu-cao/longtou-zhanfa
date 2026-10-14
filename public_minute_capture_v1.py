#!/usr/bin/env python3
"""Capture and measure public 1-minute bars for all prior limit-up candidates.

The measurements make open-support language reproducible without screenshots.
They are minute-bar proxies, not tick/Level-2 facts, and never authorise an
entry by themselves.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import public_live_capture_v1 as live


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "v16" / "public_minute"
SHANGHAI = ZoneInfo("Asia/Shanghai")
VERSION = "v16-public-minute-capture-1"
MINUTE_POINTS = ("09:35", "10:00")
SINA_MINUTE_URL = "https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_data=/CN_MarketDataService.getKLineData"
SINA_HEADERS = live.SINA_HEADERS


def safe_float(value: Any) -> float:
    return live.safe_float(value)


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    match = re.search(r"var\s+_data=\((\[.*?\])\);", text, flags=re.S)
    if not match:
        return []
    payload = json.loads(match.group(1))
    return payload if isinstance(payload, list) else []


def parse_minute_response(text: str, code: str) -> pd.DataFrame:
    rows = []
    for item in _extract_json_array(text):
        stamp = pd.to_datetime(item.get("day"), errors="coerce")
        if pd.isna(stamp):
            continue
        rows.append({
            "code": live.code6(code), "minute": pd.Timestamp(stamp).strftime("%Y-%m-%d %H:%M:%S"),
            "open": safe_float(item.get("open")), "high": safe_float(item.get("high")),
            "low": safe_float(item.get("low")), "close": safe_float(item.get("close")),
            "volume_shares": safe_float(item.get("volume")), "amount": safe_float(item.get("amount")),
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(["code", "minute"], keep="last").sort_values("minute").reset_index(drop=True)
    return frame


@dataclass(frozen=True)
class MinuteFetchResult:
    bars: pd.DataFrame
    errors: dict[str, str]
    requested: int
    response_hashes: dict[str, str]


def _fetch_one(
    code: str, get: Callable[..., Any], timeout: float, retries: int, datalen: int,
) -> tuple[str, pd.DataFrame, str, str | None]:
    error: Exception | None = None
    params = {"symbol": live.sina_symbol(code), "scale": "1", "ma": "no", "datalen": str(datalen)}
    for attempt in range(retries + 1):
        try:
            response = get(SINA_MINUTE_URL, params=params, headers=SINA_HEADERS, timeout=timeout)
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()
            content = response.content if hasattr(response, "content") else str(response.text).encode("utf-8")
            text = content.decode("utf-8", errors="replace")
            return code, parse_minute_response(text, code), hashlib.sha256(content).hexdigest(), None
        except Exception as exc:
            error = exc
            if attempt < retries:
                time.sleep(0.25 * (attempt + 1))
    return code, pd.DataFrame(), "", str(error)[:300] if error else "unknown provider error"


def fetch_minute_bars(
    codes: Iterable[str], *, get: Callable[..., Any] = requests.get, workers: int = 6,
    timeout: float = 8.0, retries: int = 1, datalen: int = 240,
) -> MinuteFetchResult:
    normalized = sorted({live.code6(code) for code in codes if live.mainboard(live.code6(code))})
    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    hashes: dict[str, str] = {}
    with futures.ThreadPoolExecutor(max_workers=max(1, min(workers, 10))) as executor:
        jobs = {executor.submit(_fetch_one, code, get, timeout, retries, datalen): code for code in normalized}
        for job in futures.as_completed(jobs):
            code, frame, digest, error = job.result()
            if not frame.empty:
                frames.append(frame)
            if digest:
                hashes[code] = digest
            if error:
                errors[code] = error
    bars = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not bars.empty:
        bars = bars.sort_values(["code", "minute"]).reset_index(drop=True)
    return MinuteFetchResult(bars=bars, errors=errors, requested=len(normalized), response_hashes=hashes)


def minute_freshness(last_minute: Any, received_at: Any, trade_date: str) -> tuple[str, float | None]:
    stamp = pd.to_datetime(last_minute, errors="coerce")
    if pd.isna(stamp):
        return "UNAVAILABLE_NO_MINUTE_TIMESTAMP", None
    observed = live.local_timestamp(stamp)
    received = live.local_timestamp(received_at)
    age = (received - observed).total_seconds()
    if observed.date() != pd.Timestamp(trade_date).date():
        return "STALE_PREVIOUS_SESSION_MINUTES", age
    if age < -30:
        return "UNAVAILABLE_FUTURE_MINUTE_TIMESTAMP", age
    if age <= 90:
        return "VERIFIED_PUBLIC_MINUTE_BAR_FRESH", age
    if age <= 240:
        return "DELAYED_PUBLIC_MINUTE_BAR", age
    return "STALE_PUBLIC_MINUTE_BAR", age


def _ols_slope(values: pd.Series, denominator: float) -> float | None:
    valid = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(valid) < 3 or denominator <= 0:
        return None
    slope = float(np.polyfit(np.arange(len(valid), dtype=float), valid, 1)[0])
    return slope / denominator


def measure_candidate(
    bars: pd.DataFrame, previous_close: Any, received_at: Any, trade_date: str, asof: str,
) -> dict[str, Any]:
    previous = safe_float(previous_close)
    point = pd.Timestamp(f"{pd.Timestamp(trade_date).strftime('%Y-%m-%d')} {asof}:00")
    frame = bars.copy()
    frame["minute"] = pd.to_datetime(frame["minute"], errors="coerce")
    frame = frame[(frame["minute"].dt.strftime("%Y-%m-%d") == pd.Timestamp(trade_date).strftime("%Y-%m-%d"))
                  & (frame["minute"] <= point)].sort_values("minute")
    if frame.empty:
        return {
            "minute_data_status": "UNAVAILABLE_NO_BARS_BEFORE_ASOF", "minute_age_seconds": None,
            "minute_bar_count": 0, "direct_entry_authorized": False,
        }
    numeric = ["open", "high", "low", "close", "volume_shares", "amount"]
    for field in numeric:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame = frame.dropna(subset=["close", "volume_shares", "amount"])
    if frame.empty:
        return {
            "minute_data_status": "UNAVAILABLE_INVALID_BARS", "minute_age_seconds": None,
            "minute_bar_count": 0, "direct_entry_authorized": False,
        }
    status, age = minute_freshness(frame["minute"].iloc[-1], received_at, trade_date)
    cumulative_volume = frame["volume_shares"].clip(lower=0).cumsum()
    cumulative_amount = frame["amount"].clip(lower=0).cumsum()
    frame["session_vwap"] = cumulative_amount / cumulative_volume.replace(0, np.nan)
    current_vwap = safe_float(frame["session_vwap"].iloc[-1])
    current_close = safe_float(frame["close"].iloc[-1])
    limit_price = live.limit_price_10(previous)
    tolerance = .005
    frame["at_limit_close"] = frame["close"].ge(limit_price - tolerance) if not math.isnan(limit_price) else False
    frame["touch_limit"] = frame["high"].ge(limit_price - tolerance) if not math.isnan(limit_price) else False
    touch_start = frame["touch_limit"] & ~frame["touch_limit"].shift(fill_value=False)
    failed = frame["touch_limit"] & ~frame["at_limit_close"]
    below = frame["close"] < frame["session_vwap"]
    above = frame["close"] >= frame["session_vwap"]
    reclaim_positions = frame.index[above & below.shift(fill_value=False)].tolist()
    first_reclaim = frame.loc[reclaim_positions[0], "minute"] if reclaim_positions else None
    earlier_below_limit = (~frame["at_limit_close"]).cummax().shift(fill_value=False)
    reseal_rows = frame[frame["at_limit_close"] & earlier_below_limit]
    reseal_minute = None
    if not reseal_rows.empty:
        reseal_minute = max(0.0, (reseal_rows["minute"].iloc[0] - pd.Timestamp(f"{trade_date} 09:30:00")).total_seconds() / 60)

    pullback_ratio = None
    pullback_minute = None
    for index in range(1, len(frame)):
        row = frame.iloc[index]
        previous_row = frame.iloc[index - 1]
        if row["close"] < row["open"] or row["close"] < previous_row["close"]:
            reference = frame.iloc[max(0, index - 3):index]["volume_shares"].median()
            if pd.notna(reference) and reference > 0:
                pullback_ratio = float(row["volume_shares"] / reference)
            pullback_minute = row["minute"].isoformat()
            break
    last_two = frame.tail(2)["volume_shares"].mean() if len(frame) >= 2 else math.nan
    prior_two = frame.iloc[-4:-2]["volume_shares"].mean() if len(frame) >= 4 else math.nan
    below_expanding = bool(
        not math.isnan(current_vwap) and current_close < current_vwap
        and pd.notna(last_two) and pd.notna(prior_two) and prior_two > 0 and last_two > prior_two * 1.25
    )
    one_word = bool(len(frame) >= 2 and not math.isnan(limit_price)
                    and frame["low"].notna().all() and frame["low"].ge(limit_price - tolerance).all())
    return {
        "minute_data_status": status, "minute_age_seconds": age, "minute_bar_count": int(len(frame)),
        "last_minute": frame["minute"].iloc[-1].isoformat(), "last_close": current_close,
        "session_vwap_proxy": current_vwap,
        "close_to_session_vwap_proxy": current_close / current_vwap - 1 if current_vwap > 0 else None,
        "vwap_slope_5m_proxy": _ols_slope(frame.tail(6)["session_vwap"], previous),
        "vwap_slope_15m_proxy": _ols_slope(frame.tail(16)["session_vwap"], previous) if asof == "10:00" else None,
        "above_session_vwap_proxy": bool(not math.isnan(current_vwap) and current_close >= current_vwap),
        "vwap_reclaimed_proxy": bool(reclaim_positions),
        "first_vwap_reclaim_minute_proxy": first_reclaim.isoformat() if first_reclaim is not None else None,
        "first_pullback_minute_proxy": pullback_minute,
        "first_pullback_volume_ratio_proxy": pullback_ratio,
        "limit_price_10pct_inferred": limit_price,
        "limit_touch_episode_count_proxy": int(touch_start.sum()),
        "failed_seal_minute_count_proxy": int(failed.sum()),
        "failed_seal_twice_proxy": bool(failed.sum() >= 2),
        "reseal_minute_proxy": reseal_minute,
        "one_word_minute_proxy": one_word,
        "below_vwap_expanding_proxy": below_expanding,
        "evidence_level": "INFERRED_FROM_PUBLIC_1MIN_BARS",
        "direct_entry_authorized": False,
    }


def build_measurements(
    candidates: pd.DataFrame, bars: pd.DataFrame, received_at: Any, trade_date: str, asof: str,
) -> pd.DataFrame:
    rows = []
    grouped = {code: group for code, group in bars.groupby("code")} if not bars.empty else {}
    for candidate in candidates.to_dict("records"):
        code = live.code6(candidate["code"])
        measured = measure_candidate(
            grouped.get(code, pd.DataFrame()), candidate.get("signal_close"), received_at, trade_date, asof,
        )
        rows.append({
            "signal_date": candidate["signal_date"], "trade_date": trade_date, "asof": asof,
            "code": code, "name": candidate["name"], "boards": int(candidate["boards"]),
            "received_at": live.local_timestamp(received_at).isoformat(), **measured,
        })
    return pd.DataFrame(rows).sort_values(["boards", "code"], ascending=[False, True]).reset_index(drop=True)


def make_report(manifest: dict[str, Any], measurements: pd.DataFrame) -> str:
    lines = [
        f"# {manifest['trade_date']} {manifest['asof']} 公开1分钟承接代理", "",
        "- 来源：Sina公开1分钟K线接口。",
        f"- 候选覆盖：{manifest['candidate_with_bars']}/{manifest['candidate_rows']}。",
        f"- 新鲜分钟覆盖：{manifest['fresh_candidate_rows']}/{manifest['candidate_rows']}。",
        "- 指标属性：均为公开分钟OHLCV推断，不是逐笔、Level-2、真实封单或成交。",
        "- 执行权限：本层固定不授权ENTER。", "",
        "| 代码 | 名称 | 板位 | 状态 | 最新/均价 | 5分钟VWAP斜率 | 触板段 | 炸板分钟 | 回封分钟 | 跌破放量 |", "|---|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in measurements.itertuples(index=False):
        relation = "缺失" if pd.isna(getattr(row, "close_to_session_vwap_proxy", None)) else f"{row.close_to_session_vwap_proxy:.2%}"
        slope = "缺失" if pd.isna(getattr(row, "vwap_slope_5m_proxy", None)) else f"{row.vwap_slope_5m_proxy:.4%}"
        reseal = "-" if pd.isna(getattr(row, "reseal_minute_proxy", None)) else f"{row.reseal_minute_proxy:.0f}"
        lines.append(
            f"| {row.code} | {row.name} | {int(row.boards)} | {row.minute_data_status} | {relation} | {slope} | "
            f"{getattr(row, 'limit_touch_episode_count_proxy', 0)} | {getattr(row, 'failed_seal_minute_count_proxy', 0)} | {reseal} | "
            f"{getattr(row, 'below_vwap_expanding_proxy', False)} |"
        )
    lines += ["", "这些代理可用于记录和回测承接形态，但不能代替真实排队可买性。", ""]
    return "\n".join(lines)


def capture(
    candidate_path: Path, trade_date: str, asof: str, output: Path = DEFAULT_OUTPUT, *,
    now: Any | None = None, fetcher: Callable[..., MinuteFetchResult] = fetch_minute_bars,
) -> dict[str, Any]:
    if asof not in MINUTE_POINTS:
        raise ValueError(f"minute capture asof must be one of {MINUTE_POINTS}")
    requested_at = live.local_timestamp(now or datetime.now(SHANGHAI))
    candidates = live.load_candidates(candidate_path)
    fetched = fetcher(candidates["code"])
    received_at = requested_at if now is not None else live.local_timestamp(datetime.now(SHANGHAI))
    measurements = build_measurements(candidates, fetched.bars, received_at, trade_date, asof)
    fresh = int(measurements["minute_data_status"].eq("VERIFIED_PUBLIC_MINUTE_BAR_FRESH").sum())
    with_bars = int(measurements["minute_bar_count"].gt(0).sum())
    status = (
        "CONDITIONAL_PUBLIC_MINUTE_EVIDENCE_FRESH" if fresh == len(candidates) else
        "CASH_PUBLIC_MINUTE_DATA_INCOMPLETE_OR_STALE"
    )
    identity = {
        "version": VERSION, "trade_date": trade_date, "asof": asof,
        "received_at": received_at.isoformat(), "codes": candidates["code"].tolist(),
        "response_hashes": fetched.response_hashes,
    }
    capture_id = live.hash_payload(identity)[:20]
    target = output / pd.Timestamp(trade_date).strftime("%Y-%m-%d") / asof.replace(":", "") / capture_id
    if target.exists():
        raise FileExistsError("immutable public minute capture already exists")
    target.mkdir(parents=True, exist_ok=False)
    bars_path, measures_path = target / "minute_bars.csv", target / "measurements.csv"
    report_path, manifest_path = target / "report.md", target / "manifest.json"
    fetched.bars.to_csv(bars_path, index=False, encoding="utf-8-sig")
    measurements.to_csv(measures_path, index=False, encoding="utf-8-sig")
    manifest: dict[str, Any] = {
        "version": VERSION, "capture_id": capture_id, "status": status,
        "signal_date": str(candidates["signal_date"].iloc[0]),
        "trade_date": pd.Timestamp(trade_date).strftime("%Y-%m-%d"), "asof": asof,
        "requested_at": requested_at.isoformat(), "received_at": received_at.isoformat(),
        "source": "Sina public 1-minute K-line endpoint", "candidate_rows": int(len(candidates)),
        "candidate_with_bars": with_bars, "fresh_candidate_rows": fresh,
        "provider_failed_codes": fetched.errors, "provider_response_hashes": fetched.response_hashes,
        "evidence_level": "INFERRED_FROM_PUBLIC_1MIN_BARS", "direct_entry_authorized": False,
        "limitations": [
            "minute bars cannot reconstruct tick order, exact auction cancel sequence or queue priority",
            "touch/reseal/failed-seal fields are one-minute proxies and may miss intraminute events",
            "no actual order or fill is inferred",
        ],
    }
    report_path.write_text(make_report(manifest, measurements), encoding="utf-8")
    manifest["files"] = {
        "bars": {"path": str(bars_path.resolve()), "sha256": hashlib.sha256(bars_path.read_bytes()).hexdigest()},
        "measurements": {"path": str(measures_path.resolve()), "sha256": hashlib.sha256(measures_path.read_bytes()).hexdigest()},
        "report": {"path": str(report_path.resolve()), "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()},
    }
    manifest["manifest_hash"] = live.hash_payload({key: value for key, value in manifest.items() if key != "manifest_hash"})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**manifest, "capture_dir": str(target.resolve()), "manifest_path": str(manifest_path.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture public 1-minute open-support evidence for all prior limit-ups")
    parser.add_argument("--candidates", type=Path, help="auto-discovers latest standardized full pool when omitted")
    parser.add_argument("--trade-date", default=datetime.now(SHANGHAI).strftime("%Y-%m-%d"))
    parser.add_argument("--asof", choices=MINUTE_POINTS, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    candidates = args.candidates or live.discover_candidates()
    result = capture(candidates, args.trade_date, args.asof, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
