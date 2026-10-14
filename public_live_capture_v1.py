#!/usr/bin/env python3
"""Timestamped public Level-1 quote capture for the leader-relay workflow.

This module removes the need for screenshots for fields that a public quote
feed can actually verify (price, OHLC, cumulative volume/amount and quote
time).  It deliberately does *not* promote public Level-1 data to Level-2,
auction-cancel, reseal, queue or fill evidence.

Every run writes a new immutable capture directory.  A fresh public quote may
support WATCH/CANCEL decisions, but never authorises ENTER by itself.
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
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "v16" / "public_live"
DEFAULT_UNIVERSE = ROOT / "data" / "v3" / "imported" / "current-universe.csv"
SNAPSHOT_TIMES = ("09:20", "09:24:50", "09:25", "09:30", "09:35", "10:00")
MAINBOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")
SHANGHAI = ZoneInfo("Asia/Shanghai")
VERSION = "v16-public-live-capture-2"
SINA_URL = "https://hq.sinajs.cn/list={symbols}"
SINA_HEADERS = {
    "Referer": "https://finance.sina.com.cn/",
    "User-Agent": "Mozilla/5.0 (compatible; leader-research-public-quote/1.0)",
}
LINE_PATTERN = re.compile(r'var\s+hq_str_(?P<symbol>[a-z]{2}\d{6})="(?P<body>[^"]*)";')


def code6(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def mainboard(code: str) -> bool:
    return code.startswith(MAINBOARD_PREFIXES)


def sina_symbol(code: str) -> str:
    value = code6(code)
    return ("sh" if value.startswith(("600", "601", "603", "605")) else "sz") + value


def local_timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize(SHANGHAI)
    return stamp.tz_convert(SHANGHAI)


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def limit_price_10(previous_close: Any) -> float:
    value = safe_float(previous_close)
    if math.isnan(value) or value <= 0:
        return math.nan
    return float((Decimal(str(value)) * Decimal("1.10")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256_bytes(encoded)


def infer_asof(now: Any) -> str:
    stamp = local_timestamp(now)
    seconds = stamp.hour * 3600 + stamp.minute * 60 + stamp.second
    points = [(9 * 3600 + 20 * 60, "09:20"), (9 * 3600 + 24 * 60 + 50, "09:24:50"),
              (9 * 3600 + 25 * 60, "09:25"), (9 * 3600 + 30 * 60, "09:30"),
              (9 * 3600 + 35 * 60, "09:35"), (10 * 3600, "10:00")]
    eligible = [label for point, label in points if seconds >= point]
    return eligible[-1] if eligible else "09:20"


def capture_window_status(trade_date: str, asof: str, captured_at: Any) -> tuple[str, float]:
    if asof not in SNAPSHOT_TIMES:
        raise ValueError(f"asof must be one of {SNAPSHOT_TIMES}")
    captured = local_timestamp(captured_at)
    target = local_timestamp(f"{pd.Timestamp(trade_date).strftime('%Y-%m-%d')} {asof}:00" if len(asof) == 5
                             else f"{pd.Timestamp(trade_date).strftime('%Y-%m-%d')} {asof}")
    delta = (captured - target).total_seconds()
    # Auction changes in seconds; later opening snapshots tolerate one minute.
    early, late = ((-10.0, 20.0) if asof == "09:24:50" else
                   (-20.0, 45.0) if asof in {"09:20", "09:25"} else
                   (-20.0, 75.0))
    if captured.date() != target.date() or captured.dayofweek >= 5:
        return "MARKET_DATE_OR_WEEKEND_MISMATCH", delta
    if delta < early:
        return "EARLY", delta
    if delta > late:
        return "LATE", delta
    return "ON_TIME", delta


def parse_sina_response(text: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for match in LINE_PATTERN.finditer(text):
        symbol = match.group("symbol")
        values = match.group("body").split(",")
        if len(values) < 33 or not values[0].strip():
            continue
        quote_stamp = pd.to_datetime(f"{values[30]} {values[31]}", errors="coerce")
        if pd.notna(quote_stamp):
            quote_stamp = pd.Timestamp(quote_stamp).tz_localize(SHANGHAI).isoformat()
        else:
            quote_stamp = None
        rows.append({
            "code": symbol[-6:], "provider_name": values[0].strip(),
            "open": safe_float(values[1]), "previous_close": safe_float(values[2]),
            "last": safe_float(values[3]), "high": safe_float(values[4]),
            "low": safe_float(values[5]), "bid": safe_float(values[6]),
            "ask": safe_float(values[7]), "volume_shares": safe_float(values[8]),
            "amount": safe_float(values[9]), "quote_timestamp": quote_stamp,
            "provider_status": values[32].strip(),
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates("code", keep="last").sort_values("code").reset_index(drop=True)
    return frame


@dataclass(frozen=True)
class FetchResult:
    quotes: pd.DataFrame
    errors: list[str]
    requested: int
    raw_hashes: list[str]


def _fetch_chunk(
    symbols: list[str], get: Callable[..., Any], timeout: float, retries: int,
) -> tuple[pd.DataFrame, str, str | None]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = get(
                SINA_URL.format(symbols=",".join(symbols)), headers=SINA_HEADERS, timeout=timeout,
            )
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()
            content = response.content if hasattr(response, "content") else bytes(str(response.text), "gb18030")
            text = content.decode("gb18030", errors="replace")
            return parse_sina_response(text), sha256_bytes(content), None
        except Exception as exc:  # provider/network errors are output evidence
            last_error = exc
            if attempt < retries:
                time.sleep(0.25 * (attempt + 1))
    return pd.DataFrame(), "", str(last_error)[:300] if last_error else "unknown provider error"


def fetch_sina_quotes(
    codes: Iterable[str], *, get: Callable[..., Any] = requests.get, chunk_size: int = 300,
    workers: int = 4, timeout: float = 8.0, retries: int = 1,
) -> FetchResult:
    normalized = sorted({code6(code) for code in codes if mainboard(code6(code))})
    chunks = [normalized[index:index + chunk_size] for index in range(0, len(normalized), chunk_size)]
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    hashes: list[str] = []
    with futures.ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as executor:
        jobs = {
            executor.submit(_fetch_chunk, [sina_symbol(code) for code in chunk], get, timeout, retries): chunk
            for chunk in chunks
        }
        for job in futures.as_completed(jobs):
            frame, digest, error = job.result()
            if not frame.empty:
                frames.append(frame)
            if digest:
                hashes.append(digest)
            if error:
                errors.append(f"{jobs[job][0]}..{jobs[job][-1]}: {error}")
    quotes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not quotes.empty:
        quotes = quotes.drop_duplicates("code", keep="last").sort_values("code").reset_index(drop=True)
    return FetchResult(quotes=quotes, errors=errors, requested=len(normalized), raw_hashes=sorted(hashes))


def quote_freshness(quote_timestamp: Any, captured_at: Any) -> tuple[str, float | None]:
    if not quote_timestamp or pd.isna(quote_timestamp):
        return "UNAVAILABLE_NO_PROVIDER_TIMESTAMP", None
    try:
        quote = local_timestamp(quote_timestamp)
        captured = local_timestamp(captured_at)
    except (TypeError, ValueError):
        return "UNAVAILABLE_INVALID_PROVIDER_TIMESTAMP", None
    age = (captured - quote).total_seconds()
    if age < -30:
        return "UNAVAILABLE_FUTURE_PROVIDER_TIMESTAMP", age
    if quote.date() != captured.date():
        return "STALE_PREVIOUS_SESSION", age
    if age <= 15:
        return "VERIFIED_PUBLIC_QUOTE_FRESH", age
    if age <= 120:
        return "DELAYED_PUBLIC_QUOTE", age
    return "STALE_PUBLIC_QUOTE", age


def _rename_candidate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "代码": "code", "名称": "name", "连板数": "boards", "成交额": "signal_amount",
        "最新价": "signal_close", "日期": "signal_date",
    }
    result = frame.rename(columns={key: value for key, value in aliases.items() if key in frame.columns}).copy()
    if "amount" in result and "signal_amount" not in result:
        result["signal_amount"] = result["amount"]
    if "last_price" in result and "signal_close" not in result:
        result["signal_close"] = result["last_price"]
    return result


def load_candidates(path: Path) -> pd.DataFrame:
    frame = _rename_candidate_columns(pd.read_csv(path, dtype={"code": "string", "代码": "string"}))
    required = {"code", "name", "boards"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"candidate input missing fields: {missing}")
    if "pool_assignment" in frame:
        frame = frame[frame["pool_assignment"].eq("CLOSING_LIMIT_UP")]
    if "is_mainboard" in frame:
        frame = frame[frame["is_mainboard"].map(truthy)]
    frame["code"] = frame["code"].map(code6)
    frame = frame[frame["code"].map(mainboard)]
    frame = frame[~frame["name"].astype(str).str.upper().str.contains(r"(?:^|\s)\*?ST", regex=True)]
    frame["boards"] = pd.to_numeric(frame["boards"], errors="coerce")
    frame = frame[frame["boards"].ge(1)]
    if "signal_date" not in frame:
        raise ValueError("candidate input missing signal_date/date field")
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if frame["signal_date"].isna().any():
        raise ValueError("candidate input contains invalid signal_date")
    latest = frame["signal_date"].max()
    frame = frame[frame["signal_date"].eq(latest)].copy()
    if frame["code"].duplicated().any():
        raise ValueError("candidate input contains duplicate closing-limit codes")
    for field in ("signal_amount", "signal_close"):
        if field not in frame:
            frame[field] = math.nan
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    keep = ["signal_date", "code", "name", "boards", "signal_amount", "signal_close"]
    for optional in ("board_group", "eligibility_status", "security_status", "industry_provider"):
        if optional in frame:
            keep.append(optional)
    return frame[keep].sort_values(["boards", "code"], ascending=[False, True]).reset_index(drop=True)


def discover_candidates(root: Path = ROOT) -> Path:
    paths = list((root / "results").glob("current_full_pool_*/all_closing_and_failed_candidates.csv"))
    if not paths:
        raise FileNotFoundError("no standardized current full-pool candidate file found; pass --candidates")
    dated: list[tuple[str, Path]] = []
    for path in paths:
        try:
            sample = pd.read_csv(path, usecols=["signal_date"], nrows=10)
            dated.append((str(sample["signal_date"].max()), path))
        except Exception:
            continue
    if not dated:
        raise ValueError("current full-pool files exist but none has a readable signal_date")
    return max(dated, key=lambda item: (item[0], str(item[1])))[1]


def load_universe(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists() or not path.stat().st_size:
        return pd.DataFrame(columns=["code", "name"])
    frame = pd.read_csv(path, dtype={"code": "string"})
    if not {"code", "name"}.issubset(frame.columns):
        raise ValueError("universe input requires code and name")
    frame = frame[["code", "name"]].copy()
    frame["code"] = frame["code"].map(code6)
    frame = frame[frame["code"].map(mainboard)]
    frame = frame[~frame["name"].astype(str).str.upper().str.contains(r"(?:^|\s)\*?ST", regex=True)]
    return frame.drop_duplicates("code").reset_index(drop=True)


def build_candidate_snapshot(
    candidates: pd.DataFrame, quotes: pd.DataFrame, captured_at: Any, trade_date: str, asof: str,
) -> pd.DataFrame:
    quote_columns = ["code", "open", "previous_close", "last", "high", "low", "bid", "ask",
                     "volume_shares", "amount", "quote_timestamp", "provider_status"]
    if quotes.empty:
        quote_frame = pd.DataFrame({column: pd.Series(dtype="object") for column in quote_columns})
    else:
        quote_frame = quotes.copy()
        for column in quote_columns:
            if column not in quote_frame:
                quote_frame[column] = math.nan
        quote_frame = quote_frame[quote_columns]
    merged = candidates.merge(quote_frame, on="code", how="left", validate="one_to_one")
    freshness = merged["quote_timestamp"].map(lambda value: quote_freshness(value, captured_at))
    merged["quote_data_status"] = freshness.map(lambda value: value[0])
    merged["quote_age_seconds"] = freshness.map(lambda value: value[1])
    merged["trade_date"] = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    merged["asof"] = asof
    merged["captured_at"] = local_timestamp(captured_at).isoformat()
    merged["auction_gap_proxy"] = merged["open"] / merged["previous_close"] - 1
    merged.loc[merged["open"].le(0) | merged["previous_close"].le(0), "auction_gap_proxy"] = math.nan
    merged["cumulative_amount_to_signal_ratio_proxy"] = merged["amount"] / merged["signal_amount"]
    merged.loc[merged["signal_amount"].le(0), "cumulative_amount_to_signal_ratio_proxy"] = math.nan
    merged["limit_price_10pct_inferred"] = merged["previous_close"].map(limit_price_10)
    merged["at_limit_price_proxy"] = (
        merged["last"].notna() & merged["limit_price_10pct_inferred"].notna()
        & merged["last"].ge(merged["limit_price_10pct_inferred"] - 0.005)
    )
    merged["public_level1_buyability"] = merged["at_limit_price_proxy"].map(
        lambda value: "UNAVAILABLE_NO_QUEUE" if value else "PRICE_NOT_LOCKED_LIMIT_PROXY"
    )

    def decision(row: pd.Series) -> str:
        if row["quote_data_status"] != "VERIFIED_PUBLIC_QUOTE_FRESH":
            return "CASH_DATA_NOT_FRESH"
        if row["at_limit_price_proxy"]:
            return "WATCH_AT_LIMIT_BUYABILITY_UNKNOWN"
        gap = safe_float(row["auction_gap_proxy"])
        if not math.isnan(gap) and gap > 0.07:
            return "WATCH_OVERHEATED"
        if not math.isnan(gap) and gap < -0.03:
            return "CANCEL_WEAK_OPEN"
        return "WATCH_PUBLIC_PRICE_ONLY"

    merged["public_feed_decision"] = merged.apply(decision, axis=1)
    merged["direct_entry_authorized"] = False
    merged["unavailable_execution_fields"] = (
        "auction_cancel_ratio,exact_auction_amount,level2_queue,active_orders,"
        "first_pullback_volume,verified_theme_sync,reseal_sequence,actual_fill"
    )
    return merged


def breadth_summary(universe: pd.DataFrame, quotes: pd.DataFrame, captured_at: Any) -> dict[str, Any]:
    if universe.empty:
        return {
            "status": "UNAVAILABLE_NO_UNIVERSE", "universe_rows": 0, "quote_rows": 0,
            "coverage": 0.0, "fresh_coverage": 0.0,
        }
    if quotes.empty or not {"code", "last", "previous_close", "quote_timestamp"}.issubset(quotes.columns):
        return {
            "status": "UNAVAILABLE_NO_USABLE_QUOTES", "universe_rows": len(universe),
            "quote_rows": 0, "coverage": 0.0, "fresh_coverage": 0.0,
        }
    merged = universe.merge(quotes, on="code", how="left")
    usable = merged[merged["last"].notna() & merged["previous_close"].gt(0)].copy()
    if usable.empty:
        return {
            "status": "UNAVAILABLE_NO_USABLE_QUOTES", "universe_rows": len(universe),
            "quote_rows": 0, "coverage": 0.0, "fresh_coverage": 0.0,
        }
    usable["return"] = usable["last"] / usable["previous_close"] - 1
    usable["fresh"] = usable["quote_timestamp"].map(
        lambda value: quote_freshness(value, captured_at)[0] == "VERIFIED_PUBLIC_QUOTE_FRESH"
    )
    coverage = len(usable) / len(universe)
    fresh_coverage = usable["fresh"].sum() / len(universe)
    status = (
        "INFERRED_FROM_FRESH_PUBLIC_QUOTES" if coverage >= 0.95 and fresh_coverage >= 0.90 else
        "PARTIAL_OR_DELAYED_PUBLIC_QUOTES"
    )
    return {
        "status": status, "universe_rows": int(len(universe)), "quote_rows": int(len(usable)),
        "coverage": float(coverage), "fresh_coverage": float(fresh_coverage),
        "advance_count": int(usable["return"].gt(0).sum()),
        "decline_count": int(usable["return"].lt(0).sum()),
        "flat_count": int(usable["return"].eq(0).sum()),
        "decline_ratio": float(usable["return"].lt(0).mean()),
        "limit_up_proxy_count": int(usable["return"].ge(0.095).sum()),
        "limit_down_proxy_count": int(usable["return"].le(-0.095).sum()),
        "warning": "breadth uses a cached current mainboard/non-ST name universe and 9.5% proxies; it is not an official limit pool",
    }


def make_report(manifest: dict[str, Any], snapshot: pd.DataFrame) -> str:
    lines = [
        f"# {manifest['trade_date']} {manifest['asof']} 公开行情快照", "",
        "## 数据质量", "",
        f"- 来源：{manifest['source']}",
        f"- 本地接收时间：{manifest['captured_at']}",
        f"- 采集窗口：{manifest['capture_window_status']}（偏差 {manifest['capture_delta_seconds']:.1f} 秒）",
        f"- 候选覆盖：{manifest['candidate_quote_rows']}/{manifest['candidate_rows']}",
        f"- 新鲜候选：{manifest['fresh_candidate_rows']}/{manifest['candidate_rows']}",
        "- 已核验字段：公开Level-1价格、开高低、昨收、累计量额、行情时间戳。",
        "- 不可用字段：真实竞价撤单、逐笔主动单、Level-2封单队列、回封顺序、真实成交。",
        "- 执行限制：本层永不单独授权ENTER；只支持WATCH、CANCEL或CASH。", "",
        "## 全部昨日涨停候选", "",
        "| 代码 | 名称 | 板位 | 开盘代理 | 最新价 | 行情状态 | 价格结论 | 可买性 |", "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in snapshot.sort_values(["boards", "code"], ascending=[False, True]).itertuples(index=False):
        gap = "缺失" if pd.isna(row.auction_gap_proxy) else f"{row.auction_gap_proxy:.2%}"
        last = "缺失" if pd.isna(row.last) else f"{row.last:.2f}"
        lines.append(
            f"| {row.code} | {row.name} | {int(row.boards)} | {gap} | {last} | "
            f"{row.quote_data_status} | {row.public_feed_decision} | {row.public_level1_buyability} |"
        )
    lines += ["", "结论：公开行情可替代截图中的价格层，但不能替代Level-2和真实成交证据。", ""]
    return "\n".join(lines)


def capture(
    candidate_path: Path, trade_date: str, asof: str, output: Path = DEFAULT_OUTPUT,
    universe_path: Path | None = DEFAULT_UNIVERSE, *, now: Any | None = None,
    quote_fetcher: Callable[..., FetchResult] = fetch_sina_quotes,
) -> dict[str, Any]:
    request_started = local_timestamp(now or datetime.now(SHANGHAI))
    point = infer_asof(request_started) if asof == "auto" else asof
    candidates = load_candidates(candidate_path)
    if candidates.empty:
        raise ValueError("candidate input contains no mainboard closing-limit rows")
    signal_date = str(candidates["signal_date"].iloc[0])
    trade = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    universe = load_universe(universe_path)
    candidate_codes = set(candidates["code"])
    # Latency-sensitive candidates are always fetched first.  The optional
    # breadth universe is a second request and can never delay the timestamp
    # used to judge candidate freshness.
    candidate_fetch = quote_fetcher(candidate_codes)
    candidate_received = request_started if now is not None else local_timestamp(datetime.now(SHANGHAI))
    universe_codes = set(universe["code"] if not universe.empty else []) - candidate_codes
    breadth_fetch = quote_fetcher(universe_codes) if universe_codes else FetchResult(
        quotes=pd.DataFrame(), errors=[], requested=0, raw_hashes=[],
    )
    completed = request_started if now is not None else local_timestamp(datetime.now(SHANGHAI))
    frames = [frame for frame in (candidate_fetch.quotes, breadth_fetch.quotes) if not frame.empty]
    all_quotes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not all_quotes.empty:
        all_quotes = all_quotes.drop_duplicates("code", keep="first").sort_values("code").reset_index(drop=True)
    fetched = FetchResult(
        quotes=all_quotes,
        errors=[f"candidate: {error}" for error in candidate_fetch.errors]
               + [f"breadth: {error}" for error in breadth_fetch.errors],
        requested=candidate_fetch.requested + breadth_fetch.requested,
        raw_hashes=sorted(candidate_fetch.raw_hashes + breadth_fetch.raw_hashes),
    )
    window_status, delta = capture_window_status(trade, point, candidate_received)
    snapshot = build_candidate_snapshot(candidates, candidate_fetch.quotes, candidate_received, trade, point)
    breadth = breadth_summary(universe, all_quotes, completed)
    quote_rows = int(snapshot["quote_timestamp"].notna().sum())
    fresh_rows = int(snapshot["quote_data_status"].eq("VERIFIED_PUBLIC_QUOTE_FRESH").sum())
    overall_status = (
        "CONDITIONAL_PUBLIC_PRICE_FRESH" if window_status == "ON_TIME" and fresh_rows == len(snapshot) else
        "CASH_PUBLIC_DATA_INCOMPLETE_OR_NOT_ON_TIME"
    )
    manifest: dict[str, Any] = {
        "version": VERSION, "status": overall_status, "signal_date": signal_date,
        "trade_date": trade, "asof": point, "request_started_at": request_started.isoformat(),
        "captured_at": candidate_received.isoformat(),
        "candidate_response_received_at": candidate_received.isoformat(),
        "capture_completed_at": completed.isoformat(),
        "capture_window_status": window_status, "capture_delta_seconds": float(delta),
        "source": "Sina public Level-1 quote endpoint",
        "candidate_file": str(candidate_path.resolve()), "candidate_rows": int(len(snapshot)),
        "candidate_quote_rows": quote_rows, "fresh_candidate_rows": fresh_rows,
        "candidate_requested_quote_codes": candidate_fetch.requested,
        "breadth_requested_quote_codes": breadth_fetch.requested,
        "requested_quote_codes": fetched.requested, "returned_quote_codes": int(len(fetched.quotes)),
        "provider_errors": fetched.errors, "provider_response_hashes": fetched.raw_hashes,
        "market_breadth": breadth, "direct_entry_authorized": False,
        "decision_scope": "WATCH_CANCEL_CASH_ONLY",
        "limitations": [
            "public Level-1 quotes are not broker/terminal Level-2 data",
            "quote freshness does not prove order-queue stability or actual buyability at limit-up",
            "cumulative amount is not labelled as exact auction amount after continuous trading starts",
            "no automatic order placement is performed",
        ],
    }
    identity = {
        "version": VERSION, "signal_date": signal_date, "trade_date": trade, "asof": point,
        "captured_at": candidate_received.isoformat(), "candidate_codes": snapshot["code"].tolist(),
        "provider_response_hashes": fetched.raw_hashes,
    }
    capture_id = hash_payload(identity)[:20]
    target = output / trade / point.replace(":", "") / capture_id
    if target.exists():
        raise FileExistsError("immutable public quote capture already exists")
    target.mkdir(parents=True, exist_ok=False)
    quotes_path = target / "candidate_quotes.csv"
    breadth_path = target / "market_breadth.json"
    report_path = target / "report.md"
    manifest_path = target / "manifest.json"
    snapshot.to_csv(quotes_path, index=False, encoding="utf-8-sig")
    breadth_path.write_text(json.dumps(breadth, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["files"] = {
        "candidate_quotes": {"path": str(quotes_path.resolve()), "sha256": sha256_bytes(quotes_path.read_bytes())},
        "market_breadth": {"path": str(breadth_path.resolve()), "sha256": sha256_bytes(breadth_path.read_bytes())},
    }
    report_path.write_text(make_report(manifest, snapshot), encoding="utf-8")
    manifest["files"]["report"] = {"path": str(report_path.resolve()), "sha256": sha256_bytes(report_path.read_bytes())}
    manifest["capture_id"] = capture_id
    manifest["manifest_hash"] = hash_payload({key: value for key, value in manifest.items() if key != "manifest_hash"})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**manifest, "capture_dir": str(target.resolve()), "manifest_path": str(manifest_path.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture timestamped public Level-1 quotes for all closing-limit candidates")
    parser.add_argument("--candidates", type=Path, help="standardized full-pool candidate CSV; auto-discovers latest when omitted")
    parser.add_argument("--trade-date", default=datetime.now(SHANGHAI).strftime("%Y-%m-%d"))
    parser.add_argument("--asof", choices=(*SNAPSHOT_TIMES, "auto"), default="auto")
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE,
                        help="cached current universe for optional breadth; missing file is reported, not imputed")
    parser.add_argument("--no-breadth", action="store_true", help="capture candidates only for lower latency")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    candidate_file = args.candidates or discover_candidates()
    universe_file = None if args.no_breadth else args.universe
    result = capture(candidate_file, args.trade_date, args.asof, args.output, universe_file)
    print(json.dumps(result, ensure_ascii=False, indent=2))
