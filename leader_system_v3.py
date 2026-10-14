"""A-share main-board continuous-limit leader system V3.

The system is deliberately research-first.  It separates:
1) end-of-day candidate probability;
2) 09:25-10:00 confirmation;
3) observable false-strength and large-loss risk;
4) execution and deployment gates.

It never interprets missing auction/Level-2 data as zero and never claims to infer
the intent of a so-called "main force".  Outputs are conditional research signals.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Collection, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "v3"
RAW = DATA / "raw"
IMPORTED = DATA / "imported"
RESULTS = ROOT / "results" / "v3"
MODEL_PATH = RESULTS / "model_card.json"
FORWARD = RESULTS / "forward"
ACTIVE_FREEZE_PATH = FORWARD / "active_freeze.json"
FORWARD_LEDGER_PATH = FORWARD / "forward_ledger.jsonl"
FORWARD_SIGNALS_PATH = DATA / "forward_signals.csv"
PAPER_TRADES_PATH = DATA / "paper_trades.csv"

SNAPSHOT_TIMES = ("09:20", "09:24:50", "09:25", "09:30", "09:35", "10:00")
ACTIVE_STATES = {"启动", "升温", "加速", "修复"}
STATE_NAMES = ("启动", "升温", "加速", "一致", "大分歧", "退潮", "修复")
STATE_PROBABILITY_COLUMNS = {
    "启动": "emotion_next_start_probability",
    "升温": "emotion_next_warming_probability",
    "加速": "emotion_next_acceleration_probability",
    "一致": "emotion_next_consensus_probability",
    "大分歧": "emotion_next_divergence_probability",
    "退潮": "emotion_next_retreat_probability",
    "修复": "emotion_next_repair_probability",
}
EMOTION_ADVERSE_MODEL_FEATURES = [
    "breadth_health", "relay_appetite", "ladder_health", "loss_effect_health",
    "crowding_risk", "regime_fragility", "false_acceleration",
    "promotion_overall", "promotion_mid", "space_feedback", "space_feedback_3d",
    "failed_rate", "down_limit_count", "previous_limit_tail_loss", "yesterday_premium",
    "height_change", "limit_change", "max_board", "middle_density", "ladder_gap_count",
    "one_word_share", "emotion_state_age_scaled", "emotion_adverse_markov_probability",
    "emotion_transition_entropy", "state_start", "state_warming", "state_acceleration",
    "state_consensus", "state_divergence", "state_retreat", "state_repair",
]

BASE_FEATURES = [
    "boards", "height_gap", "is_space_leader", "unique_space", "height_share",
]
EMOTION_FEATURES = [
    "emotion_score", "limit_count_scaled", "consecutive_scaled", "down_limit_rate",
    "failed_rate", "seal_rate", "height_change_scaled", "limit_change_scaled",
    "promotion_overall", "promotion_1to2", "promotion_2to3", "promotion_3to4",
    "promotion_4to5", "promotion_mid", "space_feedback", "space_feedback_3d",
    "middle_density", "ladder_gap_count", "breadth_height_divergence",
    "yesterday_premium", "failed_tail_loss", "previous_limit_tail_loss", "state_start", "state_warming",
    "state_acceleration", "state_consensus", "state_divergence", "state_retreat",
    "state_repair", "breadth_health", "relay_appetite", "ladder_health",
    "loss_effect_health", "crowding_risk", "regime_fragility",
    "false_acceleration", "emotion_data_completeness",
    "emotion_adverse_next_probability", "emotion_supportive_next_probability",
    "emotion_persistence_probability", "emotion_transition_entropy",
    "emotion_transition_evidence_scaled", "emotion_state_age_scaled",
    "failed_rate_percentile_60d", "down_limit_percentile_60d",
    "promotion_mid_percentile_60d", "yesterday_premium_percentile_60d",
    "tail_loss_percentile_60d",
]
THEME_FEATURES = [
    "theme_support", "theme_middle_support", "theme_first_support", "theme_promotion",
    "theme_concentration", "theme_fragmentation", "same_height_rivals",
    "capacity_anchor_health", "industry_support_proxy",
]
INDIVIDUAL_FEATURES = [
    "turnover", "turnover_vs_prev", "amount_float_cap", "sealed_fund_float_cap",
    "first_seal_minute", "last_seal_minute", "bomb_count", "one_word_proxy",
    "one_word_streak", "volume_ratio", "amplitude", "relative_strength",
    "price", "float_cap_log",
]
OPEN_FEATURES = [
    "auction_gap", "auction_amount_ratio", "auction_cancel_ratio", "expectation_gap",
    "vwap_slope_5m", "vwap_slope_15m", "pullback_volume_ratio", "vwap_reclaimed",
    "reseal_minute", "theme_sync_open", "rivals_eliminated", "below_vwap_expanding",
    "failed_seal_twice",
]
RISK_FEATURES = [
    "announcement_risk", "abnormal_10d_distance", "abnormal_30d_distance",
    "reduction_risk", "unlock_risk", "earnings_risk", "rumor_only",
    "crowding_score", "sentiment_unanimity",
]
FEATURE_GROUPS = {
    "baseline": BASE_FEATURES,
    "emotion": EMOTION_FEATURES,
    "theme": THEME_FEATURES,
    "individual": INDIVIDUAL_FEATURES,
    "open": OPEN_FEATURES,
    "risk": RISK_FEATURES,
}
EOD_FEATURES = BASE_FEATURES + EMOTION_FEATURES + THEME_FEATURES + INDIVIDUAL_FEATURES + RISK_FEATURES
OPEN_MODEL_FEATURES = EOD_FEATURES + OPEN_FEATURES


def ensure_dirs() -> None:
    for path in (DATA, RAW, IMPORTED, RESULTS, FORWARD):
        path.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "" or pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def clamp(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    number = safe_float(value)
    if math.isnan(number):
        return math.nan
    return max(lo, min(hi, number))


def linear_score(value: Any, bad: float, good: float) -> float:
    """Map a market variable to [0, 1] without converting missing data to zero."""
    number = safe_float(value)
    if math.isnan(number):
        return math.nan
    if good == bad:
        return math.nan
    return clamp((number - bad) / (good - bad))


def json_ready(value: Any) -> Any:
    """Return strict-JSON values; unavailable market data becomes null, never NaN."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def json_text(value: Any) -> str:
    return json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def forward_event_hash(event: dict[str, Any]) -> str:
    payload = {key: value for key, value in event.items() if key != "event_hash"}
    return canonical_hash(payload)


def read_forward_events(ledger_path: Path = FORWARD_LEDGER_PATH) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        return []
    events = []
    for line_number, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"forward ledger line {line_number} is invalid JSON") from exc
    return events


def verify_forward_ledger(ledger_path: Path = FORWARD_LEDGER_PATH,
                          root: Path = ROOT) -> dict[str, Any]:
    """Verify the local append-only hash chain and every referenced score snapshot.

    This is tamper-evident local evidence, not an external trusted timestamp.
    """
    events = read_forward_events(ledger_path)
    previous = "GENESIS"
    errors: list[str] = []
    for position, event in enumerate(events, 1):
        if int(event.get("sequence", -1)) != position:
            errors.append(f"sequence mismatch at event {position}")
        if event.get("previous_event_hash") != previous:
            errors.append(f"previous hash mismatch at event {position}")
        expected = forward_event_hash(event)
        if event.get("event_hash") != expected:
            errors.append(f"event hash mismatch at event {position}")
        references = []
        if event.get("event_type") == "SIGNAL_SNAPSHOT":
            references.append((event.get("snapshot_file"), event.get("snapshot_sha256"), "snapshot"))
        elif event.get("event_type") == "MODEL_FREEZE":
            references.append((event.get("manifest_file"), event.get("manifest_sha256"), "freeze manifest"))
        for relative, expected_sha, label in references:
            referenced_path = root / str(relative) if relative else None
            if referenced_path is None or not referenced_path.exists():
                errors.append(f"{label} missing at event {position}")
            elif sha256_file(referenced_path) != expected_sha:
                errors.append(f"{label} hash mismatch at event {position}")
        previous = str(event.get("event_hash", ""))
    return {
        "status": "PASS" if not errors else "FAIL",
        "events": len(events),
        "head_event_hash": previous,
        "errors": errors,
        "assurance": "LOCAL_TAMPER_EVIDENT_ONLY",
    }


def append_forward_event(event: dict[str, Any], ledger_path: Path = FORWARD_LEDGER_PATH,
                         root: Path = ROOT) -> dict[str, Any]:
    verification = verify_forward_ledger(ledger_path, root)
    if verification["status"] != "PASS":
        raise ValueError("forward ledger failed verification; refusing to append")
    enriched = dict(event)
    enriched["sequence"] = int(verification["events"]) + 1
    enriched["previous_event_hash"] = verification["head_event_hash"]
    enriched["recorded_at"] = datetime.now().isoformat(timespec="seconds")
    enriched["event_hash"] = forward_event_hash(enriched)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(json_ready(enriched), ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False) + "\n")
    return enriched


def zfill_code(value: Any) -> str:
    text = str(value).split(".")[0].strip()
    return text.zfill(6)


def is_mainboard(code: Any, name: Any = "", st_flag: Any = False, new_flag: Any = False) -> bool:
    code = zfill_code(code)
    name = str(name)
    st = str(st_flag).strip().lower() in {"1", "true", "yes", "是"}
    new = str(new_flag).strip().lower() in {"1", "true", "yes", "是"}
    return bool(code.startswith(("60", "00")) and not st and not new
                and not any(x in name.upper() for x in ("ST", "退")))


def round_limit_price(pre_close: Any, ratio: float = 0.10) -> float:
    price = Decimal(str(pre_close)) * (Decimal("1") + Decimal(str(ratio)))
    return float(price.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def parse_clock(value: Any) -> float:
    if value is None or value == "" or pd.isna(value):
        return math.nan
    text = "".join(ch for ch in str(value).split(".")[0] if ch.isdigit()).zfill(6)
    try:
        hour, minute, second = int(text[:2]), int(text[2:4]), int(text[4:6])
        return hour * 60 + minute + second / 60
    except ValueError:
        return math.nan


def weighted_available(parts: Iterable[tuple[float, Any]]) -> float:
    valid = [(w, safe_float(v)) for w, v in parts if not math.isnan(safe_float(v))]
    if not valid:
        return math.nan
    total = sum(w for w, _ in valid)
    return sum(w * v for w, v in valid) / total


def first_existing(row: pd.Series | dict[str, Any], names: Iterable[str], default: Any = math.nan) -> Any:
    for name in names:
        if name in row and not pd.isna(row[name]):
            return row[name]
    return default


def normalize_pool(frame: pd.DataFrame, date: str) -> pd.DataFrame:
    mapping = {
        "代码": "code", "名称": "name", "最新价": "price", "涨跌幅": "pct_change",
        "成交额": "amount", "流通市值": "float_cap", "总市值": "total_cap",
        "换手率": "turnover", "封板资金": "sealed_fund", "首次封板时间": "first_seal",
        "最后封板时间": "last_seal", "炸板次数": "bomb_count", "连板数": "boards",
        "所属行业": "industry", "量比": "volume_ratio", "振幅": "amplitude",
    }
    out = frame.rename(columns=mapping).copy()
    out["date"] = pd.Timestamp(date).strftime("%Y-%m-%d")
    if "code" not in out:
        out["code"] = ""
    if "name" not in out:
        out["name"] = ""
    out["code"] = out["code"].map(zfill_code)
    for col in ("price", "pct_change", "amount", "float_cap", "total_cap", "turnover",
                "sealed_fund", "bomb_count", "boards", "volume_ratio", "amplitude"):
        if col not in out:
            out[col] = math.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ("first_seal", "last_seal", "industry"):
        if col not in out:
            out[col] = ""
    return out[out.apply(lambda r: is_mainboard(r["code"], r["name"]), axis=1)].reset_index(drop=True)


def collect_command(args: argparse.Namespace) -> None:
    ensure_dirs()
    try:
        import akshare as ak
    except ImportError as exc:
        raise SystemExit("collect requires akshare; install it or use the import command") from exc

    if args.history_start:
        collect_free_history(ak, args)
        if not args.date:
            return
    if not args.date:
        raise SystemExit("collect requires --date or --history-start/--history-end")
    date = pd.Timestamp(args.date).strftime("%Y%m%d")
    day_dir = RAW / date
    day_dir.mkdir(parents=True, exist_ok=True)

    calls = {
        "limit_up": ak.stock_zt_pool_em,
        "failed_board": ak.stock_zt_pool_zbgc_em,
        "limit_down": ak.stock_zt_pool_dtgc_em,
        "previous_limit": ak.stock_zt_pool_previous_em,
        "strong": ak.stock_zt_pool_strong_em,
    }
    manifest = {"date": date, "collected_at": datetime.now().isoformat(timespec="seconds"), "datasets": {}}
    for kind, func in calls.items():
        try:
            frame = func(date=date)
            frame.to_csv(day_dir / f"{kind}.csv", index=False, encoding="utf-8-sig")
            manifest["datasets"][kind] = {"status": "verified", "rows": int(len(frame)), "source": "Eastmoney via AKShare"}
        except Exception as exc:  # Network/provider failures must be recorded, not hidden.
            manifest["datasets"][kind] = {"status": "unavailable", "error": str(exc)[:300]}

    snapshot_rows: list[dict[str, Any]] = []
    if args.asof:
        signal_date = pd.Timestamp(args.signal_date or args.date).strftime("%Y-%m-%d")
        candidates_path = DATA / "leader_candidates.csv"
        candidates = pd.read_csv(candidates_path, dtype={"code": str}) if candidates_path.exists() else pd.DataFrame()
        if not candidates.empty:
            candidates = candidates[candidates["date"] == signal_date]
        spot_map: dict[str, dict[str, Any]] = {}
        is_today = pd.Timestamp(args.date).date() == pd.Timestamp.now().date()
        if is_today and not candidates.empty:
            try:
                spot = ak.stock_zh_a_spot_em().rename(columns={
                    "代码": "code", "名称": "name", "最新价": "last", "今开": "open",
                    "成交量": "volume", "成交额": "amount", "换手率": "turnover",
                })
                spot["code"] = spot["code"].map(zfill_code)
                spot_map = spot.set_index("code").to_dict("index")
            except Exception as exc:
                manifest["datasets"]["live_spot"] = {"status": "unavailable", "error": str(exc)[:300]}
        for _, row in candidates.iterrows():
            code = zfill_code(row["code"])
            spot = spot_map.get(code, {})
            previous_close = safe_float(row.get("price"))
            open_price = safe_float(spot.get("open"))
            live_amount = safe_float(spot.get("amount"))
            prior_amount = safe_float(row.get("signal_amount"))
            auction_gap = open_price / previous_close - 1 if open_price > 0 and previous_close > 0 else math.nan
            amount_ratio = live_amount / prior_amount if live_amount >= 0 and prior_amount > 0 else math.nan
            verified_fields = [x for x, value in (("auction_gap", auction_gap), ("auction_amount_ratio", amount_ratio))
                               if not math.isnan(value)]
            unavailable = [name for name in OPEN_FEATURES if name not in verified_fields]
            snapshot_rows.append({
                "signal_date": signal_date, "trade_date": pd.Timestamp(args.date).strftime("%Y-%m-%d"),
                "asof": args.asof, "code": code, "name": row.get("name", ""),
                "source": "Eastmoney public spot via AKShare" if spot else "snapshot placeholder",
                "source_timestamp": datetime.now().isoformat(timespec="seconds"),
                "data_status": "partial" if verified_fields else "unavailable",
                "unavailable_fields": ",".join(unavailable),
                **{name: (auction_gap if name == "auction_gap" else amount_ratio if name == "auction_amount_ratio" else math.nan)
                   for name in OPEN_FEATURES},
            })
        if spot_map:
            manifest["datasets"]["live_spot"] = {"status": "delayed", "rows": len(spot_map),
                                                   "source": "Eastmoney public spot via AKShare",
                                                   "limitation": "not Level-2; no cancel sequence"}
        if snapshot_rows:
            append_dedup(pd.DataFrame(snapshot_rows), DATA / "open_snapshots.csv",
                         ["signal_date", "trade_date", "asof", "code"])
    (day_dir / "manifest.json").write_text(json_text(manifest), encoding="utf-8")
    print(json_text(manifest))


def current_mainboard_universe(ak: Any) -> tuple[list[tuple[str, str]], str]:
    """Resolve a current (not point-in-time complete) universe with provider fallback."""
    cache_path = IMPORTED / "current-universe.csv"
    errors: list[str] = []
    try:
        spot = ak.stock_zh_a_spot_em().rename(columns={"代码": "code", "名称": "name"})
        source = "current Eastmoney spot universe via AKShare"
    except Exception as exc:
        errors.append(f"stock_zh_a_spot_em: {str(exc)[:160]}")
        try:
            spot = ak.stock_info_a_code_name().rename(columns={"代码": "code", "名称": "name"})
            source = "current A-share code/name list via AKShare fallback"
        except Exception as fallback_exc:
            errors.append(f"stock_info_a_code_name: {str(fallback_exc)[:160]}")
            if not cache_path.exists():
                raise RuntimeError("all current-universe providers failed: " + " | ".join(errors)) from fallback_exc
            spot = pd.read_csv(cache_path, dtype={"code": str})
            source = "cached current universe (provider unavailable)"
    if "code" not in spot or "name" not in spot:
        raise ValueError("current-universe provider did not return code/name columns")
    spot = spot[["code", "name"]].copy()
    spot["code"] = spot["code"].map(zfill_code)
    spot["name"] = spot["name"].fillna("").astype(str)
    spot = spot[spot.apply(lambda row: is_mainboard(row["code"], row["name"]), axis=1)]
    spot = spot.drop_duplicates("code").sort_values("code")
    spot["collected_at"] = datetime.now().isoformat(timespec="seconds")
    spot["source"] = source
    spot.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return list(zip(spot["code"], spot["name"])), source


def collect_free_history(ak: Any, args: argparse.Namespace) -> None:
    if not args.history_end:
        raise SystemExit("--history-start requires --history-end")
    start = pd.Timestamp(args.history_start)
    end = pd.Timestamp(args.history_end)
    fetch_start = (start - pd.Timedelta(days=35)).strftime("%Y%m%d")
    fetch_end = end.strftime("%Y%m%d")
    if args.codes_file:
        source = pd.read_csv(args.codes_file, dtype=str)
        code_column = "code" if "code" in source else "代码"
        name_column = "name" if "name" in source else "名称" if "名称" in source else None
        universe = [(zfill_code(r[code_column]), str(r[name_column]) if name_column else "") for _, r in source.iterrows()]
        universe_complete = bool("universe_complete" in source and source["universe_complete"].astype(str).str.lower().isin({"1", "true", "yes"}).all())
        universe_source = str(Path(args.codes_file).resolve())
    else:
        try:
            universe, universe_source = current_mainboard_universe(ak)
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        universe_complete = False
    if args.max_codes:
        universe = universe[:args.max_codes]
    skip_eastmoney_history = "fallback" in universe_source.lower() or "cached" in universe_source.lower()

    requested_start, requested_end = str(start.date()), str(end.date())
    progress_path = IMPORTED / "history-progress.csv"
    parts_dir = IMPORTED / "history-parts" / f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if progress_path.exists() and not getattr(args, "refresh_history", False):
        progress = pd.read_csv(progress_path, dtype={"code": str})
        matched = progress[(progress["requested_start"] == requested_start)
                           & (progress["requested_end"] == requested_end)
                           & (progress["status"] == "success")]
        completed = set(matched["code"].map(zfill_code))
    completed.update(path.stem for path in parts_dir.glob("*.csv"))
    pending_universe = [item for item in universe if item[0] not in completed]

    def fetch_one(item: tuple[str, str]) -> tuple[pd.DataFrame | None, str]:
        code, name = item
        provider_errors: list[str] = []
        hist: pd.DataFrame | None = None
        provider = ""
        if not skip_eastmoney_history:
            for _ in range(2):
                try:
                    hist = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=fetch_start,
                                              end_date=fetch_end, adjust="")
                    if hist.empty:
                        provider_errors.append("Eastmoney: empty response")
                        continue
                    hist = hist.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
                                                "最低": "low", "成交量": "volume", "成交额": "amount",
                                                "换手率": "turnover"})
                    hist["float_cap"] = math.nan
                    provider = "Eastmoney unadjusted daily via AKShare"
                    break
                except Exception as exc:
                    provider_errors.append(f"Eastmoney: {str(exc)[:220]}")
        if hist is None or hist.empty:
            symbol = ("sh" if code.startswith("6") else "sz") + code
            for _ in range(2):
                try:
                    hist = ak.stock_zh_a_daily(
                        symbol=symbol, start_date=fetch_start, end_date=fetch_end, adjust="")
                    if hist.empty:
                        provider_errors.append("Sina: empty response")
                        continue
                    hist = hist.rename(columns={"outstanding_share": "_outstanding_share"})
                    hist["turnover"] = pd.to_numeric(hist.get("turnover"), errors="coerce") * 100
                    outstanding = pd.to_numeric(hist.get("_outstanding_share"), errors="coerce")
                    hist["float_cap"] = outstanding * pd.to_numeric(hist.get("close"), errors="coerce")
                    provider = "Sina unadjusted daily via AKShare fallback"
                    break
                except Exception as exc:
                    provider_errors.append(f"Sina: {str(exc)[:220]}")
        if hist is None or hist.empty:
            return None, " | ".join(provider_errors)[:600] or "unknown provider failure"
        hist["date"] = pd.to_datetime(hist["date"]).dt.strftime("%Y-%m-%d")
        hist = hist.sort_values("date")
        hist["pre_close"] = pd.to_numeric(hist["close"], errors="coerce").shift(1)
        earliest = pd.Timestamp(hist["date"].min())
        looks_new = earliest > start - pd.Timedelta(days=20)
        hist["new_listing"] = False
        if looks_new:
            hist.loc[hist.index[:10], "new_listing"] = True
        hist["code"] = code
        hist["name"] = name
        hist["st"] = "ST" in name.upper()
        hist["status_verified"] = False
        hist["industry"] = ""
        hist["theme"] = ""
        hist["benchmark_return"] = math.nan
        hist["source"] = provider
        hist["universe_complete"] = universe_complete
        keep = ["date", "code", "name", "open", "high", "low", "close", "pre_close", "volume",
                "amount", "turnover", "float_cap", "st", "new_listing", "status_verified",
                "industry", "theme", "benchmark_return", "source", "universe_complete"]
        return hist[keep].dropna(subset=["pre_close"]), ""

    progress_rows: list[dict[str, Any]] = []
    successful_codes: list[str] = []
    rows_this_run = 0
    failed_this_run = 0
    provider_row_counts: dict[str, int] = {}

    def flush_progress() -> None:
        nonlocal progress_rows
        if progress_rows:
            append_dedup(pd.DataFrame(progress_rows), progress_path,
                         ["requested_start", "requested_end", "code"])
            progress_rows = []

    with futures.ThreadPoolExecutor(max_workers=max(1, min(5, args.workers))) as executor:
        future_map = {executor.submit(fetch_one, item): item for item in pending_universe}
        for completed_count, future in enumerate(futures.as_completed(future_map), 1):
            code, name = future_map[future]
            try:
                frame, error = future.result()
            except Exception as exc:
                frame, error = None, str(exc)[:600]
            if frame is not None and not frame.empty:
                frame.to_csv(parts_dir / f"{code}.csv", index=False, encoding="utf-8-sig")
                status = "success"
                row_count = len(frame)
                successful_codes.append(code)
                rows_this_run += row_count
                for source, count in frame["source"].value_counts().items():
                    provider_row_counts[str(source)] = provider_row_counts.get(str(source), 0) + int(count)
            else:
                status = "failed"
                row_count = 0
                failed_this_run += 1
            progress_rows.append({
                "requested_start": requested_start, "requested_end": requested_end,
                "code": code, "name": name, "status": status, "rows": row_count,
                "error": error, "attempted_at": datetime.now().isoformat(timespec="seconds"),
            })
            if completed_count % 25 == 0:
                flush_progress()
    flush_progress()
    target = IMPORTED / "daily-bars.csv"
    part_paths = sorted(parts_dir.glob("*.csv"))
    combined = (pd.concat((pd.read_csv(path, dtype={"code": str}) for path in part_paths), ignore_index=True)
                if part_paths and (successful_codes or not target.exists()) else pd.DataFrame())
    if not combined.empty:
        append_dedup(combined, target, ["date", "code"])
    elif not target.exists() and pending_universe:
        raise SystemExit("free history collection returned no rows; see history-progress.csv")
    if target.exists():
        archived_meta = pd.read_csv(target, dtype={"code": str}, usecols=["code", "source"])
        archived_provider_rows = {str(key): int(value) for key, value in archived_meta["source"].value_counts().items()}
        archived_rows = int(len(archived_meta))
        archived_codes = int(archived_meta["code"].nunique())
    else:
        archived_provider_rows, archived_rows, archived_codes = {}, 0, 0
    manifest = {
        "collected_at": datetime.now().isoformat(timespec="seconds"), "requested_start": requested_start,
        "requested_end": requested_end, "codes_requested": len(universe),
        "codes_already_complete": len(completed), "codes_attempted": len(pending_universe),
        "codes_returned_this_run": len(successful_codes),
        "codes_failed_this_run": failed_this_run,
        "rows_this_run": rows_this_run,
        "history_provider_rows_this_run": provider_row_counts,
        "history_provider_rows": archived_provider_rows,
        "archived_rows": archived_rows, "archived_codes": archived_codes,
        "universe_source": universe_source, "universe_complete": universe_complete,
        "point_in_time_status_complete": False,
        "survivorship_bias": not universe_complete,
        "warning": ("current-universe free history cannot include delisted names; historical ST/name status is unverified"
                    if not universe_complete else "historical ST/name status remains unverified"),
    }
    (IMPORTED / "daily-bars-manifest.json").write_text(json_text(manifest), encoding="utf-8")
    print(json_text(manifest))


IMPORT_REQUIRED = {
    "daily-bars": {"date", "code", "name", "open", "high", "low", "close", "pre_close", "st", "new_listing", "universe_complete"},
    "auction": {"signal_date", "trade_date", "asof", "code"},
    "themes": {"date", "code", "theme"},
    "announcements": {"code", "published_at", "risk_type"},
    "outcomes": {"signal_date", "code", "promoted_next_day"},
}


def validate_outcomes_import(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject internally inconsistent execution labels before they reach training."""
    checked = frame.copy()
    binary_fields = ("promoted_next_day", "false_strength", "large_loss", "sell_blocked", "rumor_only")
    for field in binary_fields:
        if field not in checked:
            continue
        values = pd.to_numeric(checked[field], errors="coerce")
        invalid = values.notna() & ~values.isin([0, 1])
        if invalid.any():
            raise ValueError(f"outcomes {field} must be 0/1 or blank; invalid rows={invalid[invalid].index.tolist()[:10]}")
        checked[field] = values
    for field in ("entry_price", "t1_close", "actual_exit_price"):
        if field in checked:
            checked[field] = pd.to_numeric(checked[field], errors="coerce")
            invalid = checked[field].notna() & (checked[field] <= 0)
            if invalid.any():
                raise ValueError(f"outcomes {field} must be positive")
    costs = (pd.to_numeric(checked["round_trip_cost"], errors="coerce").fillna(0.003)
             if "round_trip_cost" in checked else pd.Series(0.003, index=checked.index))
    if {"entry_price", "t1_close"}.issubset(checked.columns):
        available = checked["entry_price"].notna() & checked["t1_close"].notna()
        mark = checked["t1_close"] / checked["entry_price"] - 1 - costs
        checked["mark_to_market_return"] = np.where(available, mark, math.nan)
        if "large_loss" in checked:
            expected = (mark <= -0.07).astype(float)
            conflict = available & checked["large_loss"].notna() & (checked["large_loss"] != expected)
            if conflict.any():
                raise ValueError("outcomes large_loss conflicts with entry-to-T+1 close return")
    if "sell_blocked" in checked and "actual_exit_price" in checked and "net_return" in checked:
        net = pd.to_numeric(checked["net_return"], errors="coerce")
        unresolved = (checked["sell_blocked"] == 1) & checked["actual_exit_price"].isna() & net.notna()
        if unresolved.any():
            raise ValueError("sell_blocked rows without actual_exit_price must leave net_return blank")
    return checked


def import_command(args: argparse.Namespace) -> None:
    ensure_dirs()
    source = Path(args.file).resolve()
    if not source.exists():
        raise SystemExit(f"file not found: {source}")
    frame = pd.read_csv(source, dtype={"code": str, "代码": str})
    frame = frame.rename(columns={"代码": "code", "日期": "date", "名称": "name"})
    missing = IMPORT_REQUIRED[args.kind] - set(frame.columns)
    if missing:
        raise SystemExit(f"{args.kind} missing columns: {sorted(missing)}")
    if "code" in frame:
        frame["code"] = frame["code"].map(zfill_code)
    if args.kind == "outcomes":
        try:
            frame = validate_outcomes_import(frame)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    frame["imported_at"] = datetime.now().isoformat(timespec="seconds")
    frame["source_file"] = str(source)
    target = DATA / "open_snapshots.csv" if args.kind == "auction" else IMPORTED / f"{args.kind}.csv"
    keys = {
        "daily-bars": ["date", "code"], "auction": ["signal_date", "trade_date", "asof", "code"],
        "themes": ["date", "code"], "announcements": ["code", "published_at", "risk_type"],
        "outcomes": ["signal_date", "code"],
    }[args.kind]
    append_dedup(frame, target, keys)
    print(f"imported {len(frame)} rows -> {target}")


def append_dedup(frame: pd.DataFrame, path: Path, keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, dtype={"code": str})
        frame = pd.concat([old, frame], ignore_index=True, sort=False)
    frame = frame.drop_duplicates(keys, keep="last").sort_values(keys)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def pools_from_local_cache() -> dict[str, pd.DataFrame]:
    pools: dict[str, pd.DataFrame] = {}
    for path in sorted((ROOT / "data" / "2026-08").glob("zt_pool_*.csv")):
        date = pd.Timestamp(path.stem[-8:]).strftime("%Y-%m-%d")
        pools[date] = normalize_pool(pd.read_csv(path, dtype={"代码": str}), date)
    for day_dir in sorted(RAW.glob("20*")):
        path = day_dir / "limit_up.csv"
        if path.exists():
            date = pd.Timestamp(day_dir.name).strftime("%Y-%m-%d")
            pools[date] = normalize_pool(pd.read_csv(path, dtype={"代码": str}), date)
    return pools


def append_only_recent_pools(historical: dict[str, pd.DataFrame],
                             recent: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Append only dates beyond the frozen historical archive.

    Recent provider pools contain useful seal-time fields but must never silently
    rewrite the dates on which the trained model was built.
    """
    combined = dict(historical)
    historical_end = max(historical) if historical else ""
    for date, pool in recent.items():
        if not historical or str(date) > historical_end:
            combined[str(date)] = pool
    return combined


def prepare_daily_bars(path: Path) -> pd.DataFrame:
    """Normalize daily bars and derive only close-known, point-in-time-safe fields."""
    bars = pd.read_csv(path, dtype={"code": str})
    bars["date"] = pd.to_datetime(bars["date"]).dt.strftime("%Y-%m-%d")
    bars["code"] = bars["code"].map(zfill_code)
    for optional, default in (("st", False), ("new_listing", False), ("turnover", math.nan),
                              ("amount", math.nan), ("float_cap", math.nan), ("industry", "")):
        if optional not in bars:
            bars[optional] = default
    bars = bars[bars.apply(lambda r: is_mainboard(r["code"], r.get("name", ""), r["st"], r["new_listing"]), axis=1)]
    for field in ("open", "high", "low", "close", "pre_close", "turnover", "amount", "float_cap"):
        bars[field] = pd.to_numeric(bars.get(field), errors="coerce")
    if "limit_price" not in bars:
        bars["limit_price"] = bars["pre_close"].map(round_limit_price)
    else:
        bars["limit_price"] = pd.to_numeric(bars["limit_price"], errors="coerce")
    bars["down_limit_price"] = bars["pre_close"].map(lambda value: round_limit_price(value, -0.10))
    bars["is_limit"] = (pd.to_numeric(bars["close"], errors="coerce") - bars["limit_price"]).abs() <= 0.011
    bars["touched_limit"] = bars["high"] >= bars["limit_price"] - 0.011
    bars["failed_board_daily"] = bars["touched_limit"] & ~bars["is_limit"]
    bars["is_limit_down"] = (bars["close"] - bars["down_limit_price"]).abs() <= 0.011
    bars["one_word_daily"] = (
        bars[["open", "high", "low", "close"]].sub(bars["limit_price"], axis=0).abs().le(0.011).all(axis=1)
    )
    bars = bars.sort_values(["code", "date"])
    market_dates = sorted(bars["date"].unique())
    previous_market_date = {right: left for left, right in zip(market_dates, market_dates[1:])}
    streaks: list[int] = []
    previous_limits: list[bool] = []
    last_code, last_date, last_limit, streak = None, None, False, 0
    for row in bars.itertuples():
        if row.code != last_code:
            last_code, last_date, last_limit, streak = row.code, None, False, 0
        consecutive_session = last_date == previous_market_date.get(row.date)
        previous_limits.append(bool(consecutive_session and last_limit))
        streak = (streak + 1 if consecutive_session else 1) if row.is_limit else 0
        streaks.append(streak)
        last_date, last_limit = row.date, bool(row.is_limit)
    bars["boards"] = streaks
    bars["previous_is_limit"] = previous_limits
    return bars


def derive_daily_market_context(bars: pd.DataFrame) -> pd.DataFrame:
    """Infer historical loss-effect fields unavailable from recent pool APIs."""
    rows: list[dict[str, Any]] = []
    limit_counts = bars.groupby("date")["is_limit"].sum().to_dict()
    dates = sorted(bars["date"].unique())
    previous_date = {right: left for left, right in zip(dates, dates[1:])}
    for date, group in bars.groupby("date"):
        feedback = (group.loc[group["previous_is_limit"], "close"]
                    / group.loc[group["previous_is_limit"], "pre_close"] - 1).dropna()
        prior_count = int(limit_counts.get(previous_date.get(date, ""), 0))
        rows.append({
            "date": date,
            "failed_count": int(group["failed_board_daily"].sum()),
            "down_limit_count": int(group["is_limit_down"].sum()),
            "previous_limit_mean_return": float(feedback.mean()) if len(feedback) else math.nan,
            "previous_limit_median_return": float(feedback.median()) if len(feedback) else math.nan,
            # Backward-compatible feature name.  Its statistic is explicitly
            # the median and must not be rendered as an arithmetic average.
            "yesterday_premium": float(feedback.median()) if len(feedback) else math.nan,
            "previous_limit_tail_loss": float(feedback.quantile(0.10)) if len(feedback) else math.nan,
            "previous_limit_feedback_count": int(len(feedback)),
            "previous_limit_feedback_coverage": (len(feedback) / prior_count if prior_count else math.nan),
            "loss_fields_status": "inferred_from_unadjusted_daily_ohlc",
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def pools_from_prepared_daily_bars(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pools: dict[str, pd.DataFrame] = {}
    for date, group in bars[bars["is_limit"]].groupby("date"):
        frame = pd.DataFrame({
            "date": date, "code": group["code"], "name": group["name"], "price": group["close"],
            "pct_change": (group["close"] / group["pre_close"] - 1) * 100,
            "amount": group["amount"], "float_cap": group["float_cap"], "turnover": group["turnover"],
            "sealed_fund": math.nan,
            "first_seal": np.where(group["one_word_daily"], "092500", ""),
            "last_seal": np.where(group["one_word_daily"], "092500", ""),
            "bomb_count": np.where(group["one_word_daily"], 0.0, math.nan),
            "boards": group["boards"], "industry": group["industry"], "volume_ratio": math.nan,
            "amplitude": (group["high"] - group["low"]) / group["pre_close"] * 100,
        })
        pools[date] = frame.reset_index(drop=True)
    return pools


def pools_from_daily_bars(path: Path) -> dict[str, pd.DataFrame]:
    return pools_from_prepared_daily_bars(prepare_daily_bars(path))


def raw_count(date: str, kind: str) -> float:
    path = RAW / pd.Timestamp(date).strftime("%Y%m%d") / f"{kind}.csv"
    if not path.exists():
        return math.nan
    try:
        frame = pd.read_csv(path)
        if "代码" in frame:
            frame = frame[frame.apply(lambda r: is_mainboard(r.get("代码", ""), r.get("名称", "")), axis=1)]
        return float(len(frame))
    except Exception:
        return math.nan


def raw_previous_feedback(date: str) -> tuple[float, float, float]:
    path = RAW / pd.Timestamp(date).strftime("%Y%m%d") / "previous_limit.csv"
    if not path.exists():
        return math.nan, math.nan, math.nan
    try:
        frame = pd.read_csv(path, dtype={"代码": str})
        frame = frame[frame.apply(lambda r: is_mainboard(r.get("代码", ""), r.get("名称", "")), axis=1)]
        values = pd.to_numeric(frame.get("涨跌幅"), errors="coerce").dropna() / 100
        if values.empty:
            return math.nan, math.nan, math.nan
        return float(values.mean()), float(values.median()), float(values.quantile(0.10))
    except Exception:
        return math.nan, math.nan, math.nan


def load_optional(kind: str) -> pd.DataFrame:
    path = IMPORTED / f"{kind}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"code": str})


def promotion_rates(previous: pd.DataFrame | None, current: pd.DataFrame) -> dict[str, float]:
    defaults = {"promotion_overall": math.nan, "promotion_1to2": math.nan, "promotion_2to3": math.nan,
                "promotion_3to4": math.nan, "promotion_4to5": math.nan, "promotion_mid": math.nan,
                "space_feedback": math.nan}
    if previous is None or previous.empty:
        return defaults
    current_map = dict(zip(current["code"], current["boards"]))

    def rate(rows: pd.DataFrame) -> float:
        if rows.empty:
            return math.nan
        values = [safe_float(current_map.get(code)) == safe_float(board) + 1 for code, board in zip(rows["code"], rows["boards"])]
        return float(np.mean(values))

    eligible = previous[previous["boards"] >= 1]
    mid = previous[previous["boards"].between(2, 4)]
    prev_max = previous["boards"].max()
    space = previous[previous["boards"] == prev_max]
    result = {"promotion_overall": rate(eligible), "promotion_mid": rate(mid), "space_feedback": rate(space)}
    for board in range(1, 5):
        result[f"promotion_{board}to{board+1}"] = rate(previous[previous["boards"] == board])
    return result


def rolling_emotion_context(previous_rows: list[dict[str, Any]], current: dict[str, Any],
                            window: int = 60, minimum: int = 20) -> dict[str, float]:
    """Causal empirical percentiles using only market rows before the current close."""
    mapping = {
        "failed_rate": "failed_rate_percentile_60d",
        "down_limit_count": "down_limit_percentile_60d",
        "promotion_mid": "promotion_mid_percentile_60d",
        "yesterday_premium": "yesterday_premium_percentile_60d",
        "previous_limit_tail_loss": "tail_loss_percentile_60d",
    }
    history = previous_rows[-window:]
    result: dict[str, float] = {}
    for source, target in mapping.items():
        value = safe_float(current.get(source))
        values = np.array([safe_float(row.get(source)) for row in history], dtype=float)
        values = values[np.isfinite(values)]
        if math.isnan(value) or len(values) < minimum:
            result[target] = math.nan
            continue
        below = float((values < value).sum())
        equal = float(np.isclose(values, value, rtol=0, atol=1e-12).sum())
        result[target] = (below + 0.5 * equal) / len(values)
    return result


def emotion_dimensions(row: dict[str, Any]) -> dict[str, Any]:
    """Separate heat from relay quality, loss effect and crowding.

    All inputs are available at the signal-day close.  Missing observations stay
    missing inside each weighted component and are exposed through completeness.
    This prevents a busy first-board tape from masking a broken middle ladder.
    """
    previous_limit_tail_loss = safe_float(row.get("previous_limit_tail_loss"))
    if math.isnan(previous_limit_tail_loss):
        previous_limit_tail_loss = safe_float(row.get("failed_tail_loss"))
    failed_percentile = safe_float(row.get("failed_rate_percentile_60d"))
    down_percentile = safe_float(row.get("down_limit_percentile_60d"))
    premium_percentile = safe_float(row.get("yesterday_premium_percentile_60d"))
    tail_percentile = safe_float(row.get("tail_loss_percentile_60d"))
    breadth = weighted_available([
        (0.45, linear_score(row.get("limit_count"), 20, 80)),
        (0.25, linear_score(row.get("limit_change"), -20, 25)),
        (0.30, linear_score(row.get("seal_rate"), 0.65, 0.90)),
    ])
    relay = weighted_available([
        (0.22, linear_score(row.get("promotion_overall"), 0.12, 0.48)),
        (0.28, linear_score(row.get("promotion_mid"), 0.10, 0.45)),
        (0.18, clamp(row.get("space_feedback"))),
        (0.17, clamp(row.get("space_feedback_3d"))),
        (0.15, linear_score(row.get("yesterday_premium"), -0.03, 0.04)),
    ])
    ladder = weighted_available([
        (0.30, linear_score(row.get("promotion_mid"), 0.10, 0.45)),
        (0.20, 1 - linear_score(row.get("ladder_gap_count"), 0, 3)),
        (0.18, linear_score(row.get("middle_density"), 0.20, 0.60)),
        (0.14, clamp(row.get("unique_space"))),
        (0.18, linear_score(row.get("height_change"), -2, 1)),
    ])
    loss_health = weighted_available([
        (0.30, 1 - failed_percentile if not math.isnan(failed_percentile)
         else 1 - linear_score(row.get("failed_rate"), 0.10, 0.35)),
        (0.25, 1 - down_percentile if not math.isnan(down_percentile)
         else 1 - linear_score(row.get("down_limit_count"), 0, 10)),
        (0.25, tail_percentile if not math.isnan(tail_percentile)
         else linear_score(previous_limit_tail_loss, -0.10, 0.00)),
        (0.20, premium_percentile if not math.isnan(premium_percentile)
         else linear_score(row.get("yesterday_premium"), -0.03, 0.04)),
    ])
    crowding = weighted_available([
        (0.55, linear_score(row.get("one_word_share"), 0.15, 0.50)),
        (0.25, linear_score(row.get("limit_count"), 60, 100)),
        (0.20, linear_score(row.get("max_board"), 6, 9)),
    ])
    divergence = clamp(row.get("breadth_height_divergence"))
    fragility = weighted_available([
        (0.28, 1 - relay if not math.isnan(relay) else math.nan),
        (0.28, 1 - loss_health if not math.isnan(loss_health) else math.nan),
        (0.18, crowding),
        (0.14, divergence),
        (0.12, linear_score(row.get("ladder_gap_count"), 0, 3)),
    ])

    red_flags: list[str] = []
    checks = [
        (safe_float(row.get("promotion_mid")) < 0.20, "中位晋级不足20%", row.get("promotion_mid")),
        (safe_float(row.get("failed_rate")) >= 0.22
         and (math.isnan(failed_percentile) or failed_percentile >= 0.75),
         "炸板率处于近60日高位", row.get("failed_rate")),
        (previous_limit_tail_loss <= -0.05, "昨日涨停股尾部亏损恶化", previous_limit_tail_loss),
        (safe_float(row.get("ladder_gap_count")) >= 2, "梯队断层不少于2层", row.get("ladder_gap_count")),
        (safe_float(row.get("space_feedback_3d")) < 0.40, "近期空间板反馈偏弱", row.get("space_feedback_3d")),
        (safe_float(row.get("one_word_share")) >= 0.45, "一字拥挤", row.get("one_word_share")),
    ]
    for condition, label, raw in checks:
        if not math.isnan(safe_float(raw)) and condition:
            red_flags.append(label)
    height_rising = safe_float(row.get("height_change")) > 0
    false_acceleration = float(height_rising and len(red_flags) >= 2)

    required = (
        "limit_count", "limit_change", "seal_rate", "promotion_overall", "promotion_mid",
        "space_feedback", "space_feedback_3d", "failed_rate", "down_limit_count",
        "previous_limit_tail_loss", "yesterday_premium", "one_word_share",
    )
    observed = sum(not math.isnan(safe_float(row.get(name))) for name in required)
    completeness = observed / len(required)
    quality = weighted_available([
        (0.15, breadth), (0.30, relay), (0.22, ladder), (0.25, loss_health),
        (0.08, 1 - crowding if not math.isnan(crowding) else math.nan),
    ])
    risk_level = "HIGH" if false_acceleration or (not math.isnan(fragility) and fragility >= 0.68) else (
        "MEDIUM" if not math.isnan(fragility) and fragility >= 0.45 else "LOW")
    return {
        "breadth_health": breadth, "relay_appetite": relay, "ladder_health": ladder,
        "loss_effect_health": loss_health, "crowding_risk": crowding,
        "regime_fragility": fragility, "false_acceleration": false_acceleration,
        "emotion_data_completeness": completeness, "emotion_quality_score": quality,
        "emotion_risk_level": risk_level, "emotion_warnings": "；".join(red_flags),
    }


def classify_emotion(row: dict[str, Any], previous_state: str | None = None) -> str:
    profile = emotion_dimensions(row)
    hchg = safe_float(row.get("height_change"))
    lchg = safe_float(row.get("limit_change"))
    promo_mid = safe_float(row.get("promotion_mid"))
    feedback = safe_float(row.get("space_feedback"))
    failed = safe_float(row.get("failed_rate"))
    down = safe_float(row.get("down_limit_count"))
    limits = safe_float(row.get("limit_count"))
    one_words = safe_float(row.get("one_word_share"))
    premium = safe_float(row.get("yesterday_premium"))
    failed_percentile = safe_float(row.get("failed_rate_percentile_60d"))
    down_percentile = safe_float(row.get("down_limit_percentile_60d"))
    promotion_percentile = safe_float(row.get("promotion_mid_percentile_60d"))
    premium_percentile = safe_float(row.get("yesterday_premium_percentile_60d"))
    loss_health = safe_float(profile.get("loss_effect_health"))
    fragility = safe_float(profile.get("regime_fragility"))
    false_acceleration = safe_float(profile.get("false_acceleration")) == 1
    promotion_weak = (not math.isnan(promo_mid) and promo_mid < 0.25
                      if math.isnan(promotion_percentile)
                      else promo_mid < 0.35 and promotion_percentile <= 0.25)
    down_pressure = (not math.isnan(down) and down >= 5
                     if math.isnan(down_percentile)
                     else down >= 3 and down_percentile >= 0.75)
    premium_weak = (not math.isnan(premium) and premium < 0
                    if math.isnan(premium_percentile)
                    else premium < 0.01 and premium_percentile <= 0.25)
    failed_pressure = (not math.isnan(failed) and failed >= 0.25
                       if math.isnan(failed_percentile)
                       else failed >= 0.22 and failed_percentile >= 0.75)
    weak_confirmations = sum((
        promotion_weak,
        not math.isnan(loss_health) and loss_health < 0.40,
        down_pressure,
        premium_weak,
    ))
    if (not math.isnan(hchg) and hchg <= -2 and weak_confirmations >= 1) or (
            feedback == 0 and not math.isnan(promo_mid) and promo_mid < 0.15 and not math.isnan(down) and down >= 5) or (
            not math.isnan(loss_health) and loss_health < 0.25 and not math.isnan(hchg) and hchg < 0):
        return "退潮"
    if false_acceleration or (failed_pressure and weak_confirmations >= 1) or (
            not math.isnan(hchg) and hchg < 0 and not math.isnan(promo_mid) and promo_mid < 0.25) or (
            not math.isnan(fragility) and fragility >= 0.72 and not math.isnan(loss_health) and loss_health < 0.35):
        return "大分歧"
    if (previous_state in {"退潮", "大分歧"} and not math.isnan(lchg) and lchg > 0 and feedback == 1
            and (math.isnan(loss_health) or loss_health >= 0.40)):
        return "修复"
    if (not math.isnan(one_words) and one_words >= 0.35 and not math.isnan(limits) and limits >= 50
            and not math.isnan(hchg) and hchg >= 0):
        return "一致"
    if (not math.isnan(hchg) and hchg > 0 and not math.isnan(promo_mid) and promo_mid >= 0.25
            and not false_acceleration and (math.isnan(loss_health) or loss_health >= 0.40)):
        return "加速"
    if not math.isnan(lchg) and lchg >= 15 and (math.isnan(hchg) or hchg >= 0):
        return "启动"
    score = safe_float(row.get("emotion_score"))
    if math.isnan(score):
        score = safe_float(profile.get("emotion_quality_score"))
    return "升温" if score >= 0.58 else "修复"


def attach_online_emotion_forecast(market: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    """Attach causal Markov and expanding-window adverse-regime forecasts.

    At close t, the transition t-1 -> t is known, while t -> t+1 is never used.
    A supervised challenger starts only after 60 training and 20 calibration
    transitions.  It replaces the Markov baseline only after 60 earlier causal
    forecasts and a material Brier/log-loss improvement.
    """
    if market.empty:
        return market.copy()
    ordered = market.sort_values("date").copy()
    counts = {state: {other: 0 for other in STATE_NAMES} for state in STATE_NAMES}
    previous_state: str | None = None
    state_age = 0
    state_features = {
        "state_start": "启动", "state_warming": "升温", "state_acceleration": "加速",
        "state_consensus": "一致", "state_divergence": "大分歧", "state_retreat": "退潮",
        "state_repair": "修复",
    }
    for index, row in ordered.iterrows():
        state = str(row["emotion_state"])
        if state not in counts:
            for column in STATE_PROBABILITY_COLUMNS.values():
                ordered.at[index, column] = math.nan
            continue
        if previous_state in counts:
            counts[previous_state][state] += 1
        state_age = state_age + 1 if state == previous_state else 1
        evidence = sum(counts[state].values())
        denominator = evidence + alpha * len(STATE_NAMES)
        probabilities = {
            target: (counts[state][target] + alpha) / denominator
            for target in STATE_NAMES
        }
        for target, column in STATE_PROBABILITY_COLUMNS.items():
            ordered.at[index, column] = probabilities[target]
        entropy = -sum(probability * math.log(probability) for probability in probabilities.values())
        normalized_entropy = entropy / math.log(len(STATE_NAMES))
        completeness = safe_float(row.get("emotion_data_completeness"))
        status = ("USABLE" if evidence >= 20 and (math.isnan(completeness) or completeness >= 0.75)
                  else "LOW_EVIDENCE")
        markov_adverse = probabilities["大分歧"] + probabilities["退潮"]
        ordered.at[index, "emotion_adverse_markov_probability"] = markov_adverse
        ordered.at[index, "emotion_adverse_next_probability"] = markov_adverse
        ordered.at[index, "emotion_adverse_challenger_probability"] = math.nan
        ordered.at[index, "emotion_forecast_source"] = "MARKOV_BASELINE"
        ordered.at[index, "emotion_challenger_status"] = "INSUFFICIENT_HISTORY"
        ordered.at[index, "emotion_champion_evaluation_n"] = 0
        ordered.at[index, "emotion_challenger_vs_markov_brier_skill"] = math.nan
        ordered.at[index, "emotion_supportive_next_probability"] = sum(
            probabilities[target] for target in ACTIVE_STATES)
        ordered.at[index, "emotion_persistence_probability"] = probabilities[state]
        ordered.at[index, "emotion_transition_entropy"] = normalized_entropy
        ordered.at[index, "emotion_transition_evidence"] = evidence
        ordered.at[index, "emotion_transition_evidence_scaled"] = min(1.0, evidence / 60.0)
        ordered.at[index, "emotion_transition_confidence"] = (
            "LOW" if evidence < 20 else ("MEDIUM" if evidence < 60 else "HIGH"))
        ordered.at[index, "emotion_forecast_status"] = status
        ordered.at[index, "emotion_state_age"] = state_age
        ordered.at[index, "emotion_state_age_scaled"] = min(1.0, state_age / 5.0)
        for feature, label in state_features.items():
            ordered.at[index, feature] = float(state == label)
        previous_state = state

    # The label for row s becomes observable only at the close of row s+1.
    adverse_labels = ordered["emotion_state"].shift(-1).isin({"大分歧", "退潮"}).astype(float)
    adverse_labels.iloc[-1] = math.nan
    ordered["_adverse_next_label"] = adverse_labels
    for position in range(len(ordered)):
        available = ordered.iloc[:position].dropna(subset=["_adverse_next_label"]).copy()
        if len(available) < 80:
            continue
        training, calibration = available.iloc[:-20].copy(), available.iloc[-20:].copy()
        if (training["_adverse_next_label"].value_counts().min() < 8
                or calibration["_adverse_next_label"].nunique() < 2
                or calibration["_adverse_next_label"].value_counts().min() < 3):
            ordered.iloc[position, ordered.columns.get_loc("emotion_challenger_status")] = "CLASS_IMBALANCE"
            continue
        model = RegularizedLogistic.fit(
            training, "_adverse_next_label", EMOTION_ADVERSE_MODEL_FEATURES,
            l2=0.25, epochs=800, lr=0.025)
        calibration_raw = model.predict(calibration)
        calibrator = PlattCalibrator.fit(
            calibration_raw, calibration["_adverse_next_label"].to_numpy(float))
        challenger_probability = float(calibrator.predict(model.predict(ordered.iloc[[position]]))[0])
        row_index = ordered.index[position]
        ordered.at[row_index, "emotion_adverse_challenger_probability"] = challenger_probability
        ordered.at[row_index, "emotion_challenger_status"] = "TRAINED_CAUSAL"

        earlier = ordered.iloc[:position].dropna(
            subset=["_adverse_next_label", "emotion_adverse_challenger_probability"])
        evaluation_n = len(earlier)
        ordered.at[row_index, "emotion_champion_evaluation_n"] = evaluation_n
        if evaluation_n < 60 or earlier["_adverse_next_label"].nunique() < 2:
            continue
        y = earlier["_adverse_next_label"].to_numpy(float)
        challenger = earlier["emotion_adverse_challenger_probability"].to_numpy(float)
        markov = earlier["emotion_adverse_markov_probability"].to_numpy(float)
        challenger_brier = float(np.mean((challenger - y) ** 2))
        markov_brier = float(np.mean((markov - y) ** 2))
        relative_skill = 1 - challenger_brier / markov_brier if markov_brier > 0 else math.nan
        challenger_logloss = float(-np.mean(
            y * np.log(np.clip(challenger, 1e-6, 1))
            + (1 - y) * np.log(np.clip(1 - challenger, 1e-6, 1))))
        markov_logloss = float(-np.mean(
            y * np.log(np.clip(markov, 1e-6, 1))
            + (1 - y) * np.log(np.clip(1 - markov, 1e-6, 1))))
        challenger_ece = expected_calibration_error(y, challenger)
        markov_ece = expected_calibration_error(y, markov)
        ordered.at[row_index, "emotion_challenger_vs_markov_brier_skill"] = relative_skill
        champion = (not math.isnan(relative_skill) and relative_skill >= 0.05
                    and challenger_logloss < markov_logloss
                    and challenger_ece <= markov_ece + 0.02)
        if champion:
            ordered.at[row_index, "emotion_adverse_next_probability"] = challenger_probability
            ordered.at[row_index, "emotion_forecast_source"] = "SUPERVISED_CHALLENGER"
    return ordered.drop(columns=["_adverse_next_label"]).sort_index()


def build_market_table(pools: dict[str, pd.DataFrame], daily_context: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    context_map = ({str(row["date"]): row for _, row in daily_context.iterrows()}
                   if daily_context is not None and not daily_context.empty else {})
    previous: pd.DataFrame | None = None
    previous_state: str | None = None
    space_history: list[float] = []
    for date in sorted(pools):
        pool = pools[date]
        ladder = pool[pool["boards"] >= 2]
        max_board = int(ladder["boards"].max()) if not ladder.empty else 1
        prev_row = rows[-1] if rows else None
        promos = promotion_rates(previous, pool)
        context = context_map.get(date, {})
        failed_count = raw_count(date, "failed_board")
        down_count = raw_count(date, "limit_down")
        previous_mean_return, yesterday_premium, failed_tail_loss = raw_previous_feedback(date)
        loss_fields_status = "verified_recent_pool"
        if math.isnan(failed_count):
            failed_count = safe_float(context.get("failed_count"))
            loss_fields_status = str(context.get("loss_fields_status", "unavailable"))
        if math.isnan(down_count):
            down_count = safe_float(context.get("down_limit_count"))
        if math.isnan(yesterday_premium):
            yesterday_premium = safe_float(context.get("yesterday_premium"))
        if math.isnan(previous_mean_return):
            previous_mean_return = safe_float(context.get("previous_limit_mean_return"))
        if math.isnan(failed_tail_loss):
            failed_tail_loss = safe_float(context.get("previous_limit_tail_loss"))
        limit_count = float(len(pool))
        seal_rate = limit_count / (limit_count + failed_count) if not math.isnan(failed_count) and limit_count + failed_count else math.nan
        failed_rate = failed_count / (limit_count + failed_count) if not math.isnan(failed_count) and limit_count + failed_count else math.nan
        middle = int(ladder["boards"].between(2, 4).sum())
        unique_space = float((ladder["boards"] == max_board).sum() == 1) if not ladder.empty else 0.0
        existing_levels = set(int(x) for x in ladder["boards"])
        gaps = sum(level not in existing_levels for level in range(2, max_board + 1))
        one_word_values = []
        for _, item in ladder.iterrows():
            first, last = parse_clock(item.get("first_seal")), parse_clock(item.get("last_seal"))
            bombs, turnover = safe_float(item.get("bomb_count")), safe_float(item.get("turnover"))
            locked = (not math.isnan(first) and first <= 565.02 and not math.isnan(last) and last <= 565.02
                      and (math.isnan(bombs) or bombs == 0) and (math.isnan(turnover) or turnover < 3))
            one_word_values.append(float(locked))
        height_change = max_board - int(prev_row["max_board"]) if prev_row else math.nan
        limit_change = limit_count - float(prev_row["limit_count"]) if prev_row else math.nan
        middle_density = middle / max(1, len(ladder))
        loss_health = weighted_available([
            (0.55, 1 - clamp((failed_rate if not math.isnan(failed_rate) else math.nan) / 0.30)),
            (0.45, 1 - clamp((down_count if not math.isnan(down_count) else math.nan) / 10)),
        ])
        emotion_score = weighted_available([
            (0.12, clamp((limit_count - 20) / 80)), (0.12, clamp((max_board - 2) / 6)),
            (0.08, clamp((height_change + 3) / 6)), (0.17, promos["promotion_overall"]),
            (0.12, promos["promotion_mid"]), (0.13, loss_health),
            (0.11, clamp(middle_density)), (0.08, unique_space), (0.07, promos["space_feedback"]),
        ])
        space_history.append(promos["space_feedback"])
        valid_space = [x for x in space_history[-3:] if not math.isnan(x)]
        row = {
            "date": date, "source": "Eastmoney/AkShare pool or imported daily bars",
            "source_timestamp": f"{date} 15:00:00", "data_status": "verified" if not math.isnan(failed_count) else "partial",
            "limit_count": limit_count, "consecutive_count": int(len(ladder)), "max_board": max_board,
            "failed_count": failed_count, "down_limit_count": down_count, "seal_rate": seal_rate,
            "failed_rate": failed_rate, "height_change": height_change, "limit_change": limit_change,
            "middle_density": middle_density, "ladder_gap_count": gaps, "unique_space": unique_space,
            "one_word_share": float(np.mean(one_word_values)) if one_word_values else math.nan,
            "space_feedback_3d": float(np.mean(valid_space)) if valid_space else math.nan,
            "breadth_height_divergence": float(not math.isnan(height_change) and height_change > 0 and not math.isnan(limit_change) and limit_change < -20),
            "previous_limit_mean_return": previous_mean_return,
            "previous_limit_median_return": yesterday_premium,
            # Legacy model feature retained with its original median statistic.
            "yesterday_premium": yesterday_premium,
            # Backward-compatible alias retained; the source is yesterday's entire
            # limit-up pool, so "previous_limit_tail_loss" is the precise name.
            "previous_limit_tail_loss": failed_tail_loss, "failed_tail_loss": failed_tail_loss, **promos,
            "previous_limit_feedback_count": safe_float(context.get("previous_limit_feedback_count")),
            "previous_limit_feedback_coverage": safe_float(context.get("previous_limit_feedback_coverage")),
            "loss_fields_status": loss_fields_status,
        }
        row.update(rolling_emotion_context(rows, row))
        row["emotion_heat_score"] = emotion_score
        profile = emotion_dimensions(row)
        row.update(profile)
        row["emotion_score"] = (profile["emotion_quality_score"]
                                if not math.isnan(safe_float(profile["emotion_quality_score"]))
                                else emotion_score)
        row["emotion_state"] = classify_emotion(row, previous_state)
        missing = [k for k in ("failed_count", "down_limit_count", "seal_rate", "yesterday_premium", "previous_limit_tail_loss") if math.isnan(safe_float(row[k]))]
        completeness = safe_float(row.get("emotion_data_completeness"))
        row["data_status"] = "verified" if completeness >= 0.90 else ("partial" if completeness >= 0.50 else "sparse")
        row["unavailable_fields"] = ",".join(missing)
        rows.append(row)
        previous, previous_state = pool, row["emotion_state"]
    return attach_online_emotion_forecast(pd.DataFrame(rows))


def attach_state_features(row: dict[str, Any], state: str) -> None:
    mapping = {
        "state_start": "启动", "state_warming": "升温", "state_acceleration": "加速",
        "state_consensus": "一致", "state_divergence": "大分歧", "state_retreat": "退潮",
        "state_repair": "修复",
    }
    for feature, label in mapping.items():
        row[feature] = float(state == label)


def load_context_overlay(date: str) -> dict[str, dict[str, Any]]:
    path = ROOT / f"current_snapshot_{date}.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype={"code": str})
    return {zfill_code(r["code"]): r.to_dict() for _, r in frame.iterrows()}


def announcement_features(announcements: pd.DataFrame, code: str, date: str) -> dict[str, float]:
    empty = {"announcement_risk": math.nan, "reduction_risk": math.nan,
             "unlock_risk": math.nan, "earnings_risk": math.nan}
    if announcements.empty:
        return empty
    data = announcements.copy()
    data["code"] = data["code"].map(zfill_code)
    data["published_at"] = pd.to_datetime(data["published_at"], errors="coerce")
    cutoff = pd.Timestamp(f"{date} 15:00:00")
    start = cutoff - pd.Timedelta(days=45)
    data = data[(data["code"] == code) & data["published_at"].between(start, cutoff)]
    if data.empty:
        # Presence of a supplied announcement dataset is treated as checked coverage.
        return {key: 0.0 for key in empty}
    kinds = set(data["risk_type"].astype(str).str.lower())
    reduction = float(any("减持" in kind or "reduction" in kind for kind in kinds))
    unlock = float(any("解禁" in kind or "unlock" in kind for kind in kinds))
    earnings = float(any(any(token in kind for token in ("业绩雷", "亏损", "earnings")) for kind in kinds))
    hard = float(any(any(token in kind for token in ("停牌核查", "风险提示", "监管", "处罚", "suspend", "regulatory")) for kind in kinds))
    return {"announcement_risk": max(hard, reduction * 0.8, earnings * 0.8, unlock * 0.5),
            "reduction_risk": reduction, "unlock_risk": unlock, "earnings_risk": earnings}


def build_candidate_table(pools: dict[str, pd.DataFrame], market: pd.DataFrame) -> pd.DataFrame:
    themes = load_optional("themes")
    outcomes_import = load_optional("outcomes")
    announcements = load_optional("announcements")
    theme_map = {}
    if not themes.empty:
        themes["date"] = pd.to_datetime(themes["date"]).dt.strftime("%Y-%m-%d")
        theme_map = {(r["date"], zfill_code(r["code"])): str(r["theme"]) for _, r in themes.iterrows()}
    outcome_map = {}
    if not outcomes_import.empty:
        outcome_map = {(str(r["signal_date"]), zfill_code(r["code"])): r.to_dict() for _, r in outcomes_import.iterrows()}
    market_map = market.set_index("date").to_dict("index")
    dates = sorted(pools)
    rows: list[dict[str, Any]] = []
    streaks: dict[str, int] = {}
    previous_turnover: dict[str, float] = {}
    for index, date in enumerate(dates):
        pool = pools[date]
        previous_pool = pools[dates[index - 1]] if index > 0 else None
        ladder = pool[pool["boards"] >= 2].copy()
        next_pool = pools[dates[index + 1]] if index + 1 < len(dates) else None
        after_pool = pools[dates[index + 2]] if index + 2 < len(dates) else None
        next_map = {} if next_pool is None else next_pool.set_index("code").to_dict("index")
        after_map = {} if after_pool is None else after_pool.set_index("code").to_dict("index")
        m = market_map[date]
        max_board = int(m["max_board"])
        overlay = load_context_overlay(date)
        industry_counts = pool["industry"].fillna("").value_counts().to_dict()
        height_counts = ladder["boards"].value_counts().to_dict()
        theme_counts: dict[str, int] = {}
        industry_themes: dict[str, list[str]] = {}
        for _, item in ladder.iterrows():
            theme = theme_map.get((date, item["code"]), "")
            if theme:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
                industry_themes.setdefault(str(item.get("industry", "")), []).append(theme)
        for _, item in ladder.iterrows():
            code, boards = item["code"], int(item["boards"])
            first, last = parse_clock(item.get("first_seal")), parse_clock(item.get("last_seal"))
            bombs, turnover = safe_float(item.get("bomb_count")), safe_float(item.get("turnover"))
            locked = (not math.isnan(first) and first <= 565.02 and not math.isnan(last) and last <= 565.02
                      and (math.isnan(bombs) or bombs == 0) and (math.isnan(turnover) or turnover < 3))
            streaks[code] = streaks.get(code, 0) + 1 if locked else 0
            theme = theme_map.get((date, code), "")
            industry = str(item.get("industry", ""))
            same_theme = theme_counts.get(theme, 0) if theme else math.nan
            theme_middle = math.nan
            theme_first = math.nan
            theme_promotion = math.nan
            if theme:
                themed = pool[pool["code"].map(lambda c: theme_map.get((date, c), "")) == theme]
                theme_middle = int(themed["boards"].between(2, 4).sum())
                theme_first = int((themed["boards"] == 1).sum())
                if previous_pool is not None:
                    previous_themed = previous_pool[previous_pool["code"].map(
                        lambda c: theme_map.get((dates[index - 1], c), "")) == theme]
                    if not previous_themed.empty:
                        current_map_for_theme = dict(zip(pool["code"], pool["boards"]))
                        theme_promotion = float(np.mean([
                            safe_float(current_map_for_theme.get(c)) == safe_float(b) + 1
                            for c, b in zip(previous_themed["code"], previous_themed["boards"])
                        ]))
            next_item = next_map.get(code)
            promoted = math.nan if next_pool is None else float(next_item is not None and int(next_item["boards"]) == boards + 1)
            imported = outcome_map.get((date, code), {})
            if "promoted_next_day" in imported:
                promoted = safe_float(imported["promoted_next_day"])
            next_open = safe_float(imported.get("next_open_gap"))
            entry_price = safe_float(imported.get("entry_price"))
            t1_close = safe_float(imported.get("t1_close"))
            actual_exit_price = safe_float(imported.get("actual_exit_price"))
            round_trip_cost = safe_float(imported.get("round_trip_cost"))
            if math.isnan(round_trip_cost):
                round_trip_cost = 0.003
            mark_to_market_return = (t1_close / entry_price - 1 - round_trip_cost
                                     if entry_price > 0 and t1_close > 0 else math.nan)
            sell_blocked = safe_float(imported.get("sell_blocked"))
            net_return = safe_float(imported.get("net_return"))
            if math.isnan(net_return) and entry_price > 0 and actual_exit_price > 0:
                net_return = actual_exit_price / entry_price - 1 - round_trip_cost
            elif math.isnan(net_return) and sell_blocked != 1 and not math.isnan(mark_to_market_return):
                net_return = mark_to_market_return
            false_strength = safe_float(imported.get("false_strength"))
            if math.isnan(false_strength) and not math.isnan(next_open) and not math.isnan(promoted):
                false_strength = float(next_open >= 0.05 and promoted == 0)
            large_loss = safe_float(imported.get("large_loss"))
            if math.isnan(large_loss) and not math.isnan(mark_to_market_return):
                large_loss = float(mark_to_market_return <= -0.07)
            if sell_blocked == 1 and math.isnan(actual_exit_price):
                net_return = math.nan
            execution_status = ("verified_import" if entry_price > 0 and t1_close > 0 else "unavailable")
            if sell_blocked == 1 and math.isnan(actual_exit_price):
                execution_status = "sell_blocked_unresolved"
            amount, float_cap = safe_float(item.get("amount")), safe_float(item.get("float_cap"))
            sealed = safe_float(item.get("sealed_fund"))
            prev_turn = previous_turnover.get(code, math.nan)
            context = overlay.get(code, {})
            announcement = announcement_features(announcements, code, date)
            sector_ladder = safe_float(context.get("sector_ladder"))
            identity_score = safe_float(context.get("identity"))
            row = {
                "date": date, "trade_date": dates[index + 1] if index + 1 < len(dates) else "",
                "code": code, "name": item["name"], "industry": industry, "theme": theme,
                "theme_status": "verified" if theme else "unavailable", "boards": boards,
                "height_gap": max_board - boards, "is_space_leader": float(boards == max_board),
                "unique_space": float(boards == max_board and height_counts.get(boards, 0) == 1),
                "height_share": boards / max(1, max_board), "promoted_next_day": promoted,
                "promotion_label_status": (
                    "verified_import" if "promoted_next_day" in imported
                    else "verified_from_next_daily_limit_close" if next_pool is not None else "unavailable"),
                "false_strength": false_strength, "large_loss": large_loss,
                "next_open_gap": next_open,
                "mark_to_market_return": mark_to_market_return, "net_return": net_return,
                "entry_price": entry_price, "t1_close": t1_close,
                "actual_exit_price": actual_exit_price, "round_trip_cost": round_trip_cost,
                "sell_blocked": sell_blocked, "execution_status": execution_status,
                "false_strength_label_status": (
                    "verified_import" if "false_strength" in imported and not math.isnan(safe_float(imported.get("false_strength")))
                    else "inferred_from_imported_fields" if not math.isnan(false_strength) else "unavailable"),
                "large_loss_label_status": (
                    "verified_import" if "large_loss" in imported and not math.isnan(safe_float(imported.get("large_loss")))
                    else "inferred_from_imported_fields" if not math.isnan(large_loss) else "unavailable"),
                "execution_source": "user_import" if execution_status != "unavailable" else "unavailable",
                "execution_assumption": "verified user-supplied outcome" if execution_status != "unavailable" else "",
                "emotion_state": m["emotion_state"], "emotion_score": m["emotion_score"],
                "limit_count_scaled": clamp(m["limit_count"] / 100),
                "consecutive_scaled": clamp(m["consecutive_count"] / 30),
                "down_limit_rate": clamp(safe_float(m["down_limit_count"]) / 10),
                "failed_rate": m["failed_rate"], "seal_rate": m["seal_rate"],
                "height_change_scaled": clamp((safe_float(m["height_change"]) + 3) / 6),
                "limit_change_scaled": clamp((safe_float(m["limit_change"]) + 80) / 160),
                "promotion_overall": m["promotion_overall"], "promotion_1to2": m["promotion_1to2"],
                "promotion_2to3": m["promotion_2to3"], "promotion_3to4": m["promotion_3to4"],
                "promotion_4to5": m["promotion_4to5"], "promotion_mid": m["promotion_mid"],
                "space_feedback": m["space_feedback"], "space_feedback_3d": m["space_feedback_3d"],
                "middle_density": m["middle_density"], "ladder_gap_count": m["ladder_gap_count"],
                "breadth_height_divergence": m["breadth_height_divergence"],
                "yesterday_premium": m["yesterday_premium"], "failed_tail_loss": m["failed_tail_loss"],
                "previous_limit_tail_loss": m.get("previous_limit_tail_loss", m["failed_tail_loss"]),
                "breadth_health": m.get("breadth_health", math.nan),
                "relay_appetite": m.get("relay_appetite", math.nan),
                "ladder_health": m.get("ladder_health", math.nan),
                "loss_effect_health": m.get("loss_effect_health", math.nan),
                "crowding_risk": m.get("crowding_risk", math.nan),
                "regime_fragility": m.get("regime_fragility", math.nan),
                "false_acceleration": m.get("false_acceleration", math.nan),
                "emotion_data_completeness": m.get("emotion_data_completeness", math.nan),
                "emotion_adverse_next_probability": m.get("emotion_adverse_next_probability", math.nan),
                "emotion_adverse_markov_probability": m.get("emotion_adverse_markov_probability", math.nan),
                "emotion_adverse_challenger_probability": m.get("emotion_adverse_challenger_probability", math.nan),
                "emotion_forecast_source": m.get("emotion_forecast_source", "MARKOV_BASELINE"),
                "emotion_challenger_status": m.get("emotion_challenger_status", "INSUFFICIENT_HISTORY"),
                "emotion_champion_evaluation_n": m.get("emotion_champion_evaluation_n", 0),
                "emotion_challenger_vs_markov_brier_skill": m.get(
                    "emotion_challenger_vs_markov_brier_skill", math.nan),
                "emotion_supportive_next_probability": m.get("emotion_supportive_next_probability", math.nan),
                "emotion_persistence_probability": m.get("emotion_persistence_probability", math.nan),
                "emotion_transition_entropy": m.get("emotion_transition_entropy", math.nan),
                "emotion_transition_evidence": m.get("emotion_transition_evidence", 0),
                "emotion_transition_evidence_scaled": m.get("emotion_transition_evidence_scaled", 0),
                "emotion_transition_confidence": m.get("emotion_transition_confidence", "LOW"),
                "emotion_forecast_status": m.get("emotion_forecast_status", "LOW_EVIDENCE"),
                "emotion_state_age": m.get("emotion_state_age", 1),
                "emotion_state_age_scaled": m.get("emotion_state_age_scaled", 0.2),
                "failed_rate_percentile_60d": m.get("failed_rate_percentile_60d", math.nan),
                "down_limit_percentile_60d": m.get("down_limit_percentile_60d", math.nan),
                "promotion_mid_percentile_60d": m.get("promotion_mid_percentile_60d", math.nan),
                "yesterday_premium_percentile_60d": m.get("yesterday_premium_percentile_60d", math.nan),
                "tail_loss_percentile_60d": m.get("tail_loss_percentile_60d", math.nan),
                "emotion_risk_level": m.get("emotion_risk_level", "UNKNOWN"),
                "emotion_warnings": m.get("emotion_warnings", ""),
                "theme_support": same_theme, "theme_middle_support": theme_middle,
                "theme_first_support": theme_first, "theme_promotion": theme_promotion,
                "theme_concentration": same_theme / max(1, len(pool)) if theme else math.nan,
                "theme_fragmentation": (len(set(industry_themes.get(industry, []))) / max(1, len(industry_themes.get(industry, []))))
                if theme and industry_themes.get(industry) else safe_float(context.get("sector_divergence")),
                "same_height_rivals": max(0, int(height_counts.get(boards, 0)) - 1),
                "capacity_anchor_health": safe_float(context.get("anchor_health")),
                "industry_support_proxy": int(industry_counts.get(industry, 0)) if industry else math.nan,
                "turnover": turnover / 100 if not math.isnan(turnover) else math.nan,
                "signal_amount": amount, "signal_float_cap": float_cap,
                "turnover_vs_prev": turnover / prev_turn if not math.isnan(turnover) and not math.isnan(prev_turn) and prev_turn else math.nan,
                "amount_float_cap": amount / float_cap if not math.isnan(amount) and not math.isnan(float_cap) and float_cap else math.nan,
                "sealed_fund_float_cap": sealed / float_cap if not math.isnan(sealed) and not math.isnan(float_cap) and float_cap else math.nan,
                "first_seal_minute": (first - 565) / 240 if not math.isnan(first) else math.nan,
                "last_seal_minute": (last - 565) / 240 if not math.isnan(last) else math.nan,
                "bomb_count": bombs, "one_word_proxy": float(locked), "one_word_streak": streaks[code],
                "volume_ratio": safe_float(item.get("volume_ratio")), "amplitude": safe_float(item.get("amplitude")),
                "relative_strength": safe_float(item.get("pct_change")) / 10,
                "price": safe_float(item.get("price")),
                "float_cap_log": math.log(max(float_cap, 1)) if not math.isnan(float_cap) else math.nan,
                "announcement_risk": announcement["announcement_risk"] if not math.isnan(announcement["announcement_risk"]) else (
                    1 - safe_float(context.get("announcement_safety")) if not math.isnan(safe_float(context.get("announcement_safety"))) else math.nan),
                "abnormal_10d_distance": safe_float(imported.get("abnormal_10d_distance")),
                "abnormal_30d_distance": safe_float(imported.get("abnormal_30d_distance")),
                "reduction_risk": announcement["reduction_risk"], "unlock_risk": announcement["unlock_risk"],
                "earnings_risk": announcement["earnings_risk"],
                "rumor_only": safe_float(imported.get("rumor_only")),
                "crowding_score": safe_float(imported.get("crowding_score")),
                "sentiment_unanimity": safe_float(imported.get("sentiment_unanimity")),
                "identity_score": identity_score, "sector_ladder_overlay": sector_ladder,
                "market_data_status": m.get("data_status", "partial"),
                "source": "verified daily limit-up pool", "source_timestamp": f"{date} 15:00:00",
            }
            attach_state_features(row, m["emotion_state"])
            valid_identity = (boards >= 3 and (row["is_space_leader"] == 1 or (not math.isnan(sector_ladder) and sector_ladder >= 0.60)))
            if boards == 2:
                valid_identity = bool((not math.isnan(identity_score) and identity_score >= 0.85)
                                      or (m["emotion_state"] in {"启动", "修复"} and theme and theme_first >= 2))
            row["valid_identity"] = float(valid_identity)
            row["buyable_eod"] = float(not locked)
            unavailable = [feature for feature in OPEN_FEATURES + ["theme_support", "announcement_risk"]
                           if feature not in row or math.isnan(safe_float(row.get(feature)))]
            row["data_status"] = "verified" if not unavailable else "partial"
            row["unavailable_fields"] = ",".join(unavailable)
            rows.append(row)
            if not math.isnan(turnover):
                previous_turnover[code] = turnover
    return pd.DataFrame(rows)


def attach_daily_execution_outcomes(candidates: pd.DataFrame, bars: pd.DataFrame,
                                    round_trip_cost: float = 0.003) -> pd.DataFrame:
    """Attach conservative daily-OHLC execution proxies without claiming tick fills.

    A signal is formed after the signal-date close.  On the next market session we
    only create a proxy entry when the high touches the limit price and the entire
    bar is not locked at limit-up.  Entry is pessimistically priced at the limit.
    Exit is the following market session's close.  A one-price limit-down bar is
    marked sell-blocked and its executable return remains unresolved.

    User-imported execution outcomes always take precedence.  These fields are
    future labels only and must never be included in live-scoring feature lists.
    """
    if candidates.empty or bars.empty:
        return candidates.copy()
    result = candidates.copy()
    dates = sorted(pd.Series(bars["date"].dropna().astype(str).unique()).tolist())
    next_date = {left: right for left, right in zip(dates, dates[1:])}
    result["exit_date"] = result["trade_date"].astype(str).map(next_date).fillna("")

    required = ["date", "code", "open", "high", "low", "close", "pre_close",
                "limit_price", "down_limit_price", "one_word_daily", "touched_limit"]
    daily = bars.reindex(columns=required).copy()
    daily["code"] = daily["code"].map(zfill_code)
    trade = daily.rename(columns={
        "date": "trade_date", "open": "trade_open", "high": "trade_high",
        "low": "trade_low", "close": "trade_close", "pre_close": "trade_pre_close",
        "limit_price": "trade_limit_price", "down_limit_price": "trade_down_limit_price",
        "one_word_daily": "trade_one_word_up", "touched_limit": "trade_touched_limit",
    })
    exit_bars = daily.rename(columns={
        "date": "exit_date", "open": "exit_open", "high": "exit_high",
        "low": "exit_low", "close": "exit_close", "pre_close": "exit_pre_close",
        "limit_price": "exit_limit_price", "down_limit_price": "exit_down_limit_price",
        "one_word_daily": "exit_one_word_up", "touched_limit": "exit_touched_limit",
    })
    result = result.merge(trade, on=["trade_date", "code"], how="left", validate="many_to_one")
    result = result.merge(exit_bars, on=["exit_date", "code"], how="left", validate="many_to_one")

    trade_available = result["trade_close"].notna() & result["trade_limit_price"].notna()
    touched = result["trade_touched_limit"].fillna(False).astype(bool)
    one_word_up = result["trade_one_word_up"].fillna(False).astype(bool)
    fillable = trade_available & touched & ~one_word_up
    exit_available = result["exit_close"].notna() & result["exit_down_limit_price"].notna()
    exit_one_word_down = exit_available.copy()
    for field in ("exit_open", "exit_high", "exit_low", "exit_close"):
        exit_one_word_down &= (result[field] - result["exit_down_limit_price"]).abs() <= 0.011
    result["exit_one_word_down"] = exit_one_word_down.astype(float)

    proxy_entry = result["trade_limit_price"].where(fillable)
    proxy_t1_close = result["exit_close"].where(fillable & exit_available)
    proxy_mark = (proxy_t1_close / proxy_entry - 1 - round_trip_cost).where(
        fillable & exit_available)
    proxy_blocked = exit_one_word_down.astype(float).where(fillable & exit_available)
    proxy_actual_exit = proxy_t1_close.where(~exit_one_word_down)
    proxy_net = proxy_mark.where(~exit_one_word_down)
    proxy_large_loss = (proxy_mark <= -0.07).astype(float).where(proxy_mark.notna())
    proxy_gap = (result["trade_open"] / result["trade_pre_close"] - 1).where(trade_available)
    promoted = pd.to_numeric(result["promoted_next_day"], errors="coerce")
    proxy_false_strength = ((proxy_gap >= 0.05) & (promoted == 0)).astype(float).where(
        proxy_gap.notna() & promoted.notna())

    existing_status = result.get("execution_status", pd.Series("unavailable", index=result.index)).fillna("unavailable")
    preserve_execution = existing_status.ne("unavailable")
    proxy_status = np.select(
        [~trade_available, one_word_up, ~touched, fillable & ~exit_available,
         fillable & exit_available & exit_one_word_down, fillable & exit_available],
        ["missing_trade_bar", "unbuyable_one_word_up", "no_limit_touch_no_trade",
         "missing_exit_bar", "sell_blocked_one_word_down", "inferred_daily_limit_touch_fill"],
        default="unavailable",
    )
    result["execution_status"] = np.where(preserve_execution, existing_status, proxy_status)
    for field, proxy in {
        "entry_price": proxy_entry, "t1_close": proxy_t1_close,
        "actual_exit_price": proxy_actual_exit, "mark_to_market_return": proxy_mark,
        "net_return": proxy_net, "sell_blocked": proxy_blocked,
    }.items():
        current = pd.to_numeric(result.get(field), errors="coerce")
        result[field] = current.where(preserve_execution | current.notna(), proxy)
    current_cost = pd.to_numeric(result.get("round_trip_cost"), errors="coerce")
    result["round_trip_cost"] = current_cost.where(preserve_execution, round_trip_cost)

    current_false = pd.to_numeric(result.get("false_strength"), errors="coerce")
    result["false_strength"] = current_false.where(current_false.notna(), proxy_false_strength)
    current_large = pd.to_numeric(result.get("large_loss"), errors="coerce")
    result["large_loss"] = current_large.where(current_large.notna(), proxy_large_loss)
    current_gap = pd.to_numeric(result.get("next_open_gap"), errors="coerce")
    result["next_open_gap"] = current_gap.where(current_gap.notna(), proxy_gap)

    false_status = result.get("false_strength_label_status", pd.Series("unavailable", index=result.index)).fillna("unavailable")
    large_status = result.get("large_loss_label_status", pd.Series("unavailable", index=result.index)).fillna("unavailable")
    result["false_strength_label_status"] = np.where(
        false_status.ne("unavailable"), false_status,
        np.where(proxy_false_strength.notna(), "inferred_from_unadjusted_daily_ohlc", "unavailable"))
    result["large_loss_label_status"] = np.where(
        large_status.ne("unavailable"), large_status,
        np.where(proxy_large_loss.notna(), "inferred_from_unadjusted_daily_ohlc", "unavailable"))
    result["execution_source"] = np.where(
        preserve_execution, result.get("execution_source", "user_import"),
        np.where(trade_available, "inferred_from_unadjusted_daily_ohlc", "unavailable"))
    result["execution_assumption"] = np.where(
        preserve_execution, result.get("execution_assumption", "verified user-supplied outcome"),
        "daily upper-bound proxy: any intraday limit touch fills at limit; no queue/tick proof")
    return result


def attach_terminal_height_labels(candidates: pd.DataFrame) -> pd.DataFrame:
    """Label the eventual terminal board of each observed streak without feature leakage.

    The label follows only confirmed next-session promotions.  If the available
    history ends while a streak is still alive, the row is right-censored and no
    terminal height is assigned.  These columns are outcomes and must never enter
    same-day scoring features.
    """
    if candidates.empty:
        return candidates.copy()
    result = candidates.copy().reset_index(drop=True)
    result["code"] = result["code"].map(zfill_code)
    keys = {(str(row.date), str(row.code)): index for index, row in result.iterrows()}
    terminal = np.full(len(result), math.nan)
    censored = np.ones(len(result), dtype=float)
    terminal_dates: list[str] = [""]*len(result)
    order = sorted(range(len(result)), key=lambda i: str(result.loc[i, "date"]), reverse=True)
    for index in order:
        board = int(safe_float(result.loc[index, "boards"]))
        promoted = safe_float(result.loc[index, "promoted_next_day"])
        if math.isnan(promoted):
            continue
        if promoted == 0:
            terminal[index] = board
            censored[index] = 0.0
            terminal_dates[index] = str(result.loc[index, "date"])
            continue
        next_index = keys.get((str(result.loc[index, "trade_date"]), str(result.loc[index, "code"])))
        if next_index is None or int(safe_float(result.loc[next_index, "boards"])) != board + 1:
            continue
        if not math.isnan(terminal[next_index]):
            terminal[index] = terminal[next_index]
            censored[index] = censored[next_index]
            terminal_dates[index] = terminal_dates[next_index]
    result["terminal_board"] = terminal
    result["additional_boards"] = terminal - pd.to_numeric(result["boards"], errors="coerce")
    result["terminal_height_censored"] = censored
    result["terminal_date"] = terminal_dates
    result["terminal_height_label_status"] = np.where(
        np.isfinite(terminal), "inferred_from_future_daily_limit_sequence", "right_censored")
    return result


def build_outcomes(candidates: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "trade_date", "code", "name", "promoted_next_day", "false_strength",
               "large_loss", "mark_to_market_return", "net_return", "entry_price", "t1_close",
               "actual_exit_price", "round_trip_cost", "sell_blocked", "execution_status",
               "next_open_gap", "exit_date", "trade_touched_limit", "trade_one_word_up",
               "exit_one_word_down", "false_strength_label_status", "large_loss_label_status",
               "terminal_board", "additional_boards", "terminal_height_censored", "terminal_date",
               "terminal_height_label_status", "promotion_label_status",
               "execution_source", "execution_assumption", "source", "source_timestamp", "data_status"]
    out = candidates[[c for c in columns if c in candidates]].copy()
    execution_sources = out.get("execution_source", pd.Series("unavailable", index=out.index)).astype(str)
    out["data_status"] = np.where(
        out["promoted_next_day"].isna(), "unavailable",
        np.where(execution_sources.eq("user_import"), "verified", "partial"))
    return out


def build_command(args: argparse.Namespace) -> None:
    ensure_dirs()
    bars_path = IMPORTED / "daily-bars.csv"
    history_manifest_path = IMPORTED / "daily-bars-manifest.json"
    history_manifest = (json.loads(history_manifest_path.read_text(encoding="utf-8"))
                        if history_manifest_path.exists() else {})
    point_in_time_status_complete = False
    if bars_path.exists():
        bars_header = pd.read_csv(bars_path, nrows=0).columns
        usecols = ["date", *(["universe_complete"] if "universe_complete" in bars_header else []),
                   *(["status_verified"] if "status_verified" in bars_header else [])]
        bars_meta = pd.read_csv(bars_path, usecols=usecols)
        bars_days = bars_meta["date"].nunique()
        universe_complete = bool("universe_complete" in bars_meta and bars_meta["universe_complete"].astype(str).str.lower().isin({"1", "true", "yes"}).all())
        point_in_time_status_complete = bool(
            "status_verified" in bars_meta
            and bars_meta["status_verified"].astype(str).str.lower().isin({"1", "true", "yes"}).all()
        )
    else:
        bars_days = 0
        universe_complete = False
    if bars_days >= 100:
        prepared_bars = prepare_daily_bars(bars_path)
        daily_context = derive_daily_market_context(prepared_bars)
        pools = pools_from_prepared_daily_bars(prepared_bars)
        pools = append_only_recent_pools(pools, pools_from_local_cache())
        daily_context.to_csv(DATA / "daily_market_context.csv", index=False, encoding="utf-8-sig")
    else:
        prepared_bars, daily_context = None, None
        pools = pools_from_local_cache()
    if not pools:
        raise SystemExit("no source data; run collect or import daily-bars first")
    market = build_market_table(pools, daily_context)
    candidates = build_candidate_table(pools, market)
    if prepared_bars is not None:
        candidates = attach_daily_execution_outcomes(candidates, prepared_bars)
    candidates = attach_terminal_height_labels(candidates)
    outcomes = build_outcomes(candidates)
    market.to_csv(DATA / "market_emotion.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(DATA / "leader_candidates.csv", index=False, encoding="utf-8-sig")
    outcomes.to_csv(DATA / "trade_outcomes.csv", index=False, encoding="utf-8-sig")
    if not (DATA / "open_snapshots.csv").exists():
        pd.DataFrame(columns=["signal_date", "trade_date", "asof", "code", "name", "source",
                                    "source_timestamp", "data_status", "unavailable_fields", *OPEN_FEATURES]).to_csv(
            DATA / "open_snapshots.csv", index=False, encoding="utf-8-sig")
    summary = {
        "source_mode": ("imported_daily_bars_plus_append_only_recent_pools"
                        if bars_days >= 100 else "recent_pool_diagnostic"),
        "market_days": int(market["date"].nunique()), "candidate_rows": int(len(candidates)),
        "labeled_rows": int(candidates["promoted_next_day"].notna().sum()),
        "promotion_positive_rate": safe_float(candidates["promoted_next_day"].mean()),
        "false_strength_labels": int(candidates["false_strength"].notna().sum()),
        "large_loss_labels": int(candidates["large_loss"].notna().sum()),
        "terminal_height_labels": int(candidates["terminal_board"].notna().sum()),
        "terminal_height_censored": int((candidates["terminal_height_censored"] == 1).sum()),
        "daily_execution_proxy_rows": int(candidates["execution_status"].astype(str).str.startswith(
            ("inferred_", "unbuyable_", "no_limit_", "missing_", "sell_blocked_one_word_down")).sum()),
        "verified_execution_rows": int((candidates["execution_source"] == "user_import").sum()),
        "execution_proxy_status": "inferred_daily_ohlc_upper_bound" if prepared_bars is not None else "unavailable",
        "universe_complete": universe_complete,
        "point_in_time_status_complete": point_in_time_status_complete,
        "historical_one_word_observable": bool(bars_days >= 100),
        "loss_effect_coverage": float(market["failed_count"].notna().mean()),
        "survivorship_bias": bool(bars_days >= 100 and not universe_complete),
        "historical_requested_start": history_manifest.get("requested_start"),
        "historical_requested_end": history_manifest.get("requested_end"),
        "historical_universe_source": history_manifest.get("universe_source"),
        "historical_provider_rows": history_manifest.get("history_provider_rows", {}),
        "historical_collection_warning": history_manifest.get("warning"),
    }
    (DATA / "build_manifest.json").write_text(json_text(summary), encoding="utf-8")
    print(json_text(summary))


def data_audit_command(args: argparse.Namespace) -> None:
    """Produce a machine-readable evidence gap report without changing the model."""
    ensure_dirs()
    build_manifest_path = DATA / "build_manifest.json"
    build_manifest = (json.loads(build_manifest_path.read_text(encoding="utf-8"))
                      if build_manifest_path.exists() else {})
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8")) if MODEL_PATH.exists() else {}
    candidates_path = DATA / "leader_candidates.csv"
    candidates = pd.read_csv(candidates_path, dtype={"code": str}) if candidates_path.exists() else pd.DataFrame()
    market_path = DATA / "market_emotion.csv"
    market = pd.read_csv(market_path) if market_path.exists() else pd.DataFrame()
    open_path = DATA / "open_snapshots.csv"
    snapshots = pd.read_csv(open_path, dtype={"code": str}) if open_path.exists() and open_path.stat().st_size else pd.DataFrame()
    outcomes_path = DATA / "trade_outcomes.csv"
    outcomes = pd.read_csv(outcomes_path, dtype={"code": str}) if outcomes_path.exists() else pd.DataFrame()

    market_days = int(market["date"].nunique()) if not market.empty else 0
    height_samples = {}
    if not candidates.empty:
        labeled = candidates[candidates["promoted_next_day"].notna() & candidates["boards"].between(2, 5)]
        height_samples = {str(int(board)): int(count) for board, count in labeled.groupby("boards").size().items()}
    theme_coverage = (float(candidates["theme"].fillna("").ne("").mean())
                      if not candidates.empty and "theme" in candidates else 0.0)
    required_open = ["auction_gap", "vwap_reclaimed", "theme_sync_open"]
    open_complete = (int(snapshots.reindex(columns=required_open).notna().all(axis=1).sum())
                     if not snapshots.empty else 0)
    label_counts = {
        field: int(pd.to_numeric(outcomes.get(field, pd.Series(dtype=float)), errors="coerce").notna().sum())
        for field in ("promoted_next_day", "false_strength", "large_loss", "mark_to_market_return", "net_return")
    }
    execution_status_counts = (outcomes["execution_status"].fillna("unavailable").astype(str).value_counts().to_dict()
                               if not outcomes.empty and "execution_status" in outcomes else {})
    verified_execution = int((outcomes.get("execution_source", pd.Series(dtype=str)).astype(str) == "user_import").sum())
    daily_proxy_fills = int(execution_status_counts.get("inferred_daily_limit_touch_fill", 0))
    blind = model.get("blind", {})
    threshold_evidence = model.get("candidate_model", {}).get("threshold_policy", {})
    joint_evidence = model.get("candidate_model", {}).get("joint_execution_policy", {})
    height_evidence = model.get("height_survival_backtest", {})
    calibration_evidence = model.get("candidate_model", {}).get("calibration_evidence", {})
    segment_gate = model.get("segment_reliability_gate", {})
    emotion_forecast_gate = model.get("emotion_forecast_gate", {})
    emotion_forecast_metrics = model.get("emotion_forecast_backtest", {})
    validation_integrity = model.get("validation_integrity", {})
    checks = [
        ("交易日", ">=250", market_days, market_days >= 250),
        ("2板观察", ">=100", height_samples.get("2", 0), height_samples.get("2", 0) >= 100),
        ("3板观察", ">=100", height_samples.get("3", 0), height_samples.get("3", 0) >= 100),
        ("4板观察", ">=100", height_samples.get("4", 0), height_samples.get("4", 0) >= 100),
        ("5板观察", ">=100", height_samples.get("5", 0), height_samples.get("5", 0) >= 100),
        ("完整历史证券池", "true", build_manifest.get("universe_complete", False), bool(build_manifest.get("universe_complete", False))),
        ("历史ST/上市状态", "true", build_manifest.get("point_in_time_status_complete", False), bool(build_manifest.get("point_in_time_status_complete", False))),
        ("历史一字可买性", "true", build_manifest.get("historical_one_word_observable", False), bool(build_manifest.get("historical_one_word_observable", False))),
        ("亏钱效应覆盖", ">=90%", build_manifest.get("loss_effect_coverage", 0), safe_float(build_manifest.get("loss_effect_coverage")) >= 0.90),
        ("完整开盘确认标签", ">=50", open_complete, open_complete >= 50),
        ("假强标签", ">=50", label_counts["false_strength"], label_counts["false_strength"] >= 50),
        ("大面标签", ">=50", label_counts["large_loss"], label_counts["large_loss"] >= 50),
        ("终板高度标签", ">=500", build_manifest.get("terminal_height_labels", 0),
         safe_float(build_manifest.get("terminal_height_labels")) >= 500),
        ("日线成交代理样本", ">=100", daily_proxy_fills, daily_proxy_fills >= 100),
        ("已核验真实成交结果", ">=50", verified_execution, verified_execution >= 50),
        ("部署阈值证据", "operational", threshold_evidence.get("operational", False), bool(threshold_evidence.get("operational", False))),
        ("晋级/大面联合阈值", "operational", joint_evidence.get("operational", False),
         bool(joint_evidence.get("operational", False))),
        ("终板高度两板技能", ">0", height_evidence.get("two_more_board_brier_skill"),
         safe_float(height_evidence.get("two_more_board_brier_skill")) > 0),
        ("终板高度优于众数基线", "MAE更低", height_evidence.get("mode_mae"),
         safe_float(height_evidence.get("mode_mae")) < safe_float(height_evidence.get("baseline_mode_mae"))),
        ("独立校准集证据", "USABLE", calibration_evidence.get("status", "UNAVAILABLE"), calibration_evidence.get("status") == "USABLE"),
        ("概率区间证据", "USABLE", blind.get("bootstrap_status", "UNAVAILABLE"), blind.get("bootstrap_status") == "USABLE"),
        ("Brier技能分", ">0", blind.get("brier_skill_score"), safe_float(blind.get("brier_skill_score")) > 0),
        ("概率校准ECE", "<=5%", blind.get("ece_5bin"), safe_float(blind.get("ece_5bin")) <= 0.05),
        ("盲测ENTER信号", ">=100", blind.get("policy_signals", 0), safe_float(blind.get("policy_signals")) >= 100),
        ("盲测精确率", ">=70%", blind.get("policy_precision"), safe_float(blind.get("policy_precision")) >= 0.70),
        ("盲测大面率", "<=10%", blind.get("policy_large_loss_rate"),
         not math.isnan(safe_float(blind.get("policy_large_loss_rate"))) and safe_float(blind.get("policy_large_loss_rate")) <= 0.10),
        ("分层可靠性", "PASS", segment_gate.get("passed", False), bool(segment_gate.get("passed", False))),
        ("情绪转移证据", "PASS", emotion_forecast_gate.get("passed", False), bool(emotion_forecast_gate.get("passed", False))),
        ("情绪逆风预测", "USABLE", emotion_forecast_metrics.get("status", "UNAVAILABLE"),
         emotion_forecast_metrics.get("status") == "USABLE"),
        ("未复用最终盲测", "FRESH_BLIND", validation_integrity.get("status", "UNAVAILABLE"),
         validation_integrity.get("status") == "FRESH_BLIND"),
    ]
    gate = [{"item": item, "required": required, "actual": actual, "passed": bool(passed)}
            for item, required, actual, passed in checks]
    next_actions: list[str] = []
    if market_days < 250 or not build_manifest.get("universe_complete", False):
        next_actions.append("导入含退市股票和历史成分状态的24—36个月完整主板日线")
    if not build_manifest.get("point_in_time_status_complete", False):
        next_actions.append("补齐逐日ST、上市未设涨跌幅和退市状态，禁止用当前名称回填历史")
    if theme_coverage < 0.80:
        next_actions.append("按日期补充真实炒作题材，目标候选覆盖率至少80%")
    if open_complete < 50:
        next_actions.append("持续归档9:25—10:00竞价/VWAP/板块同步快照")
    if verified_execution < 50:
        next_actions.append("补录至少50笔实际入场、T+1收盘、跌停卖出阻塞和最终退出结果；日线代理不能替代真实成交")
    if not joint_evidence.get("operational", False):
        next_actions.append("继续积累真实开盘和成交样本，直到同一验证阈值同时满足70%晋级、10%大面和每周3—5笔")
    if safe_float(height_evidence.get("two_more_board_brier_skill")) <= 0:
        next_actions.append("终板高度模型保持诊断状态，等待新鲜时间段验证两板以上生存概率")
    if blind.get("bootstrap_status") != "USABLE" or safe_float(blind.get("ece_5bin")) > 0.05:
        next_actions.append("扩大独立训练/验证期，直到日期分块概率区间可用且ECE不超过5%")
    if validation_integrity.get("status") == "REUSED_DEVELOPMENT_HOLDOUT":
        next_actions.append(validation_integrity.get(
            "next_fresh_blind_requirement", "冻结模型后等待新的未来时间段作为一次性盲测"))
    report = {
        "audited_at": datetime.now().isoformat(timespec="seconds"),
        "deployment_status": model.get("deployment_status", "NO_MODEL"),
        "market_days": market_days, "height_samples": height_samples,
        "execution_label_quality": {"status_counts": execution_status_counts,
                                    "verified_execution_rows": verified_execution,
                                    "daily_proxy_fill_rows": daily_proxy_fills,
                                    "interpretation": "daily OHLC labels are inferred upper-bound fills, not tick-verified executions"},
        "historical_data_source": {
            "requested_start": build_manifest.get("historical_requested_start"),
            "requested_end": build_manifest.get("historical_requested_end"),
            "universe_source": build_manifest.get("historical_universe_source"),
            "provider_rows": build_manifest.get("historical_provider_rows", {}),
            "warning": build_manifest.get("historical_collection_warning"),
        },
        "theme_coverage": theme_coverage, "open_complete_rows": open_complete,
        "outcome_label_counts": label_counts, "gate_checks": gate,
        "threshold_evidence": threshold_evidence,
        "joint_execution_threshold_evidence": joint_evidence,
        "height_survival_evidence": height_evidence,
        "calibration_evidence": calibration_evidence,
        "segment_reliability_gate": segment_gate,
        "emotion_forecast_gate": emotion_forecast_gate,
        "emotion_forecast_backtest": emotion_forecast_metrics,
        "validation_integrity": validation_integrity,
        "diagnostic_policy_signals": blind.get("diagnostic_policy_signals", 0),
        "diagnostic_policy_precision": blind.get("diagnostic_policy_precision"),
        "passed_checks": sum(item["passed"] for item in gate), "total_checks": len(gate),
        "next_actions": next_actions,
        "conclusion": "DATA_GATE_PASSED" if all(item["passed"] for item in gate) else "EVIDENCE_GAPS_REMAIN",
    }
    (RESULTS / "data_quality_report.json").write_text(json_text(report), encoding="utf-8")
    lines = ["# V3数据质量闸门", "", f"- 结论：`{report['conclusion']}`",
             f"- 模型状态：`{report['deployment_status']}`", f"- 通过：{report['passed_checks']}/{report['total_checks']}", "",
             "| 项目 | 要求 | 当前 | 通过 |", "|---|---:|---:|:---:|"]
    for item in gate:
        lines.append(f"| {item['item']} | {item['required']} | {item['actual']} | {'是' if item['passed'] else '否'} |")
    lines += ["", "## 下一步", ""] + [f"- {action}" for action in next_actions]
    (RESULTS / "data_quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_text(report))


@dataclass
class MatrixProcessor:
    features: list[str]
    medians: np.ndarray | None = None
    means: np.ndarray | None = None
    stds: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "MatrixProcessor":
        raw = frame.reindex(columns=self.features).apply(pd.to_numeric, errors="coerce").to_numpy(float)
        self.medians = np.array([np.nanmedian(raw[:, i]) if np.isfinite(raw[:, i]).any() else 0.0 for i in range(raw.shape[1])])
        filled = np.where(np.isnan(raw), self.medians, raw)
        self.means = filled.mean(axis=0)
        self.stds = filled.std(axis=0)
        self.stds[self.stds < 1e-8] = 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians is None or self.means is None or self.stds is None:
            raise ValueError("processor not fitted")
        raw = frame.reindex(columns=self.features).apply(pd.to_numeric, errors="coerce").to_numpy(float)
        missing = np.isnan(raw).astype(float)
        filled = np.where(np.isnan(raw), self.medians, raw)
        return np.c_[(filled - self.means) / self.stds, missing]

    def to_dict(self) -> dict[str, Any]:
        return {"features": self.features, "medians": self.medians.tolist(), "means": self.means.tolist(), "stds": self.stds.tolist()}

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "MatrixProcessor":
        return cls(obj["features"], np.array(obj["medians"]), np.array(obj["means"]), np.array(obj["stds"]))


def sigmoid(values: np.ndarray | float) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(values, -30, 30)))


@dataclass
class RegularizedLogistic:
    processor: MatrixProcessor
    weights: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, label: str, features: list[str], l2: float = 0.18,
            epochs: int = 3000, lr: float = 0.035) -> "RegularizedLogistic":
        processor = MatrixProcessor(features).fit(frame)
        x = processor.transform(frame)
        y = frame[label].to_numpy(float)
        design = np.c_[np.ones(len(x)), x]
        weights = np.zeros(design.shape[1])
        for _ in range(epochs):
            p = sigmoid(design @ weights)
            grad = design.T @ (p - y) / len(y)
            grad[1:] += l2 * weights[1:]
            weights -= lr * grad
        return cls(processor, weights)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = self.processor.transform(frame)
        return sigmoid(np.c_[np.ones(len(x)), x] @ self.weights)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "regularized_logistic", "processor": self.processor.to_dict(), "weights": self.weights.tolist()}

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "RegularizedLogistic":
        return cls(MatrixProcessor.from_dict(obj["processor"]), np.array(obj["weights"]))


@dataclass
class StumpBoosting:
    processor: MatrixProcessor
    intercept: float
    stumps: list[dict[str, float]]
    learning_rate: float

    @classmethod
    def fit(cls, frame: pd.DataFrame, label: str, features: list[str], rounds: int = 70,
            learning_rate: float = 0.08) -> "StumpBoosting":
        processor = MatrixProcessor(features).fit(frame)
        x = processor.transform(frame)
        y = frame[label].to_numpy(float)
        rate = min(1 - 1e-5, max(1e-5, y.mean()))
        intercept = math.log(rate / (1 - rate))
        score = np.full(len(y), intercept)
        stumps: list[dict[str, float]] = []
        for _ in range(rounds):
            residual = y - sigmoid(score)
            best: tuple[float, int, float, float, float] | None = None
            for feature in range(x.shape[1]):
                values = x[:, feature]
                for threshold in np.unique(np.quantile(values, [0.15, 0.3, 0.5, 0.7, 0.85])):
                    left = values <= threshold
                    if left.sum() < 5 or (~left).sum() < 5:
                        continue
                    lv, rv = residual[left].mean(), residual[~left].mean()
                    error = float(np.mean((residual - np.where(left, lv, rv)) ** 2))
                    if best is None or error < best[0]:
                        best = (error, feature, float(threshold), float(lv), float(rv))
            if best is None:
                break
            _, feature, threshold, left_value, right_value = best
            score += learning_rate * np.where(x[:, feature] <= threshold, left_value, right_value)
            stumps.append({"feature": feature, "threshold": threshold, "left": left_value, "right": right_value})
        return cls(processor, intercept, stumps, learning_rate)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = self.processor.transform(frame)
        score = np.full(len(x), self.intercept)
        for stump in self.stumps:
            feature = int(stump["feature"])
            score += self.learning_rate * np.where(x[:, feature] <= stump["threshold"], stump["left"], stump["right"])
        return sigmoid(score)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "stump_boosting", "processor": self.processor.to_dict(), "intercept": self.intercept,
                "stumps": self.stumps, "learning_rate": self.learning_rate}

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "StumpBoosting":
        return cls(MatrixProcessor.from_dict(obj["processor"]), obj["intercept"], obj["stumps"], obj["learning_rate"])


@dataclass
class PlattCalibrator:
    a: float = 1.0
    b: float = 0.0

    @classmethod
    def fit(cls, predictions: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        logits = np.log(np.clip(predictions, 1e-6, 1 - 1e-6) / np.clip(1 - predictions, 1e-6, 1))
        a, b = 1.0, 0.0
        for _ in range(1200):
            p = sigmoid(a * logits + b)
            a -= 0.02 * float(np.mean((p - labels) * logits) + 0.02 * (a - 1))
            b -= 0.02 * float(np.mean(p - labels))
        return cls(a, b)

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(predictions, 1e-6, 1 - 1e-6) / np.clip(1 - predictions, 1e-6, 1))
        return sigmoid(self.a * logits + self.b)


def model_from_dict(obj: dict[str, Any]) -> RegularizedLogistic | StumpBoosting:
    return RegularizedLogistic.from_dict(obj) if obj["kind"] == "regularized_logistic" else StumpBoosting.from_dict(obj)


def auc_score(y: np.ndarray, p: np.ndarray) -> float | None:
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return None
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 5) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        right_closed = index == bins - 1
        mask = ((p >= edges[index]) & (p <= edges[index + 1]) if right_closed
                else (p >= edges[index]) & (p < edges[index + 1]))
        if mask.any():
            error += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return error


def metric_block(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float | None = None) -> dict[str, Any]:
    if frame.empty:
        return {"n": 0}
    y = frame["promoted_next_day"].to_numpy(float)
    positive_rate = float(y.mean())
    brier = float(np.mean((probabilities - y) ** 2))
    log_loss = float(-np.mean(y * np.log(np.clip(probabilities, 1e-8, 1)) + (1-y) * np.log(np.clip(1-probabilities, 1e-8, 1))))
    brier_baseline = positive_rate * (1 - positive_rate)
    log_loss_baseline = -(positive_rate * math.log(max(positive_rate, 1e-8))
                          + (1 - positive_rate) * math.log(max(1 - positive_rate, 1e-8)))
    result = {
        "n": int(len(y)), "positive_rate": positive_rate, "auc": auc_score(y, probabilities),
        "brier": brier, "brier_baseline": brier_baseline,
        "brier_skill_score": (1 - brier / brier_baseline) if brier_baseline > 0 else None,
        "ece_5bin": expected_calibration_error(y, probabilities, bins=5),
        "log_loss": log_loss, "log_loss_baseline": log_loss_baseline,
        "log_loss_skill_score": (1 - log_loss / log_loss_baseline) if log_loss_baseline > 0 else None,
    }
    if threshold is not None:
        selected = probabilities >= threshold
        result.update({"raw_threshold_signals": int(selected.sum()),
                       "raw_threshold_precision": float(y[selected].mean()) if selected.any() else None,
                       "raw_threshold_recall": float(y[selected].sum() / max(1, y.sum())),
                       "raw_threshold_coverage": float(selected.mean())})
    return result


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1-p) / total + z*z / (4*total*total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dates = np.array(sorted(frame["date"].unique()))
    if len(dates) < 16:
        raise ValueError("at least 16 labeled dates are required for train/calibration/validation/blind split")
    a, b, c = int(len(dates) * 0.50), int(len(dates) * 0.65), int(len(dates) * 0.80)
    train_dates = dates[:max(1, a - 1)]
    calibration_dates = dates[a:max(a + 1, b - 1)]
    validation_dates = dates[b:max(b + 1, c - 1)]
    blind_dates = dates[c:]
    parts = (train_dates, calibration_dates, validation_dates, blind_dates)
    if any(len(part) == 0 for part in parts):
        raise ValueError("time split produced an empty segment")
    return (frame[frame["date"].isin(train_dates)], frame[frame["date"].isin(calibration_dates)],
            frame[frame["date"].isin(validation_dates)], frame[frame["date"].isin(blind_dates)],
            {"train": [str(train_dates[0]), str(train_dates[-1])],
             "embargo_1": str(dates[a - 1]),
             "calibration": [str(calibration_dates[0]), str(calibration_dates[-1])],
             "embargo_2": str(dates[b - 1]),
             "validation": [str(validation_dates[0]), str(validation_dates[-1])],
             "embargo_3": str(dates[c - 1]),
             "blind": [str(blind_dates[0]), str(blind_dates[-1])]})


def top_one_policy(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float,
                   minimum_probabilities: np.ndarray | None = None,
                   allowed_mask: np.ndarray | None = None) -> np.ndarray:
    temp = frame[["date"]].copy()
    temp["p"] = probabilities
    temp["eligible"] = eligibility_mask(frame)
    selected = np.zeros(len(frame), dtype=bool)
    for _, indexes in temp.groupby("date").groups.items():
        indexes = list(indexes)
        valid = [i for i in indexes
                 if bool(temp.loc[i, "eligible"])
                 and temp.loc[i, "p"] >= threshold
                 and (minimum_probabilities is None or minimum_probabilities[i] >= threshold)
                 and (allowed_mask is None or bool(allowed_mask[i]))]
        if valid:
            selected[valid[np.argmax([temp.loc[i, "p"] for i in valid])]] = True
    return selected


def eligibility_mask(frame: pd.DataFrame) -> np.ndarray:
    boards = pd.to_numeric(frame["boards"], errors="coerce")
    valid_identity = pd.to_numeric(frame.get("valid_identity", 0), errors="coerce").fillna(0)
    buyable = pd.to_numeric(frame.get("buyable_eod", 0), errors="coerce").fillna(0)
    states = frame["emotion_state"].isin(ACTIVE_STATES)
    announcement = pd.to_numeric(frame.get("announcement_risk", math.nan), errors="coerce")
    no_hard_risk = announcement.isna() | (announcement < 0.8)
    return (boards.between(2, 5) & (valid_identity == 1) & (buyable == 1) & states & no_hard_risk).to_numpy()


def choose_threshold(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, Any]:
    y = frame["promoted_next_day"].to_numpy(float)
    candidates = []
    active_dates = frame.loc[frame["emotion_state"].isin(ACTIVE_STATES), "date"].nunique()
    weeks = max(1.0, active_dates / 5)
    validation_dates = int(frame["date"].nunique())
    minimum_signals = max(10, math.ceil(weeks * 3))
    minimum_validation_dates = 20
    policy_basis = "identity_gated"
    if not eligibility_mask(frame).any():
        # A short diagnostic window can contain no 2-5 board confirmed leader at all.
        # Threshold discovery may then use the broad active/buyable pool, but this
        # fallback is recorded and can never satisfy the deployment gate.
        policy_basis = "broad_pool_fallback"

    def select_at(threshold: float) -> np.ndarray:
        if policy_basis == "identity_gated":
            return top_one_policy(frame.reset_index(drop=True), probabilities, threshold)
        temp = frame.reset_index(drop=True).copy()
        allowed = (temp["emotion_state"].isin(ACTIVE_STATES)
                   & pd.to_numeric(temp["boards"], errors="coerce").between(2, 5)
                   & (pd.to_numeric(temp.get("buyable_eod", 0), errors="coerce").fillna(0) == 1))
        selected = np.zeros(len(temp), dtype=bool)
        for _, indexes in temp.groupby("date").groups.items():
            valid = [i for i in indexes if bool(allowed.iloc[i]) and probabilities[i] >= threshold]
            if valid:
                selected[valid[np.argmax([probabilities[i] for i in valid])]] = True
        return selected

    threshold_grid = sorted(set(
        [round(float(x), 6) for x in np.arange(0.01, 0.96, 0.01)]
        + [max(0.000001, float(x) - 1e-9) for x in probabilities]
    ))
    for threshold in threshold_grid:
        selected = select_at(float(threshold))
        n = int(selected.sum())
        if n == 0:
            continue
        precision = float(y[selected].mean())
        frequency = n / weeks
        lower, upper = wilson_interval(int(y[selected].sum()), n)
        candidates.append({"threshold": round(float(threshold), 2), "signals": n, "precision": precision,
                           "signals_per_week": frequency, "target_precision": precision >= 0.70,
                           "target_frequency": 3 <= frequency <= 5,
                           "precision_wilson_95": [lower, upper],
                           "minimum_signal_evidence": n >= minimum_signals,
                           "confidence_floor": lower is not None and lower >= 0.50})
    if not candidates:
        return {"threshold": 0.90, "diagnostic_threshold": 0.90, "deployment_threshold": None,
                "signals": 0, "precision": None, "signals_per_week": 0,
                "target_precision": False, "target_frequency": False, "policy_basis": policy_basis,
                "operational": False, "validation_dates": validation_dates,
                "minimum_validation_dates": minimum_validation_dates, "minimum_signals": minimum_signals,
                "abstain_reason": "no validation threshold produced a signal"}
    operational = [x for x in candidates if x["target_precision"] and x["target_frequency"]
                   and x["minimum_signal_evidence"] and x["confidence_floor"]
                   and validation_dates >= minimum_validation_dates and policy_basis == "identity_gated"]
    diagnostic_feasible = [x for x in candidates if x["target_precision"] and x["target_frequency"]]
    if not diagnostic_feasible:
        diagnostic_feasible = [x for x in candidates if x["target_precision"]]
    if operational:
        chosen = sorted(operational, key=lambda x: (-x["signals"], -x["precision"], x["threshold"]))[0]
    elif diagnostic_feasible:
        chosen = sorted(diagnostic_feasible, key=lambda x: (-x["signals"], -x["precision"], x["threshold"]))[0]
    else:
        chosen = sorted(candidates, key=lambda x: (-x["precision"], -x["signals"], x["threshold"]))[0]
    is_operational = bool(operational and chosen in operational)
    chosen["policy_basis"] = policy_basis
    chosen["diagnostic_threshold"] = chosen["threshold"]
    chosen["deployment_threshold"] = chosen["threshold"] if is_operational else None
    chosen["operational"] = is_operational
    chosen["validation_dates"] = validation_dates
    chosen["minimum_validation_dates"] = minimum_validation_dates
    chosen["minimum_signals"] = minimum_signals
    reasons = []
    if validation_dates < minimum_validation_dates:
        reasons.append(f"validation dates {validation_dates} < {minimum_validation_dates}")
    if chosen["signals"] < minimum_signals:
        reasons.append(f"validation signals {chosen['signals']} < {minimum_signals}")
    if not chosen["target_precision"]:
        reasons.append("validation precision < 70%")
    if not chosen["target_frequency"]:
        reasons.append("validation frequency outside 3-5/week")
    if not chosen.get("confidence_floor", False):
        reasons.append("95% Wilson lower bound < 50%")
    if policy_basis != "identity_gated":
        reasons.append("identity gate unavailable")
    chosen["abstain_reason"] = "" if is_operational else "; ".join(reasons)
    return chosen


def choose_joint_execution_threshold(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, Any]:
    """Choose a validation-only threshold under promotion and tail-loss constraints."""
    temp = frame.reset_index(drop=True).copy()
    y = pd.to_numeric(temp["promoted_next_day"], errors="coerce").to_numpy(float)
    entries = pd.to_numeric(temp.get("entry_price", math.nan), errors="coerce")
    losses = pd.to_numeric(temp.get("large_loss", math.nan), errors="coerce")
    if not hasattr(entries, "notna"):
        entries = pd.Series(math.nan, index=temp.index)
    if not hasattr(losses, "notna"):
        losses = pd.Series(math.nan, index=temp.index)
    active_dates = temp.loc[temp["emotion_state"].isin(ACTIVE_STATES), "date"].nunique()
    weeks = max(1.0, active_dates / 5)
    minimum_trades = max(10, math.ceil(weeks * 3))
    candidates: list[dict[str, Any]] = []
    threshold_grid = sorted(set(
        [round(float(x), 6) for x in np.arange(0.01, 0.96, 0.01)]
        + [max(0.000001, float(x) - 1e-9) for x in probabilities]
    ))
    for threshold in threshold_grid:
        selected = top_one_policy(temp, probabilities, float(threshold))
        executed = selected & entries.notna().to_numpy()
        trade_count = int(executed.sum())
        if trade_count == 0:
            continue
        loss_available = executed & losses.notna().to_numpy()
        loss_count = int(loss_available.sum())
        precision = float(y[executed].mean())
        selection_precision = float(y[selected].mean()) if selected.any() else None
        large_loss_rate = float(losses.to_numpy(float)[loss_available].mean()) if loss_count else None
        frequency = trade_count / weeks
        lower, upper = wilson_interval(int(y[executed].sum()), trade_count)
        coverage = loss_count / trade_count
        row = {
            "threshold": round(float(threshold), 2), "selection_signals": int(selected.sum()),
            "executable_trades": trade_count, "executed_promotion_precision": precision,
            "selection_promotion_precision": selection_precision,
            "large_loss_rate": large_loss_rate, "large_loss_label_coverage": coverage,
            "trades_per_active_week": frequency, "precision_wilson_95": [lower, upper],
            "target_precision": precision >= 0.70,
            "target_large_loss": large_loss_rate is not None and large_loss_rate <= 0.10,
            "target_frequency": 3 <= frequency <= 5,
            "minimum_trade_evidence": trade_count >= minimum_trades,
            "loss_coverage_ok": coverage >= 0.90,
            "confidence_floor": lower is not None and lower >= 0.50,
        }
        candidates.append(row)
    if not candidates:
        return {"operational": False, "deployment_threshold": None, "diagnostic_threshold": None,
                "minimum_trades": minimum_trades, "validation_dates": int(temp["date"].nunique()),
                "abstain_reason": "no validation threshold produced an executable trade", "frontier": []}
    feasible = [row for row in candidates if all([
        row["target_precision"], row["target_large_loss"], row["target_frequency"],
        row["minimum_trade_evidence"], row["loss_coverage_ok"], row["confidence_floor"],
    ])]
    if feasible:
        chosen = sorted(feasible, key=lambda row: (-row["executable_trades"],
                                                   row["large_loss_rate"], -row["executed_promotion_precision"]))[0]
    else:
        dual_target = [row for row in candidates if row["target_precision"] and row["target_large_loss"]
                       and row["loss_coverage_ok"]]
        def diagnostic_score(row: dict[str, Any]) -> tuple[float, float, int]:
            loss = 1.0 if row["large_loss_rate"] is None else row["large_loss_rate"]
            constraint_penalty = (max(0.0, 0.70-row["executed_promotion_precision"])
                                  + 2*max(0.0, loss-0.10)
                                  + 0.25*abs(min(5.0, max(3.0, row["trades_per_active_week"]))
                                             - row["trades_per_active_week"]))
            return (constraint_penalty, -row["executed_promotion_precision"], -row["executable_trades"])
        chosen = (sorted(dual_target, key=lambda row: (-row["executable_trades"],
                                                       -row["executed_promotion_precision"], row["threshold"]))[0]
                  if dual_target else sorted(candidates, key=diagnostic_score)[0])
    chosen = dict(chosen)
    chosen["operational"] = bool(feasible)
    chosen["deployment_threshold"] = chosen["threshold"] if feasible else None
    chosen["diagnostic_threshold"] = chosen["threshold"]
    chosen["minimum_trades"] = minimum_trades
    chosen["validation_dates"] = int(temp["date"].nunique())
    reasons = []
    if not chosen["minimum_trade_evidence"]:
        reasons.append(f"validation executable trades {chosen['executable_trades']} < {minimum_trades}")
    if not chosen["target_precision"]:
        reasons.append("executed promotion precision < 70%")
    if not chosen["target_large_loss"]:
        reasons.append("large-loss rate > 10% or unavailable")
    if not chosen["target_frequency"]:
        reasons.append("executable frequency outside 3-5/week")
    if not chosen["loss_coverage_ok"]:
        reasons.append("large-loss label coverage < 90%")
    if not chosen["confidence_floor"]:
        reasons.append("executed precision Wilson lower bound < 50%")
    chosen["abstain_reason"] = "" if feasible else "; ".join(reasons)
    # Compact non-dominated precision/loss frontier for audit; never use blind data here.
    frontier = []
    for row in sorted(candidates, key=lambda x: x["threshold"]):
        loss = 1.0 if row["large_loss_rate"] is None else row["large_loss_rate"]
        dominated = any((other["executed_promotion_precision"] >= row["executed_promotion_precision"]
                         and (1.0 if other["large_loss_rate"] is None else other["large_loss_rate"]) <= loss
                         and other["executable_trades"] >= row["executable_trades"]
                         and (other["executed_promotion_precision"] > row["executed_promotion_precision"]
                              or (1.0 if other["large_loss_rate"] is None else other["large_loss_rate"]) < loss
                              or other["executable_trades"] > row["executable_trades"]))
                        for other in candidates)
        if not dominated:
            frontier.append({key: row[key] for key in (
                "threshold", "executable_trades", "executed_promotion_precision",
                "large_loss_rate", "trades_per_active_week")})
    chosen["frontier"] = frontier[:30]
    return chosen


def fit_candidate_model(train: pd.DataFrame, calibration: pd.DataFrame,
                        validation: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    models = [
        RegularizedLogistic.fit(train, "promoted_next_day", features),
        StumpBoosting.fit(train, "promoted_next_day", features),
    ]
    choices = []
    for model in models:
        calibration_raw = model.predict(calibration)
        calibrator = PlattCalibrator.fit(calibration_raw, calibration["promoted_next_day"].to_numpy(float))
        calibrated = calibrator.predict(model.predict(validation))
        metrics = metric_block(validation, calibrated)
        threshold = choose_threshold(validation.reset_index(drop=True), calibrated)
        policy_component = (threshold["precision"] or 0) if threshold.get("operational", False) else 0.0
        score = policy_component - metrics["brier"] + (metrics["auc"] or 0.5) * 0.25
        choices.append((score, model, calibrator, metrics, threshold))
    _, model, calibrator, metrics, threshold = max(choices, key=lambda x: x[0])
    validation_probabilities = calibrator.predict(model.predict(validation))
    joint_execution_policy = choose_joint_execution_threshold(
        validation.reset_index(drop=True), validation_probabilities)
    calibration_dates = int(calibration["date"].nunique())
    calibration_rows = int(len(calibration))
    return {"model": model.to_dict(), "calibrator": {"a": calibrator.a, "b": calibrator.b},
             "calibration_evidence": {"rows": calibration_rows, "dates": calibration_dates,
                                     "status": "USABLE" if calibration_rows >= 100 and calibration_dates >= 20 else "LOW_EVIDENCE",
                                     "rule": ">=100 rows across >=20 dates"},
             "validation": metrics, "threshold_policy": threshold,
             "joint_execution_policy": joint_execution_policy}


def fit_height_survival_model(train: pd.DataFrame, prior_strength: float = 20.0) -> dict[str, Any]:
    """Fit board-to-board empirical hazards using training dates only."""
    usable = train[train["promoted_next_day"].notna()].copy()
    y = pd.to_numeric(usable["promoted_next_day"], errors="coerce")
    global_rate = float(y.mean()) if len(y) else 0.35
    observed_max = int(pd.to_numeric(usable["boards"], errors="coerce").max()) if len(usable) else 5
    cap = max(10, min(15, observed_max + 1))
    hazards: dict[str, dict[str, Any]] = {}
    previous_probability = min(0.95, max(0.02, global_rate))
    for board in range(2, cap):
        rows = usable[pd.to_numeric(usable["boards"], errors="coerce") == board]
        n = int(len(rows))
        successes = float(pd.to_numeric(rows["promoted_next_day"], errors="coerce").sum()) if n else 0.0
        if n:
            probability = (successes + prior_strength*global_rate) / (n + prior_strength)
            source = "training_board_hazard_beta_shrunk"
        else:
            probability = min(previous_probability, global_rate*(0.85**max(1, board-5)))
            source = "conservative_extrapolation_no_training_rows"
        probability = float(min(0.95, max(0.01, probability)))
        hazards[str(board)] = {"probability": probability, "observations": n,
                               "successes": int(successes), "source": source}
        previous_probability = probability
    terminal_modes = {}
    if "terminal_board" in train:
        labeled = train[train["terminal_board"].notna()].copy()
        for board, group in labeled.groupby("boards"):
            counts = pd.to_numeric(group["terminal_board"], errors="coerce").value_counts()
            if len(counts):
                terminal_modes[str(int(board))] = int(counts.index[0])
    return {"status": "TRAINED_DIAGNOSTIC", "method": "discrete_board_hazard",
            "training_rows": int(len(usable)), "training_dates": int(usable["date"].nunique()),
            "global_promotion_rate": global_rate, "prior_strength": prior_strength,
            "maximum_reported_board": cap, "hazards": hazards,
            "baseline_terminal_modes": terminal_modes,
            "assumption": "current-to-next uses candidate probability; later steps use training-only beta-shrunk board hazards"}


def select_height_training_rows(all_candidates: pd.DataFrame, train_dates: Collection[str]) -> pd.DataFrame:
    """Purge height labels whose multi-day outcome crosses the training boundary."""
    dates = sorted(str(value) for value in train_dates)
    if not dates or "terminal_date" not in all_candidates:
        return all_candidates.iloc[0:0].copy()
    end = dates[-1]
    terminal_dates = all_candidates["terminal_date"].fillna("").astype(str)
    return all_candidates[
        all_candidates["date"].astype(str).isin(dates)
        & all_candidates["promoted_next_day"].notna()
        & (pd.to_numeric(all_candidates["boards"], errors="coerce") >= 2)
        & terminal_dates.ne("") & terminal_dates.le(end)
    ].copy()


def terminal_height_distribution(current_board: int, next_probability: float,
                                 height_model: dict[str, Any]) -> dict[str, Any]:
    if math.isnan(safe_float(next_probability)) or not height_model.get("hazards"):
        return {"status": "UNAVAILABLE", "distribution": {}}
    current = int(current_board)
    cap = max(current + 1, int(height_model.get("maximum_reported_board", 10)))
    next_p = clamp(float(next_probability), 0.0, 1.0)
    reach: dict[int, float] = {}
    survival = next_p
    for target in range(current + 1, cap + 1):
        if target > current + 1:
            hazard = safe_float(height_model.get("hazards", {}).get(str(target-1), {}).get("probability"))
            if math.isnan(hazard):
                hazard = safe_float(height_model.get("global_promotion_rate", 0.35))
            survival *= clamp(hazard, 0.0, 1.0)
        reach[target] = float(survival)
    masses: dict[int, float] = {current: 1-reach.get(current+1, 0.0)}
    for height in range(current+1, cap):
        masses[height] = max(0.0, reach[height]-reach[height+1])
    masses[cap] = max(0.0, reach.get(cap, 0.0))
    total = sum(masses.values())
    if total > 0:
        masses = {height: probability/total for height, probability in masses.items()}
    mode = max(masses, key=masses.get)
    expected = sum(height*probability for height, probability in masses.items())
    def mass_quantile(q: float) -> int:
        cumulative = 0.0
        for height, probability in sorted(masses.items()):
            cumulative += probability
            if cumulative >= q:
                return int(height)
        return int(cap)
    return {"status": "DIAGNOSTIC", "current_board": current, "mode_terminal_board": int(mode),
            "expected_terminal_board": float(expected),
            "terminal_board_interval_80": [mass_quantile(0.10), mass_quantile(0.90)],
            "probability_at_least_one_more": float(reach.get(current+1, 0.0)),
            "probability_at_least_two_more": float(reach.get(current+2, 0.0)),
            "distribution": {str(height): float(probability) for height, probability in masses.items()},
            "right_tail_is_capped": True, "cap": cap}


def evaluate_height_survival(blind: pd.DataFrame, probabilities: np.ndarray,
                             height_model: dict[str, Any]) -> dict[str, Any]:
    records = []
    cap = int(height_model.get("maximum_reported_board", 10))
    for position, (_, row) in enumerate(blind.reset_index(drop=True).iterrows()):
        actual = safe_float(row.get("terminal_board"))
        if math.isnan(actual):
            continue
        forecast = terminal_height_distribution(int(row["boards"]), float(probabilities[position]), height_model)
        distribution = forecast.get("distribution", {})
        clipped_actual = min(int(actual), cap)
        mass = max(1e-12, safe_float(distribution.get(str(clipped_actual), 0.0)))
        records.append({"boards": int(row["boards"]), "actual": int(actual),
                        "predicted": int(forecast["mode_terminal_board"]),
                        "expected": float(forecast["expected_terminal_board"]),
                        "interval_low": int(forecast["terminal_board_interval_80"][0]),
                        "interval_high": int(forecast["terminal_board_interval_80"][1]),
                        "log_loss": -math.log(mass),
                        "p_two_more": float(forecast.get("probability_at_least_two_more", 0.0)),
                        "y_two_more": float(actual >= int(row["boards"])+2),
                        "capped": int(actual) > cap})
    if not records:
        return {"status": "UNAVAILABLE", "n": 0}
    result = pd.DataFrame(records)
    baseline_predictions = []
    for row in records:
        baseline_predictions.append(int(height_model.get("baseline_terminal_modes", {}).get(
            str(row["boards"]), row["boards"])))
    baseline_predictions = np.array(baseline_predictions, dtype=float)
    actual = result["actual"].to_numpy(float)
    predicted = result["predicted"].to_numpy(float)
    two_more = result["y_two_more"].to_numpy(float)
    p_two_more = result["p_two_more"].to_numpy(float)
    baseline_two_more = float(two_more.mean())
    two_more_brier = float(np.mean((p_two_more-two_more)**2))
    baseline_brier = float(np.mean((baseline_two_more-two_more)**2))
    segments = []
    for board, group in result.groupby("boards"):
        segments.append({"boards": int(board), "n": int(len(group)),
                         "mae": float(np.mean(np.abs(group["predicted"]-group["actual"]))),
                         "within_one_accuracy": float((np.abs(group["predicted"]-group["actual"]) <= 1).mean()),
                         "mean_predicted_terminal": float(group["expected"].mean()),
                         "mean_actual_terminal": float(group["actual"].mean())})
    return {"status": "DIAGNOSTIC", "n": int(len(result)),
            "censored_rows_excluded": int(len(blind)-len(result)),
            "capped_actual_rows": int(result["capped"].sum()),
            "exact_mode_accuracy": float((predicted == actual).mean()),
            "within_one_board_accuracy": float((np.abs(predicted-actual) <= 1).mean()),
            "interval_80_coverage": float(((result["actual"] >= result["interval_low"])
                                            & (result["actual"] <= result["interval_high"])).mean()),
            "interval_80_mean_width": float((result["interval_high"]-result["interval_low"]).mean()),
            "mode_mae": float(np.mean(np.abs(predicted-actual))),
            "expected_height_mae": float(np.mean(np.abs(result["expected"].to_numpy(float)-actual))),
            "baseline_mode_mae": float(np.mean(np.abs(baseline_predictions-actual))),
            "terminal_distribution_log_loss": float(result["log_loss"].mean()),
            "two_more_board_brier": two_more_brier,
            "two_more_board_brier_baseline": baseline_brier,
            "two_more_board_brier_skill": (1-two_more_brier/baseline_brier) if baseline_brier > 0 else None,
            "by_current_board": segments,
            "warning": "terminal height is right-tail capped and remains diagnostic until evaluated on a fresh blind period"}


def select_eod_features(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[list[str], list[dict[str, Any]]]:
    groups = ["baseline", "emotion", "theme", "individual", "risk"]
    selected_features = list(BASE_FEATURES)
    report: list[dict[str, Any]] = []
    baseline_model = RegularizedLogistic.fit(train, "promoted_next_day", selected_features, epochs=1600)
    baseline_metrics = metric_block(validation, baseline_model.predict(validation))
    best_score = (baseline_metrics.get("auc") or 0.5) - 0.5 * baseline_metrics["brier"]
    report.append({"through_group": "baseline", "candidate_feature_count": len(selected_features),
                   "selected_feature_count": len(selected_features),
                   "validation_auc": baseline_metrics.get("auc"),
                   "validation_brier": baseline_metrics["brier"],
                   "selection_score": best_score, "marginal_improvement": None, "selected": True})
    for group in groups[1:]:
        candidate_features = selected_features + FEATURE_GROUPS[group]
        model = RegularizedLogistic.fit(train, "promoted_next_day", candidate_features, epochs=1600)
        p = model.predict(validation)
        metrics = metric_block(validation, p)
        score = (metrics.get("auc") or 0.5) - 0.5 * metrics["brier"]
        improvement = score - best_score
        selected = improvement >= 0.01
        if selected:
            selected_features = candidate_features
            best_score = score
        report.append({"through_group": group, "candidate_feature_count": len(candidate_features),
                       "selected_feature_count": len(selected_features),
                       "validation_auc": metrics.get("auc"), "validation_brier": metrics["brier"],
                       "selection_score": score, "marginal_improvement": improvement,
                       "selected": selected})
    return selected_features, report


def predict_artifact(model_obj: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    model = model_from_dict(model_obj["model"])
    raw = model.predict(frame)
    calibrator = PlattCalibrator(**model_obj["calibrator"])
    return calibrator.predict(raw)


def build_distribution_profile(frame: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    """Freeze pre-blind feature support for live out-of-distribution checks."""
    rows = len(frame)
    monitored: dict[str, dict[str, float]] = {}
    for feature in features:
        values = pd.to_numeric(frame.get(feature, pd.Series(index=frame.index, dtype=float)), errors="coerce")
        observed = values.dropna()
        if len(observed) < max(20, math.ceil(rows * 0.50)) or observed.nunique() < 2:
            continue
        q01, q25, q50, q75, q99 = observed.quantile([0.01, 0.25, 0.50, 0.75, 0.99]).tolist()
        iqr = max(1e-9, q75 - q25)
        monitored[feature] = {
            "observed": int(len(observed)), "missing_rate": float(values.isna().mean()),
            "median": float(q50), "lower_fence": float(q01 - 0.5 * iqr),
            "upper_fence": float(q99 + 0.5 * iqr), "iqr": float(iqr),
        }
    return {"training_rows": rows, "training_dates": int(frame["date"].nunique()),
            "monitored_features": monitored, "ood_fraction_limit": 0.15,
            "missing_surprise_limit": 3}


def distribution_shift(profile: dict[str, Any], frame: pd.DataFrame) -> list[dict[str, Any]]:
    monitored = profile.get("monitored_features", {})
    limit = safe_float(profile.get("ood_fraction_limit"))
    if math.isnan(limit):
        limit = 0.15
    missing_limit = int(profile.get("missing_surprise_limit", 3))
    results: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        extreme: list[str] = []
        missing_surprises: list[str] = []
        for feature, stats in monitored.items():
            value = safe_float(row.get(feature))
            if math.isnan(value):
                if safe_float(stats.get("missing_rate")) <= 0.05:
                    missing_surprises.append(feature)
            elif value < safe_float(stats.get("lower_fence")) or value > safe_float(stats.get("upper_fence")):
                extreme.append(feature)
        count = len(monitored)
        fraction = ((len(extreme) + len(missing_surprises)) / count) if count else 1.0
        status = ("OUT_OF_DISTRIBUTION" if fraction > limit or len(missing_surprises) >= missing_limit
                  else "DRIFT_WARNING" if fraction > 0.05 else "IN_DISTRIBUTION")
        results.append({"status": status, "ood_fraction": fraction,
                        "monitored_features": count, "extreme_features": extreme[:8],
                        "missing_surprises": missing_surprises[:8]})
    return results


def fit_bootstrap_ensemble(train: pd.DataFrame, calibration: pd.DataFrame,
                           chosen_model: dict[str, Any], features: list[str],
                           replicates: int = 25, seed: int = 20260830) -> dict[str, Any]:
    """Block-bootstrap whole trading dates; never sample future/blind rows."""
    kind = chosen_model["model"]["kind"]
    dates = np.array(sorted(train["date"].unique()))
    rng = np.random.default_rng(seed)
    members: list[dict[str, Any]] = []
    attempts = 0
    while len(members) < replicates and attempts < replicates * 4:
        attempts += 1
        sampled_dates = rng.choice(dates, size=len(dates), replace=True)
        sampled = pd.concat([train[train["date"] == date] for date in sampled_dates], ignore_index=True)
        if sampled["promoted_next_day"].nunique() < 2:
            continue
        if kind == "regularized_logistic":
            model = RegularizedLogistic.fit(sampled, "promoted_next_day", features, epochs=1600)
        else:
            model = StumpBoosting.fit(sampled, "promoted_next_day", features, rounds=55)
        calibrator = PlattCalibrator.fit(model.predict(calibration), calibration["promoted_next_day"].to_numpy(float))
        members.append({"model": model.to_dict(), "calibrator": {"a": calibrator.a, "b": calibrator.b}})
    train_dates, calibration_dates = int(len(dates)), int(calibration["date"].nunique())
    status = ("USABLE" if train_dates >= 60 and calibration_dates >= 20 and len(members) >= 20
              else "LOW_EVIDENCE" if members else "UNAVAILABLE")
    return {"method": "trading_date_block_bootstrap", "status": status,
            "members": members, "replicates_requested": replicates,
            "replicates_fitted": len(members), "training_dates": train_dates,
            "calibration_dates": calibration_dates, "quantiles": [0.10, 0.50, 0.90]}


def predict_bootstrap_ensemble(ensemble: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    members = ensemble.get("members", [])
    if not members:
        empty = np.full(len(frame), math.nan)
        return empty, empty.copy(), empty.copy()
    predictions = np.vstack([predict_artifact(member, frame) for member in members])
    return (np.quantile(predictions, 0.10, axis=0), np.quantile(predictions, 0.50, axis=0),
            np.quantile(predictions, 0.90, axis=0))


def fit_optional_risk_model(frame: pd.DataFrame, label: str, features: list[str]) -> dict[str, Any]:
    usable = frame[frame[label].notna()].copy()
    required_live_features = [feature for feature in OPEN_FEATURES
                              if feature in usable and usable[feature].notna().mean() >= 0.80]
    status_field = f"{label}_label_status"
    label_quality = (usable[status_field].fillna("unavailable").astype(str).value_counts().to_dict()
                     if status_field in usable else {"unclassified": int(len(usable))})
    verified_rows = int(sum(value for key, value in label_quality.items() if str(key).startswith("verified")))
    inferred_rows = int(sum(value for key, value in label_quality.items() if str(key).startswith("inferred")))
    if len(usable) < 50 or usable["date"].nunique() < 16 or usable[label].nunique() < 2:
        return {"status": "HEURISTIC_ONLY", "labeled_rows": int(len(usable)), "model": None,
                "label_quality": label_quality, "verified_label_rows": verified_rows,
                "inferred_label_rows": inferred_rows,
                "required_live_features": required_live_features}
    train, calibration, validation, blind, split = chronological_split(usable)
    model = RegularizedLogistic.fit(train, label, features)
    calibrator = PlattCalibrator.fit(model.predict(calibration), calibration[label].to_numpy(float))
    p = calibrator.predict(model.predict(blind))
    y = blind[label].to_numpy(float)
    positive_rate = float(y.mean())
    brier = float(np.mean((p-y)**2))
    brier_baseline = positive_rate * (1-positive_rate)
    log_loss = float(-np.mean(y*np.log(np.clip(p, 1e-8, 1)) + (1-y)*np.log(np.clip(1-p, 1e-8, 1))))
    log_loss_baseline = -(positive_rate*math.log(max(positive_rate, 1e-8))
                          + (1-positive_rate)*math.log(max(1-positive_rate, 1e-8)))
    blind_metrics = {
        "n": len(y), "positive_rate": positive_rate, "auc": auc_score(y, p),
        "brier": brier, "brier_baseline": brier_baseline,
        "brier_skill_score": (1-brier/brier_baseline) if brier_baseline > 0 else None,
        "ece_5bin": expected_calibration_error(y, p), "log_loss": log_loss,
        "log_loss_baseline": log_loss_baseline,
        "log_loss_skill_score": (1-log_loss/log_loss_baseline) if log_loss_baseline > 0 else None,
    }
    calibration_evidence = {"rows": int(len(calibration)), "dates": int(calibration["date"].nunique())}
    calibration_evidence["status"] = ("USABLE" if calibration_evidence["rows"] >= 100
                                      and calibration_evidence["dates"] >= 20 else "LOW_EVIDENCE")
    model_status = ("TRAINED_DIAGNOSTIC_VERIFIED" if verified_rows == len(usable)
                    else "TRAINED_DIAGNOSTIC_MIXED" if verified_rows
                    else "TRAINED_DIAGNOSTIC_DAILY_PROXY")
    risk_gate_passed = bool(len(y) >= 100 and safe_float(blind_metrics["auc"]) >= 0.55
                            and safe_float(blind_metrics["brier_skill_score"]) > 0)
    if not risk_gate_passed:
        model_status = ("REJECTED_NO_OUT_OF_SAMPLE_SKILL_VERIFIED" if verified_rows == len(usable)
                        else "REJECTED_NO_OUT_OF_SAMPLE_SKILL_DAILY_PROXY")
    return {"status": model_status, "labeled_rows": int(len(usable)), "split": split,
             "model": model.to_dict(), "calibrator": {"a": calibrator.a, "b": calibrator.b},
            "label_quality": label_quality, "verified_label_rows": verified_rows,
            "inferred_label_rows": inferred_rows,
            "required_live_features": required_live_features,
            "risk_gate": {"passed": risk_gate_passed, "requirements": {
                "blind_rows": ">=100", "auc": ">=0.55", "brier_skill_score": ">0"}},
             "calibration_evidence": calibration_evidence,
            "blind": blind_metrics}


def merge_training_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for feature in OPEN_FEATURES:
        if feature not in result:
            result[feature] = math.nan
    result["auction_gap_status"] = "unavailable"
    if "next_open_gap" in result:
        daily_gap = pd.to_numeric(result["next_open_gap"], errors="coerce")
        missing_gap = pd.to_numeric(result["auction_gap"], errors="coerce").isna()
        use_daily_gap = missing_gap & daily_gap.notna()
        result.loc[use_daily_gap, "auction_gap"] = daily_gap[use_daily_gap]
        result.loc[use_daily_gap, "auction_gap_status"] = "inferred_from_official_daily_open"
    path = DATA / "open_snapshots.csv"
    if not path.exists() or path.stat().st_size == 0:
        return result
    snapshots = pd.read_csv(path, dtype={"code": str})
    if snapshots.empty:
        return result
    snapshots = snapshots[snapshots["asof"] <= "10:00"].sort_values("asof")
    snapshots = snapshots.drop_duplicates(["signal_date", "code"], keep="last")
    snapshots = snapshots.rename(columns={"signal_date": "date"})
    snapshots["code"] = snapshots["code"].map(zfill_code)
    result["code"] = result["code"].map(zfill_code)
    snapshot_features = [f for f in OPEN_FEATURES if f in snapshots]
    keep = ["date", "code", *snapshot_features]
    overlay = snapshots[keep].rename(columns={f: f"{f}__snapshot" for f in snapshot_features})
    result = result.merge(overlay, on=["date", "code"], how="left", validate="many_to_one")
    for feature in snapshot_features:
        snapshot_field = f"{feature}__snapshot"
        available = pd.to_numeric(result[snapshot_field], errors="coerce").notna()
        result.loc[available, feature] = pd.to_numeric(result.loc[available, snapshot_field], errors="coerce")
        result = result.drop(columns=snapshot_field)
    if "auction_gap" in snapshot_features:
        snapshot_gap_available = snapshots["auction_gap"].notna()
        snapshot_keys = set(zip(snapshots.loc[snapshot_gap_available, "date"],
                                snapshots.loc[snapshot_gap_available, "code"].map(zfill_code)))
        result_keys = list(zip(result["date"], result["code"].map(zfill_code)))
        result.loc[[key in snapshot_keys for key in result_keys], "auction_gap_status"] = "verified_open_snapshot"
    return result


def fit_open_candidate_model(frame: pd.DataFrame) -> dict[str, Any]:
    full_required = ["auction_gap", "vwap_reclaimed", "theme_sync_open"]
    full_usable = frame[frame[full_required].notna().all(axis=1) & frame["promoted_next_day"].notna()].copy()
    if len(full_usable) >= 50 and full_usable["date"].nunique() >= 16:
        usable = full_usable
        required = full_required
        features = OPEN_MODEL_FEATURES
        status = "TRAINED_DIAGNOSTIC"
        deployment_eligible = True
        source_scope = "verified_open_confirmation"
    else:
        required = ["auction_gap"]
        usable = frame[frame["auction_gap"].notna() & frame["promoted_next_day"].notna()].copy()
        features = EOD_FEATURES + ["auction_gap"]
        status = "TRAINED_DIAGNOSTIC_DAILY_OPEN_ONLY"
        deployment_eligible = False
        source_scope = "official_daily_open_proxy_without_vwap_or_sector_sync"
    if len(usable) < 50 or usable["date"].nunique() < 16 or usable["promoted_next_day"].nunique() < 2:
        return {"status": "UNAVAILABLE", "labeled_rows": int(len(usable)), "model": None,
                "reason": "need at least 50 labeled open-confirmation rows across 16 dates"}
    train, calibration, validation, blind, split = chronological_split(usable)
    fitted = fit_candidate_model(train, calibration, validation, features)
    fitted["bootstrap_ensemble"] = fit_bootstrap_ensemble(train, calibration, fitted, features)
    fitted["feature_distribution"] = build_distribution_profile(train, features)
    p = predict_artifact(fitted, blind)
    threshold = float(fitted["threshold_policy"]["diagnostic_threshold"])
    return {"status": status, "labeled_rows": int(len(usable)), "split": split,
            "required_live_features": required, "feature_scope": source_scope,
            "deployment_eligible": deployment_eligible,
            "fitted": fitted, "blind": metric_block(blind, p, threshold)}


def optional_risk_probability(model_artifact: dict[str, Any], frame: pd.DataFrame) -> np.ndarray | None:
    if (model_artifact.get("status") == "HEURISTIC_ONLY"
            or str(model_artifact.get("status", "")).startswith("REJECTED_")
            or not model_artifact.get("model")):
        return None
    model = model_from_dict(model_artifact["model"])
    raw = model.predict(frame)
    calibrator = PlattCalibrator(**model_artifact["calibrator"])
    probabilities = calibrator.predict(raw)
    for feature in model_artifact.get("required_live_features", []):
        if feature not in frame:
            probabilities[:] = math.nan
            break
        probabilities[pd.to_numeric(frame[feature], errors="coerce").isna().to_numpy()] = math.nan
    return probabilities


def transition_matrix(market: pd.DataFrame) -> dict[str, dict[str, float]]:
    counts = {state: {other: 1 for other in STATE_NAMES} for state in STATE_NAMES}
    states = market.sort_values("date")["emotion_state"].tolist()
    for left, right in zip(states, states[1:]):
        counts[left][right] += 1
    return {left: {right: count / sum(row.values()) for right, count in row.items()} for left, row in counts.items()}


def transition_evidence(market: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Describe how much evidence sits behind the smoothed transition matrix."""
    ordered = market.sort_values("date")["emotion_state"].tolist()
    totals = {state: 0 for state in STATE_NAMES}
    for left, _ in zip(ordered, ordered[1:]):
        totals[left] += 1
    result: dict[str, dict[str, Any]] = {}
    for state, count in totals.items():
        confidence = "LOW" if count < 20 else ("MEDIUM" if count < 60 else "HIGH")
        result[state] = {
            "observed_transitions": count,
            "confidence": confidence,
            "minimum_for_medium": 20,
            "minimum_for_high": 60,
        }
    return result


def emotion_forecast_deployment_gate(market: pd.DataFrame) -> dict[str, Any]:
    """Require enough pre-existing transitions before emotion forecasts can gate money."""
    evidence = transition_evidence(market)
    checks = []
    for state in sorted(ACTIVE_STATES):
        observed = int(evidence.get(state, {}).get("observed_transitions", 0))
        checks.append({"state": state, "minimum_transitions": 20,
                       "observed_transitions": observed, "passed": observed >= 20})
    forecast_columns = ["emotion_adverse_next_probability", "emotion_transition_entropy",
                        "emotion_transition_evidence", "emotion_forecast_status"]
    coverage = (float(market.reindex(columns=forecast_columns).notna().all(axis=1).mean())
                if not market.empty else 0.0)
    passed = coverage >= 0.90 and all(item["passed"] for item in checks)
    return {"passed": passed, "active_state_checks": checks, "forecast_coverage": coverage,
            "rule": "online as-of transition forecast coverage >=90%; each active state >=20 observed transitions"}


def emotion_forecast_backtest(market: pd.DataFrame) -> dict[str, Any]:
    """Score the causal next-regime forecast directly, independent of stock returns."""
    ordered = market.sort_values("date").reset_index(drop=True)
    if len(ordered) < 2:
        return {"status": "LOW_EVIDENCE", "n": 0}
    current = ordered.iloc[:-1].copy()
    next_states = ordered["emotion_state"].shift(-1).iloc[:-1].astype(str)
    all_y = next_states.isin({"大分歧", "退潮"}).astype(float).to_numpy()
    adverse_p = pd.to_numeric(
        current.reindex(columns=["emotion_adverse_next_probability"])["emotion_adverse_next_probability"],
        errors="coerce").to_numpy(float)
    valid = np.isfinite(adverse_p)
    if not valid.any():
        return {"status": "LOW_EVIDENCE", "n": 0}
    adverse_y, adverse_p = all_y[valid], np.clip(adverse_p[valid], 1e-6, 1 - 1e-6)
    n = int(len(adverse_y))
    base_rate = float(adverse_y.mean())
    brier = float(np.mean((adverse_p - adverse_y) ** 2))
    brier_baseline = float(np.mean((base_rate - adverse_y) ** 2))
    brier_skill = (1 - brier / brier_baseline) if brier_baseline > 0 else math.nan
    binary_logloss = float(-np.mean(adverse_y * np.log(adverse_p) + (1 - adverse_y) * np.log(1 - adverse_p)))
    baseline_probability = min(1 - 1e-6, max(1e-6, base_rate))
    baseline_logloss = float(-np.mean(
        adverse_y * math.log(baseline_probability) + (1 - adverse_y) * math.log(1 - baseline_probability)))

    def comparison_block(column: str) -> dict[str, Any]:
        probabilities = pd.to_numeric(
            current.reindex(columns=[column])[column], errors="coerce").to_numpy(float)
        mask = np.isfinite(probabilities)
        if not mask.any():
            return {"n": 0}
        y = all_y[mask]
        p = np.clip(probabilities[mask], 1e-6, 1 - 1e-6)
        block_brier = float(np.mean((p - y) ** 2))
        block_logloss = float(-np.mean(
            y * np.log(p) + (1 - y) * np.log(1 - p)))
        return {"n": int(mask.sum()), "auc": auc_score(y, p), "brier": block_brier,
                "ece_5bin": expected_calibration_error(y, p), "log_loss": block_logloss}

    markov_metrics = comparison_block("emotion_adverse_markov_probability")
    challenger_metrics = comparison_block("emotion_adverse_challenger_probability")
    challenger_vs_markov_skill = math.nan
    if challenger_metrics.get("n", 0):
        challenger_values = pd.to_numeric(
            current["emotion_adverse_challenger_probability"], errors="coerce").to_numpy(float)
        challenger_mask = np.isfinite(challenger_values)
        markov_same = pd.to_numeric(
            current["emotion_adverse_markov_probability"], errors="coerce").to_numpy(float)[challenger_mask]
        y_same = all_y[challenger_mask]
        markov_same_brier = float(np.mean((markov_same - y_same) ** 2))
        challenger_vs_markov_skill = (1 - challenger_metrics["brier"] / markov_same_brier
                                      if markov_same_brier > 0 else math.nan)

    multiclass_losses: list[float] = []
    top1_hits: list[float] = []
    for position, actual in enumerate(next_states[valid]):
        probabilities = {
            state: safe_float(current.loc[current.index[valid][position], column])
            for state, column in STATE_PROBABILITY_COLUMNS.items()
        }
        probability = probabilities.get(actual, math.nan)
        if math.isnan(probability):
            continue
        multiclass_losses.append(-math.log(max(1e-6, probability)))
        top1_hits.append(float(max(probabilities, key=probabilities.get) == actual))
    multiclass_logloss = float(np.mean(multiclass_losses)) if multiclass_losses else math.nan
    uniform_logloss = math.log(len(STATE_NAMES))
    multiclass_skill = (1 - multiclass_logloss / uniform_logloss
                        if not math.isnan(multiclass_logloss) else math.nan)
    usable_rows = int((current.loc[current.index[valid], "emotion_forecast_status"] == "USABLE").sum())
    champion_rows = int((current.get("emotion_forecast_source", pd.Series(index=current.index, dtype=str))
                         == "SUPERVISED_CHALLENGER").sum())
    usable = (n >= 100 and usable_rows >= 60 and not math.isnan(brier_skill) and brier_skill > 0
              and not math.isnan(multiclass_skill) and multiclass_skill > 0)
    return {
        "status": "USABLE" if usable else "LOW_EVIDENCE", "n": n,
        "usable_evidence_rows": usable_rows, "adverse_rate": base_rate,
        "adverse_auc": auc_score(adverse_y, adverse_p), "adverse_brier": brier,
        "adverse_brier_baseline": brier_baseline, "adverse_brier_skill_score": brier_skill,
        "adverse_ece_5bin": expected_calibration_error(adverse_y, adverse_p),
        "adverse_log_loss": binary_logloss, "adverse_log_loss_baseline": baseline_logloss,
        "markov_baseline": markov_metrics, "supervised_challenger": challenger_metrics,
        "challenger_vs_markov_brier_skill": challenger_vs_markov_skill,
        "supervised_champion_rows": champion_rows,
        "multiclass_log_loss": multiclass_logloss, "uniform_multiclass_log_loss": uniform_logloss,
        "multiclass_log_loss_skill": multiclass_skill,
        "multiclass_top1_accuracy": float(np.mean(top1_hits)) if top1_hits else math.nan,
        "rule": "n>=100, >=60 rows with >=20 same-state transitions, positive adverse Brier skill and positive seven-state log-loss skill; challenger needs >=60 prior causal forecasts, >=5% Brier gain vs Markov, lower log loss and ECE no worse by >2pp",
    }


def blind_reuse_audit(split: dict[str, Any], blind: pd.DataFrame) -> dict[str, Any]:
    """Prove whether a nominal blind window has already influenced development."""
    blind_window = tuple(str(value) for value in split.get("blind", []))
    window_fingerprint = hashlib.sha256(
        json.dumps(blind_window, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    label_columns = [column for column in (
        "date", "code", "boards", "promoted_next_day", "terminal_board", "terminal_date",
        "large_loss", "execution_status") if column in blind]
    label_frame = blind[label_columns].copy().sort_values(label_columns[:-1] or label_columns)
    label_hashes = pd.util.hash_pandas_object(label_frame, index=False).to_numpy(np.uint64)
    label_fingerprint = hashlib.sha256(label_hashes.tobytes()).hexdigest()
    code_fingerprint = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prior_models: list[str] = []
    frozen_dir = RESULTS / "models"
    if frozen_dir.exists():
        for path in sorted(frozen_dir.glob("model_*.json")):
            try:
                artifact = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if tuple(str(value) for value in artifact.get("split", {}).get("blind", [])) == blind_window:
                prior_models.append(path.name)
    reuse_count = len(prior_models) + 1
    fresh = reuse_count == 1
    return {
        "status": "FRESH_BLIND" if fresh else "REUSED_DEVELOPMENT_HOLDOUT",
        "deployment_eligible": fresh, "blind_window": list(blind_window),
        "window_fingerprint_sha256": window_fingerprint,
        "label_fingerprint_sha256": label_fingerprint,
        "model_code_fingerprint_sha256": code_fingerprint,
        "reuse_count_including_current": reuse_count,
        "prior_matching_models": prior_models,
        "rule": "a blind window may be inspected once; any retraining or rule change after inspection converts it to a development holdout",
        "next_fresh_blind_requirement": f"use dates strictly after {blind_window[-1]} and freeze code before labels are inspected" if blind_window else "define a new future window",
    }


def train_command(args: argparse.Namespace) -> None:
    ensure_dirs()
    if MODEL_PATH.exists() and not getattr(args, "force", False):
        previous_artifact = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        trained = pd.Timestamp(previous_artifact.get("trained_at", "1970-01-01"))
        now = pd.Timestamp.now()
        if (trained.year, trained.quarter) == (now.year, now.quarter):
            raise SystemExit("quarterly retraining lock is active; use --force only for code/test corrections")
    candidates_path = DATA / "leader_candidates.csv"
    if not candidates_path.exists():
        raise SystemExit("run build first")
    all_candidates = pd.read_csv(candidates_path, dtype={"code": str})
    frame = all_candidates[all_candidates["promoted_next_day"].notna()
                           & all_candidates["boards"].between(2, 5)].reset_index(drop=True)
    open_frame = merge_training_snapshots(frame)
    train, calibration, validation, blind, split = chronological_split(frame)
    blind_integrity = blind_reuse_audit(split, blind)
    selected_features, feature_selection = select_eod_features(train, validation)
    candidate_model = fit_candidate_model(train, calibration, validation, selected_features)
    candidate_model["bootstrap_ensemble"] = fit_bootstrap_ensemble(
        train, calibration, candidate_model, selected_features)
    candidate_model["feature_distribution"] = build_distribution_profile(train, selected_features)
    blind_p = predict_artifact(candidate_model, blind)
    height_training_frame = select_height_training_rows(all_candidates, train["date"].unique())
    height_survival_model = fit_height_survival_model(height_training_frame)
    height_survival_backtest = evaluate_height_survival(blind, blind_p, height_survival_model)
    blind_lower, blind_median, blind_upper = predict_bootstrap_ensemble(
        candidate_model["bootstrap_ensemble"], blind)
    blind_shift = distribution_shift(candidate_model["feature_distribution"], blind)
    blind_in_distribution = np.array([item["status"] == "IN_DISTRIBUTION" for item in blind_shift], dtype=bool)
    threshold_policy = candidate_model["threshold_policy"]
    joint_policy = candidate_model.get("joint_execution_policy", {})
    diagnostic_threshold = float(threshold_policy["diagnostic_threshold"])
    diagnostic_selected = top_one_policy(blind.reset_index(drop=True), blind_p, diagnostic_threshold)
    uncertainty_operational = candidate_model["bootstrap_ensemble"].get("status") == "USABLE"
    calibration_operational = candidate_model.get("calibration_evidence", {}).get("status") == "USABLE"
    policy_operational = bool(threshold_policy.get("operational", False)
                              and joint_policy.get("operational", False)
                              and uncertainty_operational and calibration_operational)
    blind_selected = (top_one_policy(blind.reset_index(drop=True), blind_p,
                                     float(joint_policy["deployment_threshold"]),
                                     minimum_probabilities=blind_lower,
                                     allowed_mask=blind_in_distribution)
                      if policy_operational else np.zeros(len(blind), dtype=bool))
    y_blind = blind["promoted_next_day"].to_numpy(float)
    blind_metrics = metric_block(blind, blind_p, diagnostic_threshold)
    blind_metrics.update({
        "policy_signals": int(blind_selected.sum()),
        "policy_precision": float(y_blind[blind_selected].mean()) if blind_selected.any() else None,
        "policy_frequency_active_week": float(blind_selected.sum() / max(1, blind[blind["emotion_state"].isin(ACTIVE_STATES)]["date"].nunique() / 5)),
        "policy_abstained": not policy_operational,
        "policy_abstain_reason": (joint_policy.get("abstain_reason", "")
                                  or threshold_policy.get("abstain_reason", "")
                                  or ("bootstrap uncertainty evidence is not usable" if not uncertainty_operational else "")
                                  or ("independent calibration evidence is not usable" if not calibration_operational else "")),
        "diagnostic_policy_signals": int(diagnostic_selected.sum()),
        "diagnostic_policy_precision": float(y_blind[diagnostic_selected].mean()) if diagnostic_selected.any() else None,
        "diagnostic_policy_frequency_active_week": float(diagnostic_selected.sum() / max(1, blind[blind["emotion_state"].isin(ACTIVE_STATES)]["date"].nunique() / 5)),
        "bootstrap_status": candidate_model["bootstrap_ensemble"].get("status"),
        "bootstrap_mean_interval_width": float(np.nanmean(blind_upper - blind_lower)),
        "bootstrap_median_absolute_shift": float(np.nanmean(np.abs(blind_median - blind_p))),
        "ood_rows": int((~blind_in_distribution).sum()),
        "ood_rate": float((~blind_in_distribution).mean()),
        "evaluation_status": blind_integrity["status"],
        "blind_reuse_count": blind_integrity["reuse_count_including_current"],
    })
    lower, upper = wilson_interval(int(y_blind[blind_selected].sum()), int(blind_selected.sum()))
    blind_metrics["policy_precision_wilson_95"] = [lower, upper]
    diagnostic_lower, diagnostic_upper = wilson_interval(
        int(y_blind[diagnostic_selected].sum()), int(diagnostic_selected.sum()))
    blind_metrics["diagnostic_policy_precision_wilson_95"] = [diagnostic_lower, diagnostic_upper]
    selected_large_losses = pd.to_numeric(blind.reset_index(drop=True).loc[blind_selected, "large_loss"], errors="coerce").dropna()
    blind_metrics["policy_large_loss_rate"] = float(selected_large_losses.mean()) if len(selected_large_losses) else None
    policy_execution = execution_metrics(blind.reset_index(drop=True).loc[blind_selected].copy())
    blind_metrics["policy_executable_trades"] = policy_execution.get("executable_trades", 0)
    blind_metrics["policy_executed_promotion_precision"] = policy_execution.get("executed_promotion_precision")
    blind_metrics["policy_executable_frequency_active_week"] = float(
        blind_metrics["policy_executable_trades"]
        / max(1, blind[blind["emotion_state"].isin(ACTIVE_STATES)]["date"].nunique()/5))
    open_model = fit_open_candidate_model(open_frame)
    false_model = fit_optional_risk_model(open_frame, "false_strength", OPEN_MODEL_FEATURES)
    loss_model = fit_optional_risk_model(open_frame, "large_loss", OPEN_MODEL_FEATURES)
    false_model["target_definition"] = "official open gap >=5% and failed same-day promotion; auction gap is therefore a contemporaneous defining input"
    loss_model["target_definition"] = "limit-touch proxy entry to following-session close after 0.30% cost <= -7%"
    market = pd.read_csv(DATA / "market_emotion.csv")
    emotion_forecast_gate = emotion_forecast_deployment_gate(market)
    emotion_forecast_metrics = emotion_forecast_backtest(market)
    build_manifest_path = DATA / "build_manifest.json"
    build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8")) if build_manifest_path.exists() else {}
    highest = blind.loc[blind.groupby("date")["boards"].idxmax()]
    baseline_precision = safe_float(highest["promoted_next_day"].mean())
    blind_metrics["highest_board_baseline_precision"] = baseline_precision
    height_counts = frame.groupby("boards")["promoted_next_day"].agg(["count", "mean"]).reset_index().to_dict("records")
    blind_reliability_frame = blind.reset_index(drop=True).copy()
    blind_reliability_frame["distribution_status"] = [item["status"] for item in blind_shift]
    blind_segment_reliability = reliability_by_segment(
        blind_reliability_frame, blind_p, ("boards", "emotion_state", "distribution_status"))
    segment_gate = segment_reliability_gate(blind_segment_reliability)
    historical_gate = (
        market["date"].nunique() >= 250
        and all(int(x["count"]) >= 100 for x in height_counts if int(x["boards"]) in (2, 3, 4, 5))
        and blind_metrics["policy_executable_trades"] >= 100
        and (blind_metrics["policy_executed_promotion_precision"] or 0) >= 0.70
        and 3 <= blind_metrics["policy_executable_frequency_active_week"] <= 5
        and blind_metrics["policy_precision"] >= baseline_precision + 0.05
        and blind_metrics["policy_large_loss_rate"] is not None
        and blind_metrics["policy_large_loss_rate"] <= 0.10
        and safe_float(blind_metrics.get("brier_skill_score")) > 0
        and safe_float(blind_metrics.get("ece_5bin")) <= 0.05
        and candidate_model["threshold_policy"].get("operational", False)
        and candidate_model.get("joint_execution_policy", {}).get("operational", False)
        and height_survival_backtest.get("status") == "DIAGNOSTIC"
        and safe_float(height_survival_backtest.get("two_more_board_brier_skill")) > 0
        and safe_float(height_survival_backtest.get("mode_mae")) < safe_float(
            height_survival_backtest.get("baseline_mode_mae"))
        and candidate_model.get("calibration_evidence", {}).get("status") == "USABLE"
        and candidate_model["bootstrap_ensemble"].get("status") == "USABLE"
        and candidate_model["threshold_policy"].get("policy_basis") == "identity_gated"
        and open_model["status"] != "UNAVAILABLE"
        and open_model.get("deployment_eligible", False)
        and open_model.get("fitted", {}).get("calibration_evidence", {}).get("status") == "USABLE"
        and open_model.get("fitted", {}).get("bootstrap_ensemble", {}).get("status") == "USABLE"
        and str(false_model["status"]).startswith("TRAINED_DIAGNOSTIC")
        and str(loss_model["status"]).startswith("TRAINED_DIAGNOSTIC")
        and false_model.get("risk_gate", {}).get("passed", False)
        and loss_model.get("risk_gate", {}).get("passed", False)
        and false_model.get("calibration_evidence", {}).get("status") == "USABLE"
        and loss_model.get("calibration_evidence", {}).get("status") == "USABLE"
        and not build_manifest.get("survivorship_bias", True)
        and build_manifest.get("point_in_time_status_complete", False)
        and build_manifest.get("historical_one_word_observable", False)
        and safe_float(build_manifest.get("loss_effect_coverage")) >= 0.90
        and segment_gate["passed"]
        and emotion_forecast_gate["passed"]
        and emotion_forecast_metrics.get("status") == "USABLE"
        and blind_integrity["deployment_eligible"]
    )
    artifact = {
        "version": 3, "emotion_engine_version": "3.4", "probability_engine_version": "3.8",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data_start": frame["date"].min(), "data_end": frame["date"].max(),
        "market_days": int(market["date"].nunique()), "observations": int(len(frame)),
        "split": split, "candidate_model": candidate_model, "blind": blind_metrics,
        "validation_integrity": blind_integrity,
        "feature_selection": feature_selection,
        "height_survival_model": height_survival_model,
        "height_survival_backtest": height_survival_backtest,
        "open_candidate_model": open_model,
        "false_strength_model": false_model, "large_loss_model": loss_model,
        "blind_segment_reliability": blind_segment_reliability,
        "segment_reliability_gate": segment_gate,
        "emotion_forecast_gate": emotion_forecast_gate,
        "emotion_forecast_backtest": emotion_forecast_metrics,
        "height_samples": height_counts, "emotion_transition_matrix": transition_matrix(market),
        "emotion_transition_evidence": transition_evidence(market),
        "data_integrity": {"universe_complete": build_manifest.get("universe_complete", False),
                           "survivorship_bias": build_manifest.get("survivorship_bias", True),
                           "point_in_time_status_complete": build_manifest.get("point_in_time_status_complete", False),
                           "historical_one_word_observable": build_manifest.get("historical_one_word_observable", False),
                           "loss_effect_coverage": build_manifest.get("loss_effect_coverage"),
                           "historical_requested_start": build_manifest.get("historical_requested_start"),
                           "historical_requested_end": build_manifest.get("historical_requested_end"),
                           "historical_universe_source": build_manifest.get("historical_universe_source"),
                           "historical_provider_rows": build_manifest.get("historical_provider_rows", {}),
                           "execution_proxy_status": build_manifest.get("execution_proxy_status", "unavailable"),
                           "daily_execution_proxy_rows": build_manifest.get("daily_execution_proxy_rows", 0),
                           "verified_execution_rows": build_manifest.get("verified_execution_rows", 0),
                           "historical_collection_warning": build_manifest.get("historical_collection_warning")},
        "deployment_status": "HISTORICAL_GATE_PASSED" if historical_gate else "DIAGNOSTIC_ONLY",
        "execution_mode": "paper_only", "targets": {"promotion_precision": 0.70, "large_loss_rate": 0.10,
                                                      "signals_per_active_week": [3, 5]},
        "limitations": [
            "no historical Level-2 order queue", "open confirmation fields may be unavailable",
            "forum popularity is a risk-only optional overlay", "main-force intent is not inferred",
            "daily execution labels assume a fill at limit after any OHLC touch and are an inferred upper-bound proxy, not tick-verified fills",
        ],
    }
    MODEL_PATH.write_text(json_text(artifact), encoding="utf-8")
    frozen_dir = RESULTS / "models"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp(artifact["trained_at"]).strftime("%Y%m%d_%H%M%S")
    (frozen_dir / f"model_{stamp}.json").write_text(json_text(artifact), encoding="utf-8")
    print(json_text({k: artifact[k] for k in ("market_days", "observations", "split", "blind", "deployment_status")}))


def heuristic_risks(row: pd.Series) -> tuple[float | None, float | None, list[str]]:
    reasons: list[str] = []
    gap = safe_float(row.get("auction_gap"))
    if math.isnan(gap):
        return None, None, ["竞价/开盘数据缺失，不能判断假强"]
    trap = 0.06
    if gap > 0.07:
        trap += 0.22; reasons.append("高开超过7%")
    if safe_float(row.get("auction_cancel_ratio")) > 0.45:
        trap += 0.22; reasons.append("竞价匹配额衰减")
    if safe_float(row.get("one_word_streak")) >= 2:
        trap += 0.12; reasons.append("连续一字首次换手风险")
    if safe_float(row.get("bomb_count")) >= 2:
        trap += 0.10; reasons.append("反复炸板")
    if safe_float(row.get("theme_sync_open")) < 0.35:
        trap += 0.15; reasons.append("板块不同步")
    if safe_float(row.get("below_vwap_expanding")) == 1:
        trap += 0.25; reasons.append("跌破VWAP且卖量扩大")
    if safe_float(row.get("failed_seal_twice")) == 1:
        trap += 0.30; reasons.append("两次冲板不封")
    if safe_float(row.get("crowding_score")) > 0.80:
        trap += 0.08; reasons.append("社区一致性拥挤")
    trap = min(0.95, trap)
    loss = min(0.90, 0.03 + trap * 0.35 + (0.20 if row.get("emotion_state") == "退潮" else 0)
               + (0.10 if safe_float(row.get("boards")) >= 5 else 0)
               + (0.25 if safe_float(row.get("announcement_risk")) >= 0.8 else 0))
    return trap, loss, reasons


def merge_open_snapshot(frame: pd.DataFrame, signal_date: str, asof: str) -> pd.DataFrame:
    path = DATA / "open_snapshots.csv"
    if not path.exists() or path.stat().st_size == 0:
        for feature in OPEN_FEATURES:
            frame[feature] = math.nan
        return frame
    snapshots = pd.read_csv(path, dtype={"code": str})
    snapshots = snapshots[(snapshots["signal_date"] == signal_date) & (snapshots["asof"] <= asof)]
    if snapshots.empty:
        for feature in OPEN_FEATURES:
            frame[feature] = math.nan
        return frame
    snapshots = snapshots.sort_values("asof").drop_duplicates("code", keep="last")
    keep = ["code", "data_status", *[x for x in OPEN_FEATURES if x in snapshots]]
    return frame.merge(snapshots[keep], on="code", how="left", suffixes=("", "_open"))


def emotion_playbook(state: str, profile: dict[str, Any]) -> dict[str, Any]:
    """Turn the emotion diagnosis into a falsifiable next-session plan."""
    policies = {
        "启动": ("CONDITIONAL_OPEN", "优先二板先锋和三板身份确认，仓位从低开始"),
        "升温": ("OPEN_FOR_CORE", "可做换手核心，后排必须有题材梯队支持"),
        "加速": ("CORE_ONLY", "只做可成交核心，拒绝弱分支和纯一字跟随"),
        "一致": ("DEFENSIVE_CORE_ONLY", "持筹优先，新增仓等待分歧换手确认"),
        "大分歧": ("WAIT_RESEAL", "不做首次冲板，等待核心分歧转一致"),
        "退潮": ("CLOSED", "停止连板接力，空间板仅作情绪锚"),
        "修复": ("PROBE_ONLY", "仅小仓验证最先修复的唯一核心，防二次退潮"),
    }
    permission, stance = policies.get(state, ("CLOSED", "状态无法识别，停止接力"))
    dimensions = {
        "breadth_health": safe_float(profile.get("breadth_health")),
        "relay_appetite": safe_float(profile.get("relay_appetite")),
        "ladder_health": safe_float(profile.get("ladder_health")),
        "loss_effect_health": safe_float(profile.get("loss_effect_health")),
    }
    available = [(name, value) for name, value in dimensions.items() if not math.isnan(value)]
    weak_links = [name for name, value in sorted(available, key=lambda x: x[1])[:2] if value < 0.50]
    if safe_float(profile.get("false_acceleration")) == 1:
        permission, stance = "CLOSED", "高度扩张被中位/亏钱效应否定，按假加速处理"
    if safe_float(profile.get("emotion_data_completeness")) < 0.60:
        permission = "DATA_LIMITED"
        stance += "；关键反馈数据不足，只允许条件观察"
    return {
        "permission": permission,
        "stance": stance,
        "weak_links": weak_links,
        "confirmation": [
            "昨日涨停溢价不转负且中位股无批量核按钮",
            "二进三/三进四至少一档维持正反馈",
            "核心换手后收复VWAP且同题材同步",
        ],
        "invalidation": [
            "空间板断板并扩散至中位股",
            "炸板率升至25%以上或跌停家数快速增加",
            "高开无量、两次冲板不封或跌破VWAP放量",
        ],
    }


def score_command(args: argparse.Namespace) -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("run train first")
    artifact = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    frame = pd.read_csv(DATA / "leader_candidates.csv", dtype={"code": str})
    signal_date = args.date or frame["date"].max()
    current = frame[frame["date"] == signal_date].copy().reset_index(drop=True)
    if current.empty:
        raise SystemExit(f"no candidates for {signal_date}")
    current = merge_open_snapshot(current, signal_date, args.asof)
    eod_p = predict_artifact(artifact["candidate_model"], current)
    p = eod_p.copy()
    eod_ensemble = artifact["candidate_model"].get("bootstrap_ensemble", {})
    probability_lower, probability_median, probability_upper = predict_bootstrap_ensemble(eod_ensemble, current)
    uncertainty_status = np.array([str(eod_ensemble.get("status", "UNAVAILABLE")) for _ in range(len(current))], dtype=object)
    eod_shift = distribution_shift(artifact["candidate_model"].get("feature_distribution", {}), current)
    distribution_status = np.array([item["status"] for item in eod_shift], dtype=object)
    distribution_fraction = np.array([item["ood_fraction"] for item in eod_shift], dtype=float)
    distribution_reasons = np.array([
        ",".join(item["extreme_features"] + item["missing_surprises"]) for item in eod_shift
    ], dtype=object)
    model_stage = np.array(["EOD" for _ in range(len(current))], dtype=object)
    open_artifact = artifact.get("open_candidate_model", {})
    if str(open_artifact.get("status", "")).startswith("TRAINED_DIAGNOSTIC"):
        open_required = open_artifact.get("required_live_features", ["auction_gap", "vwap_reclaimed", "theme_sync_open"])
        open_ready = current.reindex(columns=open_required).notna().all(axis=1).to_numpy()
        open_p = predict_artifact(open_artifact["fitted"], current)
        p[open_ready] = open_p[open_ready]
        open_ensemble = open_artifact["fitted"].get("bootstrap_ensemble", {})
        open_lower, open_median, open_upper = predict_bootstrap_ensemble(open_ensemble, current)
        probability_lower[open_ready] = open_lower[open_ready]
        probability_median[open_ready] = open_median[open_ready]
        probability_upper[open_ready] = open_upper[open_ready]
        uncertainty_status[open_ready] = str(open_ensemble.get("status", "UNAVAILABLE"))
        open_shift = distribution_shift(open_artifact["fitted"].get("feature_distribution", {}), current)
        for position in np.flatnonzero(open_ready):
            distribution_status[position] = open_shift[position]["status"]
            distribution_fraction[position] = open_shift[position]["ood_fraction"]
            distribution_reasons[position] = ",".join(
                open_shift[position]["extreme_features"] + open_shift[position]["missing_surprises"])
        model_stage[open_ready] = "OPEN_CONFIRMATION"
    eod_policy = artifact["candidate_model"]["threshold_policy"]
    eod_joint_policy = artifact["candidate_model"].get("joint_execution_policy", {})
    eod_diagnostic_threshold = float(eod_policy["diagnostic_threshold"])
    thresholds = np.full(len(current), safe_float(eod_joint_policy.get("deployment_threshold")))
    diagnostic_thresholds = np.full(len(current), eod_diagnostic_threshold)
    eod_uncertainty_ready = eod_ensemble.get("status") == "USABLE"
    eod_calibration_ready = artifact["candidate_model"].get("calibration_evidence", {}).get("status") == "USABLE"
    threshold_operational = np.full(
        len(current), bool(eod_policy.get("operational", False)
                           and eod_joint_policy.get("operational", False) and eod_uncertainty_ready
                           and eod_calibration_ready), dtype=bool)
    eod_abstain = "; ".join(x for x in (str(eod_policy.get("abstain_reason", "")),
                                          str(eod_joint_policy.get("abstain_reason", ""))) if x)
    if not eod_uncertainty_ready:
        eod_abstain = "; ".join(x for x in (eod_abstain, "bootstrap uncertainty evidence is not usable") if x)
    if not eod_calibration_ready:
        eod_abstain = "; ".join(x for x in (eod_abstain, "independent calibration evidence is not usable") if x)
    abstain_reasons = np.array([eod_abstain for _ in range(len(current))], dtype=object)
    if str(open_artifact.get("status", "")).startswith("TRAINED_DIAGNOSTIC"):
        open_policy = open_artifact["fitted"]["threshold_policy"]
        open_joint_policy = open_artifact["fitted"].get("joint_execution_policy", {})
        thresholds[open_ready] = safe_float(open_joint_policy.get("deployment_threshold"))
        diagnostic_thresholds[open_ready] = float(open_policy["diagnostic_threshold"])
        open_uncertainty_ready = open_artifact["fitted"].get("bootstrap_ensemble", {}).get("status") == "USABLE"
        open_calibration_ready = open_artifact["fitted"].get("calibration_evidence", {}).get("status") == "USABLE"
        threshold_operational[open_ready] = bool(open_policy.get("operational", False)
                                                and open_joint_policy.get("operational", False)
                                                and open_uncertainty_ready and open_calibration_ready
                                                and open_artifact.get("deployment_eligible", False))
        open_abstain = "; ".join(x for x in (str(open_policy.get("abstain_reason", "")),
                                              str(open_joint_policy.get("abstain_reason", ""))) if x)
        if not open_uncertainty_ready:
            open_abstain = "; ".join(x for x in (open_abstain, "bootstrap uncertainty evidence is not usable") if x)
        if not open_calibration_ready:
            open_abstain = "; ".join(x for x in (open_abstain, "independent calibration evidence is not usable") if x)
        if not open_artifact.get("deployment_eligible", False):
            open_abstain = "; ".join(x for x in (open_abstain,
                "daily-open-only model lacks verified VWAP and sector synchronization") if x)
        abstain_reasons[open_ready] = open_abstain
    false_model_p = optional_risk_probability(artifact.get("false_strength_model", {}), current)
    loss_model_p = optional_risk_probability(artifact.get("large_loss_model", {}), current)
    rows = []
    for index, row in current.iterrows():
        board = int(row["boards"])
        probability = float(p[index])
        threshold = safe_float(thresholds[index])
        diagnostic_threshold = float(diagnostic_thresholds[index])
        operational_threshold = bool(threshold_operational[index])
        lower_probability = safe_float(probability_lower[index])
        upper_probability = safe_float(probability_upper[index])
        uncertainty_ready = uncertainty_status[index] == "USABLE" and not math.isnan(lower_probability)
        in_distribution = distribution_status[index] == "IN_DISTRIBUTION"
        learned_trap = None if false_model_p is None else safe_float(false_model_p[index])
        learned_loss = None if loss_model_p is None else safe_float(loss_model_p[index])
        if learned_trap is not None and math.isnan(learned_trap):
            learned_trap = None
        if learned_loss is not None and math.isnan(learned_loss):
            learned_loss = None
        height_forecast = terminal_height_distribution(
            board, probability, artifact.get("height_survival_model", {}))
        height_lower_forecast = terminal_height_distribution(
            board, lower_probability, artifact.get("height_survival_model", {}))
        height_upper_forecast = terminal_height_distribution(
            board, upper_probability, artifact.get("height_survival_model", {}))
        if board >= 6:
            decision, risk_reasons = "EMOTION_ANCHOR", ["六板以上只作情绪锚/持筹管理"]
            trap, large, _ = heuristic_risks(row)
        elif safe_float(row.get("one_word_proxy")) == 1:
            decision, risk_reasons = "EMOTION_ANCHOR", ["一字锁死，无法验证真实成交"]
            trap, large, _ = heuristic_risks(row)
        elif row["emotion_state"] in {"退潮", "大分歧"}:
            decision, risk_reasons = "REJECT", [f"情绪阶段为{row['emotion_state']}"]
            trap, large, _ = heuristic_risks(row)
        elif safe_float(row.get("announcement_risk")) >= 0.8:
            decision, risk_reasons = "REJECT", ["公告/监管硬风险"]
            trap, large, _ = heuristic_risks(row)
        elif safe_float(row.get("valid_identity")) != 1:
            decision, risk_reasons = "REJECT", ["未通过先锋/主线/空间身份闸门"]
            trap, large, _ = heuristic_risks(row)
        else:
            trap, large, risk_reasons = heuristic_risks(row)
            if learned_trap is not None:
                trap = learned_trap
                risk_reasons.append("假强概率来自训练模型")
            if learned_loss is not None:
                large = learned_loss
                risk_reasons.append("大面概率来自训练模型")
            live_missing = trap is None or any(math.isnan(safe_float(row.get(f))) for f in ("auction_gap", "vwap_reclaimed", "theme_sync_open"))
            emotion_incomplete = safe_float(row.get("emotion_data_completeness")) < 0.75
            fragile_regime = safe_float(row.get("regime_fragility")) >= 0.68
            emotion_forecast_ready = str(row.get("emotion_forecast_status", "LOW_EVIDENCE")) == "USABLE"
            adverse_next_probability = safe_float(row.get("emotion_adverse_next_probability"))
            adverse_transition_risk = (not math.isnan(adverse_next_probability)
                                       and adverse_next_probability >= 0.45)
            rule_pass = (operational_threshold and probability >= threshold
                         and uncertainty_ready and lower_probability >= threshold and in_distribution
                         and not live_missing and trap <= 0.20 and large <= 0.10
                         and not emotion_incomplete and not fragile_regime
                         and emotion_forecast_ready and not adverse_transition_risk
                         and safe_float(row.get("false_acceleration")) != 1
                         and safe_float(row.get("below_vwap_expanding")) != 1
                         and safe_float(row.get("failed_seal_twice")) != 1)
            if rule_pass and artifact["deployment_status"] != "DIAGNOSTIC_ONLY":
                decision = "ENTER"
            else:
                decision = "WATCH"
                if operational_threshold and probability < threshold:
                    risk_reasons.append("晋级概率未过验证阈值")
                if not operational_threshold:
                    risk_reasons.append("部署阈值未通过证据闸门，模型自动弃权")
                if not uncertainty_ready:
                    risk_reasons.append("概率区间证据不足，模型自动弃权")
                elif operational_threshold and lower_probability < threshold:
                    risk_reasons.append("晋级概率区间下界未过部署阈值")
                if not in_distribution:
                    risk_reasons.append(f"特征分布状态为{distribution_status[index]}，禁止外推")
                if live_missing:
                    risk_reasons.append("开盘确认字段不完整")
                if emotion_incomplete:
                    risk_reasons.append("市场情绪反馈字段完整度不足75%")
                if fragile_regime:
                    risk_reasons.append("情绪结构脆弱度过高")
                if not emotion_forecast_ready:
                    risk_reasons.append("当前情绪状态历史转移不足20次，下一状态预测不可用于入场")
                if adverse_transition_risk:
                    risk_reasons.append(f"次日大分歧/退潮概率{adverse_next_probability:.1%}，超过45%风险线")
                if artifact["deployment_status"] == "DIAGNOSTIC_ONLY":
                    risk_reasons.append("模型尚未通过历史上线闸门")
        rows.append({
            "date": signal_date, "code": zfill_code(row["code"]), "name": row["name"], "boards": board,
            "emotion_state": row["emotion_state"], "decision": decision,
            "promotion_probability": probability,
            "promotion_probability_lower": lower_probability,
            "promotion_probability_median": safe_float(probability_median[index]),
            "promotion_probability_upper": upper_probability,
            "probability_interval_width": (upper_probability - lower_probability
                                           if not math.isnan(lower_probability) and not math.isnan(upper_probability) else math.nan),
            "uncertainty_status": uncertainty_status[index],
            "distribution_status": distribution_status[index],
            "distribution_ood_fraction": distribution_fraction[index],
            "distribution_reasons": distribution_reasons[index],
            "probability_status": ("VALIDATED" if artifact["deployment_status"] != "DIAGNOSTIC_ONLY"
                                   and operational_threshold and in_distribution else "DIAGNOSTIC"),
            "promotion_threshold": threshold, "diagnostic_threshold": diagnostic_threshold,
            "threshold_operational": operational_threshold,
            "threshold_abstain_reason": "" if operational_threshold else abstain_reasons[index],
            "false_strength_probability": trap, "large_loss_probability": large,
            "terminal_board_mode": height_forecast.get("mode_terminal_board"),
            "terminal_board_expected": height_forecast.get("expected_terminal_board"),
            "terminal_board_expected_lower": height_lower_forecast.get("expected_terminal_board"),
            "terminal_board_expected_upper": height_upper_forecast.get("expected_terminal_board"),
            "terminal_board_interval_80": height_forecast.get("terminal_board_interval_80"),
            "probability_at_least_two_more_boards": height_forecast.get("probability_at_least_two_more"),
            "terminal_board_distribution": height_forecast.get("distribution", {}),
            "terminal_height_status": height_forecast.get("status", "UNAVAILABLE"),
            "emotion_quality_score": safe_float(row.get("emotion_score")),
            "regime_fragility": safe_float(row.get("regime_fragility")),
            "emotion_data_completeness": safe_float(row.get("emotion_data_completeness")),
            "emotion_adverse_next_probability": safe_float(row.get("emotion_adverse_next_probability")),
            "emotion_adverse_markov_probability": safe_float(row.get("emotion_adverse_markov_probability")),
            "emotion_adverse_challenger_probability": safe_float(row.get("emotion_adverse_challenger_probability")),
            "emotion_forecast_source": row.get("emotion_forecast_source", "MARKOV_BASELINE"),
            "emotion_challenger_status": row.get("emotion_challenger_status", "INSUFFICIENT_HISTORY"),
            "emotion_champion_evaluation_n": safe_float(row.get("emotion_champion_evaluation_n")),
            "emotion_challenger_vs_markov_brier_skill": safe_float(
                row.get("emotion_challenger_vs_markov_brier_skill")),
            "emotion_supportive_next_probability": safe_float(row.get("emotion_supportive_next_probability")),
            "emotion_persistence_probability": safe_float(row.get("emotion_persistence_probability")),
            "emotion_transition_entropy": safe_float(row.get("emotion_transition_entropy")),
            "emotion_transition_evidence": safe_float(row.get("emotion_transition_evidence")),
            "emotion_forecast_status": row.get("emotion_forecast_status", "LOW_EVIDENCE"),
            "emotion_state_age": safe_float(row.get("emotion_state_age")),
            "model_stage": model_stage[index],
            "buy_trigger": "9:25-10:00板块同步、缩量回落、收复VWAP或真实换手回封",
            "cancel_trigger": "高开>7%回落、跌破VWAP放量、两次冲板不封、板块核心破板",
            "risk_reasons": "；".join(dict.fromkeys(risk_reasons)),
            "model_status": artifact["deployment_status"], "execution_mode": "paper_only",
        })
    out = pd.DataFrame(rows).sort_values(["decision", "promotion_probability"], ascending=[True, False])
    stem = f"current_candidates_{signal_date}_{args.asof.replace(':', '')}"
    score_csv_path = RESULTS / f"{stem}.csv"
    score_json_path = RESULTS / f"{stem}.json"
    out.to_csv(score_csv_path, index=False, encoding="utf-8-sig")
    current_market = current.iloc[0].to_dict()
    profile_keys = [
        "emotion_score", "breadth_health", "relay_appetite", "ladder_health",
        "loss_effect_health", "crowding_risk", "regime_fragility",
        "false_acceleration", "emotion_data_completeness", "emotion_risk_level",
        "emotion_warnings", "emotion_adverse_next_probability",
        "emotion_adverse_markov_probability", "emotion_adverse_challenger_probability",
        "emotion_forecast_source", "emotion_challenger_status", "emotion_champion_evaluation_n",
        "emotion_challenger_vs_markov_brier_skill",
        "emotion_supportive_next_probability", "emotion_persistence_probability",
        "emotion_transition_entropy", "emotion_transition_evidence",
        "emotion_transition_confidence", "emotion_forecast_status", "emotion_state_age",
    ]
    emotion_profile = {key: current_market.get(key) for key in profile_keys}
    if pd.isna(emotion_profile.get("emotion_warnings")):
        emotion_profile["emotion_warnings"] = ""
    quality = {
        "verified": "daily limit-up ladder and close fields",
        "market_data_status": current_market.get("market_data_status", "partial"),
        "emotion_data_completeness": current_market.get("emotion_data_completeness"),
        "unavailable": "Level-2 queue and intent; open fields unless imported/captured",
        "decision": "conditional paper plan only", "asof": args.asof,
    }
    current_state = str(current.iloc[0]["emotion_state"])
    next_state_probabilities = {
        state: safe_float(current.iloc[0].get(column))
        for state, column in STATE_PROBABILITY_COLUMNS.items()
    }
    next_state_probabilities = {
        state: probability for state, probability in next_state_probabilities.items()
        if not math.isnan(probability)
    }
    transition_note = {
        "observed_transitions": int(safe_float(current.iloc[0].get("emotion_transition_evidence")))
        if not math.isnan(safe_float(current.iloc[0].get("emotion_transition_evidence"))) else 0,
        "confidence": current.iloc[0].get("emotion_transition_confidence", "LOW"),
        "status": current.iloc[0].get("emotion_forecast_status", "LOW_EVIDENCE"),
        "method": "as-of-date online Dirichlet transition forecast; no future states",
    }
    report = {"date": signal_date, "data_quality": quality, "model_status": artifact["deployment_status"],
              "validation_integrity": artifact.get("validation_integrity", {}),
              "threshold_evidence": eod_policy,
              "joint_execution_threshold_evidence": eod_joint_policy,
              "height_survival_evidence": artifact.get("height_survival_backtest", {}),
              "calibration_evidence": artifact["candidate_model"].get("calibration_evidence", {}),
              "uncertainty_evidence": {key: value for key, value in eod_ensemble.items() if key != "members"},
              "distribution_summary": dict(pd.Series(distribution_status).value_counts()),
              "current_emotion_state": current_state,
              "emotion_profile": emotion_profile,
              "emotion_playbook": emotion_playbook(current_state, emotion_profile),
              "next_emotion_probabilities": dict(sorted(next_state_probabilities.items(), key=lambda x: x[1], reverse=True)),
              "transition_evidence": transition_note,
              "candidates": out.to_dict("records")}
    score_json_path.write_text(json_text(report), encoding="utf-8")
    if getattr(args, "record_forward", False):
        registration = register_forward_snapshot(
            signal_date, args.asof, out, score_json_path, score_csv_path)
        print(json_text({"forward_registration": registration}))
    print(out.to_string(index=False, formatters={"promotion_probability": "{:.1%}".format}))


def execution_metrics(selected: pd.DataFrame) -> dict[str, Any]:
    if selected.empty:
        return {"signals": 0}
    returns = pd.to_numeric(selected["net_return"], errors="coerce").dropna()
    blocked = pd.to_numeric(selected.get("sell_blocked", math.nan), errors="coerce")
    marks_all = pd.to_numeric(selected.get("mark_to_market_return", math.nan), errors="coerce")
    entry = pd.to_numeric(selected.get("entry_price", math.nan), errors="coerce")
    if not hasattr(entry, "notna"):
        entry = pd.Series(math.nan, index=selected.index)
    executed = entry.notna() | pd.to_numeric(selected["net_return"], errors="coerce").notna()
    if hasattr(marks_all, "notna"):
        executed |= marks_all.notna()
    if hasattr(blocked, "eq"):
        executed |= blocked.eq(1)
    executed_frame = selected.loc[executed]
    result: dict[str, Any] = {"signals": int(len(selected)), "selection_signals": int(len(selected)),
                              "executable_trades": int(executed.sum()),
                              "condition_not_triggered": int((~executed).sum()),
                              "promotion_precision": float(selected["promoted_next_day"].mean()),
                              "executed_promotion_precision": (float(executed_frame["promoted_next_day"].mean())
                                                               if len(executed_frame) else None),
                              "sell_blocked_count": int((blocked == 1).sum()) if hasattr(blocked, "sum") else 0,
                              "unresolved_return_count": int((executed & pd.to_numeric(
                                  selected["net_return"], errors="coerce").isna()).sum())}
    if "execution_status" in selected:
        result["execution_status_counts"] = selected["execution_status"].fillna("unavailable").astype(str).value_counts().to_dict()
    if "execution_source" in selected:
        sources = selected["execution_source"].fillna("unavailable").astype(str).value_counts().to_dict()
        result["execution_source_counts"] = sources
        result["verified_execution_count"] = int(sources.get("user_import", 0))
        result["execution_quality"] = ("VERIFIED" if sources.get("user_import", 0) == len(selected)
                                       else "MIXED" if sources.get("user_import", 0)
                                       else "INFERRED_DAILY_OHLC")
    losses = pd.to_numeric(selected["large_loss"], errors="coerce").dropna()
    result["large_loss_rate"] = float(losses.mean()) if len(losses) else None
    marks = marks_all.dropna() if hasattr(marks_all, "dropna") else pd.Series(dtype=float)
    result["execution_label_coverage"] = float(len(marks) / max(1, int(executed.sum())))
    if len(marks):
        result["average_mark_to_market_return"] = float(marks.mean())
        result["worst_mark_to_market_return"] = float(marks.min())
    if len(returns):
        equity = (1 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1
        wins, fails = returns[returns > 0], returns[returns <= 0]
        result.update({"compound_return": float(equity.iloc[-1] - 1), "max_drawdown": float(drawdown.min()),
                       "average_trade": float(returns.mean()), "profit_factor": float(wins.sum() / abs(fails.sum())) if fails.sum() else None})
    return result


def ablation_report(train: pd.DataFrame, calibration: pd.DataFrame,
                    validation: pd.DataFrame, blind: pd.DataFrame) -> list[dict[str, Any]]:
    order = ["baseline", "emotion", "theme", "individual", "open", "risk"]
    features: list[str] = []
    rows = []
    for group in order:
        features += FEATURE_GROUPS[group]
        model = RegularizedLogistic.fit(train, "promoted_next_day", features, epochs=1800)
        calibrator = PlattCalibrator.fit(model.predict(calibration), calibration["promoted_next_day"].to_numpy(float))
        val_p = calibrator.predict(model.predict(validation))
        threshold_policy = choose_threshold(validation.reset_index(drop=True), val_p)
        blind_p = calibrator.predict(model.predict(blind))
        diagnostic_selected = top_one_policy(
            blind.reset_index(drop=True), blind_p, threshold_policy["diagnostic_threshold"])
        selected = (diagnostic_selected if threshold_policy.get("operational", False)
                    else np.zeros(len(blind), dtype=bool))
        y = blind["promoted_next_day"].to_numpy(float)
        rows.append({"through_group": group, "feature_count": len(features), "blind_auc": auc_score(y, blind_p),
                      "blind_brier": float(np.mean((blind_p-y)**2)), "signals": int(selected.sum()),
                      "precision": float(y[selected].mean()) if selected.any() else None,
                      "threshold_operational": threshold_policy.get("operational", False),
                      "diagnostic_signals": int(diagnostic_selected.sum()),
                      "diagnostic_precision": float(y[diagnostic_selected].mean()) if diagnostic_selected.any() else None})
    return rows


def walk_forward_partitions(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Expanding-origin folds with calibration/validation and three embargoes."""
    dates = np.array(sorted(frame["date"].unique()))
    segment = max(2, int(len(dates) * 0.10))
    endpoints = sorted(set(min(len(dates), max(1, math.ceil(len(dates) * share)))
                           for share in (0.70, 0.85, 1.00)))
    partitions: list[dict[str, Any]] = []
    for end in endpoints:
        test_start = end - segment
        embargo_3 = test_start - 1
        validation_end = embargo_3
        validation_start = validation_end - segment
        embargo_2 = validation_start - 1
        calibration_end = embargo_2
        calibration_start = calibration_end - segment
        embargo_1 = calibration_start - 1
        if embargo_1 < 5:
            continue
        partitions.append({
            "train_dates": dates[:embargo_1], "embargo_1": str(dates[embargo_1]),
            "calibration_dates": dates[calibration_start:calibration_end],
            "embargo_2": str(dates[embargo_2]),
            "validation_dates": dates[validation_start:validation_end],
            "embargo_3": str(dates[embargo_3]), "test_dates": dates[test_start:end],
        })
    return partitions


def walk_forward_report(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold, partition in enumerate(walk_forward_partitions(frame), 1):
        train = frame[frame["date"].isin(partition["train_dates"])]
        calibration = frame[frame["date"].isin(partition["calibration_dates"])]
        validation = frame[frame["date"].isin(partition["validation_dates"])]
        test = frame[frame["date"].isin(partition["test_dates"])]
        if any(part.empty for part in (train, calibration, validation, test)) or train["promoted_next_day"].nunique() < 2:
            continue
        model = RegularizedLogistic.fit(train, "promoted_next_day", EOD_FEATURES, epochs=1600)
        calibrator = PlattCalibrator.fit(model.predict(calibration), calibration["promoted_next_day"].to_numpy(float))
        validation_p = calibrator.predict(model.predict(validation))
        threshold = choose_threshold(validation.reset_index(drop=True), validation_p)
        joint_threshold = choose_joint_execution_threshold(validation.reset_index(drop=True), validation_p)
        p = calibrator.predict(model.predict(test))
        joint_diagnostic = safe_float(joint_threshold.get("diagnostic_threshold"))
        joint_mask = (top_one_policy(test.reset_index(drop=True), p, joint_diagnostic)
                      if not math.isnan(joint_diagnostic) else np.zeros(len(test), dtype=bool))
        joint_metrics = execution_metrics(test.reset_index(drop=True).loc[joint_mask].copy())
        profile = build_distribution_profile(train, EOD_FEATURES)
        shifts = distribution_shift(profile, test)
        rows.append({
            "fold": fold,
            "train": [str(partition["train_dates"][0]), str(partition["train_dates"][-1])],
            "embargo_1": partition["embargo_1"],
            "calibration": [str(partition["calibration_dates"][0]), str(partition["calibration_dates"][-1])],
            "embargo_2": partition["embargo_2"],
            "validation": [str(partition["validation_dates"][0]), str(partition["validation_dates"][-1])],
            "embargo_3": partition["embargo_3"],
            "test": [str(partition["test_dates"][0]), str(partition["test_dates"][-1])],
            "threshold_operational": threshold.get("operational", False),
            "joint_threshold_operational": joint_threshold.get("operational", False),
            "joint_threshold_abstain_reason": joint_threshold.get("abstain_reason", ""),
            "joint_test_executable_trades": joint_metrics.get("executable_trades", 0),
            "joint_test_executed_precision": joint_metrics.get("executed_promotion_precision"),
            "joint_test_large_loss_rate": joint_metrics.get("large_loss_rate"),
            "diagnostic_threshold": threshold.get("diagnostic_threshold"),
            "drift_or_ood_rate": float(np.mean([item["status"] != "IN_DISTRIBUTION" for item in shifts])),
            **metric_block(test, p),
        })
    return rows


def reliability_by_segment(frame: pd.DataFrame, probabilities: np.ndarray,
                           columns: tuple[str, ...] = ("boards", "emotion_state")) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    working = frame.reset_index(drop=True).copy()
    working["_probability"] = probabilities
    for column in columns:
        segments: list[dict[str, Any]] = []
        if column not in working:
            result[column] = segments
            continue
        for value, group in working.groupby(column, dropna=False):
            indexes = group.index.to_numpy()
            metrics = metric_block(group, probabilities[indexes])
            n = int(len(group))
            segments.append({"segment": str(value), "confidence": "LOW" if n < 30 else ("MEDIUM" if n < 100 else "HIGH"),
                             **metrics})
        result[column] = segments
    return result


def segment_reliability_gate(reliability: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Pre-registered deployment gate: reliability must survive height and active-regime splits."""
    board_rows = {row["segment"]: row for row in reliability.get("boards", [])}
    board_checks = []
    for board in (2, 3, 4, 5):
        row = board_rows.get(str(board), {})
        passed = int(row.get("n", 0)) >= 20 and safe_float(row.get("brier_skill_score")) > 0
        board_checks.append({"segment": str(board), "minimum_n": 20, "n": int(row.get("n", 0)),
                             "brier_skill_score": row.get("brier_skill_score"), "passed": passed})

    emotion_rows = {row["segment"]: row for row in reliability.get("emotion_state", [])}
    emotion_checks = []
    for state in sorted(ACTIVE_STATES):
        row = emotion_rows.get(state, {})
        if not row:
            continue
        passed = int(row.get("n", 0)) >= 20 and safe_float(row.get("brier_skill_score")) > 0
        emotion_checks.append({"segment": state, "minimum_n": 20, "n": int(row.get("n", 0)),
                               "brier_skill_score": row.get("brier_skill_score"), "passed": passed})
    active_state_coverage = len(emotion_checks) >= 3

    distribution_rows = {row["segment"]: row for row in reliability.get("distribution_status", [])}
    in_distribution = distribution_rows.get("IN_DISTRIBUTION", {})
    distribution_check = {
        "segment": "IN_DISTRIBUTION", "minimum_n": 100,
        "n": int(in_distribution.get("n", 0)),
        "brier_skill_score": in_distribution.get("brier_skill_score"),
    }
    distribution_check["passed"] = (
        distribution_check["n"] >= 100 and safe_float(distribution_check["brier_skill_score"]) > 0)
    passed = (all(item["passed"] for item in board_checks)
              and active_state_coverage
              and all(item["passed"] for item in emotion_checks)
              and distribution_check["passed"])
    return {"passed": passed, "board_checks": board_checks, "active_emotion_checks": emotion_checks,
            "active_state_coverage": active_state_coverage, "distribution_check": distribution_check,
            "rule": "blind: each 2-5 board n>=20 and positive Brier skill; >=3 observed active states each n>=20 and positive skill; in-distribution n>=100 and positive skill"}


def backtest_command(args: argparse.Namespace) -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("run train first")
    artifact = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    frame = pd.read_csv(DATA / "leader_candidates.csv", dtype={"code": str})
    frame = frame[frame["promoted_next_day"].notna() & frame["boards"].between(2, 5)].reset_index(drop=True)
    train, calibration, validation, blind, split = chronological_split(frame)
    p = predict_artifact(artifact["candidate_model"], blind)
    threshold_policy = artifact["candidate_model"]["threshold_policy"]
    joint_policy = artifact["candidate_model"].get("joint_execution_policy", {})
    diagnostic_threshold = float(threshold_policy["diagnostic_threshold"])
    diagnostic_mask = top_one_policy(blind.reset_index(drop=True), p, diagnostic_threshold)
    joint_diagnostic_threshold = safe_float(joint_policy.get("diagnostic_threshold"))
    joint_diagnostic_mask = (top_one_policy(blind.reset_index(drop=True), p, joint_diagnostic_threshold)
                             if not math.isnan(joint_diagnostic_threshold)
                             else np.zeros(len(blind), dtype=bool))
    ensemble = artifact["candidate_model"].get("bootstrap_ensemble", {})
    probability_lower, probability_median, probability_upper = predict_bootstrap_ensemble(ensemble, blind)
    shift = distribution_shift(artifact["candidate_model"].get("feature_distribution", {}), blind)
    in_distribution = np.array([item["status"] == "IN_DISTRIBUTION" for item in shift], dtype=bool)
    calibration_usable = artifact["candidate_model"].get("calibration_evidence", {}).get("status") == "USABLE"
    policy_operational = bool(threshold_policy.get("operational", False)
                              and joint_policy.get("operational", False)
                              and ensemble.get("status") == "USABLE" and calibration_usable)
    selected_mask = (top_one_policy(blind.reset_index(drop=True), p,
                                    float(joint_policy["deployment_threshold"]),
                                    minimum_probabilities=probability_lower,
                                    allowed_mask=in_distribution)
                     if policy_operational else np.zeros(len(blind), dtype=bool))
    selected = blind.reset_index(drop=True)[selected_mask].copy()
    selected["promotion_probability"] = p[selected_mask]
    diagnostic_selected = blind.reset_index(drop=True)[diagnostic_mask].copy()
    diagnostic_selected["promotion_probability"] = p[diagnostic_mask]
    joint_diagnostic_selected = blind.reset_index(drop=True)[joint_diagnostic_mask].copy()
    joint_diagnostic_selected["promotion_probability"] = p[joint_diagnostic_mask]
    open_stage_report: dict[str, Any] = {"status": "UNAVAILABLE"}
    open_artifact = artifact.get("open_candidate_model", {})
    if str(open_artifact.get("status", "")).startswith("TRAINED_DIAGNOSTIC"):
        open_frame = merge_training_snapshots(frame)
        open_required = open_artifact.get("required_live_features", ["auction_gap", "vwap_reclaimed", "theme_sync_open"])
        open_usable = open_frame[open_frame.reindex(columns=open_required).notna().all(axis=1)
                                 & open_frame["promoted_next_day"].notna()].copy()
        _, _, _, open_blind, open_split = chronological_split(open_usable)
        open_p = predict_artifact(open_artifact["fitted"], open_blind)
        open_threshold = float(open_artifact["fitted"]["threshold_policy"]["diagnostic_threshold"])
        open_joint_policy = open_artifact["fitted"].get("joint_execution_policy", {})
        open_joint_threshold = safe_float(open_joint_policy.get("diagnostic_threshold"))
        open_mask = (top_one_policy(open_blind.reset_index(drop=True), open_p, open_joint_threshold)
                     if not math.isnan(open_joint_threshold) else np.zeros(len(open_blind), dtype=bool))
        open_selected = open_blind.reset_index(drop=True)[open_mask].copy()
        open_selected["promotion_probability"] = open_p[open_mask]
        open_selected.to_csv(RESULTS / "open_stage_diagnostic_trades.csv", index=False, encoding="utf-8-sig")
        open_stage_report = {
            "status": open_artifact.get("status"), "feature_scope": open_artifact.get("feature_scope"),
            "deployment_eligible": open_artifact.get("deployment_eligible", False),
            "required_live_features": open_required, "split": open_split,
            "probability_metrics": metric_block(open_blind, open_p, open_threshold),
            "joint_execution_policy": open_joint_policy,
            "diagnostic_policy": execution_metrics(open_selected),
            "warning": "daily official open is valid only for the 09:30+ stage; VWAP and sector sync remain unverified",
        }
    highest = blind.loc[blind.groupby("date")["boards"].idxmax()].copy()
    reliability_frame = blind.reset_index(drop=True).copy()
    reliability_frame["distribution_status"] = [item["status"] for item in shift]
    segment_reliability = reliability_by_segment(
        reliability_frame, p, ("boards", "emotion_state", "distribution_status"))
    report = {
        "split": split, "policy": execution_metrics(selected),
        "validation_integrity": artifact.get("validation_integrity", {}),
        "calibration_evidence": artifact["candidate_model"].get("calibration_evidence", {}),
        "joint_execution_policy": joint_policy,
        "policy_abstained": not policy_operational,
        "policy_abstain_reason": (joint_policy.get("abstain_reason", "")
                                  or threshold_policy.get("abstain_reason", "")
                                  or ("bootstrap uncertainty evidence is not usable"
                                      if ensemble.get("status") != "USABLE" else "")
                                  or ("independent calibration evidence is not usable"
                                      if not calibration_usable else "")),
        "diagnostic_policy": execution_metrics(diagnostic_selected),
        "joint_diagnostic_policy": execution_metrics(joint_diagnostic_selected),
        "open_stage_diagnostic": open_stage_report,
        "height_survival_backtest": artifact.get("height_survival_backtest", {}),
        "baseline_highest_board": execution_metrics(highest),
        "probability_metrics": metric_block(blind, p, diagnostic_threshold),
        "uncertainty": {"status": ensemble.get("status", "UNAVAILABLE"),
                        "mean_interval_width": float(np.nanmean(probability_upper - probability_lower)),
                        "median_absolute_point_shift": float(np.nanmean(np.abs(probability_median - p)))},
        "distribution": {"in_distribution_rows": int(in_distribution.sum()),
                         "drift_or_ood_rows": int((~in_distribution).sum())},
        "segment_reliability": segment_reliability,
        "segment_reliability_gate": segment_reliability_gate(segment_reliability),
        "emotion_forecast_backtest": artifact.get("emotion_forecast_backtest", {}),
        "emotion_forecast_gate": artifact.get("emotion_forecast_gate", {}),
        "ablation": ablation_report(train, calibration, validation, blind),
        "walk_forward": walk_forward_report(frame),
        "integrity": {
            "future_features": "forbidden; labels separated by date", "embargo": "one trading date at each split",
            "fills": "daily proxy enters at limit only after an OHLC touch and excludes one-price limit-up; no tick/queue proof",
            "fees": "0.30% round-trip in verified or daily-proxy returns",
            "execution_quality": "INFERRED_DAILY_OHLC unless execution_source is user_import",
            "status": artifact["deployment_status"],
        },
    }
    selected.to_csv(RESULTS / "blind_selected_trades.csv", index=False, encoding="utf-8-sig")
    diagnostic_selected.to_csv(RESULTS / "blind_diagnostic_trades.csv", index=False, encoding="utf-8-sig")
    joint_diagnostic_selected.to_csv(RESULTS / "blind_joint_diagnostic_trades.csv", index=False, encoding="utf-8-sig")
    (RESULTS / "backtest_report.json").write_text(json_text(report), encoding="utf-8")
    report_title = ("# V3一次性时间盲测报告"
                    if report["validation_integrity"].get("status") == "FRESH_BLIND"
                    else "# V3开发时间留出评估报告")
    evaluation_label = ("盲测" if report["validation_integrity"].get("status") == "FRESH_BLIND"
                        else "开发留出")
    lines = [report_title, "", f"- 模型状态：`{artifact['deployment_status']}`",
             f"- 时间留出完整性：`{report['validation_integrity'].get('status', 'UNAVAILABLE')}`，"
             f"复用次数 {report['validation_integrity'].get('reuse_count_including_current')}",
             f"- {evaluation_label}部署信号：{report['policy'].get('signals', 0)}",
             f"- 诊断信号：{report['diagnostic_policy'].get('signals', 0)}",
             f"- 诊断代理成交：{report['diagnostic_policy'].get('executable_trades', 0)}；"
             f"条件未触发：{report['diagnostic_policy'].get('condition_not_triggered', 0)}",
             f"- 开盘增强层：`{report['open_stage_diagnostic'].get('status', 'UNAVAILABLE')}`；"
             f"部署资格：{report['open_stage_diagnostic'].get('deployment_eligible', False)}",
             f"- 晋级/大面联合闸门：{report['joint_execution_policy'].get('operational', False)}（"
             f"{report['joint_execution_policy'].get('abstain_reason', '')}）",
             f"- 终板高度预测：{report['height_survival_backtest'].get('status', 'UNAVAILABLE')}；"
             f"MAE {report['height_survival_backtest'].get('mode_mae')}",
             f"- 自动弃权：{report['policy_abstained']}（{report['policy_abstain_reason']}）",
             f"- 晋级精确率：{report['policy'].get('promotion_precision')}",
             f"- 大面率：{report['policy'].get('large_loss_rate')}",
             f"- Brier技能分：{report['probability_metrics'].get('brier_skill_score')}",
             f"- 5分箱ECE：{report['probability_metrics'].get('ece_5bin')}",
             f"- Bootstrap区间：{report['uncertainty'].get('status')}，平均宽度 {report['uncertainty'].get('mean_interval_width')}",
             f"- 独立校准集：{report['calibration_evidence'].get('status')}，{report['calibration_evidence'].get('dates')}日/{report['calibration_evidence'].get('rows')}条",
             f"- 分布漂移/OOD：{report['distribution'].get('drift_or_ood_rows')}条",
             f"- 情绪逆风预测：{report['emotion_forecast_backtest'].get('status')}，"
             f"{report['emotion_forecast_backtest'].get('n')}次，Brier技能分 "
             f"{report['emotion_forecast_backtest'].get('adverse_brier_skill_score')}",
             f"- 情绪挑战者：{report['emotion_forecast_backtest'].get('supervised_challenger', {}).get('n', 0)}次样本外预测，"
             f"冠军接管{report['emotion_forecast_backtest'].get('supervised_champion_rows', 0)}次",
             "", "## 因子消融", "",
             "| 截止因子组 | 因子数 | 盲测AUC | Brier | 部署信号 | 诊断信号 | 诊断精确率 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["ablation"]:
        lines.append(f"| {row['through_group']} | {row['feature_count']} | {row['blind_auc']} | {row['blind_brier']:.3f} | {row['signals']} | {row['diagnostic_signals']} | {row['diagnostic_precision']} |")
    lines += ["", "## 分层可靠性", "",
              f"- 分层闸门：{report['segment_reliability_gate']['passed']}",
              "- 规则：各板位和活跃情绪阶段必须分别满足最低样本且Brier技能分为正；漂移样本禁止ENTER。", "",
              "| 维度 | 分层 | 样本 | 置信度 | Brier技能分 | ECE |",
              "|---|---|---:|---|---:|---:|"]
    for dimension, rows in report["segment_reliability"].items():
        for row in rows:
            lines.append(f"| {dimension} | {row['segment']} | {row['n']} | {row['confidence']} | {row['brier_skill_score']} | {row['ece_5bin']} |")
    lines += ["", "结果仅用于研究；未通过历史与前向闸门时不得解释为实盘胜率。", ""]
    (RESULTS / "backtest_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json_text(report))


def freeze_model_command(args: argparse.Namespace) -> None:
    """Seal the exact code/model pair before any new forward labels are viewed."""
    ensure_dirs()
    if not MODEL_PATH.exists():
        raise SystemExit("run train first")
    model_sha = sha256_file(MODEL_PATH)
    code_sha = sha256_file(Path(__file__))
    if ACTIVE_FREEZE_PATH.exists() and not getattr(args, "new_freeze", False):
        existing = json.loads(ACTIVE_FREEZE_PATH.read_text(encoding="utf-8"))
        if existing.get("model_sha256") == model_sha and existing.get("code_sha256") == code_sha:
            print(json_text({**existing, "reused_existing_freeze": True}))
            return
        raise SystemExit("active freeze differs from current code/model; use --new-freeze only for a predeclared new cycle")
    artifact = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    created_at = datetime.now().isoformat(timespec="seconds")
    candidate_data_end = artifact.get("data_end")
    candidate_path = DATA / "leader_candidates.csv"
    candidate_data_sha256 = None
    if candidate_path.exists():
        candidate_dates = pd.read_csv(candidate_path, usecols=["date"])
        if not candidate_dates.empty:
            candidate_data_end = str(candidate_dates["date"].max())
        candidate_data_sha256 = sha256_file(candidate_path)
    basis = {
        "model_sha256": model_sha,
        "code_sha256": code_sha,
        "created_at": created_at,
        "data_end": artifact.get("data_end"),
        "candidate_data_end_at_freeze": candidate_data_end,
        "candidate_data_sha256_at_freeze": candidate_data_sha256,
        "probability_engine_version": artifact.get("probability_engine_version"),
    }
    freeze_id = canonical_hash(basis)[:16]
    manifest = {
        "freeze_id": freeze_id,
        **basis,
        "model_deployment_status": artifact.get("deployment_status", "DIAGNOSTIC_ONLY"),
        "validation_integrity_status": artifact.get("validation_integrity", {}).get("status"),
        "fresh_signal_date_rule": (
            f"signal_date must be later than candidate data {candidate_data_end} "
            f"and cannot predate freeze day {created_at[:10]}"),
        "assurance": "LOCAL_TAMPER_EVIDENT_ONLY",
        "warning": "local hashes detect later changes but are not an external trusted timestamp",
    }
    freeze_path = FORWARD / f"freeze_{freeze_id}.json"
    freeze_path.write_text(json_text(manifest), encoding="utf-8")
    ACTIVE_FREEZE_PATH.write_text(json_text(manifest), encoding="utf-8")
    append_forward_event({
        "event_type": "MODEL_FREEZE", "freeze_id": freeze_id,
        "manifest_file": str(freeze_path.relative_to(ROOT)),
        "manifest_sha256": sha256_file(freeze_path),
        "model_sha256": model_sha, "code_sha256": code_sha,
        "data_end": artifact.get("data_end"), "candidate_data_end_at_freeze": candidate_data_end,
        "candidate_data_sha256_at_freeze": candidate_data_sha256,
    })
    print(json_text(manifest))


def active_freeze(active_path: Path = ACTIVE_FREEZE_PATH) -> dict[str, Any]:
    if not active_path.exists():
        raise ValueError("no active model freeze; run freeze before recording forward signals")
    return json.loads(active_path.read_text(encoding="utf-8"))


def validate_active_freeze(manifest: dict[str, Any], model_path: Path = MODEL_PATH,
                           code_path: Path | None = None) -> None:
    code_path = code_path or Path(__file__)
    if not model_path.exists() or not code_path.exists():
        raise ValueError("frozen model or code file is missing")
    if sha256_file(model_path) != manifest.get("model_sha256"):
        raise ValueError("current model differs from active freeze; refusing a new signal")
    if sha256_file(code_path) != manifest.get("code_sha256"):
        raise ValueError("current code differs from active freeze; refusing a new signal")


def verify_active_freeze_binding(manifest: dict[str, Any], active_path: Path = ACTIVE_FREEZE_PATH,
                                 ledger_path: Path = FORWARD_LEDGER_PATH,
                                 root: Path = ROOT) -> dict[str, Any]:
    ledger = verify_forward_ledger(ledger_path, root)
    errors = list(ledger["errors"])
    matching = [event for event in read_forward_events(ledger_path)
                if event.get("event_type") == "MODEL_FREEZE"
                and event.get("freeze_id") == manifest.get("freeze_id")]
    if not matching:
        errors.append("active freeze has no matching MODEL_FREEZE event")
    elif not active_path.exists() or sha256_file(active_path) != matching[-1].get("manifest_sha256"):
        errors.append("active freeze content does not match its sealed manifest")
    return {**ledger, "status": "PASS" if not errors else "FAIL", "errors": errors,
            "active_freeze_bound": not errors}


def rebuild_forward_views(ledger_path: Path = FORWARD_LEDGER_PATH, root: Path = ROOT,
                          signals_path: Path = FORWARD_SIGNALS_PATH,
                          outcomes_path: Path = PAPER_TRADES_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    for event in read_forward_events(ledger_path):
        if event.get("event_type") == "SIGNAL_SNAPSHOT":
            snapshot_path = root / str(event["snapshot_file"])
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            for candidate in snapshot.get("candidates", []):
                signal_rows.append({
                    "freeze_id": event["freeze_id"], "signal_date": event["signal_date"],
                    "asof": event["asof"], "snapshot_sha256": event["snapshot_sha256"],
                    "signal_event_hash": event["event_hash"], **candidate,
                })
        elif event.get("event_type") == "PAPER_OUTCOME":
            outcome_rows.append({key: value for key, value in event.items()
                                 if key not in {"previous_event_hash"}})
    signals = (pd.DataFrame(signal_rows) if signal_rows else pd.DataFrame(columns=[
        "freeze_id", "signal_date", "asof", "snapshot_sha256", "signal_event_hash",
        "code", "name", "decision", "promotion_probability", "model_status",
    ]))
    outcomes = (pd.DataFrame(outcome_rows) if outcome_rows else pd.DataFrame(columns=[
        "event_type", "freeze_id", "signal_event_hash", "signal_date", "code", "name",
        "decision", "asof", "execution_status", "entry_date", "entry_price",
        "promoted_next_day", "t1_close", "round_trip_cost", "mark_to_market_return",
        "net_return", "large_loss", "sell_blocked", "sequence", "recorded_at", "event_hash",
    ]))
    signals_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(signals_path, index=False, encoding="utf-8-sig")
    outcomes.to_csv(outcomes_path, index=False, encoding="utf-8-sig")
    return signals, outcomes


def register_forward_snapshot(signal_date: str, asof: str, candidates: pd.DataFrame,
                              score_json_path: Path, score_csv_path: Path,
                              active_path: Path = ACTIVE_FREEZE_PATH,
                              ledger_path: Path = FORWARD_LEDGER_PATH,
                              forward_dir: Path = FORWARD, root: Path = ROOT,
                              model_path: Path = MODEL_PATH,
                              code_path: Path | None = None,
                              latest_candidate_date: str | None = None,
                              signals_path: Path = FORWARD_SIGNALS_PATH,
                              outcomes_path: Path = PAPER_TRADES_PATH) -> dict[str, Any]:
    manifest = active_freeze(active_path)
    binding = verify_active_freeze_binding(manifest, active_path, ledger_path, root)
    if binding["status"] != "PASS":
        raise ValueError("active freeze failed ledger binding verification")
    validate_active_freeze(manifest, model_path, code_path)
    normalized_date = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
    data_end = pd.Timestamp(manifest.get("candidate_data_end_at_freeze", manifest["data_end"])).strftime("%Y-%m-%d")
    if normalized_date <= data_end:
        raise ValueError(f"forward signal date must be later than frozen candidate data end {data_end}")
    freeze_day = pd.Timestamp(manifest.get("created_at", data_end)).strftime("%Y-%m-%d")
    if normalized_date < freeze_day:
        raise ValueError(f"forward signal date cannot predate freeze day {freeze_day}")
    if latest_candidate_date is None and (DATA / "leader_candidates.csv").exists():
        available = pd.read_csv(DATA / "leader_candidates.csv", usecols=["date"])
        latest_candidate_date = str(available["date"].max())
    if latest_candidate_date is not None and normalized_date != pd.Timestamp(latest_candidate_date).strftime("%Y-%m-%d"):
        raise ValueError("only the latest available candidate date can be preregistered; historical backfill is forbidden")
    verification = verify_forward_ledger(ledger_path, root)
    if verification["status"] != "PASS":
        raise ValueError("forward ledger failed verification")
    for prior in read_forward_events(ledger_path):
        if (prior.get("event_type") == "SIGNAL_SNAPSHOT"
                and prior.get("freeze_id") == manifest["freeze_id"]
                and prior.get("signal_date") == normalized_date
                and prior.get("asof") == asof):
            raise ValueError("this freeze/date/asof snapshot is already preregistered")
    required = {"code", "decision", "promotion_probability", "model_status"}
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"forward candidates missing fields: {missing}")
    snapshot_dir = forward_dir / "snapshots" / manifest["freeze_id"]
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"{normalized_date}_{asof.replace(':', '')}.json"
    snapshot = {
        "freeze_id": manifest["freeze_id"], "signal_date": normalized_date, "asof": asof,
        "model_sha256": manifest["model_sha256"], "code_sha256": manifest["code_sha256"],
        "score_json_sha256": sha256_file(score_json_path),
        "score_csv_sha256": sha256_file(score_csv_path),
        "candidates": candidates.to_dict("records"),
    }
    snapshot_path.write_text(json_text(snapshot), encoding="utf-8")
    try:
        snapshot_reference = str(snapshot_path.relative_to(root))
    except ValueError:
        snapshot_reference = str(snapshot_path)
    event = append_forward_event({
        "event_type": "SIGNAL_SNAPSHOT", "freeze_id": manifest["freeze_id"],
        "signal_date": normalized_date, "asof": asof,
        "snapshot_file": snapshot_reference, "snapshot_sha256": sha256_file(snapshot_path),
        "candidate_count": int(len(candidates)),
        "enter_count": int((candidates["decision"] == "ENTER").sum()),
        "watch_count": int((candidates["decision"] == "WATCH").sum()),
        "model_deployment_status": manifest.get("model_deployment_status"),
    }, ledger_path, root)
    rebuild_forward_views(ledger_path, root, signals_path, outcomes_path)
    return {"freeze_id": manifest["freeze_id"], "signal_date": normalized_date,
            "asof": asof, "snapshot_sha256": event["snapshot_sha256"],
            "event_hash": event["event_hash"], "candidate_count": len(candidates),
            "enter_count": event["enter_count"], "watch_count": event["watch_count"]}


def forward_monitor(manifest: dict[str, Any], signals: pd.DataFrame, outcomes: pd.DataFrame,
                    ledger_verification: dict[str, Any]) -> dict[str, Any]:
    frozen_signals = signals[signals["freeze_id"].astype(str) == manifest["freeze_id"]].copy()
    registered_enter = frozen_signals[frozen_signals["decision"].astype(str) == "ENTER"]
    frozen_outcomes = outcomes[outcomes["freeze_id"].astype(str) == manifest["freeze_id"]].copy()
    executed = frozen_outcomes[
        (frozen_outcomes["decision"].astype(str) == "ENTER")
        & frozen_outcomes["execution_status"].astype(str).isin(["verified_fill", "sell_blocked"])
    ].copy()
    promoted = pd.to_numeric(executed.get("promoted_next_day", pd.Series(dtype=float)), errors="coerce")
    losses = pd.to_numeric(executed.get("large_loss", pd.Series(dtype=float)), errors="coerce")
    resolved = int(promoted.notna().sum())
    precision = safe_float(promoted.mean()) if resolved else math.nan
    loss_rate = safe_float(losses.mean()) if losses.notna().any() else math.nan
    passed = bool(
        ledger_verification.get("status") == "PASS"
        and manifest.get("model_deployment_status") == "HISTORICAL_GATE_PASSED"
        and resolved >= 60 and precision >= 0.70 and loss_rate <= 0.10
    )
    return {
        "freeze_id": manifest["freeze_id"],
        "model_deployment_status_at_freeze": manifest.get("model_deployment_status"),
        "registered_enter_signals": int(len(registered_enter)),
        "resolved_executable_enter_signals": resolved,
        "promotion_precision": None if math.isnan(precision) else precision,
        "large_loss_rate": None if math.isnan(loss_rate) else loss_rate,
        "condition_not_triggered_or_unbuyable": int(len(frozen_outcomes) - len(executed)),
        "report_checkpoint": bool(resolved > 0 and resolved % 20 == 0),
        "ledger_verification": ledger_verification,
        "forward_gate_passed": passed,
        "status": "PAPER_TRADE_READY" if passed else manifest.get("model_deployment_status", "DIAGNOSTIC_ONLY"),
        "rule": "pre-registered ENTER only; require historical gate, >=60 executed forward signals, >=70% promotion precision and <=10% large-loss rate",
    }


def paper_log_command(args: argparse.Namespace) -> None:
    ensure_dirs()
    manifest = active_freeze()
    verification = verify_active_freeze_binding(manifest)
    if verification["status"] != "PASS":
        raise SystemExit("forward ledger failed verification; refusing an outcome")
    signals, outcomes = rebuild_forward_views()
    if signals.empty:
        raise SystemExit("no preregistered forward signals")
    signal_date = pd.Timestamp(args.signal_date).strftime("%Y-%m-%d")
    code = zfill_code(args.code)
    matches = signals[
        (signals.get("freeze_id", "") == manifest["freeze_id"])
        & (signals.get("signal_date", "") == signal_date)
        & (signals.get("code", "").astype(str).str.zfill(6) == code)
    ]
    if matches.empty:
        raise SystemExit("no preregistered signal for this active freeze/date/code")
    signal = matches.sort_values("asof").iloc[-1]
    decision = str(signal["decision"])
    if getattr(args, "decision", None) and args.decision != decision:
        raise SystemExit(f"decision mismatch: preregistered={decision}, supplied={args.decision}")
    for prior in read_forward_events():
        if (prior.get("event_type") == "PAPER_OUTCOME" and prior.get("freeze_id") == manifest["freeze_id"]
                and prior.get("signal_date") == signal_date and prior.get("code") == code):
            raise SystemExit("outcome already exists; forward outcomes are append-only")
    execution_status = args.execution_status
    entry = safe_float(args.entry_price)
    close = safe_float(args.t1_close)
    if execution_status in {"verified_fill", "sell_blocked"} and (entry <= 0 or close <= 0):
        raise SystemExit("filled/sell-blocked outcomes require positive entry-price and t1-close")
    filled = execution_status in {"verified_fill", "sell_blocked"}
    fee_rate = safe_float(args.fees)
    mark = close / entry - 1 - fee_rate if filled else math.nan
    net = mark if filled and execution_status != "sell_blocked" else math.nan
    event = append_forward_event({
        "event_type": "PAPER_OUTCOME", "freeze_id": manifest["freeze_id"],
        "signal_event_hash": signal["signal_event_hash"],
        "signal_date": signal_date, "code": code, "name": signal.get("name", ""),
        "decision": decision, "asof": signal["asof"],
        "execution_status": execution_status, "entry_date": args.entry_date,
        "entry_price": entry if filled else None, "promoted_next_day": int(args.promoted),
        "t1_close": close if filled else None, "round_trip_cost": fee_rate if filled else None,
        "mark_to_market_return": mark if filled else None,
        "net_return": net if not math.isnan(net) else None,
        "large_loss": int(mark <= -0.07) if filled else None,
        "sell_blocked": int(execution_status == "sell_blocked"),
    })
    signals, outcomes = rebuild_forward_views()
    monitor = forward_monitor(manifest, signals, outcomes, verify_forward_ledger())
    (RESULTS / "paper_monitor.json").write_text(json_text(monitor), encoding="utf-8")
    print(json_text({"outcome_event": event, "monitor": monitor}))


def forward_verify_command(args: argparse.Namespace) -> None:
    ensure_dirs()
    manifest = active_freeze() if ACTIVE_FREEZE_PATH.exists() else None
    verification = (verify_active_freeze_binding(manifest) if manifest is not None
                    else verify_forward_ledger())
    signals, outcomes = rebuild_forward_views()
    report: dict[str, Any] = {"ledger": verification, "active_freeze": manifest}
    if manifest is not None:
        report["monitor"] = forward_monitor(manifest, signals, outcomes, verification)
    (RESULTS / "forward_verification.json").write_text(json_text(report), encoding="utf-8")
    print(json_text(report))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A-share continuous-board leader research system V3")
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--date", help="trade date, YYYYMMDD or YYYY-MM-DD")
    collect.add_argument("--signal-date", help="previous close candidate date for an open snapshot")
    collect.add_argument("--asof", choices=SNAPSHOT_TIMES)
    collect.add_argument("--history-start", help="free unadjusted daily history start")
    collect.add_argument("--history-end", help="free unadjusted daily history end")
    collect.add_argument("--codes-file", help="optional historical universe CSV with code/name/universe_complete")
    collect.add_argument("--max-codes", type=int, help="incremental/debug limit")
    collect.add_argument("--workers", type=int, default=5)
    collect.add_argument("--refresh-history", action="store_true", help="refetch codes already marked successful for the same date range")
    imported = sub.add_parser("import")
    imported.add_argument("file")
    imported.add_argument("--kind", required=True, choices=sorted(IMPORT_REQUIRED))
    sub.add_parser("build")
    train = sub.add_parser("train")
    train.add_argument("--force", action="store_true", help="override quarterly lock for code/test corrections")
    sub.add_parser("backtest")
    sub.add_parser("audit")
    score = sub.add_parser("score")
    score.add_argument("--date", help="signal date; defaults to latest")
    score.add_argument("--asof", default="09:35", choices=SNAPSHOT_TIMES)
    score.add_argument("--record-forward", action="store_true",
                       help="preregister the latest post-freeze score snapshot in the tamper-evident ledger")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--new-freeze", action="store_true",
                        help="start a predeclared new forward cycle after an intentional model/code change")
    sub.add_parser("forward-verify")
    paper = sub.add_parser("paper-log")
    paper.add_argument("--signal-date", required=True)
    paper.add_argument("--code", required=True)
    paper.add_argument("--decision", choices=["ENTER", "WATCH", "EMOTION_ANCHOR", "REJECT"],
                       help="optional cross-check; the preregistered decision remains authoritative")
    paper.add_argument("--execution-status", required=True,
                       choices=["verified_fill", "no_trade", "unbuyable_one_word_up", "sell_blocked"])
    paper.add_argument("--entry-date")
    paper.add_argument("--entry-price")
    paper.add_argument("--promoted", required=True, choices=[0, 1], type=int)
    paper.add_argument("--t1-close")
    paper.add_argument("--fees", default="0.003", help="round-trip fee/slippage rate")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    commands = {
        "collect": collect_command, "import": import_command, "build": build_command,
        "train": train_command, "backtest": backtest_command, "audit": data_audit_command, "score": score_command,
        "freeze": freeze_model_command, "forward-verify": forward_verify_command,
        "paper-log": paper_log_command,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
