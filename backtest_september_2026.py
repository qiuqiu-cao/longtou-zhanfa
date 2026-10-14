"""2026-09 main-board consecutive-board leader backtest.

The signal is fixed before reading the post-signal outcome:
  * universe: Shanghai/Shenzhen main board (60/00), non-ST;
  * one candidate per day: the highest closing consecutive board, ranked by
    same-theme breadth, turnover proximity, and earlier sealing time;
  * only candidates with at least 3 boards, at least 30 closing limit-ups, and
    non-decreasing market height are traded in the practical sleeve;
  * entry: next session, only if the stock touches its 10% limit and is not
    an all-day locked limit-up; entry price is the limit price;
  * exit: the following session close. An all-day limit-down exit is marked
    unresolved and is not counted as a realized trade.

The limit-up pool data are from Eastmoney's public daily pool endpoint. OHLCV
is from Tencent's public historical endpoint. This is a daily-OHLC execution proxy,
not a proof of queue position or actual fills.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from datetime import datetime
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "2026-09_lianban"
RESULTS = ROOT / "results"

BUY_COST = 0.00025 + 0.0010       # 0.025% commission + 0.10% entry slippage
SELL_COST = 0.00025 + 0.0005 + 0.0010  # commission + 0.05% stamp duty + slippage


def weekdays(start="2026-09-01", end="2026-09-30"):
    return pd.date_range(start, end, freq="B")


def fetch_page(day: pd.Timestamp, refresh=False) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    ds = day.strftime("%Y%m%d")
    raw_path = RAW / f"page_{ds}.html"
    if raw_path.exists() and not refresh:
        text = raw_path.read_text(encoding="utf-8")
    else:
        url = f"https://lianban.net/days/{day:%Y-%m-%d}.html"
        text = None
        # The site rate-limits repetitive requests. A cache-busting query and
        # ordinary curl/browser user agents consistently return the static page.
        uas = ["curl/8.0", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
               "AppleWebKit/537.36 Chrome/131 Safari/537.36"]
        for attempt in range(8):
            try:
                h = {"User-Agent": uas[attempt % len(uas)],
                     "Accept": "text/html,application/xhtml+xml"}
                r = requests.get(url + f"?cb={random.randint(1, 10**9)}",
                                 headers=h, timeout=25)
                if r.status_code == 404:
                    return {}
                if r.status_code == 200:
                    m = re.search(r"var S=(\{.*?\});var CUR", r.text, re.S)
                    if m:
                        text = r.text
                        raw_path.write_text(text, encoding="utf-8")
                        break
            except requests.RequestException:
                pass
            time.sleep(2.0 + attempt * 0.5)
        if text is None:
            return {}
        time.sleep(0.25)
    m = re.search(r"var S=(\{.*?\});var CUR", text, re.S)
    if not m:
        return {}
    try:
        stocks = json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}
    cur = re.search(r'var CUR="([^"]+)"', text)
    return {"date": cur.group(1) if cur else day.strftime("%Y-%m-%d"),
            "stocks": stocks,
            "source_url": f"https://lianban.net/days/{day:%Y-%m-%d}.html"}


def mainboard(code: str, row: dict) -> bool:
    return (re.match(r"^(60|00)\d{4}$", code) is not None
            and not bool(row.get("st", False))
            and not str(row.get("n", "")).startswith(("ST", "*ST", "退")))


def num(v, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def collect_pool(refresh=False):
    """Fetch the full daily closing-limit pool from Eastmoney's public API.

    This endpoint is the same provider used by the repository's archived
    AKShare snapshots, but calling it directly lets us fill the missing
    9/22--9/24 dates without relying on a rate-limited review website.
    """
    rows = []
    pages = []
    # Two early-September Eastmoney snapshots are unavailable from the public
    # endpoint and no immutable local pool file exists for them.  The archived
    # lianban.net daily review confirms the market totals and the highest-board
    # leader; retain those observations as a clearly marked partial rebuild so
    # the market-height gate does not incorrectly skip the 9/4 signal.
    manual_gap = {
        "20260902": {
            "total": 52, "source_url": "https://lianban.net/days/2026-09-02.html",
            "rows": [{"code": "601086", "name": "国芳集团", "boards": 4,
                      "theme": "一般零售", "turnover": float("nan"),
                      "seal_time": "092500", "one_word": True,
                      "failed_board": False}],
        },
        "20260903": {
            "total": 44, "source_url": "https://lianban.net/days/2026-09-03.html",
            "rows": [{"code": "601086", "name": "国芳集团", "boards": 5,
                      "theme": "一般零售", "turnover": float("nan"),
                      "seal_time": "093000", "one_word": False,
                      "failed_board": False}],
        },
    }
    pool_dir = ROOT / "data" / "2026-09_eastmoney"
    pool_dir.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0"})
    for day in weekdays():
        ds = day.strftime("%Y%m%d")
        raw_path = pool_dir / f"zt_pool_{ds}.json"
        if raw_path.exists() and not refresh:
            try:
                payload = json.loads(raw_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = None
        else:
            params = {"ut": "7eea3edcaed734bea9cbfc24409ed989",
                      "dpt": "wz.ztzt", "Pageindex": "0",
                      "pagesize": "10000", "sort": "fbt:asc", "date": ds}
            try:
                response = sess.get("https://push2ex.eastmoney.com/getTopicZTPool",
                                    params=params, timeout=(5, 15))
                response.raise_for_status()
                payload = response.json()
                raw_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                time.sleep(0.12)
            except (requests.RequestException, ValueError):
                payload = None
        data = (payload or {}).get("data") or {}
        pool_rows = data.get("pool") or []
        if not pool_rows:
            if ds in manual_gap:
                gap = manual_gap[ds]
                pages.append({"date": ds, "total": gap["total"],
                              "source_url": gap["source_url"],
                              "quality": "partial_leader_reconstruction"})
                for x in gap["rows"]:
                    rows.append({"date": pd.Timestamp(ds), **x,
                                 "amp": None,
                                 "market_limit_count": gap["total"],
                                 "source_url": gap["source_url"],
                                 "source_quality": "partial_leader_reconstruction"})
                continue
            # Eastmoney's public endpoint retains only a rolling historical
            # window.  The repository already contains immutable snapshots for
            # 9/1--9/8; use those exact rows when the endpoint reports tc=0.
            archive = ROOT / "data" / "v3" / "raw" / ds / "limit_up.csv"
            if not archive.exists():
                continue
            try:
                old = pd.read_csv(archive, dtype={"代码": str})
            except Exception:
                continue
            total = len(old)
            pages.append({"date": ds, "total": total,
                          "source_url": str(archive.resolve())})
            for _, x in old.iterrows():
                code = str(x.get("代码", "")).zfill(6)
                name = str(x.get("名称", "")).strip()
                if not re.match(r"^(60|00)\d{4}$", code) or name.startswith(("ST", "*ST", "退")):
                    continue
                boards = int(num(x.get("连板数"), 1) or 1)
                fbt = int(num(x.get("首次封板时间"), 0) or 0)
                lbt = int(num(x.get("最后封板时间"), 0) or 0)
                zbc = int(num(x.get("炸板次数"), 0) or 0)
                rows.append({"date": pd.Timestamp(ds), "code": code, "name": name,
                             "boards": boards,
                             "theme": str(x.get("所属行业") or "未知"),
                             "turnover": num(x.get("换手率")), "amp": None,
                             "seal_time": str(fbt),
                             "one_word": fbt == lbt == 92500 and zbc == 0,
                             "failed_board": zbc > 0,
                             "market_limit_count": total,
                             "source_url": str(archive.resolve())})
            continue
        total = int(data.get("tc") or len(pool_rows))
        pages.append({"date": ds, "total": total,
                      "source_url": "https://push2ex.eastmoney.com/getTopicZTPool"})
        for x in pool_rows:
            code = str(x.get("c", "")).zfill(6)
            name = str(x.get("n", "")).strip()
            if not re.match(r"^(60|00)\d{4}$", code) or name.startswith(("ST", "*ST", "退")):
                continue
            boards = int(x.get("lbc") or (x.get("zttj") or {}).get("days") or 1)
            rows.append({"date": pd.Timestamp(ds), "code": code, "name": name,
                         "boards": boards, "theme": str(x.get("hybk") or "未知"),
                         "turnover": num(x.get("hs")), "amp": None,
                         "seal_time": str(x.get("fbt") or ""),
                         "one_word": int(x.get("fbt") or 0) == int(x.get("lbt") or 0)
                         and int(x.get("zbc") or 0) == 0,
                         "failed_board": int(x.get("zbc") or 0) > 0,
                         "market_limit_count": total,
                         "source_url": "https://push2ex.eastmoney.com/getTopicZTPool"})
    pool = pd.DataFrame(rows)
    if pool.empty:
        raise RuntimeError("No Eastmoney limit-up pool data were collected")
    return pool, sorted(pd.Timestamp(x) for x in pool["date"].unique()), pages


def select_daily(pool: pd.DataFrame) -> pd.DataFrame:
    out = []
    for dt, day in pool.groupby("date", sort=True):
        height = int(day["boards"].max())
        leaders = day[day["boards"] == height].copy()
        theme_counts = day.groupby("theme")["code"].nunique().to_dict()
        leaders["theme_support"] = leaders["theme"].map(theme_counts).fillna(1)
        leaders["turnover_score"] = -abs(leaders["turnover"].fillna(0) - 15.0)
        leaders["seal_score"] = leaders["seal_time"].map(
            lambda x: -int(str(x).replace(":", "").replace(".", "") or "999999"))
        # Higher theme support, non-one-word, non-failed, reasonable turnover,
        # and earlier sealing decide ties among the same highest board.
        leaders = leaders.sort_values(
            ["theme_support", "one_word", "failed_board", "turnover_score", "seal_score"],
            ascending=[False, True, True, False, False], kind="mergesort")
        chosen = leaders.iloc[0].to_dict()
        chosen["market_height"] = height
        chosen["height_non_decreasing"] = True
        out.append(chosen)
    sig = pd.DataFrame(out).sort_values("date").reset_index(drop=True)
    sig["height_non_decreasing"] = sig["market_height"].cummax().eq(sig["market_height"])
    # cummax equality is too strict after a height falls; use the intended
    # comparison to the prior height instead.
    prev = sig["market_height"].shift(1)
    sig["height_non_decreasing"] = prev.isna() | sig["market_height"].ge(prev)
    return sig


def fetch_history(code: str, refresh=False) -> pd.DataFrame:
    hist_dir = RAW / "hist"
    hist_dir.mkdir(parents=True, exist_ok=True)
    path = hist_dir / f"{code}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    market = ("sh" if code.startswith("6") else "sz") + code
    url = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param="
           f"{market},day,2026-08-20,2026-10-20,1000,qfq")
    try:
        r = requests.get(url, timeout=20)
        data = r.json().get("data", {}).get(market, {}).get("qfqday", [])
    except (requests.RequestException, ValueError):
        data = []
    frame = pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume"])
    if frame.empty:
        return frame.set_index(pd.DatetimeIndex([], name="date"))
    frame["date"] = pd.to_datetime(frame["date"])
    for col in ["open", "close", "high", "low", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame.to_csv(path, index=False)
    time.sleep(0.08)
    return frame.set_index("date").sort_index()


def limit_price(prev_close: float, pct=0.10) -> float:
    # A-share main-board price limit rounded to cents (half-up).
    d = (Decimal(str(prev_close)) * (Decimal("1") + Decimal(str(pct))))
    return float(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def down_limit(prev_close: float, pct=0.10) -> float:
    d = (Decimal(str(prev_close)) * (Decimal("1") - Decimal(str(pct))))
    return float(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def next_dates(dates, dt):
    later = [d for d in dates if d > dt]
    return (later[0] if len(later) > 0 else None,
            later[1] if len(later) > 1 else None)


def run_trades(sig, histories, pool_dates, mode="practical"):
    trades = []
    last_exit = None
    for _, s in sig.iterrows():
        signal_date = pd.Timestamp(s["date"])
        entry_date, exit_date = next_dates(pool_dates, signal_date)
        if entry_date is None or exit_date is None:
            continue
        if last_exit is not None and entry_date <= last_exit:
            continue
        eligible = (int(s["boards"]) >= 3 and int(s["market_limit_count"]) >= 30
                    and bool(s["height_non_decreasing"]))
        if mode == "practical" and not eligible:
            continue
        code = str(s["code"]).zfill(6)
        h = histories.get(code)
        if h is None or any(d not in h.index for d in (signal_date, entry_date, exit_date)):
            continue
        sig_close = float(h.loc[signal_date, "close"])
        prev = float(h.loc[entry_date, "close"] if entry_date == signal_date else h.loc[entry_date, "open"])
        # For a next-session limit calculation, previous close is the signal
        # day's close. Daily OHLC is enough to test the touch and locked state.
        prev_close = sig_close
        up = limit_price(prev_close)
        e = h.loc[entry_date]
        open_gap = float(e["open"]) / prev_close - 1.0
        # The project's relay rule treats 0--5% as preferred and 5--7% as
        # conditional; above 7% is an execution-risk veto. The daily proxy
        # cannot verify VWAP/real-time absorption, so the practical sleeve
        # applies the conservative 0--7% gate.
        if mode == "practical" and not (0.0 <= open_gap <= 0.07):
            continue
        touched = float(e["high"]) >= up - 0.011
        locked_up = (abs(float(e["open"]) - float(e["high"])) < 1e-8
                     and abs(float(e["low"]) - float(e["high"])) < 1e-8
                     and abs(float(e["close"]) - float(e["high"])) < 1e-8
                     and float(e["close"]) >= up - 0.011)
        if not touched or locked_up:
            continue
        x = h.loc[exit_date]
        exit_prev = float(e["close"])
        down = down_limit(exit_prev)
        locked_down = (abs(float(x["open"]) - float(x["high"])) < 1e-8
                       and abs(float(x["low"]) - float(x["high"])) < 1e-8
                       and abs(float(x["close"]) - float(x["low"])) < 1e-8
                       and float(x["close"]) <= down + 0.011)
        mark = float(x["close"])
        if locked_down:
            trades.append({"signal_date": signal_date.date().isoformat(),
                           "entry_date": entry_date.date().isoformat(),
                           "exit_date": exit_date.date().isoformat(), "code": code,
                           "name": s["name"], "boards": int(s["boards"]),
                           "market_limit_count": int(s["market_limit_count"]),
                           "entry": up, "exit": mark, "open_gap": open_gap,
                           "status": "UNRESOLVED_LIMIT_DOWN",
                           "net_return": None, "gross_return": mark / up - 1.0})
            last_exit = exit_date
            continue
        gross = mark / up - 1.0
        net = mark * (1 - SELL_COST) / (up * (1 + BUY_COST)) - 1.0
        trades.append({"signal_date": signal_date.date().isoformat(),
                       "entry_date": entry_date.date().isoformat(),
                       "exit_date": exit_date.date().isoformat(), "code": code,
                       "name": s["name"], "boards": int(s["boards"]),
                       "market_limit_count": int(s["market_limit_count"]),
                       "entry": up, "exit": mark, "open_gap": open_gap,
                       "status": "REALIZED_PROXY",
                       "net_return": net, "gross_return": gross})
        last_exit = exit_date
    return pd.DataFrame(trades)


def metrics(trades: pd.DataFrame):
    if trades.empty:
        return {"trades": 0, "realized_trades": 0}
    realized = trades[trades["status"] == "REALIZED_PROXY"].copy()
    if realized.empty:
        return {"trades": int(len(trades)), "realized_trades": 0,
                "unresolved": int(len(trades))}
    r = pd.to_numeric(realized["net_return"], errors="coerce").dropna()
    curve = (1 + r).cumprod()
    dd = curve / curve.cummax() - 1
    wins, losses = r[r > 0], r[r <= 0]
    return {"trades": int(len(trades)), "realized_trades": int(len(r)),
            "unresolved": int(len(trades) - len(r)),
            "win_rate": float((r > 0).mean()), "total_return": float(curve.iloc[-1] - 1),
            "avg_trade": float(r.mean()), "median_trade": float(r.median()),
            "max_drawdown": float(dd.min()),
            "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() else math.inf,
            "best_trade": float(r.max()), "worst_trade": float(r.min())}


def run(refresh=False):
    pool, dates, pages = collect_pool(refresh=refresh)
    sig = select_daily(pool)
    codes = sig["code"].astype(str).str.zfill(6).unique()
    histories = {code: fetch_history(code, refresh=refresh) for code in codes}
    # Extend the execution calendar through the first post-month sessions so
    # that a 9/30 signal is not silently discarded.
    calendar = set(dates)
    for h in histories.values():
        calendar.update(pd.Timestamp(x) for x in h.index if pd.Timestamp(x) >= pd.Timestamp("2026-09-01"))
    calendar = sorted(calendar)
    censored = []
    for dt in sig["date"]:
        e1, e2 = next_dates(calendar, pd.Timestamp(dt))
        if e1 is None or e2 is None:
            censored.append(pd.Timestamp(dt).date().isoformat())
    benchmark = run_trades(sig, histories, calendar, mode="benchmark")
    practical = run_trades(sig, histories, calendar, mode="practical")
    RESULTS.mkdir(exist_ok=True)
    sig.to_csv(RESULTS / "september_2026_daily_leaders.csv", index=False, encoding="utf-8-sig")
    benchmark.to_csv(RESULTS / "september_2026_benchmark_trades.csv", index=False, encoding="utf-8-sig")
    practical.to_csv(RESULTS / "september_2026_practical_trades.csv", index=False, encoding="utf-8-sig")
    report = {
        "retrieved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_range": [str(min(dates).date()), str(max(dates).date())],
        "page_count": len(pages), "pool_rows_mainboard": int(len(pool)),
        "daily_signals": int(len(sig)),
        "censored_signal_dates": censored,
        "partial_reconstruction_dates": ["2026-09-02", "2026-09-03"],
        "costs": {"buy": BUY_COST, "sell": SELL_COST},
        "execution_proxy": "practical sleeve: next-session open gap 0-7%, then touch of 10% limit excluding all-day locked up; following-session close exit",
        "source": {"limit_pool": "Eastmoney getTopicZTPool; archived Eastmoney/AKShare snapshots for 9/1-9/8",
                    "ohlcv": "Tencent fqkline qfqday"},
        "fields_verified": ["daily limit-up pool", "code/name", "consecutive-board count",
                            "daily open/high/low/close/volume"],
        "fields_unavailable": ["Level-2 queue", "tick/order sequence", "auction cancellation",
                               "actual fill and queue position", "intraday VWAP"],
        "execution_limit": "INFERRED_DAILY_OHLC; touch/reseal and exits are not broker-confirmed",
        "benchmark": metrics(benchmark), "practical": metrics(practical),
    }
    (RESULTS / "september_2026_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))
    print("\nPRACTICAL TRADES")
    print(practical.to_string(index=False) if not practical.empty else "none")
    print("\nBENCHMARK TRADES")
    print(benchmark.to_string(index=False) if not benchmark.empty else "none")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    run(ap.parse_args().refresh)
