"""2026-08 A-share main-board continuous-board leader backtest.

Data: Eastmoney public daily limit-up pool and adjusted-free daily OHLC via AkShare.
The practical rule is fixed before looking at returns; this is a one-month diagnostic,
not evidence of a durable edge.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
import time
import requests
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "2026-08"
RESULTS = ROOT / "results"
BUY_COST = 0.00025 + 0.0010       # commission + assumed entry slippage
SELL_COST = 0.00025 + 0.0005 + 0.0010  # commission + stamp duty + slippage


def weekdays():
    return pd.date_range("2026-08-01", "2026-08-28", freq="B")


def fetch_pools(refresh=False):
    RAW.mkdir(parents=True, exist_ok=True)
    frames = []
    code_map = None
    for ts in weekdays():
        ds = ts.strftime("%Y%m%d")
        path = RAW / f"zt_pool_{ds}.csv"
        if path.exists() and not refresh:
            df = pd.read_csv(path, dtype={"代码": str})
        else:
            try:
                df = ak.stock_zt_pool_em(date=ds)
            except Exception:
                df = pd.DataFrame()
            if df.empty:
                if code_map is None:
                    names = ak.stock_info_a_code_name()
                    code_map = dict(zip(names["name"], names["code"].astype(str).str.zfill(6)))
                df = fetch_lianban_fallback(ts, code_map)
            if df.empty:
                continue
            df.to_csv(path, index=False, encoding="utf-8-sig")
            time.sleep(0.15)
        if "当日涨停总数" not in df:
            df["当日涨停总数"] = len(df)
        if "data_source" not in df:
            df["data_source"] = "Eastmoney/AkShare"
        df["代码"] = df["代码"].astype(str).str.zfill(6)
        df["date"] = pd.Timestamp(ts)
        frames.append(df)
    if not frames:
        raise RuntimeError("No limit-up pool data fetched")
    return pd.concat(frames, ignore_index=True)


def fetch_lianban_fallback(ts, code_map):
    """Recover the public board ladder from lianban.net JSON-LD.

    This fallback has board/name/market total but not tick-level seal or industry data.
    """
    url = f"https://lianban.net/days/{ts.strftime('%Y-%m-%d')}.html"
    text = requests.get(url, timeout=20).text
    blocks = re.findall(r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>', text, re.S | re.I)
    parsed = []
    total = None
    for block in blocks:
        try:
            obj = json.loads(html.unescape(block))
        except json.JSONDecodeError:
            continue
        if obj.get("@type") == "Dataset":
            m = re.search(r"涨停\s*(\d+)\s*家", obj.get("description", ""))
            total = int(m.group(1)) if m else total
        if obj.get("@type") == "ItemList" and "连板天梯" in obj.get("name", ""):
            for item in obj.get("itemListElement", []):
                m = re.match(r"(.+?)\s+(\d+)板$", item.get("name", ""))
                if not m:
                    continue
                name, boards = m.group(1), int(m.group(2))
                code = code_map.get(name)
                if code and boards >= 2:
                    parsed.append({
                        "代码": code, "名称": name, "换手率": 15.0, "炸板次数": 0,
                        "首次封板时间": 999999, "最后封板时间": 999999,
                        "连板数": boards, "所属行业": "未核验", "涨停统计": f"{boards}/{boards}",
                        "当日涨停总数": total or len(parsed), "data_source": "lianban.net JSON-LD",
                    })
    print(f"fallback {ts.date()}: {len(parsed)} continuous boards")
    return pd.DataFrame(parsed)


def mainboard(df):
    code_ok = df["代码"].str.match(r"^(60|00)")
    name_ok = ~df["名称"].astype(str).str.contains(r"ST|退", case=False, regex=True)
    return df[code_ok & name_ok].copy()


def fetch_histories(codes, refresh=False):
    out = {}
    hist_dir = RAW / "hist"
    hist_dir.mkdir(parents=True, exist_ok=True)
    for code in sorted(codes):
        path = hist_dir / f"{code}.csv"
        if path.exists() and not refresh:
            h = pd.read_csv(path, dtype={"股票代码": str})
        else:
            try:
                h = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20260720",
                                       end_date="20260828", adjust="")
            except Exception as exc:
                market_code = ("sh" if code.startswith("6") else "sz") + code
                try:
                    h = ak.stock_zh_a_daily(symbol=market_code, start_date="20260720",
                                            end_date="20260828", adjust="")
                    h = h.rename(columns={"date": "日期", "open": "开盘", "high": "最高",
                                          "low": "最低", "close": "收盘", "volume": "成交量",
                                          "amount": "成交额", "turnover": "换手率"})
                except Exception:
                    print(f"history failed {code}: {exc}")
                    continue
            if h.empty:
                continue
            h.to_csv(path, index=False, encoding="utf-8-sig")
            time.sleep(0.12)
        h["日期"] = pd.to_datetime(h["日期"])
        out[code] = h.set_index("日期").sort_index()
    return out


def time_num(value):
    try:
        return int(str(value).replace(".0", "").zfill(6))
    except ValueError:
        return 999999


def quality_rank(row):
    turnover = float(row["换手率"])
    turnover_quality = -abs(turnover - 15.0)
    return (int(row["sector_support"]), int(row["炸板次数"] > 0),
            int(not row["one_word_proxy"]), turnover_quality,
            -time_num(row["最后封板时间"]))


def signals(pool):
    records = []
    dates = sorted(pool["date"].unique())
    prior_height = None
    for dt in dates:
        day_all = pool[pool.date == dt].copy()
        day = day_all[day_all["连板数"] >= 2].copy()
        if day.empty:
            prior_height = 1
            continue
        height = int(day["连板数"].max())
        industry_counts = day_all.groupby("所属行业").size().to_dict()
        leaders = day[day["连板数"] == height].copy()
        leaders["sector_support"] = leaders["所属行业"].map(industry_counts).fillna(1)
        leaders["one_word_proxy"] = (
            leaders["首次封板时间"].map(time_num).eq(92500)
            & leaders["最后封板时间"].map(time_num).eq(92500)
            & leaders["炸板次数"].eq(0)
            & leaders["换手率"].lt(3.0)
        )
        chosen = max((r for _, r in leaders.iterrows()), key=quality_rank)
        base = chosen.to_dict()
        base.update({
            "signal_date": pd.Timestamp(dt),
            "market_limit_count": int(day_all["当日涨停总数"].max()) if "当日涨停总数" in day_all else len(day_all),
            "market_height": height,
            "height_non_decreasing": prior_height is None or height >= prior_height,
        })
        records.append(base)
        prior_height = height
    return pd.DataFrame(records)


def next_dates(trading_dates, dt):
    later = [d for d in trading_dates if d > dt]
    return later[0] if len(later) >= 1 else None, later[1] if len(later) >= 2 else None


def make_trades(sig, histories, pool_dates, mode):
    trades, last_exit = [], None
    for _, s in sig.sort_values("signal_date").iterrows():
        signal_date = pd.Timestamp(s["signal_date"])
        entry_date, exit_date = next_dates(pool_dates, signal_date)
        if entry_date is None or exit_date is None:
            continue
        if last_exit is not None and entry_date < last_exit:
            continue
        if mode != "benchmark":
            eligible = (
                int(s["连板数"]) >= 3
                and int(s["market_limit_count"]) >= 30
                and bool(s["height_non_decreasing"])
            )
            if not eligible:
                continue
        code = str(s["代码"]).zfill(6)
        h = histories.get(code)
        if h is None or any(d not in h.index for d in (signal_date, entry_date, exit_date)):
            continue
        signal_close = float(h.loc[signal_date, "收盘"])
        signal_open = float(h.loc[signal_date, "开盘"])
        signal_high = float(h.loc[signal_date, "最高"])
        signal_low = float(h.loc[signal_date, "最低"])
        locked_signal = (
            abs(signal_high - signal_low) < 0.001
            and abs(signal_close - signal_open) < 0.001
        )
        if mode == "strict" and locked_signal:
            continue
        entry = float(h.loc[entry_date, "开盘"])
        exit_px = float(h.loc[exit_date, "开盘"])
        premium = entry / signal_close - 1.0
        if premium >= 0.095:  # practically unavailable at a locked main-board limit-up open
            continue
        if mode == "leader_open" and not (0.0 <= premium <= 0.07):
            continue
        if mode == "strict" and not (0.0 <= premium <= 0.05):
            continue
        gross = exit_px / entry - 1.0
        net = exit_px * (1 - SELL_COST) / (entry * (1 + BUY_COST)) - 1.0
        trades.append({
            "signal_date": signal_date.date().isoformat(), "entry_date": entry_date.date().isoformat(),
            "exit_date": exit_date.date().isoformat(), "code": code, "name": s["名称"],
            "boards": int(s["连板数"]), "industry": s["所属行业"],
            "sector_support": int(s["sector_support"]), "market_height": int(s["market_height"]),
            "entry_premium": premium, "entry": entry, "exit": exit_px,
            "gross_return": gross, "net_return": net,
        })
        last_exit = exit_date
    return pd.DataFrame(trades)


def metrics(trades):
    if trades.empty:
        return {"trades": 0}
    r = trades["net_return"].astype(float)
    curve = (1 + r).cumprod()
    dd = curve / curve.cummax() - 1
    wins, losses = r[r > 0], r[r <= 0]
    return {
        "trades": int(len(r)), "win_rate": float((r > 0).mean()),
        "total_return": float(curve.iloc[-1] - 1), "avg_trade": float(r.mean()),
        "median_trade": float(r.median()), "max_drawdown": float(dd.min()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() else math.inf,
        "best_trade": float(r.max()), "worst_trade": float(r.min()),
    }


def run(refresh=False):
    full_pool = fetch_pools(refresh)
    pool = mainboard(full_pool)
    sig = signals(pool)
    histories = fetch_histories(sig["代码"].astype(str).str.zfill(6).unique(), refresh)
    pool_dates = sorted(pd.Timestamp(x) for x in pool["date"].unique())
    benchmark = make_trades(sig, histories, pool_dates, mode="benchmark")
    practical = make_trades(sig, histories, pool_dates, mode="leader_open")
    strict = make_trades(sig, histories, pool_dates, mode="strict")
    RESULTS.mkdir(exist_ok=True)
    sig.to_csv(RESULTS / "august_2026_daily_leaders.csv", index=False, encoding="utf-8-sig")
    benchmark.to_csv(RESULTS / "august_2026_benchmark_trades.csv", index=False, encoding="utf-8-sig")
    practical.to_csv(RESULTS / "august_2026_practical_trades.csv", index=False, encoding="utf-8-sig")
    strict.to_csv(RESULTS / "august_2026_strict_trades.csv", index=False, encoding="utf-8-sig")
    report = {
        "data_range": [str(min(pool_dates).date()), str(max(pool_dates).date())],
        "pool_rows_mainboard": int(len(pool)), "daily_signals": int(len(sig)),
        "costs": {"buy": BUY_COST, "sell": SELL_COST},
        "benchmark": metrics(benchmark), "practical": metrics(practical),
        "strict_turnover": metrics(strict),
    }
    (RESULTS / "august_2026_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nPRACTICAL TRADES")
    print(practical.to_string(index=False) if not practical.empty else "none")
    print("\nSTRICT TURNOVER TRADES")
    print(strict.to_string(index=False) if not strict.empty else "none")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    run(ap.parse_args().refresh)
