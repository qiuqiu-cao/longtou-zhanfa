"""A-share continuous-board leader system v2.

Commands:
  fetch     Build a 24-month public archive from lianban.net JSON-LD.
  train     Build labels, fit a regularized discrete-hazard model, and validate by time.
  current   Score the latest board ladder with role, probability, confidence, and buyability.
  all       Run fetch, train, and current.

Research only. Historical archive data has daily board ladders and market summaries,
but no Level-2/tick queue. Current-day Eastmoney pool enriches execution fields when present.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import html
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "history_v2"
DAILY_CACHE = DATA / "daily"
RESULTS = ROOT / "results" / "v2"
MODEL_PATH = RESULTS / "hazard_model.json"
START = "2024-09-01"
END = "2026-08-28"

FEATURES = [
    "boards", "height_gap", "is_space_leader", "emotion_score",
    "limit_count_scaled", "consecutive_scaled", "down_limits_scaled",
    "seal_rate_scaled", "failed_rate", "height_change_scaled",
    "limit_change_scaled", "promotion_overall", "promotion_mid",
    "space_feedback", "middle_density", "unique_space",
    "breadth_height_divergence", "new_cycle_flag",
]


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))


def norm_name(name):
    return re.sub(r"\s+", "", str(name)).replace("★", "")


def extract_int(pattern, text, default=None):
    m = re.search(pattern, text)
    return int(m.group(1)) if m else default


def extract_float(pattern, text, default=None):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default


def parse_page(date, text, url):
    blocks = re.findall(r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>', text, re.S | re.I)
    dataset, ladder = None, []
    for block in blocks:
        try:
            obj = json.loads(html.unescape(block))
        except json.JSONDecodeError:
            continue
        if obj.get("@type") == "Dataset":
            dataset = obj
        if obj.get("@type") == "ItemList" and "连板天梯" in obj.get("name", ""):
            for item in obj.get("itemListElement", []):
                m = re.match(r"(.+?)\s+(\d+)板$", item.get("name", ""))
                if m:
                    ladder.append({"name": norm_name(m.group(1)), "boards": int(m.group(2))})
    if not dataset or not ladder:
        return None
    desc = dataset.get("description", "")
    limit_count = extract_int(r"涨停\s*(\d+)\s*家", desc)
    consecutive_count = extract_int(r"连板\s*(\d+)\s*家", desc, len(ladder))
    max_board = extract_int(r"最高\s*(\d+)\s*板", desc, max(x["boards"] for x in ladder))
    down_limits = extract_int(r"跌停\s*(\d+)\s*家", desc, 0)
    seal_rate = extract_float(r"封板率\s*([\d.]+)%", desc, 0.0)
    failed_count = extract_int(r"炸板\s*(\d+)\s*家", desc, 0)
    emotion_match = re.search(r"情绪(?:阶段)?[：:]?\s*([^。；,，]+?期)", desc)
    mainline_match = re.search(r"主线题材[：:]\s*([^。]+)", desc)
    if limit_count is None:
        return None
    return {
        "date": date, "source_url": url, "source": "lianban.net JSON-LD",
        "limit_count": limit_count, "consecutive_count": consecutive_count,
        "max_board": max_board, "down_limits": down_limits,
        "seal_rate": seal_rate, "failed_count": failed_count,
        "published_emotion": emotion_match.group(1) if emotion_match else "未提供",
        "mainlines": mainline_match.group(1) if mainline_match else "未提供",
        "ladder": ladder,
    }


def fetch_one(ts, refresh=False):
    DAILY_CACHE.mkdir(parents=True, exist_ok=True)
    ds = ts.strftime("%Y-%m-%d")
    path = DAILY_CACHE / f"{ds}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    url = f"https://lianban.net/days/{ds}.html"
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 research"})
            if response.status_code == 200:
                parsed = parse_page(ds, response.text, url)
                if parsed:
                    path.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
                    return parsed
            return None
        except requests.RequestException:
            time.sleep(0.4 * (attempt + 1))
    return None


def fetch_archive(refresh=False):
    dates = list(pd.date_range(START, END, freq="B"))
    records = []
    with cf.ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_one, ts, refresh): ts for ts in dates}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            record = fut.result()
            if record:
                records.append(record)
            if i % 100 == 0:
                print(f"archive progress {i}/{len(dates)}; trading days={len(records)}")
    records.sort(key=lambda x: x["date"])
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "archive.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"archive saved: {len(records)} trading days")
    return records


def promotion_stats(prev_ladder, current_ladder):
    current = {x["name"]: x["boards"] for x in current_ladder}
    eligible = [x for x in prev_ladder if x["boards"] >= 2]
    mid = [x for x in eligible if 2 <= x["boards"] <= 4]

    def rate(rows):
        if not rows:
            return 0.0
        return sum(current.get(x["name"]) == x["boards"] + 1 for x in rows) / len(rows)

    prev_max = max((x["boards"] for x in prev_ladder), default=1)
    prev_space = [x for x in prev_ladder if x["boards"] == prev_max]
    feedback = float(any(current.get(x["name"]) == x["boards"] + 1 for x in prev_space))
    return rate(eligible), rate(mid), feedback


def emotion_features(day, prev=None):
    ladder = [x for x in day["ladder"] if x["boards"] >= 2]
    middle = sum(2 <= x["boards"] <= 4 for x in ladder)
    max_count = sum(x["boards"] == day["max_board"] for x in ladder)
    if prev:
        promo, promo_mid, feedback = promotion_stats(prev["ladder"], day["ladder"])
        height_change = day["max_board"] - prev["max_board"]
        limit_change = day["limit_count"] - prev["limit_count"]
    else:
        promo = promo_mid = feedback = 0.5
        height_change = limit_change = 0
    failed_rate = day["failed_count"] / max(1, day["limit_count"] + day["failed_count"])
    middle_density = middle / max(1, len(ladder))
    width = clamp((day["limit_count"] - 20) / 80)
    height = clamp((day["max_board"] - 2) / 6)
    height_trend = clamp((height_change + 3) / 6)
    loss_health = 1 - clamp(0.55 * failed_rate / 0.30 + 0.45 * day["down_limits"] / 10)
    structure = clamp(0.55 * middle_density + 0.45 * (1 if max_count == 1 else 0.4))
    score = 100 * (
        0.15 * width + 0.15 * height + 0.10 * height_trend + 0.20 * promo
        + 0.15 * loss_health + 0.15 * structure + 0.10 * feedback
    )
    breadth_height_divergence = float(height_change > 0 and limit_change < -20)
    new_cycle = float(prev is not None and prev["max_board"] - day["max_board"] >= 2 and day["max_board"] <= 4)
    if height_change <= -2 or (feedback == 0 and promo_mid < 0.15 and day["down_limits"] >= 5):
        state = "退潮"
    elif breadth_height_divergence or failed_rate >= 0.25 or (height_change < 0 and promo_mid < 0.25):
        state = "大分歧"
    elif height_change > 0 and promo_mid >= 0.25 and day["limit_count"] >= 40:
        state = "加速"
    elif limit_change >= 15 and height_change >= 0 and feedback >= 0.5:
        state = "启动"
    elif score >= 70 and day["max_board"] >= 5:
        state = "一致"
    else:
        state = "修复/混沌"
    return {
        "emotion_score": score / 100, "emotion_score_100": round(score, 1), "emotion_state": state,
        "promotion_overall": promo, "promotion_mid": promo_mid, "space_feedback": feedback,
        "height_change": height_change, "limit_change": limit_change, "failed_rate": failed_rate,
        "middle_density": middle_density, "unique_space": float(max_count == 1),
        "breadth_height_divergence": breadth_height_divergence, "new_cycle_flag": new_cycle,
    }


def build_dataset(records):
    market_rows, obs = [], []
    enriched = []
    for i, day in enumerate(records):
        ef = emotion_features(day, records[i - 1] if i else None)
        item = {**day, **ef}
        enriched.append(item)
        market_rows.append({k: v for k, v in item.items() if k != "ladder"})
    for i, day in enumerate(enriched[:-1]):
        nxt = {x["name"]: x["boards"] for x in enriched[i + 1]["ladder"]}
        for x in day["ladder"]:
            if x["boards"] < 2:
                continue
            row = {
                "date": day["date"], "name": x["name"], "boards": x["boards"],
                "height_gap": day["max_board"] - x["boards"],
                "is_space_leader": float(x["boards"] == day["max_board"]),
                "promoted_next_day": float(nxt.get(x["name"]) == x["boards"] + 1),
                "emotion_state": day["emotion_state"], "emotion_score": day["emotion_score"],
                "limit_count_scaled": clamp(day["limit_count"] / 100),
                "consecutive_scaled": clamp(day["consecutive_count"] / 30),
                "down_limits_scaled": clamp(day["down_limits"] / 10),
                "seal_rate_scaled": clamp(day["seal_rate"] / 100),
                "failed_rate": day["failed_rate"],
                "height_change_scaled": clamp((day["height_change"] + 3) / 6),
                "limit_change_scaled": clamp((day["limit_change"] + 80) / 160),
                "promotion_overall": day["promotion_overall"], "promotion_mid": day["promotion_mid"],
                "space_feedback": day["space_feedback"], "middle_density": day["middle_density"],
                "unique_space": day["unique_space"],
                "breadth_height_divergence": day["breadth_height_divergence"],
                "new_cycle_flag": day["new_cycle_flag"],
            }
            obs.append(row)
    market = pd.DataFrame(market_rows)
    observations = pd.DataFrame(obs)
    DATA.mkdir(parents=True, exist_ok=True)
    market.to_csv(DATA / "market_daily.csv", index=False, encoding="utf-8-sig")
    observations.to_csv(DATA / "board_observations.csv", index=False, encoding="utf-8-sig")
    return market, observations


def local_august_records(archive_records=None):
    """Use the locally cached August pools when the long archive is permission-limited."""
    pool_dir = ROOT / "data" / "2026-08"
    archive_map = {x["date"]: x for x in (archive_records or [])}
    records = []
    for path in sorted(pool_dir.glob("zt_pool_202608*.csv")):
        df = pd.read_csv(path, dtype={"代码": str})
        if "连板数" not in df or "名称" not in df:
            continue
        df["代码"] = df["代码"].astype(str).str.zfill(6)
        df = df[df["代码"].str.match(r"^(60|00)")]
        df = df[~df["名称"].astype(str).str.contains(r"ST|退", case=False, regex=True)]
        ladder_df = df[df["连板数"] >= 2]
        if ladder_df.empty:
            continue
        date = pd.Timestamp(path.stem[-8:]).strftime("%Y-%m-%d")
        base = archive_map.get(date, {})
        industries = df.get("所属行业", pd.Series(dtype=str)).value_counts()
        records.append({
            "date": date, "source_url": base.get("source_url", "Eastmoney/AkShare local cache"),
            "source": "Eastmoney/AkShare + lianban fallback",
            "limit_count": int(df["当日涨停总数"].max()) if "当日涨停总数" in df else len(df),
            "consecutive_count": len(ladder_df), "max_board": int(ladder_df["连板数"].max()),
            "down_limits": int(base.get("down_limits", 0)),
            "seal_rate": float(base.get("seal_rate", 0)),
            "failed_count": int(base.get("failed_count", 0)),
            "published_emotion": base.get("published_emotion", "未提供"),
            "mainlines": base.get("mainlines", "、".join(industries.head(3).index.astype(str))),
            "missing_fields": ["down_limits", "seal_rate", "failed_count"] if not base else [],
            "ladder": [{"name": norm_name(r["名称"]), "boards": int(r["连板数"])}
                       for _, r in ladder_df.iterrows()],
        })
    return records


def best_available_records(records=None):
    archive = records
    if archive is None and (DATA / "archive.json").exists():
        archive = json.loads((DATA / "archive.json").read_text(encoding="utf-8"))
    archive = archive or []
    local = local_august_records(archive)
    if len(archive) < 100 and local:
        print(f"long archive insufficient ({len(archive)} days); using local diagnostic set ({len(local)} days)")
        return local
    return archive


@dataclass
class LogisticModel:
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray

    def predict(self, frame):
        x = frame[FEATURES].to_numpy(float)
        z = np.c_[np.ones(len(x)), (x - self.mean) / self.std] @ self.weights
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    def to_dict(self):
        return {"features": FEATURES, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "weights": self.weights.tolist()}

    @classmethod
    def from_dict(cls, obj):
        return cls(np.array(obj["mean"]), np.array(obj["std"]), np.array(obj["weights"]))


def fit_logistic(frame, l2=0.12, epochs=3500, lr=0.04):
    x = frame[FEATURES].to_numpy(float)
    y = frame["promoted_next_day"].to_numpy(float)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-8] = 1.0
    xs = (x - mean) / std
    design = np.c_[np.ones(len(xs)), xs]
    w = np.zeros(design.shape[1])
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-np.clip(design @ w, -30, 30)))
        grad = design.T @ (p - y) / len(y)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return LogisticModel(mean, std, w)


def auc_score(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return None
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def evaluate(frame, model):
    if frame.empty:
        return {"n": 0}
    y = frame["promoted_next_day"].to_numpy(float)
    p = model.predict(frame)
    bins = np.minimum((p * 5).astype(int), 4)
    ece = sum((bins == b).mean() * abs(p[bins == b].mean() - y[bins == b].mean())
              for b in range(5) if (bins == b).any())
    return {
        "n": int(len(y)), "positive_rate": float(y.mean()), "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(np.clip(p, 1e-8, 1)) + (1-y) * np.log(np.clip(1-p, 1e-8, 1)))),
        "auc": auc_score(y, p), "calibration_error": float(ece),
    }


def time_split(frame):
    dates = np.array(sorted(frame["date"].unique()))
    a, b = dates[int(len(dates) * 0.60)], dates[int(len(dates) * 0.80)]
    return frame[frame.date < a], frame[(frame.date >= a) & (frame.date < b)], frame[frame.date >= b]


def walk_forward(frame):
    dates = np.array(sorted(frame["date"].unique()))
    cuts = [(0.50, 0.65), (0.65, 0.80), (0.80, 1.00)]
    rows = []
    for fold, (train_end, test_end) in enumerate(cuts, 1):
        a, b = dates[int(len(dates) * train_end)], dates[min(len(dates)-1, int(len(dates) * test_end))]
        train = frame[frame.date < a]
        test = frame[(frame.date >= a) & (frame.date <= b)]
        model = fit_logistic(train)
        rows.append({"fold": fold, "train_end": a, "test_end": b, **evaluate(test, model)})
    return rows


def train_system(records=None):
    records = best_available_records(records)
    market, frame = build_dataset(records)
    train, validation, blind = time_split(frame)
    selection_model = fit_logistic(train)
    validation_metrics = evaluate(validation, selection_model)
    blind_metrics = evaluate(blind, selection_model)
    blind_leaders = evaluate(blind[blind.is_space_leader == 1], selection_model)
    wf = walk_forward(frame)
    # Final deployment model is refit only after all evaluation numbers are frozen.
    final_model = fit_logistic(frame)
    RESULTS.mkdir(parents=True, exist_ok=True)
    deployment_status = "RESEARCH_READY" if (
        len(market) >= 250 and blind_metrics.get("n", 0) >= 200
        and (blind_metrics.get("auc") or 0) >= 0.55
        and blind_metrics.get("calibration_error", 1) <= 0.08
    ) else "DIAGNOSTIC_ONLY"
    artifact = {
        "model": final_model.to_dict(), "data_start": frame.date.min(), "data_end": frame.date.max(),
        "observations": len(frame), "validation": validation_metrics, "blind": blind_metrics,
        "blind_space_leaders": blind_leaders, "walk_forward": wf,
        "deployment_status": deployment_status,
        "height_samples": frame.groupby("boards")["promoted_next_day"].agg(["count", "mean"]).reset_index().to_dict("records"),
        "limitations": ["historical topic mapping unavailable", "Level-2/tick queue unavailable",
                        "probability is board continuation, not guaranteed executable return",
                        f"available training history is only {len(market)} trading days"],
    }
    MODEL_PATH.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: artifact[k] for k in ("observations", "validation", "blind", "blind_space_leaders", "walk_forward")}, ensure_ascii=False, indent=2))
    return artifact, market, frame


def load_current_pool(date):
    path = ROOT / "data" / "2026-08" / f"zt_pool_{date.replace('-', '')}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"代码": str})
    df["name_norm"] = df["名称"].map(norm_name)
    counts = df.groupby("所属行业").size().to_dict()
    df["sector_support"] = df["所属行业"].map(counts).fillna(1)
    return df


def load_context_overlay(date):
    path = ROOT / f"current_snapshot_{date}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {norm_name(r["name"]): r.to_dict() for _, r in df.iterrows()}


def shrunk_height_probability(raw_probability, boards, artifact):
    samples = artifact["height_samples"]
    total_n = sum(int(x["count"]) for x in samples)
    global_rate = sum(int(x["count"]) * float(x["mean"]) for x in samples) / max(1, total_n)
    item = next((x for x in samples if int(x["boards"]) == int(boards)), None)
    if item:
        n, mean = int(item["count"]), float(item["mean"])
        posterior = (n * mean + 12 * global_rate) / (n + 12)
    else:
        n, posterior = 0, global_rate
    blind = artifact.get("blind", {})
    model_reliable = blind.get("n", 0) >= 200 and (blind.get("auc") or 0) >= 0.55
    blended = raw_probability if model_reliable else 0.35 * raw_probability + 0.65 * posterior
    return float(blended), n, model_reliable


def height_survival(feat, model, artifact, max_board=10):
    current = int(feat["boards"])
    survive, curve = 1.0, {}
    for target in range(current + 1, max_board + 1):
        projected = dict(feat)
        projected["boards"] = target - 1
        projected["height_gap"] = 0
        projected["is_space_leader"] = 1
        raw = float(model.predict(pd.DataFrame([projected]))[0])
        step, _, _ = shrunk_height_probability(raw, target - 1, artifact)
        survive *= step
        curve[target] = survive
    masses = {current: 1 - curve.get(current + 1, 0)}
    for h in range(current + 1, max_board):
        masses[h] = curve[h] - curve[h + 1]
    masses[max_board] = curve.get(max_board, 0)
    return curve, max(masses, key=masses.get)


def current_features(day, prev, candidate):
    ef = emotion_features(day, prev)
    row = {
        "boards": candidate["boards"], "height_gap": day["max_board"] - candidate["boards"],
        "is_space_leader": float(candidate["boards"] == day["max_board"]),
        "emotion_score": ef["emotion_score"], "limit_count_scaled": clamp(day["limit_count"] / 100),
        "consecutive_scaled": clamp(day["consecutive_count"] / 30),
        "down_limits_scaled": clamp(day["down_limits"] / 10), "seal_rate_scaled": clamp(day["seal_rate"] / 100),
        "failed_rate": ef["failed_rate"], "height_change_scaled": clamp((ef["height_change"] + 3) / 6),
        "limit_change_scaled": clamp((ef["limit_change"] + 80) / 160),
        "promotion_overall": ef["promotion_overall"], "promotion_mid": ef["promotion_mid"],
        "space_feedback": ef["space_feedback"], "middle_density": ef["middle_density"],
        "unique_space": ef["unique_space"], "breadth_height_divergence": ef["breadth_height_divergence"],
        "new_cycle_flag": ef["new_cycle_flag"],
    }
    return row, ef


def current_report(records=None, artifact=None):
    records = best_available_records(records)
    artifact = artifact or json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    model = LogisticModel.from_dict(artifact["model"])
    day, prev = records[-1], records[-2]
    pool = load_current_pool(day["date"])
    overlay = load_context_overlay(day["date"])
    pool_map = {} if pool is None else pool.set_index("name_norm").to_dict("index")
    rows = []
    for candidate in day["ladder"]:
        if candidate["boards"] < 2:
            continue
        feat, ef = current_features(day, prev, candidate)
        raw_p = float(model.predict(pd.DataFrame([feat]))[0])
        p, height_n, model_reliable = shrunk_height_probability(raw_p, candidate["boards"], artifact)
        curve, height_mode = height_survival(feat, model, artifact)
        raw = pool_map.get(candidate["name"])
        context = overlay.get(candidate["name"], {})
        if raw:
            code = str(raw.get("代码", "")).zfill(6)
            name = str(raw.get("名称", ""))
            if not re.match(r"^(60|00)", code) or re.search(r"ST|退", name, re.I):
                continue
        sector_support = int(raw.get("sector_support", 0)) if raw else 0
        industry = raw.get("所属行业", "未核验") if raw else "未核验"
        first = str(raw.get("首次封板时间", "")) if raw else ""
        last = str(raw.get("最后封板时间", "")) if raw else ""
        bombs = int(raw.get("炸板次数", 0)) if raw else 0
        turnover = float(raw.get("换手率", 0)) if raw else 0
        locked = raw is not None and first.startswith("92500") and last.startswith("92500") and bombs == 0 and turnover < 3
        sector_ladder = float(context.get("sector_ladder", 0.5 if sector_support >= 2 else 0.2))
        sector_divergence = float(context.get("sector_divergence", 0.5))
        rival_pressure = float(context.get("rival_pressure", 0.5))
        anchor_health = float(context.get("anchor_health", 0.5))
        structure_block = sector_divergence >= 0.75 or rival_pressure >= 0.80 or anchor_health <= 0.30
        roles = []
        if feat["is_space_leader"]:
            roles.append("空间龙/情绪锚")
        if sector_support >= 2 and candidate["boards"] >= 3 and sector_ladder >= 0.60:
            roles.append("主线核心候选")
        if ef["new_cycle_flag"] and candidate["boards"] >= 3:
            roles.append("新周期候选")
        old_cycle_risk = candidate["boards"] >= 6 or (
            candidate["boards"] >= 5 and ef["emotion_state"] in ("一致", "大分歧"))
        buyability = "低（一字/锁死）" if locked else ("中（需竞价回封）" if raw is not None else "未知")
        if (3 <= candidate["boards"] <= 5 and not locked and not structure_block
                and ef["emotion_state"] in ("启动", "加速", "修复/混沌")
                and ("主线核心候选" in roles or ef["new_cycle_flag"]) and not old_cycle_risk):
            roles.append("条件可交易龙")
        confidence = "高" if model_reliable and height_n >= 300 and raw is not None else (
            "中" if model_reliable and height_n >= 100 else "低")
        curve_text = " ".join(f">={h}板{prob:.1%}" for h, prob in list(curve.items())[:3])
        if structure_block:
            risk = "板块大分歧/竞争闸门"
        elif old_cycle_risk:
            risk = "旧周期高位管理"
        elif raw is None:
            risk = "板块字段未核验"
        else:
            risk = "需盘中确认"
        rows.append({
            "股票": candidate["name"], "连板": candidate["boards"], "行业": industry,
            "身份": "、".join(roles) or "连板跟随", "情绪阶段": ef["emotion_state"],
            "情绪温度": ef["emotion_score_100"], "次板概率": p, "概率置信度": confidence,
            "终止高度众数": height_mode, "高度生存曲线": curve_text,
            "可买性": buyability, "板块涨停支持": sector_support,
            "风险": risk,
        })
    out = pd.DataFrame(rows).sort_values(["连板", "次板概率"], ascending=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / f"current_roles_{day['date']}.csv", index=False, encoding="utf-8-sig")
    summary = {
        "date": day["date"], "emotion": emotion_features(day, prev),
        "data_quality": {"verified": "daily ladder, market breadth, current daily pool",
                         "unavailable": "Level-2 queue, tick cancellation, exact live fill",
                         "decision": "conditional plan only",
                         "model_status": artifact.get("deployment_status", "DIAGNOSTIC_ONLY")},
        "candidates": out.to_dict("records"),
    }
    (RESULTS / f"current_report_{day['date']}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out.to_string(index=False, formatters={"次板概率": "{:.1%}".format}))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["fetch", "train", "current", "all"])
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    records = None; artifact = None
    if args.command in ("fetch", "all"):
        records = fetch_archive(args.refresh)
    if args.command in ("train", "all"):
        artifact, _, _ = train_system(records)
    if args.command in ("current", "all"):
        current_report(records, artifact)


if __name__ == "__main__":
    main()
