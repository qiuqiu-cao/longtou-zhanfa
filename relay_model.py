"""Main-board A-share continuous-limit-up leader model (research only)."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

FEATURES = [
    "boards", "cycle", "height_expanding", "promotion", "loss_effect",
    "sector_ladder", "sector_divergence", "anchor_health", "rival_pressure",
    "identity", "turnover_quality", "auction_open", "announcement_safety",
    "buyability", "capital_consensus", "capital_continuity",
]

# Conservative prior, deliberately shrinked. Replace by time-split fitted coefficients.
PRIOR = {
    "intercept": -2.55,
    "boards": -0.13,
    "cycle": 0.48,
    "height_expanding": 0.28,
    "promotion": 0.28,
    "loss_effect": 0.30,
    "sector_ladder": 0.52,
    "sector_divergence": -0.55,
    "anchor_health": 0.45,
    "rival_pressure": -0.35,
    "identity": 0.48,
    "turnover_quality": 0.35,
    "auction_open": 0.50,
    "announcement_safety": 0.30,
    "buyability": 0.45,
    "capital_consensus": 0.35,
    "capital_continuity": 0.30,
}

WEIGHTS = {
    "cycle": 12, "height_expanding": 5, "promotion": 7, "loss_effect": 6,
    "sector_ladder": 12, "sector_divergence": -8, "anchor_health": 5,
    "rival_pressure": -5, "identity": 10, "turnover_quality": 8,
    "auction_open": 8, "announcement_safety": 5, "buyability": 5,
    "capital_consensus": 4, "capital_continuity": 3,
}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, x))))


def row_features(row: dict[str, str]) -> dict[str, float]:
    return {k: float(row[k]) for k in FEATURES}


def strength(x: dict[str, float]) -> float:
    height = min(15.0, 5.0 + 1.5 * x["boards"] + 3.0 * x["height_expanding"])
    return round(height + sum(WEIGHTS[k] * x[k] for k in WEIGHTS), 1)


def next_probability(x: dict[str, float], model: dict[str, float]) -> float:
    z = model["intercept"] + sum(model[k] * x[k] for k in FEATURES)
    return sigmoid(z)


def survival(x: dict[str, float], model: dict[str, float], max_board: int = 10):
    current = int(x["boards"])
    probs, survive = {}, 1.0
    for target in range(current + 1, max_board + 1):
        step = dict(x)
        step["boards"] = target - 1
        # Evidence decays as uncertainty compounds; height penalty remains in model.
        decay = max(0.55, 1.0 - 0.05 * (target - current - 1))
        for key in ("sector_ladder", "anchor_health", "identity", "turnover_quality",
                    "auction_open", "capital_consensus", "capital_continuity"):
            step[key] *= decay
        survive *= next_probability(step, model)
        probs[target] = survive
    masses = {current: 1.0 - probs.get(current + 1, 0.0)}
    for h in range(current + 1, max_board):
        masses[h] = probs[h] - probs[h + 1]
    masses[max_board] = probs[max_board]
    mode = max(masses, key=masses.get)
    cumulative, lo, hi = 0.0, None, max_board
    for h in range(current, max_board + 1):
        cumulative += masses[h]
        if cumulative >= 0.10 and lo is None:
            lo = h
        if cumulative >= 0.90:
            hi = h
            break
    return probs, mode, (lo if lo is not None else current, hi)


def load_model(path: str | None) -> dict[str, float]:
    if not path:
        return dict(PRIOR)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score_file(path: str, model_path: str | None):
    model = load_model(model_path)
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    output = []
    for row in rows:
        x = row_features(row)
        probs, mode, interval = survival(x, model)
        output.append((strength(x), row, probs, mode, interval))
    for s, row, probs, mode, interval in sorted(output, reverse=True, key=lambda z: z[0]):
        nxt = int(float(row["boards"])) + 1
        curve = " ".join(f">={h}板:{p:.1%}" for h, p in list(probs.items())[:4])
        print(f'{row["name"]} {int(float(row["boards"]))}板 | 强度{s:.1f} | 次板{probs[nxt]:.1%} | '
              f'终止众数{mode}板 | 80%区间{interval[0]}-{interval[1]}板 | {curve} | {row.get("risk_note", "")}')


def train(path: str, out: str, epochs: int = 4000, lr: float = 0.03, l2: float = 0.08):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 100:
        raise SystemExit("至少需要100条板级观察；建议500条以上，并按日期做样本外验证。")
    xs, ys = [row_features(r) for r in rows], [float(r["promoted_next_day"]) for r in rows]
    model = dict(PRIOR)
    for _ in range(epochs):
        grads = {"intercept": 0.0, **{k: 0.0 for k in FEATURES}}
        for x, y in zip(xs, ys):
            err = next_probability(x, model) - y
            grads["intercept"] += err
            for k in FEATURES:
                grads[k] += err * x[k]
        n = len(xs)
        model["intercept"] -= lr * grads["intercept"] / n
        for k in FEATURES:
            model[k] -= lr * (grads[k] / n + l2 * model[k])
    Path(out).write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}; n={len(rows)}. 仍须使用更晚日期的盲测集检查校准。")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("csv"); s.add_argument("--model")
    t = sub.add_parser("train"); t.add_argument("csv"); t.add_argument("output")
    args = p.parse_args()
    score_file(args.csv, args.model) if args.cmd == "score" else train(args.csv, args.output)


if __name__ == "__main__":
    main()
