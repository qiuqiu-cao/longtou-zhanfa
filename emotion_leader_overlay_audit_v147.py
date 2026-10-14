"""Test whether a close-time emotion forecast improves a frozen leader Top1.

The audit never re-ranks stocks.  It joins an already-produced daily Top1 to a
point-in-time next-session emotion forecast and asks two narrow questions:

1. Does vetoing HIGH risk improve the Top1's B+2 return and large-loss rate?
2. Does LOW risk justify a positive selection bonus?

Both claims fail closed unless directions agree and a bootstrap interval
excludes zero.  Fill is a historical proxy, not proof of executable fills.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v147-emotion-leader-overlay-1"
TOP1_FIELDS = {
    "signal_date", "fill_proxy_target", "b2_net_close_mark_LABEL_ONLY",
    "b2_big_loss_mark_LABEL_ONLY",
}
EMOTION_FIELDS = {"signal_date", "risk_band", "risk_off_probability"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binary(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.dropna().isin([0, 1]).all():
        raise ValueError("binary field contains values outside 0/1")
    return numeric


def _summary(frame):
    fill = frame.fill_proxy_target.dropna()
    filled = frame[frame.fill_proxy_target.eq(1)].copy()
    returns = filled.b2_net_close_mark_LABEL_ONLY.dropna()
    losses = filled.b2_big_loss_mark_LABEL_ONLY.dropna()
    return {
        "sessions": int(len(frame)),
        "known_fill_proxy": int(len(fill)),
        "fill_proxy_rate": float(fill.mean()) if len(fill) else None,
        "known_b2_return_among_fill_proxy": int(len(returns)),
        "b2_mean_return_among_fill_proxy": float(returns.mean()) if len(returns) else None,
        "known_b2_large_loss_among_fill_proxy": int(len(losses)),
        "b2_large_loss_rate_among_fill_proxy": float(losses.mean()) if len(losses) else None,
    }


def _bootstrap_difference(frame, left_mask, right_mask, field, seed, reps=5000):
    left = pd.to_numeric(frame.loc[left_mask, field], errors="coerce").dropna().to_numpy()
    right = pd.to_numeric(frame.loc[right_mask, field], errors="coerce").dropna().to_numpy()
    if len(left) < 5 or len(right) < 5:
        return {"difference": None, "bootstrap95": [None, None],
                "left_n": int(len(left)), "right_n": int(len(right))}
    rng = np.random.default_rng(seed)
    differences = np.empty(reps)
    for index in range(reps):
        differences[index] = (
            rng.choice(left, len(left), replace=True).mean()
            - rng.choice(right, len(right), replace=True).mean()
        )
    return {
        "difference": float(left.mean() - right.mean()),
        "bootstrap95": [float(value) for value in np.quantile(differences, [.025, .975])],
        "left_n": int(len(left)),
        "right_n": int(len(right)),
    }


def audit(top1: pd.DataFrame, emotion: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if TOP1_FIELDS - set(top1):
        raise ValueError(f"top1 fields missing: {sorted(TOP1_FIELDS - set(top1))}")
    if EMOTION_FIELDS - set(emotion):
        raise ValueError(f"emotion fields missing: {sorted(EMOTION_FIELDS - set(emotion))}")
    left, right = top1.copy(), emotion.copy()
    for frame in (left, right):
        frame["signal_date"] = pd.to_datetime(frame.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
        if frame.signal_date.isna().any() or frame.signal_date.duplicated().any():
            raise ValueError("each input must have one valid row per signal date")
    if not right.risk_band.isin({"LOW", "MIXED", "HIGH"}).all():
        raise ValueError("unknown risk band")
    probability = pd.to_numeric(right.risk_off_probability, errors="coerce")
    if probability.isna().any() or not probability.between(0, 1).all():
        raise ValueError("invalid risk probability")
    left["fill_proxy_target"] = _binary(left.fill_proxy_target)
    left["b2_big_loss_mark_LABEL_ONLY"] = _binary(left.b2_big_loss_mark_LABEL_ONLY)
    left["b2_net_close_mark_LABEL_ONLY"] = pd.to_numeric(
        left.b2_net_close_mark_LABEL_ONLY, errors="coerce"
    )
    merged = left.merge(
        right[["signal_date", "risk_band", "risk_off_probability"]],
        on="signal_date", how="inner", validate="one_to_one",
    ).sort_values("signal_date").reset_index(drop=True)
    if merged.empty:
        raise ValueError("no overlapping signal dates")

    filled = merged.fill_proxy_target.eq(1)
    high, non_high = merged.risk_band.eq("HIGH"), merged.risk_band.ne("HIGH")
    low, non_low = merged.risk_band.eq("LOW"), merged.risk_band.ne("LOW")
    high_return = _bootstrap_difference(
        merged, filled & high, filled & non_high, "b2_net_close_mark_LABEL_ONLY", 14701
    )
    high_loss = _bootstrap_difference(
        merged, filled & high, filled & non_high, "b2_big_loss_mark_LABEL_ONLY", 14702
    )
    low_return = _bootstrap_difference(
        merged, filled & low, filled & non_low, "b2_net_close_mark_LABEL_ONLY", 14703
    )
    low_loss = _bootstrap_difference(
        merged, filled & low, filled & non_low, "b2_big_loss_mark_LABEL_ONLY", 14704
    )

    def interval_strictly_above_zero(item):
        return item["bootstrap95"][0] is not None and item["bootstrap95"][0] > 0

    def interval_strictly_below_zero(item):
        return item["bootstrap95"][1] is not None and item["bootstrap95"][1] < 0

    high_directions = (
        high_return["difference"] is not None and high_return["difference"] < 0
        and high_loss["difference"] is not None and high_loss["difference"] > 0
    )
    high_strong = interval_strictly_below_zero(high_return) or interval_strictly_above_zero(high_loss)
    low_directions = (
        low_return["difference"] is not None and low_return["difference"] > 0
        and low_loss["difference"] is not None and low_loss["difference"] < 0
    )
    low_strong = interval_strictly_above_zero(low_return) or interval_strictly_below_zero(low_loss)
    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "overlap": {
            "sessions": int(len(merged)),
            "start": merged.signal_date.min(),
            "end": merged.signal_date.max(),
            "risk_band_counts": {str(k): int(v) for k, v in merged.risk_band.value_counts().items()},
        },
        "groups": {
            "all": _summary(merged),
            "high": _summary(merged[high]),
            "non_high": _summary(merged[non_high]),
            "low": _summary(merged[low]),
            "non_low": _summary(merged[non_low]),
        },
        "contrasts_left_minus_right": {
            "high_minus_non_high_b2_return": high_return,
            "high_minus_non_high_b2_large_loss": high_loss,
            "low_minus_non_low_b2_return": low_return,
            "low_minus_non_low_b2_large_loss": low_loss,
        },
        "policy_evidence": {
            "high_risk_top1_veto_supported": bool(high_directions and high_strong),
            "low_risk_positive_rank_bonus_supported": bool(low_directions and low_strong),
            "allowed_use": "context_and_confirmation_strictness_only",
            "stock_reranking_allowed": False,
        },
        "governance": {
            "emotion_forecast_retrained": False,
            "stock_top1_reranked": False,
            "filled_trade_claimed": False,
            "complete_exit_claimed": False,
            "untouched_blind_test_claimed": False,
        },
        "limitations": [
            "overlap reuses historical dates already touched by project research",
            "fill_proxy_target is not broker-confirmed execution",
            "B+2 close mark is not a frozen complete-trade exit",
            "conditional-on-fill results can differ from aggregate market damage",
        ],
    }
    return merged, report


def run(top1_path, emotion_path, output_dir):
    top_path, emotion_source, target = map(Path, (top1_path, emotion_path, output_dir))
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable audit directory")
    merged, report = audit(pd.read_csv(top_path), pd.read_csv(emotion_source))
    target.mkdir(parents=True)
    merged.to_csv(target / "joined_daily_top1.csv", index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(path.resolve()): _sha(path) for path in (top_path, emotion_source)}
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top1", required=True)
    parser.add_argument("--emotion", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.top1, args.emotion, args.output_dir), ensure_ascii=False, indent=2))

