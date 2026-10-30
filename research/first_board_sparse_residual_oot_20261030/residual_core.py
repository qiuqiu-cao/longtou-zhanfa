"""Outcome-blind residual scoring and maturity-aware research inputs."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "research/first_board_top3_40_20261030/results"
KEYS = ["signal_date", "code"]
P2 = "promotion_next_LABEL_ONLY"
P4 = "reach4_LABEL_ONLY"
Z = ["z1_seal_ratio", "z2_seal_time", "z3_breakout", "z4_height"]
RAW = ["reported_close_seal_log_amount_ratio", "reported_first_seal_trading_minutes",
       "breakout_20", "height_score"]


def read(path):
    return pd.read_csv(path, dtype={"code": str}, float_precision="round_trip", low_memory=False)


def load_inputs():
    x = read(DATA / "FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv")
    y = read(DATA / "OUTCOMES_DIAGNOSTIC_ONLY.csv")
    s = read(ROOT / "research/first_second_stage_20261020_run1/S1_ALL_PREDICTIONS.csv")
    pool = x.merge(y, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not pool.pop("_merge").eq("both").all():
        raise ValueError("Features and outcomes must have identical keys")
    for col in [P2, P4]:
        if not pool[col].dropna().isin([0, 1]).all():
            raise ValueError("Invalid outcome")
    maturity = s[KEYS + ["label_end"]]
    if len(pool) != 7254 or pool.signal_date.nunique() != 286:
        raise ValueError("Historical pool contract changed")
    return x, pool, maturity


def features(x, policy="neutral"):
    """Ignore any supplied labels. Rank observed values; missing residual = zero."""
    d = x[KEYS + ["s1_score"] + RAW].copy()
    if d.duplicated(KEYS).any() or d[KEYS].isna().any().any():
        raise ValueError("Invalid candidate keys")
    if not np.isfinite(d[["s1_score", "height_score"]]).all().all():
        raise ValueError("Missing frozen model score")
    if policy not in {"neutral", "original_imputation_diagnostic"}:
        raise ValueError("Invalid missing-value policy")
    out = d[KEYS + ["s1_score"]].copy()
    for z, raw in zip(Z, RAW):
        v = d[raw].replace([np.inf, -np.inf], np.nan)
        out[z + "_missing"] = v.isna()
        if policy == "original_imputation_diagnostic":
            v = v.fillna(240. if z == "z2_seal_time" else 0.)
        out[z] = (v.groupby(d.signal_date).rank(method="average", pct=True,
                    ascending=z != "z2_seal_time") - .5).fillna(0.)
    return out


def select(f, theta):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (len(Z),) or not np.isfinite(theta).all():
        raise ValueError("Invalid coefficient vector")
    d = f[KEYS + ["s1_score"] + Z].copy()
    if d.duplicated(KEYS).any() or not np.isfinite(d[["s1_score"] + Z]).all().all():
        raise ValueError("Invalid scoring input")
    d["score"] = d.s1_score.to_numpy() + d[Z].to_numpy() @ theta
    d = d.sort_values(["signal_date", "score", "code"], ascending=[True, False, True])
    d["slot"] = d.groupby("signal_date").cumcount() + 1
    return d.loc[d.slot.le(3), KEYS + ["slot", "score"]].reset_index(drop=True)


def mature_training(pool, maturity, fit_as_of):
    """Keep complete daily pools only when all label horizons have closed.

    A missing outcome stays unknown even if its horizon has matured. The original
    evaluation pool is never shortened by this training-only purge.
    """
    cutoff = pd.Timestamp(fit_as_of)
    d = pool.merge(maturity, on=KEYS, how="left", validate="one_to_one")
    signal = pd.to_datetime(d.signal_date, errors="raise")
    ends = pd.to_datetime(d.label_end, errors="coerce")
    if ((ends <= signal) & ends.notna()).any():
        raise ValueError("Label end must follow the signal day")
    requested = signal.le(cutoff)
    row_mature = ends.notna() & ends.le(cutoff)
    day_mature = row_mature.groupby(d.signal_date).transform("all")
    excluded = d.loc[requested & ~day_mature].copy()
    excluded["purge_reason"] = "DAILY_LABEL_HORIZON_NOT_COMPLETE_AT_FIT_CUTOFF"
    train = d.loc[requested & day_mature].copy()
    if train.empty:
        raise ValueError("No mature training dates")
    return train, excluded


def evaluate(pool, picks):
    path = ROOT / "research/first_board_math_20261030/review_gemini_cascade.py"
    spec = importlib.util.spec_from_file_location("residual_existing_contract", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    report, detail = mod.evaluate(pool, picks, sorted(pool.signal_date.unique()))
    detail = detail.merge(pool[KEYS + ["one_price"]], on=KEYS, validate="one_to_one")
    for state, name in [(0, "non_one_price"), (1, "one_price")]:
        g = detail[detail.one_price.eq(state)]
        report[name] = dict(selected=len(g), known=int(g[P2].notna().sum()),
                           hits=int(g[P2].eq(1).sum()), unknown=int(g[P2].isna().sum()))
    report["one_price_is_signal_day_status_not_next_day_buyability"] = True
    return report, detail
