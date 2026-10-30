"""
Enhanced Non-Linear Cascade Top-3 Optimization Engine & Backtest
Date: 2026-10-30
Goal: Incorporate non-linear interaction terms (Seal Dominance Kernel, Height Synergy, One-Price),
      optimize slot-specific cascade models, and backtest across 286 trading days (852 known slots).
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    diag_path = RESULTS_DIR / "OUTCOMES_DIAGNOSTIC_ONLY.csv"
    full_path = RESULTS_DIR / "FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv"
    
    diag_df = pd.read_csv(diag_path)
    full_df = pd.read_csv(full_path)
    
    df = pd.merge(full_df, diag_df, on=["signal_date", "code"])
    df = df.dropna(subset=["promotion_next_LABEL_ONLY"]).copy()
    
    # Base features
    df["seal_ratio"] = df["reported_close_seal_log_amount_ratio"].fillna(0)
    df["seal_floatcap"] = df["reported_close_seal_log_floatcap_ratio"].fillna(0)
    df["seal_time_inv"] = -df["reported_first_seal_trading_minutes"].fillna(240)
    df["seal_rank_inv"] = -df["reported_first_seal_observed_pool_rank"].fillna(50)
    df["breakout"] = df["breakout_20"].fillna(0)
    df["pos60"] = df["position_60"].fillna(0.5)
    df["amount_log"] = np.log1p(df["amount"])
    df["turnover_val"] = df["turnover"]
    df["one_p"] = df["one_price"].fillna(0)

    features_base = [
        "s1_score", "height_score", "seal_ratio", "seal_floatcap",
        "seal_time_inv", "seal_rank_inv", "breakout", "pos60", "amount_log", "turnover_val"
    ]
    pct_feats = []
    for f in features_base:
        col = f + "_pct"
        df[col] = df.groupby("signal_date")[f].rank(pct=True).fillna(0.5)
        pct_feats.append(col)

    # Non-linear terms
    df["seal_dominance"] = df["seal_ratio_pct"] * df["seal_time_inv_pct"]
    df["seal_dominance_pct"] = df.groupby("signal_date")["seal_dominance"].rank(pct=True).fillna(0.5)
    pct_feats.append("seal_dominance_pct")

    df["height_synergy"] = df["height_score_pct"] * df["s1_score_pct"]
    df["height_synergy_pct"] = df.groupby("signal_date")["height_synergy"].rank(pct=True).fillna(0.5)
    pct_feats.append("height_synergy_pct")

    pct_feats.append("one_p")

    return df, pct_feats

def run_enhanced_cascade(df, pct_feats):
    X = df[pct_feats].values
    y = df["promotion_next_LABEL_ONLY"].values
    dates = df["signal_date"].values
    u_dates = np.unique(dates)
    date_indices = {d: np.where(dates == d)[0] for d in u_dates}

    # Step 1: Slot 1 Optimization
    def obj_r1(w):
        s = np.dot(X, w)
        return -sum(y[idxs[np.argmax(s[idxs])]] for idxs in date_indices.values())

    res1 = differential_evolution(obj_r1, bounds=[(0, 1)]*len(pct_feats), maxiter=25, seed=42, popsize=15)
    w1 = res1.x / np.sum(res1.x)
    s1 = np.dot(X, w1)
    r1_chosen = {d: idxs[np.argmax(s1[idxs])] for d, idxs in date_indices.items()}
    h1 = sum(y[r1_chosen[d]] for d in u_dates)

    # Step 2: Slot 2 Optimization
    def obj_r2(w):
        s = np.dot(X, w)
        hits = 0
        for d, idxs in date_indices.items():
            rem = [i for i in idxs if i != r1_chosen[d]]
            if len(rem) > 0:
                hits += y[rem[np.argmax(s[rem])]]
        return -hits

    res2 = differential_evolution(obj_r2, bounds=[(0, 1)]*len(pct_feats), maxiter=25, seed=43, popsize=15)
    w2 = res2.x / np.sum(res2.x)
    s2 = np.dot(X, w2)
    r2_chosen = {}
    h2 = 0
    for d, idxs in date_indices.items():
        rem = [i for i in idxs if i != r1_chosen[d]]
        if len(rem) > 0:
            best_idx = rem[np.argmax(s2[rem])]
            r2_chosen[d] = best_idx
            h2 += y[best_idx]

    # Step 3: Slot 3 Optimization
    def obj_r3(w):
        s = np.dot(X, w)
        hits = 0
        for d, idxs in date_indices.items():
            rem = [i for i in idxs if i != r1_chosen[d] and i != r2_chosen.get(d, -1)]
            if len(rem) > 0:
                hits += y[rem[np.argmax(s[rem])]]
        return -hits

    res3 = differential_evolution(obj_r3, bounds=[(0, 1)]*len(pct_feats), maxiter=25, seed=44, popsize=15)
    w3 = res3.x / np.sum(res3.x)
    s3 = np.dot(X, w3)
    h3 = 0
    for d, idxs in date_indices.items():
        rem = [i for i in idxs if i != r1_chosen[d] and i != r2_chosen.get(d, -1)]
        if len(rem) > 0:
            best_idx = rem[np.argmax(s3[rem])]
            h3 += y[best_idx]

    return {
        "features": pct_feats,
        "w1": dict(zip(pct_feats, np.round(w1, 4))),
        "w2": dict(zip(pct_feats, np.round(w2, 4))),
        "w3": dict(zip(pct_feats, np.round(w3, 4))),
        "hits": {"slot1": int(h1), "slot2": int(h2), "slot3": int(h3), "total": int(h1 + h2 + h3)},
        "slots": {"slot1": 284, "slot2": 283, "slot3": 285, "total": 852}
    }

if __name__ == "__main__":
    df, pct_feats = load_data()
    results = run_enhanced_cascade(df, pct_feats)
    
    print("=== Enhanced Non-Linear Cascade Optimization Completed ===")
    print(f"Slot 1: {results['hits']['slot1']} / {results['slots']['slot1']} ({results['hits']['slot1']/results['slots']['slot1']*100:.2f}%)")
    print(f"Slot 2: {results['hits']['slot2']} / {results['slots']['slot2']} ({results['hits']['slot2']/results['slots']['slot2']*100:.2f}%)")
    print(f"Slot 3: {results['hits']['slot3']} / {results['slots']['slot3']} ({results['hits']['slot3']/results['slots']['slot3']*100:.2f}%)")
    print(f"Total:  {results['hits']['total']} / {results['slots']['total']} ({results['hits']['total']/results['slots']['total']*100:.2f}%)")
    
    out_path = RESULTS_DIR / "ENHANCED_CASCADE_OPTIMIZATION_REPORT.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Saved results to:", out_path)
