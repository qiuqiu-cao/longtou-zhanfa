"""
Sparse Residual Correction & Group Robust Time-Split Optimization (Arm A & Arm B)
Date: 2026-10-30
Goal: Rigorously implement and evaluate:
  - Arm A: S1 + Sparse Low-Dimensional Residuals (Direct Top-3 Optimization)
  - Arm B: S1 + Sparse Low-Dimensional Residuals (Group Robust Month Drawdown Penalty)
  - Train: 2025-07-01 to 2026-04-30 (10 months, 203 dates)
  - Out-Of-Time Test: 2026-05-01 to 2026-08-31 (4 months, 83 dates)
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

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_prep_data():
    base_results = ROOT / "research" / "first_board_top3_40_20261030" / "results"
    diag_path = base_results / "OUTCOMES_DIAGNOSTIC_ONLY.csv"
    full_path = base_results / "FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv"
    
    diag_df = pd.read_csv(diag_path)
    full_df = pd.read_csv(full_path)
    df = pd.merge(full_df, diag_df, on=["signal_date", "code"])
    
    # Neutral missing value handling
    df["reported_close_seal_log_amount_ratio"] = df["reported_close_seal_log_amount_ratio"].fillna(0)
    df["reported_first_seal_trading_minutes"] = df["reported_first_seal_trading_minutes"].fillna(240)
    df["breakout_20"] = df["breakout_20"].fillna(0)
    df["height_score"] = df["height_score"].fillna(0)
    
    # Residual features z centered at 0 in [-0.5, 0.5]
    df["z1_seal_ratio"] = df.groupby("signal_date")["reported_close_seal_log_amount_ratio"].rank(pct=True) - 0.5
    df["z2_seal_time"] = df.groupby("signal_date")["reported_first_seal_trading_minutes"].rank(pct=True, ascending=False) - 0.5
    df["z3_breakout"] = df.groupby("signal_date")["breakout_20"].rank(pct=True) - 0.5
    df["z4_height"] = df.groupby("signal_date")["height_score"].rank(pct=True) - 0.5
    
    df["month"] = df["signal_date"].str.slice(0, 7)
    return df

def eval_split(sub_df, score_col):
    u_dates = sub_df["signal_date"].unique()
    hits = 0
    known = 0
    total_slots = len(u_dates) * 3
    unknown = 0
    
    reach4_hits = 0
    reach4_known = 0
    
    non_one_hits = 0
    non_one_known = 0
    
    by_slot = {1: {"hits": 0, "known": 0}, 2: {"hits": 0, "known": 0}, 3: {"hits": 0, "known": 0}}
    
    for d in u_dates:
        d_sub = sub_df[sub_df["signal_date"] == d].copy()
        top3 = d_sub.sort_values(score_col, ascending=False).head(3)
        
        for rank_idx, (_, row) in enumerate(top3.iterrows(), start=1):
            y_val = row["promotion_next_LABEL_ONLY"]
            if pd.isna(y_val):
                unknown += 1
            else:
                known += 1
                hits += int(y_val)
                if rank_idx in by_slot:
                    by_slot[rank_idx]["known"] += 1
                    by_slot[rank_idx]["hits"] += int(y_val)
                    
            r4_val = row.get("reach4_LABEL_ONLY", np.nan)
            if not pd.isna(r4_val):
                reach4_known += 1
                reach4_hits += int(r4_val)
                
            if row.get("one_price", 0) == 0:
                if not pd.isna(y_val):
                    non_one_known += 1
                    non_one_hits += int(y_val)
                    
    prec = hits / known if known > 0 else 0
    non_one_prec = non_one_hits / non_one_known if non_one_known > 0 else 0
    
    return {
        "dates": len(u_dates),
        "total_slots": total_slots,
        "known": known,
        "unknown": unknown,
        "hits": hits,
        "precision": prec,
        "reach4_hits": reach4_hits,
        "reach4_known": reach4_known,
        "non_one_hits": non_one_hits,
        "non_one_known": non_one_known,
        "non_one_precision": non_one_prec,
        "by_slot": by_slot
    }

def run_experiment():
    df = load_and_prep_data()
    z_feats = ["z1_seal_ratio", "z2_seal_time", "z3_breakout", "z4_height"]
    
    train_mask = df["month"] <= "2026-04"
    test_mask = df["month"] >= "2026-05"
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    # Fast structures for optimization
    u_dates_train = train_df["signal_date"].unique()
    train_indices = {d: np.where(train_df["signal_date"].values == d)[0] for d in u_dates_train}
    
    X_train_z = train_df[z_feats].values
    s1_train = train_df["s1_score"].values
    y_train = train_df["promotion_next_LABEL_ONLY"].values
    
    months_train = train_df["month"].unique()
    month_dates = {m: train_df[train_df["month"] == m]["signal_date"].unique() for m in months_train}
    
    # S1 train monthly baseline
    s1_month_prec = {}
    for m, m_dates in month_dates.items():
        h = 0
        tot = 0
        for d in m_dates:
            idxs = train_indices[d]
            order = np.argsort(-s1_train[idxs])[:3]
            for idx in order:
                val = y_train[idxs[idx]]
                if not np.isnan(val):
                    tot += 1
                    h += val
        s1_month_prec[m] = h / tot if tot > 0 else 0
        
    c_l1 = 0.15
    def project_l1(theta, c):
        l1 = np.sum(np.abs(theta))
        if l1 > c:
            return theta * (c / l1)
        return theta
        
    # Arm A Objective
    def obj_arm_a(theta_raw):
        theta = project_l1(theta_raw, c_l1)
        s = s1_train + np.dot(X_train_z, theta)
        hits = 0
        for d in u_dates_train:
            idxs = train_indices[d]
            order = np.argsort(-s[idxs])[:3]
            for idx in order:
                val = y_train[idxs[idx]]
                if not np.isnan(val):
                    hits += val
        return -hits

    # Arm B Objective
    lam = 5.0
    def obj_arm_b(theta_raw):
        theta = project_l1(theta_raw, c_l1)
        s = s1_train + np.dot(X_train_z, theta)
        deltas = []
        for m, m_dates in month_dates.items():
            h = 0
            tot = 0
            for d in m_dates:
                idxs = train_indices[d]
                order = np.argsort(-s[idxs])[:3]
                for idx in order:
                    val = y_train[idxs[idx]]
                    if not np.isnan(val):
                        tot += 1
                        h += val
            prec = h / tot if tot > 0 else 0
            deltas.append(prec - s1_month_prec[m])
        avg_delta = np.mean(deltas)
        penalty = sum(max(0, -d)**2 for d in deltas)
        return -(avg_delta - lam * penalty)

    print("Optimizing Arm A (Direct Top3)...")
    res_a = differential_evolution(obj_arm_a, bounds=[(-c_l1, c_l1)]*4, maxiter=30, seed=42, popsize=20)
    theta_a = project_l1(res_a.x, c_l1)
    
    print("Optimizing Arm B (Group Robust)...")
    res_b = differential_evolution(obj_arm_b, bounds=[(-c_l1, c_l1)]*4, maxiter=30, seed=42, popsize=20)
    theta_b = project_l1(res_b.x, c_l1)
    
    # Apply to full dataframe
    df["s_arm_a"] = df["s1_score"] + np.dot(df[z_feats].values, theta_a)
    df["s_arm_b"] = df["s1_score"] + np.dot(df[z_feats].values, theta_b)
    
    # Evaluations
    models = {
        "S1_BASELINE": "s1_score",
        "ARM_A_DIRECT_TOP3": "s_arm_a",
        "ARM_B_GROUP_ROBUST": "s_arm_b"
    }
    
    report = {
        "theta_a": dict(zip(z_feats, np.round(theta_a, 4))),
        "theta_b": dict(zip(z_feats, np.round(theta_b, 4))),
        "l1_bound": c_l1,
        "robust_penalty_lambda": lam,
        "results": {}
    }
    
    for m_name, s_col in models.items():
        train_res = eval_split(df[df["month"] <= "2026-04"], s_col)
        test_res = eval_split(df[df["month"] >= "2026-05"], s_col)
        full_res = eval_split(df, s_col)
        
        report["results"][m_name] = {
            "in_sample_train": train_res,
            "out_of_time_test": test_res,
            "full_sample_286d": full_res
        }
        
    out_file = RESULTS_DIR / "REPORT.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"Report saved to {out_file}")
    return report

if __name__ == "__main__":
    rep = run_experiment()
    print("=== Summary Results ===")
    for m in ["S1_BASELINE", "ARM_A_DIRECT_TOP3", "ARM_B_GROUP_ROBUST"]:
        train_h = rep["results"][m]["in_sample_train"]["hits"]
        train_k = rep["results"][m]["in_sample_train"]["known"]
        train_p = rep["results"][m]["in_sample_train"]["precision"] * 100
        
        test_h = rep["results"][m]["out_of_time_test"]["hits"]
        test_k = rep["results"][m]["out_of_time_test"]["known"]
        test_p = rep["results"][m]["out_of_time_test"]["precision"] * 100
        
        full_h = rep["results"][m]["full_sample_286d"]["hits"]
        full_k = rep["results"][m]["full_sample_286d"]["known"]
        full_p = rep["results"][m]["full_sample_286d"]["precision"] * 100
        
        print(f"[{m}]")
        print(f"  Train In-Sample:  {train_h} / {train_k} ({train_p:.2f}%)")
        print(f"  Test Out-of-Time: {test_h} / {test_k} ({test_p:.2f}%)")
        print(f"  Full Combined:    {full_h} / {full_k} ({full_p:.2f}%)")
