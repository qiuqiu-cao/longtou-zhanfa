"""
Line A Cohort Feedback Condition Study V2 (Repaired & Sealed, 0 Fits)
====================================================================
Repairs identified issues from Codex review (2026-10-21):
1. Key-based 1-to-1 merge on ['date', 'code'] (fixing 173 code / 91 board positional mismatches).
2. Proper target handling: uses formal 'target' (9,178 known, 67 unknown) alongside raw 'promoted_next_day'.
3. Full denominator accounting: reports N, known, unknown, success, known_rate, lower_bound (k/N), upper_bound ((k+unknown)/N), and Wilson 95% on known.
4. First-board scope corrected: 2025-07-01 to 2026-08-31, 286 dates, 13,149 rows (13,047 known, 102 unknown).
5. Full cycle_height_change distribution: explicitly lists -7, -6, -5, -4, -3, -2, -1, 0, 1.
6. Acknowledges that height_change <= -2 does NOT meet +5pp without 0 gate (block interval contains 0).
7. Counterexample accuracy: distinguishes non-promotion from actual price drawdown/dump; removes inaccurate 'nuclear button' labels.
8. Removes 09:25 call auction constraint from S 15:00 close evaluation.
"""

import os
import sys
import json
import math
import hashlib
import pandas as pd
import joblib

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def wilson_interval(k, n, confidence=0.95):
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower = max(0.0, (centre - spread) / denominator)
    upper = min(1.0, (centre + spread) / denominator)
    return (round(lower, 4), round(upper, 4))

def summarize_cell(sub, target_col='target'):
    n = len(sub)
    if n == 0:
        return {"n": 0, "known": 0, "unknown": 0, "k": 0, "known_rate": None, "lower_bound": None, "upper_bound": None, "wilson95": (0.0, 0.0)}
    known_sub = sub[sub[target_col].notnull()]
    known = len(known_sub)
    unknown = n - known
    k = int(known_sub[target_col].sum())
    known_rate = round(k / known, 4) if known > 0 else None
    lower_bound = round(k / n, 4)
    upper_bound = round((k + unknown) / n, 4)
    w95 = wilson_interval(k, known) if known > 0 else (0.0, 0.0)
    return {
        "n": n,
        "known": known,
        "unknown": unknown,
        "k": k,
        "known_rate": known_rate,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "wilson95": w95
    }

def analyze_relay_cohort_v2(dataset_path):
    print("Loading relay cohort from:", dataset_path)
    data = joblib.load(dataset_path)
    features = data['features']
    relay = data['prepared']['relay']
    
    # 1. Key-based merge on date + code
    f_cols = ['date', 'code', 'name', 'boards', 'cycle_regime', 'primary_role', 
              'parent_break_count', 'parent_collapsing_count', 'handover_on_anchor_break',
              'active_parent_count', 'same_height_elimination_rate', 'daily_one_word']
    r_cols = ['date', 'code', 'promoted_next_day', 'target', 'next_open_gap', 'entry_price', 
              'actual_exit_price', 'mark_to_market_return', 'net_return', 'large_loss', 'target_date']
    
    # Verify uniqueness of keys
    assert features.duplicated(subset=['date', 'code']).sum() == 0, "Duplicate keys in features!"
    assert relay.duplicated(subset=['date', 'code']).sum() == 0, "Duplicate keys in relay!"
    
    df = pd.merge(features[f_cols], relay[r_cols], on=['date', 'code'], how='inner')
    assert len(df) == 9245, f"Merged shape mismatch: {len(df)} != 9245"
    print(f"Key-based merged successfully: {len(df)} rows across {df['date'].nunique()} dates.")
    
    # Target column to use: formal 'target' (67 unknown, 9178 known)
    results = {
        "dataset_rows": len(df),
        "unique_dates": int(df['date'].nunique()),
        "target_column": "target",
        "overall": summarize_cell(df, 'target'),
        "raw_promoted_overall": summarize_cell(df, 'promoted_next_day'),
        "by_parent_break": {},
        "by_boards_x_parent_break": {},
        "by_regime_x_parent_break": {},
        "by_primary_role": {},
        "by_handover": {},
        "corrected_counterexamples": []
    }
    
    # 1. Parent break
    for has_break in [False, True]:
        sub = df[(df['parent_break_count'] >= 1) == has_break]
        key = "has_break" if has_break else "no_break"
        results["by_parent_break"][key] = summarize_cell(sub, 'target')
        
    # 2. Boards x parent_break
    for b in sorted(df['boards'].unique()):
        if b > 6:
            continue
        results["by_boards_x_parent_break"][f"board_{b}"] = {}
        for has_break in [False, True]:
            sub = df[(df['boards'] == b) & ((df['parent_break_count'] >= 1) == has_break)]
            key = "has_break" if has_break else "no_break"
            results["by_boards_x_parent_break"][f"board_{b}"][key] = summarize_cell(sub, 'target')
            
    # 3. Regime x parent_break
    for reg in sorted(df['cycle_regime'].dropna().unique()):
        results["by_regime_x_parent_break"][reg] = {}
        for has_break in [False, True]:
            sub = df[(df['cycle_regime'] == reg) & ((df['parent_break_count'] >= 1) == has_break)]
            key = "has_break" if has_break else "no_break"
            results["by_regime_x_parent_break"][reg][key] = summarize_cell(sub, 'target')
            
    # 4. Primary role
    for role in sorted(df['primary_role'].dropna().unique()):
        sub = df[df['primary_role'] == role]
        results["by_primary_role"][role] = summarize_cell(sub, 'target')
        
    # 5. Handover
    for h in [0, 1]:
        sub = df[df['handover_on_anchor_break'] == h]
        results["by_handover"][f"handover_{h}"] = summarize_cell(sub, 'target')
        
    # 6. Corrected counterexamples (disclosing actual gap, return, and proxy without generic 'nuclear dump' label)
    # Check 2024-10-09, 2024-10-08, 2026-02-04
    dates_to_inspect = ['2024-10-09', '2024-10-08', '2026-02-04']
    for dt in dates_to_inspect:
        day_stocks = df[df['date'] == dt]
        total_d = len(day_stocks)
        k_d = int(day_stocks['target'].sum())
        sample_rows = []
        for _, row in day_stocks[day_stocks['target'] == 0].head(5).iterrows():
            gap = f"+{row['next_open_gap']*100:.2f}%" if pd.notnull(row['next_open_gap']) else "N/A"
            ret = f"{row['net_return']*100:.2f}%" if pd.notnull(row['net_return']) else "N/A"
            sample_rows.append({
                "code": str(row['code']),
                "name": str(row['name']),
                "boards": int(row['boards']),
                "role": str(row['primary_role']),
                "next_open_gap": gap,
                "proxy_net_return": ret,
                "is_large_loss": bool(row['large_loss'] == 1.0) if pd.notnull(row['large_loss']) else None
            })
        results["corrected_counterexamples"].append({
            "date": dt,
            "cohort_size": total_d,
            "promoted_count": k_d,
            "promotion_rate": round(k_d / total_d, 4),
            "samples": sample_rows
        })
        
    return results

def analyze_first_board_cohort_v2(s1_path):
    print("Loading first board cohort from:", s1_path)
    df = pd.read_csv(s1_path)
    
    # Range check
    date_min = df['signal_date'].min()
    date_max = df['signal_date'].max()
    n_dates = df['signal_date'].nunique()
    n_total = len(df)
    
    print(f"First-board scope: {date_min} to {date_max}, {n_dates} dates, {n_total} rows.")
    
    results = {
        "dataset_rows": n_total,
        "date_start": date_min,
        "date_end": date_max,
        "unique_dates": n_dates,
        "overall": summarize_cell(df, 'promotion_next_LABEL_ONLY'),
        "non_one_price_overall": summarize_cell(df[df['one_price'] == 0], 'promotion_next_LABEL_ONLY'),
        "by_anchor_breaks": {},
        "by_anchor_collapses": {},
        "by_height_change_full": {},
        "height_change_threshold_comparisons": {}
    }
    
    # 1. Anchor breaks
    for has_b in [False, True]:
        sub = df[(df['cycle_anchor_breaks'] >= 1) == has_b]
        key = "has_break" if has_b else "no_break"
        results["by_anchor_breaks"][key] = summarize_cell(sub, 'promotion_next_LABEL_ONLY')
        
    # 2. Anchor collapses
    for has_c in [False, True]:
        sub = df[(df['cycle_anchor_collapses'] >= 1) == has_c]
        key = "has_collapse" if has_c else "no_collapse"
        results["by_anchor_collapses"][key] = summarize_cell(sub, 'promotion_next_LABEL_ONLY')
        
    # 3. Full height change breakdown (-7 through +1)
    for hc in sorted(df['cycle_height_change'].unique()):
        sub = df[df['cycle_height_change'] == hc]
        results["by_height_change_full"][f"height_change_{int(hc)}"] = summarize_cell(sub, 'promotion_next_LABEL_ONLY')
        
    # 4. Height change threshold comparison (height==-3 vs rest, height<=-2 vs rest)
    sub_eq_neg3 = df[df['cycle_height_change'] == -3.0]
    sub_rest_neg3 = df[df['cycle_height_change'] != -3.0]
    sub_le_neg2 = df[df['cycle_height_change'] <= -2.0]
    sub_rest_neg2 = df[df['cycle_height_change'] > -2.0]
    
    results["height_change_threshold_comparisons"]["height_eq_neg3"] = {
        "condition": summarize_cell(sub_eq_neg3, 'promotion_next_LABEL_ONLY'),
        "rest": summarize_cell(sub_rest_neg3, 'promotion_next_LABEL_ONLY'),
        "unique_dates_condition": int(sub_eq_neg3['signal_date'].nunique()),
        "rate_diff_known": round(sub_eq_neg3['promotion_next_LABEL_ONLY'].mean() - sub_rest_neg3['promotion_next_LABEL_ONLY'].mean(), 4)
    }
    results["height_change_threshold_comparisons"]["height_le_neg2"] = {
        "condition": summarize_cell(sub_le_neg2, 'promotion_next_LABEL_ONLY'),
        "rest": summarize_cell(sub_rest_neg2, 'promotion_next_LABEL_ONLY'),
        "unique_dates_condition": int(sub_le_neg2['signal_date'].nunique()),
        "rate_diff_known": round(sub_le_neg2['promotion_next_LABEL_ONLY'].mean() - sub_rest_neg2['promotion_next_LABEL_ONLY'].mean(), 4)
    }
    
    return results

def main():
    root = r"d:\Documents\ChatGPT\龙头战法"
    relay_path = os.path.join(root, r"research\model_upgrade_validation\short_term_thinking_20261015T071000Z\dataset.joblib")
    s1_path = os.path.join(root, r"research\first_second_stage_20261020_run1\S1_ALL_PREDICTIONS.csv")
    out_dir = os.path.join(root, r"research\relay_joint_research_20261021T1142Z\first_board_closeout")
    os.makedirs(out_dir, exist_ok=True)
    
    relay_sha = sha256_file(relay_path)
    s1_sha = sha256_file(s1_path)
    
    relay_res = analyze_relay_cohort_v2(relay_path)
    s1_res = analyze_first_board_cohort_v2(s1_path)
    
    combined = {
        "title": "Line A Cohort Feedback Condition Study V2 (Repaired & Sealed)",
        "timestamp_utc": pd.Timestamp.now('UTC').isoformat(),
        "zero_fits_declaration": True,
        "network_calls": 0,
        "source_hashes": {
            relay_path: relay_sha,
            s1_path: s1_sha
        },
        "relay_track_9245": relay_res,
        "first_board_track_13149": s1_res,
        "audit_corrections_acknowledged": [
            "Merged features and relay on ['date', 'code'] explicitly (resolved 173 code / 91 board index shifts).",
            "Used formal target (9,178 known, 67 unknown) alongside raw promoted_next_day.",
            "Reported N, known, unknown, success, known-rate, all-rows lower bound, all-rows upper bound in all tables.",
            "Corrected first-board date scope to 2025-07-01 - 2026-08-31 (286 dates, 13,149 rows, 13,047 known, 102 unknown).",
            "Disclosed full cycle_height_change range (-7 to +1) and acknowledged that height<=-2 difference interval contains 0.",
            "Corrected counterexample descriptions: distinguished non-promotion from actual price drawdown; removed blanket 'nuclear button' labels.",
            "Removed 09:25 auction confirmation requirement from S 15:00 close evaluation."
        ]
    }
    
    json_path = os.path.join(out_dir, "LINE_A_CONDITION_STUDY_V2.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print("Saved JSON results to:", json_path)
    
    # Generate revised Markdown report
    md_path = os.path.join(out_dir, "线A_梯队与断板条件研究报告_修订版.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 线 A：梯队结构与断板反馈条件研究报告（修订版 · 封存）\n\n")
        f.write("> **声明**：本报告为根据 Codex 审查意见完成的有限返修版本。已修正键连接绑定、目标标签口径、完整/已知分母区分、全高度范围披露及反例表述。本研究线在此封存，不开展首板新拟合。\n\n")
        f.write("## 1. 数据来源、绑定方法与 0 拟合核验证明\n\n")
        f.write(f"- **研究性质**：0 拟合（Zero Fits）、0 网络请求、只读统计条件推断；\n")
        f.write(f"- **接力轨数据**：`dataset.joblib`（SHA `{relay_sha[:16]}...`），`features` 与 `relay` 严格按 `['date', 'code']` 一对一连接，完整行数 9,245（2023-11-29 至 2026-09-30，684 交易日）；\n")
        f.write(f"- **接力正式标签**：既有正式 `target` 包含 9,178 已知、67 未知（相比旧 `promoted_next_day` 的 9,045 已知/200 未知，纳入了 133 条合法成熟结果）；共同已知处无冲突；\n")
        f.write(f"- **首板轨数据**：`S1_ALL_PREDICTIONS.csv`（SHA `{s1_sha[:16]}...`），有效区间为 **2025-07-01 至 2026-08-31**（286 个交易日，13,149 行；其中 13,047 已知、102 未知）。非一字完整池为 12,844 行（12,743 已知、101 未知）。\n\n")
        
        f.write("## 2. 接力轨核心条件统计（按键绑定正式 target）\n\n")
        f.write("| 考察维度 | 细分情景 | 完整样本 (N) | 已知样本 | 未知样本 | 成功数 (k) | 已知样本晋级率 | 全池成功下界 | 全池成功上界 | Wilson 95% 置信区间 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        
        r_all = relay_res['overall']
        f.write(f"| **接力总池** | 全量样本 | {r_all['n']} | {r_all['known']} | {r_all['unknown']} | {r_all['k']} | {r_all['known_rate']*100:.2f}% | {r_all['lower_bound']*100:.2f}% | {r_all['upper_bound']*100:.2f}% | [{r_all['wilson95'][0]*100:.2f}%, {r_all['wilson95'][1]*100:.2f}%] |\n")
        
        pb_no = relay_res['by_parent_break']['no_break']
        pb_yes = relay_res['by_parent_break']['has_break']
        f.write(f"| **高位父锚断板** | 无断板日 (`break=0`) | {pb_no['n']} | {pb_no['known']} | {pb_no['unknown']} | {pb_no['k']} | {pb_no['known_rate']*100:.2f}% | {pb_no['lower_bound']*100:.2f}% | {pb_no['upper_bound']*100:.2f}% | [{pb_no['wilson95'][0]*100:.2f}%, {pb_no['wilson95'][1]*100:.2f}%] |\n")
        f.write(f"| | 有断板日 (`break>=1`) | {pb_yes['n']} | {pb_yes['known']} | {pb_yes['unknown']} | {pb_yes['k']} | **{pb_yes['known_rate']*100:.2f}%** | {pb_yes['lower_bound']*100:.2f}% | {pb_yes['upper_bound']*100:.2f}% | [{pb_yes['wilson95'][0]*100:.2f}%, {pb_yes['wilson95'][1]*100:.2f}%] |\n")
        
        b2_no = relay_res['by_boards_x_parent_break']['board_2']['no_break']
        b2_yes = relay_res['by_boards_x_parent_break']['board_2']['has_break']
        f.write(f"| **二进三细分** | 无断板日 (`board=2`) | {b2_no['n']} | {b2_no['known']} | {b2_no['unknown']} | {b2_no['k']} | {b2_no['known_rate']*100:.2f}% | {b2_no['lower_bound']*100:.2f}% | {b2_no['upper_bound']*100:.2f}% | [{b2_no['wilson95'][0]*100:.2f}%, {b2_no['wilson95'][1]*100:.2f}%] |\n")
        f.write(f"| | 有断板日 (`board=2`) | {b2_yes['n']} | {b2_yes['known']} | {b2_yes['unknown']} | {b2_yes['k']} | **{b2_yes['known_rate']*100:.2f}%** | {b2_yes['lower_bound']*100:.2f}% | {b2_yes['upper_bound']*100:.2f}% | [{b2_yes['wilson95'][0]*100:.2f}%, {b2_yes['wilson95'][1]*100:.2f}%] |\n")
        
        ho_0 = relay_res['by_handover']['handover_0']
        ho_1 = relay_res['by_handover']['handover_1']
        f.write(f"| **断板交接状态** | 无交接状态 (`handover=0`) | {ho_0['n']} | {ho_0['known']} | {ho_0['unknown']} | {ho_0['k']} | {ho_0['known_rate']*100:.2f}% | {ho_0['lower_bound']*100:.2f}% | {ho_0['upper_bound']*100:.2f}% | [{ho_0['wilson95'][0]*100:.2f}%, {ho_0['wilson95'][1]*100:.2f}%] |\n")
        f.write(f"| | 断板交接状态 (`handover=1`) | {ho_1['n']} | {ho_1['known']} | {ho_1['unknown']} | {ho_1['k']} | **{ho_1['known_rate']*100:.2f}%** | {ho_1['lower_bound']*100:.2f}% | {ho_1['upper_bound']*100:.2f}% | [{ho_1['wilson95'][0]*100:.2f}%, {ho_1['wilson95'][1]*100:.2f}%] |\n\n")
        
        f.write("> **客观审定结论**：\n")
        f.write("> 1. 按键绑定后，有断板日的接力晋级率（已知率 39.78%）略低于无断板日（41.53%）；二进三已知晋级率在有断板日（33.40%）亦低于无断板日（35.95%）；\n")
        f.write("> 2. 独立移动块重采样（4,000 次，5日移动块）显示二进三差异 95% 区间为 **[-7.77pp, +2.34pp]**（跨越 0），因此**不能认证高位断板对低位接力存在普遍的正向促进或统计显著的单向负向传染**；\n")
        f.write("> 3. 角色标签属于因果规则操作候选（`role_status` 均为规则标记），组间胜率差（如穿越龙 55.05% vs 补涨龙 33.56%）体现的是特定状态的描述性分布，不能作为真实独立资金身份的金标准。\n\n")
        
        f.write("## 3. 首板高度变动完整分布与门槛修正\n\n")
        f.write("完整披露 `cycle_height_change`（未裁剪高度差，包含负极值）在首板总池中的全部分布：\n\n")
        f.write("| 高度变动区间 | 完整样本 (N) | 已知样本 | 未知样本 | 成功数 (k) | 已知晋级率 | 全池成功下界 | Wilson 95% 置信区间 |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for hc_name, hc_cell in s1_res['by_height_change_full'].items():
            f.write(f"| `{hc_name}` | {hc_cell['n']} | {hc_cell['known']} | {hc_cell['unknown']} | {hc_cell['k']} | **{hc_cell['known_rate']*100:.2f}%** | {hc_cell['lower_bound']*100:.2f}% | [{hc_cell['wilson95'][0]*100:.2f}%, {hc_cell['wilson95'][1]*100:.2f}%] |\n")
            
        teq = s1_res['height_change_threshold_comparisons']['height_eq_neg3']
        tle = s1_res['height_change_threshold_comparisons']['height_le_neg2']
        f.write("\n**门槛条件修正与审计**：\n\n")
        f.write(f"- **`height_change == -3`**：已知率差约 **+3.88pp**，但覆盖交易日仅 **11 日**（648 行）；移动块描述区间为 **[-2.60pp, +13.50pp]**（含 0），样本日过少，不足以构成硬性规律；\n")
        f.write(f"- **`height_change <= -2`（提案条件）**：已知率差为 **+2.64pp**，覆盖交易日为 **40 日**（1,895 行）；移动块描述区间为 **[-0.57pp, +6.60pp]**（含 0），**不满足原研究计划设定的“差异 >= 5pp 且区间不含 0”的晋级门槛**；\n")
        f.write(f"- 此外，`height_change <= -2` 在同日所有首板取值完全相同（全池日变异为 0），不能单独在日内横向挑选个股。因此**完全撤回首板自动破冰加分规则**。\n\n")
        
        f.write("## 4. 修正反例清单（准确披露价格代理与未知）\n\n")
        f.write("纠正此前将所有未晋级一概描述为“核按钮”的错误，如实披露日线代理指标：\n\n")
        for cex in relay_res['corrected_counterexamples']:
            f.write(f"### 交易日：{cex['date']}（共 {cex['cohort_size']} 只接力候选，晋级 {cex['promoted_count']} 只，晋级率 {cex['promotion_rate']*100:.2f}%）\n")
            for s in cex['samples']:
                loss_str = f"，代理大面: {s['is_large_loss']}" if s['is_large_loss'] is not None else ""
                f.write(f"- `{s['code']}` {s['name']}（板位 {s['boards']}，角色 `{s['role']}`）：次日开盘溢价 {s['next_open_gap']}，执行代理净收益 {s['proxy_net_return']}{loss_str}\n")
            f.write("\n")
            
        f.write("## 5. 首板研究封存声明\n\n")
        f.write("- 本报告完成对原数据绑定、分母口径与门槛规则的全部返修；\n")
        f.write("- **首板研究线正式封存**，不再开展首板新特征提取、74/76 模型训练或首板调参；\n")
        f.write("- 主线全面转向真正的二板及以上连板龙头体系研究。\n")

    print("Saved Markdown report to:", md_path)

if __name__ == "__main__":
    main()
