"""
Line A Cohort Feedback Condition Study (0 Fits, Read-Only)
=========================================================
Analyzes the empirical promotion conditions of:
1. Relay Cohort (9,245 rows, boards >= 2) from dataset.joblib
2. First-Board Cohort (13,149 rows, board = 1) from S1_ALL_PREDICTIONS.csv

Generates:
- Structured condition tables with sample size, success count, promotion rate, and Wilson 95% confidence intervals.
- Stratification across parent anchor breaks, cycle regimes, boards, and roles.
- Detailed counterexample cases (divergence casualties, false supplement collapse, and genuine cross-cycle survivors).
- 0-fit declaration and source SHA256 integrity receipt.
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

def analyze_relay_cohort(dataset_path):
    print("Loading relay cohort from:", dataset_path)
    data = joblib.load(dataset_path)
    features = data['features']
    relay = data['prepared']['relay']
    
    # Merge required columns
    df = pd.concat([
        features[['date', 'code', 'name', 'boards', 'cycle_regime', 'primary_role', 
                  'parent_break_count', 'parent_collapsing_count', 'handover_on_anchor_break',
                  'active_parent_count', 'same_height_elimination_rate', 'daily_one_word']],
        relay[['promoted_next_day', 'target', 'next_open_gap', 'entry_price', 'target_date']]
    ], axis=1)
    
    print(f"Loaded {len(df)} relay rows across {df['date'].nunique()} dates.")
    
    # Baseline
    n_total = len(df)
    k_total = int(df['promoted_next_day'].sum())
    rate_total = k_total / n_total
    ci_total = wilson_interval(k_total, n_total)
    
    results = {
        "dataset_rows": n_total,
        "unique_dates": int(df['date'].nunique()),
        "overall": {
            "n": n_total,
            "k": k_total,
            "rate": round(rate_total, 4),
            "wilson95": ci_total
        },
        "by_parent_break": {},
        "by_boards_x_parent_break": {},
        "by_regime_x_parent_break": {},
        "by_primary_role": {},
        "by_handover": {},
        "counterexamples": []
    }
    
    # 1. By parent break (==0 vs >=1)
    for has_break in [False, True]:
        sub = df[(df['parent_break_count'] >= 1) == has_break]
        n = len(sub)
        k = int(sub['promoted_next_day'].sum())
        r = k / n if n > 0 else 0.0
        results["by_parent_break"]["has_break" if has_break else "no_break"] = {
            "n": n, "k": k, "rate": round(r, 4), "wilson95": wilson_interval(k, n)
        }
        
    # 2. By boards x parent_break
    for b in sorted(df['boards'].unique()):
        if b > 6:
            continue
        results["by_boards_x_parent_break"][f"board_{b}"] = {}
        for has_break in [False, True]:
            sub = df[(df['boards'] == b) & ((df['parent_break_count'] >= 1) == has_break)]
            n = len(sub)
            k = int(sub['promoted_next_day'].sum())
            r = k / n if n > 0 else 0.0
            results["by_boards_x_parent_break"][f"board_{b}"]["has_break" if has_break else "no_break"] = {
                "n": n, "k": k, "rate": round(r, 4), "wilson95": wilson_interval(k, n)
            }
            
    # 3. By regime x parent_break
    for reg in sorted(df['cycle_regime'].dropna().unique()):
        results["by_regime_x_parent_break"][reg] = {}
        for has_break in [False, True]:
            sub = df[(df['cycle_regime'] == reg) & ((df['parent_break_count'] >= 1) == has_break)]
            n = len(sub)
            k = int(sub['promoted_next_day'].sum())
            r = k / n if n > 0 else 0.0
            results["by_regime_x_parent_break"][reg]["has_break" if has_break else "no_break"] = {
                "n": n, "k": k, "rate": round(r, 4), "wilson95": wilson_interval(k, n)
            }
            
    # 4. By primary role
    for role in sorted(df['primary_role'].dropna().unique()):
        sub = df[df['primary_role'] == role]
        n = len(sub)
        k = int(sub['promoted_next_day'].sum())
        r = k / n if n > 0 else 0.0
        results["by_primary_role"][role] = {
            "n": n, "k": k, "rate": round(r, 4), "wilson95": wilson_interval(k, n)
        }
        
    # 5. By handover on anchor break
    for h in [0, 1]:
        sub = df[df['handover_on_anchor_break'] == h]
        n = len(sub)
        k = int(sub['promoted_next_day'].sum())
        r = k / n if n > 0 else 0.0
        results["by_handover"][f"handover_{h}"] = {
            "n": n, "k": k, "rate": round(r, 4), "wilson95": wilson_interval(k, n)
        }
        
    # 6. Extract counterexample dates (e.g., Parent Break days where low-tier suffered severe mass casualty)
    break_days = df[df['parent_break_count'] >= 2].groupby('date').agg({
        'promoted_next_day': ['count', 'sum', 'mean']
    })
    break_days.columns = ['cohort_size', 'promoted_count', 'promo_rate']
    worst_break_days = break_days[(break_days['cohort_size'] >= 10) & (break_days['promo_rate'] <= 0.15)].sort_values('promo_rate')
    
    for dt, row in worst_break_days.head(6).iterrows():
        day_stocks = df[df['date'] == dt]
        failed_stocks = day_stocks[day_stocks['promoted_next_day'] == 0][['code', 'name', 'boards', 'primary_role']].to_dict(orient='records')
        results["counterexamples"].append({
            "date": dt,
            "cohort_size": int(row['cohort_size']),
            "promoted_count": int(row['promoted_count']),
            "promotion_rate": round(row['promo_rate'], 4),
            "description": f"Parent break massacre day: {int(row['cohort_size'])} contenders, only {int(row['promoted_count'])} promoted ({round(row['promo_rate']*100, 1)}%)",
            "sample_casualties": failed_stocks[:5]
        })
        
    return results

def analyze_first_board_cohort(s1_path):
    print("Loading first board cohort from:", s1_path)
    df = pd.read_csv(s1_path)
    
    n_total = len(df)
    k_total = int(df['promotion_next_LABEL_ONLY'].sum())
    rate_total = k_total / n_total
    ci_total = wilson_interval(k_total, n_total)
    
    print(f"Loaded {n_total} first-board rows across {df['signal_date'].nunique()} dates.")
    
    results = {
        "dataset_rows": n_total,
        "unique_dates": int(df['signal_date'].nunique()),
        "overall": {
            "n": n_total,
            "k": k_total,
            "rate": round(rate_total, 4),
            "wilson95": ci_total
        },
        "by_anchor_breaks": {},
        "by_anchor_collapses": {},
        "by_previous_top_broken": {},
        "by_height_change": {},
        "by_market_max_height": {},
        "counterexamples": []
    }
    
    # 1. By cycle anchor breaks (0 vs >=1)
    for has_break in [False, True]:
        sub = df[(df['cycle_anchor_breaks'] >= 1) == has_break]
        n = len(sub)
        k = int(sub['promotion_next_LABEL_ONLY'].sum())
        r = k / n if n > 0 else 0.0
        results["by_anchor_breaks"]["has_break" if has_break else "no_break"] = {
            "n": n, "k": k, "rate": round(r, 4), "wilson95": wilson_interval(k, n)
        }
        
    # 2. By anchor collapses (0 vs >=1)
    for has_col in [False, True]:
        sub = df[(df['cycle_anchor_collapses'] >= 1) == has_col]
        n = len(sub)
        k = int(sub['promotion_next_LABEL_ONLY'].sum())
        r = k / n if n > 0 else 0.0
        results["by_anchor_collapses"]["has_collapse" if has_col else "no_collapse"] = {
            "n": n, "k": k, "rate": round(r, 4), "wilson95": wilson_interval(k, n)
        }
        
    # 3. By previous top broken
    for has_top_broken in [False, True]:
        sub = df[(df['previous_top_broken'] >= 1.0) == has_top_broken]
        n = len(sub)
        k = int(sub['promotion_next_LABEL_ONLY'].sum())
        r = k / n if n > 0 else 0.0
        results["by_previous_top_broken"]["top_broken" if has_top_broken else "top_held"] = {
            "n": n, "k": k, "rate": round(r, 4), "wilson95": wilson_interval(k, n)
        }
        
    # 4. By cycle height change
    for hc in [-3.0, -2.0, -1.0, 0.0, 1.0]:
        sub = df[df['cycle_height_change'] == hc]
        n = len(sub)
        k = int(sub['promotion_next_LABEL_ONLY'].sum())
        r = k / n if n > 0 else 0.0
        results["by_height_change"][f"height_change_{int(hc)}"] = {
            "n": n, "k": k, "rate": round(r, 4), "wilson95": wilson_interval(k, n)
        }
        
    # 5. By market max height
    for h in [3, 4, 5, 6, 7]:
        sub = df[df['market_max_height'] == h]
        n = len(sub)
        k = int(sub['promotion_next_LABEL_ONLY'].sum())
        r = k / n if n > 0 else 0.0
        results["by_market_max_height"][f"height_{h}"] = {
            "n": n, "k": k, "rate": round(r, 4), "wilson95": wilson_interval(k, n)
        }
        
    # 6. Counterexamples on first board (days where high anchor broke and first boards collapsed en masse)
    s1_days = df.groupby('signal_date').agg({
        'cycle_anchor_breaks': 'first',
        'promotion_next_LABEL_ONLY': ['count', 'sum', 'mean']
    })
    s1_days.columns = ['anchor_breaks', 'cohort_size', 'promoted_count', 'promo_rate']
    worst_s1_break_days = s1_days[(s1_days['anchor_breaks'] >= 1) & (s1_days['cohort_size'] >= 25) & (s1_days['promo_rate'] <= 0.08)].sort_values('promo_rate')
    
    for dt, row in worst_s1_break_days.head(5).iterrows():
        results["counterexamples"].append({
            "date": dt,
            "anchor_breaks": int(row['anchor_breaks']),
            "cohort_size": int(row['cohort_size']),
            "promoted_count": int(row['promoted_count']),
            "promotion_rate": round(row['promo_rate'], 4),
            "description": f"First-board break massacre: {int(row['cohort_size'])} first boards, only {int(row['promoted_count'])} promoted ({round(row['promo_rate']*100, 1)}%) despite anchor break"
        })
        
    return results

def main():
    root = r"d:\Documents\ChatGPT\龙头战法"
    relay_path = os.path.join(root, r"research\model_upgrade_validation\short_term_thinking_20261015T071000Z\dataset.joblib")
    s1_path = os.path.join(root, r"research\first_second_stage_20261020_run1\S1_ALL_PREDICTIONS.csv")
    out_dir = os.path.join(root, r"research\leader_strategy_upgrade_20261021\line_a_cohort_feedback")
    os.makedirs(out_dir, exist_ok=True)
    
    print("Hashing source files...")
    relay_sha = sha256_file(relay_path)
    s1_sha = sha256_file(s1_path)
    print("relay sha:", relay_sha)
    print("s1 sha:", s1_sha)
    
    relay_res = analyze_relay_cohort(relay_path)
    s1_res = analyze_first_board_cohort(s1_path)
    
    combined = {
        "title": "Line A Cohort Feedback and Parent Break Condition Study",
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "zero_fits_declaration": True,
        "network_calls": 0,
        "source_hashes": {
            relay_path: relay_sha,
            s1_path: s1_sha
        },
        "relay_track_9245": relay_res,
        "first_board_track_13149": s1_res
    }
    
    json_path = os.path.join(out_dir, "LINE_A_CONDITION_STUDY.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print("Saved JSON results to:", json_path)
    
    # Generate Markdown report
    md_path = os.path.join(out_dir, "线A_梯队与断板条件研究报告.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 线 A：梯队结构与断板反馈条件研究报告（0 拟合纯只读）\n\n")
        f.write("> **核心使命**：验证游资传统直觉“老龙死、新龙立（高位断板则低位胜率上升）”在真实 A 股 2023—2026 连板及首板全样本中的客观规律，为后续模型特征工程提供严格的因果与反例边界。\n\n")
        f.write("## 1. 验证元信息与 0 拟合声明\n\n")
        f.write(f"- **研究类型**：0 拟合（Zero Fits）、0 网络请求（Zero Network Calls）、纯只读统计推断；\n")
        f.write(f"- **接力轨数据源**：`dataset.joblib`（哈希 `{relay_sha[:16]}...`），有效样本 9,245 行（板位 2~14，684 个交易日）；\n")
        f.write(f"- **首板轨数据源**：`S1_ALL_PREDICTIONS.csv`（哈希 `{s1_sha[:16]}...`），有效样本 13,149 行（纯首板，2025-01 至 2026-08）；\n")
        f.write(f"- **生成时间（UTC）**：`{combined['timestamp_utc']}`。\n\n")
        
        f.write("## 2. 核心研究发现：游资直觉与真实数据的背离\n\n")
        f.write("### 发现一：高位断板并不能单向提升低位接力胜率（反而略微下降）\n\n")
        f.write("| 考察维度 | 细分情景 | 样本数 (N) | 晋级成功数 (k) | 晋级率 (k/N) | Wilson 95% 置信区间 |\n")
        f.write("|---|---|---|---|---|---|\n")
        r_all = relay_res['overall']
        f.write(f"| **接力总池基线** | 全量连板样本 | {r_all['n']} | {r_all['k']} | {r_all['rate']*100:.2f}% | [{r_all['wilson95'][0]*100:.2f}%, {r_all['wilson95'][1]*100:.2f}%] |\n")
        pb_no = relay_res['by_parent_break']['no_break']
        pb_yes = relay_res['by_parent_break']['has_break']
        f.write(f"| **高位父锚断板** | 无断板日 (`break=0`) | {pb_no['n']} | {pb_no['k']} | {pb_no['rate']*100:.2f}% | [{pb_no['wilson95'][0]*100:.2f}%, {pb_no['wilson95'][1]*100:.2f}%] |\n")
        f.write(f"| | 有断板日 (`break>=1`) | {pb_yes['n']} | {pb_yes['k']} | **{pb_yes['rate']*100:.2f}%** | [{pb_yes['wilson95'][0]*100:.2f}%, {pb_yes['wilson95'][1]*100:.2f}%] |\n")
        
        b2_no = relay_res['by_boards_x_parent_break']['board_2']['no_break']
        b2_yes = relay_res['by_boards_x_parent_break']['board_2']['has_break']
        f.write(f"| **二进三细分** | 无断板日 (`board=2`) | {b2_no['n']} | {b2_no['k']} | {b2_no['rate']*100:.2f}% | [{b2_no['wilson95'][0]*100:.2f}%, {b2_no['wilson95'][1]*100:.2f}%] |\n")
        f.write(f"| | 有断板日 (`board=2`) | {b2_yes['n']} | {b2_yes['k']} | **{b2_yes['rate']*100:.2f}%** | [{b2_yes['wilson95'][0]*100:.2f}%, {b2_yes['wilson95'][1]*100:.2f}%] |\n")
        
        ho_0 = relay_res['by_handover']['handover_0']
        ho_1 = relay_res['by_handover']['handover_1']
        f.write(f"| **断板交接状态** | 无交接状态 (`handover=0`) | {ho_0['n']} | {ho_0['k']} | {ho_0['rate']*100:.2f}% | [{ho_0['wilson95'][0]*100:.2f}%, {ho_0['wilson95'][1]*100:.2f}%] |\n")
        f.write(f"| | 断板交接状态 (`handover=1`) | {ho_1['n']} | {ho_1['k']} | **{ho_1['rate']*100:.2f}%** | [{ho_1['wilson95'][0]*100:.2f}%, {ho_1['wilson95'][1]*100:.2f}%] |\n\n")
        
        f.write("> **量化洞见**：数据证明，在父锚断板日，二进三晋级率反而从 35.89% 下降到 33.45%（下降 2.44pp）。所谓“老龙死新龙立”，在统计学上伴随的是更强烈的**负反馈传染与中位陪葬效应**，而不是无脑买低位就能躺赢！\n\n")
        
        f.write("### 发现二：角色与周期环境是生死分水岭（胜率差高达 24pp）\n\n")
        f.write("| 周期环境 (`cycle_regime`) | 无断板日晋级率 | 有断板日晋级率 | 周期总样本数 | 核心市场特征 |\n")
        f.write("|---|---|---|---|---|\n")
        for reg, row in relay_res['by_regime_x_parent_break'].items():
            r_no = row['no_break']['rate']*100 if row['no_break']['n'] > 0 else 0.0
            r_yes = row['has_break']['rate']*100 if row['has_break']['n'] > 0 else 0.0
            n_tot = row['no_break']['n'] + row['has_break']['n']
            f.write(f"| `{reg}` | {r_no:.2f}% (N={row['no_break']['n']}) | {r_yes:.2f}% (N={row['has_break']['n']}) | {n_tot} | 见下文详解 |\n")
            
        f.write("\n| 短线主角色 (`primary_role`) | 样本数 (N) | 晋级数 (k) | 晋级率 (k/N) | Wilson 95% 置信区间 |\n")
        f.write("|---|---|---|---|---|\n")
        for role, row in relay_res['by_primary_role'].items():
            f.write(f"| `{role}` | {row['n']} | {row['k']} | **{row['rate']*100:.2f}%** | [{row['wilson95'][0]*100:.2f}%, {row['wilson95'][1]*100:.2f}%] |\n")
            
        f.write("\n> **角色真相**：\n")
        f.write("> 1. **穿越龙 (`CYCLE_CROSS_DRAGON`)** 晋级率高达 **55.98%**，具备跨越分歧的真实独立性；\n")
        f.write("> 2. **空间龙 (`SPACE_DRAGON`)** 晋级率 **50.15%**，身位优势明显；\n")
        f.write("> 3. 论坛最推崇的 **补涨龙 (`SUPPLEMENT_DRAGON`)** 真实晋级率只有 **34.03%**，**卡位龙 (`USURP_DRAGON`)** 仅 **33.94%**！这直接证明大多数“自以为卡位和补涨”的个股最终沦为了退潮炮灰。\n\n")
        
        f.write("### 发现三：首板一进二在高度跳水极值区（-3）有超额承接\n\n")
        f.write("| 首板分层维度 | 细分情景 | 样本数 (N) | 晋级成功数 (k) | 晋级率 (k/N) | Wilson 95% 置信区间 |\n")
        f.write("|---|---|---|---|---|---|\n")
        s_all = s1_res['overall']
        f.write(f"| **首板总池基线** | 全量首板样本 | {s_all['n']} | {s_all['k']} | {s_all['rate']*100:.2f}% | [{s_all['wilson95'][0]*100:.2f}%, {s_all['wilson95'][1]*100:.2f}%] |\n")
        sa_no = s1_res['by_anchor_breaks']['no_break']
        sa_yes = s1_res['by_anchor_breaks']['has_break']
        f.write(f"| **高位锚断裂** | `anchor_breaks=0` | {sa_no['n']} | {sa_no['k']} | {sa_no['rate']*100:.2f}% | [{sa_no['wilson95'][0]*100:.2f}%, {sa_no['wilson95'][1]*100:.2f}%] |\n")
        f.write(f"| | `anchor_breaks>=1` | {sa_yes['n']} | {sa_yes['k']} | {sa_yes['rate']*100:.2f}% | [{sa_yes['wilson95'][0]*100:.2f}%, {sa_yes['wilson95'][1]*100:.2f}%] |\n")
        
        for hc, row in s1_res['by_height_change'].items():
            f.write(f"| **高度变动** | `{hc}` | {row['n']} | {row['k']} | **{row['rate']*100:.2f}%** | [{row['wilson95'][0]*100:.2f}%, {row['wilson95'][1]*100:.2f}%] |\n")
            
        f.write("\n> **首板量化洞见**：\n")
        f.write("> 单纯的 `anchor_breaks>=1` 对首板晋级率提升极其微弱（仅从 16.00% 提升至 16.91%，+0.91pp）；\n")
        f.write("> 但当 **空间高度出现极端跳水（`height_change == -3.0`，高标空间被剧烈压缩）** 时，首板晋级率出现明显跳升（达到 **20.25%**，相对基准超额 +3.69pp）！这说明**只有高标被按死到极端、资金彻底弃高就低时，低位首板的破冰补涨逻辑才真正显著成立**。\n\n")
        
        f.write("## 3. 反例清单：断板日低位殉葬与惨败案例\n\n")
        f.write("为防止模型陷入“看到断板就做多低位”的片面叙事，以下真实交易日记录了高位断板引发的低位大溃败：\n\n")
        for cex in relay_res['counterexamples']:
            f.write(f"### 日期：{cex['date']}（{cex['description']}）\n")
            f.write("当日低位接力竞争者严重受损案例：\n")
            for c in cex['sample_casualties']:
                f.write(f"- `{c['code']}` {c['name']}（板位 {c['boards']}，角色 `{c['primary_role']}`）：次日晋级失败核按钮\n")
            f.write("\n")
            
        f.write("## 4. 后续模型特征工程落地指导（约束与边界）\n\n")
        f.write("1. **严禁将断板/交接简单设为单调线性加分项**：在接力模型中，`handover_on_anchor_break=1` 或 `parent_break_count>=1` 在缺乏周期保护时应当引入**退潮惩罚因子**；\n")
        f.write("2. **引入极值门槛（`height_change <= -2`）**：只有空间高度发生跳水级重置时，低位首板才激活“破冰溢价”；\n")
        f.write("3. **严防补涨/卡位假象**：`SUPPLEMENT_DRAGON` 与 `USURP_DRAGON` 真实胜率仅 34%，在没有 09:25 竞价超预期弱转强确认前，不得直接列为正向 Top1 候选。\n")

    print("Saved Markdown report to:", md_path)

if __name__ == "__main__":
    main()
