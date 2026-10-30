"""
Line B Seal Quality and Turnover Binning Qualification Audit V2 (Repaired & Sealed, 0 Fits)
==========================================================================================
Repairs identified issues from Codex review (2026-10-21):
1. Key-based 1-to-1 merge on ['date', 'code'].
2. Re-names '真实次日净收益' to 'S+1假设触板入场、S+2收盘退出的日线执行代理', disclosing observable denominators:
   - Non-one-word: 3,018 observed for large_loss, 2,956 for net_return.
   - One-word: 1,016 observed for large_loss, 991 for net_return.
   - Discloses that conditional subset of failed promotions with proxy execution is not the entire failed population.
3. Corrects definition of 'verified_turnover_boards':
   - It is an observation node count in the streak, not current board height (e.g. 527 of 4964 are board=3, 205 are board=4).
   - On first boards (S1), it can only be 0, 1, or unknown; verified_turnover_boards == 2 cannot distinguish first boards.
   - Completely removes the invalid gate 'verified_turnover_boards == 2 && EBB_HANDOVER' from first-board design.
4. Corrects turnover monotonicity statement:
   - Dips from <3% (16.12%) to 3-5% (14.64%), then increases; thus not strictly monotonic across full range.
   - Notes that turnover is already present in BASE 62 features (turnover, turnover_rank, turnover_ratio_5).
   - Reports 1 unknown turnover row.
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

def main():
    root = r"d:\Documents\ChatGPT\龙头战法"
    relay_path = os.path.join(root, r"research\model_upgrade_validation\short_term_thinking_20261015T071000Z\dataset.joblib")
    s1_path = os.path.join(root, r"research\first_second_stage_20261020_run1\S1_ALL_PREDICTIONS.csv")
    out_dir = os.path.join(root, r"research\relay_joint_research_20261021T1142Z\first_board_closeout")
    os.makedirs(out_dir, exist_ok=True)
    
    relay_sha = sha256_file(relay_path)
    s1_sha = sha256_file(s1_path)
    
    # 1. Audit Relay Dataset (9245) with key-based merge
    data = joblib.load(relay_path)
    feat = data['features']
    relay = data['prepared']['relay']
    
    f_cols = ['date', 'code', 'boards', 'daily_one_word', 'verified_turnover_boards', 
              'observed_nonword_boards', 'first_open_after_one_word', 'relative_volume']
    r_cols = ['date', 'code', 'target', 'promoted_next_day', 'large_loss', 'net_return', 'mark_to_market_return', 'next_open_gap']
    
    assert set(zip(feat.date, feat.code)) == set(zip(relay.date, relay.code))
    df_relay = pd.merge(feat[f_cols], relay[r_cols], on=['date', 'code'], how='inner', validate='one_to_one')
    assert len(df_relay) == 9245
    
    # Audit fields
    fields_to_audit = ['daily_one_word', 'verified_turnover_boards', 'observed_nonword_boards', 'first_open_after_one_word']
    relay_audit = {}
    for f in fields_to_audit:
        nn = int(df_relay[f].notnull().sum())
        nulls = int(df_relay[f].isnull().sum())
        relay_audit[f] = {
            "non_null_count": nn,
            "null_count": nulls,
            "non_null_rate": round(nn / len(df_relay), 4),
            "qualification_status": "NOT_ASSESSED_BY_THIS_NULL_COUNT_AUDIT",
            "coverage_status": "OBSERVED_COUNTS_ONLY"
        }
        
    # Analysis of daily_one_word with explicit observable denominators
    one_word_stats = {}
    for val, name in [(0.0, "non_one_word"), (1.0, "one_word")]:
        sub = df_relay[df_relay['daily_one_word'] == val]
        n_full = len(sub)
        known_target = sub[sub['target'].notnull()]
        k_target = int(known_target['target'].sum())
        
        # Large loss observable denominator
        sub_loss = sub[sub['large_loss'].notnull()]
        n_loss_obs = len(sub_loss)
        k_loss = int(sub_loss['large_loss'].sum()) if n_loss_obs > 0 else 0
        loss_rate_obs = round(k_loss / n_loss_obs, 4) if n_loss_obs > 0 else None
        
        # Net return observable denominator
        sub_ret = sub[sub['net_return'].notnull()]
        n_ret_obs = len(sub_ret)
        mean_ret_obs = round(float(sub_ret['net_return'].mean()), 4) if n_ret_obs > 0 else None
        
        # Gap observable denominator
        sub_gap = sub[sub['next_open_gap'].notnull()]
        n_gap_obs = len(sub_gap)
        mean_gap_obs = round(float(sub_gap['next_open_gap'].mean()), 4) if n_gap_obs > 0 else None
        
        # Failure subset with loss proxy
        sub_fail = sub[(sub['target'] == 0.0) & (sub['large_loss'].notnull())]
        n_fail_loss_obs = len(sub_fail)
        k_fail_loss = int(sub_fail['large_loss'].sum()) if n_fail_loss_obs > 0 else 0
        fail_loss_rate_obs = round(k_fail_loss / n_fail_loss_obs, 4) if n_fail_loss_obs > 0 else None
        
        one_word_stats[name] = {
            "total_rows": n_full,
            "known_target_rows": len(known_target),
            "unknown_target_rows": n_full - len(known_target),
            "promoted_count": k_target,
            "promotion_known_rate": round(k_target / len(known_target), 4),
            "all_rows_lower_bound": round(k_target / n_full, 4),
            "large_loss_observable_denominator": n_loss_obs,
            "large_loss_observable_rate": loss_rate_obs,
            "failed_with_loss_observable_denominator": n_fail_loss_obs,
            "failed_loss_rate_in_observable_subset": fail_loss_rate_obs,
            "net_return_observable_denominator": n_ret_obs,
            "mean_execution_proxy_net_return": mean_ret_obs,
            "next_open_gap_observable_denominator": n_gap_obs,
            "mean_next_open_gap": mean_gap_obs
        }
        
    # Analysis of verified_turnover_boards height composition
    turnover_board_composition = {}
    for tb in [0.0, 1.0, 2.0, 3.0, 4.0]:
        sub = df_relay[df_relay['verified_turnover_boards'] == tb]
        board_dist = sub['boards'].value_counts().to_dict()
        turnover_board_composition[f"turnover_nodes_{int(tb)}"] = {
            "total_rows": len(sub),
            "board_height_distribution": {str(k): int(v) for k, v in sorted(board_dist.items())}
        }
        
    # 2. Audit First-Board Dataset (13149)
    df_s1 = pd.read_csv(s1_path)
    
    # Non-one-price turnover bins
    nop_s1 = df_s1[df_s1['one_price'] == 0].copy()
    bins = [-1, 3, 5, 10, 15, 20, 100]
    labels = ['<3%', '3-5%', '5-10%', '10-15%', '15-20%', '>20%']
    nop_s1['to_bin'] = pd.cut(nop_s1['turnover'], bins=bins, labels=labels)
    
    s1_turnover_bins = {}
    for lbl in labels:
        sub = nop_s1[nop_s1['to_bin'] == lbl]
        n_full = len(sub)
        known_sub = sub[sub['promotion_next_LABEL_ONLY'].notnull()]
        known = len(known_sub)
        unknown = n_full - known
        k = int(known_sub['promotion_next_LABEL_ONLY'].sum())
        s1_turnover_bins[lbl] = {
            "n": n_full,
            "known": known,
            "unknown": unknown,
            "k": k,
            "known_rate": round(k / known, 4) if known > 0 else None,
            "lower_bound": round(k / n_full, 4),
            "wilson95": wilson_interval(k, known)
        }
        
    receipt = {
        "title": "Line B Seal Quality and Turnover Binning Audit Receipt V2 (Repaired & Sealed)",
        "timestamp_utc": pd.Timestamp.now('UTC').isoformat(),
        "zero_fits_declaration": True,
        "network_calls": 0,
        "source_hashes": {
            relay_path: relay_sha,
            s1_path: s1_sha
        },
        "relay_field_audit": relay_audit,
        "relay_one_word_analysis_with_observable_denominators": one_word_stats,
        "verified_turnover_boards_composition": turnover_board_composition,
        "first_board_turnover_binnings_corrected": s1_turnover_bins,
        "audit_corrections_acknowledged": [
            "Merged on ['date', 'code'] explicitly.",
            "Re-named return metric to 'S+1假设触板入场、S+2收盘退出的日线执行代理'.",
            "Disclosed observable denominators for large_loss (3,018 non-one-word, 1,016 one-word) and net_return (2,956 and 991).",
            "Disclosed that conditional subset of failed promotions with proxy execution is not the entire failed population.",
            "Corrected verified_turnover_boards definition: it counts historical turnover nodes in streak, not current board height (527 of 4964 are board=3, 205 are board=4).",
            "Deleted the invalid gate 'verified_turnover_boards == 2 && EBB_HANDOVER' from first-board design.",
            "Withdrew strictly monotonic turnover claim across full range (dips from <3% 16.12% to 3-5% 14.64%).",
            "Acknowledged that turnover is already part of BASE 62 features."
        ]
    }
    
    json_path = os.path.join(out_dir, "LINE_B_AUDIT_RECEIPT_V2.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)
    print("Saved JSON receipt to:", json_path)
    
    # Generate revised Markdown report
    md_path = os.path.join(out_dir, "线B_封板质量与换手分桶资格审计报告_修订版.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 线 B：封板质量与换手分桶资格审计报告（修订版 · 封存）\n\n")
        f.write("> **声明**：本报告为根据 Codex 审查意见完成的有限返修版本。已更正执行代理术语、披露可观察分母、修正连续换手节点计数定义、纠正换手非严格单调表述，并删除首板无效闸门。本研究线在此封存。\n\n")
        
        f.write("## 1. 字段口径与执行代理定义审计\n\n")
        f.write("本次字段非空率只表示覆盖，未据任意空值阈值重新认证来源或截止时点。已有授权的历史重建资格与来源等级仍独立保留；本次收据不授予新训练资格。分桶按右闭区间(-1,3]、(3,5]、(5,10]、(10,15]、(15,20]、(20,100]计算；另保留1条换手率未知记录，不从完整池删除。\n\n")
        f.write("1. **收益与损失代理口径纠正**：\n")
        f.write("   - 既有字段 `net_return` 与 `large_loss` 源自日线执行假设：**信号日 S 收盘发出，次日 S+1 盘中触板且非全天一字时假设以涨停价成交，S+2 收盘退出，扣除 0.3% 摩擦费用**；\n")
        f.write("   - 它没有真实逐笔成交或排队凭证，**不得称为“真实次日净收益”**，而是明确命名为**“S+1假设触板入场、S+2收盘退出的日线执行代理”**；\n")
        f.write("   - 该代理指标仅在**满足上述入场假设的子集**上可观察，不能冒充全量样本的真实回报。\n\n")
        
        f.write("## 2. 真实一字与非一字形态及可观察执行代理全景\n\n")
        f.write("| 板位形态 | 完整行数 | 目标已知行 | 目标未知 | 已知晋级率 | 全池成功下界 | 代理大面可观察样本数 | 全可观察样本大面率 | 失败子集中有大面标签数 | 失败子集大面率 | 代理净收益可观察样本数 | 平均代理净收益 | 开盘溢价样本数 | 平均开盘溢价 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        
        for name, row in one_word_stats.items():
            name_zh = "一字板 (`daily_one_word=1`)" if name == "one_word" else "非一字换手板 (`daily_one_word=0`)"
            fail_rate_str = f"{row['failed_loss_rate_in_observable_subset']*100:.2f}%" if row['failed_loss_rate_in_observable_subset'] is not None else "N/A"
            mean_ret_str = f"{row['mean_execution_proxy_net_return']*100:.2f}%" if row['mean_execution_proxy_net_return'] is not None else "N/A"
            f.write(f"| **{name_zh}** | {row['total_rows']} | {row['known_target_rows']} | {row['unknown_target_rows']} | **{row['promotion_known_rate']*100:.2f}%** | {row['all_rows_lower_bound']*100:.2f}% | {row['large_loss_observable_denominator']} | {row['large_loss_observable_rate']*100:.2f}% | {row['failed_with_loss_observable_denominator']} | {fail_rate_str} | {row['net_return_observable_denominator']} | {mean_ret_str} | {row['next_open_gap_observable_denominator']} | +{row['mean_next_open_gap']*100:.2f}% |\n")
            
        f.write("\n> **量化审计说明**：\n")
        f.write("> 1. 非一字板的 `large_loss` 仅在 3,018 行有标签（占 7,073 行的 42.67%）；一字板仅在 1,016 行有标签（占 2,126 行的 47.79%）；\n")
        f.write("> 2. 所谓“一旦开板失败，大面率 43.90%”此前混淆了分母，43.90% 是全部 1,016 条可观察样本的大面率；而在晋级失败且有代理标签的 333 条一字样本中，大面率为 74.47%；在非一字失败有代理标签的 1,198 条样本中，大面率为 66.11%。两者均只是代理入场子集，不能当作全池所有股票的真实概率。\n\n")
        
        f.write("## 3. `verified_turnover_boards` 字段定义纠偏与首板闸门删除\n\n")
        f.write("1. **字段本质**：`verified_turnover_boards` 是本轮连板序列中经核验的换手观测节点累计计数，**不是当前个股的板位高度**，也不要求节点连续；\n")
        f.write("2. **数据分布**：在值为 2 的 4,964 行中：\n")
        f.write("   - 二板个股：4,133 行（83.26%）；\n")
        f.write("   - 三板个股：527 行（10.62%）；\n")
        f.write("   - 四板个股：205 行（4.13%）；\n")
        f.write("   - 五至十四板：99 行（1.99%）；\n")
        f.write("3. **首板闸门无效裁决**：\n")
        f.write("   - 在首板截面上，连板序列长度仅有 1 天，该计数正确值只能为 0、1 或未知；\n")
        f.write("   - 因此此前 Arm B 提议的 `verified_turnover_boards == 2 && EBB_HANDOVER` 在首板截面上根本无法触发，若填入次日才生成的 2 则是严重未来函数；\n")
        f.write("   - **裁决：从首板设计中彻底删除此闸门，二板及以上另在连板主线中按当时可见信息独立评价**。\n\n")
        
        f.write("## 4. 首板换手率分桶修正（非全区间严格单调递增）\n\n")
        f.write("在首板（S1 排除一字板）12,844 只完整样本（已知 12,743，未知 101）中：\n\n")
        f.write("| 换手率区间 (`turnover`) | 完整样本 (N) | 已知样本 | 未知样本 | 晋级成功数 (k) | 已知晋级率 | 全池成功下界 | Wilson 95% 置信区间 |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for b_name, b_info in s1_turnover_bins.items():
            f.write(f"| **{b_name}** | {b_info['n']} | {b_info['known']} | {b_info['unknown']} | {b_info['k']} | **{b_info['known_rate']*100:.2f}%** | {b_info['lower_bound']*100:.2f}% | [{b_info['wilson95'][0]*100:.2f}%, {b_info['wilson95'][1]*100:.2f}%] |\n")
            
        f.write("\n> **实证修正**：\n")
        f.write("> 1. 晋级率从 `<3%` 的 16.12% 下降到 `3-5%` 的 14.64%，随后在 `5-10%` (15.05%)、`10-15%` (16.07%)、`15-20%` (17.52%)、`>20%` (19.38%) 递增；因此**全区间并非严格单调递增**；\n")
        f.write("> 2. 换手率 `turnover`、`turnover_rank`、`turnover_ratio_5` 均已在原 BASE 62 字段中，换手分箱属于**表示形式变换**，未提供全新增量承接信息；\n")
        f.write("> 3. 撤回凭此条件表直接在首板加分的提议，首板 74/76 方案不冻结、不训练。\n\n")
        
        f.write("## 5. 首板审计封存结论\n\n")
        f.write("- 线 B 资格审计返修完成，正式确认首板不再开展 74/76 或新特征拟合；\n")
        f.write("- 首板研究在此封存，全面进入连板龙头接力与抗分歧响应主线。\n")

    print("Saved Markdown audit to:", md_path)

if __name__ == "__main__":
    main()
