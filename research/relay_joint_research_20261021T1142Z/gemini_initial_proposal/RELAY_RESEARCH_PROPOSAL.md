# 连板龙头接力与抗分歧响应研究提案（讨论稿 · 0 拟合）

- **提案主体**：实际反重力 Gemini
- **协作审查**：Codex
- **状态**：方案研讨阶段，未共同冻结，0 拟合、0 网络调用

---

### 一、 研究对象、完整分母与分层架构

1. **研究对象**：
   - 信号日收盘（$S$ 日 15:00）全市场已有 2 板及以上（$\ge 2$ 板）的主板连板候选；
   - 绝不与首板混为一谈，彻底解耦首板 BASE62 / 52 / 117 指标。
2. **完整数据分母（9,245 行，684 个交易日）**：
   - 数据源：`research/model_upgrade_validation/short_term_thinking_20261015T071000Z/dataset.joblib`；
   - 正式目标标签：`target`，已知 9,178 行，未知 67 行；
   - **板位分层结构（基数基准）**：
     - **2 进 3 梯队**：5,528 行（占 59.80%），已知 5,491，成功 1,833，已知晋级率 **33.38%**；
     - **3 进 4 梯队**：1,892 行（占 20.46%），已知 1,876，成功 899，已知晋级率 **47.92%**；
     - **4 板及以上高标梯队**：1,825 行（占 19.74%），已知 1,811，成功 971，已知晋级率 **53.62%**。
3. **输出要求**：
   - 全池统一打分，输出每日全池 Top-1、Top-2；
   - 评估必须分栏输出：全池指标、2 进 3 子集、3 进 4 子集、4+ 高标子集，以及非一字专用通道。

---

### 二、 核心假设 A：梯队负反馈路径与分歧传导

#### 1. 为什么旧的静态断板计数（`parent_break_count`）必然失败？
- 静态计数只回答了“今天有没有老龙断板”，但忽略了**情绪出清的时间滞后性**；
- 当天老龙断板往往引发市场极端恐慌与负反馈传导，中低位大面积跟跌（如 2024-10-08、10-09）；
- 真正的“老周期出清、新周期确立”，需要经历 **1~2 个交易日的中位淘汰与恐慌释放**。

#### 2. 可计算的新增路径定义与公式

$$
\text{CascadeRisk}(t) = \mathbb{I}(\Delta t_{\text{break}} \le 1) \times \left(1 + \text{FailedMidTiers}(t)\right)
$$

$$
\text{CycleResetSignal}(t) = \mathbb{I}(\Delta t_{\text{break}} \in [2, 4]) \times \mathbb{I}(\text{MidTierVoid}(t) == 1) \times \text{RegimeEbb}(t)
$$

- **变量说明（均在 $S$ 日 15:00 收盘可得）**：
  - $\Delta t_{\text{break}}$：最近一次高位空间锚（$\ge 4$板）断板至今经过的官方交易日数；
  - $\text{FailedMidTiers}(t)$：当日 3 板及 4 板中炸板或跌停的个股数量；
  - $\text{MidTierVoid}(t)$：当日 3 板与 4 板是否出现断层（存活数为 0）；
  - $\text{RegimeEbb}(t)$：市场是否处于退潮向破冰过渡阶段。
- **与旧失败的本质区别**：
  - 不是简单的单日静态计数字段，而是**断板衰减时间差 $\times$ 中位梯队断层度**的动态事件交互。

---

### 三、 核心假设 B：题材群体受损下的个股抗分歧响应

#### 1. 为什么静态概念/名称平均涨幅（旧 72/75）无效？
- 平均涨幅容易被个别个股拉高，且“报喜不报忧”，无法识别同组后排的炸板风险（如今日盛洋科技炸板拖累同线）；
- 必须构建**同组受损排雷机制**与**逆势个股的抗分歧度量**。

#### 2. 可计算的群体受损与个体韧性公式

1. **同组受损率（Group Damage Ratio, $D_g$）**：

$$
D_g(t) = \frac{\sum_{j \in \text{Group}} \left(\mathbb{I}(\text{kind}_j == \text{'failed\_board'}) + \mathbb{I}(\text{kind}_j == \text{'down\_limit'})\right)}{N_{\text{Group}}(t)}
$$

2. **个股抗分歧响应（Individual Resilience Score, $R_i$）**：

$$
R_i(t) = \begin{cases}
\text{TurnoverScore}_i \times \text{SealSpeedRank}_i, & \text{若 } D_g(t) == 0 \\
\text{TurnoverHealth}_i \times \left(1 - \text{OpenCount}_i\right) \times \mathbb{I}(\text{Close}_i == \text{Limit}), & \text{若 } D_g(t) > 0
\end{cases}
$$

- **逻辑说明**：
  - 若同题材/同叙事内无人掉队（$D_g = 0$），常规按首封速度与换手打分；
  - 若同组出现炸板或跌停（$D_g > 0$），后排开板多次、尾盘勉强封板的个股直接触发大幅惩罚（排雷盛洋类）；
  - 只有在同组遭遇负反馈时，自身**能够烂板弱转强回封、换手处于 10%~20% 承接区间且未破分时均线**的个股（独立穿越者），才能获得逆势承接加分。

---

### 四、 至多两臂比较结构（讨论稿 · 待共同冻结）

| 对照臂 | 字段集合与特征构成 | 核心假设 | 边界约束 |
|---|---|---|---|
| **Arm A (`RELAY_CASCADE_PATH`)** | 保存连板 BASE 特征 + 梯队断层与出清滞后路径字段（$\Delta t_{\text{break}}$、$\text{MidTierVoid}$、板位归一化排名） | 解决高标断板与中位绞肉时的时序出清识别 | 仅使用 $S$ 日收盘可得数据；不回填次日信息 |
| **Arm B (`RELAY_RESILIENCE_DUAL`)** | Arm A + 同组受损率 $D_g$ 与个股抗分歧韧性 $R_i$ | 在梯队路径基础上，解决主线分歧时核心与后排的分流排雷 | 缺乏真实题材时仅使用严格名称代理，明确标注代理局限 |

- **冻结约束**：
  - 双方达成一致并经 Codex 审查签署前，**绝对不开训、不跑新 fit**；
  - 若经数据资格审计发现 $D_g$ 历史样本覆盖率不足，则仅保留 Arm A，绝不强凑第二臂。

---

### 五、 固定连板基准与验收标准（拒绝套用首板指标）

1. **唯一合法基准**：
   - 采用 [delivery/model_validation_results_VALIDATED.json](file:///d:/Documents/ChatGPT/龙头战法/research/model_upgrade_validation/short_term_thinking_20261015T071000Z/delivery/model_validation_results_VALIDATED.json) 中保存的连板 `BASE`；
   - 2025H2 非一字 F1 基线为 **9.36%**，Recall 为 **5.21%**；9 月非一字 F1 基线为 **24.0%**，Top-1 命中为 **7/13**。
2. **硬性验收门槛（必须全部过门，且过门前不改门）**：
   - **门槛 1（全日历覆盖）**：完整 684 个交易日（或指定评估窗）决策日覆盖率 $\ge 95\%$，未知日计入分母；
   - **门槛 2（非一字超额）**：非一字通道 F1 必须超越保存连板 BASE 至少 **+3.0pp**，且 Precision 不下降超过 3pp；
   - **门槛 3（Top-1 排序改善）**：在完整检验日历上，每日 Top-1 命中次日收盘涨停天数严格高于连板 BASE；
   - **门槛 4（大面防御）**：所选候选在次日日线执行代理下的大面率（亏损 $>7\%$）低于全池同身位平均水平。

---

### 六、 交付与执行状态

- 本提案与配套 [RESPONSE_TO_CODEX.md](file:///d:/Documents/ChatGPT/龙头战法/research/relay_joint_research_20261021T1142Z/gemini/RESPONSE_TO_CODEX.md) 已同步落盘；
- 保持 0 拟合，等待 Codex 审查评议后，再确定是否进行零拟合特征资格审计。
