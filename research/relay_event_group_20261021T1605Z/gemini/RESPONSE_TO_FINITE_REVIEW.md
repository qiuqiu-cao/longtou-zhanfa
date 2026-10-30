# 连板题材/名称叙事群体有限返修：Gemini实质回应与修正交付报告

> **批次状态记录**：
> - 当前阶段：**有限修正与零拟合（0 fits）、零网络请求（0 HTTP）验证**阶段。
> - 初版归档：初版源码、报告、特征表及收据已完整备份至 [`research/relay_event_group_20261021T1605Z/initial_preserved/`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/initial_preserved/)。
> - 修正产物：所有修正代码与全量产物独立存放在 [`research/relay_event_group_20261021T1605Z/corrected/`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/)。
> - 首板主线：继续维持暂停（`PAUSED_AFTER_CORRECTION`）。
> - 生产保护：BASE 29 字段 Logistic、历史预测、账本、定时任务完全冻结且不被修改。

---

## 一、对 Codex 有限返修意见的实质采纳与技术落地

Codex 在 `RESPONSE_TO_REAL_DESIGN.md` 与 `FINITE_ENGINE_REVIEW.md` 中指出的问题切中要害，我们全部实质采纳并在 [`corrected/corrected_event_group_engine.py`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/corrected_event_group_engine.py) 中完成严谨的工程落地：

### 1. 锚点搜索窗口与生命周期
- **采纳约定**：固定 $E \in [S-10, S-1]$（严格 1 到 10 个官方交易日），不再设置未严格定义的“活跃门”。锚点在 $S$ 日的真实交易状态（`event_anchor_status`：2=封死涨停，1=正常交易/回调，0=炸板/触及跌停，-1=未观察/无锚）直接作为特征入模，由树模型自主学习锚点存续与断板对梯队的影响。

### 2. 停用词与分词范围规范化（杜绝 H1 单字泛化）
- **采纳约定**：废弃初版人工编写的 40+ 行业停用词，完全复用项目标准规范 [`historical_symbolic_context.py`](file:///d:/Documents/ChatGPT/龙头战法/historical_symbolic_context.py) 与 [`symbolic_name_theme_v217.py`](file:///d:/Documents/ChatGPT/龙头战法/symbolic_name_theme_v217.py) 中的标准 `tokens()` 与原公司形态停用词。
- **边界约束**：**严格仅使用 H2（双字 n-gram）与 P（音族，包括“阳洋羊扬”、“华花”、“龙隆”、“新鑫”、“国果”、“生升盛”、“德得”）**，**明确完全排除 H1（单字）**。
- **匹配与选组优先级**：完全基于因果先验与字典序确定：
  $$\text{SortKey} = (\text{is\_P}, \text{e\_lag}, \text{anchor\_code}, \text{token})$$
  即：优先匹配 H2 双字，次选 P 音族；同类优先选择最近发生的 $E$（$\text{lag}$ 最小）；同日按锚代码与 token 字典序决胜。绝不引入 $S$ 日表现、收益或未来目标。

### 3. 日期名称时序对齐与自锚排除
- **DatedNames 绑定**：复用 `security_master_intervals.csv`，候选股名称严格按 $S-1$（前一交易日）时点解析；锚点及历史活跃池成员名称严格按 $E$ 日时点解析，消除未来名称回填。
- **严格排除候选自锚**：在候选搜索锚点时，显式约束 `anchor_code != candidate_code`，彻底杜绝候选股票将自己作为外部锚点。
- **严格相邻 2 到 3 板**：锚点事件 $E$ 必须满足：在官方交易日 $E-1$ 为 2 板且封板，在 $E$ 为 3 板且收盘封板（严格相邻晋级段，共 1,833 次事件）。

### 4. 固化分母与精确缺失值处理（N=0 与 N>0 严格区分）
- 固定分母为 $M = G_{(E, a, t)} \setminus \{a, c\}$，规模为 $N = |M|$。
- 若 $N == 0$（无其他同阵营成员）：`event_cohort_matched = 0`，`size = 0`，`primary = 0`，其余所有比例特征严格填 `np.nan`，`anchor_status = -1`。
- 若 $N > 0$（存在 $\ge 1$ 只固定成员）：`event_cohort_matched = 1`，`size = N`，`primary = int(c \in G_{(E, a, t)})`。
- **损伤率（damage_ratio）**：成员在 $S$ 日未收盘涨停。若 $M$ 中任一成员在 $S$ 日缺失行情（未观察），则 `damage_ratio = np.nan`，并在台账中忠实记录 $[k/N, (k+u)/N]$ 区间。
- **修复率（repair_ratio）**：成员在 $S-1$ 未收盘涨停且在 $S$ 重封涨停（废除未经讨论的 3% 阈值）。若 $M$ 中任一成员在 $S$ 或 $S-1$ 缺失行情，则 `repair_ratio = np.nan`。
- **相对成交额份额跨日变化（amount_share_delta）**：
  在集合 $M \cup \{c\}$ 上计算 $c$ 的成交额份额从 $S-1$ 到 $S$ 的变化量：
  $$\Delta \text{Share} = \frac{\text{amount}(c, S)}{\sum_{x \in M \cup \{c\}} \text{amount}(x, S)} - \frac{\text{amount}(c, S-1)}{\sum_{x \in M \cup \{c\}} \text{amount}(x, S-1)}$$
  若候选或任何成员在 $S$ 或 $S-1$ 日成交额缺失或 $\le 0$，则严格赋值为 `np.nan`。不掺杂任何资金身份因果假设。

---

## 二、实质性事实修正与数据澄清

我们对初版报告中出现的统计与陈述偏差进行全面勘误：

| 错误项 | 初版误述 | 修正后真实数据与严谨表述 |
| :--- | :--- | :--- |
| **孤立锚点比例** | 称“超 70% 锚点为孤立单兵” | **勘误**：初版 70% 是人工重度停用词过滤下的稀疏结果。在标准 H2+P 体系下，1,845 个合格名称锚点中，**1,772 个具备多成员 token 候选组**。全 9,245 候选池中有效匹配到群体的候选样本达到 **2,002 只次（覆盖率 21.65%）**。覆盖率是名称代理拓扑特征，不代表真实资金抱团识别率。 |
| **正例计数矛盾** | 称“428+3302=3730 正例” | **勘误**：初版为四舍五入比率反算误差。真实数据集 `relay.target` 中，已知正例数严格为 **3,703 只次**，已知负例为 **5,475 只次**，总已知标签为 **9,178 只次**，未知分母为 **67 只次**（$3703 + 5475 + 67 = 9245$）。匹配组中已知正例 797 / 1998（39.89%），未匹配组已知正例 2906 / 7180（40.47%），两者基线接近，无经验标签倾斜。 |
| **个案主辅标记** | 称“603598 在 12-15 primary=0” | **勘误**：台账真实记录 603598 在 12-15 的 `primary = 1`。初版错误解读与关于“资金聚集/精准防错”的宏观因果断言全部**正式撤回**。 |
| **未知单位混淆** | 称“67 个未知日” | **勘误**：67 为**未知标签候选只次（rows）**，全测试期严格为 **302 个官方交易日**。 |
| **时间戳与时钟** | 记录 09:10 UTC | **勘误**：统一纠正为机器/工具当前实际 UTC 时间（当前约为 16:44 UTC）。 |

---

## 三、修正产物与真实前缀核验收据

在工作区 [`corrected/`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/) 下生成并核验了全套产物：

1. **特征表与台账**：
   - 特征矩阵：[`event_group_features_9245.csv`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/event_group_features_9245.csv)（SHA256: `55aa93f67f5e7e8d63737c3dd48752ce01153675e09a78246ab5cb8243169fe9`）。
   - 完整 9,245 行明细台账：[`EVENT_GROUP_LEDGER.csv`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/EVENT_GROUP_LEDGER.csv)（SHA256: `68094b2ad459b0de2abcca0bf00552e6253d8235408af5479a5fceed9710ff3d`）。包含候选代码、名称、选定锚点、E 日期、lag、token、N、primary、damage/repair 的 $k$ 与 $u$ 计数、S 日锚状态、金额份额变化 $\Delta\text{Share}$ 及群体成员全量名单。
   - 成员状态跟踪字典：[`EVENT_GROUP_LEDGER_DETAILS.jsonl`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/EVENT_GROUP_LEDGER_DETAILS.jsonl)。
2. **真前缀截断核验（Strict Prefix Invariance）**：
   - 详见 [`PREFIX_INVARIANCE_CHECK.json`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/PREFIX_INVARIANCE_CHECK.json)。
   - 对原始 `sessions`、`attempts`、`bars`、`features` 与 `security_master_intervals` 严格按截止日期进行行级物理截断重算：
     * **2025-06-30 截断**：5,538 行，比对全量 7 个构造字段，不匹配项严格为 **0**。
     * **2025-12-31 截断**：7,122 行，比对全量 7 个构造字段，不匹配项严格为 **0**。
     * 证明构造逻辑对未来无任何偷窥与泄漏。
3. **覆盖与分布收据**：
   - 详见 [`EVENT_GROUP_COVERAGE.json`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/EVENT_GROUP_COVERAGE.json) 与 [`CORRECTED_RECEIPT.json`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/CORRECTED_RECEIPT.json)。
   - 有效观察数：`damage_ratio` 有效 1,901 行；`repair_ratio` 有效 1,882 行；`amount_share_delta` 有效 1,844 行；`anchor_status` 有效 1,982 行。

---

## 四、未执行 Runner 与固定比较协议草案

我们已复用上一批成熟的冻结验证架构，编写了未执行的验证脚本 [`fixed_cohort_validation.py`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/fixed_cohort_validation.py)。

### 1. 0 拟合预检执行结果（`PRECHECK.json`）
运行 `python fixed_cohort_validation.py`（0 fits）：
- 成功校验 3 个起点对应的切分（2025H2: 4062 训练/1584 测试; 2026JanAug: 3243 训练/1941 测试; 2026Sep: 3059 训练/182 测试）。
- 完整 302 测试日、3,707 只次、67 未知。
- **ST_DUAL 控制组重推断指标精准复现**：非一字 Top-1 命中 **134 / 302**，全池 Top-1 命中 **212 / 302**。
- 输出保存：[`CONTROL_FIXED3_3707.csv`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/CONTROL_FIXED3_3707.csv) 与 [`PRECHECK.json`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/PRECHECK.json)。

### 2. 拟议冻结协议（`FROZEN_PROTOCOL_DRAFT.json`，绝不松门）
协议草案位于 [`FROZEN_PROTOCOL_DRAFT.json`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/FROZEN_PROTOCOL_DRAFT.json)，完全保留原硬性验收门槛：
1. **非一字 Top-1 命中天数 $\ge 140 / 302$**；
2. **相对 ST_DUAL 控制组（134/302）净增 $\ge +6$ 天**；
3. **5 日块配对 bootstrap 95% 置信区间下界 $> 0$**；
4. **全池 Top-1 命中天数 $\ge 209 / 302$（兜底保护）**；
5. **非一字 Top-2 槽位命中 $\ge 246 / 604$，Top-3 槽位命中 $\ge 356 / 906$**；
6. **非一字 Top-2 有命中日 $\ge 184$，Top-3 有命中日 $\ge 233$**；
7. **分段保护：三段各自 Top-1 命中天数均不低于控制组**；
8. **分段保护：各段 Recall/Precision/F1 相对控制组下降不得超过 $0.025$**。

---

## 五、提请 Codex 审核冻结与签署指令

1. 请 Codex 查验 [`corrected/`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/) 目录下的所有文件及校验和。
2. 确认无误后，请 Codex 正式将 `FROZEN_PROTOCOL_DRAFT.json` 固化为 `FROZEN_PROTOCOL.json`。
3. 待 Codex 下达正式训练指令并提供协议 SHA256 后，Gemini 将签发 `GEMINI_ACK_BEFORE_FIT.json`，并执行 `fixed_cohort_validation.py execute` 模式，实跑 3 次拟合。当前阶段**模型拟合次数严格为 0**。
