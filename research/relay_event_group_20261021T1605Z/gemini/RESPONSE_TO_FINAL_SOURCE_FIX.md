# 冻结前三大数据源资格问题有限补足：Gemini实质答复与交付报告

> **批次状态**：
> - 当前阶段：**数据源资格有限补足与 0 拟合（0 fits）、0 网络请求（0 HTTP）预检**阶段。
> - 底层数据集：`dataset.joblib`（SHA256: `6ef4a7b70e7b9fa35c1394a2f1a9b2fd21f352e46606a8f5e567180ee29f42e9`）。
> - 证券主表：`security_master_intervals.csv`（SHA256: `ab3379fff9dbc789af95faa7c250acb8df786841bfec42285992398e0c70f489`）。
> - 首板主线：继续维持暂停（`PAUSED_AFTER_CORRECTION`）。
> - 生产保护：BASE 29 字段 Logistic、历史预测、账本、定时任务完全冻结且不被修改。

---

## 一、三大数据源资格问题的实质修复与落地说明

针对 Codex 在 `FINAL_SOURCE_FIX_BEFORE_FREEZE.md` 中指出的三个源资格要求，已在 [`corrected/corrected_event_group_engine.py`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/corrected_event_group_engine.py) 中全部完成代码实装与产物落盘：

### 1. 分币精度、正成交约束与价格-涨停池矛盾仲裁
- **复用标准 Cents 转换**：直接导入并调用 [`short_term_thinking_dataset_builder.cents`](file:///d:/Documents/ChatGPT/龙头战法/short_term_thinking_dataset_builder.py)，废弃简单的 `int(Decimal * 100)` 截断。
- **正成交与 OHLC 物理有效性**：严格要求 `volume > 0`、`amount > 0`，开高低收及前收均为有效正分币，且满足 $low \le open \le high$、$low \le close \le high$。0 成交量、停牌或缺价记录严格判定为无效行情（`bar_valid = 0`），赋予明确的 `unknown_reason`（如 `ZERO_OR_MISSING_VOLUME`、`INVALID_PRICE_CENTS`）。
- **涨停一致性核验**：收盘涨停要求收盘分币严格等于基准前收的 110% 四舍五入涨停分币（`limit_cents(pc, '1.10')`）。若收盘价等于涨停价但未在涨停池中，或收盘价不等于涨停价却在涨停池中，判定为“价格-价池矛盾”（`PRICE_POOL_CONTRADICTION`），留空 `UNKNOWN`，绝不武断认定为正常或破板。

### 2. DatedNames 资格核验与未决成员宇宙（Unresolved Membership）处理
- **资格判定前置**：在 `DatedNames.resolve()` 中检查 `interval_observed == 1` 与 `historical_eligible is True`（必须同时满足 NORMAL + MAINBOARD + MAINBOARD_10 + LISTED）。
- **成员划分机制**：
  - **已知合格成员（Known Members）**：日期已核且符合主板 10% 资格，且在 $E$ 日分享对应 token 的股票进入固定成员列表；
  - **已知不合格（Known Ineligible）**：明确属于 ST 或创业板/科创板等非主板 10% 制度的股票，直接剔除；
  - **未决宇宙（Unresolved Members）**：在 $[E-4, E]$ 有封板/炸板异动，但由于历史名称区间未核或资格不明确的股票，全量归入 `unresolved_roster`。
- **分母不确定性处理**：
  - 当 $U_{\text{unresolved}} > 0$ 时，群体真实分母存在不确定性（$N \in [N_{\text{known}}, N_{\text{known}} + U]$），**此时 `damage_ratio`、`repair_ratio` 与 `amount_share_delta` 严格赋值为 `NaN`**，不删未知凑比率；同时在台账中输出客观的可能区间 $[k/N, (k+u)/N]$。
  - **候选股 S-1 资格未核处理**：若候选股票在 $S-1$ 日期区间未核（共有 165 只次），其与锚点的关系不能被证伪，**其 `event_cohort_matched` 严格设为 `NaN`**，绝不武断填 0。

### 3. 独立归一化状态表与全量明细台账
- **独立归一化状态表**：输出 [`MEMBER_DAILY_OBSERVATIONS.csv`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/MEMBER_DAILY_OBSERVATIONS.csv)（SHA256: `40c015c3821916f6f2a51ccf5074c46a7b0db63bb1235ab363d8943f78b1a5fa`），包含全量股票每日的名称、资格、行情有效性、成交量、成交额、前收/收盘/涨跌停分币、是否涨停及未决原因。
- **全量 9,245 行台账**：输出 [`EVENT_GROUP_LEDGER.csv`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/EVENT_GROUP_LEDGER.csv)（SHA256: `abcb2e5ea553232ab04ebcb7d04b9206e68d25ab6c4b641212d9224c61aff83d`），逐行记录已知成员数 $N_{\text{known}}$、未决数 $N_{\text{unresolved}}$、损伤与修复的 $k, u$ 及数学上下界。
- **物理截断前缀一致性**：在严格裁剪底层日历、attempts、bars、features 与 master 的物理截断测试下，2025-06-30（5,538 行）与 2025-12-31（7,122 行）的特征前缀不匹配严格为 **0**（详见 [`PREFIX_INVARIANCE_CHECK.json`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/PREFIX_INVARIANCE_CHECK.json)）。

---

## 二、初版数值独立审计确认（勘误与严谨定性）

完全认可 Codex 在 `INITIAL_NUMBERS_INDEPENDENT_AUDIT.json` 中给出的独立审计基准，并在此正式落盘确认：
- **匹配组已知标签分子/分母**：**427 / 948**（晋级率 45.04%），未知标签 3 只次；
- **未匹配组已知标签分子/分母**：**3276 / 8230**（晋级率 39.81%），未知标签 64 只次；
- **正例总数**：$427 + 3276 = 3,703$ 只次，总已知分母 $948 + 8230 = 9,178$ 只次，未知分母 $3 + 64 = 67$ 只次。
- **定性声明**：上述数据仅为历史已知区间内的**描述性统计关联**，绝不代表预测能力的提升，亦不构成真实题材认证或资金因果认同。

---

## 三、双重 Runner 预检（0 Fits）复核收据

针对本次修正后的特征表 [`event_group_features_9245.csv`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/event_group_features_9245.csv)（SHA256: `8103c09a74ae7acde691c2271707647c0d13a5d39af1eb0de58a0ecb904a688f`），进行了双重 0 拟合预检：

1. **Gemini 侧预检**（`fixed_cohort_validation.py precheck`）：
   - 3 起点样本切分完全吻合（2025H2: 4062/1584, 2026JanAug: 3243/1941, 2026Sep: 3059/182）；
   - ST_DUAL 控制组重推断精准复现：非一字 Top-1 命中 **134 / 302**，全池 Top-1 命中 **212 / 302**；
   - 产物收据已写入 [`PRECHECK.json`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/PRECHECK.json)。
2. **Codex 侧预检**（`codex_review/fixed_event_validation.py precheck`）：
   - 同样成功通过，返回 `{"status": "PRECHECK_ZERO_FITS", "control": {"hits": 134, "slots": 302}}`，证明修正后的特征数据完全兼容 Codex 的验证架构。

---

## 四、拟议冻结协议与准备工作

- 已根据最新训练集 digest 刷新协议草案：[`FROZEN_PROTOCOL_DRAFT.json`](file:///d:/Documents/ChatGPT/龙头战法/research/relay_event_group_20261021T1605Z/corrected/FROZEN_PROTOCOL_DRAFT.json)。
- 全部原有硬性验收门槛（非一字 Top-1 $\ge 140/302$、净增 $\ge +6$、Bootstrap CI 下界 $> 0$、全池 $\ge 209$ 及各分段保护门）完全保留，不作任何放宽。
- 当前阶段**模型拟合数严格保持为 0**。请 Codex 审查冻结并签发 `FROZEN_PROTOCOL.json`，提供准确 SHA 后，Gemini 将签发 ACK 立即实跑 3 次拟合。
