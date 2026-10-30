# Gemini 关于共享实训代码、嵌套因果时钟与精确拟合清单的独立技术审计报告

- **编写主体**：反重力 Gemini 3.8 Flash High
- **研究批次**：[`research/gemini_codex_joint_ladder_emotion_design_20261021T071259Z/`](file:///d:/Documents/ChatGPT/龙头战法/research/gemini_codex_joint_ladder_emotion_design_20261021T071259Z/)
- **被审代码**：
  - 主训练脚本：[`first_board_joint_ladder_emotion_model.py`](file:///d:/Documents/ChatGPT/龙头战法/first_board_joint_ladder_emotion_model.py)
  - 情绪预测头脚本：[`first_board_emotion_forecast.py`](file:///d:/Documents/ChatGPT/龙头战法/first_board_emotion_forecast.py)
  - 精确拟合清单：[`EXACT_FIT_PLAN_BEFORE_ANY_FIT.json`](file:///d:/Documents/ChatGPT/龙头战法/research/gemini_codex_joint_ladder_emotion_design_20261021T071259Z/EXACT_FIT_PLAN_BEFORE_ANY_FIT.json)
- **协议状态**：**代码与拟合清单独立审计完全通过（IMPLEMENTATION_AND_FIT_PLAN_AUDIT_PASSED），确认当前尚未创建数值批次，实际模型拟合严格为 0 (`model_fits = 0`)，网络请求严格为 0 (`external_requests = 0`)**。

---

## 一、 核心架构与因果防泄漏机制独立穿透审计

Gemini 独立逐行审查了 [`first_board_emotion_forecast.py`](file:///d:/Documents/ChatGPT/龙头战法/first_board_emotion_forecast.py) 与 [`first_board_joint_ladder_emotion_model.py`](file:///d:/Documents/ChatGPT/龙头战法/first_board_joint_ladder_emotion_model.py)，确认其在时序因果与防泄漏机制上符合**最高级别的量化防穿越标准**：

### 1. 42 日股票校准集的严格嵌套因果律（Zero Calibration Leakage）
- **实现位置**：`first_board_emotion_forecast.py` 第 145–188 行 `fit_stock_head()` 与 `calibration_parts()`；
- **机制核验**：
  - 训练集按时间倒序切出最后 42 个交易日作为校准集 `cal`（`signal_date >= cut`），其前为内层训练集 `inner`（`label_end < cut`）；
  - 宏观头对校准集的特征预测 `cal_predictions = emotion.predict_at(cut, ...)`，调用的宏观模型 cutoff 严格设定为 `cut`；
  - 宏观训练样本在 `mature_members()` 中被强制约束为：`daily.label_end.lt(cut) & daily.signal_date.lt(cut)`；
  - **核心结论**：用于生成 42 日校准集宏观特征的 Ridge 模型，**完全未吸收任何校准期内的个股标签或宏观目标**；股票内层模型 `inner_model` 亦只在 `inner` 上拟合。两者在校准起始日截点前严格冻结，杜绝了校准期信息向内渗透。

### 2. 2025H1 验证集的单一固定宏观头（Zero Intra-Validation Learning）
- **实现位置**：`fit_stock_head()` 第 164–166 行；
- **机制核验**：
  - 验证集部署前，宏观截点被硬性绑定为验证集首日前一日：`boundary = score.signal_date.min()[:7] + "-01"`（即 `2025-01-01`）；
  - 验证集 117 天所消费的宏观预测值 `deployed_predictions`，**统一由以 2025-01-01 为截点的单一固定宏观头输出**；
  - 该宏观头仅学习 2025 年前完全成熟的数据，在 2025H1 验证期内**权重绝对静态冻结，严禁按月偷学或动态更新**。

### 3. 宏观模型缓存与不可篡改持久化校验（Cache Integrity）
- **实现位置**：`first_board_emotion_forecast.py` 第 75–117 行 `model_at()`；
- **机制核验**：
  - 命中磁盘缓存时，系统不仅比对当前截点的样本描述符（`desc == receipt`），还**强制校验 `ridge.joblib` 权重的 SHA-256 哈希值与收据中的 `model_sha256` 必须 100% 吻合**；
  - 对不足 63 天的冷启动截点，写入 `COLD_START_NO_FIT.json` 并返回 `None`（使得后续特征置为 `NaN`，绝不因缺少特征而非法删除股票行）；
  - 若截点目录存在但既无完成收据也无冷启动收据，系统直接抛出 `FileExistsError` 中断执行，**杜绝复用半成品或损坏缓存**。

---

## 二、 拟合清单（`EXACT_FIT_PLAN_BEFORE_ANY_FIT.json`）精确复核

Gemini 对清单中的全部 46 个宏观截点及各外层窗口的子模型规划进行了逐项核验：

| 核验项目 | 清单数值 / 范围 | Gemini 独立复算与确认结论 |
| :--- | :---: | :--- |
| **注册宏观截点总数** | 46 个 cut | 完整覆盖 2023-11 至 2026-08 各月向前 OOF 及嵌套校准截点 |
| **冷启动截点数（不足 63 日）** | **5 个**（2023-11 至 2024-03） | 样本天数分别为 0, 1, 22, 44, 59，均 $<63$ 天，标记为 `fit_required: false`，正确 |
| **主验证集宏观 Ridge 拟合数** | **10 次** | 从 2024-04 至 2025-01，满足 $\ge 63$ 日且 $\le 252$ 日窗口，正确 |
| **14 个月全量宏观 Ridge 拟合数（若 B 胜）** | **41 次** | 46 个注册截点扣减 5 个冷启动截点，实际需拟合 41 次，算力预算确定 |
| **验证期股票 HGB 核心拟合数** | **Arm A: 2 次，Arm B: 2 次** | 包含 1 次校准前缀子模型 + 1 次全量主模型，完全对称且最小化 |
| **Platt 标量校准拟合数** | **Arm A: 1 次，Arm B: 1 次** | 仅对 1 维 logit 进行单变量 Logistic 回归，严格受斜率正向约束 |
| **旧 BASE 与 S2 二阶段重训数** | **严格为 0** | 原受保护模型与历史账本完全冻结，0 篡改 |
| **当前实际已执行模型拟合数** | **严格为 0 (`model_fits = 0`)** | 确认所有代码与清单均处于拟合前审查状态 |

---

## 三、 两臂特征维度与入模字段核实

- **Arm A（`LADDER_SEAL_DIRECT_80`，由 Gemini 牵头）**：
  $$\text{BASE\_62} (62) + \text{LADDER\_EMOTION} (13) + \text{TIME} (2) + \text{FUND} (2) + \text{REOPEN} (1) = \mathbf{80 \text{ 维}}$$
- **Arm B（`LADDER_EMOTION_HIERARCHICAL_81`，由 Codex 牵头）**：
  $$\text{Arm A} (80) + \text{emotion\_next\_first_promotion\_estimate} (1) = \mathbf{81 \text{ 维}}$$
- 字段名与顺序已在 `first_board_joint_ladder_emotion_model.py` 第 23–26 行与 `ARMS` 字典中完成硬编码对齐。

---

## 四、 边界声明与后续实训准备

1. **当前阶段定性**：
   本文档是拟合前的最终独立技术审查，**当前尚未正式创建或签署新的数值批次目录**；
2. **严守零消耗纪律**：
   在正式签署共同冻结协议之前，Gemini 与 Codex 双侧均**严格不执行任何 `model.fit()`，不发起任何外部 HTTP 请求**；
3. **后续实训行动**：
   Gemini 确认上述代码、时钟、防泄漏约束及拟合清单完全合格，**同意以此代码作为联合实验的唯一实现**。静待 Codex 建立新的 30 分钟联合冻结批次并签署 `JOINT_FROZEN_PROTOCOL.json`，随后立即按协议分工开展训练（Gemini 跑 Arm A，Codex 跑 Arm B）！
