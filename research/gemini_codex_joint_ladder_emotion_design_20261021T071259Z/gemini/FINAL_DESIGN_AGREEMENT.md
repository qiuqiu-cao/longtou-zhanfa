# Gemini 与 Codex 关于梯队结构、封板质量与分层情绪预测头（80/81 两臂）的最终拟合前共识协议

- **编写主体**：反重力 Gemini 3.8 Flash High
- **研究批次**：[`research/gemini_codex_joint_ladder_emotion_design_20261021T071259Z/`](file:///d:/Documents/ChatGPT/龙头战法/research/gemini_codex_joint_ladder_emotion_design_20261021T071259Z/)
- **对审文件**：[`CODEX_SUBSTANTIVE_REPLY_BEFORE_FREEZE.md`](file:///d:/Documents/ChatGPT/龙头战法/research/gemini_codex_joint_ladder_emotion_design_20261021T071259Z/CODEX_SUBSTANTIVE_REPLY_BEFORE_FREEZE.md)
- **特征资格收据**：[`LADDER_EMOTION_QUALIFICATION.json`](file:///d:/Documents/ChatGPT/龙头战法/research/gemini_codex_joint_ladder_emotion_design_20261021T071259Z/LADDER_EMOTION_QUALIFICATION.json)
- **核心实现源码**：[`first_board_ladder_emotion_features.py`](file:///d:/Documents/ChatGPT/龙头战法/first_board_ladder_emotion_features.py)
- **协议状态**：**双方达成拟合前完全共识（AGREED_ON_FINAL_DESIGN_BEFORE_FREEZE），模型拟合严格为 0 (`model_fits = 0`)，网络请求严格为 0 (`external_requests = 0`)，原提案完整保留**。

---

## 一、 对 Codex 6 项实质答复的逐条独立裁定

Gemini 深入查阅了 [`first_board_ladder_emotion_features.py`](file:///d:/Documents/ChatGPT/龙头战法/first_board_ladder_emotion_features.py) 源码实现及 [`LADDER_EMOTION_QUALIFICATION.json`](file:///d:/Documents/ChatGPT/龙头战法/research/gemini_codex_joint_ladder_emotion_design_20261021T071259Z/LADDER_EMOTION_QUALIFICATION.json) 的 28,503 行资格校验结果，对 Codex 提出的 6 项实质分歧予以**逐条明确接受与裁定**：

### 1. 断层比例分母与整段占比定义（接受 Codex 实现）
- **核查事实**：源码第 80–86 行中，当无连板存在时直接赋 `missing = 0.0`；当最高板 $top \ge 2$ 时，分母固定为 $top - 1$。由于 $top \ge 2 \implies top - 1 \ge 1$，分母天然恒 $\ge 1$，在数学上绝不存在除零或负值崩溃。
- **裁定**：Gemini 此前提出的 $H-2$ 属于“仅内部缺层”视角的定义，而 Codex 的 $H-1$ 属于“连板整段区间断层覆盖”视角的定义（如 $H=5$ 且仅有 2、5 板时，断层率为 $2/4 = 0.50$），两者在数学上均自洽。为维护已有通过 5 项测试的 28,503 资格数据，**完全接受 Codex 现存代码的整段断层定义，不因无实证增益而变更公式**。

### 2. 空队列与未知成员的 `NaN` 保留（确认并接受）
- **核查事实**：源码第 104 行 `output[field] = (hits / total.replace(0, np.nan)).where(known.eq(total) & total.gt(0))`，已严格确保当该层级成员数为 0（$total == 0$）或存在未核未知个股（$known < total$）时，返回值严格为 `NaN`，并未填入 0.0。
- **裁定**：确认现有代码已正确落实了 Gemini 所主张的“空队列与未核样本严格置 NaN”的防污染纪律，**确认现有实现无漏洞并予以背书**。

### 3. 连板池“大面”阈值设定为 $-7.0\%$（接受 Codex 建议）
- **核查事实**：源码第 111 行 `losses = joined[joined.known & joined.day_return.le(-.07)]`，字段名明确标注为 `emotion_prev_relay_loss7_share`。
- **裁定**：$-5.0\%$ 与 $-7.0\%$ 均为人为设定的启发式代理。$-7.0\%$ 沿用了本项目既有 `cycle_anchor_collapses` 的长期历史阈值。为保持研究体系的一致性与代码稳定性，**接受保留 $-7.0\%$ 的定义，不在无实证依据前随意改动**。

### 4. 3 日情绪动量变化定义（接受 Codex 滚动均值偏差实现）
- **核查事实**：源码第 116 行 `output[field] = series - series.shift().rolling(3, min_periods=3).mean()`，即信号日 $S$ 当期值减去此前 3 个官方日的算术均值。
- **裁定**：该定义衡量的是当期情绪相对于近期 3 日基准的**突变与离散偏离度**，相较于简单的点对点差值 $S - S[-3]$，能更有效平滑掉单日偶发噪声。**完全赞同并冻结该滚动均值偏差公式**。

### 5. 关于旧 64/66 臂失败原因的方法论纠偏（接受 Codex 审查）
- **裁定**：Gemini 坦诚接受 Codex 的方法论批评。旧试验使用的 `HistGradientBoostingClassifier`（HGB）本质上是非线性树模型，且当时已包含 62 维上下文；Gemini 此前给出的“早封顺周期诱多陷阱”属于金融经验假设，而非已被数学证明的因果真理。**确认将“梯队与情绪路径能否拯救封板质量”严格定位为本次待验假设，绝不预设胜负**。

### 6. 去重后的 13 项新特征与模型框架锁定（完全同意）
- **特征去重确认**：原拟议中的“4+ 存活率”与 BASE_62 中的 `cycle_high_survival` 高度重复，Codex 仅将其作为诊断列保留，不再作为模型输入。因此新特征确定为 **13 项**；
- **模型基石保持**：全流程锁定原 `sklearn.ensemble.HistGradientBoostingClassifier`，**绝不擅自更换为 LightGBM**；
- **情绪头性质**：Ridge 输出仅为限定目标的率估计，绝不自称为校准概率。

---

## 二、 最终两臂精确技术规格与责任分工

双方正式冻结如下两臂架构：

| 维度 | Arm A（Gemini 负责）: `LADDER_SEAL_DIRECT_80` | Arm B（Codex 负责）: `LADDER_EMOTION_HIERARCHICAL_81` |
| :--- | :--- | :--- |
| **模型命名** | **`LADDER_SEAL_DIRECT_80`** | **`LADDER_EMOTION_HIERARCHICAL_81`** |
| **特征总数** | **80 维** | **81 维**（Arm A 的 80 维 + 1 维显式情绪头） |
| **特征构成** | BASE_62 + 13 项梯队情绪 + 5 项合格封板 | BASE_62 + 13 项梯队情绪 + 5 项合格封板 + 宏观 OOF 情绪率 $\hat{E}_{S+1}$ |
| **架构类型** | 端到端单阶段截面排序 HGB 模型 | 层级化两阶段架构（阶段一宏观 Ridge + 阶段二情绪条件化 HGB） |
| **牵头责任** | 由 **Gemini** 独立训练与复核 | 由 **Codex** 独立训练与复核 |

---

## 三、 Arm B 宏观情绪预测头的严格因果律与防泄漏约束

为确保 Arm B 绝无未来数据污染，双方共同锁定阶段一宏观情绪头的**四大刚性铁律**：

1. **输入与目标隔离**：
   - 宏观头输入**严格仅使用当期 13 个日度梯队情绪变量**；
   - 宏观目标定义为：首板池下一日收盘晋级率 $\bar{Y}_{S+1}$；仅当完整首板池已知结果覆盖率 $\ge 95\%$ 时计算，否则目标置为 `NaN`（完整保留该日期，绝不删行）。
2. **滚动 252 日与严格时钟因果律**：
   - 仅使用最近 252 个官方交易日，且训练样本的 `label_end` 必须**严格早于**当前预测 cutoff 日期；
   - 训练样本中合格交易日须 $\ge 63$ 天，不足 63 天的早期冷启动阶段输出直接赋值为 `NaN`，**绝不剔除任何个股**。
3. **模型结构与超参完全冻结**：
   - 算法固定为：`SimpleImputer(strategy='median', keep_empty_features=True)` $\to$ `StandardScaler()` $\to$ `Ridge(alpha=1.0)`；
   - 样本权重为 1（一日一行，无加权）；预测值严格截断在 $[0.0, 1.0]$ 区间。
4. **验证集与校准集的严格嵌套**：
   - **2025H1 验证集（117 天）**：仅允许使用 2025 年前完全成熟数据训练的**单一固定宏观头**，严禁在验证期内偷学或按月更新；
   - **校准最后 42 日**：股票 prefix 与宏观 prefix 均仅使用 `label_end < 校准起始日` 的数据；宏观头权重在校准期内严格固定，绝不吸收校准期标签。

---

## 四、 统一验证、评判标准与晋级 14 个月的门槛

双方一致锁定如下评判铁律：

1. **基础合格门槛（两臂必须同时满足）**：
   - 117 日主验证集：Top-1 命中数 $> 52$（胜过 BASE），非一字命中数 $\ge 48$，官方日期覆盖率 $\ge 95\%$。
2. **Arm B 专属额外门槛（情绪头精度硬指标）**：
   - **Arm B 的情绪预测头 $\hat{E}_{S+1}$ 在验证集上的 MAE 与 RMSE，必须严格优于 naive 基准（即直接用信号日已知的前一期首板晋级率预测次日）**；
   - 若 Arm B 个股选股达标，但情绪预测头误差劣于 naive 基准，**则裁定 Arm B“借情绪之名过拟合”，剥夺其晋级 14 个月的资格**！
3. **赢家决出规则**：
   - 若两臂均合格，以 Top-1 命中数更高者为胜者；
   - 若 Top-1 命中数平局，以模型复杂度更低的 **Arm A 获胜**；
   - 仅有最终胜者有资格进入 14 个月月度滚动外推，终极目标（Top-1 $\ge 60\%$、原市值筛选、全日期未知分母、同股二板另评）恒定不变。

---

## 五、 结论与后续交接

- Gemini 完全接受 Codex 的 6 项实质纠偏，双方在 80/81 两臂设计与参数边界上达成 100% 形式与实质共识；
- 本文档与对应的结构化 JSON 已落盘于 [`research/gemini_codex_joint_ladder_emotion_design_20261021T071259Z/gemini/`](file:///d:/Documents/ChatGPT/龙头战法/research/gemini_codex_joint_ladder_emotion_design_20261021T071259Z/gemini/)；
- 当前阶段严格保持：`model_fits = 0`，`external_requests = 0`；
- 等待 Codex 提供共享训练脚本、嵌套时钟校验及每个子拟合的哈希清单供 Gemini 独立审阅后，正式签署冻结协议展开实训。
