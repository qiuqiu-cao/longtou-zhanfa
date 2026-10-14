# V10 冻结模型与前向影子账本报告

## 结论

V10已把连板影子研究从“可重跑的CSV”升级为“冻结模型 + 盘前预注册 + 盘中只追加 + T+1只追加”的哈希链。这解决了前向验证中最容易伪造高胜率的三个问题：事后换候选、事后选盘中截图、只结算成功交易。

当前仍是 `FORWARD_SHADOW_ONLY`，`ENTER=0`。

## 冻结模型

- 冻结ID：`9738e2c165075798`
- 开发数据截止：2026-08-31
- 晋级模型：V8时间稳定 HGB，AUC 0.680，Brier技能分 +0.102
- 可成交代理：V8时间稳定 HGB，AUC 0.573，Brier技能分 +0.019
- 至少再走两板：V8时间稳定 HGB，AUC 0.681，Brier技能分 +0.066
- 联合排序：`0.65*p_two_more + 0.25*p_promotion + 0.10*p_fill_proxy`
- 大面模型：`REJECTED_NOT_USED`
- 研究范围：活跃情绪中的主板3—5板、前一日可买、非一字，每日最多一只。

冻结文件同时绑定训练数据、V7情绪输入、V8模型卡、V8代码、V10代码和序列化模型的SHA-256。当前校验为 `PASS`，保证等级为 `LOCAL_TAMPER_EVIDENT_ONLY`，不是第三方时间戳。

## 事件链

### 1. `PLAN`

- 只能在信号日15:05后、交易日9:15前创建；
- 信号日必须晚于2026-08-31；
- 必须已有同日V7冻结情绪预测；
- 冻结全部候选概率、每日第一名、候选数据哈希和两个模型冻结ID；
- 无合格研究对象时也会登记 `NO_SHADOW_CANDIDATE`，不会删掉空仓日。

### 2. `CONFIRMATION`

- 只能绑定已预注册代码和09:25/09:35/10:00快照；
- 快照必须来自V9可核验原始数据重算档案；
- 09:25超过2分钟、09:35/10:00超过5分钟永久标记迟到；
- 确认状态和原因会在校验时从原快照重算，不能手改。

### 3. `SETTLEMENT`

- 每个计划都允许结算，不论是确认、取消、数据缺失还是未成交；
- 只有按时的09:35/10:00 `CONFIRM_RESEAL` 才产生分钟回封成交代理；
- 入场价为交易日涨停价，结算使用下一个已归档交易日收盘，扣除买卖费用和滑点；
- T+1一字跌停时，按收盘盯市判定是否大面，但 `net_return` 保持未解决，不伪造可卖成交；
- 没有盘中确认的对象不生成代理买入。

## 当前前向状态

| 项目 | 当前 |
|---|---:|
| V10账本校验 | PASS |
| PLAN | 0 |
| CONFIRMATION | 0 |
| SETTLEMENT | 0 |
| 代理成交 | 0 |
| `ENTER` | 0 |

V9旧的2026-08-31演练CSV没有盘前V10预注册，因此从此不计入影子样本数。就绪度已回到0/100个合法影子观察，这是消除事后污染后的正确起点。

## 运行命令

```powershell
python relay_shadow_model_v10.py verify
python intraday_forward_v10.py prepare --input data\v7\point_in_time_rebuild\leader_candidates.csv --signal-date YYYY-MM-DD --trade-date YYYY-MM-DD
python intraday_forward_v10.py confirm --session-id SESSION_ID --asof 09:35
python intraday_forward_v10.py settle --session-id SESSION_ID --outcome outcome.csv
python intraday_forward_v10.py verify
python intraday_forward_v10.py monitor
python intraday_readiness_v10.py
```

T+1结算模板：`data/v10/templates/t1_outcome_primitive.csv`。

Level-2排队和真实成交仍为 `UNAVAILABLE`，所有成交结果都必须保留 `INFERRED_MINUTE_RESEAL_FILL` 证据等级。
