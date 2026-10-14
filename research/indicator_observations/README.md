# 三类观察指标研究入口

此链路只生成独立研究观测，不改生产预测、现模型分数或前5顺序。题材支持继续已有固定C1/C2；封板和09:35相对强度尚未完成预测增量检验。

## 已实现

`research_indicator_pipeline.py run`一次留存完整原池、来源绑定原竞争组、封板指标、开盘强度状态及全候选CSV。支持原V275档案和`capture_signal_theme_inputs.py`实际收盘来源；原登记候选另标，研究池不会扩大原11只结算分母。

`collect-signal`复用已有采集链路，真信号日15:00—16:30采四池、独立日线与明确报价，随后生成新日期研究观察附页。参数含--signal、--calendar、--protocol、--output，可附--prior-roster/--prior-roster-sha256和已审阅--theme-evidence/--theme-bindings。来源不齐停在保存的完整节点；它不采题材新闻、不认证证券/全A、不自动作者审阅或冻结预测。

`collect-open`用事前冻结原组，真实目标日09:35之后每码一次采集、6线程、timeout(5,8)。保存全部响应、日期范围、失败及取得时间；旧日行不当今天行情，只提取唯一目标日09:35行，未来行/重复目标行拒绝。当前Sina源的分钟起止、单位与复权语义未认证，因此归档不自动形成有效强度，不得手工标成已认证。

有效09:35输入可用`--minute-rows`接入绑定包：JSON对象records，每条含literal code、raw_path绝对路径、raw_sha256、raw_record_index、真实available_at。原raw须为JSONlist，code/bar_end/close/interval_seconds=60/timestamp_side=end/adjustment=none/price_unit=yuan/source从原记录重建。绑定包的语义或价格与原记录冲突会拒绝。供应商声明语义仍不是交易所认证。

先前收盘可由`--daily-archive`读取原V293或新信号来源；完整原始重建后只提取原信号日不复权close，独立于分钟来源。任何必需价格缺失、无同行或来源迟于预定截止，强度主值未知。09:35行09:36才取得最早只能09:36使用。回放不增加真实前向预测样本。

## 本批实际结果

完整原48只留存，原11只登记完整保留；46行首封分位、48行原报告开板次数均只是当前旧档诊断，盘后原截止可用0行。精确事件时长缺完整事件序列，全部未知。目标10月8日09:35尚未发生，48行待发生。59项针对测试及14个子检查通过；没有训练或计算预测提升。题材10份直接绑定来源哈希未变，细分已核验成员仍0，不能重抓旧资料回填9月30日。

当前交付位于`implementation_20261007T080458Z/complete_archive_run/交付说明.md`，全候选见同目录all_candidate_observations.csv，原组及后续参数见research_roster.json和collection_plan.json。

## 已准备的目标日原组采集命令

以下命令仅在真实目标交易日09:35之后执行，今天会在请求和建目录前拒绝；输出须唯一，不覆盖已有档案。它仍不能认证Sina分钟语义，也不会变更模型。

```powershell
D:\python\python.exe research_indicator_pipeline.py collect-open --roster "research/indicator_observations/implementation_20261007T080458Z/complete_archive_run/research_roster.json" --roster-sha256 "713d2c8587a8b932057fe824bf1f71757b76c2a8f9b203aea81a35f4e3232de3" --calendar "data/v279/exchange_calendar_audit_20261002_v1/calendar.json" --output "research/indicator_observations/open_capture_20261008_unique"
```

接入合格分钟包后用`run --roster <同一原组> --roster-sha256 <同一哈希> --calendar <原日历> --daily-archive <独立日线来源> --minute-rows <原始绑定包> --seal-csv <V275档案> --seal-report <其原report> --output <唯一新目录>`。新信号收盘档用--signal-archive替代V275二参数。封板原响应为供应商报告，不当逐笔或队列成交认证。

这是程序与来源准备完成，不是效果检验通过。各项等合格输入后按单一假设/固定基准/至多两挑战预登记，缺失信息对照和完整原组都保留，不在旧窗口搜参数。平台目标及定时任务未启用或改写。
