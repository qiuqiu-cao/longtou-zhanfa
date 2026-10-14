# 题材与同身位竞争：输入接入说明

本入口是 `prepare_theme_competition_inputs.py`。它复用V293原始行情重建、V295上下文重建、V150题材角色及V114竞争关系计算，不改生产模型或V298冻结入口。

## 已完成的输入能力

- 保存供应商声明范围内的全部收盘候选，以及全部2板以上候选；题材未知不删行。
- 供应商分类只作描述，保留原分类、重叠标签、原始取得时间和缺失。技术标签不自动成为业务题材。
- 未知题材的同题材竞争数、角色和成员状态保留空值；全供应商范围的同板位代码另列，不能替代同题材竞争。
- 具体题材证据须来自单独审阅的原文、实际取得时间和哈希。现有V150只支持一只股票一个已登记主题材；全部供应商重叠标签另存，不自动挑选主归属。
- 题材角色和竞争身份描述当前声明范围内的结构，不确认真实龙头，也不代表下一日晋级。
- 证券状态、独立涨停价格和交易所全市场完整性不明时保留未知。此入口不会生成这些项目的verified状态，V151训练门槛继续阻止通过。
- 没有模型原始分数时留空；不从板位、名次或晋级判断补分。
- 派生档案的可用时间包含实际生成时间。研究截止固定为信号日16:30（Asia/Shanghai），不改变原生产调度或旧登记契约。

## 当期运行

在下一次具备真实新交易日输入时，传入该日期的V293归档；可额外传入绑定同一行情归档的V295上下文。输出必须是全新目录。

```powershell
& 'D:\python\python.exe' .\prepare_theme_competition_inputs.py `
  --pool-archive '<当期V293归档目录>' `
  --signal-date '2026-10-08' `
  --protocol '.\research\theme_same_height\protocol.json' `
  --context-archive '<当期V295归档目录>' `
  --output '<全新研究输入目录>'
```

没有V295归档时可省略context参数，分类保留未知。没有题材原文时保持空草稿，不能标已审阅。已有历史底稿可用于验证接入，晚于知识截止的档案不计时点合格样本。

## 审阅题材原文后接入

另外准备符合 `templates/specific_theme_evidence_v150.json` 的实际证据，以及来源绑定JSON，再同时传入 `--theme-evidence` 和 `--theme-bindings`。不修改上述历史运行产物。

绑定顶层字段为signal_date、recorded_at、reviewed_at、review_status、reviewer、review_notes、evidence_sha256和groups。只有实际审阅原文后，review_status才可填写REVIEWED_SOURCE_TEXT。日期和哈希须对应提交的V150证据文件。

groups以实际theme_group_id为键。每组分别绑定membership、driver、assessment三个来源。每个来源含绝对raw_path、raw_sha256、captured_at以及能在UTF-8原文中找到的excerpt；membership和driver另含与V150一致的url、published_at及publication_time_basis=REVIEWED_EXACT_TIMESTAMP。membership摘录必须明确包含该组全部代码。出版时间、取得时间、审阅时间、档案形成时间分别核验；禁止倒填。

原文定位和哈希检查不能自动证明业务归属或催化事实。来源层级、事实判断、发布时间仍须实际审阅。空模板、供应商概念分类和未经核验叙事不能代替这项工作。首轮正式比较前还要登记精确算法、特征、训练与验证边界；本工具不授权拟合或部署。

## 本次实际检查

现有9月30日底稿：48只全部保留，其中11只连板候选与原登记一致。已核验的当时题材仍为0；同题材竞争字段均为空。实际档案形成于10月3日，属于历史接入检查，不能成为9月30日事前题材证据，也不能改写原预测。

9项定向合成测试通过；最初失败保留在 `input_preparation_validation.jsonl`。实际输入包的来源和产物哈希、完整候选以及V151拒绝未核验证券的行为均已检查。新增真实前向样本为0；没有准确率改善结论。
