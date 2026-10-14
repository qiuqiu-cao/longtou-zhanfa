# V13 前向大面风险监控

- 状态：`DATA_COLLECTION_REQUIRED`
- 已结算：0，纸面成交代理：0，大面标签：0
- 大面率：NA，95% Wilson上界：NA
- 晋级精确率：NA
- 结论：need 60 more resolved executed-proxy large-loss labels

该监控只接受V10哈希链中按时确认并完成T+1结算的记录；缺失结果不会按安全样本计入。
即使风险组件通过，也不能单独把系统升级为实盘状态。
