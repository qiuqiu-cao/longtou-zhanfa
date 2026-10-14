# SUPERSEDED

该注册表的首个快照把CSV空值序列化为非标准JSON `NaN`。账本哈希仍可验证，但该格式不满足严格JSON消费者要求，因此停止追加事件。后续前向记录使用`same_height_pk_forward_v2`；旧文件只保留审计，不得继续结算。
