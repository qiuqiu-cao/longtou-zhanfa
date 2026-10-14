# 历史5分钟数据补齐请求包

- 请求总数：1721
- 已有完整48根数据：245
- 仍缺：1476
- 缺失范围：仅首板；连板不在本请求包内。

导出未复权5分钟K线，必须包含 `code,date,time,open,high,low,close,volume,amount`。
时间必须声明为K线开始时间或结束时间；成交量必须声明为股或手；成交额必须声明为元或万元。
不要补写停牌/无成交K线，不要把缺失值填0。每个股票交易日应有48根5分钟K线。

示例导入：

`python import_minute_csv_v47.py --input D:\your_data\historical_5min.csv --output data\v49\imported_5min_batch1 --source 同花顺导出 --interval 5 --timestamp-side end --volume-unit shares --amount-unit yuan --adjustment unadjusted --requests D:\Documents\ChatGPT\龙头战法\data\v49\minute_request_pack_20260906\remaining_requests.json`

导入结果仍标记为 `USER_IMPORTED_NOT_INDEPENDENTLY_VERIFIED`，不会被宣称为真实排队成交。
