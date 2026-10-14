# 历史5分钟数据补齐请求包

- 候选总数：2243
- 已通过当前回放口径：968
- 仍缺：1275
- 缺失板组：{'FIRST_BOARD': 1273, 'CONSECUTIVE_BOARD': 2}。
- 修复原因：{'MISSING_ARCHIVE': 1266, 'DAILY_OHLC_CONTRADICTION': 9}。

导出未复权5分钟K线，必须包含 `code,date,time,open,high,low,close,volume,amount`。
时间必须声明为K线开始时间或结束时间；成交量必须声明为股或手；成交额必须声明为元或万元。
不要补写停牌/无成交K线，不要把缺失值填0。每个股票交易日应有48根5分钟K线。

示例导入：

`python import_minute_csv_v47.py --input D:\your_data\historical_5min.csv --output data\v49\imported_5min_batch1 --source 同花顺导出 --interval 5 --timestamp-side end --volume-unit shares --amount-unit yuan --adjustment unadjusted --requests D:\Documents\ChatGPT\龙头战法\data\v49\minute_request_pack_20260907_v2\remaining_requests.json`

导入结果仍标记为 `USER_IMPORTED_NOT_INDEPENDENTLY_VERIFIED`，不会被宣称为真实排队成交。
