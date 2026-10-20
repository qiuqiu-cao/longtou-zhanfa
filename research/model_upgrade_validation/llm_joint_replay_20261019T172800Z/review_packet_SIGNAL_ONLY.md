# 双模型固定历史回放：仅信号资料，无答案

完整原归档连板池；日期/代码/名称/BASE及ROLL分数/次日行情全部遮蔽。null=未知，不是0。封板分钟为自09:30起的日内墙钟分钟，午休不扣除，0=09:30。换手/振幅单位%，amount/cap只给同日相对名次；历史量单位未统一，不解释为股或手。one_price为当日OHLC同价形态，不是可成交性。没有合格历史题材/论坛/公告全文，所以所有候选叙事及公告风险未知。

表内 T-2/T-1/T 为市场过去三天。股票表完整保留每日归档连板候选。过去五日OHLC相对昨收及换手另见JSON；需要时必须阅读，不另取原数据库。

## D01 完整池 7只

[{"limit_count": 92.0, "consecutive_count": 14.0, "max_board": 4.0, "failed_count": 14.0, "down_limit_count": 1.0, "failed_rate": 0.1321, "seal_rate": 0.8679, "yesterday_premium": 0.0237, "emotion_state": "加速"}, {"limit_count": 69.0, "consecutive_count": 15.0, "max_board": 4.0, "failed_count": 25.0, "down_limit_count": 6.0, "failed_rate": 0.266, "seal_rate": 0.734, "yesterday_premium": -0.0063, "emotion_state": "修复"}, {"limit_count": 32.0, "consecutive_count": 7.0, "max_board": 3.0, "failed_count": 20.0, "down_limit_count": 135.0, "failed_rate": 0.3846, "seal_rate": 0.6154, "yesterday_premium": -0.0544, "emotion_state": "退潮"}]

|id|boards|one_price|turnover_pct|amplitude_pct|first_seal_min|last_seal_min|reported_open_count|amount_pctile|float_cap_pctile|volume_vs_previous|observed_nonword_streak|first_open_after_one_price|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|S01|2|0.0|0.1057|7.1257|?|?|?|28.5714|28.5714|2.8476|2.0|0.0|
|S02|2|0.0|0.0833|1.2195|?|?|?|42.8571|57.1429|3.9715|1.0|1.0|
|S03|3|1.0|0.0456|0.0|0.0|0.0|0.0|14.2857|100.0|0.5342|1.0|0.0|
|S04|2|0.0|0.077|7.9629|?|?|?|71.4286|71.4286|2.9186|1.0|1.0|
|S05|3|0.0|0.1026|8.2474|?|?|?|100.0|85.7143|0.8791|3.0|0.0|
|S06|2|0.0|0.1817|7.7396|?|?|?|85.7143|42.8571|4.9103|2.0|0.0|
|S07|3|0.0|0.1811|3.7557|?|?|?|57.1429|14.2857|2.4171|3.0|0.0|

## D02 完整池 11只

[{"limit_count": 26.0, "consecutive_count": 4.0, "max_board": 6.0, "failed_count": 14.0, "down_limit_count": 26.0, "failed_rate": 0.35, "seal_rate": 0.65, "yesterday_premium": -0.0125, "emotion_state": "大分歧"}, {"limit_count": 81.0, "consecutive_count": 7.0, "max_board": 7.0, "failed_count": 30.0, "down_limit_count": 3.0, "failed_rate": 0.2703, "seal_rate": 0.7297, "yesterday_premium": 0.0323, "emotion_state": "修复"}, {"limit_count": 109.0, "consecutive_count": 11.0, "max_board": 4.0, "failed_count": 18.0, "down_limit_count": 5.0, "failed_rate": 0.1417, "seal_rate": 0.8583, "yesterday_premium": 0.0152, "emotion_state": "升温"}]

|id|boards|one_price|turnover_pct|amplitude_pct|first_seal_min|last_seal_min|reported_open_count|amount_pctile|float_cap_pctile|volume_vs_previous|observed_nonword_streak|first_open_after_one_price|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|S01|3|0.0|0.1133|12.5|?|?|?|81.8182|63.6364|1.0148|3.0|0.0|
|S02|2|0.0|0.1516|9.5949|?|?|?|72.7273|45.4545|1.0346|2.0|0.0|
|S03|2|0.0|0.0434|5.1651|?|?|?|54.5455|81.8182|0.5629|2.0|0.0|
|S04|2|0.0|0.063|4.9928|?|?|?|90.9091|100.0|0.4036|2.0|0.0|
|S05|2|0.0|0.0742|6.3836|?|?|?|100.0|90.9091|1.1128|2.0|0.0|
|S06|2|0.0|0.0478|4.472|?|?|?|36.3636|54.5455|2.2432|2.0|0.0|
|S07|2|0.0|0.0345|1.9084|?|?|?|45.4545|72.7273|0.2889|2.0|0.0|
|S08|2|0.0|0.1539|6.9136|?|?|?|63.6364|36.3636|2.464|2.0|0.0|
|S09|2|0.0|0.0628|8.1448|?|?|?|18.1818|9.0909|1.3012|2.0|0.0|
|S10|2|0.0|0.0594|9.0452|?|?|?|27.2727|27.2727|2.5669|2.0|0.0|
|S11|4|1.0|0.0199|0.0|0.0|0.0|0.0|9.0909|18.1818|2.1002|1.0|0.0|

## D03 完整池 8只

[{"limit_count": 54.0, "consecutive_count": 14.0, "max_board": 6.0, "failed_count": 30.0, "down_limit_count": 8.0, "failed_rate": 0.3571, "seal_rate": 0.6429, "yesterday_premium": 0.0089, "emotion_state": "大分歧"}, {"limit_count": 38.0, "consecutive_count": 12.0, "max_board": 4.0, "failed_count": 22.0, "down_limit_count": 16.0, "failed_rate": 0.3667, "seal_rate": 0.6333, "yesterday_premium": -0.0123, "emotion_state": "退潮"}, {"limit_count": 44.0, "consecutive_count": 8.0, "max_board": 5.0, "failed_count": 18.0, "down_limit_count": 20.0, "failed_rate": 0.2903, "seal_rate": 0.7097, "yesterday_premium": -0.0102, "emotion_state": "修复"}]

|id|boards|one_price|turnover_pct|amplitude_pct|first_seal_min|last_seal_min|reported_open_count|amount_pctile|float_cap_pctile|volume_vs_previous|observed_nonword_streak|first_open_after_one_price|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|S01|2|0.0|0.1353|5.3451|?|?|?|25.0|37.5|1.7979|2.0|0.0|
|S02|2|0.0|0.1977|8.567|?|?|?|87.5|100.0|2.5564|2.0|0.0|
|S03|5|0.0|0.2391|4.9374|?|?|?|75.0|75.0|10.5217|4.0|0.0|
|S04|4|1.0|0.0022|0.0|0.0|0.0|0.0|12.5|50.0|0.8897|0.0|0.0|
|S05|3|0.0|0.5505|13.8462|?|?|?|62.5|12.5|4.0783|3.0|0.0|
|S06|2|0.0|0.2093|18.533|?|?|?|37.5|25.0|10.2788|1.0|1.0|
|S07|2|0.0|0.1399|5.2104|?|?|?|50.0|62.5|1.7497|2.0|0.0|
|S08|2|0.0|0.2567|6.616|?|?|?|100.0|87.5|8.6546|1.0|1.0|

## D04 完整池 7只

[{"limit_count": 45.0, "consecutive_count": 5.0, "max_board": 3.0, "failed_count": 20.0, "down_limit_count": 4.0, "failed_rate": 0.3077, "seal_rate": 0.6923, "yesterday_premium": -0.007, "emotion_state": "大分歧"}, {"limit_count": 25.0, "consecutive_count": 8.0, "max_board": 4.0, "failed_count": 14.0, "down_limit_count": 5.0, "failed_rate": 0.359, "seal_rate": 0.641, "yesterday_premium": 0.0057, "emotion_state": "加速"}, {"limit_count": 26.0, "consecutive_count": 7.0, "max_board": 5.0, "failed_count": 20.0, "down_limit_count": 15.0, "failed_rate": 0.4348, "seal_rate": 0.5652, "yesterday_premium": 0.0085, "emotion_state": "大分歧"}]

|id|boards|one_price|turnover_pct|amplitude_pct|first_seal_min|last_seal_min|reported_open_count|amount_pctile|float_cap_pctile|volume_vs_previous|observed_nonword_streak|first_open_after_one_price|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|S01|2|0.0|0.1828|8.5443|?|?|?|85.7143|85.7143|2.378|2.0|0.0|
|S02|2|0.0|0.0998|8.5399|?|?|?|42.8571|28.5714|1.2997|2.0|0.0|
|S03|2|0.0|0.0736|9.9602|?|?|?|100.0|100.0|1.0865|2.0|0.0|
|S04|5|1.0|0.0291|0.0|0.0|0.0|0.0|28.5714|14.2857|1.6187|1.0|0.0|
|S05|3|0.0|0.1195|1.8919|?|?|?|57.1429|57.1429|3.8851|1.0|1.0|
|S06|3|1.0|0.0084|0.0|0.0|0.0|0.0|14.2857|42.8571|0.9449|1.0|0.0|
|S07|5|0.0|0.2674|11.9508|?|?|?|71.4286|71.4286|2.1369|3.0|0.0|

## D05 完整池 7只

[{"limit_count": null, "consecutive_count": null, "max_board": null, "failed_count": null, "down_limit_count": null, "failed_rate": null, "seal_rate": null, "yesterday_premium": null, "emotion_state": null}, {"limit_count": null, "consecutive_count": null, "max_board": null, "failed_count": null, "down_limit_count": null, "failed_rate": null, "seal_rate": null, "yesterday_premium": null, "emotion_state": null}, {"limit_count": 31.0, "consecutive_count": 7.0, "max_board": 5.0, "failed_count": 10.0, "down_limit_count": 56.0, "failed_rate": 0.2439, "seal_rate": 0.7561, "yesterday_premium": -0.0405, "emotion_state": "修复"}]

|id|boards|one_price|turnover_pct|amplitude_pct|first_seal_min|last_seal_min|reported_open_count|amount_pctile|float_cap_pctile|volume_vs_previous|observed_nonword_streak|first_open_after_one_price|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|S01|2|0.0|0.2733|11.8182|?|?|?|100.0|?|?|1.0|?|
|S02|3|0.0|0.078|10.4734|?|?|?|28.5714|?|3.6699|1.0|?|
|S03|3|0.0|0.1697|7.8563|?|?|?|85.7143|?|2.1304|1.0|?|
|S04|2|0.0|0.0761|4.8554|?|?|?|42.8571|?|1.3115|1.0|?|
|S05|2|0.0|0.1889|10.3061|?|?|?|71.4286|?|?|1.0|?|
|S06|3|0.0|0.1869|9.9698|?|?|?|57.1429|?|1.2154|1.0|?|
|S07|5|1.0|0.0048|0.0|?|?|?|14.2857|?|0.84|0.0|0.0|

## D06 完整池 11只

[{"limit_count": 31.0, "consecutive_count": 7.0, "max_board": 5.0, "failed_count": 10.0, "down_limit_count": 56.0, "failed_rate": 0.2439, "seal_rate": 0.7561, "yesterday_premium": -0.0405, "emotion_state": "修复"}, {"limit_count": 51.0, "consecutive_count": 9.0, "max_board": 6.0, "failed_count": 8.0, "down_limit_count": 10.0, "failed_rate": 0.1356, "seal_rate": 0.8644, "yesterday_premium": 0.0132, "emotion_state": "加速"}, {"limit_count": 48.0, "consecutive_count": 11.0, "max_board": 7.0, "failed_count": 11.0, "down_limit_count": 8.0, "failed_rate": 0.1864, "seal_rate": 0.8136, "yesterday_premium": -0.0121, "emotion_state": "大分歧"}]

|id|boards|one_price|turnover_pct|amplitude_pct|first_seal_min|last_seal_min|reported_open_count|amount_pctile|float_cap_pctile|volume_vs_previous|observed_nonword_streak|first_open_after_one_price|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|S01|2|0.0|0.1865|3.9062|?|?|?|72.7273|?|2.7438|1.0|?|
|S02|3|0.0|0.1707|19.9461|?|?|?|81.8182|?|2.1327|2.0|?|
|S03|2|1.0|0.0192|0.0|?|?|?|36.3636|?|0.5712|0.0|0.0|
|S04|7|1.0|0.0134|0.0|?|?|?|45.4545|?|0.8936|0.0|0.0|
|S05|2|1.0|0.0011|0.0|?|?|?|9.0909|?|0.6795|0.0|0.0|
|S06|2|0.0|0.0576|13.2383|?|?|?|54.5455|?|1.3268|1.0|?|
|S07|3|1.0|0.0261|0.0|?|?|?|27.2727|?|0.4477|0.0|0.0|
|S08|4|0.0|0.1965|18.7713|?|?|?|100.0|?|9.5572|2.0|?|
|S09|2|0.0|0.1387|8.0157|?|?|?|90.9091|?|6.5175|1.0|?|
|S10|2|1.0|0.0177|0.0|?|?|?|18.1818|?|0.4948|0.0|0.0|
|S11|3|0.0|0.0459|2.9049|?|?|?|63.6364|?|1.8286|2.0|?|

## D07 完整池 7只

[{"limit_count": null, "consecutive_count": null, "max_board": null, "failed_count": null, "down_limit_count": null, "failed_rate": null, "seal_rate": null, "yesterday_premium": null, "emotion_state": null}, {"limit_count": null, "consecutive_count": null, "max_board": null, "failed_count": null, "down_limit_count": null, "failed_rate": null, "seal_rate": null, "yesterday_premium": null, "emotion_state": null}, {"limit_count": 38.0, "consecutive_count": 7.0, "max_board": 5.0, "failed_count": 44.0, "down_limit_count": 9.0, "failed_rate": 0.5366, "seal_rate": 0.4634, "yesterday_premium": -0.0178, "emotion_state": "大分歧"}]

|id|boards|one_price|turnover_pct|amplitude_pct|first_seal_min|last_seal_min|reported_open_count|amount_pctile|float_cap_pctile|volume_vs_previous|observed_nonword_streak|first_open_after_one_price|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|S01|2|1.0|0.0252|0.0|?|?|?|28.5714|?|?|0.0|0.0|
|S02|2|0.0|0.0409|7.3171|?|?|?|42.8571|?|?|1.0|?|
|S03|5|0.0|0.0727|9.4059|?|?|?|57.1429|?|?|1.0|?|
|S04|2|?|0.153|14.9225|?|?|?|71.4286|?|?|?|?|
|S05|2|0.0|0.1981|9.0158|?|?|?|100.0|?|1.0794|2.0|0.0|
|S06|2|1.0|0.0129|0.0|?|?|?|14.2857|?|0.4506|1.0|0.0|
|S07|2|0.0|0.1169|13.6364|?|?|?|85.7143|?|1.2089|2.0|0.0|

## D08 完整池 11只

[{"limit_count": 40.0, "consecutive_count": 8.0, "max_board": 7.0, "failed_count": 24.0, "down_limit_count": 2.0, "failed_rate": 0.375, "seal_rate": 0.625, "yesterday_premium": 0.0083, "emotion_state": "大分歧"}, {"limit_count": 76.0, "consecutive_count": 7.0, "max_board": 3.0, "failed_count": 39.0, "down_limit_count": 14.0, "failed_rate": 0.3391, "seal_rate": 0.6609, "yesterday_premium": 0.0112, "emotion_state": "退潮"}, {"limit_count": 53.0, "consecutive_count": 11.0, "max_board": 3.0, "failed_count": 23.0, "down_limit_count": 6.0, "failed_rate": 0.3026, "seal_rate": 0.6974, "yesterday_premium": 0.0055, "emotion_state": "退潮"}]

|id|boards|one_price|turnover_pct|amplitude_pct|first_seal_min|last_seal_min|reported_open_count|amount_pctile|float_cap_pctile|volume_vs_previous|observed_nonword_streak|first_open_after_one_price|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|S01|2|0.0|0.0794|7.2755|?|?|?|72.7273|81.8182|1.2035|2.0|0.0|
|S02|2|0.0|0.0759|4.3169|?|?|?|45.4545|36.3636|2.7119|2.0|0.0|
|S03|2|0.0|0.1519|7.1823|?|?|?|36.3636|9.0909|1.0874|2.0|0.0|
|S04|2|0.0|0.0805|6.5848|?|?|?|54.5455|63.6364|2.5562|2.0|0.0|
|S05|2|0.0|0.0352|10.6061|?|?|?|63.6364|100.0|1.4241|2.0|0.0|
|S06|2|0.0|0.0846|3.9801|?|?|?|27.2727|18.1818|0.8211|2.0|0.0|
|S07|3|1.0|0.0111|0.0|0.0|0.0|0.0|9.0909|45.4545|0.6136|1.0|0.0|
|S08|2|0.0|0.03|2.953|?|?|?|18.1818|27.2727|0.8182|2.0|0.0|
|S09|2|0.0|0.1435|7.3322|?|?|?|100.0|54.5455|1.768|2.0|0.0|
|S10|2|0.0|0.0438|8.8296|?|?|?|81.8182|90.9091|1.0807|2.0|0.0|
|S11|2|0.0|0.1079|4.8928|?|?|?|90.9091|72.7273|2.601|2.0|0.0|

## D09 完整池 11只

[{"limit_count": 56.0, "consecutive_count": 14.0, "max_board": 4.0, "failed_count": 21.0, "down_limit_count": 1.0, "failed_rate": 0.2727, "seal_rate": 0.7273, "yesterday_premium": 0.0225, "emotion_state": "升温"}, {"limit_count": 45.0, "consecutive_count": 16.0, "max_board": 5.0, "failed_count": 58.0, "down_limit_count": 0.0, "failed_rate": 0.5631, "seal_rate": 0.4369, "yesterday_premium": 0.0126, "emotion_state": "大分歧"}, {"limit_count": 46.0, "consecutive_count": 11.0, "max_board": 5.0, "failed_count": 23.0, "down_limit_count": 19.0, "failed_rate": 0.3333, "seal_rate": 0.6667, "yesterday_premium": -0.0134, "emotion_state": "修复"}]

|id|boards|one_price|turnover_pct|amplitude_pct|first_seal_min|last_seal_min|reported_open_count|amount_pctile|float_cap_pctile|volume_vs_previous|observed_nonword_streak|first_open_after_one_price|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|S01|2|1.0|0.0118|0.0|0.0|0.0|0.0|9.0909|27.2727|1.1763|0.0|0.0|
|S02|2|0.0|0.0883|5.3283|?|?|?|63.6364|72.7273|0.3858|2.0|0.0|
|S03|2|1.0|0.0312|0.0|0.0|0.0|0.0|45.4545|81.8182|6.7803|0.0|0.0|
|S04|3|1.0|0.0182|0.0|0.0|0.0|0.0|36.3636|54.5455|0.1493|2.0|0.0|
|S05|2|0.0|0.1271|10.231|?|?|?|81.8182|63.6364|2.1385|2.0|0.0|
|S06|2|0.0|0.27|8.0876|?|?|?|72.7273|9.0909|1.2534|2.0|0.0|
|S07|2|0.0|0.0609|7.9435|?|?|?|100.0|100.0|0.9429|2.0|0.0|
|S08|5|1.0|0.0167|0.0|0.0|0.0|0.0|18.1818|18.1818|0.877|1.0|0.0|
|S09|2|1.0|0.012|0.0|0.0|0.0|0.0|27.2727|36.3636|0.3343|1.0|0.0|
|S10|4|0.0|0.0838|2.8292|?|?|?|54.5455|45.4545|0.3946|4.0|0.0|
|S11|3|0.0|0.0964|11.6955|?|?|?|90.9091|90.9091|1.63|2.0|0.0|

## D10 完整池 15只

[{"limit_count": 86.0, "consecutive_count": 13.0, "max_board": 6.0, "failed_count": 15.0, "down_limit_count": 2.0, "failed_rate": 0.1485, "seal_rate": 0.8515, "yesterday_premium": 0.0513, "emotion_state": "修复"}, {"limit_count": 70.0, "consecutive_count": 18.0, "max_board": 4.0, "failed_count": 34.0, "down_limit_count": 0.0, "failed_rate": 0.3269, "seal_rate": 0.6731, "yesterday_premium": 0.028, "emotion_state": "升温"}, {"limit_count": 48.0, "consecutive_count": 15.0, "max_board": 5.0, "failed_count": 26.0, "down_limit_count": 7.0, "failed_rate": 0.3514, "seal_rate": 0.6486, "yesterday_premium": 0.0002, "emotion_state": "大分歧"}]

|id|boards|one_price|turnover_pct|amplitude_pct|first_seal_min|last_seal_min|reported_open_count|amount_pctile|float_cap_pctile|volume_vs_previous|observed_nonword_streak|first_open_after_one_price|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|S01|2|0.0|0.1135|7.8212|?|?|?|46.6667|?|1.6756|2.0|0.0|
|S02|2|0.0|0.0943|4.0169|?|?|?|66.6667|?|3.0144|2.0|0.0|
|S03|2|0.0|0.0476|7.0326|?|?|?|60.0|?|2.1883|2.0|0.0|
|S04|2|1.0|0.0291|0.0|?|?|?|6.6667|?|0.4215|1.0|0.0|
|S05|2|0.0|0.1694|10.0|?|?|?|93.3333|?|2.3065|2.0|0.0|
|S06|2|0.0|0.0344|4.3478|?|?|?|40.0|?|0.491|2.0|0.0|
|S07|3|0.0|0.2604|14.9813|?|?|?|73.3333|?|4.7716|2.0|0.0|
|S08|3|1.0|0.0324|0.0|?|?|?|33.3333|?|0.3205|2.0|0.0|
|S09|2|1.0|0.0852|0.0|?|?|?|26.6667|?|1.8325|1.0|0.0|
|S10|2|0.0|0.1039|9.8434|?|?|?|53.3333|?|2.3065|2.0|0.0|
|S11|5|?|0.1879|9.3886|?|?|?|80.0|?|?|?|?|
|S12|3|0.0|0.2837|8.1818|?|?|?|100.0|?|3.1468|3.0|0.0|
|S13|2|0.0|0.1497|11.8765|?|?|?|86.6667|?|1.4667|2.0|0.0|
|S14|2|1.0|0.0138|0.0|?|?|?|20.0|?|0.1981|0.0|0.0|
|S15|3|1.0|0.0219|0.0|?|?|?|13.3333|?|0.6887|1.0|0.0|

## D11 完整池 10只

[{"limit_count": 75.0, "consecutive_count": 5.0, "max_board": 7.0, "failed_count": 21.0, "down_limit_count": 1.0, "failed_rate": 0.2188, "seal_rate": 0.7812, "yesterday_premium": 0.0271, "emotion_state": "修复"}, {"limit_count": 78.0, "consecutive_count": 23.0, "max_board": 8.0, "failed_count": 18.0, "down_limit_count": 1.0, "failed_rate": 0.1875, "seal_rate": 0.8125, "yesterday_premium": 0.0424, "emotion_state": "加速"}, {"limit_count": 36.0, "consecutive_count": 10.0, "max_board": 3.0, "failed_count": 20.0, "down_limit_count": 12.0, "failed_rate": 0.3571, "seal_rate": 0.6429, "yesterday_premium": -0.0048, "emotion_state": "退潮"}]

|id|boards|one_price|turnover_pct|amplitude_pct|first_seal_min|last_seal_min|reported_open_count|amount_pctile|float_cap_pctile|volume_vs_previous|observed_nonword_streak|first_open_after_one_price|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|S01|2|0.0|0.093|5.2632|?|?|?|60.0|90.0|6.1318|2.0|0.0|
|S02|3|0.0|0.1331|6.0222|?|?|?|50.0|40.0|0.9868|3.0|0.0|
|S03|2|0.0|0.0407|1.8568|?|?|?|30.0|70.0|0.8526|2.0|0.0|
|S04|3|0.0|0.1569|8.2066|?|?|?|80.0|80.0|1.0656|3.0|0.0|
|S05|3|0.0|0.24|4.5728|?|?|?|70.0|50.0|18.1161|2.0|0.0|
|S06|2|0.0|0.1566|4.0077|?|?|?|40.0|10.0|3.8769|1.0|1.0|
|S07|2|0.0|0.0535|6.5401|?|?|?|20.0|20.0|1.2336|2.0|0.0|
|S08|2|0.0|0.3699|4.8282|?|?|?|100.0|60.0|3.9031|2.0|0.0|
|S09|3|1.0|0.0168|0.0|0.0|0.0|0.0|10.0|30.0|1.1789|0.0|0.0|
|S10|2|0.0|0.0798|15.4472|?|?|?|90.0|100.0|1.2916|2.0|0.0|

## D12 完整池 17只

[{"limit_count": 56.0, "consecutive_count": 7.0, "max_board": 5.0, "failed_count": 43.0, "down_limit_count": 7.0, "failed_rate": 0.4343, "seal_rate": 0.5657, "yesterday_premium": 0.0253, "emotion_state": "升温"}, {"limit_count": 52.0, "consecutive_count": 13.0, "max_board": 6.0, "failed_count": 25.0, "down_limit_count": 3.0, "failed_rate": 0.3247, "seal_rate": 0.6753, "yesterday_premium": 0.0119, "emotion_state": "大分歧"}, {"limit_count": 48.0, "consecutive_count": 17.0, "max_board": 7.0, "failed_count": 25.0, "down_limit_count": 5.0, "failed_rate": 0.3425, "seal_rate": 0.6575, "yesterday_premium": 0.0144, "emotion_state": "大分歧"}]

|id|boards|one_price|turnover_pct|amplitude_pct|first_seal_min|last_seal_min|reported_open_count|amount_pctile|float_cap_pctile|volume_vs_previous|observed_nonword_streak|first_open_after_one_price|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|S01|3|1.0|0.0078|0.0|0.0|0.0|0.0|23.5294|82.3529|0.5768|1.0|0.0|
|S02|3|1.0|0.0132|0.0|0.0|0.0|0.0|11.7647|17.6471|0.7608|1.0|0.0|
|S03|7|0.0|0.3488|11.9522|?|?|?|94.1176|41.1765|0.885|5.0|0.0|
|S04|2|0.0|0.1538|5.9146|?|?|?|76.4706|64.7059|1.2549|2.0|0.0|
|S05|2|0.0|0.0661|6.4067|?|?|?|47.0588|76.4706|3.2675|2.0|0.0|
|S06|2|0.0|0.0402|5.9591|?|?|?|88.2353|100.0|1.4437|2.0|0.0|
|S07|3|0.0|0.1354|9.1384|?|?|?|41.1765|23.5294|0.9917|3.0|0.0|
|S08|3|1.0|0.0082|0.0|0.0|0.0|0.0|17.6471|47.0588|0.793|1.0|0.0|
|S09|2|0.0|0.3141|13.1915|?|?|?|82.3529|35.2941|0.9599|2.0|0.0|
|S10|2|0.0|0.147|4.6154|?|?|?|35.2941|5.8824|2.5897|2.0|0.0|
|S11|2|1.0|0.0053|0.0|0.0|0.0|0.0|5.8824|11.7647|0.5255|0.0|0.0|
|S12|3|0.0|0.0282|6.758|?|?|?|64.7059|94.1176|2.7038|3.0|0.0|
|S13|5|0.0|0.2945|13.9401|?|?|?|100.0|70.5882|2.1852|4.0|0.0|
|S14|2|0.0|0.0534|3.7249|?|?|?|29.4118|29.4118|1.8674|2.0|0.0|
|S15|2|0.0|0.0629|8.973|?|?|?|70.5882|88.2353|2.8209|2.0|0.0|
|S16|2|0.0|0.0912|4.6185|?|?|?|58.8235|52.9412|0.8348|2.0|0.0|
|S17|3|0.0|0.0812|2.0581|?|?|?|52.9412|58.8235|6.6534|2.0|0.0|
