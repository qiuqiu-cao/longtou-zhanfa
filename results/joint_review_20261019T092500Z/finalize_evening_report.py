"""Publish a readable evening append; keep afternoon forecasts and report."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pandas as pd
import short_term_thinking_dataset_builder as b
import relay_multiday_scenarios as ev
OUT=Path(__file__).resolve().parent
def main():
 original=OUT/'independent_full39_BEFORE_MODEL_AND_GEMINI.json';newpath=OUT/'independent_full39_EVENING_APPEND_AFTER_MODEL.json'
 old=b.read(original);new=b.read(newpath);review=b.read(OUT/'evening_evidence_review_APPEND_ONLY.json');b.verify(review['sources'])
 rows=[];oldmap={r['code']:r for r in old['rows']}
 for r in new['rows']:
  rows.append(dict(code=r['code'],name=r['name'],boards=r['boards'],afternoon_close_limit_call=oldmap[r['code']]['next_close_limit_up_call'],evening_close_limit_call=r['next_close_limit_up_call'],priority_rank=r['priority_rank'],new_evidence=r.get('evening_additional_evidence','未取得影响本股方向的新证据'),afternoon_formed_at=old['formed_at'],evening_formed_at=new['formed_at'],after_model_exposure=True))
 pd.DataFrame(rows).to_csv(OUT/'完整39只_下午与晚间预测对照.csv',index=False,encoding='utf-8-sig')
 text='''# 10月19日晚间公告补核与20日最终观察更新

最新真实补核在北京时间19日22:29—22:33进行。原下午完整39只独立预测、模型评分、47次研究训练和原结算均保留。本页属于看过模型与双方讨论后的追加证据判断，不伪装新盲审。

**20日相对优先顺序不变：吉大正元003029、保隆科技603197、万丰奥威002085。另观察东方嘉盛002889、亨通光电600487。概率未验证。** 东方嘉盛从下午的偏晋级改为未知／观察，多日方向也改未知；原YES和原多日路径到期仍须单独结算。

|股票|20日收盘涨停|21日累计|22日累计|26日累计|最重要的新增限制|
|---|---|---|---|---|---|
|吉大正元003029|偏晋级4板|偏上|偏上|未知|AI／抗量子与低空收入小，半年亏损；一字开板需实际承接|
|保隆科技603197|偏晋级3板|偏上|偏上|有条件偏上|新异动公告没有认证飞行汽车订单；离职激励回购不当增持催化|
|万丰奥威002085|偏晋级3板|偏上，允许收阴|偏上，允许震荡|有条件偏上|26日预约三季报披露窗口，不能预设财报利好|
|东方嘉盛002889|未知／观察|未知|未知|未知|补得公告镜像的“跌幅”与已核连续涨停日线方向冲突，暂未澄清|
|亨通光电600487|未知／趋势观察|有条件偏上|有条件偏上|有条件偏上|业绩预告未审计、承接成交规模较大，不预测连板|

累计指相应官方交易日收盘高于19日收盘，不等于每一天上涨。吉大和保隆的原2日连续板主情景仍保留，3日与5日不押逐日上涨／连续涨停。万丰允许中途震荡。原完整逐事件判断与取消条件在主报告和原JSON。

## 新公告怎样影响判断

1. 保隆2026-080公告称经营正常、无应披露未披露重大事项。不能从这份公告推出新的特斯拉飞行汽车订单已落实。另28,000股按18.68元回购注销对应3名离职激励对象，不是二级市场增持利好；预计22日注销不保证股价上涨。维持相对第二。[异动公告](https://pdf.dfcfw.com/pdf/H2_AN202610191830687043_1.pdf)
2. 万丰2026-046公告称经营正常、未有应披露未披露重大事项；三季报预约26日披露，当前相关业绩数据未向第三方提供。h5判断加入财报窗口风险，维持相对第三。[公告](https://pdf.dfcfw.com/pdf/H2_AN202610191830687134_1.pdf)
3. 东方原超时公告已取得，但2026-034镜像原文写15／16日“跌幅”偏离23.61%，与本批日线15／16连续涨停相矛盾，另一转载同文。没有证据判定哪边错误，保留冲突，不从公告反改价格、也不擅自把“跌”改“涨”。从主动正向降为未知观察。[公告镜像](https://pdf.dfcfw.com/pdf/H2_AN202610161830618206_1.pdf)
4. 京城临2026-025披露半年归母亏损6263.39万元；控股子公司北洋天青于9月18日因重组信息涉嫌虚假记载等被立案。是新取得的旧风险与本次披露，不倒填原时点；不把BASE第三名转成接力交易许可，执行关注排除，方向维持未知。[公告](https://pdf.dfcfw.com/pdf/H2_AN202610191830688539_1.pdf)
5. 东安、文灿、大业分别披露半年归母亏损893.47、21590.20、6399.14万元；长春披露静态PE221.72、行业27.75；海洋王提示AI智慧照明收入有限。这些反证加入完整同位比较，不把它们简化为第二天必跌，也不把概念共现当业绩兑现。

## 实际修正的数据采集经验

晚间当前可公开取得的文件，可能落款下一官方交易日。仅把索引结束日期设为19日，会漏掉19日晚公开、封面20日的公告。对11只连板加亨通的有界检查，发现9家公司共12篇这类索引，实际取得11篇PDF，一篇控股股东回函失败保留。原索引前两篇为激励回购与法律意见时，保隆异动公告位列第三；已针对该异动公告单独补读，不能用“取前两篇”代替风险优先审阅。

修复采取“可查到下一官方封面日期＋真实请求／收件时钟＋原文哈希＋风险文件优先”的来源契约。只使用19日晚实际收到的材料，不制造20日未来取得；首次公开时刻未取得者继续未知。现有任务、生产采集和模型权重没有被自动替换，单独脚本与完整回执在facts下。本次数据补强未证明准确率改善。

## 到期评价

下午版本与晚间版本都保留完整39只，20日结算收盘涨停及触板；21／22／26日分别结算2／3／5日累计、逐日与连续涨停。东方的弃判不删除下午原YES。所有429个新事件当前都尚未成熟；没有新增成交或真实正确结果。

47次多日研究、完整11121行比较的验收结论不变：2／3／5日三个目标均未通过，不部署。2日目标对旧代理的合并增益有局部统计支持，但未稳超恒看涨，不能称次日涨停准确率提升。

[完整39只下午与晚间对照](完整39只_下午与晚间预测对照.csv) · [晚间完整逐股版本](independent_full39_EVENING_APPEND_AFTER_MODEL.json) · [晚间证据与决策](evening_evidence_review_APPEND_ONLY.json)
'''
 late=OUT/'10月19日晚间公告补核_20日最终观察更新.md'
 with late.open('x',encoding='utf-8') as f:f.write(text)
 report=OUT/'10月19日联合复盘与20日预测_模型优化实测.md';s=report.read_text(encoding='utf-8')
 marker='\n\n**晚间最新补核：** 北京时间19日22:29—22:33已加入落款20日、19日晚实际公开的公告。前三仍为吉大／保隆／万丰；东方由原偏晋级降为未知观察。以下完整39只与双方原判断保留下午版本，最新逐股版本及公告反证见[晚间更新](10月19日晚间公告补核_20日最终观察更新.md)。这次追加已看模型，不声称新的盲审。\n'
 title=s.find('\n');s=s[:title]+marker+s[title:];report.write_text(s,encoding='utf-8')
 daily=b.read(OUT/'facts/normalized_daily.json');bars=pd.DataFrame([dict(code=c,date=d,**{k:r[k]/100 for k in ['open','high','low','close']},volume=r['volume_native']) for c,it in daily.items() for d,r in it['days'].items()]);sessions=b.read(ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json')['sessions']
 evaluation=ev.evaluate(new,bars,sessions,b.now());assert evaluation['new_mature_labels']==0 and len(evaluation['rows'])==429
 b.write_new(OUT/'evening429_events_PENDING_at_creation.json',evaluation)
 b.write_new(OUT/'evening_delivery_receipt.json',dict(formed_at=b.now(),top3=new['top3'],only_direction_changed='002889 YES->UNKNOWN',all39_retained=True,all429_pending=True,model_refits_in_evening=0,original_afternoon_hash_unchanged=b.sha(original)==new['original_sha256'],sources={str(p.resolve()):b.sha(p) for p in [report,late,newpath,OUT/'evening_evidence_review_APPEND_ONLY.json',OUT/'完整39只_下午与晚间预测对照.csv',OUT/'10月19日联合复盘_下午版_保留.md']}))
 print(dict(late_report=str(late.resolve()),original39_unchanged=True,latest39_retained=True,all_pending=True))
if __name__=='__main__':main()
