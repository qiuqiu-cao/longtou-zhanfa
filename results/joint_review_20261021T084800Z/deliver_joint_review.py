from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import pandas as pd
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
now=lambda:datetime.now(timezone.utc).isoformat()
ack_path=OUT/'gemini/JOINT_CONCLUSION_ACK.json';ack_md=OUT/'gemini/JOINT_CONCLUSION_ACK.md'
assert ack_path.exists() and ack_md.exists(),'Await actual Gemini agreement; do not call a proposal joint consensus'
ack=read(ack_path);acktext=json.dumps(ack,ensure_ascii=False)+ack_md.read_text(encoding='utf-8')
assert all(c in acktext for c in ['002724','002696','603138','603278','002313'])
facts=read(OUT/'facts/complete_enriched_facts_SCORES_MASKED.json');dataqa=read(OUT/'facts/DATA_AND_SCOPE_VERIFICATION.json')
own=read(OUT/'codex/INDEPENDENT_ROUND1_BEFORE_PEER_AND_MODEL.json');ownreceipt=read(OUT/'codex/INDEPENDENT_FREEZE_RECEIPT.json')
assert sha(OUT/'codex/INDEPENDENT_ROUND1_BEFORE_PEER_AND_MODEL.json')==ownreceipt['sha256']
base=read(OUT/'BASE_scores_AFTER_INDEPENDENT.json');baseby={r['code']:r for r in base['rows']}
by={r['code']:r for r in facts['rows']};cards=read(OUT/'EXECUTION_SCENARIOS_DRAFT_UNVALIDATED.json');cardby={r['code']:r for r in cards['cards']}
codes=['002724','002696','603138','603278','002313']
paths={
'002724':dict(d2='若22日晋级且低位继续响应，23日累计偏强，但更高板位的换手分歧增加',d3='26日需低位不断补位支持；否则偏高位震荡',d5='28日累计方向未知，不押连续五日上涨或五连板',support='20日实际换手后21日再加速；现有四板身位，名称情绪在百洋/九阳及低位有实际价格响应',counter='21日换手仅0.82%，不能证明次日新承接；五板挑战遇到市场缩量与淘汰端负反馈',call='偏晋级，4进5相对首位',role='本组高位情绪锚候选，未认证唯一总龙头'),
'002696':dict(d2='若22日晋级，23日累计条件偏上，旧高区域与接力分歧需观察',d3='26日名称和水产共振仍在才保留偏强；否则震荡兑现',d5='28日方向未知，高位震荡风险仍高，不押逐日上涨',support='09:36后供应商未再报告开板，有实际换手；旧名称线修复与水产群体当期响应并存',counter='16日与19日明显亏钱路径尚在，今日成交量仅昨日0.545倍；不能由15.63%换手或零报告开板认证健康锁筹',call='偏晋级，2进3；高淘汰环境下仍需承接',role='旧高位修复的中位情绪候选，非全新低位启动'),
'603138':dict(d2='若22日1进2成功，23日累计条件偏上，同时迎来新线首轮淘汰检验',d3='26日若数据方向形成持续梯队才看延续，否则偏兑现',d5='28日累计方向未知，不将首日催化直接外推五日',support='09:30:07早封、流通约44.35亿；星光/南威/航天/太极及三维/慧辰提供当期群体响应',counter='久远银海/税友/深桑达炸板，首日发酵可能次日兑现；早封不认证主力因果或独立龙头',call='偏晋级；本轮首板1进2相对首选',role='数据要素叙事低位候选；润建相关观察而非唯一父锚'),
'603278':dict(d2='23日累计方向待换手检验；开板后是否恢复前排最关键',d3='26日机器人中位能修复才继续看强，否则分化',d5='28日方向未知，不押连续涨停',support='四板一价保持市场前排，泰尔等仍有低位响应；行业关联弱公告不否定市场叙事',counter='机器人中位与后排淘汰较多，京城/宁波/福莱等负反馈；两日一价后潜在兑现压力',call='保留正向假设，但降为分歧观察',role='四板空间竞争者，与海洋王/东安可共同晋级或失败'),
'002313':dict(d2='若22日续板且同为等不共振转弱，23日累计条件偏上',d3='26日观察同位分化，不假定必胜泰尔/同为',d5='28日方向未知，低换手后的兑现仍需检验',support='二板一价维持端侧AI相关叙事身位，原市值门内',counter='今日换手1.78%，尚缺次日真实交换证据；移远容量分歧可能传导',call='偏晋级的观察项，2进3',role='端侧AI一价身位候选，非已认证资金控盘')}
final=[]
for i,c in enumerate(codes,1):
 r=by[c];p=paths[c];s=cardby[c]
 assert r['screen_cap_9326859896_52']
 final.append(dict(rank=i,code=c,name=r['name'],list_role='TOP3' if i<=3 else 'OBSERVATION',close=r['close'],limit_reference=s['next_limit_reference'],prediction=p['call'],probability='UNVALIDATED_NO_NUMERIC_PROBABILITY',support=p['support'],counter=p['counter'],identity_hypothesis=p['role'],cumulative_direction_20261023=p['d2'],cumulative_direction_20261026=p['d3'],cumulative_direction_20261028=p['d5'],every_day_up='UNKNOWN_SEPARATE_EVENT',consecutive_limit_up='UNKNOWN_SEPARATE_EVENT',execution=s,stock_specific_model_probability=None,BASE_score_uncalibrated=baseby.get(c,{}).get('score_display'),BASE_call=baseby.get(c,{}).get('BASE_direction')))
record=dict(formed_at=now(),signal_date='2026-10-21',target_date='2026-10-22',status='JOINT_CONCLUSION_AFTER_ACTUAL_PEER_ACK',actual_peer='Antigravity Gemini 3.8 Flash High',conversation_id='cbfaa95f-7eb8-46fd-8195-6ace0c4fe35e',scope='No singled-out stock seat; 67 current mainboard rows retained, original cap/thin screen respected; full universe context kept',exposure='Codex round1 archived before peer/model; subsequent joint revisions after both round1 and BASE seen; not blind or erase-memory review',market='前排抱团、中位淘汰、低位轮动。下一日基准分歧分化，核心可能修复，尚非全面普涨或全面退潮',market_scenarios={'base':'指数偏弱背景下先分歧，关注高位是否稳定和低位能否持续承接','repair':'四板前排不共振杀跌、九阳百洋不出现共同严重负反馈、数据/传媒低位持续出现相对领先与回封，淘汰端止跌','deterioration':'四板普遍走弱、名称中位和数据低位同时冲高回落、昨日失败股继续大面，则执行层降预期；原S收盘预测与完整验收日期仍保留'},top3=codes[:3],observations=codes[3:],first_board_top1='603138',candidates=final,full_current_rows=67,source_counts=facts['source_counts'],prior46_feedback=read(OUT/'facts/COMPLETE_PRIOR_COHORT_FEEDBACK.json')['layers'],peer_ack_path=str(ack_path),peer_ack_sha256=sha(ack_path),codex_acceptance_path=str(OUT/'codex/ROUND3_ACCEPTANCE_AND_REQUIRED_LIMITS.md'),forum_current21_qualified_body=0,forum_scope='2 readable original bodies dated Oct20/Oct15; third selected original body unavailable; search snippets separated',new_fits=0,new_orders=0,new_formal_registry_predictions=0,unchanged_production_BASE=True,formal_registration='Independent joint scenario archive; not production registration or model60% acceptance')
write(OUT/'JOINT_FINAL_PREDICTION_BEFORE_NEXT_SESSION.json',record)
full=[]
for r in own['rows']:
 c=r['code'];v=dict(r)
 v['joint_summary_rank']=codes.index(c)+1 if c in codes else None
 v['joint_summary_prediction']=paths[c]['call'] if c in paths else None
 v['full_row_author']='Codex complete-pool independent direction; Gemini explicitly agrees focal3+2 only'
 v['BASE_score_uncalibrated']=baseby.get(c,{}).get('score_display');v['BASE_call']=baseby.get(c,{}).get('BASE_direction')
 v['current_first_board_top1']=c=='603138'
 if c=='600271':v['counter_evidence']='首封晚于海量，流通约137.11亿超过原筛选，作为完整市场反馈保留'
 if c=='600293':v['continuity_status']='供应商报2板；19日官方参考日行情缺口，严格连续资格未知，BASE仅补充来源诊断'
 full.append(v)
pd.DataFrame(full).to_csv(OUT/'完整67只_独立方向与共同摘要及BASE对照.csv',index=False,encoding='utf-8-sig')
lines=['# 10月21日收盘复盘与10月22日共同预期',f'形成时间：{record["formed_at"]}（UTC）；所有交易日与报价日期按北京时间。','',
'**共同结论：前排抱团、中位淘汰、低位轮动。10月22日先按分歧分化预期，重点看核心能否稳定并带来新晋级；尚不判断为全面普涨，也不机械认定全面退潮。**','',
'经Codex与实际反重力Gemini逐轮事实核对和实质分歧讨论，重点前三：**海洋王、百洋股份、海量数据**；另观察**大业股份、日海智能**。仅选一只首板1进2时，本轮相对首选是**海量数据**。这些是未验证概率的独立判断，不能把共识当作60%已达标。','',
'## 收盘事实与情绪变化','',
'沪指3713.03（-0.60%），深成指11704.35（-0.82%），创业板2694.71（-1.62%）；两市成交约14294亿元，较前日少267亿元；含北交所上涨1979、下跌3360。上涨下跌与两市成交统计范围不同，不拼成同一分母。来源：[澎湃新闻收盘报道](https://finance.eastmoney.com/a/202610213902615867.html)。','',
'返回全市场涨停50、炸板19、跌停5；本次主板完整并集为涨停44、炸板18、跌停5，共67只。50只涨停与5只跌停另与[界面新闻盘后报道](https://finance.eastmoney.com/a/202610213902596617.html)交叉核对。完整源并集不等于独立认证交易所全市场无遗漏。','',
'今日主板供应商梯队：四板3只（海洋王/大业/东安）、三板1只（润建）、二板12只、首板28只。四板前排仍在，最高连续板并未较昨日提高。旧高位新华传媒今日收涨约6.45%，九阳/百洋仍修复；与时代万恒等负反馈并存，不能只选成功者或只看跌停。','',
'| 昨日完整46只主板池 | 今日晋级 | 含义 |','|---|---:|---|',
'| 首板→二板 | 12/33，36.36% | 有低位轮动，仍有21只未晋级 |',
'| 二板→三板 | 1/7，14.29% | 中位承接最弱，只润建晋级 |',
'| 三板→四板 | 3/5，60.00% | 前排保持，淘汰的长春/京城分别跌8.98%/7.52% |',
'| 四板→五板 | 0/1 | 吉大当日失败，只有一个样本，不能推导明日必失败 |','',
'这些是今天的主板样本描述，**不是任何一只股票明天的概率**。主板封板比例44/(44+18)=70.97%，也不是隔日晋级胜率。','',
'## 题材情绪怎样判断','',
'1. **名称情绪与实际水产响应仍在。**海洋王四板、百洋/九阳二板，以及低位的当日实际封板，支持群体认可继续存在；但盛洋科技炸板是直接反例。名字相似只产生关系假设，不自动认证资金因果、核心身份或晋级。九阳是旧锚反馈，不能因为盘子大就从情绪地图删掉。','',
'2. **数据要素与AI传媒出现群体响应，但次日兑现风险在。**数据侧海量、星光、南威、航天、太极、三维、慧辰，与久远/税友/深桑达失败共同看；传媒侧粤传媒、天威、出版、读者、中广、文投、兆讯，与永吉等失败及旧锚新华共同看。两类叙事有交叉，不能把所有算力、通信、软件硬并成单一父龙链。','',
'3. **机器人前排与中后排已经分化。**大业四板、泰尔二板仍强，京城、宁波及福莱等失败说明高标强不保证跟风承接。因此大业从Codex原摘要第二位降为观察；这来自梯队反证，而非公司主营关联否认。','',
'## 下一交易日候选与多日路径','',
'下表2/3/5日分别为10月23日、10月26日、10月28日相对10月21日的**累计方向/情景**。逐日上涨、每天封板和累计上涨分开，后两者没有校准概率，不据单日预测外推。','',
'| 相对顺序 | 股票与今收→22日涨停参考 | 22日判断 | 支持及主要反证 | 可交易性 |','|---|---|---|---|---|']
for i,c in enumerate(codes,1):
 r=by[c];p=paths[c];s=cardby[c]
 trade='一价预期可能排不到，开板需重新检验' if c in ['002724','603278','002313'] else '有换手路径，但成交和收益未知'
 lines.append(f'| {i}'+('重点' if i<=3 else '观察')+f' | {r["name"]} {c}：{r["close"]:.2f}→{s["next_limit_reference"]:.2f} | {p["call"]}；概率未验证 | {p["support"]}；反证：{p["counter"]} | {trade} |')
lines+=['','| 股票 | 2日：10/23 | 3日：10/26 | 5日：10/28 |','|---|---|---|---|']
for c in codes:lines.append('| '+' | '.join([by[c]['name'],paths[c]['d2'],paths[c]['d3'],paths[c]['d5']])+' |')
lines+=['','## 22日竞价与打板观察预案','',
'**以下价格情景为本次人工观察预案，未经优化/回测认证，不是新生产阈值。**最终开盘、匹配额、未匹配量、开盘后成交路径均未发生。没有相同口径的原时点对照，不编最低竞价金额、量比或健康换手百分比。','',
'| 股票 | 09:25强于预期 | 符合预期 | 低于预期/中性 | 完整竞争确认 |','|---|---|---|---|---|']
for c in codes:
 s=cardby[c];lines.append('| '+' | '.join([by[c]['name'],s['auction_strong'],s['auction_normal'],s['auction_weak'],'、'.join(s['competition'])])+' |')
for c in codes:
 s=cardby[c]
 lines+=['',f'### {s["name"]}：{s["previous_close"]:.2f} / {s["next_limit_reference"]:.2f}',s['scenario']+'。',
 '09:25：'+s['auction_amount']+'。',
 '09:30：'+s['after_open_0930']+'。',
 '09:35后：'+s['after_open_0935']+'。',
 '取消积极执行的情景：'+s['cancel']+'。']
lines+=['',cards['common'],'',
'本批未完成逐股全部实时硬门，不能写公告无硬雷或抗核通过；也不因行业占比或主营弱关联机械否决市场叙事。仓位沿原账户/纸面模式，不新增交易许可。若22日成交，普通A股最早正常卖出为23日，22日盘中失效不能假定当天卖出；排队、确认、成交、费用、卖不出与未平仓分别记录。执行预案取消不能删除本次收盘预测或弱行情日期。','',
'## 双方如何形成一致','',
'Codex原首版总体前三为海洋王/大业/百洋，首板第一为海量。Gemini首版把用户点名股和顺灏放入前三，且部分报价错误；用户要求重做后，其自行改为百洋/出版/海量。随后核对原始行情、保留真实反证：Gemini接受一价可延续与可成交分离，Codex接受机器人中位失败对大业的压制，最终收敛海洋王/百洋/海量与两只观察。','',
'共同讨论不等于两份独立投票，更不等于已经验证准确。首版已知道用户提示，重做不能冒称消除记忆；后续共同修订也已看过BASE。原始聊天、各轮来源/收据和实际Gemini确认独立保存。Gemini在用户重做时更新了本次ROUND1草稿，原首读哈希在BEFORE_BASE_BOTH_ROUND1_HASHES.json、完整首读文本在本聊天工具记录；重做稿另存peer_snapshots，不能声称所有草稿字节从未改变。旧生产预测/账本不改。','',
'## 与原BASE的真实分歧','',
'仅双方首版留档后，对原BASE做16只连板补充来源推理，0次拟合。分数未校准，80.5分不是80.5%概率；本次首板相对排序也不是已经上线的新数值模型。','',
'| 股票 | 原BASE分数/方向 | 共同结论 |','|---|---|---|']
for c in ['002724','002696','603278','002313']:
 b=baseby[c];lines.append(f'| {by[c]["name"]} | {b["score_display"]:.1f} / {b["BASE_direction"]} | {paths[c]["call"]} |')
lines+=['','百洋是关键分歧：BASE30.8分偏不晋级，双方因当期群体修复与相对地位给正向情景，保留冲突并等待完整结果；不能用共识掩盖分歧。三峡76.8分仅是补充源诊断，其严格连续资格仍有源缺口；未纳本次重点。','',
'## 完整池、来源范围与到期验证','',
'- 67只保留：44涨停、18炸板、5跌停；完整逐股Codex方向及共同3+2摘要在同目录CSV，未知不填0。Gemini明确确认共同重点结论，不冒称其逐股给67只相同方向。',
'- 原流通市值上限93.2685989652亿元；44涨停中8只超门仅退出原小盘筛选候选，不删除市场情绪样本。原薄换手代理规则不变，本日无新增代理排除；三板组资金身份缺乏合格时点证据，不冒称全部排查干净。',
'- 供应商报16连板，其中15只相邻官方连续价检通过，三峡19日官方参考行情缺口单列；不以最近交易日冒充前一官方交易日。天洋的换手腾讯/替代页15.1818%与东财16.1509%口径差保留，不拼接成精确资金结论。',
'- 原始返回101份哈希及95只日线重解析核对（实际只数以DATA_AND_SCOPE_VERIFICATION.json为准），全部67只今日价格一致；不认证真实Level2队列或特定资金行为。',
'- 股吧限定三篇：两篇可读原帖正文分别为[杭城小七哥10/20](https://m.tgb.cn/a/2w31Z4KV77h)、[延边刺客10/15](https://m.tgb.cn/a/2vUNIn9W0UH)，只提取锚点、扩散与失败的假设；自报成功率/席位身份不当真值。第三篇[作手阿一10/20](https://www.jiuyangongshe.com/a/3y8n00z1uzb)本次原始HTML无合格正文，搜索摘要单列；截至本次未取得合格21日盘后原帖，不冒称已读。评论不继承正文时间。',
'- 22日结算次日收盘涨停Top1/Top3及完整67方向；23/26/28日分别检验累计、逐日上涨及连续涨停。未来竞价、成交与买后收益另评；不以竞价回填21日收盘预测，不删失败日期、弱行情或未知分母。',
'- 本次0fit、0新订单、0生产登记变动；训练60%目标尚未达到，模型研究设计继续保持原待复核状态。','',
'证据索引：JOINT_FINAL_PREDICTION_BEFORE_NEXT_SESSION.json、gemini/JOINT_CONCLUSION_ACK.json、facts/DATA_AND_SCOPE_VERIFICATION.json、完整67只_独立方向与共同摘要及BASE对照.csv。']
text='\n'.join(lines)+'\n'
text=text.replace('101份哈希及95只日线重解析',f'{dataqa["verified_raw_files"]}份哈希及{dataqa["daily_raw_replays"]}只日线重解析')
report=OUT/'10月21日盘面复盘与22日共同预测.md';report.write_text(text,encoding='utf-8')
write(OUT/'COMPLETION_JOINT_REVIEW.json',dict(formed_at=now(),status='COMPLETE_ACTUAL_PEER_ACK_RECEIVED',report=str(report),report_sha256=sha(report),prediction_sha256=sha(OUT/'JOINT_FINAL_PREDICTION_BEFORE_NEXT_SESSION.json'),peer_ack_sha256=sha(ack_path),own_round1_unchanged=True,full_current67_retained=True,top3=codes[:3],observations=codes[3:],first_board_top1='603138',new_fits=0,new_orders=0,formal_new_predictions=0,model_60pct_achieved=False))
print(json.dumps({'report':str(report),'top3':codes[:3],'observations':codes[3:],'full_rows':len(full),'report_characters':len(text),'peer_ack_sha256':sha(ack_path)},ensure_ascii=False))
