from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
import json,hashlib,copy
O=Path(__file__).resolve().parent;R=O.parents[1]
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def new(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
f=read(O/'facts/RECONCILED_COMPLETE_FACTS_SCORES_MASKED.json');daily=read(O/'facts/normalized_daily.json');raw={r['c']:r for r in read(O/'facts/raw/2026-10-27_failed_board.json')['data']['pool']}
for r in f['rows']:
 if r['code'] in ['600826','600865']:
  z=raw[r['code']];r.update(reported_float_cap_cny=z['ltsz'],reported_total_cap_cny=z['tshare'],primary_amount_cny=z['amount'],original_cap_screen=z['ltsz']<=9326859896.52,thin_proxy_excluded=False,screen_status='WITHIN_ORIGINAL_SCREEN_CLOSING_PRICE_CONFIRMED_POOL_CONFLICT',reported_open_count=z['zbc'])
f['source_bindings'][str(O/'minutes/additional_capture_receipt.json')]=sha(O/'minutes/additional_capture_receipt.json');f['formed_at']=now()
new(O/'facts/RESOLVED_FACTS_v2_SCORES_MASKED.json',f)
# Scope corrections are projections with original source rows retained, not overwrites of raw/provider facts.
for name in ['prepare_features.py','score_original_BASE.py','audit_firstboard_base_inputs.py']:
 p=O/name;s=p.read_text(encoding='utf-8-sig');s=s.replace("OUT/'facts/qualified_full_facts_SCORES_MASKED.json'","OUT/'facts/RESOLVED_FACTS_v2_SCORES_MASKED.json'")
 if name=='score_original_BASE.py':s=s.replace("prep['full_relay']==14","prep['full_relay']==12").replace('complete14=True,independent_full90_unchanged=True','complete12=True,independent_full83_unchanged=True')
 p.write_text(s,encoding='utf-8')
context='''# 2026-10-27有限股吧预读、事实与推理分栏

上海17:30左右复核作者博客主页，最新主帖仍显示1026；未取得合格1027新主帖，不把搜索到的2025年1027或盘前文章冒充今天盘后。复用并重新取得两篇10月26日原帖正文：延边刺客《1026》 https://m.tgb.cn/a/2wd3CQePUzU ，页面26日16:34，27日09:30:27 UTC取得；率土之滨非魔也《10-26涨停55只…》 https://m.tgb.cn/a/2wdbgSIcLdH ，页面26日18:21、正文自写16:05生成，27日09:30:28 UTC取得。仅正文，未读评论/回复/图片。原文范围、哈希见limited_context_capture.json及昨日完整原文。

采用：将旧周期兑现、同位淘汰、消费/应用/名称分支的实际响应分开检验。否决：主控账户/量化搬家等资金身份；只要多伦晋级便认证全线主升；同字自动亲缘；一字封单等于锁仓和可成交；13点封板叫尾盘。观点不因作者名气或粉丝数成为真值。

今天验证：消费青岛/桂发祥/阳光晋级且零售有新首板，支持消费扩散；应用延华高开8.61%后收跌约3%，新华传媒和慈文偏弱，否决泛应用普遍持续；旧机器人八只二板无一晋级、正裕/林州跌停，却有三只四板存活，说明高位抱团与中位杀跌并存。海洋王反包而大西洋尾盘炸、东方海洋炸、百洋下跌，否决名称全体受益。所有关系仍为可检验假设，不认证账户资金流动。

时间轴：正裕09:35附近跌停时九百/青岛/光洋强封，说明未同跌；14:00前后1/3/5分钟华天科技偏强、新华传媒偏弱，但随后尾盘华天与指数也回落，不能把传媒下跌单因归给科技吸血。大西洋14:54后从6.12下滑至5.72，九百/青岛/襄阳没有同步打开；深创和沪指同时加速下行，存在共同风险收缩解释。多伦日线低9.66与14:49末封提示尾盘打开；分钟收盘点全9.85不能反证区间内无打开。赛福全天OHLC同价但供应商末封15:00，保留队列/时序冲突，不编造换手过程。

格隆汇哥吉拉《涨停潮来了，谁在点火》仅作当前零售叙事来源。页面只有“7小时前”，准确发布时间未知；混合26日食品和27日零售，不采用其宏观数值、估值安全或确定资金转移作为本次事实。

公告：青岛、九百、多伦、襄阳、赛福及阳光等全文已取得。主营机器人占比极小不单独否决情绪；常规风险提示不直接当停牌。深赛格低换手一字且子公司担保代偿进展，暂不作为可接力首选。东贝报告业绩转亏为已知基本面信息，但不据此否定其当日价格认可；本次优先比较更清晰的群体地位。江顺今晚Q3取得为盘后信息，不能倒填今天09:35封板原因。茂业最新季度全文单次请求失败，保留风险核验缺口，条件关注需盘前补核；不以索引成功冒称全文已清雷。首板所有输入不足，不输出BASE。

池口径修正：兰生、百大在两供应商炸板池，但同池收盘价、本日未复权日线、15:00分钟与报价均为涨停参考价9.04/12.62。按收盘价格事件补列兰生二板、百大首板，原炸板身份、封板时序和队列未知保留。原36收盘池加2价格确认，成为38只价格涨停、24炸板、21跌停，83唯一股票。不是宣称原供应商池已无冲突，也未修改昨天预测或原始响应。
'''
(O/'context/论坛预读与事实取舍.md').write_text(context,encoding='utf-8')
# Explicit author-written complete-pool calls, with observed support and counter-evidence.
v={
'600838':('YES','唯一五板，09:30封稳，7.49%换手且消费二板与新零售首板同时扩散','五板后加速兑现风险，未经历大分歧；指数下跌时的抱团未必能穿越下一日'),
'000678':('YES','四板09:36稳定、20.57%换手，在正裕/林州跌停时保持机器人前排','连续高换手可能拥挤也可能释放抛压，缺真实持仓证据；和多伦/赛福可共同强也可共跌'),
'603528':('UNKNOWN','四板且换手11.65%，大部分时间封住，仍有相对地位','末封14:49、日线低9.66说明尾盘真实分歧；八只二板全灭及应用弱使次日溢价并不等同一字强'),
'603028':('YES','四板一价、换手10.31%已不命中昨日低换手薄流动性规则','今日数字筛选通过不回填昨日；一价不等于可买，末封15:00与页面09:25冲突，队列未知'),
'001219':('YES','食品二板、09:38稳定、换手7.77%，在消费扩散中保持先发','开盘涨停后低14.51，仍有分歧；补量后不代表未来每天都能承接'),
'002820':('YES','食品二板10:11封住且之后稳定，流通22.52亿换手17.86%','首轮曾跌破昨收，日内摆动大；比青岛晚，明日再加速可能兑现'),
'001318':('UNKNOWN','低开后转强二板，食品/阳叙事双侧可观察','末封14:20、4次打开，晚于青岛桂发；常规异动附业绩下降不直接否决，但情绪地位仍待确认'),
'002093':('UNKNOWN','二板09:39稳定，AI应用/通信有个股独立性','流通79.89亿，延华/传媒弱说明应用未形成普遍接力；不能因跨题材名称认证新龙'),
'002708':('UNKNOWN','二板早封，机器人汽车/洋叙事仍有群体','流通87.76亿近原上限、换手仅3.87%加速；此前活跃不等于早期新龙'),
'001331':('UNKNOWN','能源/机器人二板且13:34末封，存活显示局部认可','流通110.10亿超原范围，不能用大市值提高首板路线命中'),
'605333':('UNKNOWN','二板、流通92.39亿在上限内，尾盘回封','换手4.18%，14:50末封且接近市值上限，强度不及早封同位'),
'600826':('UNKNOWN','收盘价9.04两类价格源确认，按价格事件为二板','9次打开、午后修复，供应商仍归炸板；来源资格和封板时间未合格，不列买入主选'),
'600865':('UNKNOWN','零售午后上涨，收12.62由日线/15:00点和同池价格确认','原供应商炸板、6次打开且换手19.16%，不把收盘价达涨停等同稳定早封'),
'600828':('YES','零售首板09:40封住，九百/青岛及广百形成群体认可','换手仅1.57%、流通77.39亿偏大但未超原上限；缺首次补量验证，季度全文请求失败，需补核'),
'603758':('YES','09:33封稳，低开转强，流通46.47亿；同制造方向有完整前排与新股响应','换手2.95%需补量，传感器叙事不自动认证龙头；早已公开Q3利润下降不能拿来称业绩推动'),
'001400':('YES','昨天炸板今天09:35反包且稳定，流通25.86亿、换手5.84%，液冷/制造方向相对独立','高价80.28和反包后的兑现压力；盘后Q3不是今天09:35已知催化，独立不等于必能穿越'),
'601956':('UNKNOWN','09:40稳定、低价流通34.44亿，换手4.15%，机器人电机/制冷可观察','Q3已披露转亏，但不否决当日市场认可；群体地位和秦安/江顺相比不足，不能仅低价升为龙头'),
'601566':('UNKNOWN','09:31首封、消费衣着与名称叙事可比较','13:12才末封，行业不等于食品，不能因九字认证九百父子关系'),
'002187':('UNKNOWN','零售09:43首封、流通41.49亿且换手5.12%','13:28末封，比茂业稳定慢，消费同线可共同晋级但目前先发地位稍弱'),
'000058':('UNKNOWN','09:25一价首板，电子市场/半导体检测与零售共现','仅0.76%换手，无实质分歧承接；担保代偿进展需看，数值范围内不等于可接一字'),
'603376':('UNKNOWN','汽车电子09:59稳定，流通8.42亿换手11.04%','极小盘需谨慎实际流动性，2次打开且先发稍晚；不认证特定资金操盘'),
'603370':('UNKNOWN','机器人铁芯首板11:05稳定，流通26.02亿、换手12.71%','相对较晚且旧机器人中位亏钱，不能把同业概念直接当接力认可'),
'603037':('UNKNOWN','机器人/汽车小盘13:01一次封住，流通31.59亿','午后启动比早封弱；收购叙事不代替市场地位'),
'002823':('UNKNOWN','汽车连接器/机器人首板，流通30.09亿换手10.28%','13:20末封，辨识度弱于早盘制造同位'),
'002284':('UNKNOWN','09:30首封、换手6.79%，公开受让股权与汽车方向可核','4次打开14:49才稳定，与大盘末段同波动，封板质量偏弱'),
'600626':('UNKNOWN','上海汽车内饰午后首板，流通52.04亿','5次打开14:46末封，地域共现不证明九百助攻'),
'600785':('UNKNOWN','昨日零售炸板今日尾盘反包，换手19.86%','14:55末封、拥挤，不能替代茂业早盘先发，解套卖压是假设需承接证据'),
'601086':('UNKNOWN','零售14:52单封，流通84.05亿','尾盘确认且接近原市值上限，群体扩散后排不直接升龙头'),
'600715':('NO','AI漫剧低价首板，流通53.53亿有交易','09:41首封至14:56反复5次，新华/慈文弱，不将低价弱板视为情绪先发'),
'002724':('UNKNOWN','旧六板断后反包，流通74.15亿，显示旧高标尚有修复','换手28.85%、6次打开至14:55，百洋/东方/大西洋未跟，不能认证洋线已统一修复'),
'002249':('UNKNOWN','电机/洋叙事首板，机器人仍有高位认可','流通144.18亿超范围、14:55末封，不纳入小盘主选'),
'603565':('UNKNOWN','物流业绩/激励首板09:31启动、13:34末封','流通264.19亿超原范围，独立容量对照非情绪龙头主选'),
'002623':('UNKNOWN','玻璃基板/光伏午后13:19封住，流通36.18亿','非早盘先发且板块缺清晰梯队，盘后Q3不倒填日内催化'),
'000888':('UNKNOWN','旅游消费10:29首封13:03末封，流通53.22亿','旅游不是食品零售自动助攻，消费板块内部需各自确认'),
'002205':('UNKNOWN','管道小流通21.19亿10:49单封','同线缺梯队、独立一日脉冲尚难区分，不仅凭小盘入前三'),
'600303':('UNKNOWN','跌停后反包在首板池，流通21.07亿但缺当日日线','价格/流动性及连续性资格未知，保留完整分母不列主选'),
'603813':('UNKNOWN','汽车物流10:34单封，流通25.41亿','换手仅1.89%，同方向上升不代表本股承接已充分，不优于早封者'),
'002011':('UNKNOWN','温控09:30首封且稳定，容量方向有认可','流通120.69亿超范围；即使稳也不能为提高命中率进入本路线'),
}
assert len(v)==38,len(v)
failed_reasons={'002865':'光伏首板炸板，缺强同位支持','002434':'机器人汽车后排冲板未稳','003026':'半导体冲板失败，容量修复不是全体晋级','002590':'汽车后排冲板未稳','002222':'光学科技冲板失败，不能用华天强替代本股承接','002196':'电机机器人冲板失败，同类有存活也有失败','603366':'东方名称不能等同海洋王统一修复','002703':'汽车冲板未稳，低位题材分层','603489':'电机冲板未稳，需反包新判断','002639':'雪人机器人后排未稳','002086':'今日早封后14点后走弱，洋名修复非普遍，收2.39未晋级','600617':'能源独立冲板未稳','603337':'昨日二板今天炸板未三板，中位亏钱保留','002181':'传媒反抽炸板，新华/慈文弱的反证','002995':'应用冲板未稳，不泛化软件全强','001282':'制造后排未稳','600558':'日线缺失，分钟收5.72与炸板池一致，昨日首板推荐未晋级；14:54后尾盘走弱','002741':'材料科技后排未稳','603390':'汽车后排未稳','002929':'算力应用后排未稳，价格较高且放大波动','002376':'零售新北洋炸板，九百强不普遍助攻','001311':'汽车后排未稳','002300':'电缆独立冲板未稳','603121':'汽车后排未稳'}
rp=['600838','001219','000678','603528','002820','002093','603028','001318','002708','600826','605333','001331']
fp=['600828','603758','001400','601956','002187','601566','603376','000058','603370','603037','002823','002284','600626','600785','601086','002623','000888','002205','603813','002724','600865','600715','600303','002011','002249','603565']
rows=[]
for r in f['rows']:
 c=r['code'];kind=r['kind']
 if kind=='limit_up':call,support,counter=v[c]
 elif kind=='failed_board':
  call='UNKNOWN' if c in ['600558','002086','603337'] else 'NO';support='盘中触及涨停但当前未收盘封住，完整反包观察池保留';counter=failed_reasons[c]
 else:
  call='NO';support='跌停负反馈作为完整样本保留，未来反核须另日新判断';counter='当前抛压未获修复，不能凭超跌即认证次日晋级；正裕/林州/睿能/奇精/南方的同链亏钱效应须进入前排闸门'
 rows.append(dict(code=c,name=r['name'],source_kind=kind,boards=r.get('boards'),close=r.get('close'),next_close_limit_up_call=call,priority_rank=rp.index(c)+1 if c in rp else None,first_board_relative_rank=fp.index(c)+1 if c in fp else None,support=support,counter_evidence=counter,cancel_condition='模型冲突/重要风险/缺合格源者只观察，不因竞价高开直接升级；主选须09:25真实成交核验且09:35后换手回封，失守开盘区并反抽失败或同线亏钱扩大则取消；取消不自动买下名。',original_screen=r.get('screen_status'),source_qualification=r.get('qualification'),probability='UNVALIDATED'))
inputs=[O/'facts/RESOLVED_FACTS_v2_SCORES_MASKED.json',O/'minutes/capture_receipt.json',O/'minutes/additional_capture_receipt.json',O/'minutes/RELATIONSHIP_WINDOWS.json',O/'context/notices/focus_fulltext_manifest.json',O/'context/limited_context_capture.json',O/'context/论坛预读与事实取舍.md']
ind=dict(formed_at=now(),author='Codex',signal_date='2026-10-27',target_date='2026-10-28',horizon_dates={'1':'2026-10-28','2':'2026-10-29','3':'2026-10-30','4':'2026-11-02','5':'2026-11-03'},model_output_seen=False,actual_Gemini_current_opinion_obtained=False,full_rows=len(rows),information_clock='Shanghai after close 2026-10-27 evening; newly received evening filings separate, not 15:00 point-in-time signal',leader_candidate='600838',relay_top3=rp[:3],first_board_top3=fp[:3],first_board_top1=fp[0],relay_priority_definition='Independent emotion/continuation comparison, not BASE or buy permission. Final rank will separately disclose source gate, model conflicts and execution state.',market='指数午后加速下行且缩量，高位抱团与中位淘汰；消费食品/零售持续，应用弱，洋名旧高标反包和跟风失败同现。主情景先检查尾盘风险传导和核心首次分歧，不能照昨日惯性加速。',falsifiers=['今日消费可与机器人共同强，并非固定跷跷板','海洋王反包未带动全部洋字，名称只能候选关系','华天午后强于新华但尾盘同回落，竞争假设有共同卖压反例','兰生百大收盘价确认不等于原供应商封板成员或真实队列已合格','昨日仅观察九百晋级不追认其成为昨日可买主选'],source_hashes={str(p.relative_to(O)):sha(p) for p in inputs},rows=rows)
assert len(rows)==83 and len(set(r['code'] for r in rows))==83
new(O/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json',ind)
new(O/'INDEPENDENT_FREEZE_RECEIPT.json',dict(formed_at=now(),sha256=sha(O/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json'),full_rows=83,complete_relay=12,complete_first=26,complete_failed=24,complete_down=21,raw_closing_pool=36,additional_price_confirmed_pool_conflicts=2,model_output_seen=False,new_fits=0))
print(json.dumps(dict(frozen=83,independent_relay=rp[:3],independent_first=fp[:3],source_counts=f['resolved_closing_counts']),ensure_ascii=False))
