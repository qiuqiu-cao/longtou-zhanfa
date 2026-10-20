"""Literal author judgments, saved before reading any current BASE scores."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import pandas as pd
OUT=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def h(c,e,l,path):return dict(cumulative_up_call=c,every_day_up_call=e,consecutive_limit_ups_call=l,path=path)

def main():
    facts=read(OUT/'facts/qualified_full_facts_SCORES_MASKED.json')
    # Author decisions; no model output, score, learned threshold or unseen result.
    authored={
    '603278':('YES',1,'机器人盘面情绪映射的3板前排候选，非已认证产业龙头',
      '昨天分歧板今天转3板一价，成交量为19日0.176倍；与福莱、宁波、荣泰等2板及多只机器人首板形成盘面层级，强于仅剩单票的叙事。机器人当前强度比昨天汽车组整体更有扩散支持。',
      '一价没有经历今日开板压力，不能证明明日承接；21日署名但20日晚已公开的风险公告提示亏损及非理性炒作，并未认证新的机器人商业进展。不能用页面机器人标签替代公司新订单。',
      '21日竞价低于11.99且开盘未能修复；或打开涨停后反复回封失败、福莱/宁波/荣泰同步跌回20日收盘，取消主动接力。若继续一字只作为预测与情绪观察，不假定买到。',
      {'2':h('YES','UNKNOWN','UNKNOWN','至22日累计偏上，21日争4板；22日更可能首次充分换手，连续两日封板不作主承诺。'),'3':h('YES','NO','NO','至23日累计偏上，但预计4至5板附近有分歧，允许收阴与冲高回落。'),'5':h('UNKNOWN','NO','NO','至27日是否高于11.99依赖换手后同组延续，预期至少一次明显分歧，五日累计方向未定。')}),
    '605488':('YES',2,'机器人材料/传感器叙事中的2板前排假设，业务催化资格保留',
      '非一价2板，09:30:11后供应商未报告再开板；换手1.6063%、量0.573倍、振幅4.76%，比荣泰/科森的最后封板更早、当日价格分歧较小。与大业可共同晋级而非二选一。',
      '很低换手未必有足够卖压测试；本次所读公司异动公告仅确认经营无重大变化并提示炒作，未确认新增传感器订单。页面柔性传感器叙事只能作为待核交易解释。',
      '21日跌回30.99后持续不能收回，或大业开板大跌、宁波/荣泰同时弱化，取消强延续；高开超过7%后放量跌破开盘均价不追。',
      {'2':h('YES','UNKNOWN','UNKNOWN','至22日累计偏上，21日争3板、22日预期补量换手；不承诺每天上涨或双日连板。'),'3':h('YES','NO','NO','至23日先延续后震荡，累计仍偏上，若机器人梯队缩减则改为区间。'),'5':h('YES','NO','NO','至27日有条件累计走强；须守住30.99附近并有同组续板，不能只有自身缩量独撑。')}),
    '003029':('YES',3,'独立4板高度锚，AI安全/低空安全情绪候选',
      '东方嘉盛断板后成为当前唯一4板；20日换手24.16%、量5.885倍后09:57:30再封，收26.61，形态从缩量加速转为真实已发生交换。高度支持与换手风险同时存在，不以单一量比否决。',
      '11.24亿元成交与24.16%换手也可能是兑现而非主动承接；20日晚取得21日署名公告强调热点业务未规模化、半年亏损，股价脱离基本面，不能包装为业绩主升。缺完整逐笔路径及真实排队。',
      '21日竞价无溢价并持续跌破26.61、冲板反复不能封，或大业/福莱等同周期集体转弱，取消5板主动预期。高开回落破均价不因唯一高度硬接。',
      {'2':h('UNKNOWN','NO','NO','21日争5板，但22日高位补量/兑现压力大，至22日累计方向保留未知。'),'3':h('UNKNOWN','NO','NO','至23日偏高位分歧，不预测连续上涨，若21日未封且失守26.61则主升路径失效。'),'5':h('UNKNOWN','NO','NO','至27日预期大幅震荡，能否高于26.61待换手检验；不押五日必涨。')}),
    '600267':('YES',4,'有发行人经营数据支持的医药业绩2板候选',
      '2板日内振幅2.24%、换手4.92%，从一价首板放量5.919倍后09:33:37封住；同日昂利康、东亚等医药首板共存。公司预计前三季归母11.5亿元、扣非9.8亿元，分别增149.46%与99.59%，是已读原文的业绩线索。',
      '业绩为未经审计初步核算，营收总口径下降且有股权处置收益，不能把归母增幅全部当经常性增长；医药不是机器人同一主线。早封和窄振幅仍不等于真实买盘队列。',
      '21日不能维持13.27附近并反抽失败，或昂利康/东亚等医药同步回落，取消短线延续；若不涨停但保持正收益，涨停预测仍按失败独立记账。',
      {'2':h('YES','UNKNOWN','NO','至22日累计偏上，21日争3板，22日更偏换手延续而非锁死连板。'),'3':h('YES','NO','NO','至23日业绩线震荡上行倾向，允许中途收阴。'),'5':h('YES','NO','NO','至27日累计偏上，靠经营数据与量价延续检验，不预测五日逐日上涨。')}),
    '600178':('YES',5,'低空/动力情绪中3板存活候选，收入兑现逻辑否决',
      '与大业同为3板一价，换手1.6243%、量0.484倍；在保隆/万丰/文灿未晋级时仍保持形态，足以保留相对晋级候选，不把同组失败机械移植。',
      '20日晚公告确认2026年1至9月飞行汽车业务尚未形成收入、半年由盈转亏，不能以订单业绩兑现背书。低空原同组多只失败且一价补量未测试，优先级低于大业。',
      '21日打开后跌破12.41并不能修复、同组无新增晋级，或反复封不住，取消主动接力；只观察确认，不根据一价假定可成交。',
      {'2':h('UNKNOWN','NO','NO','21日仍有冲4板倾向，22日补量风险更大，至22日累计方向未知。'),'3':h('UNKNOWN','NO','NO','至23日倾向分歧震荡，须与机器人分支区分，不以J/名称联想续强。'),'5':h('UNKNOWN','NO','NO','至27日高度与实际业务脱节风险高，不预测连续上涨或必然累计上行。')}),
    '002929':('NO',None,'算力服务2板分歧候选','合同公告索引存在，20日换手15.45%、量6.654倍后最终封板，确有晋级事实。','14:31末封、振幅12.09%、4次报告开板；同位福莱/海正/慕思更早更窄，当前相对承接确定性较低。合同全文本轮超时，不能把未核订单利润写成确定支持，亦不以缺文单独判负。','若21日显著弱转强、在完整同位前排稳定封板，偏断板被证伪；低开且51.04不能收回则延续负向。',{}),
    '002164':('YES',None,'机器人减速器叙事换手2板竞争者','量1.239倍、换手12.99%、09:53末封，机器人当期梯队增强；实际股东增持文件已读，不能把增持当订单。','换手和7.22%振幅高于福莱，3次报告开板，可能被更早/更低换手同组分流；所谓批量供货尚未本轮原文核实。','21日跌回16.16不能修复或福莱/荣泰同步走弱，取消续板预期。',{}),
    '603119':('UNKNOWN',None,'机器人/汽车2板竞争者','2板换手3.24%、10:04末封，量1.292倍，仍有同组共振支持。','9.74%振幅较福莱更大；晚间公告未认证新热点催化，不从与特斯拉标签相邻推断特定供货。','若21日弱于福莱/宁波并跌回50.46不修复，观察失效；若先转强可追加新判断，原弃判保留。',{}),
    '603626':('NO',None,'结构件2板，公告事件需单独下调','今日缩量0.945倍、换手6.63%、10:01末封具备上涨事实。','20日晚原文披露对外投资暨购买资产暂缓推进、存在终止风险，不能继续把收购定增叙事当确定利好；与福莱/荣泰相比新增反证更具体。','若21日消化公告后反而持续转强并封板，负向预测被证伪，仍不回写原NO。',{}),
    '001323':('UNKNOWN',None,'智能睡眠/端侧智能映射2板，行业不等同机器人','量1.784倍、换手1.87%、振幅2.72%、09:30早封，真实价量支持独立保留。','半年净利下降48.13%，晚间公告无新重大事项；床垫智能化不能当同一机器人本体业务，不能仅因低换手判YES。','21日失守22.23且同位2板多数不续，观察失效；转强时另追加事前修正。',{}),
    '600860':('UNKNOWN',None,'装备制造3板，交易与风险分栏','3板换手11.49%、09:44末封，较长春更早；与大业可共同晋级，并非必互相淘汰。','控股子公司北洋天青涉嫌重组信息披露问题被立案，主营半年亏损；机器集成叙事不是新增订单。列风险回避，不因风险存在声称必断板。','若21日新增风险扩大或9.57失守不修复，继续回避；预测结果仍完整结算。',{}),
    '002724':('UNKNOWN',None,'专业照明3板，AI照明/名称情绪连接待检验','换手10.85%、量9.243倍来自19日极低基数；09:38末封，价格分歧小于长春。九阳/百洋当期反包可作为名称叙事背景，不能认证补涨因果。','公告确认AI智慧照明收入有限；羊/Y线并非业务共同体，不能继承九阳父锚身份；完整同位有大业/东安/京城/长春竞争。','21日跌回9.26不修复且同位继续强，观察失效；如转强则单独追加，不把旧未知改为事前YES。',{}),
    '600148':('NO',None,'军工/汽车3板尾盘分歧候选','14:56:52回封成功，必须按收盘3板，不采用午间炸板状态；今天真实换手22%。','量14.52倍来自一价低基数，不能只凭倍数判死；但同位大业/东安全天一价、京城/海洋更早封，尾盘才稳定显示当前相对分歧较大。','若21日弱转强且先于完整同位封板，偏断板被证伪；高开回落失守19.70则延续弱假设。',{})}
    first={
    '002596':('NO','尾段14:55才末封、量4.15倍、非当前主要连板分支，强延续证据弱；若21日转强封板将证伪。'),
    '603893':('UNKNOWN','端侧AI首板但38.34亿元成交、5次开板、14:46末封，趋势持续与次日连板分别判断。'),
    '605177':('UNKNOWN','医药首板，14:44末封且振幅10.82%，需与海正/昂利康完整比较；未覆盖不等于无催化。'),
    '603636':('NO','14:13后发首板、振幅12.00%，AI短剧/政务页面标签尚未原文认证，难优于更早模组集群；转强可证伪。'),
    '603366':('UNKNOWN','储能/热泵独立首板、14:11末封，不能把不同设备业务并入机器人主线；方向保留未知。'),
    '002696':('UNKNOWN','旧周期反包，换手28.67%、量1.026倍，并非新的低位补涨启动；九阳同日回归支持情绪线索但不认证资金因果。'),
    '002076':('UNKNOWN','量子/安防标签与吉大语义相邻，9次开板至13:35末封，不能称安全线主动核心。'),
    '603341':('UNKNOWN','端侧AI/终端首板可与美格/移远比较，6次开板至13:26，板块共振支持与筹码分歧并存。'),
    '603683':('UNKNOWN','材料/柔性传感叙事首板、13:13封，不能把页面标签直接等同福莱已获新订单，保留逐股身份缺口。'),
    '002940':('UNKNOWN','医药13:06首板，海正已有2板身位；自身也可能共同晋级，缺乏足够相对证据判定。'),
    '603400':('NO','13:00后发设备首板、换手11.55%，完整机器人竞争中缺乏早封和低分歧优势；若弱转强将证伪。'),
    '002403':('YES','机器人叙事首板，与大业及多只2板共振，量1.899倍、11:26末封；振幅11.24%和较晚封仍是反证，非前5。'),
    '603657':('UNKNOWN','家庭清洁设备首板，11:26封、量0.847倍；家庭机器人与工业本体不能混为直接梯队。'),
    '603696':('NO','食品首板午前11:21封，量1.177倍、无当期连续食品梯队确认，明日封板相对倾向偏弱；不宣称没有催化。'),
    '001288':('UNKNOWN','输送设备/矿山机器人叙事，首板量1.253倍；业绩预告原文已归档，未以泛机器人身份自动转正。'),
    '001388':('UNKNOWN','电力巡检/机器人叙事首板，量5.29倍且11:03末封；可能题材扩散，也可能短线兑现，保留未知。'),
    '001283':('UNKNOWN','消费电池与AI终端/机器人配套叙事，3次开板、10:58末封；不把电池所有需求认证直接本体业务。'),
    '603068':('UNKNOWN','端侧无线芯片首板，量1.327倍、10:33封；与瑞芯微同属芯片子组，不能由模组封板推定其必连板。'),
    '605389':('UNKNOWN','液压/AGV减速器映射首板、10:31封且换手0.68%，低换手不独立认证承接；保留同组比较。'),
    '603407':('UNKNOWN','材料次新首板，与机器人/端侧AI不是同一产品链；10:20封、换手10.76%，无法确认次日连板优势。'),
    '002881':('YES','端侧AI模组首板，与日海/移远/芯片集群同时封板，量1.40倍、换手9.16%、10:19末封；具体新订单未原文核实且非前5。'),
    '002242':('UNKNOWN','旧周期九阳反包、量0.774倍、5次开板，百洋同步可支持名称情绪重现；不倒填新补涨父子真值。'),
    '603236':('YES','端侧AI模组首板、10:09末封，相关同组多只封板而非孤立名称；21.92亿元成交要求持续增量，已读回购公告不是新订单证明，非前5。'),
    '001239':('UNKNOWN','盾构/风电首板，与机器人为不同业务，10:01封但量2.56倍；不从出口限制题材直接认证订单。'),
    '002313':('YES','端侧模组首板、09:41末封、换手4.17%、量1.812倍，较同组早且分歧较小；已读股权转让进展，未以AI标签冒称新业绩，非前5。'),
    '002347':('UNKNOWN','工业机器人装备首板、09:39封、量2.017倍；有板块共振但自身相对主动身份未核，未知不自动负向。'),
    '603386':('UNKNOWN','PCB兼机器人映射首板，09:36封、量0.625倍；原PCB高标周期和新机器人扩散需区分，不把所有概念累加。'),
    '002088':('UNKNOWN','耐火隔热材料首板，量3.86倍，控制权/电池叙事未据本轮原文认证；不能并作机器人核心。'),
    '002181':('UNKNOWN','传媒反弹首板、09:30早封，旧传媒多数仍分歧；早封不能独立认证新周期，但也不直接判NO。'),
    '002479':('UNKNOWN','热电/回收业绩分支，量0.747倍、09:30封，属于独立首板观察而非机器人直接同位。'),
    '600293':('UNKNOWN','重组复牌首板，20日日线存在但19日停牌无相邻参考，原公司重组/复牌公告已归档；保留报价与价格资格缺口，不作连板候选。'),
    '600661':('UNKNOWN','教育一价首板，换手5.35%、量1.57倍，不能机械因一价转正或因非主线转负，等待完整教育扩散。'),
    '002835':('UNKNOWN','AI安防一价首板，换手1.88%、量0.940倍；与吉大安全主题相邻但不是同一产品，真实队列与新催化未知。')}
    limits=[r for r in facts['rows'] if r['kind']=='limit_up'];assert len(limits)==46
    assert set(authored)|set(first)=={r['code'] for r in limits}
    rows=[]
    for r in limits:
        c=r['code']
        if c in authored:call,rank,role,support,counter,cancel,hs=authored[c]
        else:
            call,support=first[c];rank=None;role='完整首板池独立观察';counter='当期行业/交易叙事线索与公告认证分栏；首板可同时晋级或共同失败，未核内容不自动当作不存在。'
            cancel=f"21日跌回20日收盘附近不能修复且同组转弱，则取消主动观察；若先强于同组封板可证伪原偏负或弃判，保留原稿。";hs={}
        for hkey in ['2','3','5']:
            if hkey not in hs:hs[hkey]=h('UNKNOWN','UNKNOWN','UNKNOWN','非重点多日承诺；保留完整候选与到期价格，不继承前5路径。')
        peers=[p['code'] for p in limits if p['boards']==r['boards'] and p['code']!=c]
        rows.append(dict(code=c,name=r['name'],boards=r['boards'],signal_close=r.get('close'),next_close_limit_up_call=call,next_intraday_touch_call='YES' if call=='YES' else 'UNKNOWN',priority_rank=rank,role_hypothesis=role,support=support,counter_evidence=counter,cancel_conditions=cancel,complete_same_height_peers=peers,horizons=hs,probability='未验证',tradeability='竞价/开盘确认待发生；一价不假定可买；Level2真实队列/完整订单流不可核。T+1当日买入不能当天卖出止损。'))
    forum=dict(formed_at=now(),preferred_current20_body_qualified=0,actual_read_relevant19_bodies=2,
       scope='限定四作者20日搜索未获合格当前正文；另读两篇19日相关原文。正文/评论分开，未读评论不继承正文时间。',
       assessments=[dict(author='率土之滨非魔也',url='https://www.tgb.cn/a/2w1r0zXNKSD',published='2026-10-19 16:30+08:00',kind='19日正文',accept='完整同位和不同路径对照',reject='封单厚/一字即最可能晋级的推断；保隆和万丰20日失败形成反证；其全市场46与本项目39主板口径分开，不承接其未核资金净流入'),dict(author='主升龙头空空龙',url='https://www.tgb.cn/a/2w1lHvU4cKB',published='2026-10-19 15:15+08:00',kind='19日正文，不取后续评论当原判断',accept='先看完整竞价/层级，再区分低空与AI分支',reject='千亿资金识别/谁带谁未独立认证；19日竞价大封单不能推出20日保隆必续；也不将19日个别医药失败整体移植海正20日。')])
    sourcepaths=[OUT/'facts/qualified_full_facts_SCORES_MASKED.json',OUT/'features/current29_BEFORE_MODEL.json',OUT/'context/capture_manifest.json',OUT/'context/evening_fulltext_manifest.json']
    for folder in [OUT/'facts/notices',OUT/'context/notices']:
        sourcepaths+=list(folder.glob('*.txt'))
    sourcepaths += [OUT/'context/forum_19_rates.txt',OUT/'context/forum_19_counter.txt',OUT/'context/market_thepaper.txt']
    record=dict(formed_at=now(),signal_date='2026-10-20',target_date='2026-10-21',horizon_dates=facts['horizon_dates'],author='Codex live independent author',model_exposure='CURRENT_MODEL_OUTPUT_NOT_REVIEWED_SELF_ATTESTED',scope='COMPLETE_RETURNED_MAINBOARD_NON_ST_CLOSING_UNION',full_candidates=46,full_relay=13,rows=rows,forum_preread=forum,
      mainline=dict(primary='机器人/物理AI盘面延续，端侧AI向模组/芯片扩散；均为当期叙事/价格集群而非全业务认证',secondary='医药业绩分支',negative='低空汽车内部分化，旧传媒/染料不整体认定修复',stage='修复中的分化；明日主预期先分歧后局部修复，不押普涨加速',next_day_scenarios=dict(base='21日高开或冲高后分歧，机器人与端侧AI保留部分续强，指数震荡。',strong='大业延续、吉大换手后再封，福莱/宁波/荣泰至少两只续强，机器人扩散并有成交配合。',weak='大业/吉大开板难回封，福莱等2板批量失守今日收盘，端侧首板高开回落，停止主动追价。')),
      source_bindings={str(p):sha(p) for p in sourcepaths},new_fits=0,old_models_changed=False,new_formal_predictions=0,registration_status='LOCAL_INDEPENDENT_BEFORE_MODEL; original formal source gate not passed, not a formal ledger entry')
    write(OUT/'independent_full46_BEFORE_MODEL.json',record)
    write(OUT/'independent_freeze_receipt_BEFORE_MODEL.json',dict(formed_at=now(),path=str(OUT/'independent_full46_BEFORE_MODEL.json'),sha256=sha(OUT/'independent_full46_BEFORE_MODEL.json'),full_rows=46,models_displayed=False,target_session_not_started=True))
    pd.DataFrame(rows).to_csv(OUT/'完整46只_独立方向与2_3_5日原预测.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(dict(full_rows=46,relay=13,top5=[(r['code'],r['name'],r['priority_rank']) for r in rows if r['priority_rank']],directions={k:sum(r['next_close_limit_up_call']==k for r in rows) for k in ['YES','NO','UNKNOWN']},models_seen=False),ensure_ascii=False))
if __name__=='__main__':main()
