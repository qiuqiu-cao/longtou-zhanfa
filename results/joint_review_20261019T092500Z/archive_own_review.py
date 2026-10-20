"""Author's full-pool calls, saved before current model or Gemini conclusions."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
OUT=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def h(c,e,l,path):return dict(cumulative_up_call=c,every_day_up_call=e,consecutive_limit_ups_call=l,path=path)
def main():
 facts=read(OUT/'facts/qualified_current_facts_SCORES_MASKED.json')
 # These are authored judgments, not rules assigning an automatic direction.
 authored={
 '003029':('YES',1,'空间高度并列候选；低空安全/网络安全分支的情绪锚假设，不认证唯一总龙',
  '与东方同为3板，19日一价24.19、量为前一官方日0.195倍；9只2板形成低位竞争，传媒老高标失效使新分支获关注。高度与价格形态比纯名称联想更有证据。',
  '003029公告确认AI/抗量子和低空业务收入较低未规模化，半年亏损。缩量一字存在未释放筹码；若东方换手转强且它高开回落，身位并不保证晋级。',
  '20日竞价明显弱于东方并开盘后持续失守24.19、未能修复；或同组2板大面积开板跌破19日收盘，则取消主动晋级交易假设。竞价转弱即追加修正，不删除原YES。',
  {'2':h('YES','YES','YES','20日4板、21日争5板是主情景；若20日大换手未封，双日连续板情景即失效。'),'3':h('YES','NO','NO','22日前后预计首次明显换手/震荡，累计仍偏上；不预测逐日或连续涨停。'),'5':h('UNKNOWN','NO','NO','26日前经历一次以上分歧，能否高于24.19取决于换手后承接和后排续板，不押五日累计必涨。')}),
 '603197':('YES',2,'汽车/低空叙事2板前排；情绪题材种子候选',
  '2板9只完整比较中19日09:25一价，1.278亿元成交、换手1.87%；成交量为16日1.066倍，并非虚构缩量。与万丰、东安、文灿、长春形成并行晋级可能。',
  '一价不能证明能买到或实际队列；论坛所谓Roadster高压气罐没有本次发行人原文认证，不能当已获订单。其先封也不是他股让位的因果证明。',
  '20日无正溢价且开盘后跌回32.33下方不能修复；或万丰/东安/文灿同步走弱、机器人首板集群无后续，则取消强延续执行。',
  {'2':h('YES','YES','YES','20日争3板、21日争4板，届时有开板换手压力；累计偏上，二连涨停是有风险的主情景。'),'3':h('YES','NO','NO','22日预期分歧换手，累计高于32.33但未必当天上涨。'),'5':h('YES','NO','NO','26日累计偏上，路径为先延续后震荡；若只靠未证实订单叙事则撤销持续主升判断。')}),
 '002085':('YES',3,'低空/汽车容量竞争者；换手核心候选',
  '19日09:31封板，日振幅2.83%、换手4.82%、11.114亿元，量为前一日1.55倍；相对东方/大业的14%以上振幅，价格推进更平稳。与保隆可以同时晋级，不强设二选一。',
  '较大成交规模意味着持续增量要求更高；09:31早封和公开0次开板不证明午后真实买盘队列。若市场机器人纯概念后发走强，低空容量可能落后。',
  '20日开盘失守10.90且半小时不能收回，或保隆仍封而万丰持续放量回落；主题退潮同时出现则停止追价。',
  {'2':h('YES','NO','NO','20日争3板，21日预期容量分歧，累计偏上，可能中途收阴。'),'3':h('YES','NO','NO','22日争震荡后再走强，累计偏上而非每天都涨。'),'5':h('YES','NO','NO','26日仍偏趋势累计上行；以10.90承接和同组扩散为持续条件，失败改为区间或走弱。')}),
 '002889':('YES',4,'与吉大同高的换手竞争者；保留共同晋级路径',
  '非一价3板，19日放量9.42倍且20.48%换手仍封，说明已发生筹码交换；新高度与吉大可共同晋级。',
  '振幅14.95%、13:26最后封板，不能把爆量一律叫充分换手；公告索引本轮超时，半导体及机器人业务由页面摘要提及尚未本次原文核实。',
  '20日无溢价并持续跌破15.74、回封反复而同位吉大保持强势，或公告发现实质风险；不因名字J强行维持正向。',
  {'2':h('YES','NO','NO','20日争4板，21日预计较吉大更早出现大分歧；累计仍偏上。'),'3':h('UNKNOWN','NO','NO','22日是否保持高于15.74取决于20日换手承接，不预设完成卡位。'),'5':h('UNKNOWN','NO','NO','26日前高换手路径容易大幅回撤，不能从今日封板推出五日持续上涨。')}),
 '600487':('UNKNOWN',5,'业绩催化趋势观察；首板竞争池内保留',
  '发行人前三季业绩预告归母59.696207至63.696207亿元、同比151.27%至168.10%；19日98.314亿元成交且封板，具备比纯名字叙事更具体的已披露催化。',
  '业绩预告未经审计；近百亿元成交带来较大接力量要求，非小盘连板路径。PCB/光通信与低空安全并非同一产业，不用它给所有J字票背书。',
  '20日高开低走持续失守50.25且放量，或光通信同类集体回落，则取消短线持续走强假设。',
  {'2':h('YES','NO','NO','至21日累计偏上，20日更可能高开换手而非持续锁板。'),'3':h('YES','NO','NO','至22日业绩线震荡上行倾向，允许中途回落。'),'5':h('YES','NO','NO','至26日累计偏上，须量价承接和业绩线扩散；不预测连板或每天上涨。')}),
 '600148':('UNKNOWN',None,'汽车/军工2板同位候选','一价17.91、换手1.51%，与东安并行竞争，不能只因更低成交额就定为唯一前排。','与保隆、京城、海洋王同为一价，缺少独立强业务催化与真实队列对比。','20日不能维持17.91附近正溢价并被同组持续压制，保留弃判而不自动转正。',{}),
 '600178':('UNKNOWN',None,'汽车/动力2板同位竞争者','19日窄振幅2.83%、量0.678倍、09:30封，价格推进较稳；万丰可能共同晋级。','燃油动力业务不能自动等于电动垂直起降或特定订单，本轮未取得新业务催化原文。','20日失守11.28且反抽失败、同组持续弱化，取消正向观察。',{}),
 '600860':('UNKNOWN',None,'名称叙事与2板先封候选','一价8.70、量0.337倍，是实际晋级事实；林曜正文讨论京城但末段写京投。','不得把京投600683当京城600860；氢储气设备不能直接认证飞行汽车订单，名字联想不是资金因果。','20日失守8.70、同组封板缩减或叙事无独立业务支持，观察失效。',{}),
 '002724':('UNKNOWN',None,'军工/专业照明2板同位候选','一价8.42、量0.141倍，与长春/甘化在军工概念报道中共现。','严重缩量有补量分歧，专业照明与具体航展订单未核；并非天然低空核心。','开板量剧增而无法封住、跌回8.42且同组不跟随，取消观察。',{}),
 '603348':('UNKNOWN',None,'汽车结构件2板竞争者','09:30封、振幅3.14%、量1.19倍，与保隆和万丰均可并行。','自身题材不等同所有特斯拉新产品供货，完整同位中没有价格/催化足以排入Top3的独立优势。','20日失守16.84且修复失败或汽车组普跌，取消观察。',{}),
 '000576':('NO',None,'军工2板中放量分歧候选','早封但量3.60倍、换手13.42%，与海洋王/长春等共同竞争。','有真实上涨事实，若20日缩量转强而对手转弱，会证伪偏断板；不因爆量必判失败。','若20日站稳9.65并比长春/海洋王更强，可追加上修，但原NO留档。',{}),
 '603278':('NO',None,'汽车/机器人2板后排分歧候选','19日13:36封、振幅13.02%、量2.96倍、换手14.04%，同组有更早且更窄振幅的保隆/万丰/东安/文灿。','后发换手也可能卡位；封板后机器人首板群增强是反证，不能由晚封直接断言失败。','20日若显著强于完整同组并持续封住，偏断板被证伪；若失守10.90且回封弱则延续负向。',{})}
 first={
 '605058':('NO','21天12板是滚动形态并非12连板；19日4次开板、尾段14:55封，16.62%换手，反包后再接力度待核。'),
 '002343':('NO','今日地天反包、振幅19.94%、换手46.52%；读者/文轩/华媒等同时跌停，反包成功不修复原15→16误报。'),
 '605255':('NO','立案调查与无中昊资产注入等风险公告已取得，不因低换手给正向交易资格。'),
 '001211':('NO','14:41尾段首板、振幅10.56%，消费线未形成足以支持次日封板的同期梯队。'),
 '003021':('UNKNOWN','机器人午后首板集群之一，14:16封、量2.70倍；与荣泰/日盈/新坐标/宁波/科森/福莱比较，不能事后任选为核心。'),
 '603119':('UNKNOWN','机器人午后首板集群，13:53封、换手2.51%；本次取得减持提前结束文件但不等于新订单。'),
 '603286':('UNKNOWN','机器人午后首板、13:52封，振幅15.86%，主题扩散支持与大分歧并存。'),
 '603040':('UNKNOWN','13:50机器人/汽车叙事首板、换手2.02%，业务具体新进展未本次核实。'),
 '002164':('NO','13:35封、10.48%换手，减速器主题集群共现不认证持续核心。'),
 '002626':('NO','盘中记录59次开板、13:32末封，分歧较重；缺同题材连续梯队支持。'),
 '603859':('UNKNOWN','工业软件AI下午首板，量2.17倍；可与太极形成软件观察但不同于低空订单。'),
 '603626':('UNKNOWN','机器人下午首板、换手7.01%，需与兆威/荣泰/福莱等完整组竞争，未排入重点5只。'),
 '603188':('NO','13:16最后封、换手16.44%、染料叙事独立，同期连续梯队支持不清。'),
 '001216':('NO','5次开板、换手17.43%，公告索引超时保留；不认定MLCC新催化已核。'),
 '600329':('UNKNOWN','中药首板、13:00末封、量3.42倍，医药强于旧消费但不是当前连板主线。'),
 '002886':('UNKNOWN','PCB/材料叙事首板、换手14.49%及2次开板，公告索引超时，不将多概念等价强度。'),
 '605488':('UNKNOWN','11:21机器人材料首板、换手2.80%，理财公告不是机器人供货认证。'),
 '605258':('NO','11:13末封、换手11.15%、振幅13.87%，PCB同组澳弘/沃特等分歧较大。'),
 '601778':('UNKNOWN','新能源电力组11:01首板，与江苏/浙江新能并行，暂缺独立身位优势。'),
 '600686':('UNKNOWN','汽车首板10:44末封，与2板汽车9只中的子组有层级差，但可作扩散观察。'),
 '603330':('NO','10:39首板、振幅14%，热熔胶题材与机器人/低空业务不能混同。'),
 '600032':('UNKNOWN','10:11电力首板，江苏/晶科完整同组并存，不机械三选一。'),
 '603693':('UNKNOWN','10:02电力首板，早于另两只但无证据认证独立电力核心。'),
 '001323':('NO','09:46末封、消费家居首板，不能由早封反推板块主升。'),
 '002368':('UNKNOWN','09:33软件首板，数据库/信创与吉大安全概念仅语义相邻，缺完整共同催化认证。'),
 '002929':('UNKNOWN','09:30末封、缩量0.471倍且有一次开板，页面合同金额线索本次未取得发行人原文。'),
 '600267':('UNKNOWN','业绩线一价首板，量0.352倍；有订单/预增须进一步原文核验，与亨通不是同一行业。')}
 rows=[];relay=[r for r in facts['rows'] if r['kind']=='limit_up' and r['boards']>=2]
 relcodes=sorted(r['code'] for r in relay)
 for r in facts['rows']:
  if r['kind']!='limit_up':continue
  c=r['code'];peers=[p['code'] for p in relay if p['boards']==r['boards'] and p['code']!=c] if r['boards']>=2 else sorted(k for k in first if k!=c)+['600487']
  if c in authored:
   call,rank,role,support,counter,cancel,hs=authored[c]
  else:
   call,support=first[c];rank=None;role='首板完整竞争池观察';counter='催化或行业关联仅据页面/媒体线索，未知不等于已证伪；实际次日封板可证伪负向或弃判。';cancel='仅在独立催化核实且20日相对完整同组转强时追加上修，原判断保留。';hs={}
  for k in ['2','3','5']:
   if k not in hs:hs[k]=h('UNKNOWN','UNKNOWN','UNKNOWN','未列入重点多日预测，不自动继承Top5走强情景；仍保留全池到期价格结算。')
  rows.append(dict(code=c,name=r['name'],boards=r['boards'],signal_close=r['close'],next_close_limit_up_call=call,next_intraday_touch_call='YES' if call=='YES' else 'UNKNOWN',priority_rank=rank,role_hypothesis=role,support=support,counter_evidence=counter,cancel_conditions=cancel,complete_same_height_peers=peers,horizons=hs,probabilities='未验证',tradeability='一价不可据此视为成交；非一价须竞价/开盘承接确认，T+1下当日买入不能当日止损卖出。',facts_reference='facts/qualified_current_facts_SCORES_MASKED.json#'+c))
 assert len(rows)==39 and sorted(r['code'] for r in rows if r['boards']>=2)==relcodes and len(relcodes)==11
 forum=dict(formed_at=now(),capture_receipt='forum/limited_capture_receipt.json',current19_selected_preferred_authors_body_count=0,
  sources=[dict(id='linyao_1016',author='林曜01',published='2026-10-16T16:18:00+08:00',kind='原文正文与后续评论分栏',adopt='检查高位断板时首板种子、完整身位与新分支',reject='名字/字母和上板先后可作叙事线索，不能直接认证业务/资金让位/次日晋级；京城与末段京投歧义保留',comments='10月18/19日回复不继承16日正文时间'),dict(id='ale_1019',author='节奏大师阿乐',published='2026-10-19T05:35:00+08:00',kind='搜索工具读取全文；直接HTTP空页未冒称成功',adopt='19日早评点名保隆/万丰，更新分别09:23:57、09:29:20、09:30:34；完整竞争与撤销条件意识',reject='高压气罐供货、明星选手能排到一字、最快即核心均未独立认证；09:23更新不倒填05:35，20日预测不借今日事后检索伪装事前'),dict(id='counterpoint_1016',author='jl韭菜抄家',published='2026-10-16T21:30:00+08:00',kind='原文正文',adopt='题材市场传播与产业兑现分开；反复盈利未退出暴露持仓规则问题',reject='作者情绪及名气不作为胜率；本文没有证明任何具体特斯拉产品订单或因果流向')],
  full_cards=[dict(code=r['code'],name=r['name'],direct_forum_coverage=('linyao_1016' if r['code'] in ['003029','002889','603197','600860','603278'] else 'ale_1019正文/分时更新' if r['code'] in ['002085','600178','600148'] else 'UNKNOWN_NOT_NEGATIVE'),official_or_price_check='逐股对应本表facts_reference与已取得公告，页面AI业务摘要不是事实',independent_handling='ACCEPT_PRICE_AND_STRUCTURE; REJECT_UNVERIFIED_CAUSALITY; UNKNOWN_BUSINESS_WHERE_NO_ISSUER_TEXT') for r in rows])
 write(OUT/'forum/full39_preread_and_independent_adoption.json',forum)
 sources=[OUT/'facts/qualified_current_facts_SCORES_MASKED.json',OUT/'facts/603337_source_conflict_APPEND_CORRECTION.json',OUT/'forum/full39_preread_and_independent_adoption.json',OUT/'facts/notices/fulltext_manifest.json',OUT/'facts/notices/index_manifest.json',Path(__file__)]
 record=dict(version='19close20target_independent_full39_v1',formed_at=now(),signal_date='2026-10-19',target_date='2026-10-20',official_horizon_targets={'1':'2026-10-20','2':'2026-10-21','3':'2026-10-22','5':'2026-10-26'},
  scope='完整替代收盘页主板非ST39封板；11连板；额外603337保留来源冲突不冒称全交易所已核',full_pool_count=39,relay_count=11,extra_source_conflict_count=1,
  model_exposure='CURRENT19_SCORES_NOT_DISPLAYED_SELF_ATTESTED',gemini_current_views_exposed=False,external_blinding_verified=False,current_forecast_formal=False,
  market_thesis='缩量指数修复但广度仍弱，老传媒/消费负反馈，新低空/汽车/军工与午后机器人分化；20日偏前排续强后排淘汰，不押全面反转。',
  top3=['003029','603197','002085'],watch2=['002889','600487'],rows=rows,
  not_a_calibrated_probability=True,not_actual_fill=True,new_fits=0,new_formal_predictions=0,
  sources={str(p.resolve()):sha(p) for p in sources})
 path=OUT/'independent_full39_BEFORE_MODEL_AND_GEMINI.json';write(path,record)
 write(OUT/'independent_archive_receipt_BEFORE_MODEL.json',dict(formed_at=now(),full39_file=str(path.resolve()),sha256=sha(path),top3=record['top3'],watch2=record['watch2'],self_attested_before_model_and_gemini=True,external_blinding_verified=False))
 print(json.dumps(dict(formed_at=record['formed_at'],rows=len(rows),relay=11,top3=record['top3'],watch2=record['watch2']),ensure_ascii=False))
if __name__=='__main__':main()
