"""Human-authored complete-pool calls, archived before any current BASE display."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,csv
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_new(p,x):
 with p.open('x',encoding='utf-8') as h:json.dump(x,h,ensure_ascii=False,indent=2,allow_nan=False)
f=read(OUT/'facts/complete_enriched_facts_SCORES_MASKED.json')
# Directions are authored below, not generated from a model, keyword or cutoff.
# Columns: code | next-close-limit call | stock-specific support | strongest counterevidence.
text='''
603278|YES|上午指数低点附近已显著强于海洋，13:09供应商封板后低位机器人扩散，当前最像高度情绪锚|6板换手33.42%且监管事项未有解除证据，历史浮盈兑现压力；仅观察不能把涨停倾向当许可
002724|UNKNOWN|跌停至涨停修复完成6板，与大业共同晋级说明不是机械零和|午盘仍跌停、最终封至14:54才稳定，羊线九阳未同步转强；地天次日获利兑现与再次分歧
600838|YES|3板且全天一价，零售新华百货和百大形成低位响应|一字缩量没有检验新增接力承接；风险提示且机器人可能吸引注意，不能无条件顶一字
603089|YES|09:36完成回封，指数弱时保持二板，换手6.83%，相较襄阳开板和筹码交换压力较小|南方/睿能/赛福等同位强封竞争；6.83%换手不证明筹码锁定，机器人板块转弱可整体失败
603280|YES|09:30封住二板且日线验证，属机器人低位早期强封，相比午后随指数封板有先手|31.82至31.97的狭窄日内区间、缩量近一字，尚未经历充分分歧，首次开板兑现风险
603933|YES|二板一字，机器人低位群体强，工业自动化/重组叙事可作当日背景|一字流动性和真实承接未验证，重组问询延期带来不确定性，不以概念收入小直接否决情绪
603028|YES|二板全天一字，首板后快速取得同位认可，机器人群体配合|本日原薄盘代理触发且一字不可假定成交；预测倾向与剔除参与资格并存
003021|UNKNOWN|二板且机器人容量方向修复，午后回封使高低位共同受益|流通市值超原主选上限且多次炸板，容量趋势不等于小盘情绪龙头，次日封板方向保留未知
603528|UNKNOWN|二板午后一次封住，成交换手真实可观察且同题材群体支持|13:19才封，早封同位正裕南方更强，普涨日跟随修复不充分证明独立性
002535|UNKNOWN|低价二板早封、换手7.51%，结构上有竞争力|最新异动文本确认此前已收到财务涉嫌虚增收入的处罚事先告知，主选剔除；风险事实不自动等于次日跌停
000678|UNKNOWN|二板在10:49恢复封住，早于午后指数修复，属于有效的晨间承接反例|换手27.27%且多次开板，新增接力需吸收大量交换筹码；与较稳定正裕相比压力较重
603677|YES|09:44触板10:22回封，早于午后普涨，流通31.39亿和换手6.64%符合早期小盘入口|机器人同线大量首板，低成交额不证明独立控盘；需二板同位和早盘承接证明它不是一日脉冲
002009|YES|09:43已有冲板，经历分歧后13:23回封，换手17.19%，同题材有高标及二板梯队|筹码交换明显多于奇精，早盘冲板失败者周一有兑现需求；热题材回流也可能只给一日修复
603081|YES|大业封板后13:13一次封住，流通45亿，成交较前5日明显放大，属于较早低位扩散|仍晚于早盘先行股，换手3.94%不等于充分承接；与13:17派斯林仅数分钟差，不能据此认证补涨龙
600215|UNKNOWN|13:17一次封板，流通28.97亿且换手5.36%，低价低位机器人候选|与大丰接近、尚无独立带动证据，小盘本身不能证明更强；列第四相对观察不自动补位
002026|UNKNOWN|13:31回封且换手5.12%，机器人扩散中有真实交易|封住晚于大丰派斯林且缺领先证据，需竞争检验不能因主题相同加固定分
001298|UNKNOWN|10:46封住早于午后普涨，存储与机器人标签可提供两方向观察|双标签不代表双重资金承接，同日大港和机器人早封股竞争，尚未证实跨题材核心
603721|UNKNOWN|午后传媒/AI回流并最终封板，约28亿小盘具情绪弹性|分钟核验否决供应商09:55首封，实际13:44触板13:59分钟封住，不能误判为早盘先锋
002343|UNKNOWN|老周期反包，09:34早触板且尾盘再封，有辨识度与传媒修复|换手42.36%、多次回落、新华未封与旧锚分化；反包首板不是低位新周期确认
002474|UNKNOWN|09:31早触板10:18回封，政务AI与信创交易叙事有观察价值|较多开板及17.22%换手，行业扩散弱于机器人，周一需相对强度不能只看低价
002421|UNKNOWN|09:45曾触板且13:22回封，低价AI方向交易活跃|换手26.50%并多次开板，短线交换压力重于早封低位主选，不能因低价自动加分
002354|UNKNOWN|09:58已回封，AI与机器人方向早期活跃，对题材扩散有参考意义|流通130.65亿超原上限，只作容量及情绪对照，不纳入本路线主选
002945|UNKNOWN|10:23封板与指数低点后金融修复相邻，是市场风险偏好确认对象|流通414.72亿超原范围且次日持续未证实，不把指数反弹工具当小盘龙头主选
603901|UNKNOWN|14:00一次封板，低位工业机器人扩散且量较前5日显著提升|下午后排且换手4.41%，一日普涨不能确认新增领导地位
603286|UNKNOWN|机器人电子皮肤/汽车链叙事下首次封板，流通59.48亿符合原范围|14:37才封、此前已有涨幅，低位身份不如奇精明确，先等待群体内领先
002931|UNKNOWN|机器人收购叙事和一次封板，流通78亿未越上限|14:16后排且重大事项进展未作完整资格确认，本批不把交易标签当确定催化
002380|UNKNOWN|工业自动化方向首板，价格回升量能扩大，流通39亿|14:34末封且多次开板，弱于早封同题材，周期性回弹与新龙需继续区分
002988|UNKNOWN|低位轻量化方向配合机器人，14:31最终回封|封板后有抛压且换手仅3.64%，受板块修复带动的替代解释强
001283|UNKNOWN|机器人电池/眼镜叙事首板且换手16.87%可见交易|尾盘至14:55仍需回封、6次开板，封板质量弱于早封同位
603448|UNKNOWN|小流通盘、午后机器人线活跃并最终封住|换手24.99%且多次炸板，次新价格波动大，不能仅因盘小提高优先级
603466|UNKNOWN|物理AI叙事随午后机器人扩散，最终一次封住|流通87亿接近原上限，换手16.70%且已有前期活跃，可能只是分歧后的回流
001359|UNKNOWN|PCB/电子布方向放量首板，流通45亿|14:54末封且强板块并非该线，较高价位和此前回撤使承接需另验
600303|UNKNOWN|低价车桥/军工链首板有弹性，换手13.77%|14:46后排封板，不能由低价推断必有接力，监管/风险完整性尚无充分正证
603838|UNKNOWN|摘帽后涨停且量比提升，小市值不超范围|摘帽特殊背景、临近收盘反复开板，和新机器人先行线辨识度不同
002687|UNKNOWN|低价服装方向放量首板，可作独立消费观察|没有与强零售核心等同的群体证据，不把消费泛标签当助攻
002612|UNKNOWN|医美方向13:20封板，换手与规模处原范围|相对独立但同主题连板梯队不足，无法仅凭低位判断次日封住
600785|UNKNOWN|上海九百一字时早触板、13:05回封，零售低位有真实响应|换手16.94%且回封晚，九百打开或机器人分流都会降低接力预期
600865|UNKNOWN|13:04封板，零售高低位扩散得到盘面支持|近期回撤修复也可解释首板，强于新华的独立证据有限，九百溢价必须维持
001216|UNKNOWN|MLCC/材料叙事首板，早触板午后回封|近几日反复涨跌且换手18%，前期浮盈与套牢压力假设并存，不认定唯一主题核心
002322|UNKNOWN|13:02封板且量能提升，软件业绩方向可见|缺完整同主题梯队和盘中领先证据，不能把业绩标签直接映射连板
001328|UNKNOWN|早盘触板、10:07稳定封住，业绩与消费背景|此前多次开板而换手仅3.36%，盘口强弱需核，消费竞争已有九百及低位跟随
002077|UNKNOWN|09:59回封且10.72%换手，半导体早盘局部强势|华天及通富并未收涨停，封测/测试与机器人不是同一确定主线，次日持续性未知
600506|UNKNOWN|09:54回封、10.31%换手，液冷方向出现早期主动性|其他液冷梯队缺充分确认，独立机会和孤立首板需要周一表现区分
000972|UNKNOWN|供应商报告首板和低位消费背景|本次缺当日合格不复权日线，不补造价位、连续资格和参与条件
002982|UNKNOWN|一字首板体现竞价封板需求|流通17亿换手2.62%触发原薄盘代理，缺交易承接，不进入主选
000156|UNKNOWN|广电与机器人题材背景下尾盘封板|流通131亿超原上限、14:47后排，保留容量对照
002812|UNKNOWN|电池方向普涨中封板、容量较大|流通384亿超原上限，不用大市值反弹抬高早期情绪龙头命中率
002747|UNKNOWN|机器人容量中军涨停，与小票可能助攻共存|流通229亿超原范围，容量确认不等于小盘龙头主选
002439|UNKNOWN|安全软件尾盘涨停可作科技扩散对照|流通110亿超原范围，14:37后排且持续性未验证
603341|UNKNOWN|机器人与AI眼镜方向首板形成交叉观察|流通118亿超原范围，多标签不等于领涨，保留对照
000700|UNKNOWN|人形机器人/汽配题材午后回封|流通109亿超原范围且尾盘回封，不列本路线早期主选
002979|UNKNOWN|机器人伺服方向14:15封板可确认中军支持|流通121亿超原上限，不纳入首板主选
002241|UNKNOWN|歌尔涨停支持容量科技与机器人共振假设|流通731亿远超主选范围，不将大票一进二当小票情绪龙头成果
000810|UNKNOWN|眼镜/智能终端午后回封显示科技活跃|流通112亿超原上限，题材扩散不自动保障连板
603171|UNKNOWN|数字化软件与AI方向回封，体现容量广度|流通196亿超原范围，作为方向对照
601519|UNKNOWN|午后金融方向封板参与指数修复|流通184亿超原范围，指数弹性工具不同于本路线小盘龙头
000559|UNKNOWN|机器人中军13:55封板，支持群体修复|流通384亿超原范围，只留主题容量对照
603859|UNKNOWN|物理AI和工业软件13:43封住|流通107亿超原范围，不改变原上限凑候选
603160|UNKNOWN|芯片端侧方向13:43封板，科技修复广度增加|流通263亿超原范围，不能以容量行情混入小盘样本
605488|UNKNOWN|多日活跃且机器人柔性触觉方向再涨停，主题辨识度值得观察|流通114亿超原范围，此次是反包首板而非未经涨幅的新起点
603713|UNKNOWN|业绩与物流方向低位封板，可观察独立线|流通116亿超原上限、题材关联与资金认可未验证
001301|UNKNOWN|锂电材料容量首板支持成长修复|流通119亿超原范围，不当情绪小票主选
603337|UNKNOWN|机器人工业线13:00首板，午后修复较早|流通183亿超原范围，保留背景不提高主选概率
600094|UNKNOWN|地产方向首板并有量能扩大|流通93.34亿略超原93.27亿上限，不能为微差擅自放宽
603186|NO|盘中曾到涨停说明有风险偏好响应|收盘未封且换手21.64%，与多数封板科技相比弱，尚无重新领先证据
603559|NO|通信方向盘中触板|普涨背景仍未封、同主题梯队不强，不把触板当连续接力认可
001360|NO|机械方向盘中触板有局部买盘|换手20.84%且回封失败，在强机器人日相对弱
603626|UNKNOWN|机器人链涨幅接近涨停，保留修复关注|最终未封，不能把接近涨停误写封板；与早封同位相比缺口仍在
001282|NO|锻造汽配可与机器人形成观察线索|同线普涨仍炸板，换手6.11%未使封板成立，偏不续强封
002253|NO|航空空管方向盘中有涨停尝试|回封失败、题材扩散证据弱于核心，次日封板倾向不正向
002164|UNKNOWN|机器人群体修复带来较强上冲|最终炸板、换手19.92%，是主题并非全面成功的重要反例
603728|UNKNOWN|机器人容量方向盘中触板|大市值且炸板，不能以容量强就认定全部小票受益
603600|UNKNOWN|供应商列入炸板池|本次当日合格日线缺失，不补造精确强弱和结果
002307|NO|基建方向盘中触板|强修复日仍炸板、换手16.42%，相对弱于核心题材
600851|NO|医药方向盘中曾冲板|收盘未封且缺完整梯队，暂不把独立分支当强接力
002913|NO|此前成交路径可提出抛压假设但账户身份未知|普涨日仍跌停，个股风险/负反馈没有随指数解除
002578|NO|有色方向存在独立波动|普涨中仍跌停，缺少次日强反包正证
002811|NO|前期高波动股有反抽可能|当日跌停负反馈延续，未观察到稳定修复
600202|NO|昨日二板曾有题材关注|今日跌停且二板整体淘汰，不能因市场回暖就自动看修复
001368|NO|昨日弱势可有回补需求但尚未确认|当日继续跌停，承接仍缺证据
002388|NO|低位风险股可能反抽但无积极证据|缩量跌停路径下可交易性和接力证据不足
'''
manual={}
for line in text.strip().splitlines():
 code,call,support,counter=line.split('|');assert code not in manual;manual[code]=(call,support,counter)
assert set(manual)=={r['code'] for r in f['rows']},(set(manual)-{r['code'] for r in f['rows']},{r['code'] for r in f['rows']}-set(manual))
first_rank=['603677','002009','603081','600215','002026','001298','603721','002343','002474','002421','603901','603286','002931','002380','002988','600785','600865','600506','002077','002322','603466','001216','001283','603448','001328','002612','001359','600303','002687','603838','000972','002982','000156','002812','002747','002439','603341','000700','002979','002241','000810','603171','601519','000559','603859','603160','605488','603713','001301','603337','600094']
# Only qualified original-screen first boards receive a main-scope rank.
qualified={r['code'] for r in f['rows'] if r['kind']=='limit_up' and r['boards']==1 and r.get('screen_status')=='WITHIN_ORIGINAL_RESEARCH_SCREEN'}
first_rank=[c for c in first_rank if c in qualified]
assert set(first_rank)==qualified,(qualified-set(first_rank))
relay_top=['603089','603280','603278'];first_top=['603677','002009','603081']
rows=[]
for r in f['rows']:
 call,support,counter=manual[r['code']]
 rows.append(dict(code=r['code'],name=r['name'],source_kind=r['kind'],boards=r.get('boards'),close=r.get('close'),next_close_limit_up_call=call,
  first_board_relative_rank=first_rank.index(r['code'])+1 if r['code'] in first_rank else None,
  priority_rank=relay_top.index(r['code'])+1 if r['code'] in relay_top else first_top.index(r['code'])+1 if r['code'] in first_top else None,
  support=support,counter_evidence=counter,cancel_condition='原筛选不通过/重要监管风险未清不参与；本股转弱失守收盘参考且题材或同位不能承接则取消。各重点股具体竞价和触发见独立执行卡。',
  original_screen=r.get('screen_status','NOT_LIMIT_UP_CANDIDATE'),source_qualification=r.get('qualification'),probability='UNVALIDATED'))
bank={str(p.relative_to(OUT)):sha(p) for p in [OUT/'facts/complete_enriched_facts_SCORES_MASKED.json',OUT/'minutes/capture_receipt.json',OUT/'minutes/additional_capture_receipt.json',OUT/'context/notices/focus_9_fulltext_manifest.json']}
archive=dict(formed_at=now(),author='Codex',signal_date='2026-10-23',target_date='2026-10-26',
 horizon_dates={'1':'2026-10-26','2':'2026-10-27','3':'2026-10-28','4':'2026-10-29','5':'2026-10-30'},
 model_output_seen=False,actual_Gemini_current_opinion_obtained=False,internal_research_assistance=True,full_rows=len(rows),
 information_clock='Formed after close, Shanghai 2026-10-24; includes separately attributed evening disclosures. Not an as-of-15:00 signal.',
 leader_candidate='603278',relay_top3=relay_top,first_board_top3=first_top,first_board_top1=first_top[0],
 relay_priority_definition='综合研究/参与关注排序：正裕与南方为条件候选，大业第3为仅观察情绪锚；不是原BASE排序或涨停概率排序',
 market='指数由早盘急跌转午后强修复，低位机器人扩散、高标双晋级，但昨日二/三板仍大面积淘汰，旧羊线未普遍修复；下周先看修复分歧与新二板淘汰，不能因总涨停增加无差别接力。',
 forum_use='有限两篇狼王行千里10/23午盘与盘后原帖，作者观点作线索；不采纳自报胜率、量化账户身份或单因果。',
 falsifiers=['机器人容量与小票可同涨，华天/新华不固定互斥','大丰与派斯林时间只差约4分钟不能单独证明优劣','正裕/南方/襄阳同位可共同晋级，不是强制二选一','大业监管状态与最像龙头身份判断分开'],
 source_hashes=bank,rows=rows,new_fits=0,new_orders=0)
write_new(OUT/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json',archive)
write_new(OUT/'INDEPENDENT_FREEZE_RECEIPT.json',dict(formed_at=now(),sha256=sha(OUT/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json'),full_rows=len(rows),model_output_seen=False,signal_date=archive['signal_date'],target_date=archive['target_date']))
with (OUT/'完整81只_独立判断_模型展示前.csv').open('x',encoding='utf-8-sig',newline='') as h:
 w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(dict(rows=len(rows),first_rank_count=len(first_rank),relay_top3=relay_top,first_top3=first_top,sha256=sha(OUT/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json')),ensure_ascii=False))
