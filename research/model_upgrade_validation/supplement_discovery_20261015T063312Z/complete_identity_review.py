"""Explicit retrospective author decisions, never supervised gold labels."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import stock_judgement_review as o
import relay_review_workbench as w

out=Path(__file__).parent/'delivery_final'
facts_path=out/'facts_with_identity_discovery.json'
facts=o.read(facts_path)
context=o.read(out/'discovery/discovery.json')
draft=o.read(out/'author_identity_PENDING.json')
# These decisions are authored from the checked source/time/competition facts;
# no direction, role truth or missing entry is automatically filled.
decisions={
'002696':('CYCLE_SUPPLEMENT_CANDIDATE',['002242'],
    '九阳周期提前启动的名称/情绪补涨候选，当前相对阳光身位领先。',
    '百洋10月9日起1/2/3/4板，九阳同期5/6后13日未续；原正文名称链及独立评论有九阳—百洋连接。阳光同周期2板，百洋4板是本池身位领先事实。',
    '农业避险可以独立解释部分表现；九阳不是已证唯一父龙。14日百洋为一价日、量为前交易日约0.092倍，这体现价型和交易约束，不证明可成交或资金迁移。',
    '名称/情绪线相对同周期持续转弱，或独立行业解释获得更强证据，则撤销核心补涨假设；一价封死不能据此声称买到。',{}),
'001318':('OLD_FOLLOWER_RESTART',['002242'],
    '九阳消费/名称线的旧跟风重起候选，与百洋的提前启动补涨分开。',
    '9月30日已出现阳光首板；本轮13/14日1/2板，启动与九阳首次未续板同日，旧消费段落列阳光并讨论九阳/均瑶。',
    '旧跟风重起不等于新周期核心；百洋同周期4板领先。均瑶是有原文背景的替代锚，尚不足认定单一父龙。',
    '消费/名称线缺扩散、相对百洋及完整二板组持续落后，或重起时序被反证，则降低该身份判断。',
    {'000678':('REJECT','只有阳字相同，行业与段落没有逐股连接，不能由襄阳轴承推父锚。'),
     '605388':('PENDING','原消费段落有均瑶背景，历史2板后未续；缺本轮阳光与均瑶的直接关系和领导地位证据。')}),
'603096':('THEME_RELAY_COMPETING',['600825'],
    '新华传媒周期的传媒补涨核心候选，与慈文共同保留竞争。',
    '自主发现新华传媒原8/9/10板后13日未续；新经典9日起1/2/3/4板跨过该节点，出版/IP语义匹配传媒，当前与慈文同为4板；完整当期传媒语义范围还保留新华文轩、中国科传首板。',
    '作者称补涨龙只是观点。14日新经典从22.00开、21.64最低回到23.07收，较前日量约7.906倍；慈文亦回到日高，单凭放量及收高无法证明新经典控制全组。',
    '传媒梯队失去持续性、相对慈文的主动性证据未兑现，或独立题材解释更强，则撤销核心候选；不以作者称号确认唯一龙。',{}),
'002343':('THEME_RELAY_COMPETING',['600825'],
    '新华传媒周期的传媒接力/伴飞竞争者，核心地位仍需与新经典比较。',
    '慈文与新经典同为9日起1/2/3/4板，传媒属性与原新华高标形成时序相容的接力假设；13日一价后14日出现换手日并收在日高；还保留新华文轩、中国科传首板背景。',
    '14日量为前日约3.545倍，与新经典同日共同回到日高，不能预设二选一。原作者伴飞说法与新经典补涨龙说法均非资金因果真值。',
    '失去同板位主动性或传媒梯队转弱时撤销接力核心假设；保留共同晋级、共同失败、分化三条路径。',{}),
'603968':('RELATED_UNRESOLVED',[],
    '化工晋级候选，具体补涨父锚未成立。',
    '醋化12/13/14日1/2/3板已核；工业化工与历史亚邦同大类，足以列研究线索。',
    '亚邦较早2板不是已证本轮父龙；正文跟随新华的主体是善水，不能移给醋化。食品饲料添加剂也不能把醋化并入九阳消费补涨。',
    '只有取得醋化与具体父锚的直接连接、兼容时序和同题材领先证据才升级身份；化工放量本身不够。',
    {'603188':('PENDING','化学制品同大类但节点更早且仅2板，缺本轮逐股父锚或核心证明。')}),
'600847':('THEME_RELAY_COMPETING',['600241'],
    '时代万恒退潮节点后的电池同题材接力候选，尚不能叫补涨龙。',
    '时代原6板后13日未续，万里13/14日1/2板，万里电池/储能叙事与时代电池行业存在时间与属性交集。',
    '同电池行业不能证明资金转移；紫竹、雄韬、丰元、龙蟠等替代历史锚同时保留，万里缺直接关系正文和核心控制证据。',
    '若电池梯队无共同反馈、万里缺同位主动性，或独立催化更强，则降为仅行业相关。',
    {'002058':('PENDING','历史4板固态背景可作替代周期，但本轮万里直接连接缺失。'),
     '002733':('PENDING','历史电池2板只提供同类背景，缺领导地位与逐股连接。'),
     '002805':('PENDING','历史电池2板及其失败保留，不能仅因同业改称父龙。'),
     '603906':('PENDING','历史电池2板背景较弱，未核本轮父子关系。')}),
'000530':('UNKNOWN',[],
    '冰雪/液冷设备叙事有独立解释，补涨父锚未知。',
    '13/14日1/2板路径已核，原当期理由含冰雪工程、液冷、储能温控。',
    '储能温控是配套设备，不能由储能二字直接归入电池高标补涨；没有合格更早同题材父锚。',
    '先核冰雪/设备完整竞争及逐股催化，取得直接父锚与时序后再改身份；未知不作断板理由。',{}),
'000668':('NOT_SUPPORTED',[],
    '对丰元补涨连接不采纳，其余角色仍未知。',
    '荣丰13/14日1/2板成立，扫描保留了丰字线索供审查。',
    '与丰元只有单字重合，无逐股原文、题材或父锚证据；这不是对荣丰晋级方向的否定。',
    '出现可核名称链、具体父锚时序及竞争依据时重新审阅。',
    {'002805':('REJECT','丰字相同不能建立荣丰与电池股丰元的关系。')}),
'000823':('UNKNOWN',[],
    '电子方向候选，补涨身份未知。',
    '13/14日1/2板已核，完整二板同位对象保留。',
    '现有历史完整池没有可支持的更早电子父锚与逐股连接，不能把未检出说成没有催化或独立龙。',
    '补齐具体电子题材连接、父锚节点和同位主动性后才升级身份。',{}),
'002813':('UNKNOWN',[],
    '复牌连续板资格未齐，父锚与核心身份未知。',
    '当期来源报2板但13日收盘池和日线均没有本轮连续交易记录，14日有实际日线。',
    '不能把停牌前上次交易日当13日，不能因报2板便构造补涨启动日。',
    '先核复牌及连续涨停口径；不足前保留全部候选并拒绝角色确证。',{}),
'002860':('RELATED_UNRESOLVED',[],
    '复牌资产收购/家电线索待核，不认定九阳补涨。',
    '14日存在一价日线，当前家电零部件语义可列九阳/均瑶背景。',
    '13日日线缺失，本轮连续板未通过；业务邻近不是已核关系，尔字也不能连接格尔软件。',
    '先补复牌和连续板资格，再核直接叙事与竞争；一价不能确认可成交。',
    {'002242':('PENDING','家电属性相邻，但复牌连续板资格与本轮直接关系均缺。'),
     '605388':('PENDING','只有宽泛消费邻近，缺本轮逐股连接。'),
     '603232':('REJECT','尔字重合，软件与资产收购/家电没有合格关系。')}),
'002917':('NOT_SUPPORTED',[],
    '否决博纳名称误连，其他补涨角色未知。',
    '13/14日1/2板已核，完整二板竞争仍保留。',
    '金奥博与博纳仅博字重合，无法从影视历史锚推出其补涨身份；军工等当期叙事需独立核验。',
    '具体同题材高标和直接连接取得后重新分析，不由单字或放量升级。',
    {'001330':('REJECT','博字重合不构成题材或名称交易链。')}),
'600202':('UNKNOWN',[],
    '设备候选，补涨父锚未知。',
    '13/14日1/2板路径已核，保留完整同位竞争。',
    '现有父锚扫描没有合格直接连接，不能由空调行业自动指定九阳为父龙。',
    '需具体题材、父锚与竞争证据后改判身份，不因来源未覆盖转负。',{}),
'600391':('UNKNOWN',[],
    '航空军工候选，补涨父锚未知。',
    '13/14日1/2板路径已核，当期来源出现航空叙事。',
    '航空事件不能单独证明与某历史高标存在补涨关系，也不能单独证明核心。',
    '核同题材完整梯队和节点连接，取得相对领先依据后再升级角色。',{}),
'601188':('NOT_SUPPORTED',[],
    '否决龙字误连，交通/冰雪线仍需独立判断。',
    '13/14日1/2板已核，同位对象完整保留。',
    '龙蟠和龙版只与龙江共享龙字，电池或传媒高标不能据此成为其父龙；未支持不等于已证无补涨。',
    '仅在取得交通/冰雪或真实名称链及兼容时序后重开父锚假设。',
    {'603906':('REJECT','龙字相同没有电池到交通的已核叙事连接。'),
     '605577':('REJECT','龙字相同没有出版到交通的已核叙事连接。')}),
}
for row in draft['rows']:
    call,accepted,relative,support,counter,cancel,others=decisions[row['code']]
    ctx=next(r for r in context['rows'] if r['code']==row['code'])
    row['identity_discovery_assessment']=dict(reviewed=True,identity_call=call,accepted_anchor_codes=accepted,
        relative_role_hypothesis=relative,support=support,counterevidence=counter,cancel_condition=cancel,
        rejected_or_pending_anchors=[dict(anchor_code=c,decision=v[0],reason=v[1]) for c,v in sorted(others.items())],
        same_cycle_competitors=sorted({c for g in ctx['same_cycle_competitors'] for c in g['members'] if c!=row['code']}),
        reviewed_at=o.now().isoformat(),role_truth_certified=False)
draft.update(author_review_completed=True,reviewed_at=o.now().isoformat(),
    identity_review_scope='ALREADY_MODEL_EXPOSED_RETROSPECTIVE_HYPOTHESES_NOT_ROLE_TRAINING_TRUTH')
completed=out/'author_identity_completed.json'
o.write_new(completed,draft)
w.validate_completed(facts_path,draft)
w.review_context(facts_path,completed,out/'completed_identity_review.json')
w.render(facts_path,out/'review_completed',completed)
print(dict(complete_identity_reviews=len(draft['rows']),new_forecasts=0,new_role_gold_labels=0))
