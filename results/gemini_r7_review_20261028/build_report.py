"""Write a separate Gemini-model comparison; preserve the original review."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
OLD=ROOT/'results/daily_review_20261028T1020Z'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
scores=read(OUT/'GEMINI_R7_SCORES.json')['rows'];gate=read(OUT/'REGIME_GATE.json');checks=read(OUT/'VERIFICATION.json')
original=OLD/'20261028复盘与1029预期预测_完整双榜.md'
before=sha(original);report=original.read_text(encoding='utf8')
now=datetime.now(timezone.utc);bj=now.astimezone(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
report=report.replace('# 2026年10月28日复盘与10月29日预期预测','# 2026年10月28日 Gemini 第七轮模型对照复盘与10月29日预期',1)
intro=f'''
本次新增形成时间：{bj}（北京时间）。用户明确要求实际使用 Gemini 新模型。本报告在原完整复盘上追加独立模型推理和参与状态，不覆盖原报告、原预测或账本。下文原行情、公告、论坛和独立判断沿用已绑定归档，并非本次重新采集。已有 BASE 和人工结论已见，不称盲审。

**本次确实加载并运行了 Gemini 第七轮修正版模型。三板范围内：青岛食品第1，光洋股份第2；配套情绪闸门0.5417未达到0.55，严格按整套流程仍为观察。襄阳轴承五板、五只二板及51只首板不适用三板模型，绝不伪造其R7分数。**

模型原文件为 `dragon_round7_capacity_dragon_20261028T094500Z/round7_all_fold_models.joblib`；按运行前协议固定使用第4折五个任务头，训练截止2025-11-30，不重训、不挑折。主排序沿用原 Arm3：0.40×空间分+0.35×5板分+0.25×3进4分。Arm2等仅作对照，不按效果切换。该模型在10月28日收盘后形成，本次也不属于模型在收盘前已冻结的完全盲测；10月29日结果尚未取得。
'''
pos=report.index('\n## 1 大盘');report=report[:pos]+'\n'+intro+report[pos:]
old_note='模板新增潮汐/大龙模型未取得经本次资格核验的当日推理包，不给分、不套旧日期分、不照抄模板62.7%/翻倍等战绩。三板训练模型也不外推给五板、二板或首板。'
assert old_note in report
report=report.replace(old_note,'原报告未调用Gemini新模型；本次已按用户新指令追加实际推理，结果如下。原BASE分数及方向完全保留。')
section='''
## 3A Gemini 第七轮实际模型推理

输入范围为10月28日完整59只收价涨停事件中的8只连板，原生三板仅青岛食品与光洋股份2只，均保留。与原BASE目标不同：本模型各任务头分别学习3进4、未来空间高标代理、连续达到5板、未来10个交易日最高价涨幅至少46.4%的路径事件，以及附带早期断板限制的18%反弹代理；后两者不是可实现交易收益，也不直接认证市场情绪龙头。

| 三板排名 | 股票 | 3进4分 | 空间高标分 | 达5板分 | 46.4%路径分 | 18%反弹分 | Arm3综合分 | Arm2空间综合分 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
'''
for r in scores:
    section+=f"| {r['rank_arm3']} | {r['name']} {r['code']} | {r['score_3to4']:.4f} | {r['score_space']:.4f} | {r['score_5b']:.4f} | {r['score_gain46']:.4f} | {r['score_reb18']:.4f} | **{r['score_arm3_capacity']:.4f}** | {r['score_arm2_space']:.4f} |\n"
section+='''
所有值均为未经校准的模型输出；0.7182不等于71.82%实际晋级概率，0.5773也不等于57.73%成龙概率。青岛的3进4与空间头均高于光洋；光洋的反弹代理头更高，但不能据此把当日接力取消改成自动低吸，更不能承诺大二波。

**配套情绪闸门：0.54166874 < 固定阈值0.55，关闭。** 使用第六轮修正版保存的第4折闸门，这是第七轮历史评估实际配套的版本；并非旧未修版第5轮模型。该结果只表示这套研究规则的收紧条件，不等于市场必跌。阈值不因相差0.0083而临时降低。按这一组合，青岛即使个股分较高，今晚也未获得完整策略准入；光洋更不能靠换排序绕过原BASE三板约束。明日看到竞价上涨不会自动回填改写今晚闸门。

数据资格与真实局限：

- 70个原始字段中，两股各有2个缺失字段：`promotion_3to4`（昨日无三板、分母0）和市场`crowding_risk`（当前包未暴露）。股票模型严格使用保存的每个任务头训练中位数填补；闸门使用其原生缺失值分支。未知没有填成真实0或安全通过。80字段顺序与保存模型一致。
- 市场情绪质量、部分衍生维度复用原归档，继承其昨日37行而非事实38行的局限；本次可直接重算的整体/各板位晋级率和首板扩散，单独使用完整38只昨日池。整体晋级8/38、中位3/11、二进三2/8、首板进二5/26。原BASE输入和分数不回改。
- 行业扩散严格复用训练所用旧行业映射，不能临时把所有机器人概念塞入同组。光洋对应“汽车配件”，当前同组6只、首板4只、二板1只、三板1只，首板较昨日增加1只；青岛在该旧映射中未知，单独隔离，不能把模型中的“唯一”解释为真实题材身位认证。
- `total_limit_up`在该模型原代码中实际是二板以上候选数，故本次为8；涨停家数缩放字段另用59。行业成交额占比也沿用原构建器只汇总二板以上金额的口径。名称容易误解，不擅改原特征定义。
- 当日缺14:30与15:00同口径封单队列，Seal_Delta未知，不能把分钟累计成交量当加撤单，更不能宣布真龙特赦完成。

保存模型复核：第4折444条历史记录、五个任务头共2220个输出重新加载重放，最大绝对差异小于1e-12；模型文件SHA与原manifest一致。OOF CSV自身manifest哈希不一致的旧收据问题仍记录实际哈希；本次以当前模型、当前CSV实算逐项比对通过，不声称全部原交付元数据无误。重放一致只证明计算复现，不证明未来有效。

### 新旧模型怎样改变这次判断

| 股票 | 原BASE | R7结果 | 当前参与结论 |
|---|---:|---|---|
| 青岛食品 | 76.3，YES | 三板第1，综合0.5773；闸门未过 | 三板优先验证对象；R7完整组合下仅观察，一字也不能当可买 |
| 光洋股份 | 43.0，NO | 三板第2，综合0.3160；反弹头较高 | 原三板低分且无容量豁免，只观察；不以反弹分改低吸 |
| 襄阳轴承 | 47.0，NO | 五板，不适用 | 原人工高度观察保留；不能称R7推荐；尾盘队列和财报待核 |
| 东贝集团 | 45.5，NO | 二板，不适用 | 原人工二进三观察，不是R7第三名 |
| 文投控股 | 42.5，NO | 二板，不适用 | 保留原观察状态 |
| 凯众股份 | 40.6，NO | 二板，不适用 | 保留原观察状态 |
| 茂业商业 | 38.7，NO | 二板，不适用 | 保留原取消/不选状态 |
| 江顺科技 | 20.8，NO | 二板，不适用 | 保留原低优先级观察 |

另有10月28日专项竞价模型 `auction_dragon_model_20261028/model_auction_relay_dragon.joblib`，目标为已知开盘条件下当日封板，与三板收盘选未来高标不同。它需要10月29日真实开盘价才能构造相对昨高、昨均价、均线等输入，现在未知，故本次未调用它。未用10月28日开盘代替10月29日，也未假设明日高开。该竞价模型现有时点/回测问题也不能因本次试用被视为解决。

### 对10月29日及后续路径的预测

青岛食品：模型支持它在两只三板中优先争四，较光洋更偏向连续高度路径；人工反证是今日1.76%换手的一字使兑现压力延后，消费同伴没有全面恢复。29日若继续一字，只观察；若爆量开板，须看实际承接和九百、茂业、桂发祥的反馈。30日进入进一步高度争夺的前提是29日先通过真实分歧；11月2日、11月4日观察是否仍有群体认可，不预报连续每日涨停。

光洋股份：模型对立即连续高度的偏好较弱，18%反弹代理头较高，合理检验是未来可能出现分歧后的反抽，而不是预测明日必断板或之后必反包。29日仍看能否守住轴承题材地位，与襄阳可同强同弱；30日、11月2日和11月4日必须随实际断板/修复重新判断，不在三板信号上偷用后来信息。当前原BASE硬门未过，不设新增买入授权。

襄阳轴承：仍是人工观察的5进6高标；本次R7并没有给出它6板、7板或总龙的模型预测。东贝及三只首板的2/3/5日路径也仍属原人工情景，具体见下列执行卡，不伪装为新模型结果。

'''
report=report.replace('\n## 4 首板完整BASE','\n'+section+'\n## 4 首板完整BASE',1)
report=report.replace('## 5 最终连板前三','## 5 综合观察前三（人工结论，不冒充R7三名）',1)
report=report.replace('## 6 最终首板前三','## 6 首板人工前三（R7不适用，首板BASE未输出）',1)
pos=report.index('\n## 6 ')
report=report[:pos]+'''\n本次应用新模型后的区别：青岛获得三板相对首位支持，但配套闸门不通过，**仅作三板优先观察，不视为完整策略主选准入**；光洋仍未获准入。襄阳、东贝的原人工观察保留在各自适用范围，不据此把两只三板模型强凑成全市场前三。\n'''+report[pos:]
report=report.replace('## 7 次日竞价与执行卡','## 7 次日竞价与执行卡（人工观察预案；R7闸门未通过不自动启用）',1)
pos=report.index('\n所有价格按10月28日')
report=report[:pos]+'''\n**本次状态覆盖说明**：以下保留人工预案供验证，价格区间和成交额门槛均不是Gemini训练出的最优阈值。青岛与光洋在R7组合下当前只观察，不能因明日落入区间就宣称整套策略准入。其他板位和首板卡没有获得R7背书。\n'''+report[pos:]
report+='''

## 9 本次新增交付与核验

本目录保存运行前协议、70字段原始输入、每头中位数填补值、五头分数、完整8只适用性、配套闸门、历史重放与受保护文件哈希。新增拟合0、订单0；旧报告/BASE/账本原字节保持。模型仅用于用户授权的单独研究复盘，不替换生产版本。

本次文件： [实际模型分数](GEMINI_R7_SCORES.json)、[闸门](REGIME_GATE.json)、[输入与缺失](INPUTS_BEFORE_SCORES.json)、[复现核验](VERIFICATION.json)。原112只全池独立判断与51只首板全表沿用原归档；完整分母不因R7只适用两只而缩小。
'''
# Use absolute artifact links when the report is opened independently.
for name in ['GEMINI_R7_SCORES.json','REGIME_GATE.json','INPUTS_BEFORE_SCORES.json','VERIFICATION.json']:
    report=report.replace(']('+name+')',']('+str((OUT/name).resolve()).replace('\\','/')+')')
path=OUT/'20261028_Gemini第七轮模型复盘与1029预期.md'
with path.open('x',encoding='utf8') as f:f.write(report)
assert before==sha(original)
assert len(scores)==2 and gate['pass_gate'] is False
receipt=dict(formed_at_utc=now.isoformat(),report=str(path),report_sha256=sha(path),old_report_unchanged=True,
 actual_r7_inference_rows=2,full_relay_applicability_rows=8,first_board_full_table_rows=51,
 r7_combined_entry_gate='CLOSED_OBSERVE_ONLY',r7_native_rank=[r['code'] for r in scores],
 original_human_relay_observers=['000678','001219','601956'],first_observers=['003028','603897','000670'],
 no_fits=True,no_orders=True,calibrated_probabilities=False)
with (OUT/'DELIVERY_RECEIPT.json').open('x',encoding='utf8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print(json.dumps(receipt,ensure_ascii=False,indent=2))
