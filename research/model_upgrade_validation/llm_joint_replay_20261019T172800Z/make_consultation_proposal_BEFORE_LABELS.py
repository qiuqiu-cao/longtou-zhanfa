from run_joint_replay import *
import copy
c=read(HERE/'codex_independent_BEFORE_GEMINI_AND_LABELS.json')
g=read(HERE/'gemini_first_received_snapshot_BEFORE_DISCUSSION.json')
p=read(HERE/'review_packet_SIGNAL_ONLY_v2.json')
changes={
('D01','S05'):('YES','采纳Gemini对连续换手且量0.88倍的延续解释，但“核心”仅是假设，不能认证资金抱团。'),
('D02','S01'):('YES','采纳升温109涨停环境中三板、连续三日换手且量1.01倍的延续倾向；保留平开低-2.5及宽振幅反证。'),
('D02','S07'):('UNKNOWN','保留当前窄幅与缩量强形态，也接受其此前反复断板/冲高失败的供给解释；不因缩量就断言无资金。'),
('D03','S07'):('YES','采纳强开后低点仍4.81的相对强度；量1.75倍与S01的1.80倍相近，两只可以共同晋级，非必二选一。'),
('D04','S02'):('UNKNOWN','量1.30倍与9.98%换手较温和，接受Gemini的低位竞争假设；但8.54%波幅及43.48%炸板率使正向不足。'),
('D04','S07'):('UNKNOWN','接受五板换手存活有延续路径，撤回仅以深分歧偏断的单向倾向；但26.74%换手、低-1.93不足认证真龙。'),
('D05','S04'):('YES','采纳二板4.86%波幅、7.61%换手、量1.31倍的相对延续，允许在历史资料缺口下依据已知信号作预测；不认证承接。'),
('D05','S06'):('UNKNOWN','量1.22倍与中位三板支持温和交换解释，但18.69%换手/近10%波幅及56跌停反证仍重，原始OHLC缺失。'),
('D06','S08'):('UNKNOWN','采纳唯一四板可能换手接替的假设，不把巨量自动判死；但9.56倍量、18.77%波幅、历史OHLC缺失不能认证主动核心。'),
('D07','S03'):('YES','采纳唯一五板在完整池存活、7.27%换手的相对高度延续假设；日线及量比缺口明确保留，资金抱团未证实。'),
('D07','S05'):('UNKNOWN','19.81%换手及量1.08倍也可解释为持续交换，接受Gemini反证，不再单向偏断；9.02%波幅和53.66%炸板仍阻止确认。'),
('D08','S09'):('YES','采纳两日正开正低点、量1.77倍及完整同位中较高成交相对量的延续解释；不能称真实流入最强或大资金核心。'),
('D09','S07'):('UNKNOWN','换手6.09%、量0.94倍有序，撤回仅以非一价波幅偏断；但大成交相对第一不是主动进攻证明，负溢价和19跌停仍有冲突。'),
('D10','S02'):('YES','采纳近涨停开、低5.92、4.02%波幅的相对强度；三倍量对应9.43%绝对换手，不与极高换手一概否决。'),
('D10','S12'):('UNKNOWN','28.37%换手可由一日前9.02%低基数部分解释，更早首板19.05%；接受持续交换可能，但低1.82及3.15倍量不认证真龙。'),
('D11','S02'):('YES','采纳并列最高三板且量0.99倍、低点等于开盘3.98的相对连续形态；8板坍塌使跨周期身份仍未知。'),
('D11','S04'):('UNKNOWN','量1.07倍、15.69%换手并非异常膨胀，采纳高位持续交换可能；但开盘强度连续衰减且低1.80，不直接确认。'),
('D12','S03'):('YES','采纳七板多日换手后由39.41%回落34.88%、量0.89倍且市场溢价为正的延续解释；仍有连续负低点及宽幅反证。'),
('D12','S07'):('UNKNOWN','三板量0.99倍、换手13.54%支持有序交换，撤回仅因开盘衰减偏断；0.78开盘和9.14%波幅仍不足确认主导。'),
('D12','S10'):('YES','低点即开盘5.45、4.62%波幅，放量2.59倍对应14.70%换手；在正溢价与梯队增长背景采纳正向，未认证核心。')}
tops={'D01':'S05 S03 S02','D02':'S11 S01 S05','D03':'S04 S07 S01','D04':'S04 S05 S06','D05':'S07 S04 S06','D06':'S04 S11 S07','D07':'S03 S06 S05','D08':'S07 S06 S09','D09':'S08 S10 S04','D10':'S08 S02 S04','D11':'S09 S02 S03','D12':'S03 S01 S17'}
out=copy.deepcopy(c);out['reviewer']='Codex consultation proposal, not final joint agreement';out['created_at_utc']=now();out['other_reviewer_decisions_seen']=True
gl={(d['case'],r['id']):r for d in g['days'] for r in d['stocks']};cl={(d['case'],r['id']):r for d in c['days'] for r in d['stocks']};pl={(d['case'],r['id']):r for d in p['days'] for r in d['stocks']}
principles=[
'目标是次日收盘涨停；一价并非必晋级，但“开板难接/买不到”不能直接作次日断板证据，预测与执行分列。',
'成交额同池第一不证明主动买入、资金抱团、换手龙或共识；packet没有分时、盘口、题材/公告。禁止把假设写成事实。',
'float_cap_pctile只有同池相对名次，不足称超大市值；volume_vs_previous是成交量比，不是封单、净流入或换手比。',
'放量先看基数和价格分歧。D12-S17当前换手8.12%低于首板13.17%，6.65倍来自昨日1.22%低基数，日内低7.99/波幅2.06。',
'D10-S11当前日线只到T-8且价型资格不足，不能凭五板认证真龙或确定性最高；应保留未知或给具体合格反证。',
'完整同位允许共进、共败、分化；D04两五板、D12多三板不预设唯一胜者。最高高度、唯一活口或一价本身都不认证身份。',
'本提案接受Gemini若干真实换手路径，同时保留其主要反证。没有看结果；请反驳有证据的具体行，不机械附和。']
out['consultation_principles']=principles
for d in out['days']:
    case=d['case'];d['top3']=tops[case].split();d['market_reason']+=' 联合提议：市场事实优先，原始题材与流向未知；用双方已核价量分歧审阅，不把资金叙事当真值。'
    for r in d['stocks']:
        key=(case,r['id']);a=cl[key];b=gl[key]
        r['codex_independent']=a['decision'];r['gemini_independent']=b['decision']
        if key in changes:r['decision'],r['reason']=changes[key]
        elif a['decision']==b['decision']:r['reason']='双方独立方向一致；'+a['reason']+' Gemini支持/反证：'+b['reason']
        else:r['reason']='暂保留Codex倾向，待Gemini逐项反驳。'+a['reason']+' Gemini不同解释：'+b['reason']+' 角色/资金因果仅作假设。'
write(HERE/'CODEX_joint_proposal_BEFORE_LABELS.json',out)
lines=['# 遮蔽答案的双模型讨论提案（非最终共同决定）','']+['- '+v for v in principles]+['',
'Codex和Gemini独立稿均已冻结。下列显示所有方向分歧；20处明确修正只是协商提议，结果尚未读取。其余相同方向仍保留逐股原理由。','',
'|案例/股号|板数/一价|换手%/波幅%/量比|Codex|Gemini|共同提议|提议理由|','|---|---|---|---|---|---|---|']
for d in out['days']:
    for r in d['stocks']:
        key=(d['case'],r['id']);a=cl[key];b=gl[key];s=pl[key]
        if a['decision']!=b['decision'] or key in changes:
            lines.append(f"|{key[0]}/{key[1]}|{s['boards']}/{s['one_price']}|{s['turnover_pct']}/{s['amplitude_pct']}/{s['volume_vs_previous']}|{a['decision']}|{b['decision']}|{r['decision']}|{r['reason']}|")
lines+=['','拟议相对Top3：']+[f"- {k}: {v}" for k,v in tops.items()]
(HERE/'consultation_CODEX_proposal_SIGNAL_ONLY.md').write_text('\n'.join(lines),encoding='utf-8')
write(HERE/'consultation_proposal_receipt_BEFORE_LABELS.json',{'created_at_utc':now(),'codex_sha256':sha(HERE/'codex_independent_BEFORE_GEMINI_AND_LABELS.json'),'gemini_first_snapshot_sha256':sha(HERE/'gemini_first_received_snapshot_BEFORE_DISCUSSION.json'),'proposal_sha256':sha(HERE/'CODEX_joint_proposal_BEFORE_LABELS.json'),'rows':122,'outcome_reads':0,'independent_disagreements':sum(cl[k]['decision']!=gl[k]['decision'] for k in cl),'explicit_proposed_changes_from_codex':len(changes)})
print(json.dumps({'disagreements':sum(cl[k]['decision']!=gl[k]['decision'] for k in cl),'changes':len(changes),'proposed_counts':pd.Series([r['decision'] for d in out['days'] for r in d['stocks']]).value_counts().to_dict()}))
