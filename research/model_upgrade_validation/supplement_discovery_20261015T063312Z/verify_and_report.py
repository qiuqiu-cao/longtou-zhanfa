"""Independently audit saved identity context; preserve original predictions."""
from collections import Counter
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import json
import sys
import tempfile
sys.path.insert(0,str(Path.cwd()))
import stock_judgement_review as o
import supplement_identity_discovery as identity
import relay_review_workbench as workbench

root=Path(__file__).parent
delivery=root/'delivery_final'
facts_path=delivery/'facts_with_identity_discovery.json'
review_path=delivery/'author_identity_completed.json'
facts,review=o.read(facts_path),o.read(review_path)
context=identity.checked(facts)
status=workbench.validate_completed(facts_path,review)
o.check_sources(facts['source_bindings'])
o.check_sources(review['source_bindings'])
start=o.read(root/'execution_start.json')
protected={p:o.sha(p)==sha for p,sha in start['protected_hashes'].items()}
assert all(protected.values())
assert facts['rows']==o.read(context['facts_path'])['rows']
assert {r['code'] for r in review['rows']}=={r['code'] for r in context['rows']}
assert all(r['promotion_judgement'] is None for r in review['rows'])

# Rebuild cents and native volume directly from the raw Tencent arrays, without
# using the discovery parser or its price_path calculation.
spec=o.read(context['evidence_path'])
calendar=o.read(o.read(spec['relation_bank_path'])['calendar_path'])['sessions']
checked_daily_cells=0
for row in context['rows']:
    path=row['daily_price_path']
    symbol=('sh' if row['code'].startswith('6') else 'sz')+row['code']
    raw=o.read(path['raw_path'])['data'][symbol]['day']
    raw_by_date={x[0]:x for x in raw}
    for observed in path['observations']:
        date=observed['date'];r=raw_by_date.get(date)
        if r is None:
            assert observed['status']=='UNKNOWN'
            continue
        prices=[int((Decimal(str(x))*100).quantize(Decimal('1'),rounding=ROUND_HALF_UP)) for x in r[1:5]]
        assert [observed[k] for k in ('open','close','high','low')]==prices
        assert observed['volume_native']==float(r[5])
        assert observed['one_price_session']==(len(set(prices))==1)
        previous=calendar[calendar.index(date)-1]
        prior=raw_by_date.get(previous)
        ratio=float(r[5])/float(prior[5]) if prior and float(prior[5])>0 else None
        assert observed['volume_ratio_vs_previous_official_session']==ratio
        checked_daily_cells+=1

# Discover historical anchor membership directly from the exact-date raw lists.
expected_anchors=set()
for snap in spec['pool_snapshots']:
    if snap['kind']!='EASTMONEY_CLOSING_POOL':continue
    data=o.read(snap['path'])['data']
    assert str(data['qdate'])==snap['date'].replace('-','') and data['tc']==len(data['pool'])
    for r in data['pool']:
        if str(r['c']).startswith(('00','60')) and 'ST' not in r['n'].upper() and '退' not in r['n'] and r['lbc']>=2:
            expected_anchors.add(r['c'])
assert expected_anchors=={a['code'] for a in context['anchor_candidates']}
old_anchors={r.get('code') for r in o.read(spec['relation_bank_path'])['anchors'] if r.get('code')}
assert '600825' not in old_anchors and '600825' in expected_anchors
assert all(next(h for h in r['anchor_hypotheses'] if h['anchor_code']=='600825')['identity_hypothesis']=='THEME_RELAY_COMPETING'
    for r in context['rows'] if r['code'] in {'603096','002343'})
assert all(not h['source_connection_supported'] for r in context['rows'] if r['code']=='603968' for h in r['anchor_hypotheses'])

negative=[]
def rejected(name,fn):
    try:fn()
    except (ValueError,KeyError) as e:negative.append(dict(case=name,error_type=type(e).__name__,reason=str(e)))
    else:raise AssertionError('invalid input accepted: '+name)

changed=deepcopy(facts);changed['rows'].pop()
rejected('full_pool_stock_removed',lambda:identity.checked(changed))
changed_review=deepcopy(review)
next(r for r in changed_review['rows'] if r['code']=='600847')['identity_discovery_assessment']['rejected_or_pending_anchors'].pop()
rejected('parent_disposition_missing',lambda:identity.validate_review(facts,changed_review))
changed_review=deepcopy(review)
next(r for r in changed_review['rows'] if r['code']=='001318')['identity_discovery_assessment']['same_cycle_competitors']=[]
rejected('same_cycle_competitor_omitted',lambda:identity.validate_review(facts,changed_review))
changed_review=deepcopy(review);changed_review['rows'][0]['promotion_judgement']='PROMOTE'
rejected('diagnostic_cannot_create_forecast',lambda:workbench.validate_completed(facts_path,changed_review))
with tempfile.TemporaryDirectory() as t:
    corrupt=Path(t)/'context.json';saved=deepcopy(context);saved['pool_size']=0;o.write_new(corrupt,saved)
    bad=deepcopy(facts);bad['identity_discovery_context_path']=str(corrupt);bad['source_bindings']|=o.bind(corrupt)
    rejected('rehashing_changed_projection_not_enough',lambda:identity.checked(bad))
    output=Path(t)/'frozen.json'
    rejected('old_diagnostic_freeze_refused_by_original_gate',lambda:workbench.freeze(facts_path,review_path,output))
    assert not output.exists()
assert all(o.sha(p)==sha for p,sha in start['protected_hashes'].items())

tests=o.read(root/'final_targeted_tests_v2.json')
assert tests['failures']==tests['errors']==0
counts=Counter(r['identity_discovery_assessment']['identity_call'] for r in review['rows'])
qualified=sum(r['current_streak']['complete_closing_streak_checked'] for r in context['rows'])
accepted=sum(len(r['identity_discovery_assessment']['accepted_anchor_codes']) for r in review['rows'])
parents=sum(len(r['anchor_hypotheses']) for r in context['rows'])
previous=o.read('research/model_upgrade_validation/recall_followthrough_20261015T055212Z/completion_verification.json')
receipt=dict(completed_at=o.now().isoformat(),status='FULL15_PARENT_DISCOVERY_AND_INDEPENDENT_IDENTITY_REVIEW_VERIFIED',
    pool_size=len(context['rows']),historical_pool_snapshots=len(spec['pool_snapshots']),
    excluded_request_response_date_mismatches=len(o.read(root/'excluded_date_mismatch.json')['rows']),
    historical_anchor_candidates=len(expected_anchors),newly_recovered_parent='600825 新华传媒',
    old_manual_coded_anchor_count=len(old_anchors),parent_hypotheses=parents,
    candidates_with_nonempty_parent_hypotheses=sum(bool(r['anchor_hypotheses']) for r in context['rows']),
    author_accepted_parent_hypotheses=accepted,author_review_completed=len(review['rows']),
    source_bound_candidate_daily_paths=context['candidate_daily_paths_available'],
    fully_qualified_current_streaks=qualified,raw_daily_cells_independently_recomputed=checked_daily_cells,
    current_complete_provider_mainboard_closing_rows=len(context['current_complete_provider_closing_pool']),
    classification_counts=dict(counts),full_original_fact_rows_unchanged=True,
    protected_original_hash_checks=protected,source_hash_checks=len(facts['source_bindings']),
    completed_review_validation=status,engineering_tests_passed=tests['tests_run'],actual_negative_checks=negative,
    previous_training_cutoff_context_counts=previous['original_training_cutoff_path_context_counts'],
    current_context_is_historical_asof_training=False,independent_role_gold_labels_created=0,
    new_fits=0,new_formal_forecasts=0,new_formal_mature_samples=0,new_orders_or_fills=0,
    model_exposure=context['model_exposure'],measured_identity_accuracy=None,measured_recall_improvement=None,
    final_view_path=str((delivery/'review_final/review.html').resolve()))
o.write_new(root/'completion_final.json',receipt)
o.write_new(root/'workbench_source_compatibility_final.json',dict(
    previous_workbench_exact_preimage_path=str((root/'source_preimages/relay_review_workbench.py').resolve()),
    previous_workbench_sha256=o.sha(root/'source_preimages/relay_review_workbench.py'),
    current_workbench_sha256=o.sha('relay_review_workbench.py'),
    additive_features=['prepare identity evidence','discover','mandatory identity review when attached','score-masked view','compare revalidation'],
    protected_core_unchanged=all(protected.values()),original_fact_rows_unchanged=True,
    original_gates_bypassed=False,new_role_weights_deployed=False))

labels={'CYCLE_SUPPLEMENT_CANDIDATE':'周期补涨候选','OLD_FOLLOWER_RESTART':'旧跟风重起',
    'THEME_RELAY_COMPETING':'同题材接力竞争','RELATED_UNRESOLVED':'相关线索待核',
    'UNKNOWN':'身份未知','NOT_SUPPORTED':'否决已发现的弱关系，其他身份未知'}
lines=['# 补涨身份识别补强与完整15只结论','',
    '已经补上“从历史完整池找父锚—核启动与断板节点—比较同周期及同位候选—独立形成身份判断”的入口，并实际完成10月14日这15只的诊断审阅。此前遗漏的新华传媒600825已由归档扫描找回。','',
    f'本次读取6个日期快照，排除4份错日期响应，扫描{len(expected_anchors)}个历史高标候选；15只均有可从原始响应重建的不复权日线，{qualified}只本轮连续板路径完整。路畅科技、星帅尔在13日缺相邻官方交易日记录，继续保留复牌资格缺口。','',
    f'当前并非上下文全0：15份结构/日线上下文、{receipt["candidates_with_nonempty_parent_hypotheses"]}只存在父锚假设、{parents}条待审关系，15只均完成逐股取舍。5条关系被接受为研究假设，其余有逐项否决或待核理由。接受假设不等于认证父龙。','',
    '| 股票 | 独立身份区分 | 主要依据和实际边界 |','|---|---|---|']
bycode=o.unique_rows(review['rows'])
for row in sorted(context['rows'],key=lambda r:(-r['boards'],r['code'])):
    a=bycode[row['code']]['identity_discovery_assessment']
    lines.append(f'| {row["name"]} {row["code"]}（{row["boards"]}板） | {labels[a["identity_call"]]} | {a["relative_role_hypothesis"]} |')
lines += ['',
    '百洋与阳光现在分开：百洋先于九阳未续板启动，名称链/评论与1→4板时序相容；阳光早在9月30日出现过，13日重起，对应旧跟风重起，且本池身位落后于百洋。两者均没有已核资金迁移或唯一父龙证明。','',
    '新经典与慈文现在能进入新华传媒周期比较：两者都从9日首板走到14日4板，原新华传媒8/9/10板后13日未续。原作者“新补涨龙/伴飞”只作假设。14日两股都放量并收于日高，量约为前交易日7.906/3.545倍，单凭这两项不能宣布新经典控制全组。当前背景还保留新华文轩601811、中国科传601858首板；新华文轩与父锚新华传媒不是同一只股票。','',
    '已排除的语义误判：善水—新华的原文关系不转移给醋化；食品饲料添加剂不把醋化归为九阳消费补涨；消费电子不等同消费食品，储能温控不等同直接电池，单个“龙/丰/博/尔/阳”字不构成补涨证据。万里与时代仅保留同电池及时间节点相容的接力假设，其他电池历史锚逐项待核。','',
    '工作台prepare新增`--identity-evidence`，已有完整事实使用`discover`追加；每只须实际填写身份、父锚取舍、反证、完整同周期竞争和取消条件。freeze/compare先核本层，再运行原全部门。原事实行、数值分数、原方向和账本保持。完整逐股证据及反证见[审阅工作台](delivery_final/review_final/review.html)，机器上下文见[discovery.json](delivery_final/discovery/discovery.json)。','',
    f'验证：{tests["tests_run"]}项相关工程测试通过，{len(negative)}项实际错误输入均拒绝，原来源哈希{len(facts["source_bindings"])}项通过，日线{checked_daily_cells}个股票日直接从原始数组独立复算；原受保护核心与三份账本哈希不变。这些是链路与事实校验，不能算识别准确率提升。','',
    '**训练仍需区分分母：**当前诊断上下文已经补入；原2025-06-30/2025-12-31/2026-08-31三个训练截止前完整时点上下文仍为0/0/0，独立SUPPLEMENT身份金标准新增0。本次没有数值拟合，不能把当下形成的判断、作者称号或次日是否晋级直接当历史补涨标签。旧9245条的合格晋级标签研究仍可按既定证据队列推进，并未被全面禁止。','',
    '本批拟合0、新正式预测0、新正式成熟0、新成交0；旧日期已看模型，只追加诊断，不回填事前预测。识别准确率与Recall改善仍待新日期前向验证。中间试跑目录、错误和源码前像保留；最后加固日线请求与替代页生成须在信号日收盘后，真实原件均通过；当前有效交付以delivery_final及completion_final.json为准。','']
(root/'补涨身份识别补强与完整15只结论.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({k:receipt[k] for k in ['pool_size','historical_anchor_candidates','parent_hypotheses','source_bound_candidate_daily_paths','fully_qualified_current_streaks','source_hash_checks','engineering_tests_passed','new_fits']},ensure_ascii=False))
