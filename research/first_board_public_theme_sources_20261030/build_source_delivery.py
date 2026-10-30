"""Reproducible no-label public-source ingestion and full-pool integration."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import sys

import joblib
import pandas as pd

from public_theme_sources import (ROOT, HERE, KEYS, sha, parse_ths, page_lines,
    read_archive, unmark, validate_detail_link, reviewed_memberships,
    attach_to_pool, as_provider, guard_sparse_features)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def read_csv(path, **kwargs):
    return pd.read_csv(path, dtype={'code': str}, float_precision='round_trip', **kwargs)


def link_results():
    result = {}
    for path in sorted((HERE / 'link_checks').glob('*.json')):
        obj = json.loads(path.read_text(encoding='utf-8'))
        text = obj['result'].get('value', '')
        if not isinstance(text, str):
            continue
        for part in text.split('-' * 80):
            m = re.search(r'Source: click\((\{[^}]+\})\)', part)
            if not m:
                continue
            source = json.loads(m[1])
            key = (source['ref_id'], source['id'])
            value = {'source_ref': key[0], 'detail_link_id': key[1],
                     'result': {'status': 'fulfilled', 'value': part},
                     'verification_path': str(path.relative_to(ROOT)), 'verification_sha256': sha(path)}
            if key in result and result[key]['result']['value'] != part:
                # A duplicated detail link in the original page must not be used
                # to silently assign two different dates. Validator checks dates.
                result[key]['duplicate_observation'] = True
            else:
                result[key] = value
    return result


def groups(now):
    names_path = ROOT / 'research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv'
    names = read_csv(names_path, usecols=['code', 'name']).drop_duplicates()
    name_map = names.groupby('name').code.agg(lambda x: sorted(set(x))).to_dict()
    cases = read_csv(ROOT / 'research/first_board_dynamic_narrative_review_20261030/corrected/CASE_POOL_NO_LABELS.csv')
    for row in cases.itertuples():
        name_map.setdefault(row.name, []).append(row.code)
    name_map = {n: sorted(set(c)) for n, c in name_map.items()}
    specs = []
    # Resolve the exact stock links in the author's original article; this adds
    # identities, never substitutes current prices or new group members.
    identity_path = HERE/'raw/forum_member_code_links.json'
    _, identity_text = read_archive(identity_path)
    _, forum_text = read_archive(HERE/'raw/forum_2vMRu2sQ4BP.json')
    linked_codes={}
    for part in identity_text.split('-'*80):
        source=re.search(r'Source: click\((\{[^}]+\})\)',part)
        target=re.search(r'Redirected to URL: https://www\.tgb\.cn/quotes/[a-z]+(\d{6})',part)
        if not source or not target:
            continue
        origin=json.loads(source[1])
        if origin['ref_id']!='turn130view0':
            continue
        label=re.search(r'【'+str(origin['id'])+r'†([^†】]+)(?:†[^】]+)?】',forum_text)
        if label:
            linked_codes[re.sub(r'\s+','',label[1])]=(target[1],origin['id'])

    def add(file, group, event, displayed, author, span, members, relation, counter):
        path = HERE / 'raw' / (file + '.json')
        converted = []
        for source_name, name, explicit_code in members:
            candidates = name_map.get(name, []) if name else []
            link_identity = linked_codes.get(name) if file=='forum_2vMRu2sQ4BP' else None
            code = explicit_code or (link_identity[0] if link_identity else None) or (candidates[0] if len(candidates) == 1 else None)
            converted.append({'source_name': source_name, 'name': name, 'code': code,
                'code_resolution': 'EXPLICIT_SOURCE_CODE' if explicit_code else
                    'ORIGINAL_AUTHOR_STOCK_LINK_TARGET' if link_identity else
                    'UNIQUE_EXISTING_NAME_MAP_NOT_IDENTITY_LABEL' if code else 'UNRESOLVED',
                'identity_source_path':str(identity_path.relative_to(ROOT)) if link_identity else None,
                'identity_source_sha256':sha(identity_path) if link_identity else None,
                'identity_original_link_id':link_identity[1] if link_identity else None})
        specs.append({'source_file': str(path.relative_to(ROOT)), 'source_sha256': sha(path),
            'theme_id': group, 'event_date': event, 'displayed_publication_time': displayed,
            'publication_time': None, 'publication_timezone_verified': False,
            'author': author, 'source_span': span, 'members': converted, 'body_scope': 'original_body',
            'relation': relation, 'reviewed_at': now, 'complete_within_reviewed_group': True,
            'counterevidence': counter})

    _, text = read_archive(HERE / 'raw/forum_1781454602.json')
    span = next(t for n, t in page_lines(text) if n == 57)
    add('forum_1781454602', 'FORUM:1781454602:NAME_CHAIN', '2026-10-12',
        '2026-10-12 17:28:51; site timezone not independently verified', '抄底冠军1573', span,
        [('三羊', None, None), ('九阳', '九阳股份', '002242'),
         ('百洋', '百洋股份', '002696'), ('圣阳', '圣阳股份', '002580')],
        'name_narrative_hypothesis_not_causal_edge', '单帖链条，不证明完整成员、父子因果或市场共识；三羊未解析。')

    _, text = read_archive(HERE / 'raw/forum_2vvUuPqu2Tl.json')
    begin, end = text.index('L59:'), text.index('L65:')
    add('forum_2vvUuPqu2Tl', 'FORUM:2vvUuPqu2Tl:Y_CHEMICAL', '2026-09-30',
        '26-09-30 16:39; site timezone not independently verified', '延边刺客', text[begin:end],
        [('新亚制程','新亚制程',None),('建业股份','建业股份',None),('兴业股份','兴业股份',None),
         ('九阳','九阳股份',None),('均瑶','均瑶健康',None)],
        'name_narrative_hypothesis_not_causal_edge', '作者明确先当气氛组、脱离后再确认；未将所有Y同音股票自动纳入。')

    _, text = read_archive(HERE / 'raw/forum_2vMRu2sQ4BP.json')
    span = text[text.index('三、主线归因'):text.index('四、')]
    media = ['新华传媒','中文在线','掌阅科技','芒果超媒','吉视传媒','上海电影','龙版传媒',
             '博纳影业','欢瑞世纪','新经典','新华文轩','粤传媒','慈文传媒']
    add('forum_2vMRu2sQ4BP','FORUM:2vMRu2sQ4BP:MEDIA13','2026-10-09',
        '2026-10-10 21:46; body says generated 2026-10-09 18:10; publication wins',
        '率土之滨非魔也',span,[(n if n!='粤传媒' else '粤 传 媒',n,None) for n in media],
        'author_theme_group_not_confirmed_leadership', '13为该作者列举组，不是完整市场；文中动因仅4条，其余按概念聚类；发布日期晚于事件日。')
    for name, start, stop, members in [
        ('BATTERY6','第二聚落','第三聚落',['时代万恒','紫竹高科','雄韬股份','丰元股份','领湃科技','科力远']),
        ('SOFTWARE6','第三聚落','三大聚落',['大智慧','绿盟科技','天融信','麒麟信安','格尔软件','中新赛克'])]:
        part = span[span.index(start):span.index(stop)]
        add('forum_2vMRu2sQ4BP','FORUM:2vMRu2sQ4BP:'+name,'2026-10-09',
            '2026-10-10 21:46; site timezone not independently verified','率土之滨非魔也',part,
            [(n,n,None) for n in members],'author_theme_group_not_confirmed_leadership',
            '完整保留原列举组；不采信作者资金身份、买卖建议或自报成功率，不与传媒机械互斥。')

    for file, line, ident, shown, author in [
        ('ths_news_military_20261016',53,'MILITARY','2026-10-16 10:18:42; mobile display differs by 8h','财闻／同花顺转载'),
        ('ths_news_software_20261016',44,'SOFTWARE','2026-10-16 09:34:21; site timezone not independently verified','同花顺7x24快讯')]:
        _, text = read_archive(HERE / 'raw' / (file+'.json'))
        part = next(t for n,t in page_lines(text) if n==line)
        members = [(n,n,c) for n,c in re.findall(r'【\d+†([^†【】（）]+)（(\d{6})）†stockpage\.10jqka\.com\.cn】',part)]
        members += [(n,n,c) for n,c in re.findall(r'([\u4e00-\u9fff]+)\((\d{6})\.BJ\)',part)]
        # Explicit stock links only: sector indices and fund codes are excluded.
        members = list(dict.fromkeys(members))
        add(file,'THS_NEWS:20261016:'+ident,'2026-10-16',shown,author,part,members,
            'reported_common_theme_not_lead_lag_causality',
            '保留全部列举成员含未涨停股；盘中表现不是收盘结果，时区及历史版本未验证。')
    claims = []
    for spec in specs:
        claims += reviewed_memberships(spec, ROOT / spec['source_file'])
    return specs, claims, names_path


def main(out):
    out.mkdir(exist_ok=False)
    now = datetime.now(timezone.utc).isoformat()
    checks = link_results()
    events, statuses = [], []
    for path in sorted((HERE/'raw').glob('ths_[0-9]*.json')):
        rows, status = parse_ths(path)
        statuses.append(status)
        for row in rows:
            v = checks.get((row['source_ref'],row['detail_link_id']))
            row['event_date_binding'] = ('DETAIL_LINK_DATE_VERIFIED' if v and validate_detail_link(row,v)
                else 'DETAIL_LINK_UNVERIFIED_OR_MISMATCH')
            if v:
                row.update({k:v[k] for k in ('verification_path','verification_sha256')})
            events.append(row)
    specs, claims, names_path = groups(now)
    write_json(out/'THS_DATED_EVENTS.json',events)
    write_json(out/'SOURCE_FETCH_STATUS.json',statuses)
    write_json(out/'REVIEWED_GROUP_SPECS.json',specs)
    write_json(out/'REVIEWED_GROUP_MEMBERS.json',claims)
    summary = {}
    sources = {'CASE92':ROOT/'research/first_board_dynamic_narrative_review_20261030/corrected/CASE_POOL_NO_LABELS.csv',
               'FULL7254':ROOT/'research/first_board_top3_40_20261030/results/FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv'}
    for name,path in sources.items():
        original = read_csv(path)
        joined = attach_to_pool(original,events,claims)
        pd.testing.assert_frame_equal(original,joined[original.columns],check_exact=True)
        joined.to_csv(out/f'{name}_PUBLIC_THEME_CONTEXT_NO_LABELS.csv',index=False,encoding='utf-8-sig')
        summary[name] = {'rows':len(original),'dates':int(original.signal_date.nunique()),
            'ths_same_date_rows':int(joined.public_ths_event_count.gt(0).sum()),
            'reviewed_group_same_date_rows':int(joined.public_reviewed_group_count.gt(0).sum()),
            'either_context_rows':int(joined.public_theme_context_status.ne('UNKNOWN').sum()),
            'actual_historical_eligible_rows':int(joined.public_context_actual_historical_available.sum()),
            'original_columns_exactly_unchanged':True}
        joined.assign(has_ths=joined.public_ths_event_count.gt(0),has_group=joined.public_reviewed_group_count.gt(0)).groupby('signal_date').agg(
            candidates=('code','size'),ths_same_date=('has_ths','sum'),groups_same_date=('has_group','sum')).to_csv(
                out/f'{name}_DAILY_COVERAGE.csv',encoding='utf-8-sig')
    # Materialize the existing mathematical feature interface over every original
    # candidate, without fitting coefficients or using any future outcome.
    state_path = ROOT/'research/gemini_codex_joint_theme_feedback_design_20261021T074448Z/CAUSAL_PRICE_STATES_WITHOUT_TARGETS.joblib'
    state = joblib.load(state_path)
    provider = as_provider(events)
    provider['calendar_qualified'] = provider.signal_date.isin(state['sessions'])
    source_export = provider.copy(); source_export['themes']=source_export.themes.map(lambda x:json.dumps(x,ensure_ascii=False))
    source_export.to_csv(out/'THS_PROVIDER_ADAPTER_NO_LABELS.csv',index=False,encoding='utf-8-sig')
    feature_path = ROOT/'research/first_board_dynamic_rank_20261030/dynamic_features.py'
    spec = importlib.util.spec_from_file_location('existing_dynamic_theme_features',feature_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    full_pool = read_csv(sources['FULL7254'])
    subset_features = mod.build(full_pool[KEYS],provider[provider.calendar_qualified],state['states'],state['sessions'])
    features = guard_sparse_features(subset_features,mod.FIELDS)
    features.to_csv(out/'FULL7254_PUBLIC_THEME_FEATURES_NO_LABELS.csv',index=False,encoding='utf-8-sig')
    summary['numeric_interface'] = {'candidate_rows':len(features),
        'theme_context_rows':int(features.theme_count.gt(0).sum()),
        'finite_fields':{c:int(features[c].notna().sum()) for c in mod.FIELDS},
        'observed_subset_finite_fields_NOT_FULL_UNIVERSE':{c:int(features['observed_subset_'+c].notna().sum()) for c in mod.FIELDS},
        'universe_coverage_gate_enabled':True,
        'fits':0,'ranking_changed':False,'probability_claim':False,
        'sampling_scope':'case-day selected securities, sparse historical attachment; NOT a fair all-history source sampling experiment'}
    summary.update({'created_at':now,'queried_securities':len(statuses),
        'fetch_status':dict(Counter(x['status'] for x in statuses)),
        'parsed_events':len(events),'date_binding':dict(Counter(x['event_date_binding'] for x in events)),
        'provider_events_on_existing_calendar':int(provider.calendar_qualified.sum()),
        'provider_events_outside_existing_calendar':int((~provider.calendar_qualified).sum()),
        'reviewed_groups':len(specs),'group_members_including_unresolved':len(claims),
        'unresolved_member_count':sum(x['code'] is None for x in claims),
        'disallowed_uses':['no original forecast rewrite','no proof of 40 percent','no automatic name matching bonus',
                           'no winner-only rule','no forum publication date from body-generated time'],
        'ths_direct_http_probe':'requests returned 403; ordinary public web read used, no challenge bypass',
        'current_full_history_benchmark_unchanged':True})
    bindings = {str(p.relative_to(ROOT)):sha(p) for p in list((HERE/'raw').glob('*.json'))+list((HERE/'link_checks').glob('*.json'))}
    for p in [Path(__file__),HERE/'public_theme_sources.py',names_path,state_path,feature_path,*sources.values()]:
        bindings[str(p.relative_to(ROOT))]=sha(p)
    write_json(out/'INPUT_BINDINGS.json',bindings)
    write_json(out/'DELIVERY.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'results')
    main(parser.parse_args().output.resolve())
