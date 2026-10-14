"""Rebuild official amount observations by response date, category and stated unit."""
from __future__ import annotations
from datetime import datetime,timezone
from decimal import Decimal
import html
import json
from pathlib import Path
import re
import pandas as pd
from market_interval_research_v300 import sha,write_json,CALENDAR,ROOT
from collect_exchange_turnover_v301 import request_spec

SSE_SCHEMA=ROOT/'data/v301/turnover_sse_schema_20261003_v1'


def clean(value):
    return re.sub(r'\s+','',re.sub(r'<[^>]+>','',html.unescape(str(value))))


def money(value,unit):
    text=str(value).strip()
    if not re.fullmatch(r'(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?',text):
        raise ValueError('amount is missing or not a nonnegative decimal')
    amount=Decimal(text.replace(',',''))
    if unit not in ['元','亿元']: raise ValueError('explicit currency unit required')
    factor=Decimal(100000000) if unit=='亿元' else Decimal(1)
    precision=Decimal(10)**amount.as_tuple().exponent*factor
    return amount*factor,precision


def category_row(venue,day,key,value,unit,scope):
    yuan,precision=money(value,unit)
    return {'date':day,'venue':venue,'category':key,'amount_yuan':str(yuan),
        'amount_yi_yuan':float(yuan/Decimal(100000000)), 'reported_unit':unit,
        'reported_amount':str(value),'reported_precision_yuan':str(precision),
        'scope':scope,'excludes_st':False,'genuine_historical_availability_certified':False}


def check_sum(total,parts):
    # All amounts are published rounded; no synthetic equality by modifying a component.
    difference=abs(Decimal(total['amount_yuan'])-sum(Decimal(p['amount_yuan']) for p in parts))
    tolerance=sum(Decimal(p['reported_precision_yuan']) for p in [total,*parts])/2
    if difference>tolerance: raise ValueError('stock category amount reconciliation conflict')


def parse_sse(payload,day):
    if (not isinstance(payload,dict) or payload.get('sqlId')!='COMMON_SSE_SJ_GPSJ_CJGK_MRGK_C'
            or payload.get('actionErrors') or payload.get('fieldErrors') or payload.get('isPagination')!='false'):
        raise ValueError('SSE response schema/errors/pagination invalid')
    rows=payload.get('result')
    if not isinstance(rows,list) or not rows: raise ValueError('SSE daily rows unavailable')
    code_map={'01':('MAINBOARD_A','SSE_MAINBOARD_A_INCLUDING_ST'),
              '02':('MAINBOARD_B','SSE_MAINBOARD_B'),
              '03':('STAR_BOARD','SSE_STAR_BOARD_INCLUDING_ST_CDR_NOT_SEPARATELY_DISCLOSED'),
              '17':('STOCKS_TOTAL','SSE_STOCKS_A_AND_B')}
    known={};seen=set()
    for row in rows:
        code=row.get('PRODUCT_CODE')
        if code in seen: raise ValueError('SSE duplicate product category')
        seen.add(code)
        if row.get('TRADE_DATE')!=day.replace('-',''): raise ValueError('SSE response date mismatch')
        if code not in set(code_map)|{'11'}: raise ValueError('SSE unrecognized product category')
        if code in code_map:
            key,scope=code_map[code]
            known[key]=category_row('SSE',day,key,row.get('TRADE_AMT'),'亿元',scope)
    if set(known)!={v[0] for v in code_map.values()}: raise ValueError('SSE required category missing')
    check_sum(known['STOCKS_TOTAL'],[known[k] for k in ['MAINBOARD_A','MAINBOARD_B','STAR_BOARD']])
    return list(known.values())


def parse_szse(payload,day):
    if not isinstance(payload,list): raise ValueError('SZSE list response required')
    selected=[r for r in payload if r.get('metadata',{}).get('tabkey')=='tab1']
    if len(selected)!=1: raise ValueError('SZSE main tab missing/duplicated')
    tab=selected[0];meta=tab['metadata']
    if tab.get('error') or meta.get('catalogid')!='1803_sczm' or meta.get('subname')!=day:
        raise ValueError('SZSE error/catalog/actual report date mismatch')
    if meta.get('pagesize')!=-1: raise ValueError('SZSE unexpected paginated category table')
    conditions=[c for c in meta.get('conditions',[]) if c.get('name')=='txtQueryDate']
    if len(conditions)!=1 or conditions[0].get('defaultValue')!=day:
        raise ValueError('SZSE response query-date mismatch')
    header=clean(meta.get('cols',{}).get('cjje',''))
    if header not in ['成交金额(亿元)','成交金额(元)']: raise ValueError('SZSE amount unit unknown')
    unit='亿元' if header=='成交金额(亿元)' else '元'
    keys={'股票':('STOCKS_TOTAL','SZSE_STOCKS_A_AND_B_AND_REPORTED_CDR'),
          '主板A股':('MAINBOARD_A','SZSE_MAINBOARD_A_INCLUDING_ST'),
          '主板B股':('MAINBOARD_B','SZSE_MAINBOARD_B'),
          '创业板A股':('CHINEXT_A','SZSE_CHINEXT_A_INCLUDING_ST'),
          '主板CDR':('MAINBOARD_CDR','SZSE_MAINBOARD_CDR'),
          '创业板CDR':('CHINEXT_CDR','SZSE_CHINEXT_CDR')}
    known={};seen=set();rows=tab.get('data')
    if not isinstance(rows,list) or not rows: raise ValueError('SZSE rows unavailable')
    for r in rows:
        label=clean(r.get('lbmc',''))
        if label in seen: raise ValueError('SZSE duplicate category')
        seen.add(label)
        if label in keys:
            key,scope=keys[label];known[key]=category_row('SZSE',day,key,r.get('cjje'),unit,scope)
    if not {'STOCKS_TOTAL','MAINBOARD_A','MAINBOARD_B','CHINEXT_A'}.issubset(known):
        raise ValueError('SZSE required stock component missing')
    check_sum(known['STOCKS_TOTAL'],[r for k,r in known.items() if k!='STOCKS_TOTAL'])
    return list(known.values())


def schema_evidence():
    m=json.loads((SSE_SCHEMA/'manifest.json').read_text(encoding='utf-8'))
    item=next(r for r in m['requests'] if r['key']=='sse_stockdata')
    raw=SSE_SCHEMA/item['raw_path']
    if item['status']!='RECEIVED' or item.get('http_status')!=200 or sha(raw)!=item['sha256']:
        raise ValueError('official SSE field-definition source invalid')
    text=raw.read_text(encoding='utf-8')
    for phrase in ['股票17 主板A 01 主板B 02 科创板 03 股票回购 11','v.TRADE_AMT.str_toFixed(2)','成交金额(亿元)']:
        if phrase not in text: raise ValueError('SSE official schema evidence missing')
    return {str(raw.resolve()):sha(raw),str((SSE_SCHEMA/'manifest.json').resolve()):sha(SSE_SCHEMA/'manifest.json')}


def aggregate(rows,days):
    result=[];by_key={(r['date'],r['venue'],r['category']):r for r in rows}
    for day in days:
        required=[by_key.get((day,*key)) for key in [('SSE','MAINBOARD_A'),('SZSE','MAINBOARD_A'),
                                                   ('SSE','STAR_BOARD'),('SZSE','CHINEXT_A')]]
        row={'date':day,'scope_mainboard':'SHSZ_MAINBOARD_A_INCLUDING_ST',
             'scope_stock_categories':'SHSZ_MAINBOARD_A_PLUS_STAR_BOARD_PLUS_CHINEXT_A_INCLUDING_ST_CDR_SCOPE_UNRESOLVED_EXCLUDING_B_BSE',
             'matches_original_non_st_pool_scope':False,'st_excluded_amount':None,
             'genuine_historical_availability_certified':False}
        for key,components in [('mainboard_a_amount_yi',required[:2]),('shsz_stock_category_amount_yi',required)]:
            known=all(p is not None for p in components)
            row[key]=float(sum(Decimal(p['amount_yuan']) for p in components)/Decimal(100000000)) if known else None
            row[key+'_status']='OFFICIAL_COMPONENT_SUM_INFERRED_RETROSPECTIVE' if known else 'COMPONENT_UNAVAILABLE'
            row[key+'_available_at_utc']=max(p['available_at_utc'] for p in components) if known else None
        result.append(row)
    return pd.DataFrame(result)


def normalize(archives,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    sessions=json.loads(CALENDAR.read_text(encoding='utf-8-sig'))['sessions']
    sources=schema_evidence();sources.update({str(CALENDAR.resolve()):sha(CALENDAR),str(Path(__file__).resolve()):sha(__file__)})
    observations=[];audit=[];seen=set();days=set()
    for archive in map(Path,archives):
        manifest_path=archive/'manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['calendar_sha256']!=sha(CALENDAR): raise ValueError('calendar source changed')
        sources[str(manifest_path.resolve())]=sha(manifest_path)
        for path,digest in manifest.get('parent_source_hashes',{}).items():
            if sha(path)!=digest: raise ValueError('parent collection source altered')
            sources[path]=digest
        for record in manifest['requests']:
            day=record['requested_date'];venue=record['venue'];days.add(day)
            url,params,_=request_spec(venue,day)
            if record.get('url')!=url or record.get('parameters')!=params:
                raise ValueError('receipt official origin/query parameters mismatch')
            if (day,venue) in seen or day not in sessions: raise ValueError('duplicate/nonofficial receipt date')
            seen.add((day,venue));entry={'date':day,'venue':venue,'receipt_status':record['status']}
            try:
                if record['status']!='RECEIVED' or record['http_status']!=200: raise ValueError('source receipt unavailable')
                raw=archive/record['raw_path']
                if sha(raw)!=record['sha256']: raise ValueError('raw response altered')
                sources[str(raw.resolve())]=sha(raw)
                payload=json.loads(raw.read_bytes())
                values=(parse_sse if venue=='SSE' else parse_szse)(payload,day)
                received=datetime.fromisoformat(record['received_at_utc'])
                started=datetime.fromisoformat(record['requested_at_utc'])
                completed=datetime.fromisoformat(manifest['recorded_at_utc'])
                if not started<=received<=completed or received.tzinfo is None:
                    raise ValueError('source receipt clock invalid')
                for value in values:
                    value.update(available_at_utc=record['received_at_utc'],
                                 source_manifest=str(manifest_path.resolve()),raw_sha256=record['sha256'])
                observations.extend(values);entry.update(status='DATE_UNIT_COMPONENTS_VERIFIED_RETROSPECTIVE',rows=len(values))
            except (ValueError,KeyError,TypeError,json.JSONDecodeError) as error:
                entry.update(status='UNAVAILABLE_RETAINED',error=str(error))
            audit.append(entry)
    pd.DataFrame(observations).to_csv(out/'exchange_category_amounts.csv',index=False)
    combined=aggregate(observations,sorted(days));combined.to_csv(out/'market_amounts_NOT_ORIGINAL_POOL_SCOPE.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'source_audit.csv',index=False)
    report={'version':'v301-official-turnover-observations-2-scope-corrected','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'OFFICIAL_DAILY_COMPONENTS_WITH_EXPLICIT_SCOPE_RETROSPECTIVE', 'requested_dates':len(days),
        'requested_venue_days':len(audit),'valid_venue_days':sum(r['status']=='DATE_UNIT_COMPONENTS_VERIFIED_RETROSPECTIVE' for r in audit),
        'rows':len(observations),'complete_mainboard_dates':int(combined.mainboard_a_amount_yi.notna().sum()),
        'complete_stock_category_dates':int(combined.shsz_stock_category_amount_yi.notna().sum()),
        'scope_matches_original_non_st_pool':False,'same_scope_turnover_slot_complete':False,
        'genuine_historical_availability_certified':False,'forward_evidence':False,
        'core_upgrade_authorized':False,'stable_60_proved':False,'goal_complete':False,
        'source_hashes':sources,'artifact_hashes':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/'report.json',report)
    return {k:report[k] for k in ['status','requested_dates','valid_venue_days','rows','complete_mainboard_dates','complete_stock_category_dates']}
