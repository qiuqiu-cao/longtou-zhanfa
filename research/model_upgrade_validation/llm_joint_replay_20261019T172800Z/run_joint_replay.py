"""Isolated, immutable two-LLM historical replay. No fitting or production writes."""
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import sys
import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUNDLE = ROOT / 'research/model_upgrade_validation/historical_symbolic_context_20261015T081200Z/latest15_prepared_v2/engine_context_WITH_CURRENT15.joblib'
SEED = 'human-request-two-actual-llms-historical-consultation-20261019-v1'
PERIODS = [('2025H2', '2025-07-01', '2025-12-31'), ('2026JanAug', '2026-01-01', '2026-08-31'), ('2026Sep', '2026-09-01', '2026-09-30')]

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.file_digest(open(p, 'rb'), 'sha256').hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, (np.integer, np.bool_)): return x.item()
    if isinstance(x, (float, np.floating)): return round(float(x), 4) if np.isfinite(x) else None
    if pd.isna(x): return None
    return x
def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f: json.dump(clean(x), f, ensure_ascii=False, indent=2, allow_nan=False)
def score(s): return hashlib.sha256((SEED + ':' + s).encode()).hexdigest()
def num(v):
    try: return float(v) if pd.notna(v) and np.isfinite(float(v)) else None
    except (TypeError, ValueError): return None
def ratio(a, b):
    a, b = num(a), num(b)
    return a / b if a is not None and b is not None and b > 0 else None

def prepare():
    protocol = {
        'created_at_utc': now(), 'experiment': 'CODEX_GEMINI38_HISTORICAL_CONSULTATION_V1',
        'human_request': '两个真实模型用以往当时数据各自判断、商量、共同决策后揭晓结果',
        'seed': SEED, 'sample_rule': 'Each of three fixed periods: select the four signal dates with the lowest SHA256(seed:period:date). Keep the entire archived relay pool on each selected date. Order cases by a separate hash. No label, score, performance or pool-size selection.',
        'periods': PERIODS, 'days_per_period': 4,
        'main_event': 'Next adjacent official session CLOSE limit-up, using archived promotion target corroborated by unadjusted exact-cent 10% prices; missing/conflicting qualification remains UNKNOWN.',
        'secondary_event': 'Next adjacent official session intraday TOUCH limit-up, separately counted.',
        'reviewers': ['Codex (this live assistant)', 'Actual Antigravity Gemini 3.8 Flash High'],
        'arms': ['codex_independent', 'gemini_independent', 'discussed_joint'],
        'reference_arms': ['archived_ROLL252_strict_gt_0.5', 'always_no', 'height_rank_Top3'],
        'sequence': ['protocol and selection frozen', 'explicit signal-only packet; dates/codes/names masked; no target or model score', 'each reviewer saves independent all-pool decisions before seeing other decisions', 'one label-hidden consultation and all-pool joint decision freeze', 'only then extract labels and evaluate all arms'],
        'metrics': ['TP/FP/TN/FN, unknown labels and abstentions', 'judged-known accuracy and full-known correct fraction', 'precision, recall, F1, balanced accuracy, coverage', 'Top1/Top3 with unknown slots retained', 'real daily non-one-price subset', 'paired consultation corrections and induced errors', 'three period strata', 'fixed seed 10000 day-cluster bootstrap joint-minus-each accuracy difference'],
        'abstention_policy': 'UNKNOWN is never a correct NO. Report judged denominator plus complete known denominator. Recall retains abstained positive labels as misses, coverage separate.',
        'top3_policy': 'Rank exactly min(3,pool size) relative candidates, even if direction NO/UNKNOWN; relative rank is not a positive forecast. Top3 slots with unknown outcomes retained.',
        'end_rule': 'One fixed sample, one independent pass each, one consultation. No outcome-driven date replacement, parameter adjustment, post-reveal decision change, retraining, production deployment or formal-ledger writes.',
        'limits': ['Previously researched historical universe and late-acquired sources: masked replay is NOT independently certified blind or genuine forward evidence.', 'Historical contemporaneous theme/forum/announcement coverage is insufficient; packet tests price-volume/emotion/competition reasoning only, not a full narrative strategy.', 'Archived pool and bars retain original survivor and supplier scope biases. No claim of all-A-share performance.', 'No fill, profit or calibrated individual event probability is inferred.'],
        'signal_fields': ['boards','turnover','amplitude','first_seal_minute','last_seal_minute','bomb_count','signal_amount','signal_float_cap'],
        'bar_fields': ['date','code','open','high','low','close','pre_close','volume','amount','turnover'],
        'market_fields': ['limit_count','consecutive_count','max_board','failed_count','down_limit_count','failed_rate','seal_rate','yesterday_premium','emotion_state'],
    }
    write(HERE/'protocol_BEFORE_DATA_AND_DECISIONS.json', protocol)
    source_hash = sha(BUNDLE)
    data = joblib.load(BUNDLE)
    full = data['prepared']['relay']
    # Project the signal columns before any row inspection or sampling. No outcome is read here.
    relay = full[['date','code','name','price'] + protocol['signal_fields']].copy()
    relay['date'] = relay['date'].astype(str).str[:10]
    relay['code'] = relay['code'].astype(str).str.zfill(6)
    selected = []
    for period, start, end in PERIODS:
        dates = sorted(relay.loc[relay.date.between(start,end),'date'].unique())
        for day in sorted(dates, key=lambda d: score(period+':'+d))[:4]: selected.append((period,day))
    selected.sort(key=lambda v: score('case:'+v[1]))
    market = data['prepared']['market'][['date']+protocol['market_fields']].copy()
    market['date'] = market['date'].astype(str).str[:10]
    bars = data['bars'][protocol['bar_fields']].copy()
    bars['date'] = bars['date'].astype(str).str[:10]
    bars['code'] = bars['code'].astype(str).str.zfill(6)
    codes = set(relay.loc[relay.date.isin([d for _,d in selected]),'code'])
    bars = bars.loc[bars.code.isin(codes)].sort_values(['code','date'])
    assert not bars.duplicated(['code','date']).any()
    features = data['features'][['date','code','daily_one_word','relative_volume','verified_turnover_boards','first_open_after_one_word']].copy()
    features['date'] = features['date'].astype(str).str[:10]
    features['code'] = features['code'].astype(str).str.zfill(6)
    features = features.set_index(['date','code'])
    groups = {c:g for c,g in bars.groupby('code')}
    packet = {'experiment':protocol['experiment'], 'event':'下一相邻官方交易日收盘涨停', 'outcome_fields_in_packet':False,
        'notes':'完整原归档连板池；日期/代码/名称/BASE及ROLL分数/次日行情全部遮蔽。null=未知，不是0。封板分钟为自09:30起的日内墙钟分钟，午休不扣除，0=09:30。换手/振幅单位%，amount/cap只给同日相对名次；历史量单位未统一，不解释为股或手。one_price为当日OHLC同价形态，不是可成交性。没有合格历史题材/论坛/公告全文，所以所有候选叙事及公告风险未知。', 'days':[]}
    private = []
    for i,(period,day) in enumerate(selected,1):
        cid = f'D{i:02}'
        rows = relay.loc[relay.date.eq(day)].copy()
        rows = rows.assign(_shuffle=rows.code.map(lambda c:score(day+':'+c))).sort_values('_shuffle')
        assert len(rows) >= 3 and not rows.code.duplicated().any()
        for field in ['signal_amount','signal_float_cap']:
            rows[field+'_pctile'] = pd.to_numeric(rows[field],errors='coerce').rank(pct=True)*100
        histm = market.loc[market.date.le(day)].tail(3)
        marketrows = [{k:r[k] for k in protocol['market_fields']} for _,r in histm.iterrows()]
        stockrows=[]
        for j,(_,r) in enumerate(rows.iterrows(),1):
            sid=f'S{j:02}'
            f=features.loc[(day,r.code)]
            hist=groups.get(r.code,pd.DataFrame(columns=bars.columns))
            hist=hist.loc[hist.date.le(day)].tail(5)
            assert hist.empty or hist.date.max() <= day
            hist_rows=[]
            for _,h in hist.iterrows():
                p=num(h.pre_close)
                hist_rows.append({'o':100*(ratio(h.open,p)-1) if ratio(h.open,p) is not None else None,
                    'h':100*(ratio(h.high,p)-1) if ratio(h.high,p) is not None else None,
                    'l':100*(ratio(h.low,p)-1) if ratio(h.low,p) is not None else None,
                    'c':100*(ratio(h.close,p)-1) if ratio(h.close,p) is not None else None,
                    'turnover':h.turnover})
            item={'id':sid,'boards':r.boards,'one_price':f.daily_one_word,'turnover_pct':r.turnover,
                'amplitude_pct':r.amplitude,'first_seal_min':r.first_seal_minute,'last_seal_min':r.last_seal_minute,
                'reported_open_count':r.bomb_count,'amount_pctile':r.signal_amount_pctile,'float_cap_pctile':r.signal_float_cap_pctile,
                'volume_vs_previous':f.relative_volume,'observed_nonword_streak':f.verified_turnover_boards,
                'first_open_after_one_price':f.first_open_after_one_word,'past5_ohlc_pct_vs_previous_close':hist_rows}
            stockrows.append(item)
            private.append({'case':cid,'id':sid,'period':period,'date':day,'code':r.code,'name':r['name'],'pool_price':r.price})
        packet['days'].append({'case':cid,'complete_pool_count':len(rows),'past3_market_old_to_new':marketrows,'stocks':stockrows})
    write(HERE/'PRIVATE_mapping_DO_NOT_READ_BEFORE_JOINT_FREEZE.json',private)
    write(HERE/'review_packet_SIGNAL_ONLY.json',packet)
    lines=['# 双模型固定历史回放：仅信号资料，无答案','',packet['notes'],'','表内 T-2/T-1/T 为市场过去三天。股票表完整保留每日归档连板候选。过去五日OHLC相对昨收及换手另见JSON；需要时必须阅读，不另取原数据库。','']
    cols=['id','boards','one_price','turnover_pct','amplitude_pct','first_seal_min','last_seal_min','reported_open_count','amount_pctile','float_cap_pctile','volume_vs_previous','observed_nonword_streak','first_open_after_one_price']
    def fmt(v):
        v=clean(v)
        return '?' if v is None else str(v)
    for d in packet['days']:
        lines+=['## '+d['case']+' 完整池 '+str(d['complete_pool_count'])+'只','',json.dumps(clean(d['past3_market_old_to_new']),ensure_ascii=False),'','|'+ '|'.join(cols)+'|','|'+'|'.join(['---']*len(cols))+'|']
        for r in d['stocks']:lines.append('|'+ '|'.join(fmt(r[k]) for k in cols)+'|')
        lines.append('')
    (HERE/'review_packet_SIGNAL_ONLY.md').write_text('\n'.join(lines),encoding='utf-8')
    schema={'reviewer':'YOUR_NAME','packet_sha256':sha(HERE/'review_packet_SIGNAL_ONLY.json'),'created_at_utc':'Use actual clock or state not independently verified','outcomes_seen':False,
        'days':[{'case':d['case'],'top3':['Sxx','Syy','Szz'],'market_reason':'independent context','stocks':[{'id':r['id'],'decision':'YES/NO/UNKNOWN','reason':'signal support; counterevidence/cancellation'} for r in d['stocks']]} for d in packet['days']]}
    write(HERE/'decision_SCHEMA_NOT_A_PREDICTION.json',schema)
    write(HERE/'preparation_receipt.json',{'created_at_utc':now(),'source':str(BUNDLE),'source_sha256':source_hash,
        'protocol_sha256':sha(HERE/'protocol_BEFORE_DATA_AND_DECISIONS.json'),'packet_sha256':sha(HERE/'review_packet_SIGNAL_ONLY.json'),
        'mapping_sha256':sha(HERE/'PRIVATE_mapping_DO_NOT_READ_BEFORE_JOINT_FREEZE.json'),'days':len(selected),'rows':len(private),'outcomes_extracted':False,'selected_only_by_dates':True,
        'protected_files':{str(p):sha(p) for p in [ROOT/'data/independent_stock_judgement/registry.jsonl',ROOT/'data/independent_stock_judgement/settlement_ledger.jsonl'] if p.exists()}})
    print(json.dumps({'days':len(selected),'rows':len(private),'case_sizes':{d['case']:d['complete_pool_count'] for d in packet['days']},'outcomes_extracted':False},ensure_ascii=False))

if __name__=='__main__':
    if sys.argv[1]=='prepare': prepare()
    else: raise SystemExit('Unknown operation')
