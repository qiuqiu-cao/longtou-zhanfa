"""One-off read-only-source audit; writes a new dated review evidence directory."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/review_evidence_20260907'
RAW = ROOT / 'data/v3/raw/20260907'

def mainboard(f):
    return f[f['代码'].str.match(r'^(00|60)') & ~f['名称'].str.contains('ST|退', regex=True)].copy()

def read(name, date='20260907'):
    return pd.read_csv(ROOT / f'data/v3/raw/{date}/{name}.csv', dtype={'代码': str})

def main():
    OUT.mkdir(exist_ok=False)
    up = mainboard(read('limit_up'))
    failed = mainboard(read('failed_board'))
    prev = mainboard(read('previous_limit'))
    prior = mainboard(read('limit_up', '20260904'))
    assert not up['代码'].duplicated().any()
    assert not set(up['代码']) & set(failed['代码'])
    joined = prior[['代码','连板数']].merge(prev, on='代码', how='left', validate='one_to_one')
    joined['promoted'] = joined['代码'].isin(up['代码'])
    actual = up.set_index('代码')['连板数']
    mismatches = []
    for r in joined[joined.promoted].to_dict('records'):
        if actual[r['代码']] != r['连板数'] + 1:
            mismatches.append(r['代码'])
    data = up.rename(columns={'代码':'code','名称':'name','连板数':'boards'})
    data['signal_date'] = '2026-09-07'
    data['pool_assignment'] = 'CLOSING_LIMIT_UP'
    data['is_mainboard'] = True
    data['price_data_status'] = '已核验_单一供应商收盘档案'
    data['security_status'] = '延迟_仅代码名称初筛_官方有效期需更新'
    data['specific_theme'] = '不可获得_未逐股核验当日驱动'
    data['announcement_fulltext_status'] = '不可获得_待逐股全文核验'
    data['tick_l2_status'] = '不可获得'
    data['continuation_probability_2_3d'] = None
    data.to_csv(OUT/'all_closing_candidates.csv', index=False, encoding='utf-8-sig')
    failed.to_csv(OUT/'failed_boards.csv', index=False, encoding='utf-8-sig')
    joined.to_csv(OUT/'previous_limit_feedback.csv', index=False, encoding='utf-8-sig')
    losses = (failed['最新价']/failed['涨停价']-1)*100
    stats = {
        'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'signal_date':'2026-09-07','plan_date':'2026-09-08',
        'full_market_completeness':'不可获得_接口93涨停19炸板与媒体95及25未对齐',
        'mainboard_scope':'代码名称初筛_未完成当日官方证券状态核验',
        'closing_count':len(up),'boards':{str(k):int(v) for k,v in up['连板数'].value_counts().items()},
        'failed_count':len(failed),'limit_down_count':len(mainboard(read('limit_down'))),
        'seal_rate_within_received_pool':len(up)/(len(up)+len(failed)),
        'previous_limit_count':len(joined),'previous_feedback_available':int(joined['涨跌幅'].notna().sum()),
        'previous_mean_pct':float(joined['涨跌幅'].mean()),'previous_median_pct':float(joined['涨跌幅'].median()),
        'previous_q10_pct':float(joined['涨跌幅'].quantile(.1)),
        'previous_negative_count':int(joined['涨跌幅'].lt(0).sum()),
        'previous_minus7_count':int(joined['涨跌幅'].le(-7).sum()),
        'promotions':int(joined.promoted.sum()),'board_mismatches':mismatches,
        'feedback_by_previous_board':{str(k):{'n':len(g),'promotions':int(g.promoted.sum()),'mean_pct':float(g['涨跌幅'].mean())} for k,g in joined.groupby('连板数')},
        'failed_limit_price_to_close_proxy_mean_pct':float(losses.mean()),
        'failed_limit_price_to_close_proxy_worst_pct':float(losses.min()),
        'failed_limit_price_to_close_proxy_minus7_count':int(losses.le(-7).sum()),
        'failed_loss_status':'推断_涨停价假设成交至当日收盘_非T+1完整交易大面率',
        'raw_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RAW.glob('*.csv')},
        'decision':'CASH','model_status':'DIAGNOSTIC_ONLY',
    }
    (OUT/'audit.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 2026-09-07 全量已获取涨停池明细（完整性未通过）','',
           '来源：东方财富经AKShare，采集开始时间2026-09-07 17:28:28，收盘口径。主板代码/名称初筛86只；未证明全市场无遗漏。',
           '报价、首次/末次封板、开板次数、成交额、换手为单源档案已核验；首次封板时间保留供应商秒级标记，不代表真实可成交时间。',
           '所有行的真实当日细分题材及公告全文风险均未完成核验，记“不可获得”；行业不是当日涨停原因。证券资格官方证据只到9月4日，记“延迟”。','']
    for title, frame in [('首板',data[data.boards.eq(1)]),('连续板',data[data.boards.gt(1)])]:
        lines += [f'## {title}：{len(frame)}只','', '|代码|名称|板位|收盘元|首次封板|最后封板|开板次数|成交亿元|换手%|供应商行业（非真实题材）|','|---|---|---:|---:|---|---|---:|---:|---:|---|']
        for r in frame.to_dict('records'):
            tm=lambda v: str(int(v)).zfill(6)[:2]+':'+str(int(v)).zfill(6)[2:4]+':'+str(int(v)).zfill(6)[4:]
            lines.append(f"|{r['code']}|{r['name']}|{r['boards']}|{r['最新价']:.2f}|{tm(r['首次封板时间'])}|{tm(r['最后封板时间'])}|{r['炸板次数']}|{r['成交额']/1e8:.2f}|{r['换手率']:.2f}|{r['所属行业']}|")
        lines += ['']
    lines += ['## 炸板池（单独，不混入收盘涨停）','', '、'.join(f"{r['名称']}({r['代码']})" for r in failed.to_dict('records')), '', '## 数据审计','', '```json', json.dumps(stats,ensure_ascii=False,indent=2),'```']
    (OUT/'full_pool.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
