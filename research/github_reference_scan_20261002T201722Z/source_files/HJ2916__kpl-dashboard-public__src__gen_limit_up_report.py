# -*- coding: utf-8 -*-
"""
涨停个股深度分析报告 - 数据采集+HTML生成
获取过去10日每日涨停股票的全部数据
"""
import sys
sys.path.insert(0, r'D:\trae\traeworkspace\kpl-dashboard\lib')
sys.path.insert(0, r'D:\trae\traeworkspace\kpl-dashboard\scripts')

import sqlite3
import json
import time
from datetime import datetime, timedelta
from kaipanla_crawler import KaipanlaCrawler
from pathlib import Path

DB = r'D:\trae\traeworkspace\kpl-dashboard\data\kpl_dashboard.db'
OUTPUT = Path(r'D:\trae\traeworkspace\kpl-dashboard\output')

crawler = KaipanlaCrawler()

# 获取过去10个交易日
def get_trading_dates():
    """从数据库获取已有板块强度的最近10个交易日"""
    conn = sqlite3.connect(DB)
    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT date FROM sector_strength ORDER BY date DESC LIMIT 10"
    ).fetchall()]
    conn.close()
    return sorted(dates)

def collect_limit_up_data(dates):
    """收集每日涨停数据"""
    from datetime import datetime
    today_str = datetime.now().strftime('%Y-%m-%d')
    all_data = {}
    for date in dates:
        print(f"\n采集 {date} 涨停数据...")
        day_data = {'date': date, 'stocks': [], 'broken': [], 'ladder': {}, 'ranking': []}
        is_today = (date == today_str)

        # 1. 涨停股数据
        if is_today:
            # 今日: 用get_realtime_board_stocks获取全部涨停股(含封单/主力/连板等)
            print("  [今日] 使用get_realtime_board_stocks获取全部涨停股...")
            all_board_stocks = []
            for bt in range(1, 6):
                try:
                    stocks = crawler.get_realtime_board_stocks(board_type=bt)
                    all_board_stocks.extend(stocks)
                    time.sleep(0.2)
                except Exception as e:
                    print(f"  get_realtime_board_stocks({bt})失败: {e}")

            # 按涨停原因分组
            reason_groups = {}
            for s in all_board_stocks:
                reason = s.get('limit_up_reason', '') or '未分类'
                if reason not in reason_groups:
                    reason_groups[reason] = []
                stock = {
                    'code': s.get('stock_code', ''),
                    'name': s.get('stock_name', ''),
                    'limit_up_reason': reason,
                    'concepts': s.get('concepts', ''),
                    'consecutive': f"{s.get('consecutive_days', 1)}板",
                    'consecutive_count': s.get('consecutive_days', 1),
                    'seal_amount': s.get('seal_amount', 0),
                    'main_capital': s.get('main_net_inflow', 0),
                    'circulating_mv': s.get('circulating_market_cap', 0),
                    'total_mv': s.get('total_market_cap', 0),
                    'limit_up_price': s.get('limit_up_price', 0),
                    'change_pct': s.get('change_pct', 0),
                    'sector_code': s.get('sector_code', ''),
                }
                reason_groups[reason].append(stock)
                day_data['stocks'].append(stock)

            for reason, stocks in reason_groups.items():
                day_data['ranking'].append({
                    'sector_code': stocks[0].get('sector_code', ''),
                    'sector_name': reason,
                    'stock_count': len(stocks),
                    'stocks': stocks
                })
            print(f"  涨停板块: {len(day_data['ranking'])}个, 涨停股: {len(day_data['stocks'])}只")
        else:
            # 历史日期: 用get_sector_ranking(注意: API返回固定9板块, 非真实历史数据)
            try:
                ranking = crawler.get_sector_ranking(date=date)
                if ranking and ranking.get('sectors'):
                    summary = ranking.get('summary', {})
                    day_data['summary'] = summary
                    for sec in ranking['sectors']:
                        sector_info = {
                            'sector_code': sec.get('sector_code', ''),
                            'sector_name': sec.get('sector_name', ''),
                            'stock_count': len(sec.get('stocks', [])),
                            'stocks': []
                        }
                        for s in sec.get('stocks', []):
                            stock = {
                                'code': s.get('股票代码', ''),
                                'name': s.get('股票名称', ''),
                                'limit_up_reason': s.get('涨停原因', ''),
                                'concepts': s.get('概念标签', ''),
                                'consecutive': s.get('连板天数', ''),
                                'consecutive_count': s.get('连板次数', ''),
                                'theme': s.get('主题', ''),
                                'seal_amount': s.get('封单额', 0),
                                'main_capital': s.get('主力资金', 0),
                                'limit_up_time': s.get('首次封板时间', ''),
                                'circulating_mv': s.get('流通市值', 0),
                                'total_mv': s.get('总市值', 0),
                                'is_first_board': s.get('是否首板', False),
                            }
                            sector_info['stocks'].append(stock)
                            day_data['stocks'].append(stock)
                        day_data['ranking'].append(sector_info)
                    print(f"  涨停板块: {len(day_data['ranking'])}个, 涨停股: {len(day_data['stocks'])}只 (历史数据, API限制)")
            except Exception as e:
                print(f"  get_sector_ranking失败: {e}")

        time.sleep(0.5)

        # 2. 连板梯队(get_consecutive_limit_up, timeout=8s per call)
        try:
            ladder = crawler.get_consecutive_limit_up(date=date, timeout=8)
            if ladder and ladder.get('ladder'):
                day_data['ladder'] = {
                    'max_consecutive': ladder.get('max_consecutive', 0),
                    'max_stocks': ladder.get('max_consecutive_stocks', []),
                    'max_concepts': ladder.get('max_consecutive_concepts', []),
                    'ladder': {}
                }
                for height, stocks in ladder.get('ladder', {}).items():
                    day_data['ladder']['ladder'][height] = stocks
                print(f"  连板梯队: 最高{day_data['ladder']['max_consecutive']}板")
        except Exception as e:
            print(f"  get_consecutive_limit_up失败: {e}")

        time.sleep(0.5)

        # 3. 炸板股(get_historical_broken_limit_up)
        try:
            broken = crawler.get_historical_broken_limit_up(date=date)
            if broken:
                day_data['broken'] = [{
                    'code': s.get('stock_code', ''),
                    'name': s.get('stock_name', ''),
                    'change_pct': s.get('change_pct', 0),
                    'limit_up_time': s.get('limit_up_time', ''),
                    'open_time': s.get('open_time', ''),
                    'yesterday_consecutive': s.get('yesterday_consecutive_text', ''),
                    'sector': s.get('sector', ''),
                    'main_capital': s.get('main_capital_net', 0),
                    'turnover': s.get('turnover_amount', 0),
                } for s in broken]
                print(f"  炸板股: {len(day_data['broken'])}只")
        except Exception as e:
            print(f"  get_historical_broken_limit_up失败: {e}")

        time.sleep(0.5)

        # 4. 大幅回撤(get_sharp_withdrawal)
        try:
            withdrawal = crawler.get_sharp_withdrawal(date=date)
            if withdrawal is not None and len(withdrawal) > 0:
                day_data['withdrawal'] = []
                for _, row in withdrawal.iterrows():
                    day_data['withdrawal'].append({
                        'code': str(row.get('股票代码', '')),
                        'name': str(row.get('股票名称', '')),
                        'change_pct': float(row.get('当日涨跌幅(%)', 0) or 0),
                        'pullback': float(row.get('回撤幅度(%)', 0) or 0),
                    })
                print(f"  大幅回撤: {len(day_data['withdrawal'])}只")
        except Exception as e:
            print(f"  get_sharp_withdrawal失败: {e}")

        time.sleep(0.5)

        all_data[date] = day_data

    return all_data


def collect_sector_news(top_sectors, growth_sectors):
    """获取板块要闻"""
    all_news = {'top': [], 'growth': []}

    for label, sectors_list in [('top', top_sectors), ('growth', growth_sectors)]:
        for code, name, strength in sectors_list:
            print(f"  获取板块要闻: {name} ({code})")
            try:
                result = crawler.get_plate_news(plate_id=code, page_size=15)
                if result and result.get('news_list'):
                    all_news[label].append({
                        'sector_code': code,
                        'sector_name': name,
                        'strength': strength,
                        'news': result['news_list'][:10]
                    })
                    print(f"    {len(result['news_list'])}条要闻")
                else:
                    all_news[label].append({
                        'sector_code': code, 'sector_name': name,
                        'strength': strength, 'news': []
                    })
            except Exception as e:
                print(f"    失败: {e}")
            time.sleep(1)

    return all_news


def get_top_sectors(conn, today):
    """获取今日强度前3和增长前3的板块"""
    # 今日强度前3
    top3 = conn.execute(
        "SELECT sector_code, sector_name, strength FROM sector_strength WHERE date=? ORDER BY strength DESC LIMIT 3",
        (today,)
    ).fetchall()

    # 找昨天的日期
    yest_date_row = conn.execute(
        "SELECT date FROM sector_strength WHERE date < ? ORDER BY date DESC LIMIT 1",
        (today,)
    ).fetchone()
    growth3 = []
    if yest_date_row:
        yesterday = yest_date_row[0]
        growth_rows = conn.execute("""
            SELECT a.sector_code, a.sector_name, (a.strength - b.strength) as growth
            FROM sector_strength a
            JOIN sector_strength b ON a.sector_code = b.sector_code
            WHERE a.date = ? AND b.date = ?
            ORDER BY growth DESC LIMIT 3
        """, (today, yesterday)).fetchall()
        growth3 = [(r[0], r[1], r[2]) for r in growth_rows]

    return top3, growth3


def generate_frequent_stocks_html(frequent_stocks):
    """生成高频涨停个股表格HTML"""
    rows = []
    for i, (code, v) in enumerate(frequent_stocks):
        if len(v['dates']) < 2:
            continue
        reasons = ', '.join(v['reasons'])[:50]
        concepts = ' '.join(f'<span class="tag tag-blue">{c}</span>' for c in list(v['concepts'])[:5])
        sectors = ', '.join(v['sectors'])[:40]
        rows.append(f'<tr><td>{i+1}</td><td style="color:#0984e3;">{code}</td><td style="color:#fff;">{v["name"]}</td><td style="color:#e74c3c;font-weight:bold;">{len(v["dates"])}</td><td style="color:#fdcb6e;">{v["max_consecutive"]}板</td><td>{reasons}</td><td>{concepts}</td><td style="color:#888;font-size:12px;">{sectors}</td></tr>')
    return ''.join(rows)


def generate_html(all_data, news_data, dates):
    """生成涨停个股分析HTML"""
    today = datetime.now().strftime('%Y-%m-%d')
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 统计汇总
    total_stocks = sum(len(d['stocks']) for d in all_data.values())
    total_broken = sum(len(d.get('broken', [])) for d in all_data.values())
    total_withdrawal = sum(len(d.get('withdrawal', [])) for d in all_data.values())
    all_stock_codes = set()
    for d in all_data.values():
        for s in d['stocks']:
            all_stock_codes.add(s['code'])

    # 按板块统计
    sector_stats = {}
    for date, day_data in all_data.items():
        for sec in day_data.get('ranking', []):
            code = sec['sector_code']
            if code not in sector_stats:
                sector_stats[code] = {'name': sec['sector_name'], 'days': 0, 'total_stocks': 0}
            sector_stats[code]['days'] += 1
            sector_stats[code]['total_stocks'] += sec['stock_count']

    top_sectors = sorted(sector_stats.items(), key=lambda x: x[1]['total_stocks'], reverse=True)[:15]

    # 按涨停原因统计
    reason_stats = {}
    for date, day_data in all_data.items():
        for s in day_data.get('stocks', []):
            reason = s.get('limit_up_reason', '未知')
            if reason not in reason_stats:
                reason_stats[reason] = {'count': 0, 'stocks': set()}
            reason_stats[reason]['count'] += 1
            reason_stats[reason]['stocks'].add(s['code'])

    top_reasons = sorted(reason_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:20]

    # 个股统计(出现多次的)
    stock_stats = {}
    for date, day_data in all_data.items():
        for s in day_data.get('stocks', []):
            code = s['code']
            if code not in stock_stats:
                stock_stats[code] = {
                    'name': s['name'], 'dates': [], 'reasons': set(),
                    'concepts': set(), 'max_consecutive': 0, 'sectors': set()
                }
            stock_stats[code]['dates'].append(date)
            if s.get('limit_up_reason'):
                stock_stats[code]['reasons'].add(s['limit_up_reason'])
            if s.get('concepts'):
                stock_stats[code]['concepts'].update(s['concepts'].split('、'))
            consec = s.get('consecutive', '')
            if isinstance(consec, str) and '板' in consec:
                try:
                    n = int(''.join(filter(str.isdigit, consec)))
                    stock_stats[code]['max_consecutive'] = max(stock_stats[code]['max_consecutive'], n)
                except:
                    pass
            for sec in day_data.get('ranking', []):
                for st in sec.get('stocks', []):
                    if st['code'] == code:
                        stock_stats[code]['sectors'].add(sec['sector_name'])

    # 按出现次数排序
    frequent_stocks = sorted(stock_stats.items(), key=lambda x: len(x[1]['dates']), reverse=True)[:30]

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>涨停个股深度分析 {today}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #0d1117; color: #e0e0e0; font-family: 'Microsoft YaHei', sans-serif; padding: 15px; max-width: 1400px; margin: 0 auto; }}
h1 {{ color: #fff; font-size: 22px; margin-bottom: 5px; }}
h2 {{ color: #00b894; font-size: 18px; margin: 20px 0 10px; padding-bottom: 5px; border-bottom: 2px solid #1e2433; }}
h3 {{ color: #fdcb6e; font-size: 15px; margin: 15px 0 8px; }}
.meta {{ color: #888; font-size: 12px; margin-bottom: 15px; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin-bottom: 20px; }}
.summary-card {{ background: #161b22; padding: 12px; border-radius: 6px; border: 1px solid #1e2433; }}
.summary-card .num {{ font-size: 24px; font-weight: bold; color: #00b894; }}
.summary-card .label {{ font-size: 12px; color: #888; margin-top: 3px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 15px; }}
th {{ background: #161b22; padding: 8px; text-align: left; color: #aaa; border-bottom: 2px solid #333; position: sticky; top: 0; }}
td {{ padding: 6px 8px; border-bottom: 1px solid #1e2433; }}
tr:hover {{ background: #111827; }}
.tag {{ display: inline-block; background: #1e2433; padding: 2px 6px; border-radius: 3px; font-size: 11px; margin: 1px; }}
.tag-red {{ color: #e74c3c; }}
.tag-yellow {{ color: #fdcb6e; }}
.tag-green {{ color: #2ecc71; }}
.tag-blue {{ color: #0984e3; }}
.news-item {{ background: #161b22; padding: 10px; border-radius: 4px; margin-bottom: 8px; border-left: 3px solid #00b894; }}
.news-title {{ color: #fff; font-size: 14px; margin-bottom: 3px; }}
.news-meta {{ color: #666; font-size: 11px; }}
.news-content {{ color: #aaa; font-size: 12px; margin-top: 5px; }}
.sector-news {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 15px; }}
.day-section {{ background: #0f1419; border-radius: 8px; padding: 15px; margin-bottom: 15px; border: 1px solid #1e2433; }}
.day-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
.day-date {{ color: #fff; font-size: 16px; font-weight: bold; }}
.day-stats {{ color: #888; font-size: 12px; }}
.ladder-badge {{ display: inline-block; background: #2d3436; padding: 3px 8px; border-radius: 3px; margin: 2px; font-size: 12px; }}
.ladder-badge .height {{ color: #e74c3c; font-weight: bold; }}
</style>
</head>
<body>
<h1>涨停个股深度分析</h1>
<div class="meta">数据范围: {dates[0]} ~ {dates[-1]} | 更新时间: {now_str} | 数据源: KPL</div>

<div class="summary">
    <div class="summary-card"><div class="num">{len(dates)}</div><div class="label">交易日</div></div>
    <div class="summary-card"><div class="num">{total_stocks}</div><div class="label">涨停总次数</div></div>
    <div class="summary-card"><div class="num">{len(all_stock_codes)}</div><div class="label">涨停个股数</div></div>
    <div class="summary-card"><div class="num">{total_broken}</div><div class="label">炸板总次数</div></div>
    <div class="summary-card"><div class="num">{total_withdrawal}</div><div class="label">大幅回撤次数</div></div>
    <div class="summary-card"><div class="num">{len(sector_stats)}</div><div class="label">涉及板块数</div></div>
    <div class="summary-card"><div class="num">{len(reason_stats)}</div><div class="label">涨停原因数</div></div>
</div>

<h2>一、板块要闻</h2>
<div class="sector-news">
    <div>
        <h3>今日强度前3板块要闻</h3>
        {generate_news_html(news_data.get('top', []))}
    </div>
    <div>
        <h3>今日增长前3板块要闻</h3>
        {generate_news_html(news_data.get('growth', []))}
    </div>
</div>

<h2>二、涨停原因(细分板块)排行</h2>
<table>
<tr><th>排名</th><th>涨停原因</th><th>涨停次数</th><th>涉及个股</th></tr>
{''.join(f'<tr><td>{i+1}</td><td style="color:#fdcb6e;">{name}</td><td style="color:#e74c3c;font-weight:bold;">{v["count"]}</td><td>{len(v["stocks"])}只</td></tr>' for i, (name, v) in enumerate(top_reasons))}
</table>

<h2>三、板块涨停统计</h2>
<table>
<tr><th>排名</th><th>板块</th><th>出现天数</th><th>涨停总次数</th></tr>
{''.join(f'<tr><td>{i+1}</td><td style="color:#00b894;">{v["name"]}</td><td>{v["days"]}</td><td style="color:#e74c3c;font-weight:bold;">{v["total_stocks"]}</td></tr>' for i, (code, v) in enumerate(top_sectors))}
</table>

<h2>四、高频涨停个股(出现2次以上)</h2>
<table>
<tr><th>排名</th><th>代码</th><th>名称</th><th>涨停天数</th><th>最大连板</th><th>涨停原因</th><th>概念标签</th><th>所属板块</th></tr>
{generate_frequent_stocks_html(frequent_stocks)}
</table>

<h2>五、每日涨停详情</h2>
{''.join(generate_day_html(date, day_data) for date, day_data in sorted(all_data.items(), reverse=True))}

</body>
</html>"""
    return html


def generate_news_html(news_list):
    if not news_list:
        return '<div style="color:#666;padding:10px;">暂无要闻数据</div>'
    html = ''
    for item in news_list:
        html += f'<div style="margin-bottom:15px;"><div style="color:#00b894;font-weight:bold;margin-bottom:5px;">{item["sector_name"]} (强度:{item.get("strength",0)})</div>'
        if item.get('news'):
            for n in item['news'][:5]:
                type_color = '#e74c3c' if n.get('type') == 1 else '#0984e3'
                type_name = n.get('type_name', '')
                html += f'<div class="news-item"><div class="news-title">{n.get("title","")}</div><div class="news-meta">{n.get("datetime","")} | <span style="color:{type_color};">{type_name}</span></div><div class="news-content">{n.get("content","")[:150]}...</div></div>'
        else:
            html += '<div style="color:#555;font-size:12px;padding:5px;">无要闻</div>'
        html += '</div>'
    return html


def fmt_amt(v):
    """格式化金额: 亿/万"""
    try:
        v = float(v or 0)
        if v == 0:
            return '-'
        if abs(v) >= 1e8:
            return f'{v/1e8:.2f}亿'
        if abs(v) >= 1e4:
            return f'{v/1e4:.1f}万'
        return f'{v:.0f}'
    except:
        return '-'


def generate_day_html(date, day_data):
    stocks = day_data.get('stocks', [])
    broken = day_data.get('broken', [])
    withdrawal = day_data.get('withdrawal', [])
    ladder = day_data.get('ladder', {})
    summary = day_data.get('summary', {})

    # 连板梯队
    ladder_html = ''
    if ladder and ladder.get('ladder'):
        for height in sorted(ladder['ladder'].keys(), key=lambda x: int(x) if str(x).isdigit() else 0, reverse=True):
            stocks_list = ladder['ladder'][height]
            if isinstance(stocks_list, list):
                stock_names = ', '.join([f'{s.get("股票代码","")}{s.get("股票名称","")}' if isinstance(s, dict) else str(s) for s in stocks_list[:5]])
                ladder_html += f'<span class="ladder-badge"><span class="height">{height}板</span> {stock_names}</span>'

    # 涨停股表格(含封单额/主力资金/流通市值等)
    stocks_html = ''
    if stocks:
        stocks.sort(key=lambda x: x.get('consecutive_count', 0), reverse=True)
        for s in stocks:
            consec = s.get('consecutive', '')
            consec_color = '#e74c3c' if ('3' in str(consec) or '4' in str(consec) or '5' in str(consec)) else ('#fdcb6e' if '2' in str(consec) else '#aaa')
            concepts_html = ' '.join(f'<span class="tag tag-blue">{c}</span>' for c in s.get('concepts', '').split('、')[:4] if c)
            seal = fmt_amt(s.get('seal_amount', 0))
            main_cap = fmt_amt(s.get('main_capital', 0))
            circ_mv = fmt_amt(s.get('circulating_mv', 0))
            main_color = '#e74c3c' if (s.get('main_capital', 0) or 0) > 0 else '#2ecc71'
            stocks_html += f'<tr><td style="color:#0984e3;">{s["code"]}</td><td style="color:#fff;">{s["name"]}</td><td style="color:{consec_color};font-weight:bold;">{consec}</td><td style="color:#fdcb6e;font-size:12px;">{s.get("limit_up_reason","")}</td><td>{concepts_html}</td><td style="color:#666;font-size:12px;">{seal}</td><td style="color:{main_color};font-size:12px;">{main_cap}</td><td style="color:#888;font-size:12px;">{circ_mv}</td><td style="color:#888;font-size:12px;">{s.get("limit_up_time","")}</td></tr>'

    # 炸板股
    broken_html = ''
    if broken:
        for b in broken[:15]:
            broken_html += f'<tr><td style="color:#e74c3c;">{b["code"]}</td><td>{b["name"]}</td><td style="color:#e74c3c;">{b.get("change_pct",0):.2f}%</td><td style="color:#888;font-size:12px;">{b.get("limit_up_time","")}</td><td style="color:#888;font-size:12px;">{b.get("open_time","")}</td><td style="color:#888;font-size:12px;">{b.get("yesterday_consecutive","")}</td><td style="color:#888;font-size:12px;">{b.get("sector","")}</td><td style="color:#666;font-size:12px;">{fmt_amt(b.get("main_capital",0))}</td><td style="color:#666;font-size:12px;">{fmt_amt(b.get("turnover",0))}</td></tr>'

    # 大幅回撤股
    withdrawal_html = ''
    if withdrawal:
        for w in withdrawal[:10]:
            withdrawal_html += f'<tr><td style="color:#fdcb6e;">{w["code"]}</td><td>{w["name"]}</td><td style="color:#e74c3c;">{w.get("change_pct",0):.2f}%</td><td style="color:#e74c3c;font-weight:bold;">{w.get("pullback",0):.2f}%</td></tr>'

    stats_parts = [f'涨停{len(stocks)}只', f'炸板{len(broken)}只']
    if withdrawal:
        stats_parts.append(f'回撤{len(withdrawal)}只')
    if summary:
        stats_parts.append(f'{summary.get("涨停数",0)}涨停/{summary.get("跌停数",0)}跌停')

    return f"""
<div class="day-section">
    <div class="day-header">
        <div class="day-date">{date}</div>
        <div class="day-stats">{' | '.join(stats_parts)}</div>
    </div>
    {f'<div style="margin-bottom:10px;"><span style="color:#888;font-size:12px;">连板梯队:</span> {ladder_html}</div>' if ladder_html else ''}
    <table>
    <tr><th>代码</th><th>名称</th><th>连板</th><th>涨停原因</th><th>概念标签</th><th>封单额</th><th>主力净额</th><th>流通市值</th><th>封板时间</th></tr>
    {stocks_html}
    </table>
    {f'<h3 style="color:#e74c3c;">炸板股 ({len(broken)}只)</h3><table><tr><th>代码</th><th>名称</th><th>涨跌幅</th><th>封板时间</th><th>开板时间</th><th>昨日连板</th><th>板块</th><th>主力净额</th><th>成交额</th></tr>{broken_html}</table>' if broken_html else ''}
    {f'<h3 style="color:#fdcb6e;">大幅回撤 ({len(withdrawal)}只)</h3><table><tr><th>代码</th><th>名称</th><th>涨跌幅</th><th>回撤幅度</th></tr>{withdrawal_html}</table>' if withdrawal_html else ''}
</div>"""


def main():
    dates = get_trading_dates()
    print(f"过去10个交易日: {dates}")

    # 1. 采集涨停数据
    all_data = collect_limit_up_data(dates)

    # 2. 获取板块要闻
    conn = sqlite3.connect(DB)
    today = dates[-1]
    top3, growth3 = get_top_sectors(conn, today)
    print(f"\n今日强度前3: {[(n, s) for _, n, s in top3]}")
    print(f"今日增长前3: {[(n, g) for _, n, g in growth3]}")
    conn.close()

    news_data = collect_sector_news(top3, growth3)

    # 3. 生成HTML
    html = generate_html(all_data, news_data, dates)
    _time_tag = datetime.now().strftime('%Y%m%d-%H%M')
    html_path = OUTPUT / f'涨停个股深度分析_{_time_tag}.html'
    html_path.write_text(html, encoding='utf-8')
    print(f"\nHTML报告已生成: {html_path}")
    print(f"文件大小: {html_path.stat().st_size / 1024:.0f}KB")

if __name__ == '__main__':
    main()
