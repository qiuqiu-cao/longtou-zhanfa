# -*- coding: utf-8 -*-
"""
KPL数据看板 - 数据采集 v7 (3个月数据版)
获取2026-04-01到2026-07-06约3个月的交易日数据
"""
import sys
from pathlib import Path
# 自动定位项目lib目录
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'lib'))
sys.path.insert(0, r'D:\trae\traeworkspace\kpl')  # 兼容旧路径
import sqlite3
import time
import requests
import pandas as pd
import uuid
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')
from kaipanla_crawler import KaipanlaCrawler

DB_PATH = Path(r'd:\trae\traeworkspace\kpl-dashboard\data\kpl_dashboard.db')
KPL_FILE = Path(r'<YOUR_HOME>\.trae-cn\attachments\6a4aff607bc6d9ac89e12170\bc2b5462-4380-4706-aa8b-27f4e6514b06_开盘啦4月22日板块分类20260706.xlsx')
HEADERS = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://quote.eastmoney.com/'}
START_DATE = '2026-04-01'
END_DATE = '2026-07-06'
MAX_WORKERS = 8

KPL_API_URL = "https://apphis.longhuvip.com/w1/api/index.php"
KPL_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "User-Agent": "Dalvik/2.1.0 (Linux; U; Android 9; 23116PN5BC Build/PQ3A.190605.01141736)",
    "Host": "apphis.longhuvip.com",
    "Connection": "Keep-Alive",
    "Accept-Encoding": "gzip",
}


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS sector_list (
        sector_code TEXT, sector_name TEXT, kpl_strength REAL,
        source TEXT DEFAULT 'excel',
        PRIMARY KEY (sector_code, sector_name)
    );
    CREATE TABLE IF NOT EXISTS sector_strength (
        date TEXT, sector_code TEXT, sector_name TEXT, strength REAL,
        PRIMARY KEY (date, sector_code)
    );
    CREATE TABLE IF NOT EXISTS new_high (
        date TEXT PRIMARY KEY,
        total_high INTEGER,
        daily_new INTEGER
    );
    CREATE TABLE IF NOT EXISTS sector_new_high (
        date TEXT, sector_code TEXT, sector_name TEXT,
        total_high INTEGER, daily_new INTEGER,
        PRIMARY KEY (date, sector_code)
    );
    CREATE TABLE IF NOT EXISTS hs300_kline (
        date TEXT PRIMARY KEY, open REAL, close REAL, high REAL, low REAL,
        volume REAL, pct_chg REAL
    );
    """)
    conn.commit()
    return conn


def get_trading_dates_in_range(nh_records):
    """从新高数据中获取指定日期范围内的交易日"""
    dates = sorted([r[0] for r in nh_records])
    filtered = [d for d in dates if START_DATE <= d <= END_DATE]
    return filtered


def fetch_hs300_kline(start_date, end_date):
    url = "http://48.push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": "1.000300",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57",
        "klt": "101", "fqt": "0",
        "beg": start_date.replace("-", ""),
        "end": end_date.replace("-", ""),
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=15)
    data = r.json()
    klines = data.get("data", {}).get("klines", [])
    result = []
    for k in klines:
        row = k.split(',')
        result.append((row[0], float(row[1]), float(row[2]), float(row[3]),
                      float(row[4]), float(row[5]), float(row[6]) if len(row) > 6 else 0))
    return result


def fetch_market_new_high():
    data = {
        "a": "GetDayNewHigh_W28", "st": "360", "c": "StockNewHigh",
        "PhoneOSNew": "1", "DeviceID": str(uuid.uuid4()),
        "VerSion": "5.21.0.2", "Index": "0",
        "GroupID": "ALL", "apiv": "w42", "Type": "0_0_0_0_0"
    }
    r = requests.post(KPL_API_URL, data=data, headers=KPL_HEADERS, verify=False,
                      proxies={'http': None, 'https': None}, timeout=60)
    result = r.json()
    x_data = result.get("x", [])
    records = []
    for item in x_data:
        parts = item.split("_")
        if len(parts) >= 3:
            date_str = parts[0]
            total_high = int(parts[1])
            daily_new = int(parts[2])
            formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            records.append((formatted, total_high, daily_new))
    return records


def fetch_sector_new_high(args):
    sector_code, sector_name = args
    try:
        data = {
            "a": "GetDayNewHigh_W28", "st": "360", "c": "StockNewHigh",
            "PhoneOSNew": "1", "DeviceID": str(uuid.uuid4()),
            "VerSion": "5.21.0.2", "Index": "0",
            "GroupID": sector_code, "apiv": "w42", "Type": "0_0_0_0_0"
        }
        r = requests.post(KPL_API_URL, data=data, headers=KPL_HEADERS, verify=False,
                          proxies={'http': None, 'https': None}, timeout=60)
        result = r.json()
        x_data = result.get("x", [])
        records = []
        for item in x_data:
            parts = item.split("_")
            if len(parts) >= 3:
                date_str = parts[0]
                total_high = int(parts[1])
                daily_new = int(parts[2])
                formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                records.append((formatted, sector_code, sector_name, total_high, daily_new))
        return records
    except:
        return []


def fetch_one_sector_strength(args):
    sector_code, sector_name, date = args
    try:
        crawler = KaipanlaCrawler()
        result = crawler.get_sector_strength(sector_code, date)
        if result.get('success'):
            return (date, sector_code, sector_name, float(result['strength']))
        else:
            return (date, sector_code, sector_name, 0.0)
    except:
        return (date, sector_code, sector_name, 0.0)


def discover_extra_sectors(crawler, trading_dates, existing_codes):
    extra_sectors = {}
    skipped = set()
    for date in trading_dates:
        try:
            ranking = crawler.get_sector_ranking(date=date)
            if isinstance(ranking, dict) and 'sectors' in ranking:
                for sec in ranking['sectors']:
                    code = str(sec.get('sector_code', ''))
                    name = sec.get('sector_name', '')
                    if not code or code == '0' or code == 'None':
                        if code not in skipped:
                            skipped.add(code)
                        continue
                    if code not in existing_codes and code not in extra_sectors and code not in skipped:
                        extra_sectors[code] = name
                        print(f"    发现额外板块: {name}({code}) [来源: {date}]")
        except Exception as e:
            pass
        time.sleep(0.15)
    return extra_sectors


def main():
    print(f"{'='*80}")
    print(f"KPL数据看板 - 数据采集 v7 (3个月数据) | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"日期范围: {START_DATE} ~ {END_DATE}")
    print(f"{'='*80}")

    conn = init_db()
    crawler = KaipanlaCrawler()

    # 检查现有数据
    existing_strength_dates = set()
    try:
        rows = conn.execute("SELECT DISTINCT date FROM sector_strength").fetchall()
        existing_strength_dates = set(r[0] for r in rows)
    except:
        pass
    total_existing = conn.execute("SELECT COUNT(*) FROM sector_strength").fetchone()[0]
    print(f"\n[状态] 已有板块强度: {len(existing_strength_dates)}天 ({total_existing}条)")

    # 0. 加载Excel板块列表
    print(f"\n[0] 加载精选板块列表")
    df_kpl = pd.read_excel(KPL_FILE, sheet_name='Sheet1')
    excel_codes = set()
    excel_sectors = {}
    for _, row in df_kpl.iterrows():
        code = str(row['板块代码'])
        name = row['板块名称']
        conn.execute("INSERT OR REPLACE INTO sector_list VALUES(?,?,?,?)",
                     (code, name, float(row['强度']), 'excel'))
        excel_codes.add(code)
        if code not in excel_sectors:
            excel_sectors[code] = name
    conn.commit()
    print(f"    Excel: {len(df_kpl)} 行, {len(excel_codes)} 个唯一代码")

    # 1. 沪深300 K线
    print(f"\n[1] 沪深300日K线 ({START_DATE} ~ {END_DATE})")
    hs300 = fetch_hs300_kline(START_DATE, END_DATE)
    conn.executemany("INSERT OR REPLACE INTO hs300_kline VALUES(?,?,?,?,?,?,?)", hs300)
    conn.commit()
    print(f"    已保存 {len(hs300)} 条K线")

    # 2. 全市场百日新高
    print(f"\n[2] 全市场百日新高 (1次API调用)")
    nh_records = fetch_market_new_high()
    for date, total, new in nh_records:
        conn.execute("INSERT OR REPLACE INTO new_high VALUES(?,?,?)", (date, total, new))
    conn.commit()
    trading_dates = get_trading_dates_in_range(nh_records)
    print(f"    已保存 {len(nh_records)} 天数据")
    print(f"    目标交易日({len(trading_dates)}天): {trading_dates[0]} ~ {trading_dates[-1]}")

    # 3. 动态发现额外板块 (检查全部交易日)
    print(f"\n[3] 动态发现额外板块 (检查全部{len(trading_dates)}个交易日)")
    extra_sectors = discover_extra_sectors(crawler, trading_dates, excel_codes)
    if extra_sectors:
        for code, name in extra_sectors.items():
            conn.execute("INSERT OR REPLACE INTO sector_list VALUES(?,?,?,?)",
                         (code, name, 0, 'dynamic'))
        conn.commit()
        print(f"    发现 {len(extra_sectors)} 个额外板块")
    else:
        print(f"    未发现额外板块")

    all_sectors = dict(excel_sectors)
    all_sectors.update(extra_sectors)
    print(f"    总板块数: {len(all_sectors)}")

    # 4. 板块百日新高
    existing_snh_codes = set()
    try:
        existing_snh_codes = set(r[0] for r in conn.execute("SELECT DISTINCT sector_code FROM sector_new_high").fetchall())
    except:
        pass
    missing_snh = [(code, name) for code, name in all_sectors.items() if code not in existing_snh_codes]

    if not missing_snh:
        print(f"\n[4] 板块百日新高: 全部已有, 跳过")
    else:
        print(f"\n[4] 板块百日新高 ({len(missing_snh)}板块)")
        t_start = time.time()
        success_count = 0
        total_records = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(fetch_sector_new_high, task): task for task in missing_snh}
            for i, future in enumerate(as_completed(futures)):
                records = future.result()
                if records:
                    conn.executemany("INSERT OR REPLACE INTO sector_new_high VALUES(?,?,?,?,?)", records)
                    success_count += 1
                    total_records += len(records)
                if (i + 1) % 50 == 0:
                    conn.commit()
                    print(f"    进度: {i+1}/{len(missing_snh)}")
        conn.commit()
        print(f"    完成! {success_count}板块, {total_records}条, 耗时{time.time()-t_start:.0f}秒")

    # 5. 板块强度 (增量)
    missing_dates = [d for d in trading_dates if d not in existing_strength_dates]

    if not missing_dates:
        print(f"\n[5] 板块强度: 所有日期已有, 跳过")
    else:
        total_calls = len(missing_dates) * len(all_sectors)
        print(f"\n[5] 板块强度: {len(missing_dates)}天 x {len(all_sectors)}板块 = {total_calls}次调用")
        print(f"    {MAX_WORKERS}线程并发...")

        tasks = []
        for date in missing_dates:
            for code, name in all_sectors.items():
                tasks.append((code, name, date))

        success_count = 0
        t_start = time.time()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(fetch_one_sector_strength, task): task for task in tasks}
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result:
                    conn.execute("INSERT OR REPLACE INTO sector_strength VALUES(?,?,?,?)", result)
                    success_count += 1
                if (i + 1) % 500 == 0:
                    conn.commit()
                    elapsed = time.time() - t_start
                    speed = (i + 1) / elapsed
                    remaining = (len(tasks) - i - 1) / speed
                    print(f"    进度: {i+1}/{len(tasks)} ({speed:.0f}次/秒, 剩余{remaining:.0f}s)")

        conn.commit()
        elapsed = time.time() - t_start
        print(f"    完成! {success_count}条, 耗时{elapsed:.0f}秒")

    # 6. 统计
    print(f"\n{'='*60}")
    print(f"采集完成统计")
    print(f"{'='*60}")
    total_s = conn.execute("SELECT COUNT(*) FROM sector_strength").fetchone()[0]
    dates_s = conn.execute("SELECT COUNT(DISTINCT date) FROM sector_strength").fetchone()[0]
    sectors_s = conn.execute("SELECT COUNT(DISTINCT sector_code) FROM sector_strength").fetchone()[0]
    sl_s = conn.execute("SELECT COUNT(*) FROM sector_list").fetchone()[0]
    sl_dynamic = conn.execute("SELECT COUNT(*) FROM sector_list WHERE source='dynamic'").fetchone()[0]
    nh_s = conn.execute("SELECT COUNT(*) FROM new_high").fetchone()[0]
    snh_s = conn.execute("SELECT COUNT(*) FROM sector_new_high").fetchone()[0]
    kl_s = conn.execute("SELECT COUNT(*) FROM hs300_kline").fetchone()[0]
    print(f"    板块列表: {sl_s}个 (Excel:{sl_s-sl_dynamic} + 动态:{sl_dynamic})")
    print(f"    板块强度: {total_s}条 ({dates_s}天 x {sectors_s}板块)")
    print(f"    百日新高: {nh_s}天")
    print(f"    板块新高: {snh_s}条")
    print(f"    沪深300: {kl_s}条")

    # 验证
    row = conn.execute("SELECT strength FROM sector_strength WHERE sector_code='801572' AND date='2026-07-06'").fetchone()
    if row:
        print(f"\n    验证 中报增长(801572) 7月6日: {row[0]}")

    conn.close()
    print(f"\n数据库: {DB_PATH}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
