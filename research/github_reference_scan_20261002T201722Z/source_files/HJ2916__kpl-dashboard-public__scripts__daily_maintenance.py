# -*- coding: utf-8 -*-
"""
KPL数据看板 - 每日定时维护 (蚂蚁搬家式)
执行顺序:
1. 获取当日新数据(板块强度+新高+沪深300)
2. 蚂蚁搬家补全2天历史数据
3. 刷新看板
运行时间: 每个交易日17:30 (收盘后)
"""
import sys
from pathlib import Path
import time
import random

_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

# 直接调用各子模块
from kpl_data_collector import main as collect_today
from backfill_history import main as backfill
from gen_dashboard import main as gen_html


def main():
    print(f"{'='*70}")
    print(f"KPL每日定时维护 | {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")

    # Step 1: 获取当日新数据(增量)
    print("\n>>> Step 1: 获取当日新数据...")
    try:
        collect_today()
    except Exception as e:
        print(f"  当日数据获取失败: {e}")

    # 随机休息
    pause = random.uniform(10, 30)
    print(f"\n  休息 {pause:.0f}秒...")
    time.sleep(pause)

    # Step 2: 蚂蚁搬家补全历史
    print("\n>>> Step 2: 蚂蚁搬家补全历史数据...")
    try:
        backfill()
    except Exception as e:
        print(f"  历史补全失败: {e}")

    # 随机休息
    pause = random.uniform(5, 15)
    print(f"\n  休息 {pause:.0f}秒...")
    time.sleep(pause)

    # Step 3: 刷新看板
    print("\n>>> Step 3: 生成最新看板...")
    try:
        gen_html()
    except Exception as e:
        print(f"  看板生成失败: {e}")

    print(f"\n{'='*70}")
    print(f"每日维护完成 | {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
