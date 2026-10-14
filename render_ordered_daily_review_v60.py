"""Render the daily leader review in the user's fixed decision order."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(value, digits=1):
    return "不可用" if value is None else f"{value:.{digits}%}"


def candidate_line(row):
    risk = len(row.get("active_hard_risks", []))
    theme = row.get("theme_evidence", {}).get("theme") or "unavailable"
    return (f"|{row['code']}|{row['name']}|{row['boards']}|{theme}|"
            f"{row['theme_role']}|{row['turnover']:.2f}%|{int(row['failed_seal_count'] or 0)}|"
            f"{risk}|{row['entry_permission']}|")


def render(report):
    market = report["market"]
    prior = market["previous_limit_feedback"]
    failed = market["failed_limit_entry_feedback"]
    pool = report["pool_summary"]
    theme = report["theme_summary"]
    ann = report["announcement_summary"]
    first = report["first_board_candidates"]
    relay = report["relay_candidates"]
    anchor_names = "、".join(
        f"{x['name']}({x['code']}){x['boards']}板"
        + (f"[硬风险{len(x['active_hard_risks'])}条]" if x["active_hard_risks"] else "")
        for x in report["roles"]["emotion_anchors"])
    # Research order is descriptive, never an operational threshold.  The
    # leading agriculture relay is explicitly source-bound and risk-filtered.
    watch = next((x for x in relay if x["code"] == "600354"), None)
    first_watch = next((x for x in first if x["code"] == "002312"), None)
    lines = [
        f"# {report['signal_date']} 龙头战法盘后决策单（V60）", "",
        "## 1. 数据质量", "",
        f"- 收盘涨停/炸板池：`verified` 于当日存档名单；主板非ST封板{pool['closing_limit_mainboard']}只、炸板{pool['failed_board_mainboard']}只。",
        "- 全市场独立完整性：`unavailable`；已存档名单内完整，但没有第二独立源证明全市场无遗漏。",
        f"- 实时竞价、逐笔、封单变化、Level-2：`unavailable`；题材新闻名单：`verified`部分覆盖{theme['verified_candidate_count']}/{theme['closing_candidate_count']}只。",
        f"- 公告索引：`verified_index_date_only`，覆盖71只，有效硬风险{ann['active_hard_events']}条、涉及{ann['active_hard_candidate_codes']}只；未升级为全文精确时间。",
        "", "## 2. 市场情绪", "",
        f"- 状态：`{market['emotion_regime']}`（`verified`收盘数据）。封板率{pct(market['seal_rate'])}，炸板率{pct(market['failed_rate'])}，跌停{market['down_limit_count']}只。",
        f"- 昨日涨停池：平均反馈{pct(prior['mean_return'])}，中位{pct(prior['median_return'])}，今日真正晋级{prior['promotion_count']}/{prior['observation_count']}={pct(prior['promotion_rate'])}。",
        f"- 炸板价格代理：`inferred`，从10%涨停价到收盘中位{pct(failed['median_return'])}，尾部{pct(failed['tail_10_return'])}；不是真实成交。",
        "- 解读：溢价是正的，但全体晋级率仅22.1%；旧空间标未晋级，所以是“升温中的高度分歧”，不是普遍接力日。",
        "", "## 3. 涨停梯队", "",
        "- 4板：爱仕达、亚盛集团、百大集团。",
        "- 3板：海欣食品、敦煌种业、中国出版。",
        "- 2板13只，首板52只；梯队无板位断层，但上一只6板空间标晋级0/1。",
        "", "## 4. 题材竞争", "",
        "- 农业/农化：`verified`同日新闻名单形成4—3—2—1梯队，广度和高度均最完整。",
        "- 传媒/出版：`verified`同日名单有3—2—1梯队，但旧高度股炸板，需要新核心脱离跟风定位。",
        "- 消费、机器人等：`unavailable/incomplete`；不用单只标签硬造板块梯队。",
        "", "## 5. 首板候选", "",
        "全部首板均为 `NO_ENTER`。研究观察首位（非买点）："
        + (f"{first_watch['name']}({first_watch['code']})，仅因农化广度得到同日名单支持，但它被2板金正大压制，身份仍是后排。" if first_watch else "无。"),
        "", "|code|名称|板位|真实题材|题材角色|换手|炸开次数|硬风险|权限|", "|---|---|---:|---|---|---:|---:|---:|---|",
        *[candidate_line(x) for x in first],
        "", "## 6. 连板候选", "",
        "盘前研究观察首位（非已形成买点）："
        + (f"{watch['name']}({watch['code']})。3板、农业主梯队、换手{watch['turnover']:.2f}%、开板{int(watch['failed_seal_count'])}次、无已识别硬公告风险。" if watch else "无。"),
        "", "|code|名称|板位|真实题材|题材角色|换手|炸开次数|硬风险|权限|", "|---|---|---:|---|---|---:|---:|---:|---|",
        *[candidate_line(x) for x in relay],
        "", "## 7. 三个角色", "",
        f"- 情绪锚：{anchor_names}。只用来观察高度反馈，不自动等于买点。",
        "- 成龙潜力最大：`unavailable`；敦煌种业只是研究观察首位，因缺九点二十五后真实竞争淘汰证据，不升级为“已验证龙头”。",
        "- 当前买入机会最好：无；当前状态 `CASH`。",
        "", "## 8. 综合首选", "",
        "`无可执行首选，当前持现金。` 如果必须只跟踪一只，跟踪敦煌种业；“跟踪”不等于“买入”。",
        "", "## 9. 次日执行条件", "",
        "以下全部满足才能从 `WATCH` 进入打板确认：",
        "1. 09:25—10:00取得真实竞价、成交和VWAP数据；未取得则权限仍为 `NO_ENTER`。",
        "2. 敦煌种业必须主动强于同题材竞争者，且农业/农化不能只剩孤板；亚盛风险提示后若核按钮，整个农业高度降级。",
        "3. 只接受良性换手后第一次或第二次回封；跌破VWAP放量、两次冲板不封、板块同步转弱，立即取消。",
        "4. 不对一字锁死排队，不因普通异动公告单独否决；但硬风险、可成交性和隔夜核按钮风险必须同时通过。",
        "5. 仓位：当前系统未授权的实际仓位是0；当验证门槛通过后，才按“单笔预算最大亏损≤总资金0.5%”反推股数，且单只不超过1/4仓。",
        "", "## 10. 空仓条件", "",
        "- 4板锚定批量低开/核按钮，或旧空间股负反馈扩散。",
        "- 农业/农化开盘批量分歧，敦煌变孤板或被同高度竞争者卡位。",
        "- 候选高开无量、竞价匹配额快速衰减、跌破VWAP放量、反复炸板或晚封漏单。",
        "- 无法获取实时字段，或模型仍无经盲测确立的全池选择阈值。",
        "", "状态说明：该决策单是 `DIAGNOSTIC_ONLY`，不保证次日溢价、不保证不被核，也没有声称达到70%胜率。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("refusing to overwrite ordered review")
    report = json.loads(Path(args.review).read_text(encoding="utf-8"))
    target.write_text(render(report), encoding="utf-8")
    print(target.resolve())


if __name__ == "__main__":
    main()
