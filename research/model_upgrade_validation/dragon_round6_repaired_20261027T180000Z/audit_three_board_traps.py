"""audit_three_board_traps.py

三板组专属排雷与标签审计模块（依据 2026-10-28 AGENTS.md 最新龙头战法执行约束与盘口突破规则严格实现）
用于精确识别三进四竞争中的【诱多出货套利组】与【真正换手龙头】，并自动打标、特赦真龙与一票否决。
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional

# 三板组核心标签定义
TAG_CORE_DRAGON = "【三板-换手核心真龙】"      # 绿色：早盘换手充分、板块身位第一、无恶庄历史
TAG_ONE_WORD_PIONEER = "【三板-身位一字先锋】" # 绿色：身位一字封死、带动低位扩散
TAG_LATE_ADD_SEAL = "【尾盘加单-真龙锁仓】"   # 绿色：14:30后主力逆势加单顶板，大分歧完全转一致；特赦真龙！
TAG_LATE_SEAL_TRAP = "【三板-尾盘偷袭诱多】"   # 红色：14:00后炸板跳水、尾盘勉强回封；一票否决！
TAG_LATE_CANCEL_SEAL = "【三板-尾盘撤单心虚】" # 红色：14:30后封单大幅撤单缩水，准备次日抢跑；一票否决！
TAG_CEILING_VILLAIN = "【三板-历史天花板恶庄】" # 红色：近1年历史连板极限即3板，且见顶必跌停/核按钮；一票否决！
TAG_PEER_FOLLOWER = "【三板-同位跟风陪跑】"   # 黄色：梯队>3只时的中后排跟风；存量淘汰！
TAG_FAKE_CONCEPT = "【三板-虚假概念套现】"     # 黄色：主营脱节、盘后澄清无收入、亏损套现；强制剔除！


def audit_three_board_candidate(
    code: str,
    name: str,
    board_count: int,
    is_theme_leader: bool,
    first_seal_time: Optional[str] = None,          # e.g. "09:36", "14:28", or None
    last_seal_time: Optional[str] = None,           # e.g. "09:36", "14:55", or None
    has_afternoon_drop: bool = False,               # 14:00 以后是否炸板大幅跳水
    turnover_rate: float = 0.0,                     # 换手率 (%)
    amount: float = 0.0,                            # 成交额 (元)
    historical_board_max: int = 0,                  # 历史连板最高板数
    history_had_limit_down_after_top: bool = False, # 历史上见顶后是否出现过一字跌停/核按钮
    base_score: Optional[float] = None,             # BASE 模型分数 (0-100 或 0.0-1.0)
    has_risk_announcement: bool = False,            # 是否有澄清/巨亏风险公告
    cohort_3board_size: int = 1,                    # 当日三板梯队总家数
    is_industry_unknown: bool = False,              # 题材是否未知
    late_order_action: Optional[str] = None         # 14:30 以后封单行为: "ADD"(加单), "CANCEL"(撤单), "STABLE", or None
) -> Dict[str, Any]:
    """
    审计三板候选股，返回命中的标签列表、接力处置建议及一票否决状态。
    严格执行 AGENTS.md 2026-10-28 最新约束：
    1. 命中红色/黄色标签者，100% 严禁列入最终连板前三或买入建议。
    2. 尾盘撤单（CANCEL）触发一票否决；尾盘加单（ADD）享有真龙溢价预期。
    3. 大容量换手真龙（成交超8亿或换手超15%，且早盘进攻、尾盘加单顶板、无恶庄记录），
       强制启动【BASE 评分豁免特赦】，消除日线量化模型的技术指标误判偏见！
    """
    tags = []
    hard_veto = False
    reasons = []

    if board_count != 3:
        return {
            "code": code,
            "name": name,
            "applicable": False,
            "message": f"当前板位为 {board_count} 板，非三板组，不适用三板专项审计。"
        }

    # 1. 历史天花板恶庄查重 (红色标签 - 绝对硬杀)
    if historical_board_max <= 3 and history_had_limit_down_after_top:
        tags.append(TAG_CEILING_VILLAIN)
        hard_veto = True
        reasons.append("近1年历史最高连板为3板，且曾发生见顶即核按钮一字跌停的前科（逢三必核）。")

    # 2. 尾盘偷袭回封查重 (红色标签 - 绝对硬杀)
    if has_afternoon_drop or (first_seal_time is not None and first_seal_time >= "14:00" and turnover_rate > 8.0):
        tags.append(TAG_LATE_SEAL_TRAP)
        hard_veto = True
        reasons.append("14:00后发生严重炸板跳水或尾盘勉强回封，主力高位倒筹码、散户接盘迹象明显。")

    # 3. 尾盘撤单心虚查重 (红色标签 - 绝对硬杀)
    if late_order_action == "CANCEL":
        tags.append(TAG_LATE_CANCEL_SEAL)
        hard_veto = True
        reasons.append("14:30后主力大单撤单缩水，准备次日竞价抢跑出逃，触发一票否决！")

    # 4. 虚假概念与风险公告查重 (黄色标签 - 强制剔除)
    if has_risk_announcement:
        tags.append(TAG_FAKE_CONCEPT)
        hard_veto = True
        reasons.append("盘后发布异动风险提示澄清概念无收入或严重亏损，缺乏顶监管的基本面支撑。")

    # 5. 同位梯队拥挤与跟风排位查重 (黄色标签 - 存量博弈必杀跟风)
    if cohort_3board_size > 3 and not is_theme_leader and not is_industry_unknown:
        tags.append(TAG_PEER_FOLLOWER)
        hard_veto = True
        reasons.append(f"当日三板梯队拥挤（共{cohort_3board_size}只>3只），本股非题材身位第一，属于中后排跟风。")

    # 6. 大容量换手真龙识别与 BASE 评分豁免特赦机制 (最新 2026-10-28 突破规则)
    is_turnover_dragon = False
    if (turnover_rate >= 15.0 or amount >= 8e8) and (first_seal_time is None or first_seal_time <= "10:30") and (not hard_veto):
        if late_order_action == "ADD" or is_theme_leader:
            is_turnover_dragon = True

    # BASE 量化模型底线与特赦检验
    norm_base = None
    if base_score is not None and not (isinstance(base_score, float) and (sys.float_info.max == base_score or base_score != base_score)):
        norm_base = base_score * 100.0 if base_score <= 1.0 else base_score

    if norm_base is None:
        if not is_turnover_dragon:
            hard_veto = True
            reasons.append("BASE 量化模型打分缺失，非特赦大容量真龙，执行一票否决。")
        else:
            reasons.append("BASE 量化打分缺失，但个股命中大容量换手真龙且尾盘承接健康，给予观察资格。")
    elif norm_base < 50.0:
        if is_turnover_dragon:
            reasons.append(f"BASE 量化模型打分 {norm_base:.1f} 分（<50分），但命中【大容量换手真龙特赦豁免】：成交额/换手充沛且主力主动顶板，豁免机械低分偏见，准予列为主选！")
        else:
            hard_veto = True
            reasons.append(f"BASE 量化模型打分仅 {norm_base:.1f} 分（<50分，明确判定偏不晋级），严禁人为主观调高，一票否决。")

    # 7. 未知题材单独标记
    if is_industry_unknown:
        reasons.append("题材分类属于独立未知标的，题材身位保持待核，不可假定身位助攻。")

    # 8. 正向真龙与先锋认定 (未被硬否决时方可评定)
    if not hard_veto:
        if late_order_action == "ADD":
            tags.append(TAG_LATE_ADD_SEAL)
        if first_seal_time is not None and first_seal_time <= "09:35" and turnover_rate < 5.0:
            tags.append(TAG_ONE_WORD_PIONEER)
        elif is_turnover_dragon or (first_seal_time is not None and first_seal_time <= "10:30" and turnover_rate >= 15.0 and is_theme_leader):
            tags.append(TAG_CORE_DRAGON)
        elif is_theme_leader and not is_industry_unknown:
            tags.append("【三板-题材候选观察】")

    decision = "【一票否决-严禁接力】" if hard_veto else "【准予列入三进四换手真龙主选】"

    return {
        "code": code,
        "name": name,
        "applicable": True,
        "tags": tags,
        "veto": hard_veto,
        "veto_reasons": reasons,
        "decision": decision,
        "execution_warning": (
            "次日 09:25 竞价若低开（开绿盘），买入条件单立即永久注销，严禁抄底！"
            if not hard_veto else "已触发一票否决，不设次日买入预案。"
        )
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("测试 1：正裕工业 (603089) - 恶庄诱多测试")
    res_zy = audit_three_board_candidate(
        code="603089",
        name="正裕工业",
        board_count=3,
        is_theme_leader=False,
        first_seal_time="09:25",
        last_seal_time="14:55",
        has_afternoon_drop=True,
        turnover_rate=10.44,
        amount=2.3e8,                   # 2.3亿小成交
        historical_board_max=3,
        history_had_limit_down_after_top=True,
        base_score=31.2,
        has_risk_announcement=True,
        cohort_3board_size=5,
        late_order_action="CANCEL"      # 尾盘大额撤单出逃
    )
    print(json.dumps(res_zy, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("测试 2：襄阳轴承 (000678) - 大容量换手真龙与尾盘加单特赦测试")
    res_xy = audit_three_board_candidate(
        code="000678",
        name="襄阳轴承",
        board_count=3,
        is_theme_leader=True,           # 机器人/减速器核心
        first_seal_time="09:36",        # 早盘进攻封死
        last_seal_time="09:36",
        has_afternoon_drop=False,
        turnover_rate=20.57,            # 20.57% 充分大换手
        amount=1.275e9,                 # 12.75 亿大成交容量！
        historical_board_max=6,         # 历史无见顶跌停
        history_had_limit_down_after_top=False,
        base_score=27.7,                # BASE仅27.7分（原先被机械误杀）
        has_risk_announcement=False,
        cohort_3board_size=3,
        late_order_action="ADD"         # 14:30 以后主力疯狂加单顶板！
    )
    print(json.dumps(res_xy, ensure_ascii=False, indent=2))
