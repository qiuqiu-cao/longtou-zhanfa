"""audit_three_board_traps.py

高位连板组（三板及以上：3进4、4进5等空间龙）专属排雷与标签审计模块
依据 2026-10-28 AGENTS.md 最新龙头战法执行约束与盘口突破规则严格实现：
1. 覆盖 3板、4板、5板等全部高位连板候选股，解决四板不适用问题（Issue 4）。
2. 尾盘盘口加撤单（ADD/CANCEL）严格基于封单比值 Seal_Delta（1.2与0.7）落地判定（Issue 2）。
3. 换手真龙特赦豁免条件铁律收紧：必须早盘进攻封死（有效格式 first_seal_time<=10:30）+ 大容量换手 + 尾盘加单锁仓（有效数值 Seal_Delta>=1.2 确认的 ADD），严禁空白时间、无穷大比值、分母为0或仅传字符串 ADD 绕过数值核验（Issue 1 与 Issue 3）。
4. 恶庄诱多（正裕工业型）与尾盘撤单（多伦科技/赛福天型）坚决一票否决！
"""

import os
import sys
import json
import math
from typing import Dict, List, Any, Optional

# 动态标签常量定义
TAG_LATE_ADD_SEAL = "【尾盘加单-真龙锁仓】"     # 绿色：14:30后主力逆势加单顶板(Seal_Delta>=1.2)，大分歧转一致！特赦真龙！
TAG_LATE_CANCEL_SEAL = "【尾盘撤单-心虚出逃】"   # 红色：14:30后封单大幅撤单缩水(Seal_Delta<0.7)，次日抢跑；一票否决！


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
    cohort_3board_size: int = 1,                    # 当日同板位梯队总家数
    is_industry_unknown: bool = False,              # 题材是否未知
    late_order_action: Optional[str] = None,        # 14:30 以后封单行为: "ADD", "CANCEL", "STABLE", or None
    seal_order_1430: Optional[float] = None,        # 14:30 封单挂单金额或手数
    seal_order_1500: Optional[float] = None,        # 15:00 收盘封单金额或手数
    seal_order_delta: Optional[float] = None        # 封单变动比值 (seal_order_1500 / seal_order_1430)
) -> Dict[str, Any]:
    """
    审计高位连板候选股（支持 board_count >= 3：3进4、4进5及以上高标）。
    严格执行 AGENTS.md 2026-10-28 最新约束：
    1. 必须证据齐全方可触发换手真龙特赦豁免；严禁缺失证据默认豁免。
    2. 尾盘加撤单落地盘口 Seal_Delta 计算（>=1.2 为加单，<0.7 为撤单一票否决）。
    3. 严禁空白字符串、无穷大比值、分母为0或裸字符串 ADD 旁路豁免。
    """
    # 覆盖 3板及以上全部高位连板候选
    if board_count < 3:
        return {
            "code": code,
            "name": name,
            "board_count": board_count,
            "applicable": False,
            "message": f"当前板位为 {board_count} 板，非高位连板组（<3板），不适用连板高位排雷审计。"
        }

    b_prefix = "三板" if board_count == 3 else ("四板" if board_count == 4 else f"{board_count}板")
    TAG_CORE_DRAGON = f"【{b_prefix}-换手核心真龙】"
    TAG_ONE_WORD_PIONEER = f"【{b_prefix}-身位一字先锋】"
    TAG_LATE_SEAL_TRAP = f"【{b_prefix}-尾盘偷袭诱多】"
    TAG_CEILING_VILLAIN = f"【{b_prefix}-历史天花板恶庄】"
    TAG_PEER_FOLLOWER = f"【{b_prefix}-同位跟风陪跑】"
    TAG_FAKE_CONCEPT = f"【{b_prefix}-虚假概念套现】"

    tags = []
    hard_veto = False
    reasons = []

    # 1. 严格清洗首封时间（封堵空白字符串、占位值与非正常时间格式）
    clean_first_seal_time = None
    if isinstance(first_seal_time, str):
        s = first_seal_time.strip()
        # 必须是有效的 5 字符 "HH:MM" 格式，且在交易时段内 (09:15 <= HH:MM <= 15:00)
        if len(s) == 5 and s[2] == ':' and s[:2].isdigit() and s[3:].isdigit():
            h, m = int(s[:2]), int(s[3:])
            if (9 <= h <= 15) and (0 <= m < 60) and not (h == 9 and m < 15):
                clean_first_seal_time = f"{h:02d}:{m:02d}"

    # 2. 严格核验盘口封单数值（封堵无穷大、分母为0、负数、NaN及纯字符串假加单旁路）
    eff_delta = None
    if seal_order_delta is not None:
        try:
            d_val = float(seal_order_delta)
            if math.isfinite(d_val) and d_val > 0.0:
                eff_delta = d_val
        except (ValueError, TypeError):
            eff_delta = None

    if eff_delta is None and seal_order_1430 is not None and seal_order_1500 is not None:
        try:
            v1430 = float(seal_order_1430)
            v1500 = float(seal_order_1500)
            # 分母必须为严格正数（>0），分子必须为非负数，比值必须有限
            if math.isfinite(v1430) and math.isfinite(v1500) and v1430 > 0.0 and v1500 >= 0.0:
                calc_delta = v1500 / v1430
                if math.isfinite(calc_delta) and calc_delta >= 0.0:
                    eff_delta = calc_delta
        except (ValueError, TypeError, ZeroDivisionError):
            eff_delta = None

    # 3. 确定尾盘盘口行为：
    # 铁律：特赦豁免依赖的“ADD (加单)”必须有真实合法的数值比值（eff_delta >= 1.2）支撑！
    # 严禁仅凭一个裸字符串 late_order_action="ADD" 绕过数值核验获得特赦！
    eff_late_action = None
    if eff_delta is not None:
        if eff_delta >= 1.2:
            eff_late_action = "ADD"
        elif eff_delta < 0.7:
            eff_late_action = "CANCEL"
        else:
            eff_late_action = "STABLE"
    else:
        # 无有效数值时，仅允许由外部明确标记的 CANCEL（作为保守风控排雷，一票否决），
        # 绝对禁止将未经验证的裸字符串 "ADD" 认定为有效加单！
        if late_order_action == "CANCEL":
            eff_late_action = "CANCEL"
        elif late_order_action == "ADD":
            # 裸字符串 ADD 缺少数值证据支持，拒绝认可，降级为 None
            eff_late_action = None
        elif late_order_action == "STABLE":
            eff_late_action = "STABLE"

    # 4. 尾盘撤单心虚查重 (红色标签 - 绝对硬杀一票否决)
    if eff_late_action == "CANCEL":
        tags.append(TAG_LATE_CANCEL_SEAL)
        hard_veto = True
        delta_str = f"（Seal_Delta={eff_delta:.2f} < 0.7）" if eff_delta is not None else "（外部排雷标记）"
        reasons.append(f"14:30后主力大额撤单缩水超过30%{delta_str}，准备次日竞价抢跑出逃，触发一票否决！")

    # 5. 历史天花板恶庄查重 (红色标签 - 逢顶必核硬杀)
    if historical_board_max <= board_count and history_had_limit_down_after_top:
        tags.append(TAG_CEILING_VILLAIN)
        hard_veto = True
        reasons.append(f"近1年历史最高连板为{historical_board_max}板，且曾发生见顶即核按钮一字跌停的前科（逢顶必核）。")

    # 6. 尾盘偷袭回封查重 (红色标签 - 绝对硬杀)
    if has_afternoon_drop or (clean_first_seal_time is not None and clean_first_seal_time >= "14:00" and turnover_rate > 8.0):
        tags.append(TAG_LATE_SEAL_TRAP)
        hard_veto = True
        reasons.append("14:00后发生严重炸板跳水或尾盘勉强回封，主力高位倒筹码、散户接盘迹象明显。")

    # 7. 虚假概念与风险公告查重 (黄色标签 - 强制剔除)
    if has_risk_announcement:
        tags.append(TAG_FAKE_CONCEPT)
        hard_veto = True
        reasons.append("盘后发布异动风险提示澄清概念无收入或严重亏损，缺乏顶监管的基本面支撑。")

    # 8. 同位梯队拥挤与跟风排位查重 (黄色标签 - 存量博弈淘汰)
    if cohort_3board_size > 3 and not is_theme_leader and not is_industry_unknown:
        tags.append(TAG_PEER_FOLLOWER)
        hard_veto = True
        reasons.append(f"当日{board_count}板梯队拥挤（共{cohort_3board_size}只>3只），本股非题材身位第一，属于中后排跟风。")

    # 9. 大容量换手真龙识别与 BASE 评分豁免特赦机制（严格证据齐全方可豁免）
    # 严格约束：
    # a. 早盘进攻封死：clean_first_seal_time 必须存在且 <= "10:30"（缺失或格式不合规严禁默认合格！）
    # b. 容量换手充沛：turnover_rate >= 15.0% OR amount >= 8e8
    # c. 尾盘加单锁仓：eff_late_action 必须为 "ADD" 且由合法数值 eff_delta >= 1.2 支撑（严禁用题材领先替代，严禁裸字符串旁路！）
    # d. 无恶庄历史且未被硬否决
    is_turnover_dragon = False
    if not hard_veto:
        has_early_seal = (clean_first_seal_time is not None and clean_first_seal_time <= "10:30")
        has_capacity = (turnover_rate >= 15.0 or amount >= 8e8)
        has_late_add = (eff_late_action == "ADD" and eff_delta is not None and eff_delta >= 1.2)
        
        if has_early_seal and has_capacity and has_late_add:
            is_turnover_dragon = True

    # BASE 量化模型底线与特赦检验
    norm_base = None
    if base_score is not None and not (isinstance(base_score, float) and (sys.float_info.max == base_score or base_score != base_score)):
        norm_base = base_score * 100.0 if base_score <= 1.0 else base_score

    if norm_base is None:
        if not is_turnover_dragon:
            hard_veto = True
            reasons.append("BASE 量化模型打分缺失，非证据齐全的特赦大容量真龙，执行一票否决。")
        else:
            reasons.append("BASE 量化打分缺失，但大容量换手真龙盘口证据确凿（早封+大容量+尾盘加单），给予重点观察资格。")
    elif norm_base < 50.0:
        if is_turnover_dragon:
            reasons.append(
                f"BASE 量化模型打分 {norm_base:.1f} 分（<50分），但命中断然齐备的【大容量换手真龙特赦豁免】：早盘进攻封死(<=10:30)+成交额/换手充沛+尾盘主力加单顶封(Seal_Delta>=1.2)，豁免机械低分偏见，准予列为主选！"
            )
        else:
            hard_veto = True
            reasons.append(
                f"BASE 量化模型打分仅 {norm_base:.1f} 分（<50分，明确判定偏不晋级），且未满足全部真龙特赦证据（需早封+大成交+尾盘加单），严禁人为主观调高，一票否决。"
            )

    # 10. 未知题材标记
    if is_industry_unknown:
        reasons.append("题材分类属于独立未知标的，题材身位保持待核，不可假定身位助攻。")

    # 11. 正向真龙与先锋认定 (未被硬否决时方可评定)
    if not hard_veto:
        if eff_late_action == "ADD" and eff_delta is not None and eff_delta >= 1.2:
            tags.append(TAG_LATE_ADD_SEAL)
        if clean_first_seal_time is not None and clean_first_seal_time <= "09:35" and turnover_rate < 5.0:
            tags.append(TAG_ONE_WORD_PIONEER)
        elif is_turnover_dragon or (clean_first_seal_time is not None and clean_first_seal_time <= "10:30" and turnover_rate >= 15.0 and is_theme_leader):
            tags.append(TAG_CORE_DRAGON)
        elif is_theme_leader and not is_industry_unknown:
            tags.append(f"【{b_prefix}-题材候选观察】")

    decision = "【一票否决-严禁接力】" if hard_veto else f"【准予列入{board_count}进{board_count+1}换手真龙主选】"

    return {
        "code": code,
        "name": name,
        "board_count": board_count,
        "applicable": True,
        "seal_delta": round(eff_delta, 3) if eff_delta is not None else None,
        "late_order_action": eff_late_action,
        "tags": tags,
        "veto": hard_veto,
        "veto_reasons": reasons,
        "decision": decision,
        "is_turnover_dragon_exempted": is_turnover_dragon,
        "execution_warning": (
            "次日 09:25 竞价若低开（开绿盘），买入条件单立即永久注销，严禁抄底！"
            if not hard_veto else "已触发一票否决，不设次日买入预案。"
        )
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 70)
    print("【高位连板排雷与盘口规则演示算例（Synthetic Demonstration Probes）】")
    print("【数据资质声明】：本演示中的 14:30 封单挂单金额（如襄阳轴承 9500 万）为人工设定算例探针，")
    print("用于检验规则算子边界；现有分钟档案记录的是成交量与成交额，未采集盘中 14:30 挂单队列，故本部分属于规则算子验证，非历史真实分笔队列证明。")
    print("=" * 70)

    print("测试 1：正裕工业 (603089) - 3板恶庄出货与尾盘撤单测试")
    res_zy = audit_three_board_candidate(
        code="603089",
        name="正裕工业",
        board_count=3,
        is_theme_leader=False,
        first_seal_time="09:25",
        last_seal_time="14:55",
        has_afternoon_drop=True,
        turnover_rate=10.44,
        amount=2.3e8,
        historical_board_max=3,
        history_had_limit_down_after_top=True,
        base_score=31.2,
        has_risk_announcement=True,
        cohort_3board_size=5,
        seal_order_1430=10000000,
        seal_order_1500=4000000     # Seal_Delta = 0.4 < 0.7 尾盘撤单
    )
    print(json.dumps(res_zy, ensure_ascii=False, indent=2))

    print("\n" + "=" * 70)
    print("测试 2：襄阳轴承 (000678) - 4板大容量换手真龙与尾盘加单特赦演示算例 (4进5场景)")
    res_xy = audit_three_board_candidate(
        code="000678",
        name="襄阳轴承",
        board_count=4,              # 4板直接适用！
        is_theme_leader=True,
        first_seal_time="09:36",    # 早盘进攻封死 <= 10:30
        last_seal_time="09:36",
        has_afternoon_drop=False,
        turnover_rate=20.57,        # 充分大换手 20.57%
        amount=1.49e9,              # 14.9亿大容量
        historical_board_max=6,
        history_had_limit_down_after_top=False,
        base_score=27.7,            # BASE仅27.7分
        has_risk_announcement=False,
        cohort_3board_size=3,
        seal_order_1430=95000000,   # 人工规则演示输入（非历史已采集分笔档案）
        seal_order_1500=128000000   # 15:00 收盘静态挂单计算值：Seal_Delta = 1.347 >= 1.2 尾盘加单
    )
    print(json.dumps(res_xy, ensure_ascii=False, indent=2))

    print("\n" + "=" * 70)
    print("测试 3：多伦科技 (603528) - 4板尾盘撤单与炸板诱多演示算例 (4进5场景)")
    res_dl = audit_three_board_candidate(
        code="603528",
        name="多伦科技",
        board_count=4,              # 4板
        is_theme_leader=False,
        first_seal_time="09:25",
        last_seal_time="14:49",
        has_afternoon_drop=True,    # 14:49炸板跳水
        turnover_rate=11.65,
        amount=8.1e8,
        historical_board_max=5,
        history_had_limit_down_after_top=False,
        base_score=42.0,
        has_risk_announcement=False,
        cohort_3board_size=3,
        seal_order_1430=150000000,
        seal_order_1500=45000000    # Seal_Delta = 0.30 < 0.7 尾盘撤单
    )
    print(json.dumps(res_dl, ensure_ascii=False, indent=2))

    print("\n" + "=" * 70)
    print("测试 4：赛福天 (603028) - 4板尾盘撤单心虚演示算例 (4进5场景)")
    res_sft = audit_three_board_candidate(
        code="603028",
        name="赛福天",
        board_count=4,              # 4板
        is_theme_leader=False,
        first_seal_time="09:25",
        last_seal_time="15:00",
        has_afternoon_drop=False,
        turnover_rate=10.31,
        amount=3.0e8,
        historical_board_max=4,
        history_had_limit_down_after_top=False,
        base_score=38.5,
        has_risk_announcement=False,
        cohort_3board_size=3,
        seal_order_delta=0.496      # Seal_Delta = 0.496 < 0.7 尾盘撤单
    )
    print(json.dumps(res_sft, ensure_ascii=False, indent=2))

    print("\n" + "=" * 70)
    print("测试 5：负面防漏测试 - 证据不全严禁豁免（襄阳轴承若缺失尾盘加单数值证据，BASE<50必被一票否决）")
    res_xy_leak = audit_three_board_candidate(
        code="000678",
        name="襄阳轴承_无尾盘加单证据",
        board_count=4,
        is_theme_leader=True,
        first_seal_time="09:36",
        turnover_rate=20.57,
        amount=1.49e9,
        base_score=27.7,
        late_order_action=None      # 缺失尾盘加单证据
    )
    print(json.dumps(res_xy_leak, ensure_ascii=False, indent=2))
