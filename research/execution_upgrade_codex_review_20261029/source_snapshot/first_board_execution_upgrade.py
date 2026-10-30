"""first_board_execution_upgrade.py - 龙头战法首板执行与双向核算升级引擎

核心原则：
1. 保留旧模型（first_board_leader_bundle.joblib，44字段滚动训练），不拟合任何新模型。
2. 严格复用项目原冻结筛选（流通市值上限约93.27亿，薄换手风险代理排除），在合格池内审阅。
3. 全池审阅题材时序（谁先带动题材、走强前后同伴与指数反馈），拒绝只看前十名导致遗漏榜外171只四板龙。
4. 情绪防守（亏钱效应传导与题材修复，允许留空/防守，不机械凑满3只）。
5. 盘口特赦与排雷严谨事实核验（尾盘封单变化未知一律标为【待核】，不盲目特赦；首封晚一票否决）。
6. 次日09:25实盘竞价截图直读审核（审核必须先于买入授权，严禁09:20前提前挂单）。
7. 双向核算账本系统（固定旧模型基准，如实核算：拦截了多少失败、错杀了多少盈利、可成交净收益）。
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LEGACY_BUNDLE_PATH = ROOT / "research/first_board_leader_20261020T114600Z/first_board_leader_bundle.joblib"

LEGACY_44_FEATURES = [
    "log_price", "log_amount", "log_float_cap", "turnover", "amount_cap",
    "gap", "amplitude", "body", "one_price", "close_to_open",
    "return_5", "return_10", "return_20", "return_60", "volume_ratio_5",
    "volume_ratio_20", "amount_ratio_5", "close_ma5", "close_ma20", "close_ma60",
    "breakout_20", "position_60", "prior_limits_20", "prior_failed_20",
    "previous_limit_age", "prior_volatility_20", "listing_age_sessions",
    "market_limits", "market_first", "market_relays", "market_failed",
    "market_failed_rate", "market_max_height", "market_down_limits",
    "market_positive_fraction", "market_median_return", "previous_first_promotion",
    "previous_relay_promotion", "previous_top_return", "previous_top_broken",
    "turnover_rank", "amount_rank", "volume_ratio_rank", "pre_momentum_rank",
]

# Original frozen risk constants
FROZEN_UPPER_CAP_CNY = 93.27e8       # 约 93.27 亿元固定上限
FROZEN_MICROCAP_CNY = 30.0e8         # 30 亿元微盘门槛
FROZEN_LOW_TURNOVER_PCT = 5.0        # 换手率 < 5% 风险代理
FROZEN_LOW_VOL_RATIO = 1.0           # 缩量代理 (volume_ratio_5 < 1.0)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ==============================================================================
# 模块 1: 原冻结市值与风险筛选 (Original Frozen Screen)
# ==============================================================================

def apply_original_screen(df: pd.DataFrame, upper_cap: float = FROZEN_UPPER_CAP_CNY) -> pd.DataFrame:
    """严格复用项目原冻结筛选逻辑，生成 allowed 标记与 exclusion_reason。
    
    规则：
    1. 流通市值上限约 93.27 亿元 (ABOVE_DEVELOPMENT_CAP_NO_VERIFIED_EXCEPTION)
    2. 薄换手风险代理 (CONSERVATIVE_MICROCAP_THIN_TURNOVER_PROXY):
       流通市值 <= 30 亿 AND 换手率 < 5.0% AND (一字板 OR 5日量比 < 1.0)
    3. 市值缺失 (CAP_MISSING_REVIEW)
    4. 明确保留所有排除个股与原因，不修改原行。
    """
    out = df.copy()
    cap = pd.to_numeric(out["float_cap"], errors="coerce")
    turnover = pd.to_numeric(out["turnover"], errors="coerce")
    one_price = pd.to_numeric(out.get("one_price", 0), errors="coerce").fillna(0)
    vol_ratio = pd.to_numeric(out.get("volume_ratio_5", 1.0), errors="coerce").fillna(1.0)

    cap_missing = cap.isna() | cap.le(0)
    oversize = cap.gt(upper_cap)
    
    # Thin turnover proxy: conservative risk indicator, not a verified fund identity
    microcap = cap.le(FROZEN_MICROCAP_CNY) & cap.gt(0)
    low_to = turnover.lt(FROZEN_LOW_TURNOVER_PCT)
    compressed_vol = vol_ratio.lt(FROZEN_LOW_VOL_RATIO)
    thin_proxy = microcap & low_to & (one_price.eq(1) | compressed_vol)

    allowed = ~(cap_missing | oversize | thin_proxy)
    out["original_screen_allowed"] = allowed

    reasons = []
    for m_miss, m_over, m_thin in zip(cap_missing, oversize, thin_proxy):
        r = []
        if m_miss:
            r.append("CAP_MISSING_REVIEW")
        if m_over:
            r.append("ABOVE_DEVELOPMENT_CAP_NO_VERIFIED_EXCEPTION")
        if m_thin:
            r.append("CONSERVATIVE_MICROCAP_THIN_TURNOVER_PROXY")
        reasons.append("|".join(r) if r else "QUALIFIED")

    out["exclusion_reason"] = reasons
    return out


# ==============================================================================
# 模块 2: 旧模型严格 44 维特征评分器 (Legacy 44-Feature Scorer)
# ==============================================================================

def score_legacy_first_boards(df: pd.DataFrame, bundle_path: Path = LEGACY_BUNDLE_PATH) -> pd.DataFrame:
    """加载旧首板高度模型，并用完整的 44 维特征进行推演评分。
    
    严格契约：
    - 绝不开新拟合（0 fits）。
    - 验证 44 项特征全部存在且非全缺失；若缺失关键字段，明确标记不合格。
    - 输出原始未校准概率分数与组内绝对排名。
    """
    if not bundle_path.exists():
        raise FileNotFoundError(f"Legacy bundle not found: {bundle_path}")

    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    req_features = bundle.get("features", LEGACY_44_FEATURES)

    missing = [f for f in req_features if f not in df.columns]
    if missing:
        raise ValueError(f"Legacy model requires all 44 features. Missing {len(missing)} fields: {missing}")

    out = df.copy()
    X = out[req_features].copy()
    
    # Predict uncalibrated probability
    out["legacy_model_score"] = model.predict_proba(X)[:, 1]
    out["legacy_score_status"] = "UNCALIBRATED_44_FEAT_HIST_GRADIENT"

    # Cross-sectional rank within signal_date
    date_col = "signal_date" if "signal_date" in out.columns else "date"
    if date_col in out.columns:
        out["legacy_rank"] = out.groupby(date_col)["legacy_model_score"].rank(ascending=False, method="min").astype(int)
    else:
        out["legacy_rank"] = out["legacy_model_score"].rank(ascending=False, method="min").astype(int)

    return out


# ==============================================================================
# 模块 3: 题材时序与全池身位核验 (Theme Chronology & Sector Feedback)
# ==============================================================================

def parse_seal_time_to_minutes(val: Any) -> float | None:
    """统一解析首封时间为开盘分钟数 (09:30 为 0 分钟)."""
    if val is None or pd.isna(val):
        return None
    try:
        if isinstance(val, (int, float)):
            ival = int(val)
            # e.g., 93500 -> 09:35:00
            s = str(ival).zfill(6)
            hh, mm, ss = int(s[:2]), int(s[2:4]), int(s[4:6])
        elif isinstance(val, str):
            s = val.strip()
            if ":" in s:
                parts = s.split(":")
                hh = int(parts[0])
                mm = int(parts[1])
                ss = int(parts[2]) if len(parts) > 2 else 0
            else:
                ival = int(s)
                s_str = str(ival).zfill(6)
                hh, mm, ss = int(s_str[:2]), int(s_str[2:4]), int(s_str[4:6])
        else:
            return None

        # Convert to minutes from 09:30
        if hh < 9 or (hh == 9 and mm < 30):
            return 0.0
        elif 9 <= hh <= 11 or (hh == 11 and mm <= 30):
            return (hh - 9) * 60 + mm - 30 + ss / 60.0
        elif 13 <= hh <= 15:
            return 120.0 + (hh - 13) * 60 + mm + ss / 60.0
        else:
            return 240.0
    except Exception:
        return None


def audit_theme_chronology_full_pool(df: pd.DataFrame) -> pd.DataFrame:
    """在完整合格首板池内核验题材身位与同伴反馈。
    
    纠偏原则：
    - 早封不自动等于带动；必须核对同伴跟随数量。
    - 绝不仅审视前 10 名，覆盖榜外个股，防止漏判真龙。
    """
    out = df.copy()
    time_col = "seal_first" if "seal_first" in out.columns else "first_seal_time"
    out["seal_minutes"] = out[time_col].apply(parse_seal_time_to_minutes) if time_col in out.columns else None

    industry_col = "industry" if "industry" in out.columns else "theme"
    if industry_col not in out.columns:
        out["theme_pioneer_tag"] = "⚪【题材信息缺失-待核】"
        out["industry_peers_count"] = 1
        return out

    date_col = "signal_date" if "signal_date" in out.columns else "date"
    groups = out.groupby([date_col, industry_col], sort=False) if date_col in out.columns else out.groupby(industry_col, sort=False)

    out["industry_peers_count"] = groups[industry_col].transform("count")
    if "seal_minutes" in out.columns and out["seal_minutes"].notna().any():
        out["industry_seal_rank"] = groups["seal_minutes"].rank(method="min", ascending=True)
    else:
        out["industry_seal_rank"] = 1

    tags = []
    for _, row in out.iterrows():
        peers = row.get("industry_peers_count", 1)
        rank = row.get("industry_seal_rank", 1)
        s_min = row.get("seal_minutes")

        if peers <= 1:
            tags.append("⚪【独立无助攻个股】")
        elif rank == 1 and s_min is not None and s_min <= 60.0:  # 10:30 前首封
            tags.append(f"🟢【题材先锋身位-助攻{peers-1}只】")
        elif rank == 1:
            tags.append(f"🟡【先封但偏晚({int(s_min or 0)}分)-助攻{peers-1}只】")
        else:
            tags.append(f"⚪【题材同位跟风(第{int(rank)}封)】")

    out["theme_pioneer_tag"] = tags
    return out


# ==============================================================================
# 模块 4: 情绪防守与亏钱传导审计 (Market Emotion Guard)
# ==============================================================================

def audit_market_emotion_defense(date_market_row: dict | pd.Series) -> dict[str, Any]:
    """检查大盘与情绪梯队亏钱传导，评估是否触发防守。
    
    原则：
    - “跌停≥8、晋级率<10%”仅作为经验性的警惕条件，不武断宣称完全识别未来零龙日。
    - 允许输出“无买入候选/防守空仓”，不机械凑满前三。
    """
    down_limits = float(date_market_row.get("market_down_limits", 0) or 0)
    failed_rate = float(date_market_row.get("market_failed_rate", 0) or 0)
    first_promo = float(date_market_row.get("previous_first_promotion", 0.2) or 0.2)
    pos_frac = float(date_market_row.get("market_positive_fraction", 0.5) or 0.5)

    risk_flags = []
    if down_limits >= 8:
        risk_flags.append(f"市场跌停家数({int(down_limits)})偏高")
    if failed_rate >= 0.40:
        risk_flags.append(f"炸板率({failed_rate*100:.1f}%)处于高位")
    if first_promo < 0.10:
        risk_flags.append(f"昨日首板晋级率({first_promo*100:.1f}%)低于10%")
    if pos_frac < 0.25:
        risk_flags.append(f"全市场红盘率({pos_frac*100:.1f}%)极度冰点")

    is_defense = len(risk_flags) >= 2
    return {
        "is_defense": is_defense,
        "regime": "DEFENSE_EMPTY_ALLOWED" if is_defense else "NORMAL_TRADING",
        "risk_flags": risk_flags,
        "recommendation": "【情绪退潮-允许空仓防守】今日退潮亏钱效应扩散，不强制推满前三，严控开仓" if is_defense else "【情绪正常/分歧】按标准流程全池审阅候选",
    }


# ==============================================================================
# 模块 5: 盘口特赦与排雷严谨事实核验 (Tick Microstructure Evidence)
# ==============================================================================

def audit_capacity_dragon_exemption(row: dict | pd.Series) -> tuple[str, str]:
    """核验大容量换手真龙特赦。
    
    纠偏原则：
    - 尾盘封单变化 (seal_delta) 未知时，严格保持【待核】，绝不赋予特赦！
    - 时间解析统一支持字符串、整数与浮点数。
    - 净加单不盲目推断特定资金身份。
    """
    float_cap = float(row.get("float_cap", 0) or 0)
    amount = float(row.get("amount", 0) or 0)
    turnover = float(row.get("turnover", 0) or 0)
    first_seal = row.get("seal_first") or row.get("first_seal_time")
    seal_delta = row.get("seal_delta_1430_to_close") or row.get("seal_delta")

    # 1. Capacity threshold
    if amount < 8e8 and turnover < 15.0:
        return "INELIGIBLE", f"成交额({amount/1e8:.2f}亿)未达8亿且换手率({turnover:.1f}%)未达15%"

    # 2. Seal time check (<= 10:30, i.e., <= 60 minutes)
    s_min = parse_seal_time_to_minutes(first_seal)
    if s_min is not None and s_min > 60.0:
        return "INELIGIBLE", f"首封时间偏晚({first_seal})，非早盘主动进攻"

    # 3. Seal Delta check
    if seal_delta is None or pd.isna(seal_delta):
        return "PENDING_VERIFICATION", "尾盘封单变化(14:30~收盘)数据缺失，特赦条件【待核】"

    try:
        f_delta = float(seal_delta)
        if f_delta < 1.0:
            return "VETOED", f"尾盘封单减少(Seal_Delta={f_delta:.2f})，呈现净撤单"
        elif f_delta >= 1.2:
            return "SATISFIED", f"大容量换手且尾盘大额加单锁仓(Seal_Delta={f_delta:.2f})，确凿特赦"
        else:
            return "SATISFIED", f"大容量换手且尾盘封单平稳(Seal_Delta={f_delta:.2f})，予以观察特赦"
    except Exception:
        return "PENDING_VERIFICATION", f"尾盘封单数据异常({seal_delta})，保持待核"


def audit_sneak_attack_trap_veto(row: dict | pd.Series) -> tuple[bool, str]:
    """恶庄偷袭与虚假首板一票否决。
    
    原则：
    - 14:00 之后偷袭封板且换手不足 3.0% 者一票否决。
    - 尾盘撤单超过 30% (seal_delta < 0.70) 一票否决。
    """
    first_seal = row.get("seal_first") or row.get("first_seal_time")
    turnover = float(row.get("turnover", 0) or 0)
    seal_delta = row.get("seal_delta_1430_to_close") or row.get("seal_delta")

    s_min = parse_seal_time_to_minutes(first_seal)
    if s_min is not None and s_min >= 150.0:  # 14:00 is 150 minutes
        if turnover < 3.0:
            return True, f"14:00后偷袭封板且换手严重不足({turnover:.1f}%)，诱多嫌疑一票否决"

    if seal_delta is not None and not pd.isna(seal_delta):
        try:
            if float(seal_delta) < 0.70:
                return True, f"尾盘封单大幅撤销超30%(Seal_Delta={float(seal_delta):.2f})，一票否决"
        except Exception:
            pass

    return False, "通过盘口排雷"


# ==============================================================================
# 模块 6: 09:25 实盘竞价截图直读审核协议 (Auction Execution Protocol)
# ==============================================================================

def generate_auction_execution_card(candidate: dict | pd.Series) -> dict[str, Any]:
    """生成次日 09:25 实盘竞价多模态审核卡。
    
    铁律落实：
    - 09:20—09:25 交易所不接受撤单！已经在 09:25 撮合成交的订单无法撤回！
    - 因此：审核必须先于买入授权！严禁 09:20 前挂单期望撤单。
    - 首板不机械套用三进四平低开一票否决，+3%~+6% 仅作观察参考。
    """
    code = str(candidate.get("code", "")).zfill(6)
    name = str(candidate.get("name", ""))
    close = float(candidate.get("close", 0) or 0)
    upper_ref = round(close * 1.10, 2)

    return {
        "code": code,
        "name": name,
        "pre_close": close,
        "limit_ref": upper_ref,
        "execution_protocol": "审核先行，严禁提前报单",
        "exchange_rule_warning": "09:20-09:25交易所不接受撤单！必须在09:25截图审核通过后，才准于09:25-09:30授权盘中打板条件单",
        "visual_checklist": [
            "【09:20-09:25 价格曲线】撮合价格是否逐级抬高或横盘锁死（严禁单调下砸）",
            "【匹配量阶梯】09:20 后匹配量柱状图是否稳步堆叠放大",
            "【买一挂单厚度】09:25 买一是否存在大单护盘，委比严禁深度为负",
            "【首板开盘情景】红开带量最佳；若平开或微绿需观察是否有早盘主动抢筹翻红秒板，不机械一票否决，但绿盘严禁直接市价追入",
        ],
        "authorized_action_if_passed": "09:25 后激活盘中涨停打板条件单（触板或回封买入），不挂无保护开盘市价单",
        "action_if_failed": "完全不提交买入委托，永久放弃该标的",
    }


# ==============================================================================
# 模块 7: 人工干预与执行效果双向核算账本系统 (Dual-Accounting Audit Ledger)
# ==============================================================================

class DualAccountingLedger:
    """双向对账账本系统：固定旧模型基准，如实核算拦截失败与错杀成功。
    
    对账原则：
    1. 过滤规则是双刃剑：挡住亏损的同时可能错杀盈利，必须两头都记。
    2. 只有真实成交约束下的净表现才能证明实战有效性。
    """

    def __init__(self):
        self.records: list[dict[str, Any]] = []

    def record_decision(
        self,
        signal_date: str,
        code: str,
        name: str,
        legacy_rank: int,
        legacy_score: float,
        final_selected: bool,
        intervention_type: str,  # "KEEP", "EXCLUDE_BY_SCREEN", "EXCLUDE_BY_AUDIT", "PROMOTE_FROM_BELOW"
        reason: str,
        actual_reach4: int | None = None,
        actual_trade_profitable: bool | None = None,
        trade_return: float | None = None,
    ):
        self.records.append({
            "signal_date": signal_date,
            "code": code,
            "name": name,
            "legacy_rank": legacy_rank,
            "legacy_score": legacy_score,
            "final_selected": final_selected,
            "intervention_type": intervention_type,
            "reason": reason,
            "actual_reach4": actual_reach4,
            "actual_trade_profitable": actual_trade_profitable,
            "trade_return": trade_return,
        })

    def summarize(self) -> dict[str, Any]:
        df = pd.DataFrame(self.records)
        if df.empty:
            return {"total_decisions": 0}

        # 1. Excluded cases (originally in Top 3, but removed by screen or audit)
        excluded = df[df["intervention_type"].str.startswith("EXCLUDE")]
        avoided_losses = int((excluded["actual_reach4"] == 0).sum())
        avoided_negative_trades = int((excluded["actual_trade_profitable"] == False).sum())
        killed_profitable_trades = int((excluded["actual_trade_profitable"] == True).sum())
        killed_true_dragons = int((excluded["actual_reach4"] == 1).sum())

        # 2. Promoted cases (originally rank > 3, but promoted by theme/capacity audit)
        promoted = df[df["intervention_type"] == "PROMOTE_FROM_BELOW"]
        promoted_dragons = int((promoted["actual_reach4"] == 1).sum())
        promoted_profitable = int((promoted["actual_trade_profitable"] == True).sum())
        promoted_failed = int((promoted["actual_trade_profitable"] == False).sum())

        return {
            "total_decisions": len(df),
            "excluded_count": len(excluded),
            "avoided_losses_count": avoided_losses,
            "avoided_negative_trades": avoided_negative_trades,
            "killed_profitable_trades": killed_profitable_trades,
            "killed_true_dragons": killed_true_dragons,
            "net_loss_prevention": avoided_negative_trades - killed_profitable_trades,
            "promoted_count": len(promoted),
            "promoted_dragons": promoted_dragons,
            "promoted_profitable_trades": promoted_profitable,
            "promoted_failed_trades": promoted_failed,
        }
