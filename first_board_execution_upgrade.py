"""Frozen-model review utilities. No fitting, order submission or profit claims."""
from __future__ import annotations
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import math
import re
import joblib
import numpy as np
import pandas as pd
from first_board_leader_model import FEATURES as _ORIGINAL_FEATURES

ROOT = Path(__file__).resolve().parent
LEGACY_BUNDLE_PATH = ROOT / "research/first_board_leader_20261020T114600Z/first_board_leader_bundle.joblib"
LEGACY_44_FEATURES = list(_ORIGINAL_FEATURES)
FROZEN_UPPER_CAP_CNY = 9326859896.52
FROZEN_MICROCAP_CNY = 30e8
FROZEN_LOW_TURNOVER_PCT = 5.0
FROZEN_LOW_VOL_RATIO = 1.0

def now_utc_iso(): return datetime.now(timezone.utc).isoformat()

def _number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError): return None

def _first_value(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None and not pd.isna(value): return value
    return None

def _true(value):
    return str(value).lower() in {"true", "1", "1.0"}

def _known_bool(value):
    text = str(value).lower()
    if text in {"true", "1", "1.0"}: return True
    if text in {"false", "0", "0.0"}: return False
    return None

def apply_original_screen(df, upper_cap=FROZEN_UPPER_CAP_CNY):
    """Reproduce original proxy flags; incomplete review evidence stays pending."""
    required = {"float_cap", "turnover", "one_price", "volume_ratio_5"}
    if required - set(df): raise ValueError(f"Missing screen fields: {sorted(required-set(df))}")
    out = df.copy()
    cap, turnover, one_price, vr = [pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
                                   for c in ("float_cap", "turnover", "one_price", "volume_ratio_5")]
    cap_missing = cap.isna() | cap.le(0)
    restructure = out.get("verified_restructuring_preclose", pd.Series(False, index=out.index)).eq(1)
    risk = out.get("verified_threeboard_risk_preclose", pd.Series(False, index=out.index)).eq(1)
    oversize = cap.gt(upper_cap) & ~restructure
    thin = cap.le(FROZEN_MICROCAP_CNY) & cap.gt(0) & turnover.lt(5) & (one_price.eq(1) | vr.lt(1))
    out["original_screen_allowed"] = ~(cap_missing | oversize | thin | risk)
    missing_proxy = turnover.isna() | (cap.le(FROZEN_MICROCAP_CNY) & turnover.lt(5) &
                                      ~one_price.eq(1) & (one_price.isna() | vr.isna()))
    out["screen_review_status"] = np.where(~out.original_screen_allowed, "EXCLUDED",
                                           np.where(missing_proxy, "PENDING_VERIFICATION", "QUALIFIED"))
    labels = ["CAP_MISSING_REVIEW", "ABOVE_DEVELOPMENT_CAP_NO_VERIFIED_EXCEPTION",
              "CONSERVATIVE_MICROCAP_THIN_TURNOVER_PROXY", "VERIFIED_THREEBOARD_RISK"]
    out["exclusion_reason"] = ["|".join(label for flag, label in zip(flags, labels) if flag) or
                                ("SCREEN_EVIDENCE_PENDING" if pending else "QUALIFIED")
                                for flags, pending in zip(zip(cap_missing, oversize, thin, risk), missing_proxy)]
    return out

def _rank(rows, score):
    ordered = rows.sort_values(["signal_date", score, "code"], ascending=[True, False, True])
    return ordered.groupby("signal_date").cumcount().add(1).reindex(rows.index)

def score_legacy_first_boards(df, bundle_path=LEGACY_BUNDLE_PATH):
    """New dates only. Historical audits use saved monthly OOF scores."""
    bundle = joblib.load(bundle_path)
    features = bundle["features"]
    if len(features) != 44 or set(features) != set(LEGACY_44_FEATURES): raise ValueError("Unexpected model features")
    missing = set(features + ["signal_date", "code", "first", "eligible", "scope_known"]) - set(df)
    if missing: raise ValueError(f"Missing {len(missing)} fields: {sorted(missing)}")
    if df.empty or df.duplicated(["signal_date", "code"]).any(): raise ValueError("Empty or duplicate candidate keys")
    if not df.code.astype(str).str.fullmatch(r"\d{6}").all(): raise ValueError("Invalid six-digit code")
    if pd.to_datetime(df.signal_date, format="%Y-%m-%d", errors="coerce").isna().any(): raise ValueError("Invalid dates")
    if not df.signal_date.gt(bundle["knowledge_cutoff"]).all(): raise ValueError("Historical dates require saved monthly OOF scores")
    for c in ("first", "eligible", "scope_known"):
        if not df[c].map(_true).all(): raise ValueError(f"Unverified candidate scope: {c}")
    X = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    for _, group in X.groupby(df.signal_date):
        if group.isna().all().any(): raise ValueError("A required feature is entirely missing on the signal date")
    out = df[[c for c in df if "LABEL_ONLY" not in c and c not in ("label_end", "label_observed")]].copy()
    out["legacy_model_score"] = bundle["model"].predict_proba(X)[:, 1]
    out["legacy_score_status"] = "UNCALIBRATED_FROZEN_44_FEATURES_NEW_DATE"
    out["legacy_rank"] = _rank(out, "legacy_model_score")
    return out

def load_historical_legacy_scores(facts, predictions):
    """Bind exact complete cohorts to the original OOF score, no fallback."""
    keys = ["signal_date", "code"]
    if facts.empty or any(d.duplicated(keys).any() for d in (facts, predictions)):
        raise ValueError("Empty or duplicate candidate keys")
    for c in keys + ["model_score"]:
        if c not in predictions: raise ValueError(f"Missing historical prediction field: {c}")
    a = set(map(tuple, facts[keys].to_numpy()))
    b = set(map(tuple, predictions[keys].to_numpy()))
    if a != b: raise ValueError("Historical prediction cohort mismatch; no static-model fallback")
    out = facts.merge(predictions[keys+["model_score"]], on=keys, validate="one_to_one")
    if not np.isfinite(pd.to_numeric(out.model_score, errors="coerce")).all(): raise ValueError("Unknown historical scores")
    out["legacy_model_score"] = out.model_score
    out["legacy_score_status"] = "SAVED_MONTHLY_252_OOF_UNCALIBRATED"
    out["legacy_rank"] = _rank(out, "model_score")
    return out

def _seal_seconds(value):
    if value is None or pd.isna(value): return None
    s = str(value).strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) not in (2, 3) or not all(x.isdigit() for x in parts): return None
        h, m = map(int, parts[:2]); sec = int(parts[2]) if len(parts) == 3 else 0
    else:
        number = _number(value)
        if number is None or number != int(number): return None
        s = str(int(number)).zfill(6)
        if len(s) != 6 or not s.isdigit(): return None
        h, m, sec = int(s[:2]), int(s[2:4]), int(s[4:])
    if not (0 <= h < 24 and 0 <= m < 60 and 0 <= sec < 60): return None
    t = h*3600+m*60+sec
    return t if (9*3600+25*60 <= t <= 11*3600+30*60) or (13*3600 <= t <= 15*3600) else None

def parse_seal_time_to_minutes(value):
    t = _seal_seconds(value)
    if t is None: return None
    return (t-(9*3600+30*60))/60 if t < 13*3600 else 120+(t-13*3600)/60

def audit_theme_chronology_full_pool(df):
    """Describe observed group order only, without causal or role labels."""
    out = df.copy()
    if not out.index.is_unique: raise ValueError("Duplicate frame index")
    time_col = "seal_first" if "seal_first" in out else "first_seal_time"
    times = out[time_col] if time_col in out else pd.Series(None, index=out.index, dtype=object)
    out["seal_minutes"] = times.map(parse_seal_time_to_minutes)
    seconds = times.map(_seal_seconds)
    out["theme_pioneer_tag"] = "⚪【题材关系与首封次序待核】"
    out["chronology_status"] = "PENDING_VERIFICATION"
    out["industry_peers_count"] = np.nan
    out["industry_seal_rank"] = np.nan
    for n in (1, 3, 5): out[f"same_group_later_seals_{n}m"] = np.nan
    group_col = "theme" if "theme" in out else "industry" if "industry" in out else None
    out["group_basis"] = "THEME_MEMBERSHIP_UNVERIFIED" if group_col == "theme" else "INDUSTRY_PROXY" if group_col else "UNKNOWN"
    if not group_col or "signal_date" not in out: return out
    for _, group in out.groupby(["signal_date", group_col], dropna=False, sort=False):
        if group[group_col].isna().any() or group[group_col].astype(str).str.strip().eq("").any(): continue
        idx = group.index
        out.loc[idx, "industry_peers_count"] = len(group)
        out.loc[idx, "industry_seal_rank"] = seconds.loc[idx].rank(method="min")
        complete = seconds.loc[idx].notna().all()
        out.loc[idx, "chronology_status"] = "OBSERVED_ORDER_ONLY" if complete else "INCOMPLETE_GROUP_TIMES"
        for i in idx:
            t = seconds.loc[i]
            if pd.isna(t): continue
            for n in (1, 3, 5):
                out.loc[i, f"same_group_later_seals_{n}m"] = int(((seconds.loc[idx]>t)&(seconds.loc[idx]<=t+n*60)).sum())
            out.loc[i, "theme_pioneer_tag"] = "⚪【同组首封次序观察-带动关系待核】" if complete else "⚪【同组时刻不全-先后关系待核】"
    return out

def audit_market_emotion_defense(row):
    fields = ["market_down_limits", "market_failed_rate", "previous_first_promotion", "market_positive_fraction"]
    values = {c:_number(row.get(c)) for c in fields}
    unknown = [c for c,v in values.items() if v is None or (v < 0 or (c != "market_down_limits" and v > 1))]
    flags = []
    for c, predicate, text in [
        (fields[0], lambda v:v>=8, "跌停家数偏高"), (fields[1], lambda v:v>=.4, "炸板率偏高"),
        (fields[2], lambda v:v<.1, "昨日首板晋级率低于10%"), (fields[3], lambda v:v<.25, "红盘率低于25%")]:
        if c not in unknown and predicate(values[c]): flags.append(text)
    defense = True if len(flags)>=2 else None if unknown else False
    return dict(is_defense=defense, regime="DEFENSE_REVIEW" if defense else "UNKNOWN" if defense is None else "NO_DEFENSE_TRIGGER_OBSERVED",
                risk_flags=flags, unknown_fields=unknown, rule_status="UNVALIDATED_MANUAL_OBSERVATION_RULE",
                recommendation="允许防守留空，须核验梯队传导与题材修复" if defense else "情绪证据待核" if defense is None else "尚未触发本观察规则；仍须逐股审阅")

def audit_capacity_dragon_exemption(row):
    if _true(row.get("has_bad_history")): return "VETOED", "已标记历史连续跌停风险，拒绝特赦"
    cap = _number(row.get("float_cap"))
    if cap is None or cap<=0: return "PENDING_VERIFICATION", "流通市值待核"
    allowed = row.get("original_screen_allowed")
    if cap>FROZEN_UPPER_CAP_CNY or (allowed is not None and not pd.isna(allowed) and not _true(allowed)):
        return "INELIGIBLE", "不在首板原筛选范围内"
    amount, turnover = _number(row.get("amount")), _number(row.get("turnover"))
    capacity = (amount is not None and amount>=8e8) or (turnover is not None and turnover>=15)
    if not capacity:
        return ("PENDING_VERIFICATION", "成交额或换手率缺失") if amount is None or turnover is None else ("INELIGIBLE", "未达8亿成交额或15%换手条件")
    first = _first_value(row, "seal_first", "first_seal_time")
    minutes = parse_seal_time_to_minutes(first)
    if minutes is None: return "PENDING_VERIFICATION", "首封时间缺失或异常，待核"
    if minutes>60: return "INELIGIBLE", "10:30后首封，非本特赦的早封条件"
    delta = _number(_first_value(row, "seal_delta_1430_to_close", "seal_delta"))
    if delta is None or delta<0: return "PENDING_VERIFICATION", "尾盘封单变化缺失或异常，待核"
    if not _true(row.get("seal_delta_verified")) or not row.get("seal_delta_source"):
        return "PENDING_VERIFICATION", "缺少已核同口径14:30与收盘封单来源，待核"
    if delta<.7: return "VETOED", "已核尾盘封单减少超过30%"
    if delta<1.2: return "INELIGIBLE", "尾盘封单比未达1.2加单条件"
    if not _true(row.get("history_verified")) or _known_bool(row.get("has_bad_history")) is not False or not row.get("history_source"):
        return "PENDING_VERIFICATION", "近一年连续跌停风险核查待核"
    return "SATISFIED", "满足本人工复核条件；不认证资金身份或直接授权买入"

def audit_sneak_attack_trap_veto(row):
    if _true(row.get("has_bad_history")): return True, "已标记历史连续跌停风险，取消参与"
    minutes = parse_seal_time_to_minutes(_first_value(row, "seal_first", "first_seal_time"))
    turnover = _number(row.get("turnover"))
    if minutes is not None and minutes>=180 and turnover is not None and turnover<3:
        return True, "14:00后首封且低换手风险代理，取消参与；不认证恶庄身份"
    delta = _number(_first_value(row, "seal_delta_1430_to_close", "seal_delta"))
    if delta is not None and 0<=delta<.7 and _true(row.get("seal_delta_verified")) and row.get("seal_delta_source"):
        return True, "已核尾盘封单减少超过30%，取消参与"
    return False, "未命中已核否决条件；未知盘口与历史风险仍待核，不授权买入"

def generate_auction_execution_card(candidate):
    code = str(candidate.get("code", "")).zfill(6)
    close = _number(candidate.get("close"))
    if not re.fullmatch(r"\d{6}", code) or close is None or close<=0: raise ValueError("Invalid code or previous close")
    if not code.startswith(("000", "001", "002", "003", "600", "601", "603", "605")):
        raise ValueError("This card supports regular 10% main-board references only")
    upper = float((Decimal(str(close))*Decimal("1.10")).quantize(Decimal(".01"), rounding=ROUND_HALF_UP))
    target = candidate.get("target_date")
    return dict(code=code, name=str(candidate.get("name", "")), pre_close=close, limit_ref=upper,
        signal_date=candidate.get("signal_date"), target_date=target, reference_status="REGULAR_10_PERCENT_REFERENCE_REQUIRES_CORPORATE_ACTION_CHECK",
        execution_protocol="审核先行，未触发已记录条件不提交买入委托", authorization_status="PENDING_HUMAN_VISUAL_REVIEW" if target else "PENDING_TARGET_DATE",
        exchange_rule_warning="09:20-09:25交易所不接受撤单；09:25已撮合成交不可撤回。待截图审核后评估盘中触发条件。",
        visual_checklist=["核09:20-09:25撮合价格曲线及截图范围", "核实际竞价成交、未匹配方向与量，缺数据标待核",
                          "结合昨日形态、同位和题材确认对象审阅，不机械套统一高开区间", "逐项记录09:30及09:35后的触发与取消条件，竞价通过不等于可买"],
        authorized_action_if_passed="人工审阅后，仍须满足另行记录的盘中触发与成交条件；本卡不自动授权下单",
        action_if_failed="不提交新的买入委托，记录取消原因")

class DualAccountingLedger:
    """Trade profit is known only for closed scenarios, separately from height."""
    def __init__(self):
        self.records = []
        self._keys = set()

    def record_decision(self, signal_date, code, name, legacy_rank, legacy_score, final_selected,
                        intervention_type, reason, actual_reach4=None, actual_trade_profitable=None,
                        trade_return=None, trade_status=None, trade_net_pnl=None):
        key = (signal_date, code)
        if key in self._keys: raise ValueError("Duplicate ledger key")
        if actual_reach4 is not None and (pd.isna(actual_reach4) or actual_reach4 not in (0,1)): raise ValueError("Invalid four-board outcome")
        ret = _number(trade_return)
        observed = trade_status=="SCENARIO_CLOSED" and ret is not None
        if actual_trade_profitable is not None and (not observed or bool(actual_trade_profitable)!=(ret>0)):
            raise ValueError("Profit flag requires a consistent closed scenario net_return")
        self.records.append(dict(signal_date=signal_date, code=code, name=name, legacy_rank=int(legacy_rank),
            legacy_score=float(legacy_score), final_selected=bool(final_selected), intervention_type=intervention_type,
            reason=reason, actual_reach4=actual_reach4, trade_status=trade_status or "NO_SCENARIO_RECORD",
            profit_observed=observed, actual_trade_profitable=(ret>0) if observed else None,
            trade_return=ret if observed else None, trade_net_pnl=_number(trade_net_pnl) if observed else None,
            recorded_at=now_utc_iso(), decision_clock="HISTORICAL_DIAGNOSTIC_NOT_FORWARD_PREDICTION"))
        self._keys.add(key)

    def summarize(self):
        df = pd.DataFrame(self.records)
        if df.empty: return {"total_decisions":0}
        excluded = df[df.intervention_type.str.startswith("EXCLUDE")]
        promoted = df[df.intervention_type.eq("PROMOTE_FROM_BELOW")]
        known = excluded[excluded.profit_observed]
        return dict(total_decisions=len(df), excluded_count=len(excluded),
            excluded_non_reach4=int(excluded.actual_reach4.eq(0).sum()),
            killed_four_board_paths=int(excluded.actual_reach4.eq(1).sum()),
            excluded_reach4_unknown=int(excluded.actual_reach4.isna().sum()),
            excluded_closed_scenario_count=len(known), avoided_negative_trades=int(known.trade_return.lt(0).sum()),
            killed_profitable_trades=int(known.trade_return.gt(0).sum()), break_even_excluded_trades=int(known.trade_return.eq(0).sum()),
            excluded_profit_unknown=len(excluded)-len(known),
            promoted_count=len(promoted), promoted_four_board_paths=int(promoted.actual_reach4.eq(1).sum()),
            promoted_reach4_unknown=int(promoted.actual_reach4.isna().sum()),
            promoted_profit_observed=int(promoted.profit_observed.sum()),
            promoted_profitable_trades=int((promoted.profit_observed & promoted.trade_return.gt(0)).sum()),
            promoted_negative_trades=int((promoted.profit_observed & promoted.trade_return.lt(0)).sum()),
            promoted_profit_unknown=int((~promoted.profit_observed).sum()),
            interpretation="路径与盈亏分别核算；原Top1情景不能证明替补股票、Top3策略或新组合的可成交收益",
            actual_trades=0, new_strategy_net_return=None)

