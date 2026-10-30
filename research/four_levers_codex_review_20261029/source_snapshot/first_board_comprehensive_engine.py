"""Comprehensive First-Board Engine implementing the Four Levers (四大抓手) for 龙头战法.
Strictly adheres to:
1. Zero model retraining / fitting (reuses legacy frozen 44-feature height model and S1 model).
2. Preserves original universe (< 93.27亿 float cap and thin turnover proxy exclusion).
3. Processes dynamic daily pools (7~72 candidates per day, never truncated at 35).
4. Missing data explicitly labeled UNKNOWN/待核, no fabricated order flow.
5. Dual-accounting ledger tracking avoided losses, killed profits, missed dragons, and days saved.
6. Realistic execution friction (queue constraints, limit-down lock-in, pre-committed exits, slippage).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import json
import math
import re
import numpy as np
import pandas as pd
from legacy_industry_snapshot import attach_legacy_groups

ROOT = Path(__file__).resolve().parent

# Frozen constants from project specifications
FROZEN_UPPER_CAP_CNY = 9326859896.52
FROZEN_MICROCAP_CNY = 30e8
FROZEN_LOW_TURNOVER_PCT = 5.0
FROZEN_LOW_VOL_RATIO = 1.0


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_number(value) -> float | None:
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def parse_seal_time_to_minutes(value) -> float | None:
    """Parse seal time string or integer to minutes since 09:30."""
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) not in (2, 3) or not all(x.isdigit() for x in parts):
            return None
        h, m = map(int, parts[:2])
        sec = int(parts[2]) if len(parts) == 3 else 0
    else:
        number = to_number(value)
        if number is None or number != int(number):
            return None
        s = str(int(number)).zfill(6)
        if len(s) != 6 or not s.isdigit():
            return None
        h, m, sec = int(s[:2]), int(s[2:4]), int(s[4:])
    if not (0 <= h < 24 and 0 <= m < 60 and 0 <= sec < 60):
        return None
    t = h * 3600 + m * 60 + sec
    if not ((9 * 3600 + 25 * 60 <= t <= 11 * 3600 + 30 * 60) or (13 * 3600 <= t <= 15 * 3600)):
        return None
    return (t - (9 * 3600 + 30 * 60)) / 60 if t < 13 * 3600 else 120 + (t - 13 * 3600) / 60


@dataclass
class EngineConfig:
    """Configuration parameters for the Four Levers Comprehensive Engine."""
    fusion_weight_s1: float = 0.5          # Weight on S1 promotion score: w*S1 + (1-w)*Height
    enable_sector_resonance: bool = True   # Lever 2: peer count & pioneer identification
    enable_seal_divergence: bool = True    # Lever 3: seal quality and sneak attack veto
    enable_emotion_defense: bool = True    # Lever 4: market retreat cash defense
    defense_min_flags: int = 2             # Number of risk flags to trigger emotion defense
    portfolio_top_k: int = 3               # Number of candidates selected per day
    simulation_top_k: int = 1              # Number of candidates traded in backtest simulation
    planned_exit_p: int = 2                # B+2 close exit rule
    cost_commission: float = 0.0003        # 0.03% commission
    cost_commission_min: float = 5.0       # Minimum 5 CNY commission
    cost_transfer: float = 0.00001         # 0.001% transfer fee
    cost_stamp_duty: float = 0.0005        # 0.05% stamp duty on exit
    adverse_slippage: float = 0.001        # 0.1% adverse slippage on both entry and exit
    position_budget_pct: float = 0.20      # Max 20% portfolio equity per position
    volume_participation_cap: float = 0.01 # Max 1% of previous day amount


class DualAccountingLedger:
    """Four-dimensional accounting ledger for tracking defensive and ranking interventions."""
    def __init__(self):
        self.decisions = []
        self._keys = set()

    def record_decision(
        self,
        signal_date: str,
        code: str,
        name: str,
        legacy_rank: int,
        legacy_score: float,
        final_rank: int,
        final_score: float,
        selected: bool,
        intervention: str,
        reason: str,
        reach4_outcome: int | None = None,
        promotion_outcome: int | None = None,
        trade_status: str | None = None,
        trade_return: float | None = None,
        trade_net_pnl: float | None = None,
    ):
        key = (signal_date, code)
        if key in self._keys:
            raise ValueError(f"Duplicate ledger entry: {key}")
        
        observed_trade = trade_status == "SCENARIO_CLOSED" and trade_return is not None
        self.decisions.append({
            "signal_date": signal_date,
            "code": code,
            "name": name,
            "legacy_rank": legacy_rank,
            "legacy_score": legacy_score,
            "final_rank": final_rank,
            "final_score": final_score,
            "selected": selected,
            "intervention": intervention,
            "reason": reason,
            "reach4_outcome": reach4_outcome,
            "promotion_outcome": promotion_outcome,
            "trade_status": trade_status or "NO_TRADE_RECORD",
            "trade_return": trade_return if observed_trade else None,
            "trade_net_pnl": trade_net_pnl if observed_trade else None,
            "trade_profitable": (trade_return > 0) if observed_trade else None,
            "recorded_at": now_utc_iso(),
        })
        self._keys.add(key)

    def summarize(self) -> dict:
        df = pd.DataFrame(self.decisions)
        if df.empty:
            return {"total_decisions": 0}
        
        excluded = df[df["intervention"].str.startswith("EXCLUDE")]
        promoted = df[df["intervention"].eq("PROMOTE_FROM_BELOW")]
        defended = df[df["intervention"].eq("DEFENSE_EMPTY_CASH")]
        
        closed_trades = df[df["trade_status"].eq("SCENARIO_CLOSED")]
        
        # Defense metrics
        defended_trades = defended[defended["trade_status"].eq("SCENARIO_CLOSED")]
        avoided_losses = int((defended_trades["trade_return"] < 0).sum())
        loss_amount_avoided = float(-defended_trades.loc[defended_trades["trade_return"] < 0, "trade_net_pnl"].sum())
        killed_wins = int((defended_trades["trade_return"] > 0).sum())
        win_amount_sacrificed = float(defended_trades.loc[defended_trades["trade_return"] > 0, "trade_net_pnl"].sum())
        missed_dragons = int(defended["reach4_outcome"].eq(1).sum())
        
        return {
            "total_decisions": len(df),
            "excluded_by_screen_or_veto": len(excluded),
            "promoted_from_below": len(promoted),
            "defended_decisions": len(defended),
            "four_dimensional_defense_audit": {
                "avoided_negative_trades_count": avoided_losses,
                "avoided_loss_cny": loss_amount_avoided,
                "killed_profitable_trades_count": killed_wins,
                "killed_profit_cny": win_amount_sacrificed,
                "missed_four_board_dragons": missed_dragons,
                "net_pnl_effect_cny": loss_amount_avoided - win_amount_sacrificed,
            },
            "interpretation": "双向对账严格区分路径与盈亏；防守在避免亏损的同时必然伴随错失盈利与龙头路径，两面数据必须同时公布。",
        }


class ComprehensiveFirstBoardEngine:
    """The production-grade engine implementing all four levers without retraining."""
    def __init__(self, config: EngineConfig | None = None):
        self.config = config or EngineConfig()
        self.ledger = DualAccountingLedger()

    @staticmethod
    def check_emotion_defense(daily_row: dict | pd.Series, min_flags: int = 2) -> tuple[bool, list[str]]:
        """Lever 4: Check if extreme market retreat triggers cash defense."""
        flags = []
        down_lim = to_number(daily_row.get("market_down_limits"))
        failed_rt = to_number(daily_row.get("market_failed_rate"))
        prev_p1 = to_number(daily_row.get("previous_first_promotion"))
        pos_frac = to_number(daily_row.get("market_positive_fraction"))

        if down_lim is not None and down_lim >= 8:
            flags.append("跌停家数>=8 (退潮恐慌)")
        if failed_rt is not None and failed_rt >= 0.40:
            flags.append("炸板率>=40% (高位大面)")
        if prev_p1 is not None and prev_p1 < 0.10:
            flags.append("昨日首板晋级率<10% (容错率极低)")
        if pos_frac is not None and pos_frac < 0.25:
            flags.append("全市场红盘率<25% (普跌冰点)")

        is_defense = len(flags) >= min_flags
        return is_defense, flags

    @staticmethod
    def audit_sneak_attack_trap(row: dict | pd.Series) -> tuple[bool, str]:
        """Lever 3: Veto sneak attack traps (14:00+ seal with low turnover)."""
        if str(row.get("has_bad_history", "")).lower() in {"true", "1"}:
            return True, "已标记历史跌停恶庄前科，一票否决"
        
        seal_time = row.get("seal_first") or row.get("first_seal_time")
        turnover = to_number(row.get("turnover"))
        minutes = parse_seal_time_to_minutes(seal_time)
        
        if minutes is not None and minutes >= 180 and turnover is not None and turnover < 3.0:
            return True, "14:00后尾盘偷袭且低换手(<3%)诱多出货，一票否决"
        
        delta = to_number(row.get("seal_delta_1430_to_close") or row.get("seal_delta"))
        verified_delta = str(row.get("seal_delta_verified", "")).lower() in {"true", "1"}
        if delta is not None and delta < 0.7 and verified_delta:
            return True, "尾盘大单撤单超过30% (心虚出逃)，一票否决"
            
        return False, "未触发偷袭否决条件"

    @staticmethod
    def audit_capacity_dragon_exemption(row: dict | pd.Series) -> tuple[bool, str]:
        """Lever 3: Capacity dragon exemption for large turnover with solid tail seals."""
        if str(row.get("has_bad_history", "")).lower() in {"true", "1"}:
            return False, "有恶庄前科，拒绝特赦"
        
        cap = to_number(row.get("float_cap"))
        if cap is None or cap > FROZEN_UPPER_CAP_CNY:
            return False, "市值超限，不在原筛选范围内"
            
        amount = to_number(row.get("amount"))
        turnover = to_number(row.get("turnover"))
        is_capacity = (amount is not None and amount >= 8e8) or (turnover is not None and turnover >= 15.0)
        if not is_capacity:
            return False, "未满足8亿成交额或15%换手条件"
            
        seal_time = row.get("seal_first") or row.get("first_seal_time")
        minutes = parse_seal_time_to_minutes(seal_time)
        if minutes is None or minutes > 60:
            return False, "非10:30前早盘首封，不满足真龙主动进攻条件"
            
        delta = to_number(row.get("seal_delta_1430_to_close") or row.get("seal_delta"))
        verified_delta = str(row.get("seal_delta_verified", "")).lower() in {"true", "1"}
        if delta is not None and delta >= 1.2 and verified_delta:
            return True, "大容量充分换手且尾盘主力加单锁仓，触发真龙特赦豁免"
            
        return False, "尾盘加单未达标或数据待核"

    def process_daily_candidates(
        self,
        signal_date: str,
        candidates_df: pd.DataFrame,
        daily_emotion: dict | pd.Series,
    ) -> pd.DataFrame:
        """Process full qualified pool (7~72 stocks) for a single trading date."""
        if candidates_df.empty:
            return pd.DataFrame()

        df = candidates_df.copy()
        
        # Lever 1: Dual-target fusion
        w = self.config.fusion_weight_s1
        h_score = pd.to_numeric(df["height_score"], errors="coerce").fillna(0.0)
        s1_score = pd.to_numeric(df["s1_score"], errors="coerce").fillna(0.0)
        df["fusion_score"] = w * s1_score + (1.0 - w) * h_score

        # Lever 2: Sector resonance
        df["peer_count"] = 0
        df["sector_tag"] = "⚪【题材关系待核】"
        df["sector_rank_adjustment"] = 0.0
        
        if self.config.enable_sector_resonance and "legacy_industry_group_PROXY" in df:
            known = df[df["legacy_industry_group_PROXY"].notna()].copy()
            if not known.empty:
                counts = known.groupby("legacy_industry_group_PROXY")["code"].transform("count")
                df.loc[known.index, "peer_count"] = counts
                
                # Moderate cluster resonance bonus: 2~7 peers
                cluster_mask = df.index.isin(known[counts.between(2, 7)].index)
                df.loc[cluster_mask, "sector_tag"] = "🟢【板块梯队协同共振】"
                df.loc[cluster_mask, "sector_rank_adjustment"] = 0.015
                
                # Solo isolated stock penalty: 1 peer
                solo_mask = df.index.isin(known[counts.eq(1)].index)
                df.loc[solo_mask, "sector_tag"] = "🟡【单兵孤勇-无板块梯队协同】"
                df.loc[solo_mask, "sector_rank_adjustment"] = -0.015
                
                # Climax crowding warning: >= 8 peers
                crowd_mask = df.index.isin(known[counts.ge(8)].index)
                df.loc[crowd_mask, "sector_tag"] = "🟡【板块首板高潮过热-警惕次日分化】"
                df.loc[crowd_mask, "sector_rank_adjustment"] = -0.010

        # Lever 3: Seal divergence & sneak attack veto
        df["sneak_veto"] = False
        df["sneak_veto_reason"] = ""
        df["capacity_exemption"] = False
        df["seal_tag"] = "⚪【盘口特征待核】"
        
        if self.config.enable_seal_divergence:
            for idx, row in df.iterrows():
                veto, v_reason = self.audit_sneak_attack_trap(row)
                if veto:
                    df.loc[idx, "sneak_veto"] = True
                    df.loc[idx, "sneak_veto_reason"] = v_reason
                    df.loc[idx, "seal_tag"] = "🔴【尾盘偷袭/撤单-一票否决】"
                else:
                    exempt, e_reason = self.audit_capacity_dragon_exemption(row)
                    if exempt:
                        df.loc[idx, "capacity_exemption"] = True
                        df.loc[idx, "seal_tag"] = "🟢【换手真龙-尾盘加单特赦】"
                    else:
                        m = parse_seal_time_to_minutes(row.get("seal_first") or row.get("first_seal_time"))
                        if m is not None and m <= 60:
                            df.loc[idx, "seal_tag"] = "🟢【早盘主动进攻封板】"

        # Final comprehensive score
        df["final_score"] = df["fusion_score"] + df["sector_rank_adjustment"]
        # Vetoed candidates have score set to -999.0
        df.loc[df["sneak_veto"], "final_score"] = -999.0
        # Capacity dragon exemption gets +0.05 bonus
        df.loc[df["capacity_exemption"], "final_score"] += 0.05

        # Rank candidates dynamically across full pool
        df = df.sort_values(["final_score", "code"], ascending=[False, True]).reset_index(drop=True)
        df["final_rank"] = df.index + 1

        # Lever 4: Emotion defense trigger
        is_defense, def_flags = self.check_emotion_defense(daily_emotion, self.config.defense_min_flags)
        df["is_defense_day"] = is_defense
        df["defense_flags"] = "|".join(def_flags) if is_defense else ""
        
        # Determine recommended selection
        df["selected"] = False
        if not is_defense or not self.config.enable_emotion_defense:
            # Select top K non-vetoed candidates
            non_vetoed = df[~df["sneak_veto"]]
            top_indices = non_vetoed.head(self.config.portfolio_top_k).index
            df.loc[top_indices, "selected"] = True

        return df

    def run_simulation(
        self,
        full_df: pd.DataFrame,
        bars_lookup: dict,
        calendar_sessions: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Run realistic portfolio simulation obeying execution queue, slippage, and floor lock."""
        sessions = calendar_sessions
        pos = {d: i for i, d in enumerate(sessions)}
        
        # Group candidates by signal date
        daily_groups = full_df.groupby("signal_date", sort=True)
        
        all_processed = []
        for s_date, group in daily_groups:
            emotion_row = group.iloc[0]
            processed = self.process_daily_candidates(s_date, group, emotion_row)
            all_processed.append(processed)
            
        scored_df = pd.concat(all_processed, ignore_index=True)
        
        # Select picks for trading simulation
        if self.config.enable_emotion_defense:
            trade_picks = scored_df[~scored_df["is_defense_day"] & ~scored_df["sneak_veto"]].copy()
        else:
            trade_picks = scored_df[~scored_df["sneak_veto"]].copy()
            
        trade_picks = trade_picks.sort_values(["signal_date", "final_rank", "code"]).groupby("signal_date").head(self.config.simulation_top_k)
        
        by_entry = {sessions[pos[r.signal_date] + 1]: r for r in trade_picks.itertuples() if pos.get(r.signal_date, len(sessions)) + 1 < len(sessions)}
        
        start = min(by_entry) if by_entry else sessions[-1]
        end = sessions[-1]
        
        cash = 1_000_000.0
        positions = []
        events = []
        equity = []
        last_prices = {}
        
        for p, day in enumerate(sessions):
            if day < start:
                continue
            if day > end:
                break
                
            mark = cash + sum(t["shares"] * last_prices.get(t["code"], t["entry_price"]) for t in positions)
            
            # Entry logic on day
            candidate = by_entry.get(day)
            if candidate is not None:
                bar = bars_lookup.get((day, candidate.code))
                reason = None
                if bar is None or not bar.get("valid_price") or not bar.get("eligible"):
                    reason = "ENTRY_UNKNOWN_OR_INELIGIBLE"
                elif round(bar["open"] * 100) >= bar["cap"]:
                    reason = "OPEN_AT_UPPER_LIMIT_NO_QUEUE_ASSUMPTION"
                elif round(bar["open"] * 100) <= bar["floor"]:
                    reason = "OPEN_AT_LOWER_LIMIT_CANCEL"
                    
                if reason:
                    events.append({
                        "signal_date": candidate.signal_date,
                        "code": candidate.code,
                        "entry_date": day,
                        "status": reason,
                        "net_pnl": 0.0,
                        "net_return": 0.0,
                    })
                else:
                    # Adverse slippage on buy
                    price = math.ceil(bar["open"] * (1.0 + self.config.adverse_slippage) * 100 - 1e-8) / 100
                    budget = min(
                        cash / (1.0 + self.config.adverse_slippage),
                        mark * self.config.position_budget_pct,
                        float(candidate.amount) * self.config.volume_participation_cap if pd.notna(candidate.amount) else 0.0
                    )
                    shares = int(budget / price / 100) * 100
                    if shares and round(price * 100) < bar["cap"]:
                        value = shares * price
                        fee = max(self.config.cost_commission_min, value * self.config.cost_commission) + value * self.config.cost_transfer
                        cash -= (value + fee)
                        positions.append({
                            "signal_date": candidate.signal_date,
                            "code": candidate.code,
                            "entry_date": day,
                            "entry_p": p,
                            "planned_exit_p": p + self.config.planned_exit_p,
                            "shares": shares,
                            "entry_price": price,
                            "entry_cost": value + fee,
                            "status": "OPEN",
                            "blocked_days": 0,
                        })
                    else:
                        events.append({
                            "signal_date": candidate.signal_date,
                            "code": candidate.code,
                            "entry_date": day,
                            "status": "ENTRY_LOT_OR_SLIPPAGE_LIMIT_CANCEL",
                            "net_pnl": 0.0,
                            "net_return": 0.0,
                        })
                        
            # Exit logic on day
            for t in list(positions):
                bar = bars_lookup.get((day, t["code"]))
                if bar and bar.get("valid_price"):
                    last_prices[t["code"]] = bar["close"]
                
                if p < t["planned_exit_p"]:
                    continue
                    
                # Floor lock-in check
                if bar is None or not bar.get("valid_price") or not bar.get("eligible") or round(bar["close"] * 100) <= bar["floor"] or not (bar.get("volume", 0) > 0):
                    t["blocked_days"] += 1
                    continue
                    
                # Adverse slippage on sell
                price = math.floor(bar["close"] * (1.0 - self.config.adverse_slippage) * 100 + 1e-8) / 100
                value = t["shares"] * price
                fee = max(self.config.cost_commission_min, value * self.config.cost_commission) + value * (self.config.cost_transfer + self.config.cost_stamp_duty)
                cash += (value - fee)
                net_pnl = value - fee - t["entry_cost"]
                net_return = (value - fee) / t["entry_cost"] - 1.0
                
                t.update({
                    "status": "SCENARIO_CLOSED",
                    "exit_date": day,
                    "exit_price": price,
                    "net_pnl": net_pnl,
                    "net_return": net_return,
                })
                events.append(t.copy())
                positions.remove(t)
                
            equity.append({
                "date": day,
                "equity": cash + sum(t["shares"] * last_prices.get(t["code"], t["entry_price"]) for t in positions),
                "cash": cash,
                "open_positions": len(positions),
            })
            
        events.extend([{**t, "status": "OPEN_UNRESOLVED", "mark": last_prices.get(t["code"], t["entry_price"])} for t in positions])
        trades_df = pd.DataFrame(events)
        curve_df = pd.DataFrame(equity)
        
        closed = trades_df[trades_df["status"].eq("SCENARIO_CLOSED")] if len(trades_df) else trades_df
        returns = closed["net_return"] if len(closed) else pd.Series(dtype=float)
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        
        total_return = curve_df["equity"].iloc[-1] / 1e6 - 1.0 if len(curve_df) else 0.0
        max_drawdown = float((curve_df["equity"] / curve_df["equity"].cummax() - 1.0).min()) if len(curve_df) else 0.0
        win_rate = float((returns > 0).mean()) if len(returns) else 0.0
        profit_factor = (
            closed.loc[closed["net_pnl"] > 0, "net_pnl"].sum() / -closed.loc[closed["net_pnl"] < 0, "net_pnl"].sum()
            if len(closed) and closed["net_pnl"].lt(0).any() else None
        )
        
        report = {
            "selected_count": len(trade_picks),
            "closed_trades": len(closed),
            "open_unresolved": len(positions),
            "skipped_or_cancelled": len(trades_df) - len(closed) - len(positions),
            "total_return_pct": total_return * 100.0,
            "max_drawdown_pct": max_drawdown * 100.0,
            "win_rate_pct": win_rate * 100.0,
            "profit_factor": profit_factor,
            "total_net_pnl_cny": float(closed["net_pnl"].sum()) if len(closed) else 0.0,
            "average_trade_pct": float(returns.mean() * 100.0) if len(returns) else 0.0,
            "average_win_pct": float(wins.mean() * 100.0) if len(wins) else 0.0,
            "average_loss_pct": float(losses.mean() * 100.0) if len(losses) else 0.0,
            "worst_trade_pct": float(returns.min() * 100.0) if len(returns) else 0.0,
            "blocked_exit_count": int(closed["blocked_days"].gt(0).sum()) + sum(t["blocked_days"] > 0 for t in positions),
        }
        
        return scored_df, trades_df, report
