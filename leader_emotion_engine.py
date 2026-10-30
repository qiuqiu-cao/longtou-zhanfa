"""Computable leader emotion engine across market ladder, theme/name proxy, and relative competition tiers.

Strictly research and author review aid:
- Separate promotion rate, close-to-close return, and intraday price paths by board level.
- Multi-tier columns: Market Ladder, Bound Theme / Name Proxy, Stock Relative Competition.
- Propagate unknowns and missing members cleanly.
- Uncalibrated context: does not add arbitrary scores or mutate baseline rankings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

import historical_symbolic_context as symbolic


def fractional_rank(values: List[float], ascending: bool = True) -> List[Optional[float]]:
    """Compute normalized fractional rank in [0.0, 1.0] for complete sets; ties receive average rank."""
    n = len(values)
    res: List[Optional[float]] = [None] * n
    if n <= 1:
        return res
    if any(v is None or not np.isfinite(v) for v in values):
        return res
    valid = [(i, float(v)) for i, v in enumerate(values)]
    valid.sort(key=lambda x: x[1] if ascending else -x[1])
    i = 0
    while i < n:
        j = i
        while j < n and valid[j][1] == valid[i][1]:
            j += 1
        avg_rank = (i + (j - 1)) / 2.0
        norm_rank = float(avg_rank / (n - 1))
        for k in range(i, j):
            res[valid[k][0]] = norm_rank
        i = j
    return res


def compute_market_ladder_emotion(prior_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute ladder-specific feedback from yesterday's complete mainboard cohort."""
    tiers: Dict[int, List[Dict[str, Any]]] = {}
    for r in prior_rows:
        b = r.get("prior_boards")
        if b is not None:
            tiers.setdefault(int(b), []).append(r)

    market_ladder: Dict[str, Any] = {}
    for b in sorted(tiers):
        grp = tiers[b]
        den = len(grp)
        has_unknown_promotion = any(x.get("promoted") is None for x in grp)
        known_promoted = sum(1 for x in grp if x.get("promoted") == 1)
        known_failed = sum(1 for x in grp if x.get("promoted") == 0)
        unknown_promoted = sum(1 for x in grp if x.get("promoted") is None)

        # Precise promotion rate: None if any member has unknown outcome
        promotion_rate = round(float(known_promoted / den), 4) if (den > 0 and not has_unknown_promotion) else None
        promoted_lower_bound_rate = round(float(known_promoted / den), 4) if den > 0 else 0.0

        returns = [float(x["return_pct"]) for x in grp if x.get("return_pct") is not None and np.isfinite(x["return_pct"])]
        open_gaps = [float(x["open_gap_pct"]) for x in grp if x.get("open_gap_pct") is not None and np.isfinite(x["open_gap_pct"])]

        if returns:
            mean_pct = round(float(np.mean(returns)), 2)
            median_pct = round(float(np.median(returns)), 2)
            min_pct = round(float(np.min(returns)), 2)
            max_pct = round(float(np.max(returns)), 2)
            positive_close_count = sum(1 for r in returns if r > 0)
            positive_close_rate = round(float(positive_close_count / len(returns)), 4)
        else:
            mean_pct = None
            median_pct = None
            min_pct = None
            max_pct = None
            positive_close_count = None
            positive_close_rate = None

        # Severe loss (closing down <= -7.0%)
        severe_members = [
            {"code": x["code"], "name": x["name"], "return_pct": round(float(x["return_pct"]), 2)}
            for x in grp if x.get("return_pct") is not None and float(x["return_pct"]) <= -7.0
        ]
        # Intraday chase drawdown: open > 0 and close < open
        chase_losses = [
            {"code": x["code"], "name": x["name"], "open_gap_pct": round(float(x["open_gap_pct"]), 2), "return_pct": round(float(x["return_pct"]), 2)}
            for x in grp if x.get("open_gap_pct") is not None and x.get("return_pct") is not None
            and float(x["open_gap_pct"]) > 0 and float(x["return_pct"]) < float(x["open_gap_pct"])
        ]
        # Severe gap and dump: open gap >= 5.0% but closing down
        gap_and_dumps = [
            {"code": x["code"], "name": x["name"], "open_gap_pct": round(float(x["open_gap_pct"]), 2), "return_pct": round(float(x["return_pct"]), 2)}
            for x in grp if x.get("open_gap_pct") is not None and x.get("return_pct") is not None
            and float(x["open_gap_pct"]) >= 5.0 and float(x["return_pct"]) < 0.0
        ]

        ladder_key = f"tier_{b}_into_{b+1}"
        market_ladder[ladder_key] = {
            "prior_boards": b,
            "target_ladder": f"{b}进{b+1}",
            "denominator": den,
            "promoted_count": known_promoted,
            "failed_count": known_failed,
            "unknown_promotion_count": unknown_promoted,
            "promotion_rate": promotion_rate,
            "promoted_lower_bound_rate": promoted_lower_bound_rate,
            "return_distribution": {
                "mean_pct": mean_pct,
                "median_pct": median_pct,
                "min_pct": min_pct,
                "max_pct": max_pct,
                "observed_returns_count": len(returns),
                "positive_close_count": positive_close_count,
                "positive_close_rate": positive_close_rate
            },
            "intraday_price_path": {
                "observed_open_gaps_count": len(open_gaps),
                "mean_open_gap_pct": round(float(np.mean(open_gaps)), 2) if open_gaps else None,
                "chase_drawdown_count": len(chase_losses),
                "severe_loss_count": len(severe_members),
                "severe_loss_members": severe_members,
                "gap_and_dump_count": len(gap_and_dumps),
                "gap_and_dump_members": gap_and_dumps
            },
            "interpretation_caution": "未晋级不等于崩盘亏损，收红亦不等于追入盈利；区分收盘回报与开盘后持仓体验。"
        }
    return market_ladder


def compute_theme_name_proxy_feedback(row: Dict[str, Any], current_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract candidate name/phonetic proxy tokens and observe cluster response in current pool."""
    code = row["code"]
    name = row.get("name", "")
    row_tokens = sorted(symbolic.tokens(name)) if name else []

    if not row_tokens:
        return {
            "has_name_proxy": False,
            "tokens": [],
            "proxy_disclaimer": "无有效字辈/同音代理",
            "active_external_anchors": [],
            "low_tier_diffusion": [],
            "diffusion_count": 0,
            "damaged_peers": [],
            "damaged_count": 0
        }

    # Find external anchors (boards >= 3, code != candidate)
    token_set = set(row_tokens)
    anchors = []
    diffusion = []
    damaged = []
    for peer in current_rows:
        if peer["code"] == code:
            continue
        p_name = peer.get("name", "")
        p_tokens = set(symbolic.tokens(p_name)) if p_name else set()
        matched = sorted(token_set & p_tokens)
        if matched:
            raw_b = peer.get("boards")
            p_boards = int(raw_b) if raw_b is not None else 0
            p_kind = str(peer.get("kind", "limit_up"))
            item = {
                "code": peer["code"],
                "name": p_name,
                "boards": p_boards if raw_b is not None else None,
                "kind": p_kind,
                "matched_tokens": matched,
                "close": peer.get("close"),
                "turnover_pct": peer.get("turnover_pct")
            }
            # Only strictly sealed limit_up is active anchor or positive diffusion!
            is_successful_limit_up = (p_kind == "limit_up")
            if is_successful_limit_up and p_boards >= 3:
                anchors.append(item)
            elif is_successful_limit_up and p_boards in (1, 2):
                diffusion.append(item)
            else:
                damaged.append(item)

    return {
        "has_name_proxy": bool(row_tokens),
        "tokens": row_tokens,
        "proxy_type": "NAME_PHONETIC_PROXY",
        "proxy_disclaimer": "字辈/同音共现属于市场情绪代理，非经审计的主营业务题材，不据此认证资金因果或自动加分。",
        "active_external_anchors": anchors,
        "anchor_observed_count": len(anchors),
        "low_tier_diffusion": diffusion,
        "diffusion_count": len(diffusion),
        "damaged_peers": damaged,
        "damaged_count": len(damaged)
    }


def compute_stock_relative_competition(row: Dict[str, Any], current_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute candidate's standing within its board height tier or failed/down category."""
    code = row["code"]
    raw_b = row.get("boards")
    boards = int(raw_b) if raw_b is not None else None
    kind = row.get("kind", "limit_up")

    if boards is not None:
        same_tier = [r for r in current_rows if r.get("boards") == boards]
        tier_label = f"{boards}板组"
    else:
        same_tier = [r for r in current_rows if r.get("kind") == kind and r.get("boards") is None]
        tier_label = "炸板组" if kind in ("failed_board", "failed_limit_up") else ("跌停组" if kind == "limit_down" else f"{kind}组")

    # Rank by reported seal_first among tier
    valid_seal_first = []
    for r in same_tier:
        sf = r.get("seal_first")
        try:
            val = float(sf) if sf is not None else np.nan
        except (ValueError, TypeError):
            val = np.nan
        valid_seal_first.append(val)

    ranks = fractional_rank(valid_seal_first, ascending=True)
    my_idx = [r["code"] for r in same_tier].index(code)
    my_rank = ranks[my_idx]

    # Earliest flag
    if len(same_tier) <= 1 or any(np.isnan(v) for v in valid_seal_first):
        is_earliest = None
    else:
        min_seal = min(valid_seal_first)
        is_earliest = bool(valid_seal_first[my_idx] == min_seal)

    raw_open = row.get("reported_open_count")
    if raw_open is not None:
        try:
            open_cnt = int(str(raw_open).replace("次", "").strip())
        except (ValueError, TypeError):
            open_cnt = raw_open
    else:
        open_cnt = None

    one_price_val = row.get("one_price")

    return {
        "boards": boards,
        "tier_label": tier_label,
        "same_height_peer_count": len(same_tier),
        "seal_first": row.get("seal_first"),
        "seal_last": row.get("seal_last"),
        "seal_first_rank_in_tier": round(my_rank, 4) if my_rank is not None else None,
        "is_earliest_in_tier": is_earliest,
        "reported_open_count": open_cnt,
        "turnover_pct": row.get("turnover_pct"),
        "volume_ratio_previous_official": row.get("volume_ratio_previous_official"),
        "one_price": one_price_val,
        "cautions": "首封早与零开板为盘口稳定性观察，不等于全天无分歧或机构控盘；换手率增减反映筹码交换量，买卖动机未知。"
    }


def build_emotion_context(current_facts: Dict[str, Any], prior_feedback: Dict[str, Any]) -> Dict[str, Any]:
    """Construct complete multi-tier emotion context bundle."""
    cur_sig = current_facts.get("signal_date")
    prior_sig = prior_feedback.get("signal_date")
    if cur_sig and prior_sig and cur_sig != prior_sig:
        raise ValueError(f"Signal date mismatch: current facts ({cur_sig}) vs prior feedback ({prior_sig})")

    prior_date = prior_feedback.get("prior_date")
    if cur_sig and prior_date and str(prior_date) >= str(cur_sig):
        raise ValueError(f"Prior date ({prior_date}) must be strictly earlier than signal date ({cur_sig})")

    current_rows = current_facts.get("rows", [])
    prior_rows = prior_feedback.get("rows", [])

    # Check for duplicate codes
    seen_codes = set()
    for r in current_rows:
        c = r.get("code")
        if c in seen_codes:
            raise ValueError(f"Duplicate candidate code in current facts: {c}")
        seen_codes.add(c)

    seen_prior = set()
    for r in prior_rows:
        c = r.get("code")
        if c in seen_prior:
            raise ValueError(f"Duplicate code in prior cohort feedback: {c}")
        seen_prior.add(c)

    market_ladder = compute_market_ladder_emotion(prior_rows)

    candidate_cards: Dict[str, Any] = {}
    for r in current_rows:
        code = r["code"]
        raw_b = r.get("boards")
        boards = int(raw_b) if raw_b is not None else None
        tier_market_feedback = market_ladder.get(f"tier_{boards}_into_{boards+1}") if boards is not None else None

        candidate_cards[code] = {
            "code": code,
            "name": r.get("name", ""),
            "boards": boards,
            "kind": r.get("kind", "limit_up"),
            "market_ladder_feedback": tier_market_feedback,
            "theme_name_proxy": compute_theme_name_proxy_feedback(r, current_rows),
            "stock_competition": compute_stock_relative_competition(r, current_rows)
        }

    return {
        "version": "leader-emotion-engine-1",
        "signal_date": current_facts.get("signal_date"),
        "target_date": current_facts.get("target_date"),
        "current_pool_size": len(current_rows),
        "prior_cohort_size": len(prior_rows),
        "market_ladder_emotion": market_ladder,
        "candidate_emotion_cards": candidate_cards,
        "constraints": {
            "model_fits": 0,
            "orders": 0,
            "arbitrary_score_addition_prohibited": True,
            "uncalibrated_author_review_aid_only": True
        }
    }


def attach_emotion_to_facts(facts_path: Path | str, prior_feedback_path: Path | str, output_path: Path | str) -> Dict[str, Any]:
    """Attach computable emotion bundle to facts JSON for workbench projection and review."""
    facts = json.loads(Path(facts_path).read_text(encoding="utf-8"))
    prior_fb = json.loads(Path(prior_feedback_path).read_text(encoding="utf-8"))

    emotion_bundle = build_emotion_context(facts, prior_fb)
    facts["leader_emotion_context"] = emotion_bundle

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    return emotion_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-facts", required=True, help="Path to current signal date facts JSON")
    parser.add_argument("--prior-feedback", required=True, help="Path to prior cohort feedback JSON")
    parser.add_argument("--output", required=True, help="Path to output enriched facts JSON or emotion context JSON")
    args = parser.parse_args()

    res = attach_emotion_to_facts(args.current_facts, args.prior_feedback, args.output)
    print(json.dumps({
        "status": "LEADER_EMOTION_ENGINE_COMPUTATION_SUCCESS",
        "current_pool_size": res["current_pool_size"],
        "prior_cohort_size": res["prior_cohort_size"],
        "market_ladder_tiers": list(res["market_ladder_emotion"].keys())
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
