"""Causal closing-time relay state machine shared by history and live review.

Rule roles are operational hypotheses, not independently annotated identities.
All features are built from an explicit signal-field allowlist. Outcome columns
are joined only by the research runner after this engine has finished.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from symbolic_name_theme_v217 import _name_tokens

VERSION = "causal-short-term-thinking-1"
ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "research/model_upgrade_validation/short_term_thinking_20261015T071000Z"
RECENCY = ROOT / "research/model_upgrade_validation/recency_walkforward_20261013T134351Z"
REGIMES = ("ICE_BREAKING", "DIVERGENCE_TEST", "CLIMAX_CONSENSUS", "EBB_HANDOVER", "BALANCED")
ROLES = ("SPACE_DRAGON", "CYCLE_CROSS_DRAGON", "SUPPLEMENT_DRAGON", "USURP_DRAGON", "TURNOVER_DRAGON", "THEME_LEADER", "THEME_SEED")
SIGNAL_FIELDS = ("date", "code", "name", "boards", "price", "turnover", "signal_amount", "amplitude",
    "signal_float_cap", "industry", "theme", "first_seal_minute", "last_seal_minute", "bomb_count")
MARKET_FIELDS = ("date", "max_board", "limit_count", "failed_rate", "down_limit_count", "yesterday_premium", "promotion_mid", "one_word_share")
DYNAMIC_FIELDS = [
    "cycle_height", "cycle_height_delta_5", "cycle_promotion", "cycle_bomb_rate", "cycle_down_count", "cycle_premium",
    "cycle_limit_rank20", "cycle_promotion_rank20", "is_ice_breaker_seed", "is_divergence_survivor", "climax_backrow_penalty",
    "active_parent_count", "parent_accelerating_count", "parent_heavy_count", "parent_break_count", "parent_collapsing_count",
    "parent_unobserved_count", "launch_lag_days", "supplement_coupling_score", "handover_on_anchor_break", "relation_channel_match",
    "same_height_success_count", "same_height_failed_count", "same_height_elimination_rate", "same_height_turnover_rank",
    "same_height_amount_rank", "same_height_amplitude_rank", "same_height_cap_rank", "sole_nonword_survivor",
    "verified_turnover_boards", "observed_nonword_boards", "streak_observed_fraction", "first_open_after_one_word",
    "daily_one_word", "daily_identity_observed", "relative_volume", "high_to_low_range", "theme_first_support",
    "theme_relay_support", "name_cluster_members", "isolated_volume_without_support", "parent_history_observed_fraction",
] + ["regime_" + k for k in REGIMES] + ["role_" + k for k in ROLES]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)


def verify(bindings):
    for path, digest in bindings.items():
        if sha(path) != digest:
            raise ValueError("source hash changed: " + path)


def finite(value):
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def cents(value):
    value = Decimal(str(value))
    if not value.is_finite() or value <= 0 or value * 100 != (value * 100).to_integral_value():
        raise ValueError("exact positive cents required")
    return int(value * 100)


def cap_cents(value):
    return int((Decimal(str(value)) * 110).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def price_state(bar, pool_price=None):
    if not bar:
        return dict(one_word=None, status="MISSING_DAILY_OHLC", range=None)
    try:
        op, hi, lo, cl = [cents(bar[k]) for k in ("open", "high", "low", "close")]
        if not lo <= min(op, cl) <= max(op, cl) <= hi:
            raise ValueError("invalid OHLC range")
        if pool_price is not None and (cl != cents(pool_price) or hi != cl):
            return dict(one_word=None, status="POOL_PRICE_OR_HIGH_CONFLICT", range=None)
        return dict(one_word=int(op == hi == lo == cl), status="OBSERVED_DAILY_PRICE_SHAPE",
            range=(hi - lo) / cl)
    except (ValueError, KeyError):
        return dict(one_word=None, status="INVALID_DAILY_PRICE", range=None)


def name_channels(left, right):
    a = {x for x, n in _name_tokens(left) if n == 1}
    b = {x for x, n in _name_tokens(right) if n == 1}
    shared = sorted(a & b)
    # This fixed dictionary encodes orthographic/phonetic candidates, not causal
    # links. A single weak initial or shared character never certifies a role.
    phonetic_groups = ("阳洋羊扬", "华花", "龙隆", "新鑫", "国果", "生升盛", "德得")
    phonetic = [g for g in phonetic_groups if any(c in a for c in g) and any(c in b for c in g)]
    return shared, phonetic


def classify_cycle(row, past):
    height = finite(row.get("max_board"))
    promotion = finite(row.get("promotion_mid"))
    bomb = finite(row.get("failed_rate"))
    down = finite(row.get("down_limit_count"))
    premium = finite(row.get("yesterday_premium"))
    limit = finite(row.get("limit_count"))
    prior = past[-1] if past else {}
    def rank(field, value):
        values = [finite(x.get(field)) for x in past[-20:]]
        values = [v for v in values if v is not None]
        return sum(v <= value for v in values) / len(values) if value is not None and values else None
    lr, pr = rank("limit_count", limit), rank("promotion_mid", promotion)
    previous_height = finite(prior.get("max_board"))
    expanding_down = down is not None and finite(prior.get("down_limit_count")) is not None and down > prior["down_limit_count"]
    two_cold = len(past) >= 2 and all(finite(r.get("promotion_mid")) is not None and r["promotion_mid"] < .25 for r in past[-2:])
    warming = two_cold and promotion is not None and promotion > past[-1]["promotion_mid"]
    if premium is not None and premium < 0 and ((height is not None and previous_height is not None and height < previous_height) or expanding_down):
        regime = "EBB_HANDOVER"
    elif (height is not None and height <= 3) or warming:
        regime = "ICE_BREAKING"
    elif (bomb is not None and bomb > .30) or (limit is not None and finite(prior.get("limit_count")) is not None and limit < prior["limit_count"]):
        regime = "DIVERGENCE_TEST"
    elif lr is not None and pr is not None and lr >= .80 and pr >= .80 and finite(row.get("one_word_share")) is not None and row["one_word_share"] >= .30:
        regime = "CLIMAX_CONSENSUS"
    else:
        regime = "BALANCED"
    return regime, lr, pr


def causal_features(candidates, market, attempts, bars, sessions, *, narratives=()):
    """No targets accepted: a pure chronological state machine.

    Missing sessions never become broken boards, missing OHLC never becomes a
    non-word board, and rolling percentiles never include a later session.
    """
    if candidates.duplicated(["date", "code"]).any() or market.duplicated("date").any():
        raise ValueError("duplicate dated input")
    if list(sessions) != sorted(set(sessions)):
        raise ValueError("official calendar must be sorted and unique")
    c = candidates[[k for k in SIGNAL_FIELDS if k in candidates]].copy()
    for k in SIGNAL_FIELDS:
        if k not in c:
            c[k] = np.nan
    c["code"] = c.code.astype(str).str.zfill(6)
    c = c.sort_values(["date", "code"])
    idx = {d: i for i, d in enumerate(sessions)}
    if any(d not in idx for d in c.date):
        raise ValueError("candidate outside official trading calendar")
    m = {r["date"]: r for r in market[[k for k in MARKET_FIELDS if k in market]].to_dict("records")}
    pools = defaultdict(dict)
    failed = defaultdict(Counter)
    attempt_dates = set()
    for r in attempts.to_dict("records"):
        day = r.get("date", r.get("signal_date"))
        if day not in idx:
            continue
        attempt_dates.add(day)
        if r.get("pool_assignment") == "CLOSING_LIMIT_UP":
            pools[day][r["code"]] = dict(code=r["code"], name=r.get("name", ""), boards=int(r["board_height"]),
                price=r.get("close"), industry=r.get("industry", ""), theme=r.get("theme", ""))
        elif r.get("pool_assignment") == "FAILED_BOARD_OBSERVATION":
            failed[day][int(r["board_height"])] += 1
    for r in c.to_dict("records"):
        pools[r["date"]][r["code"]] = r
    bar_index = {(r["date"], r["code"]): r for r in bars.to_dict("records")}
    byday = {d: f.to_dict("records") for d, f in c.groupby("date", sort=True)}
    past, rows, parent_edges, last_high, supplement_memory = [], [], [], {}, {}
    all_days = [d for d in sessions if min(pools) <= d <= max(c.date)]
    for day in all_days:
        t = idx[day]
        current = pools.get(day, {})
        day_market = dict(m.get(day, {}), date=day)
        if finite(day_market.get("max_board")) is None and current:
            day_market["max_board"] = max(r["boards"] for r in current.values())
        regime, lr, pr = classify_cycle(day_market, past)
        group = byday.get(day, [])
        high = max([r["boards"] for r in group], default=0)
        parent_pool = {code: r for code, r in last_high.items() if 0 < t - idx[r["date"]] <= 5}
        parent_states = {}
        previous_day = sessions[t - 1] if t else None
        for code, anchor in parent_pool.items():
            bar, prev = bar_index.get((day, code)), bar_index.get((previous_day, code))
            p = current.get(code)
            prev_p = pools.get(previous_day, {}).get(code)
            pshape = price_state(bar, p.get("price") if p else None)
            ratio = None
            if bar and prev and finite(bar.get("volume")) is not None and finite(prev.get("volume")) and bar["volume"] > 0:
                ratio = bar["volume"] / prev["volume"]
            broken = False
            if bar and prev:
                try:
                    broken = cents(bar["close"]) < cap_cents(prev["close"])
                except ValueError:
                    pass
            if p and p["boards"] >= anchor["boards"] and p["boards"] >= 4:
                state = "ANCHOR_ACCELERATING" if pshape["one_word"] == 1 or (ratio is not None and ratio <= 1) else "ANCHOR_HEAVY_TURNOVER" if ratio is not None and ratio >= 1.5 else "ANCHOR_CONTINUING"
            elif broken and prev_p and prev_p["boards"] >= 4:
                state = "ANCHOR_FIRST_BROKEN"
            elif broken and prev and bar["close"] / prev["close"] - 1 <= -.07:
                state = "ANCHOR_COLLAPSING"
            elif broken:
                state = "ANCHOR_POST_BREAK"
            else:
                state = "ANCHOR_UNOBSERVED"
            parent_states[code] = (anchor, state, ratio)
        shape_by_code = {r["code"]: price_state(bar_index.get((day, r["code"])), r["price"]) for r in group}
        for stock in group:
            code, n = stock["code"], int(stock["boards"])
            peers = [r for r in group if r["boards"] == n]
            sh = shape_by_code[code]
            def rank(field, ascending=True):
                vals = pd.Series({r["code"]: finite(r.get(field)) for r in peers}, dtype=float)
                return finite(vals.rank(pct=True, method="average", ascending=ascending).get(code))
            start = max(0, t - n + 1)
            streak_dates = sessions[start:t + 1]
            observed, nonword, verified, prices = 0, 0, 0, []
            for sd in streak_dates:
                b = bar_index.get((sd, code))
                p = pools.get(sd, {}).get(code)
                s = price_state(b, p.get("price") if p else None)
                # Require an explicitly observed closing board at each node.
                if p and s["one_word"] is not None:
                    observed += 1
                    prices.append((sd, s["one_word"]))
                    nonword += s["one_word"] == 0
                    if s["one_word"] == 0 and b and (cents(b["open"]) != cents(b["low"]) or s["range"] > .03):
                        verified += 1
            first_open = None
            if len(prices) == n:
                first_open = int(n > 1 and all(v == 1 for _, v in prices[:-1]) and prices[-1][1] == 0)
            elif sh["one_word"] == 1:
                first_open = 0
            bar, prevbar = bar_index.get((day, code)), bar_index.get((previous_day, code))
            volume_ratio = (bar["volume"] / prevbar["volume"] if bar and prevbar and finite(prevbar.get("volume")) and finite(bar.get("volume")) is not None else None)
            current_themes = set()
            for key in ("industry", "theme"):
                value = stock.get(key)
                if isinstance(value, str) and value.strip() and value.lower() not in ("unknown", "unavailable"):
                    current_themes.add(value.strip())
            theme_peers = [r for r in current.values() if any(isinstance(r.get(k), str) and r.get(k) in current_themes for k in ("industry", "theme"))]
            first_support = sum(r["boards"] == 1 for r in theme_peers) if current_themes else None
            relay_support = sum(r["boards"] >= 2 for r in theme_peers) if current_themes else None
            tokens = {token for token, size in _name_tokens(stock["name"]) if size == 1}
            name_members = max([sum(any(x == token for x, size in _name_tokens(r["name"]) if size == 1) for r in current.values()) for token in tokens], default=1)
            hypotheses = []
            for anchor_code, (anchor, state, ratio) in parent_states.items():
                if anchor_code == code:
                    continue
                parent_start = idx[anchor["date"]] - int(anchor["boards"]) + 1
                lag = start - parent_start
                shared, phonetic = name_channels(stock["name"], anchor["name"])
                theme_match = bool(current_themes & {str(anchor.get("industry", "")), str(anchor.get("theme", ""))})
                narrative_ids = [r["id"] for r in narratives if r.get("child_code") == code and r.get("anchor_code") == anchor_code and r.get("published_date", "9999") <= day and r.get("available_date", "9999") <= day]
                style = finite(stock.get("price")) is not None and finite(anchor.get("price")) and .5 <= stock["price"] / anchor["price"] <= 2
                relation_count = int(theme_match) + int(bool(shared or phonetic)) + int(bool(style)) + int(bool(narrative_ids))
                handover = state in ("ANCHOR_FIRST_BROKEN", "ANCHOR_HEAVY_TURNOVER") and 2 <= n <= 4 and 2 <= lag <= 6
                # A lone word/initial or similar price is a hypothesis only.
                substantive_relation = theme_match or bool(narrative_ids) or (bool(phonetic) and bool(style)) or len(shared) >= 2
                match = 2 <= lag <= 6 and substantive_relation
                memory_key = (code, start, anchor_code)
                if match and handover:
                    supplement_memory[memory_key] = day
                previous_handover = supplement_memory.get(memory_key)
                continuing_supplement = previous_handover is not None and t - idx[previous_handover] <= 5 and state not in ("ANCHOR_COLLAPSING", "ANCHOR_UNOBSERVED")
                score = (int(2 <= lag <= 6) + int(handover) + int(theme_match) + int(bool(shared or phonetic)) + int(bool(style)) + int(bool(narrative_ids))) / 6
                edge = dict(date=day, code=code, anchor_code=anchor_code, anchor_name=anchor["name"], parent_observation_date=anchor["date"],
                    anchor_boards=anchor["boards"], parent_anchor_state=state, launch_lag_days=lag, shared_characters=shared,
                    phonetic_families=phonetic, industry_proxy_match=theme_match, price_style_match=bool(style), narrative_ids=narrative_ids,
                    relation_channel_match=relation_count, handover_on_anchor_break=int(handover), supplement_coupling_score=score,
                    rule_supplement_candidate=bool(match and (handover or continuing_supplement)), prior_handover_day=previous_handover, independent_identity_verified=False,
                    timing_basis="ARCHIVED_REPORTED_HEIGHT_OFFICIAL_SESSION_OFFSET", counterevidence="名称/价格近似非资金因果；需独立题材及连续板核验。")
                hypotheses.append(edge)
                parent_edges.append(edge)
            compatible = [h for h in hypotheses if 2 <= h["launch_lag_days"] <= 6]
            best = sorted(compatible, key=lambda x: (-int(x["rule_supplement_candidate"]), -x["supplement_coupling_score"], x["anchor_code"]))[0] if compatible else None
            fc = failed[day][n] if day in attempt_dates else None
            elimination = fc / (fc + len(peers)) if fc is not None and fc + len(peers) else None
            amount_rank, turn_rank = rank("signal_amount"), rank("turnover")
            only_nonword = int(sh["one_word"] == 0 and sum(shape_by_code[r["code"]]["one_word"] == 0 for r in peers) == 1) if all(shape_by_code[r["code"]]["one_word"] is not None for r in peers) else None
            role_flags = dict.fromkeys(ROLES, False)
            role_flags["SPACE_DRAGON"] = n == high and high > 0
            role_flags["CYCLE_CROSS_DRAGON"] = n >= 3 and regime in ("ICE_BREAKING", "DIVERGENCE_TEST", "EBB_HANDOVER")
            role_flags["SUPPLEMENT_DRAGON"] = any(h["rule_supplement_candidate"] for h in hypotheses)
            role_flags["USURP_DRAGON"] = sh["one_word"] == 0 and elimination is not None and elimination >= .5 and len(peers) <= 2
            nonword_height = max([r["boards"] for r in group if shape_by_code[r["code"]]["one_word"] == 0], default=0)
            role_flags["TURNOVER_DRAGON"] = n >= 3 and verified >= 1 and sh["one_word"] == 0 and (n == nonword_height or (amount_rank is not None and amount_rank >= .8))
            theme_high = max([r["boards"] for r in theme_peers], default=0)
            role_flags["THEME_LEADER"] = bool(current_themes and len(theme_peers) >= 2 and first_support and n >= 3 and n == theme_high)
            role_flags["THEME_SEED"] = bool(current_themes and len(theme_peers) >= 2 and first_support and n == 2 and n == theme_high)
            active_roles = [role for role in ROLES if role_flags[role]]
            priority = ("SPACE_DRAGON", "SUPPLEMENT_DRAGON", "USURP_DRAGON", "TURNOVER_DRAGON", "CYCLE_CROSS_DRAGON", "THEME_LEADER", "THEME_SEED")
            primary = next((role for role in priority if role_flags[role]), "RELAY_CONTENDER")
            counts = Counter(x[1] for k, x in parent_states.items() if k != code)
            late = finite(stock.get("last_seal_minute"))
            out = dict(date=day, code=code, name=stock["name"], boards=n, cycle_regime=regime, primary_role=primary,
                roles=active_roles, role_status="CAUSAL_OPERATIONAL_RULE_LABEL_NOT_INDEPENDENT_TRUTH", daily_identity_status=sh["status"],
                parent_anchor_state=best["parent_anchor_state"] if best else "NO_MATCHED_PARENT" if parent_pool else "NO_ACTIVE_PARENT_IN_ARCHIVE",
                parent_anchor_code=best["anchor_code"] if best else None,
                parent_candidates=[h["anchor_code"] for h in sorted(compatible, key=lambda h: (-h["supplement_coupling_score"], h["anchor_code"]))],
                role_identity_accuracy=None, feature_cutoff=day, full_market_pit_verified=False,
                cycle_height=day_market.get("max_board"), cycle_height_delta_5=(finite(day_market.get("max_board")) - finite(past[-5].get("max_board")) if len(past) >= 5 and finite(day_market.get("max_board")) is not None and finite(past[-5].get("max_board")) is not None else None),
                cycle_promotion=day_market.get("promotion_mid"), cycle_bomb_rate=day_market.get("failed_rate"), cycle_down_count=day_market.get("down_limit_count"),
                cycle_premium=day_market.get("yesterday_premium"), cycle_limit_rank20=lr, cycle_promotion_rank20=pr,
                is_ice_breaker_seed=int(regime == "ICE_BREAKING" and n == high and 2 <= n <= 3),
                is_divergence_survivor=int(regime == "DIVERGENCE_TEST" and sh["one_word"] == 0 and elimination is not None and elimination >= .5),
                climax_backrow_penalty=(int(n < high or late > 180) if late is not None else int(n < high)) if regime == "CLIMAX_CONSENSUS" else 0,
                active_parent_count=len([p for p in parent_pool if p != code]), parent_accelerating_count=counts["ANCHOR_ACCELERATING"], parent_heavy_count=counts["ANCHOR_HEAVY_TURNOVER"],
                parent_break_count=counts["ANCHOR_FIRST_BROKEN"], parent_collapsing_count=counts["ANCHOR_COLLAPSING"], parent_unobserved_count=counts["ANCHOR_UNOBSERVED"],
                launch_lag_days=best["launch_lag_days"] if best else None, supplement_coupling_score=best["supplement_coupling_score"] if best else 0,
                handover_on_anchor_break=int(any(h["handover_on_anchor_break"] for h in compatible)), relation_channel_match=best["relation_channel_match"] if best else 0,
                same_height_success_count=len(peers), same_height_failed_count=fc, same_height_elimination_rate=elimination,
                same_height_turnover_rank=turn_rank, same_height_amount_rank=amount_rank, same_height_amplitude_rank=rank("amplitude", False),
                same_height_cap_rank=rank("signal_float_cap"), sole_nonword_survivor=only_nonword,
                verified_turnover_boards=verified if observed else None, observed_nonword_boards=nonword if observed else None,
                streak_observed_fraction=observed / n, first_open_after_one_word=first_open, daily_one_word=sh["one_word"], daily_identity_observed=int(sh["one_word"] is not None),
                relative_volume=volume_ratio, high_to_low_range=sh["range"], theme_first_support=first_support, theme_relay_support=relay_support,
                name_cluster_members=name_members, isolated_volume_without_support=int(not any(role_flags[k] for k in ("SPACE_DRAGON", "SUPPLEMENT_DRAGON", "THEME_LEADER", "THEME_SEED"))) if current_themes else None,
                parent_history_observed_fraction=sum(d in pools for d in sessions[max(0, t - 5):t]) / min(5, t) if t else 0)
            out.update({"regime_" + k: int(regime == k) for k in REGIMES})
            out.update({"role_" + k: int(v) for k, v in role_flags.items()})
            rows.append(out)
        for code, p in current.items():
            if p["boards"] >= 4:
                last_high[code] = {**p, "date": day}
        # Preserve official empty sessions so a missing predecessor is never
        # silently replaced with the last observed market row.
        past.append(day_market)
    return pd.DataFrame(rows), parent_edges


def exact_attempts(frame, sessions=None):
    """Recheck V215's 1.1-cent tolerance without modifying its original file."""
    rows = frame.copy()
    rows["code"] = rows.code.astype(str).str.zfill(6)
    high = np.rint(pd.to_numeric(rows.high) * 100).astype(int)
    close = np.rint(pd.to_numeric(rows.close) * 100).astype(int)
    cap = np.rint(pd.to_numeric(rows.limit_price_10) * 100).astype(int)
    exact_close, touch = close.eq(cap), high.ge(cap)
    valid = touch & high.le(cap) & close.le(cap)
    rows["original_pool_assignment"] = rows.pool_assignment
    rows["pool_assignment"] = np.where(exact_close, "CLOSING_LIMIT_UP", "FAILED_BOARD_OBSERVATION")
    # Heights of a recategorized closing row are not reliable failed attempts.
    invalid_changed = rows.pool_assignment.ne(rows.original_pool_assignment)
    rows = rows[valid & ~invalid_changed].copy()
    if sessions is not None:
        index = {d:i for i,d in enumerate(sessions)}
        previous, heights = {}, {}
        for r in rows.sort_values(["signal_date", "code"]).itertuples():
            p = previous.get(r.code)
            height = p[1]+1 if p and p[0] == index[r.signal_date]-1 else 1
            heights[r.Index] = height
            previous[r.code] = (index[r.signal_date], height) if r.pool_assignment == "CLOSING_LIMIT_UP" else (index[r.signal_date], 0)
        rows["original_board_height"] = rows.board_height
        rows["board_height"] = pd.Series(heights)
    return rows, dict(original=len(frame), exact_kept=int((valid & ~invalid_changed).sum()),
        exact_price_or_assignment_excluded=int((~valid | invalid_changed).sum()), scope="CURRENT_UNIVERSE_RECONSTRUCTION_NOT_FULL_MARKET_PIT")


def load_existing():
    p = read(RECENCY / "protocol_BEFORE_ANY_FIT.json")
    if sha(p["prepared_path"]) != p["prepared_sha256"]:
        raise ValueError("historical prepared data changed")
    prepared = joblib.load(p["prepared_path"])
    attempts_path = ROOT / "results/v215/historical_board_attempts_20260920_v2/historical_closing_and_failed_board_attempts.csv"
    v215_report = ROOT / "results/v215/historical_board_attempts_20260920_v2/report.json"
    if sha(attempts_path) != read(v215_report)["output_sha256"]:
        raise ValueError("V215 original byte hash changed")
    attempts, audit = exact_attempts(pd.read_csv(attempts_path, dtype={"code": str}), read(p["calendar"])["sessions"])
    columns = ["date", "code", "name", "open", "high", "low", "close", "pre_close", "volume", "amount", "turnover", "float_cap", "industry", "theme"]
    bar_path = ROOT / "data/v3/imported/daily-bars.csv"
    bars = pd.read_csv(bar_path, usecols=columns, dtype={"code": str})
    used = [Path(p["prepared_path"]), RECENCY / "protocol_BEFORE_ANY_FIT.json", Path(p["calendar"]), attempts_path, v215_report, bar_path]
    # Raw-qualified and normalized historical signal OHLC are kept as distinct
    # source tiers. The user now authorizes their retrospective feature use.
    context_path = ROOT / "research/model_upgrade_validation/early_signal_ohlc_20261013T1303Z/complete9245_provisional_price_context_DELIVERY.json"
    context = read(context_path)
    signal_bars, tiers = [], {}
    for item in context["rows"]:
        sources = item["sources"]
        if not sources:
            continue
        values = [tuple(finite(s.get(k)) for k in ("open", "high", "low", "close")) for s in sources]
        if len(set(values)) != 1:
            continue
        signal_bars.append(dict(date=item["date"], code=item["code"], **dict(zip(("open", "high", "low", "close"), values[0]))))
        tiers[(item["date"], item["code"])] = item.get("identity_source_tier", "UNCLASSIFIED_ARCHIVE")
    # Update only prices at the exact historical signal key; retain original
    # volumes and other known fields. No forward/back filling across dates.
    bi = bars.set_index(["date", "code"])
    updates = pd.DataFrame(signal_bars).set_index(["date", "code"])
    bi = bi.reindex(bi.index.union(updates.index))
    bi.loc[updates.index, ["open", "high", "low", "close"]] = updates[["open", "high", "low", "close"]]
    bars = bi.reset_index()
    used.append(context_path)
    return prepared, attempts, bars, read(p["calendar"])["sessions"], tiers, used, audit


def build(output_dir=DEFAULT_RUN):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "dataset.joblib").exists():
        raise FileExistsError("immutable dataset already exists")
    prepared, attempts, bars, sessions, tiers, used, audit = load_existing()
    features, edges = causal_features(prepared["relay"], prepared["market"], attempts, bars, sessions)
    features["price_identity_source_tier"] = [tiers.get((r.date, r.code), "MISSING") for r in features.itertuples()]
    original_keys = set(zip(prepared["relay"].date, prepared["relay"].code))
    if len(features) != len(prepared["relay"]) or original_keys != set(zip(features.date, features.code)):
        raise ValueError("all original candidates must be retained")
    if features.primary_role.isna().any() or features.primary_role.eq("UNKNOWN").any():
        raise ValueError("a rule decision is required for each candidate")
    sources = {str(p.resolve()): sha(p) for p in used + [Path(__file__), ROOT / "symbolic_name_theme_v217.py"]}
    features.to_csv(out / "complete9245_causal_features_and_roles.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(edges).to_csv(out / "all_parent_anchor_hypotheses.csv", index=False, encoding="utf-8-sig")
    joblib.dump(dict(version=VERSION, features=features, prepared=prepared, attempts=attempts, bars=bars,
        sessions=sessions, sources=sources, historical_mode="RETROSPECTIVE_CAUSAL_RECONSTRUCTION_NOT_ORIGINAL_PIT"), out / "dataset.joblib", compress=3)
    import model_upgrade_history as history
    coverage = {}
    for fold in history.FOLDS:
        for kind, base in (("training", history.training_rows(prepared["relay"], fold["start"], "relay")),
                           ("evaluation", prepared["relay"][prepared["relay"].date.between(fold["start"], fold["end"])])):
            f = features.merge(base[["date", "code"]], on=["date", "code"], validate="one_to_one")
            coverage[fold["id"] + "_" + kind] = dict(rows=len(f), rule_decision_coverage=float(f.primary_role.notna().mean()),
                daily_price_identity_coverage=float(f.daily_identity_observed.mean()),
                full_streak_price_coverage=float(f.streak_observed_fraction.eq(1).mean()),
                pk_elimination_coverage=float(f.same_height_elimination_rate.notna().mean()),
                theme_industry_proxy_coverage=float(f.theme_relay_support.notna().mean()),
                matched_parent_rows=int(f.parent_anchor_code.notna().sum()),
                field_observation_coverage={k: float(f[k].notna().mean()) for k in DYNAMIC_FIELDS})
    report = dict(version=VERSION, formed_at=now(), full_original_candidates=len(features), dates=features.date.nunique(),
        original_targets_untouched=True, rule_decision_coverage=float(features.primary_role.notna().mean()),
        primary_role_distribution=features.primary_role.value_counts().to_dict(), multi_role_distribution={k: int(features["role_" + k].sum()) for k in ROLES},
        regime_distribution=features.cycle_regime.value_counts().to_dict(), windows=coverage, exact_v215_audit=audit,
        parent_edge_rows=len(edges), independent_role_truth_rows=0, identity_recognition_accuracy=None,
        causal_time_mode="EVENT_DATE_LE_SIGNAL_ONLY; ACTUAL_RECEIPTS_NOT_BACKDATED; ARCHIVE_RESEARCH_AUTHORIZED_BY_CURRENT_USER",
        missing_policy="No feature fabrication. 100% row/state decisions is separate from each field's factual coverage.",
        limitations=["Current-universe survivorship bias and historic non-ST status remain", "OHLC implies price range, not verified order-flow absorption or intraday reseal order", "Historical industry/theme columns may be entirely missing", "Rules describe state; independent six-role accuracy is not measurable from self-labels"],
        source_bindings=sources, dataset_path=str((out / "dataset.joblib").resolve()), dataset_sha256=sha(out / "dataset.joblib"),
        numeric_fits=0, new_formal_predictions=0, new_forward_matured=0)
    write_new(out / "coverage_and_role_distribution.json", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(DEFAULT_RUN))
    a = p.parse_args(argv)
    r = build(a.output_dir)
    print(json.dumps({k: r[k] for k in ("full_original_candidates", "dates", "rule_decision_coverage", "primary_role_distribution", "exact_v215_audit")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
