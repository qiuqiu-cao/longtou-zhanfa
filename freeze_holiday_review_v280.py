"""Freeze the first real holiday forecast from already available evidence.

No refitting, future quotes, probability calibration or trading authorization.
Each source is read and hashed at the real run time; old reports remain intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from next_session_expectation_contract_v271 import now, register, digest, validate_plan
from forecast_review_bridge_v277 import attach, review_digest
from emotion_three_regime_challenger_v86 import STATE_TO_CLASS

ROOT = Path(__file__).resolve().parent
VERSION = "v280-holiday-forward-1"
SIGNAL = "2026-09-30"
SNAPSHOT = ROOT / "data/review_20260930_eod_1911"
LAB = ROOT / "results/v278/september_falsification_20261002_v6"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mainboard(frame):
    return frame[frame.code.str.startswith(("60", "00"))
                 & ~frame.name.str.contains("ST|退", case=False, regex=True, na=False)].copy()


def band(stage, metric, low, high, scope, unit="ratio", hard_veto=False, basis=None):
    return dict(stage=stage, metric=metric, low=low, high=high, scope=scope,
                unit=unit, direction="HIGHER_BETTER" if metric != "auction_distance_from_0_5pct" else "LOWER_BETTER",
                required=True, hard_veto=hard_veto, basis=basis)


def build():
    # Read all files before assigning the knowledge timestamp.
    manifest = json.loads((SNAPSHOT / "manifest.json").read_text(encoding="utf-8"))
    pools = {}
    sources = []
    for name in ("limit_up", "failed_board", "limit_down"):
        item = manifest["datasets"][name]
        path = SNAPSHOT / item["path"]
        actual = sha(path)
        if actual != item["sha256"]:
            raise ValueError("snapshot source hash mismatch: " + name)
        frame = pd.read_csv(path, dtype={"代码": str}).rename(columns={"代码": "code", "名称": "name"})
        if len(frame) != item["rows"] or frame.code.duplicated().any():
            raise ValueError("snapshot row count or duplicate mismatch: " + name)
        pools[name] = mainboard(frame)
        sources.append(dict(path=str(path), sha256=actual, role="MARKET_SNAPSHOT"))
    relay_path = LAB / "relay_predictions_LABEL_ONLY.csv"
    market_path = LAB / "market_predictions_LABEL_ONLY.csv"
    relay = pd.read_csv(relay_path, dtype={"code": str})
    relay = relay[relay.date == SIGNAL].copy()
    market = pd.read_csv(market_path)
    market = market[market.date == SIGNAL]
    if len(market) != 1 or relay.code.duplicated().any():
        raise ValueError("one market row and unique candidate rows required")
    pool = pools["limit_up"]
    pool = pool[pool["连板数"] >= 2]
    expected = dict(zip(pool.code, pool["连板数"].astype(int)))
    if dict(zip(relay.code, relay.boards.astype(int))) != expected:
        raise ValueError("model denominator must exactly match the original closing relay pool")
    calendar_path = ROOT / "data/v279/exchange_calendar_audit_20261002_v1/calendar.json"
    calendar = json.loads(calendar_path.read_text(encoding="utf-8"))
    manual_path = ROOT / "results/daily_review_20260930_v1/review.md"
    manual_path.read_text(encoding="utf-8")
    for path in (relay_path, market_path, calendar_path, manual_path,
                 LAB / "report.json", ROOT / "leader_system_v3.py",
                 ROOT / "emotion_three_regime_challenger_v86.py",
                 ROOT / "next_session_expectation_contract_v271.py",
                 ROOT / "forecast_review_bridge_v277.py", Path(__file__)):
        sources.append(dict(path=str(path.resolve()), sha256=sha(path), role="RESEARCH_DERIVED"))
    knowledge = now().astimezone(ZoneInfo("Asia/Shanghai")).isoformat()
    max_board = max(expected.values())
    candidates = []
    for row in relay.sort_values("code").to_dict("records"):
        if row["actual_one_word_daily"] not in (0, 1):
            raise ValueError("daily one-word status unavailable")
        candidates.append(dict(code=row["code"], name=row["name"], boards=int(row["boards"]),
            identity="SPACE_ANCHOR" if row["boards"] == max_board else "CHALLENGER",
            actual_one_word_daily=bool(row["actual_one_word_daily"]),
            promotion_call="PROMOTE" if row["predicted"] == 1 else "FAIL",
            call_basis="V278_PRESEPTEMBER_SELECTED_BINARY_MODEL_NO_REFIT",
            identity_evidence=["Original closing pool confirms board count; unadjusted daily OHLC confirms one-word status"],
            identity_invalidation=["Theme leadership and real absorption/reseal are unverified; promotion alone cannot certify leader identity"],
            announcement_status="UNKNOWN", probability=None))
    positive = relay[relay.predicted == 1].sort_values(["probability", "code"], ascending=[False, True])
    identity_primary = str(positive.iloc[0].code) if len(positive) else None
    nonword = relay[relay.actual_one_word_daily == 0].sort_values(["probability", "code"], ascending=[False, True])
    nonword_positive = nonword[nonword.predicted == 1]
    nonword_primary = str(nonword_positive.iloc[0].code) if len(nonword_positive) else None
    scope = "PROVIDER_REPORTED_MAINBOARD_NAME_NON_ST_CLOSING_RELAY_POOL_NOT_EXCHANGE_CERTIFIED"
    summary = {"limit_up": len(pools["limit_up"]), "failed_board": len(pools["failed_board"]),
               "limit_down": len(pools["limit_down"]), "relay": len(candidates)}
    review = dict(version=VERSION, signal_date=SIGNAL, asof=knowledge, status="WATCH",
        decision_scope=scope, pool_summary=summary, relay_candidates=candidates,
        original_manual_review=str(manual_path), sources=sources,
        data_limitations=["Provider-reported denominator; official security-status coverage unverified",
            "September 30 previous-limit feedback covers 50/51; quality baseline is partial",
            "Holiday announcement coverage incomplete; October 8 auction and intraday data not yet available",
            "Research scores are not calibrated live probabilities"],
        original_manual_conditional_watch="000678",
        model_nonword_rank_first=str(nonword.iloc[0].code) if len(nonword) else None)
    baseline = float(market.iloc[0].emotion_quality_score)
    plan = dict(signal_date=SIGNAL, target_date="2026-10-08", calendar=calendar,
        evidence_cutoff=manifest["collected_at"], knowledge_cutoff=knowledge,
        model_version="V278_PRESEPTEMBER_FROZEN_MODEL_AND_V280_AUTHOR_CONTINGENCIES",
        input_hash=digest(sources), review_hash=review_digest(review),
        primary_candidate=None, highest_accept_price=None,
        identity_primary_candidate=identity_primary, non_one_word_primary_candidate=nonword_primary,
        denominator_status="verified_complete", denominator_scope=scope,
        denominator_note="Complete within the hashed provider closing pool; not certified exchange-wide completeness",
        candidates=candidates,
        market=dict(primary="RISK_OFF", direction=str(market.iloc[0].predicted), probability=None,
            primary_basis="Author scenario: old high-board negative feedback; new cycle not confirmed",
            direction_basis="V278 existing pre-September selected direction model, not refitted",
            calibration_status="NOT_CALIBRATED", baseline_quality=baseline, baseline_status="partial",
            supporting_evidence=["8 mainboard limit-down and 11 failed boards; previous high-board losses persist"],
            opposing_evidence=["48 mainboard closing limit-ups and 37 first boards leave room for repair"],
            invalidation=["Same-definition quality proxy improves by more than 0.05; classify direction IMPROVE",
                "Verified independent theme ladder and old high-board loss relief overturn the author risk-off scenario"]),
        settlement_definition=dict(direction="Target same-formula quality minus this frozen partial baseline; >0.05 IMPROVE, <-0.05 WORSEN, otherwise CONTINUE",
            baseline_cannot_be_revised=True, target_quality_requires_complete_inputs=True,
            market_primary_mapping=STATE_TO_CLASS,
            primary_classifier_source=str(ROOT / "leader_system_v3.py"),
            promotion="Exact next-session complete closing pool: candidate boards increase by exactly one",
            identity_selection_metric="Selected identity-pool candidate promotion; this does not measure actual leader identity",
            nonword_abstention_counts_as_no_call=True),
        bands=[
            band("AUCTION", "auction_distance_from_0_5pct", 0, 0, "STOCK:000678", basis="Original manual 0%-5% preferred open; distance=max(0,-gap,gap-0.05); either excessive or low gap misses"),
            band("AUCTION", "open_gap_at_most_7pct", 1, 1, "STOCK:000678", unit="binary", hard_veto=True, basis="Original manual default reject above 7%; frozen version uses strict veto without discretionary override"),
            band("AUCTION", "announcement_review_complete_and_no_hard_negative", 1, 1, "STOCK:000678", unit="binary", hard_veto=True, basis="All material announcements available before target auction reviewed; missing coverage is UNAVAILABLE, not zero"),
            band("OPEN_0935", "price_at_or_above_verified_session_vwap", 1, 1, "STOCK:000678", unit="binary", basis="Original manual VWAP confirmation; requires timestamped intraday turnover/amount, not daily proxies"),
            band("OPEN_1000", "verified_pullback_contract_reclaim_expand_reseal", 1, 1, "STOCK:000678", unit="binary", hard_veto=True, basis="Original manual intraday path; verified tick/minute audit required, otherwise UNAVAILABLE"),
            band("OPEN_1000", "no_two_failed_limit_touches", 1, 1, "STOCK:000678", unit="binary", hard_veto=True, basis="Original manual two failed touches cancel; exact path evidence required"),
            band("CLOSE", "frozen_baseline_quality_delta", -baseline, math.nextafter(-.05, -math.inf), "FROZEN_20260930_PARTIAL_QUALITY_BASELINE", basis="WORSEN model category translated to its strict -0.05 threshold; exact boundary belongs to CONTINUE for accuracy settlement")],
        evidence=[dict(kind="MARKET_SNAPSHOT", observed_at=SIGNAL+"T15:00:00+08:00",
            available_at=manifest["collected_at"], source=str(SNAPSHOT / "manifest.json"), status="delayed"),
            dict(kind="RESEARCH_DERIVED", observed_at=knowledge, available_at=knowledge,
                 source="Hash-bound source bundle in attached review", status="inferred")],
        conditional_watch_only="000678", automatic_entry_allowed=False,
        required_unscored_checks=["Verify theme linkage and both named sector supports before considering execution",
            "Verify Xinhua Media and old high-board feedback; no contagion confirmation from daily-price proxies"],
        announcement_coverage="UNKNOWN")
    validate_plan(plan, now().isoformat())
    return review, plan


def run(output, registry):
    output = Path(output)
    if output.exists():
        raise FileExistsError("refusing to overwrite holiday forward artifacts")
    review, plan = build()
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "plan_input.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    path, registered = register(plan, registry, now().isoformat())
    bound = attach(review, registered)
    if bound["next_session_expectation"]["status"] != "BOUND":
        raise ValueError("new holiday forecast binding failed")
    (output / "review.json").write_text(json.dumps(bound, ensure_ascii=False, indent=2), encoding="utf-8")
    report = dict(version=VERSION, registered_path=str(path.resolve()), registration_hash=registered["sha256"],
        recorded_at=registered["recorded_at"], binding=bound["next_session_expectation"],
        outcome_status="PENDING_EXACT_TARGET_SESSION", accuracy_claim_allowed=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--registry", default=str(ROOT / "data/v271/expectation_registry"))
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.registry), ensure_ascii=True))
