"""Audit saved rolling scores and original screening, with no training or orders."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import first_board_execution_upgrade as up

ROOT = Path(__file__).resolve().parent
OLD = ROOT / "research/first_board_leader_20261020T114600Z"
DATA_PATH = OLD / "all_first_boards_features_and_LABEL_ONLY.csv"
SCREEN_PATH = ROOT / "research/first_board_leader_emotion_20261021T1706Z/ALL28503_PATHS_AND_ORIGINAL_SCREEN.csv"
SCENARIO_TRADES_PATH = OLD / "model_score_SCENARIO_TRADES.csv"
OUT_DIR = ROOT / "research/execution_upgrade_audit_20261029_corrected"

def read(path): return pd.read_csv(path, dtype={"code":str}, float_precision="round_trip", low_memory=False)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value):
    with path.open("x", encoding="utf-8") as f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
def metric(rows):
    known = int(rows.reach4_LABEL_ONLY.notna().sum())
    hits = int(rows.reach4_LABEL_ONLY.eq(1).sum())
    return dict(selected=len(rows), dates=int(rows.signal_date.nunique()), known=known,
                unknown=len(rows)-known, hits=hits, known_precision=hits/known if known else None)

def build_trade_lookup(df):
    required = {"signal_date", "code", "status", "net_return", "net_pnl"}
    if required-set(df): raise ValueError(f"Missing scenario fields: {sorted(required-set(df))}")
    if df.duplicated(["signal_date", "code"]).any(): raise ValueError("Duplicate scenario keys")
    result = {}
    for r in df.to_dict("records"):
        closed = r["status"]=="SCENARIO_CLOSED"
        ret = up._number(r["net_return"])
        if closed and ret is None: raise ValueError("Closed scenario has unknown net_return")
        result[(r["signal_date"], r["code"])] = dict(trade_status=r["status"],
            trade_return=ret if closed else None, trade_net_pnl=up._number(r["net_pnl"]) if closed else None)
    return result

def main(output=OUT_DIR):
    output = Path(output)
    if output.exists(): raise FileExistsError(f"Preserve existing audit outputs: {output}")
    sources = [DATA_PATH, SCREEN_PATH, SCENARIO_TRADES_PATH, OLD/"evaluation_ALL_PREDICTIONS.csv",
               up.LEGACY_BUNDLE_PATH, ROOT/"first_board_execution_upgrade.py", Path(__file__)]
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in sources}
    raw = read(DATA_PATH)
    facts = raw[raw.signal_date.ge("2025-07-01")].copy()
    saved = read(OLD/"evaluation_ALL_PREDICTIONS.csv")
    scored = up.load_historical_legacy_scores(facts, saved)
    archived = read(SCREEN_PATH)[["signal_date", "code", "allowed"]]
    screened = up.apply_original_screen(scored).merge(archived, on=["signal_date", "code"], validate="one_to_one")
    matches = screened.allowed.map(up._true).eq(screened.original_screen_allowed)
    if not matches.all(): raise ValueError("Original frozen screening mismatch")
    order = screened.sort_values(["signal_date", "legacy_rank", "code"])
    raw_top = order.groupby("signal_date").head(3)
    # Fixed original pool comparison, not permission to replace a cancelled trade.
    final_top = order[order.original_screen_allowed].groupby("signal_date").head(3)
    a = set(map(tuple, raw_top[["signal_date", "code"]].to_numpy()))
    b = set(map(tuple, final_top[["signal_date", "code"]].to_numpy()))
    trades = build_trade_lookup(read(SCENARIO_TRADES_PATH))
    ledger = up.DualAccountingLedger()
    for r in order.to_dict("records"):
        key = (r["signal_date"], r["code"])
        if key not in a|b: continue
        kind = "KEEP" if key in a&b else "EXCLUDE_BY_SCREEN" if key in a else "PROMOTE_FROM_BELOW"
        ledger.record_decision(signal_date=r["signal_date"], code=r["code"], name=r["name"],
            legacy_rank=r["legacy_rank"], legacy_score=r["model_score"], final_selected=key in b,
            intervention_type=kind, reason=r["exclusion_reason"] if kind=="EXCLUDE_BY_SCREEN" else "固定原筛选池对照",
            actual_reach4=int(r["reach4_LABEL_ONLY"]) if pd.notna(r["reach4_LABEL_ONLY"]) else None,
            **trades.get(key, {}))
    receipt = dict(timestamp=up.now_utc_iso(), status="HISTORICAL_SCREEN_DIAGNOSTIC_ONLY", fits=0,
        score_source="ORIGINAL_MONTHLY_ROLLING_252_OOF", test_dates=int(order.signal_date.nunique()), test_rows=len(order),
        archived_screen_matches=int(matches.sum()), raw_top3=metric(raw_top), original_screen_top3=metric(final_top),
        ledger_summary=ledger.summarize(), source_hashes=hashes,
        chronology_and_emotion_effect="NOT_EVALUATED_NO_HISTORICAL_EXECUTION_EVIDENCE",
        new_strategy_net_return=None, original_top1_scenario_only=True, production_deployment=False)
    assert all(sha(ROOT/p)==h for p,h in hashes.items()), "Inputs changed during audit"
    output.mkdir()
    order.to_csv(output/"complete13149_original_scores_and_screen.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ledger.records).to_csv(output/"decision_ledger.csv", index=False, encoding="utf-8-sig")
    facts_path = ROOT/"results/daily_review_20261028T1020Z/facts/RESOLVED_FACTS_v2_SCORES_MASKED.json"
    if facts_path.exists():
        live = json.loads(facts_path.read_text(encoding="utf-8"))
        cards, audits, chronology_rows = [], [], []
        names_path = facts_path.parents[1]/"完整112只_独立判断_模型展示前.csv"
        names = read(names_path).drop_duplicates("code").set_index("code").name.to_dict() if names_path.exists() else {}
        for r in live["rows"]:
            if r.get("kind")!="limit_up" or r.get("boards")!=1: continue
            candidate = dict(code=r["code"], name=names.get(r["code"],r["name"]), close=r["close"],
                             signal_date=live["signal_date"], target_date=live["target_date"])
            cards.append(up.generate_auction_execution_card(candidate))
            row = dict(float_cap=r.get("reported_float_cap_cny"), amount=r.get("amount_cny"), turnover=r.get("turnover_pct"),
                       seal_first=r.get("seal_first"), seal_delta=r.get("seal_delta_1430_to_close"))
            audits.append(dict(code=r["code"], capacity=up.audit_capacity_dragon_exemption(row),
                               veto=up.audit_sneak_attack_trap_veto(row)))
            chronology_rows.append(dict(candidate, industry=r.get("industry"), seal_first=r.get("seal_first")))
        write(output/"1028_observation_cards_NOT_AUTHORIZATIONS.json", cards)
        write(output/"1028_microstructure_audit.json", audits)
        up.audit_theme_chronology_full_pool(pd.DataFrame(chronology_rows)).to_csv(
            output/"1028_all51_group_chronology_OBSERVATIONS.csv", index=False, encoding="utf-8-sig")
        receipt["observation_cards"] = len(cards)
        receipt["current_fact_source_sha256"] = sha(facts_path)
        if names_path.exists(): receipt["name_source_sha256"] = sha(names_path)
    write(output/"AUDIT_RECEIPT_EXECUTION_UPGRADE.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))

if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    main(parser.parse_args().output)

