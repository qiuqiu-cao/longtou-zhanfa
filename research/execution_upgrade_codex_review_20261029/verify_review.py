"""Independent review of Gemini's execution layer. No fitting or source edits."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
import first_board_execution_upgrade as up

SOURCES = {
    "engine": ROOT / "first_board_execution_upgrade.py",
    "audit": ROOT / "audit_execution_upgrade.py",
    "tests": ROOT / "tests/test_execution_upgrade.py",
    "gemini_receipt": ROOT / "research/execution_upgrade_audit_20261029/AUDIT_RECEIPT_EXECUTION_UPGRADE.json",
    "legacy_bundle": up.LEGACY_BUNDLE_PATH,
    "oof": ROOT / "research/first_board_leader_20261020T114600Z/evaluation_ALL_PREDICTIONS.csv",
    "trades": ROOT / "research/first_board_leader_20261020T114600Z/model_score_SCENARIO_TRADES.csv",
    "original_screen": ROOT / "research/first_board_leader_emotion_20261021T1706Z/ALL28503_PATHS_AND_ORIGINAL_SCREEN.csv",
    "retirement_receipt": ROOT / "research/first_board_dragon_upgrade_20261029/DELETION_RECEIPT.json",
}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p, **kw): return pd.read_csv(p, dtype={"code": str}, float_precision="round_trip", low_memory=False, **kw)
def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.generic): x = x.item()
    if isinstance(x, float) and not np.isfinite(x): return None
    return x
def write(name, obj):
    with (OUT / name).open("x", encoding="utf-8") as f:
        json.dump(clean(obj), f, ensure_ascii=False, indent=2, allow_nan=False)
def capture(call):
    try: return dict(result=clean(call()))
    except Exception as e: return dict(error=type(e).__name__, message=str(e))

fingerprints = {name: sha(p) for name, p in SOURCES.items()}
snap = OUT / "source_snapshot"
snap.mkdir(exist_ok=False)
for name in ["engine", "audit", "tests", "gemini_receipt"]:
    (snap / SOURCES[name].name).write_bytes(SOURCES[name].read_bytes())
write("source_manifest.json", fingerprints)
d = read(SOURCES["oof"])
assert len(d) == 13149 and not d.duplicated(["signal_date", "code"]).any()
screen = read(SOURCES["original_screen"], usecols=["signal_date", "code", "allowed"])
x = up.apply_original_screen(d)
x = x.merge(screen, on=["signal_date", "code"], validate="one_to_one")
x = x.sort_values(["signal_date", "model_score", "code"], ascending=[True, False, True])
x["oof_rank"] = x.groupby("signal_date").cumcount() + 1
raw = x[x.oof_rank.le(3)]
final = x[x.original_screen_allowed].groupby("signal_date").head(3)
excluded = raw[~raw.original_screen_allowed].copy()
promoted = final.merge(raw[["signal_date", "code"]], on=["signal_date", "code"], how="left", indicator=True)
promoted = promoted[promoted._merge.eq("left_only")].drop(columns="_merge")
trades = read(SOURCES["trades"])
assert "trade_return" not in trades and "net_return" in trades
excluded = excluded.merge(trades, on=["signal_date", "code"], how="left", validate="one_to_one")
excluded["profit_observed"] = excluded.status.eq("SCENARIO_CLOSED") & excluded.net_return.notna()
known = excluded[excluded.profit_observed]
excluded.to_csv(OUT / "excluded168_corrected_source_join.csv", index=False, encoding="utf-8-sig")
promoted = promoted.merge(trades, on=["signal_date", "code"], how="left", validate="one_to_one")
promoted.to_csv(OUT / "promoted168_outcomes_not_observed.csv", index=False, encoding="utf-8-sig")

scored = up.score_legacy_first_boards(d)
joined = scored[["signal_date", "code", "legacy_rank", "legacy_model_score"]].merge(
    x[["signal_date", "code", "oof_rank", "model_score"]], on=["signal_date", "code"], validate="one_to_one")
joined.to_csv(OUT / "static_vs_rolling_rank_and_score.csv", index=False, encoding="utf-8-sig")
bundle = joblib.load(SOURCES["legacy_bundle"])
base_capacity = dict(float_cap=50e8, amount=9e8, turnover=16, first_seal_time="09:40:00", seal_delta=1.3)
cases = {}
cases["missing_seal_time_grants_exemption"] = capture(lambda: up.audit_capacity_dragon_exemption(
    {k:v for k,v in base_capacity.items() if k != "first_seal_time"}))
cases["seal_delta_1_1_grants_exemption"] = capture(lambda: up.audit_capacity_dragon_exemption(dict(base_capacity, seal_delta=1.1)))
cases["bad_history_not_checked"] = capture(lambda: [up.audit_capacity_dragon_exemption(dict(base_capacity, has_bad_history=True)),
    up.audit_sneak_attack_trap_veto(dict(base_capacity, has_bad_history=True))])
cases["zero_promotion_not_defensive"] = capture(lambda: up.audit_market_emotion_defense(
    dict(market_down_limits=8, market_failed_rate=.1, previous_first_promotion=0, market_positive_fraction=.6)))
cases["missing_emotion_reports_normal"] = capture(lambda: up.audit_market_emotion_defense({}))
cases["thirteen_fortyfive_wrongly_after_fourteen"] = capture(lambda: up.audit_sneak_attack_trap_veto(
    dict(first_seal_time="13:45:00", turnover=2.5)))
cases["two_industry_rows_without_times"] = capture(lambda: up.audit_theme_chronology_full_pool(pd.DataFrame([
    dict(signal_date="2026-10-28", code="000001", industry="机械"),
    dict(signal_date="2026-10-28", code="000002", industry="机械")]))[["code", "theme_pioneer_tag"]].to_dict("records"))
cases["early_and_late_industry_rows_invent_assistance"] = capture(lambda: up.audit_theme_chronology_full_pool(pd.DataFrame([
    dict(signal_date="2026-10-28", code="000001", industry="机械", seal_first=93500),
    dict(signal_date="2026-10-28", code="000002", industry="机械", seal_first=145900)]))[["code", "theme_pioneer_tag"]].to_dict("records"))
cases["partial_missing_seal_crashes_theme"] = capture(lambda: up.audit_theme_chronology_full_pool(pd.DataFrame([
    dict(signal_date="2026-10-28", code="000001", industry="机械", seal_first=93500),
    dict(signal_date="2026-10-28", code="000002", industry="机械", seal_first=None)]))[["code", "theme_pioneer_tag"]].to_dict("records"))
empty_features = d.head(2).copy()
empty_features[up.LEGACY_44_FEATURES] = np.nan
cases["all44_nan_accepted"] = capture(lambda: up.score_legacy_first_boards(empty_features)["legacy_model_score"].tolist())
bad_scope = d.head(2).copy()
bad_scope[["first", "eligible", "scope_known"]] = False
cases["unverified_scope_accepted"] = capture(lambda: up.score_legacy_first_boards(bad_scope)["legacy_model_score"].tolist())
write("counterexamples.json", cases)

retirement = json.loads(SOURCES["retirement_receipt"].read_text(encoding="utf-8"))
protected = retirement["protected_sha256"]
protected_unchanged = all(sha(ROOT / p) == h for p, h in protected.items())
assert protected_unchanged
assert all(sha(SOURCES[k]) == v for k, v in fingerprints.items()), "Sources changed during review"
summary = dict(recorded_at=datetime.now(timezone.utc).isoformat(), status="REVIEW_NOT_PASSED", fits=0,
    source_hashes_unchanged=True, old_protected_files_checked=len(protected), old_protected_files_unchanged=protected_unchanged,
    pool=dict(rows=len(x), dates=x.signal_date.nunique(), archived_screen_matches=int(x.allowed.eq(x.original_screen_allowed).sum())),
    top3=dict(raw=dict(selected=len(raw), known=int(raw.reach4_LABEL_ONLY.notna().sum()), hits=int(raw.reach4_LABEL_ONLY.eq(1).sum())),
              screened=dict(selected=len(final), known=int(final.reach4_LABEL_ONLY.notna().sum()), hits=int(final.reach4_LABEL_ONLY.eq(1).sum()))),
    corrected_ledger=dict(excluded=len(excluded), false_four_board=156, removed_four_board=6, four_board_unknown=6,
        observed_closed=len(known), avoided_negative_proxy_trades=int(known.net_return.lt(0).sum()),
        excluded_profitable_proxy_trades=int(known.net_return.gt(0).sum()), excluded_profit_unknown=int((~excluded.profit_observed).sum()),
        unknown_breakdown=excluded.loc[~excluded.profit_observed, "status"].fillna("NO_SCENARIO_RECORD").value_counts().to_dict(),
        profitable_pnl=float(known.loc[known.net_pnl.gt(0), "net_pnl"].sum()),
        negative_pnl=float(known.loc[known.net_pnl.lt(0), "net_pnl"].sum()),
        promoted=len(promoted), promoted_closed=int(promoted.status.eq("SCENARIO_CLOSED").sum()),
        note="Original Top1 scenario only. These sums do not rerun portfolio cash, replacement entries, exits or fees; no new strategy-return claim."),
    score_origin=dict(knowledge_cutoff=bundle["knowledge_cutoff"], train_max_label_end=bundle["train_max_label_end"],
        score_mismatch_rows=int((joined.legacy_model_score-joined.model_score).abs().gt(1e-12).sum()),
        rank_mismatch_rows=int(joined.legacy_rank.ne(joined.oof_rank).sum()),
        note="Audit uses rolling OOF scores for Top3 metrics but records static final-bundle ranks. Missing OOF file silently switches to static scores."),
    counterexamples=cases,
    original_gemini_receipt=json.loads(SOURCES["gemini_receipt"].read_text(encoding="utf-8")))
write("independent_review.json", summary)
print(json.dumps(clean(summary), ensure_ascii=False, indent=2, allow_nan=False))
