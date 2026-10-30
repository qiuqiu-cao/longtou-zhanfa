"""Read-only full-date error decomposition, not new fits or a threshold search."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import first_board_joint_theme_feedback_model as m

out = Path(__file__).resolve().parent / "codex_error_review"
out.mkdir(exist_ok=True)
rows = m.load_rows()
val = rows[rows.signal_date.between("2025-01-01", "2025-06-30")].copy()
old = m.risk.attach_frozen(rows, m.prior.REFERENCE / "validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
assert len(old) == 4724
diag = pd.read_csv(m.DESIGN / "COMPLETE28503_PRIOR_NAME_FEEDBACK_NO_TARGETS.csv", dtype={"code": str}, float_precision="round_trip")
diag = diag[["signal_date", "code"] + [x for x in diag if x.endswith("_DIAGNOSTIC")]]
summary = {}
paired_all = []
pick_all = []


def brief(part):
    return dict(rows=len(part), known=int(part[m.stage.TARGET].notna().sum()),
                hits=int(part[m.stage.TARGET].eq(1).sum()))


def name_scope(part):
    has = part.prior_name_anchor_present.eq(1)
    return dict(name_anchor_present_rows=int(has.sum()), follower_feedback_observed=int(part.name_prev_followers_current_mean_return.notna().sum()),
        overlapping_anchor_count_ge2=int(part.anchor_members_DIAGNOSTIC.ge(2).sum()),
        follower_count_1=int(part.follower_members_DIAGNOSTIC.eq(1).sum()),
        single_character_tokens_only_rows=int(part.matched_tokens_DIAGNOSTIC.fillna("").map(
            lambda x: bool(x) and all(t.startswith("H1:") for t in x.split("|"))).sum()))


for label in ["BASE", m.A, m.B]:
    pred = old.copy() if label == "BASE" else pd.read_csv(Path(__file__).resolve().parent / label / "VALIDATION_ALL_PREDICTIONS_BEFORE_METRICS.csv", dtype={"code": str}, float_precision="round_trip")
    pred = pred.merge(diag, on=m.KEYS, how="left", validate="one_to_one")
    by_scope = {}
    for scope, pool in [("all_screened", pred[pred.research_selection_allowed]),
                        ("non_one_price_screened", pred[pred.research_selection_allowed & pred.one_price.eq(0)])]:
        take = m.base.selected(pool, "model_score", 1).copy()
        take["model"] = label
        take["scope"] = scope
        pick_all.append(take)
        by_scope[scope] = dict(total=brief(take), signal_one_price=brief(take[take.one_price.eq(1)]),
            signal_non_one_price=brief(take[take.one_price.eq(0)]), name_scope=name_scope(take),
            picks_with_anchor=brief(take[take.prior_name_anchor_present.eq(1)]),
            picks_without_anchor=brief(take[take.prior_name_anchor_present.eq(0)]))
        if label != "BASE":
            baseline_pool = old[old.research_selection_allowed]
            if scope == "non_one_price_screened":
                baseline_pool = baseline_pool[baseline_pool.one_price.eq(0)]
            ref = m.base.selected(baseline_pool, "model_score", 1)
            cols = m.KEYS + ["name", m.stage.TARGET, "model_score", "one_price", "prior_name_anchor_present",
                "name_anchor_current_mean_return", "name_anchor_current_limit_survival", "name_prev_followers_current_mean_return",
                "name_prev_followers_current_loss7_share", "turnover", "float_cap", "provider_reported_any_reopen",
                "reported_first_seal_trading_minutes", "anchor_members_DIAGNOSTIC", "follower_members_DIAGNOSTIC",
                "matched_tokens_DIAGNOSTIC", "anchor_codes_DIAGNOSTIC", "follower_codes_DIAGNOSTIC"]
            # BASE inputs are reattached by original keys solely for diagnostics.
            ref = ref[m.KEYS + [m.stage.TARGET, "model_score"]].merge(
                pred[[x for x in cols if x not in [m.stage.TARGET, "model_score"]]], on=m.KEYS, validate="one_to_one")
            pair = take[cols].merge(ref[cols], on="signal_date", suffixes=("_new", "_base"), validate="one_to_one")
            y, z = pair[m.stage.TARGET+"_new"].eq(1), pair[m.stage.TARGET+"_base"].eq(1)
            pair["outcome_change"] = np.select([y & ~z, ~y & z, y & z], ["GAIN", "LOSS", "BOTH_HIT"], default="BOTH_MISS_OR_UNKNOWN")
            pair["scope"] = scope
            pair["model"] = label
            paired_all.append(pair)
            counts = {}
            for change in ("GAIN", "LOSS"):
                x = pair[pair.outcome_change.eq(change)]
                counts[change] = dict(dates=len(x), new_signal_one_price=int(x.one_price_new.eq(1).sum()),
                    base_signal_one_price=int(x.one_price_base.eq(1).sum()),
                    new_has_anchor=int(x.prior_name_anchor_present_new.eq(1).sum()),
                    base_has_anchor=int(x.prior_name_anchor_present_base.eq(1).sum()),
                    both_no_name_anchor=int((x.prior_name_anchor_present_new.eq(0) & x.prior_name_anchor_present_base.eq(0)).sum()))
            by_scope[scope]["paired_changes"] = counts
    summary[label] = by_scope

picks = pd.concat(pick_all, ignore_index=True)
pairs = pd.concat(paired_all, ignore_index=True)
picks.to_csv(out / "ALL117_PICKS_ALL_SCOPES.csv", index=False, encoding="utf-8-sig")
pairs.to_csv(out / "ALL117_PAIRED_CONTEXT_AND_OUTCOMES.csv", index=False, encoding="utf-8-sig")
val_context = val.merge(diag, on=m.KEYS, how="left", validate="one_to_one")
report = dict(at=m.base.now(), status="READONLY_ERROR_DECOMPOSITION_COMPLETE", new_fits=0,
    historical_validation_already_seen=True, no_posthoc_thresholds_or_revised_acceptance=True,
    validation_pool=brief(val), feature_scope=name_scope(val_context), models=summary,
    interpretation="Signal-day one-price stratification is descriptive and differs from forcing a new ranking only among non-one-price stocks. No strategy/causal attribution follows from these counts.")
m.base.write_json(out / "report.json", report)
print(json.dumps(m.base.clean(report), ensure_ascii=True), flush=True)
