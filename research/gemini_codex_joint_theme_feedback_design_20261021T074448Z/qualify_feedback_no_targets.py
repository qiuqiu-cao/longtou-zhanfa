"""Compute fixed prior-name feedback and actual source-prefix checks, without targets."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import first_board_joint_seal_model as prior
import first_board_prior_name_feedback as f
import first_board_ladder_emotion_features as price
from historical_symbolic_context import DatedNames
from security_master_v4 import validate_master

out = Path(__file__).resolve().parent
if (out / "FEEDBACK_QUALIFICATION.json").exists():
    raise FileExistsError("No qualification overwrite")
paths = [Path(f.__file__), Path(price.__file__),
    prior.ROOT / "historical_symbolic_context.py", prior.ROOT / "symbolic_name_theme_v217.py",
    prior.ROOT / "data/v4/security_master_intervals.csv", out / "NAME_SOURCE_QUALIFICATION.json",
    out / "COMPLETE28503_CANDIDATE_HISTORICAL_NAMES_NO_TARGETS.csv",
    out / "ALL_DECLARED_ARCHIVE_CLOSING_COHORT_NAMES_NO_TARGETS.csv",
    out / "CAUSAL_PRICE_STATES_WITHOUT_TARGETS.joblib", out / "DESIGN_SCOPE_BEFORE_FEATURE_RESULTS.json"]
bindings = {str(p.resolve()): prior.base.sha(p) for p in paths}
prior.base.write_json(out / "FEATURE_SCOPE_BEFORE_COMPUTE.json", dict(at=prior.base.now(), fields=f.FIELDS,
    anchor_min_previous_height=3, follower_prior_heights=[1, 2], loss_cutoff=-.07,
    selection="S-1 complete declared pool; candidate S dated name; matching earlier anchors; union/deduplicate; leave-self-out",
    source_unknown="Any previous name unknown or missing previous scope makes five fieldsNaN; any current selected cohort member missing makes its feedbackNaN",
    absent_cohort="Known no earlier anchor ->presence0,all feedbackNaN; no follower ->follower feedbackNaN",
    labels_loaded=False, model_fits=0, bindings=bindings))
candidates = pd.read_csv(paths[6], dtype={"code": str}, float_precision="round_trip")
closing = pd.read_csv(paths[7], dtype={"code": str}, float_precision="round_trip")
data = joblib.load(paths[8])
states, sessions = data["states"], data["sessions"]
full = f.build(candidates, closing, states, sessions)
full.to_csv(out / "COMPLETE28503_PRIOR_NAME_FEEDBACK_NO_TARGETS.csv", index=False, encoding="utf-8-sig")
checks = []
master = validate_master(pd.read_csv(prior.ROOT / "data/v4/security_master_intervals.csv", dtype={"code": str})).fillna("")
for cut in ("2025-06-30", "2025-12-31"):
    # Recompute raw price states from the original source truncated at cut.
    prefix_states, prefix_sessions = price.load_states(cutoff=cut)
    lookup = DatedNames(master[master.valid_from.le(cut)])
    prefix_candidates = candidates[candidates.signal_date.le(cut)].copy()
    for index, row in prefix_candidates.iterrows():
        named = lookup.resolve(row.code, row.signal_date, row["name"])
        prefix_candidates.loc[index, "historical_name"] = named["name"]
        prefix_candidates.loc[index, "name_ok"] = bool(named["interval_observed"] and named["historical_eligible"] is True)
    prefix_closing = prefix_states[prefix_states.limit][["date", "code", "p", "height"]].rename(columns={"date": "signal_date"}).copy()
    names = []
    for row in prefix_closing.itertuples(index=False):
        named = lookup.resolve(row.code, row.signal_date, "")
        names.append((named["name"], bool(named["interval_observed"] and named["historical_eligible"] is True)))
    prefix_closing["historical_name"] = [x[0] for x in names]
    prefix_closing["name_ok"] = [x[1] for x in names]
    prefix = f.build(prefix_candidates, prefix_closing, prefix_states, prefix_sessions)
    expected = full[full.signal_date.le(cut)].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix, expected)
    checks.append(dict(cutoff=cut, rows=len(prefix), raw_source_and_dated_names_recomputed=True,
                       all_fields_and_member_diagnostics_exact=True))
original = pd.read_csv(prior.FACTS, usecols=prior.KEYS + ["stage"] + prior.ORIGINAL,
                       dtype={"code": str}, float_precision="round_trip")
original = original[original.stage.eq(1)].reset_index(drop=True)
assert original[prior.KEYS].equals(full[prior.KEYS])
duplicates = {field: [old for old in prior.ORIGINAL if np.array_equal(full[field].to_numpy(dtype=float),
              original[old].to_numpy(dtype=float), equal_nan=True)] for field in f.FIELDS}
assert not any(duplicates.values()), duplicates
periods = {}
for name, mask in [("development", full.signal_date.lt("2025-01-01")),
                   ("validation2025H1", full.signal_date.between("2025-01-01", "2025-06-30")),
                   ("evaluation", full.signal_date.ge("2025-07-01"))]:
    part = full[mask]
    periods[name] = dict(rows=len(part), dates=part.signal_date.nunique(),
        present_anchor_rows=int(part.prior_name_anchor_present.eq(1).sum()),
        known_no_anchor_rows=int(part.prior_name_anchor_present.eq(0).sum()),
        unknown_context_rows=int(part.prior_name_anchor_present.isna().sum()),
        field_observation_coverage=part[f.FIELDS].notna().mean().to_dict(),
        anchor_member_unknown_rows=int(part.anchor_unknown_DIAGNOSTIC.gt(0).sum()),
        follower_member_unknown_rows=int(part.follower_unknown_DIAGNOSTIC.gt(0).sum()))
for file, digest in bindings.items():
    assert prior.base.sha(file) == digest
report = dict(at=prior.base.now(), status="FULL28503_CAUSAL_FEEDBACK_SOURCE_QUALIFICATION_PASS",
    rows=len(full), dates=full.signal_date.nunique(), labels_loaded=False, model_fits=0,
    periods=periods, prefix_checks=checks, exact_duplicate_original_fields=duplicates,
    relations_are_name_proxies_not_all_themes=True, original_HTTP_time_certified=False,
    source_nominal_price_and_current_universe_limitations_retained=True,
    core_feature_sha256=prior.base.sha(f.__file__), bindings=bindings)
prior.base.write_json(out / "FEEDBACK_QUALIFICATION.json", report)
print(json.dumps(prior.base.clean({k:v for k,v in report.items() if k != "bindings"}), ensure_ascii=True), flush=True)
