"""Source-only qualification for fixed prior-cohort name proxies; zero target reads."""
from pathlib import Path
import json
import joblib
import pandas as pd
import first_board_joint_seal_model as prior
import first_board_ladder_emotion_features as price
from historical_symbolic_context import DatedNames
from security_master_v4 import validate_master

out = Path(__file__).resolve().parent
marker = out / "NAME_SOURCE_QUALIFICATION.json"
if marker.exists():
    raise FileExistsError("Preserve original qualification")
master_path = prior.ROOT / "data/v4/security_master_intervals.csv"
master = validate_master(pd.read_csv(master_path, dtype={"code": str})).fillna("")
lookup = DatedNames(master)
keys = pd.read_csv(prior.FACTS, usecols=prior.KEYS + ["stage", "name"], dtype={"code": str})
keys = keys[keys.stage.eq(1)].reset_index(drop=True)
price.reject_targets(keys)
assert len(keys) == 28503 and keys.signal_date.nunique() == 668
states, sessions = price.load_states()
cols = ["date", "code", "p", "eligible", "valid_price", "regular_price", "limit", "height", "day_return"]
states = states[cols]
joblib.dump(dict(states=states, sessions=sessions), out / "CAUSAL_PRICE_STATES_WITHOUT_TARGETS.joblib", compress=3)
closing = states[states.limit][["date", "code", "p", "height"]].rename(columns={"date": "signal_date"}).copy()

def named(frame, archive=False):
    records = []
    for row in frame.to_dict("records"):
        resolved = lookup.resolve(row["code"], row["signal_date"], row.get("name", ""))
        row.update(historical_name=resolved["name"], name_status=resolved["status"],
            name_interval_id=resolved["interval_id"],
            interval_declared_verified=resolved["interval_observed"],
            name_interval_eligibility=resolved["historical_eligible"],
            name_ok=bool(resolved["interval_observed"] and resolved["historical_eligible"] is True
                         and resolved["name"]))
        records.append(row)
    return pd.DataFrame(records)

candidate_names = named(keys)
closing_names = named(closing)
candidate_names.to_csv(out / "COMPLETE28503_CANDIDATE_HISTORICAL_NAMES_NO_TARGETS.csv", index=False, encoding="utf-8-sig")
closing_names.to_csv(out / "ALL_DECLARED_ARCHIVE_CLOSING_COHORT_NAMES_NO_TARGETS.csv", index=False, encoding="utf-8-sig")
coverage = closing_names.groupby("p").agg(members=("code", "size"), names_known=("name_ok", "sum")).reset_index()
coverage["signal_date"] = [sessions[int(p)+1] if int(p)+1 < len(sessions) else None for p in coverage.p]
coverage["full_previous_pool_names_known"] = coverage.members.eq(coverage.names_known)
coverage["name_coverage"] = coverage.names_known / coverage.members
coverage.to_csv(out / "DAILY_PREVIOUS_POOL_NAME_COVERAGE_NO_TARGETS.csv", index=False)
periods = {}
for name, mask in [("development", candidate_names.signal_date.lt("2025-01-01")),
                   ("validation2025H1", candidate_names.signal_date.between("2025-01-01", "2025-06-30")),
                   ("evaluation", candidate_names.signal_date.ge("2025-07-01"))]:
    part = candidate_names[mask]
    daily = coverage[coverage.signal_date.isin(part.signal_date)]
    periods[name] = dict(rows=len(part), dates=part.signal_date.nunique(),
        candidate_name_coverage=float(part.name_ok.mean()),
        previous_pool_dates_with_all_names_known=int(daily.full_previous_pool_names_known.sum()),
        previous_pool_dates_total=len(daily),
        previous_pool_name_coverage=float(daily.names_known.sum()/daily.members.sum()))
paths = [master_path, Path(price.__file__), prior.FACTS] + list(prior.base.INPUTS.values())
report = dict(at=prior.base.now(), status="NAME_AND_PRIOR_COHORT_SOURCE_COUNTS_COMPLETE_NO_FITS",
    labels_read=False, research_fits=0, candidate_rows=len(keys), candidate_dates=668,
    prior_closing_rows=len(closing), periods=periods,
    historical_names_are_late_reconstruction_not_strict_original_receipt=True,
    master_verification_status_is_not_new_independent_verification=True,
    relations_are_not_verified_market_theme=True, nominal_price_and_surviving_universe_limits_retained=True,
    bindings={str(p.resolve()): prior.base.sha(p) for p in paths})
prior.base.write_json(marker, report)
print(json.dumps(prior.base.clean(report), ensure_ascii=True), flush=True)
