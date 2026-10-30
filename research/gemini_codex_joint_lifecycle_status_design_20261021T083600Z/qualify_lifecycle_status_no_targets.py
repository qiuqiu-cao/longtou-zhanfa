"""Qualify causal theme lifecycle and candidate relative status features without targets."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
import joblib
import numpy as np
import pandas as pd
import first_board_joint_seal_model as prior
import first_board_lifecycle_status_features as feat
from historical_symbolic_context import DatedNames
from security_master_v4 import validate_master

out = Path(__file__).resolve().parent

DESIGN_PRIOR = prior.ROOT / "research/gemini_codex_joint_theme_feedback_design_20261021T074448Z"
NEW_FIELDS_PRIOR = prior.ROOT / "research/first_board_joint_design_20261020T185417Z/qualification_resumed_20261021/COMPLETE28503_NEW_FIELDS_NO_LABELS.csv"

paths = [
    Path(feat.__file__),
    DESIGN_PRIOR / "COMPLETE28503_CANDIDATE_HISTORICAL_NAMES_NO_TARGETS.csv",
    DESIGN_PRIOR / "ALL_DECLARED_ARCHIVE_CLOSING_COHORT_NAMES_NO_TARGETS.csv",
    NEW_FIELDS_PRIOR,
    DESIGN_PRIOR / "CAUSAL_PRICE_STATES_WITHOUT_TARGETS.joblib",
    prior.ROOT / "historical_symbolic_context.py",
    prior.ROOT / "data/v4/security_master_intervals.csv",
]

bindings = {str(p.resolve()): prior.base.sha(p) for p in paths}

candidates = pd.read_csv(paths[1], dtype={"code": str}, float_precision="round_trip")
closing = pd.read_csv(paths[2], dtype={"code": str}, float_precision="round_trip")
facts = pd.read_csv(paths[3], dtype={"code": str}, float_precision="round_trip")
data = joblib.load(paths[4])
states, sessions = data["states"], data["sessions"]

full = feat.build(candidates, closing, facts, states, sessions)
full.to_csv(out / "COMPLETE28503_LIFECYCLE_STATUS_NO_TARGETS.csv", index=False, encoding="utf-8-sig")

original = pd.read_csv(prior.FACTS, usecols=prior.KEYS + ["stage"] + prior.ORIGINAL,
                       dtype={"code": str}, float_precision="round_trip")
original = original[original.stage.eq(1)].reset_index(drop=True)
assert original[prior.KEYS].equals(full[prior.KEYS])

# Check for exact duplicate fields against original BASE_62
duplicates = {field: [old for old in prior.ORIGINAL if np.array_equal(full[field].to_numpy(dtype=float),
              original[old].to_numpy(dtype=float), equal_nan=True)] for field in feat.ALL_FIELDS}
assert not any(duplicates.values()), duplicates

# Period coverage statistics
periods = {}
for name, mask in [("development", full.signal_date.lt("2025-01-01")),
                   ("validation2025H1", full.signal_date.between("2025-01-01", "2025-06-30")),
                   ("evaluation", full.signal_date.ge("2025-07-01"))]:
    part = full[mask]
    periods[name] = dict(
        rows=len(part),
        dates=part.signal_date.nunique(),
        field_coverage={col: float(part[col].notna().mean()) for col in feat.ALL_FIELDS},
        anchor_observed_rows=int(part.name_anchor_broken_age_sessions.notna().sum()),
        cohort_observed_rows=int(part.theme_candidate_cohort_size.notna().sum()),
        multi_cohort_rows=int(part.theme_first_seal_rank_in_group.notna().sum()),
    )

report = dict(
    at=prior.base.now(),
    status="FULL28503_CAUSAL_LIFECYCLE_STATUS_QUALIFICATION_PASS",
    rows=len(full),
    dates=full.signal_date.nunique(),
    labels_loaded=False,
    model_fits=0,
    http_requests=0,
    fields=feat.ALL_FIELDS,
    lifecycle_fields=feat.LIFECYCLE_FIELDS,
    status_fields=feat.STATUS_FIELDS,
    periods=periods,
    exact_duplicate_original_fields=duplicates,
    relations_are_name_proxies_not_all_themes=True,
    bindings=bindings,
    core_feature_sha256=prior.base.sha(feat.__file__),
)

prior.base.write_json(out / "LIFECYCLE_STATUS_QUALIFICATION.json", report)
print(json.dumps(prior.base.clean({k: v for k, v in report.items() if k != "bindings"}), ensure_ascii=True), flush=True)
