"""Freeze the jointly discussed event path comparison before any model fit."""
from pathlib import Path
import json
import shutil
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import first_board_joint_event_path_model as m

out = Path(__file__).resolve().parent
m.OUTPUT = out
assert not any((out / arm).exists() for arm in m.ARMS)
agreement_path = m.DESIGN / "gemini/FINAL_CANDIDATE_AGREEMENT.json"
agreement = json.loads(agreement_path.read_text("utf-8"))
assert set(agreement["final_candidate_arms"]) == set(m.ARMS)
for arm, item in agreement["final_candidate_arms"].items():
    assert item["fields_added"] == m.ARMS[arm][len(m.prior.ORIGINAL):]
    assert item["fields_count"] == len(m.ARMS[arm])
qualification = json.loads((m.DESIGN / "EVENT_PATH_QUALIFICATION.json").read_text("utf-8"))
assert qualification["fits"] == 0 and not qualification["labels_read"]
assert qualification["feature_source_sha256"] == m.base.sha(m.features.__file__)
assert qualification["feature_file_sha256"] == m.base.sha(m.NEW_FIELDS_FILE)
assert all(qualification["statistics"][period]["full_window_fraction"] >= .90 for period in ("development", "validation_2025H1"))
rows = m.load_rows()
original = pd.read_csv(m.prior.FACTS, dtype={"code":str}, float_precision="round_trip")
original = original[original.stage.eq(1)].reset_index(drop=True)
pd.testing.assert_frame_equal(rows[original.columns], original)
assert len(rows) == 28503 and rows.signal_date.nunique() == 668 and rows[m.stage.TARGET].isna().sum() == 126
train = m.members.train_members(rows, "2025-01-01")
old = pd.read_csv(m.prior.REFERENCE / "models/validation_S1_STAGE_CYCLE_GENE_full_members.csv", dtype={"code":str})
keys = m.KEYS + ["label_end", m.stage.TARGET]
assert set(map(tuple, train[keys].to_numpy())) == set(map(tuple, old[keys].to_numpy()))
dates = sorted(train.signal_date.unique())
cutoff = dates[-42]
inner, calibration = train[train.label_end.lt(cutoff)], train[train.signal_date.ge(cutoff)]
fields = m.features.FIELDS["sealed"][2:] + m.features.FIELDS["probe"][2:]
support = {}
for field in fields:
    support[field] = dict(inner_rows=int(inner[field].notna().sum()), inner_dates=int(inner.loc[inner[field].notna(), "signal_date"].nunique()),
        calibration_rows=int(calibration[field].notna().sum()), calibration_dates=int(calibration.loc[calibration[field].notna(), "signal_date"].nunique()))
    assert support[field]["inner_rows"] >= 1000 and support[field]["inner_dates"] >= 60
    assert support[field]["calibration_rows"] >= 300 and support[field]["calibration_dates"] >= 30
val = rows[rows.signal_date.between("2025-01-01", "2025-06-30")]
assert len(val) == 4724 and val.signal_date.nunique() == 117 and val[m.stage.TARGET].isna().sum() == 10
frozen = m.risk.attach_frozen(rows, m.prior.REFERENCE / "validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
assert m.risk.screen_metric(frozen)["hits"] == 52 and m.risk.screen_metric(frozen[frozen.one_price.eq(0)])["hits"] == 48
original_protocol = json.loads((m.prior.JOINT / "JOINT_FROZEN_PROTOCOL.json").read_text("utf-8"))
bindings = original_protocol["bindings"].copy()
for path in [Path(m.__file__), Path(m.features.__file__), Path(__file__), m.NEW_FIELDS_FILE,
    m.DESIGN / "EVENT_PATH_QUALIFICATION.json", m.DESIGN / "qualify_event_paths.py",
    m.DESIGN / "EVENT_PATH_QUALIFICATION_BEFORE_COMPUTE.json", m.DESIGN / "EVENT_FEATURE_EXISTING_TRANSFORM_AUDIT.json",
    m.DESIGN / "已有变量变换审查与候选修订.md", m.DESIGN / "sources.json", m.DESIGN / "股吧经验与六类因子候选.md",
    m.DESIGN / "gemini/GEMINI_FORUM_FACTOR_DESIGN.md", m.DESIGN / "gemini/GEMINI_FORUM_FACTOR_DESIGN_RESPONSE.md",
    m.DESIGN / "gemini/FINAL_CANDIDATE_AGREEMENT.md", agreement_path,
    ROOT / "tests/test_first_board_event_path_features.py"]:
    bindings[str(path.resolve())] = m.base.sha(path)
for path, digest in bindings.items():
    assert m.base.sha(path) == digest, path
preflight = dict(at=m.base.now(), fits=0, rows=28503, unknown126_preserved=True,
    original_fields_labels_screen_equal=True, exact_original_train_members=True,
    validation4724_unknown10_retained=True, baseline_primary_hits=52, baseline_non_one_hits=48,
    complete_source_prefixes=[15354, 20229], targeted_source_tests_passed=6,
    train_rows=len(train), inner_rows=len(inner), calibration_rows=len(calibration), actual_support=support,
    source_window_fraction={period:qualification["statistics"][period]["full_window_fraction"] for period in qualification["statistics"]})
m.base.write_json(out / "CODEX_PREFLIGHT_NO_FITS.json", preflight)
protocol = dict(at=m.base.now(), status="AWAIT_GEMINI_ACK_NO_FITS", batch_started_at="2026-10-21T03:03:34+00:00",
    batch_deadline_utc="2026-10-21T03:33:34+00:00", budget_minutes=30,
    counterpart="Actual Antigravity Gemini 3.8 Flash High; joint design completed before numerical trials",
    source_sha256=m.base.sha(m.__file__), qualification_sha256=m.base.sha(m.DESIGN / "EVENT_PATH_QUALIFICATION.json"),
    hypothesis="Prior nominal seal/probe event-aligned correction depth, worst daily return, volume contraction and S restart distinguish next-official-close promotion beyond original62. Not a retry of old generic20-day volume, outcome-memory, high-distance or static-name arms.",
    arms={arm:dict(trainer=m.OWNERS[arm], fields=fields) for arm, fields in m.ARMS.items()},
    source_definition="Latest separate nominal sealed/probe event within S-20..S-1. Correction interval E+1..S-1;currentS excluded from event search, included only restart response. No correction interval means NaN,not zero or1. Exact traded cents,all21adjacentvalidpositive-volume bars. Archive pre_close is prior raw close,not exchange ex-rights reference. No actor intent labels.",
    design_decision="Keep found/age in source diagnostics only;both found indicators and sealed age were independently confirmed transforms of existing original count/age. Only5pathmetrics per type enter models. No69/76 or draft age-based67/72 fits.",
    source_limits="Historical late source capture;existing universe survivorship and many prior uses of validation/evaluation remain. Three forum posts inspire hypotheses,never backfilled narrative truth. PidType1 old668datearchive is first-board scope;36same-close next-day appearances and remaining absence cannot certify all labels true/false. No target rewrites.",
    fixed_rules={key:original_protocol[key] for key in ("training", "validation", "evaluation", "acceptance")},
    selection="Each exact arm once. Primary hits>52 AND non-one hits>=48 AND date coverage>=95%. Only passing arm with most primary hits proceeds to14monthlyfixed252-official-sessionprefixes.67wins challenger ties;BASEwins ties. No outcome-based abstention or screen/threshold/window changes.",
    commands={arm:f"D:/python/python.exe first_board_joint_event_path_model.py train --output {out.relative_to(ROOT).as_posix()} --arm {arm} --trainer {m.OWNERS[arm]}" for arm in m.ARMS},
    evaluation_command=f"D:/python/python.exe first_board_joint_event_path_model.py evaluate --output {out.relative_to(ROOT).as_posix()}",
    evaluation_owner="Codex once after both reports;winner only;BASE/S2 refits0",
    acknowledgment="Gemini may ONLY change status to AGREED_BEFORE_NEW_FITS and add actual agreed_at/acknowledgment_note;preserve all other fields and hashes. If objections write separate receipt and do not fit.",
    production_deployment_allowed=False, actual_trades=0, new_forward_results=0, bindings=bindings)
with (out / "JOINT_FROZEN_PROTOCOL.json").open("x", encoding="utf-8") as f:
    json.dump(protocol, f, ensure_ascii=False, indent=2)
shutil.copy2(out / "JOINT_FROZEN_PROTOCOL.json", out / "PROTOCOL_PENDING_ACK_SNAPSHOT.json")
print(json.dumps(preflight, ensure_ascii=True))
