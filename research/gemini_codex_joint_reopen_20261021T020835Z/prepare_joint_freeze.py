"""Bind reviewed evidence, unchanged members and two agreed proposals before either fit."""
import json
import csv
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import first_board_joint_reopen_model as model

out = Path(__file__).resolve().parent
proposal = out / "gemini/GEMINI_REOPEN_PROPOSAL.md"
correction = out / "gemini/GEMINI_REOPEN_PROPOSAL_CORRECTION.md"
assert proposal.exists(), "Actual Gemini proposal required before common freeze"
assert correction.exists(), "Peer correction to coverage and causal wording required"
assert not any((out / arm).exists() for arm in model.ARMS)
archived = json.loads((ROOT / "research/first_board_seal_source_qualification_20261020/raw_bound_first_board_observations.json").read_text("utf-8"))
raw_bindings = {}
raw_tables = {}
for record in archived:
    raw_path = record["provider_source_path"]
    assert model.base.sha(raw_path) == record["provider_source_sha256"]
    raw_bindings[raw_path] = record["provider_source_sha256"]
    assert model.base.sha(record["receipt_path"]) == record["receipt_sha256"]
    raw_bindings[record["receipt_path"]] = record["receipt_sha256"]
    if raw_path not in raw_tables:
        with Path(raw_path).open("r", encoding="utf-8-sig", newline="") as file:
            raw_tables[raw_path] = list(csv.DictReader(file))
    original = raw_tables[raw_path][record["raw_line_number"] - 2]
    assert original[record["raw_field_columns"]["code"]].zfill(6) == record["code"]
    assert float(original[record["raw_field_columns"]["bomb_count"]]) == record["bomb_count"]
rows = model.load_rows()
assert len(rows) == 28503 and rows.signal_date.nunique() == 668
assert rows[model.stage.TARGET].isna().sum() == 126
qualification = json.loads((out / "qualification.json").read_text("utf-8"))
assert qualification["qualified"] and qualification["cross_source_matches"] == 122
train = model.members.train_members(rows, "2025-01-01")
old = pd.read_csv(model.prior.REFERENCE / "models/validation_S1_STAGE_CYCLE_GENE_full_members.csv", dtype={"code": str})
columns = model.KEYS + ["label_end", model.stage.TARGET]
assert set(map(tuple, train[columns].to_numpy())) == set(map(tuple, old[columns].to_numpy()))
val = rows[rows.signal_date.between("2025-01-01", "2025-06-30")]
frozen = model.risk.attach_frozen(rows, model.prior.REFERENCE / "validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
assert model.risk.screen_metric(frozen)["hits"] == 52
assert model.risk.screen_metric(frozen[frozen.one_price.eq(0)])["hits"] == 48
assert len(val) == 4724 and val.signal_date.nunique() == 117
assert val[model.stage.TARGET].isna().sum() == 10
source = pd.read_csv(model.NEW_FIELDS_FILE, dtype={"code": str}, float_precision="round_trip")
for cutoff in ("2025-06-30", "2025-12-31"):
    part = model.attach_flag(rows[rows.signal_date.le(cutoff)].drop(columns=model.FIELD).reset_index(drop=True), source[source.signal_date.le(cutoff)])
    np.testing.assert_allclose(part[model.FIELD], rows.loc[rows.signal_date.le(cutoff), model.FIELD], atol=0, rtol=0, equal_nan=True)
preflight = dict(at=model.base.now(), fits=0, full_rows=28503, full_dates=668, unknown126_preserved=True,
                 validation_rows=4724, validation_dates=117, validation_unknown=10,
                 all_original_fields_labels_and_screen_unchanged=True, full_members_match_baseline=True,
                 baseline_primary_hits=52, baseline_non_one_hits=48, prefix_checks=2,
                 targeted_tests="4 passed; real zero vs unknown/full-pool/no-label-source/future-prefix")
model.base.write_json(out / "CODEX_PREFLIGHT_NO_FITS.json", preflight)
old_freeze = json.loads((model.prior.JOINT / "JOINT_FROZEN_PROTOCOL.json").read_text("utf-8"))
bindings = old_freeze["bindings"].copy()
bindings.update(raw_bindings)
extra = [Path(model.__file__), Path(model.prior.__file__), out / "qualify_reopen_status.py", Path(__file__),
         model.NEW_FIELDS_FILE, out / "qualification.json", out / "PROTOCOL_BEFORE_FULL_FIELD_AUDIT.json",
         out / "ALL122_CROSS_SOURCE_REOPEN_SEMANTICS_NO_TARGETS.csv", proposal, correction,
         ROOT / "tests/test_first_board_joint_reopen_model.py",
         ROOT / "research/first_board_seal_source_qualification_20261020/raw_bound_first_board_observations.json"]
for path in extra:
    bindings[str(path.resolve())] = model.base.sha(path)
for path, digest in bindings.items():
    assert model.base.sha(path) == digest, path
freeze = dict(at=model.base.now(), status="AWAIT_GEMINI_ACK_NO_FITS", budget_minutes=30,
    batch_started_at="2026-10-21T02:08:35+00:00", batch_deadline_utc="2026-10-21T02:38:35+00:00",
    counterpart="Actual Antigravity Gemini3.8 Flash High; user explicitly authorizes joint discussion and training",
    source_sha256=model.base.sha(model.__file__), qualification_sha256=model.base.sha(out / "qualification.json"),
    hypothesis="Newly qualified supplier S-day any-reopening binary state adds intraday-path information beyond the frozen62; test alone and with previously qualified timing/fund fields, no old64/66 retraining",
    arms={arm:dict(trainer=model.OWNERS[arm], fields=fields) for arm, fields in model.ARMS.items()},
    semantics="index16 is supplier-reported any reopen, not count/actor/verified exchange queue.122/122 matched two late2026 archival dates,66zero/56positive/29counts>1. This does not independently certify all older period semantics; preserve supplier-history/late acquisition/survivorship caveats. Qualified exact S-date/cent-price andS-timestamp,NaN unknown,explicit0 retained.",
    fixed_rules={key:old_freeze[key] for key in ("training", "validation", "evaluation", "acceptance")},
    selection="Both arms once. Only passing arm with most validation primary hits enters14months;63 wins challenger ties. BASE52/48 wins ties. No label/date abstention or postresult threshold/feature changes.",
    unsupported_claims="Previous72%/68% and actor/theme causality retracted; not evidence for this hypothesis. Other prior failed models remain failed, not rerun.",
    bindings=bindings, peer_proposal_sha256=model.base.sha(proposal),
    commands={arm:f"D:/python/python.exe first_board_joint_reopen_model.py train --arm {arm} --trainer {model.OWNERS[arm]}" for arm in model.ARMS},
    evaluation_owner="Codex once after both validation reports; winner only, no BASE or S2 refit",
    acknowledgment="Gemini may ONLY change status toAGREED_BEFORE_NEW_FITS and addagreed_at/acknowledgment_note; otherwise separate objection, do notfit.",
    prospective_deployment_allowed=False, actual_trades=0, new_forward_results=0)
with (out / "JOINT_FROZEN_PROTOCOL.json").open("x", encoding="utf-8") as file:
    json.dump(freeze, file, ensure_ascii=False, indent=2)
print(json.dumps(preflight, ensure_ascii=True))
