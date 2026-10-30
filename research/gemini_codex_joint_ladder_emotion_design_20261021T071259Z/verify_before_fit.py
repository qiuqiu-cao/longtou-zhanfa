"""Read-only full-cohort, member and nested clock verification; no estimators fitted."""
import json
from pathlib import Path
import numpy as np
import first_board_joint_ladder_emotion_model as m

out = Path(__file__).resolve().parent
rows = m.load_rows()
assert len(rows) == 28503 and rows.signal_date.nunique() == 668
assert rows[m.stage.TARGET].isna().sum() == 126
assert len(m.ARMS[m.A]) == len(set(m.ARMS[m.A])) == 80
assert len(m.ARMS[m.B]) == len(set(m.ARMS[m.B])) == 81
assert set(m.ARMS[m.B])-set(m.ARMS[m.A]) == {m.macro.FIELD}
plan = json.loads((out / "EXACT_FIT_PLAN_BEFORE_ANY_FIT.json").read_text("utf-8"))
assert m.fit_plan(rows) == plan
daily, sessions = m.macro_data(rows)
assert daily.members.sum() == 28503 and (daily.members-daily.known).sum() == 126
assert daily[m.macro.TARGET].notna().sum() == daily.outcome_coverage.ge(.95).sum()
windows = []
for group, item in plan["outer_windows"].items():
    boundary = "2025-01-01" if group == "validation" else group + "-01"
    pos = int(np.searchsorted(sessions, boundary))
    train = m.members.train_members(rows, boundary, None if group == "validation" else pos-252)
    m.verify_members(train, group)
    cut, inner, cal = m.macro.calibration_parts(train)
    assert cut == item["calibration_cut"] and train.label_end.lt(boundary).all()
    assert inner.label_end.lt(cut).all() and cal.signal_date.nunique() == 42
    assert m.macro.mature_members(daily, cut, sessions).label_end.lt(cut).all()
    windows.append(dict(group=group, rows=len(train), cal_dates=42, cut=cut,
                        exact_original_members_and_labels=True, all_target_clocks_strict=True))
val = m.risk.attach_frozen(rows, m.prior.REFERENCE / "validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
baseline = m.risk.screen_metric(val)
non = m.risk.screen_metric(val[val.one_price.eq(0)])
assert baseline["hits"] == 52 and non["hits"] == 48
files = [Path(m.__file__), Path(m.macro.__file__), Path(m.features.__file__),
         Path(m.reopen.__file__), Path(m.prior.__file__), Path(m.stage.__file__), Path(m.base.__file__),
         m.prior.FACTS, m.prior.SCREEN, m.NEW_FIELDS_FILE, m.DAILY_FILE,
         m.prior.QUAL / "COMPLETE28503_NEW_FIELDS_NO_LABELS.csv", m.reopen.NEW_FIELDS_FILE,
         out / "LADDER_EMOTION_QUALIFICATION.json", out / "EXACT_FIT_PLAN_BEFORE_ANY_FIT.json"]
report = dict(at=m.base.now(), status="READ_ONLY_PREFLIGHT_PASS", research_fits=0,
    full_rows=28503, full_dates=668, unknown_outcomes=126, fields={k: len(v) for k,v in m.ARMS.items()},
    windows=windows, baseline=baseline, baseline_non_one_price=non,
    macro_target_qualified_dates=int(daily[m.macro.TARGET].notna().sum()),
    macro_target_unknown_dates=int(daily[m.macro.TARGET].isna().sum()),
    macro_target_is_promotion_rate_proxy_not_emotion_gold=True,
    old_partial_corporate_action_overlay_not_applied_to_either_side=True,
    historical_universe_and_nominal_price_limitations_retained=True,
    tests="11 passed: real source-prefix feature checks earlier plus 6 new clock/coverage/calibration tests; synthetic estimators are test doubles",
    bindings={str(p.resolve()): m.base.sha(p) for p in files})
m.base.write_json(out / "PRE_FREEZE_VERIFICATION.json", report)
print(json.dumps(m.base.clean({k:v for k,v in report.items() if k not in ("bindings","windows","baseline","baseline_non_one_price")}), ensure_ascii=True), flush=True)
