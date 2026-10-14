"""Synthetic saved-record contracts only; no real fitting or market outcomes."""
from copy import deepcopy
from datetime import date, timedelta

import pytest

from model_upgrade_metrics import (
    MARKET_CLASSES, acceptance, binary_metrics, block_interval,
    classification_metrics, raw_market_score_metrics, rollback_check,
    summarize_records,
)


@pytest.fixture
def calendar():
    # Deliberately synthetic sessions; no exchange-calendar assertion is made.
    first = date(2030, 1, 1)
    return [(first+timedelta(days=i)).isoformat() for i in range(160)]


@pytest.fixture
def protocol():
    return {"forward_sessions": 120, "min_market_pairs": 108,
            "min_class_support": 20, "min_relay_pairs": 60,
            "min_coverage": .9, "min_improvement": .03,
            "bootstrap": {"draws": 2000, "seed": 20261007,
                          "block_sessions": 5, "tails": [.00625, .99375]},
            "preferred_arm": "C1",
            "rollback": {"sessions": 60, "primary_drop": .05, "recall_drop": .10}}


def market_records(calendar, n=120, forward=True):
    result = []
    for i in range(n):
        actual = MARKET_CLASSES[i % 3]
        baseline = actual if i % 2 else MARKET_CLASSES[(i % 3+1) % 3]
        arms = {}
        for arm, prediction in (("BASE", baseline), ("C1", actual), ("C2", actual)):
            arms[arm] = {"status": "KNOWN", "prediction": prediction,
                         "raw_scores": {k: .8 if k == prediction else .1 for k in MARKET_CLASSES}}
        result.append({"module": "market", "signal_date": calendar[i], "target_date": calendar[i+1],
                       "arms": arms, "outcomes": {"market_target": actual},
                       "naive": {"majority": "CONTINUE", "persistence": "CONTINUE"},
                       "forward": forward, "target_matured": True})
    return result


def relay_records(calendar, n=120):
    result = []
    for i in range(n):
        arms = {}
        for arm in ("BASE", "C1", "C2"):
            positive_score = .4 if arm == "BASE" and i % 2 == 0 else .9
            negative_score = .6 if arm == "BASE" and i % 2 == 0 else .1
            arms[arm] = {"status": "KNOWN", "candidates": [
                {"code": "000001", "boards": 2, "actual_one_word_daily": 0,
                 "raw_score": positive_score, "predicted_positive": positive_score > .5},
                {"code": "000002", "boards": 3, "actual_one_word_daily": 0,
                 "raw_score": negative_score, "predicted_positive": negative_score > .5},
            ]}
        result.append({"module": "relay", "signal_date": calendar[i], "target_date": calendar[i+1],
                       "original_codes": ["000001", "000002"], "arms": arms,
                       "outcomes": {"candidates": {"000001": True, "000002": False}},
                       "naive": {"highest_nonword": "000002"}, "forward": True,
                       "target_matured": True})
    return result


def test_unknowns_keep_registration_and_complete_recall():
    result = binary_metrics([1, 1, 0, None], [1, None, 1, 0])
    assert result["registered_rows"] == 4
    assert result["known_outcome_rows"] == 3
    assert result["scored_rows"] == 2
    assert result["scoring_coverage"] == .5
    assert result["complete_positive_recall"] == .5
    assert result["scored_positive_recall"] == 1
    assert result["FPR"] == 1
    assert result["unknown_outcome_rows"] == 1
    assert result["unknown_prediction_rows"] == 1


def test_missing_market_class_is_not_observed_class_macro():
    result = classification_metrics(["IMPROVE", "CONTINUE"], ["IMPROVE", "CONTINUE"], MARKET_CLASSES)
    assert result["accuracy"] == 1
    assert result["macro_f1"] is None
    assert result["balanced_accuracy"] is None
    assert result["by_class"]["WORSEN"]["complete_recall"] is None


def test_multiclass_raw_brier_definition_and_invalid_scores():
    good = {"IMPROVE": .8, "CONTINUE": .1, "WORSEN": .1}
    result = raw_market_score_metrics(["IMPROVE", "CONTINUE", None], [good, {"IMPROVE": .8}, good])
    assert result["raw_score_rows"] == 1
    assert result["raw_score_Brier"] == pytest.approx(.06)
    assert result["raw_score_coverage"] == pytest.approx(1/3)
    assert result["probability_calibration"] == "NOT_CALIBRATED_INTERNAL_SCORE"


def test_official_grid_gap_is_unknown_and_bad_target_rejected(calendar):
    records = market_records(calendar, 3)
    records.pop(1)
    summary = summarize_records("market", records, calendar)
    assert summary["registered_dates"] == 3
    assert summary["source_record_dates"] == 2
    assert summary["arms"]["BASE"]["scored_rows"] == 2
    records[0]["target_date"] = calendar[2]
    with pytest.raises(ValueError, match="exact next"):
        summarize_records("market", records, calendar)


def test_family4_interval_keeps_unknown_grid_and_is_deterministic():
    values = [1.0, None, 1.0, None, 1.0]
    assert block_interval(values) == [1.0, 1.0]
    assert block_interval([None, None]) is None
    assert block_interval([1., 0., None, -1.]) == block_interval([1., 0., None, -1.])


def test_market_true_forward_fixed_node_and_predeclared_priority(calendar, protocol):
    result = acceptance("market", market_records(calendar), protocol, calendar)
    assert result["status"] == "MATURE_FIXED_NODE"
    assert result["selected_arm"] == "C1"
    assert result["gates"]["C1"]["passed"]
    assert result["gates"]["C2"]["passed"]
    assert result["gates"]["C1"]["comparison"]["paired_dates"] == 120
    assert result["automatic_deployment_performed"] is False


def test_historical_good_accuracy_cannot_pass_future_gate(calendar, protocol):
    result = acceptance("market", market_records(calendar, forward=False), protocol, calendar)
    assert result["status"] == "NOT_STARTED"
    assert result["selected_arm"] == "BASE"
    assert not result["upgrade_eligible"]


@pytest.mark.parametrize("mode", ["missing_last", "not_closed", "empty_outcomes", "short_calendar"])
def test_incomplete_or_unmature_window_cannot_upgrade(calendar, protocol, mode):
    records = market_records(calendar)
    if mode == "missing_last":
        records.pop()
    elif mode == "not_closed":
        records[-1]["target_matured"] = False
    elif mode == "empty_outcomes":
        for r in records:
            r["outcomes"] = {}
            r.pop("target_matured")
    else:
        calendar = calendar[:120]
        records = records[:-1]
    result = acceptance("market", records, protocol, calendar)
    assert result["status"] == "NOT_MATURE"
    assert result["selected_arm"] == "BASE"


def test_naive_comparison_unknown_is_not_a_free_pass(calendar, protocol):
    records = market_records(calendar)
    records[0]["naive"]["majority"] = None
    result = acceptance("market", records, protocol, calendar)
    assert not result["gates"]["C1"]["checks"]["all_naive_baselines_not_lower"]
    assert result["selected_arm"] == "BASE"


def test_one_bad_fixed_window_cannot_hide_in_overall_gain(calendar, protocol):
    records = market_records(calendar)
    for i, r in enumerate(records[:40]):
        actual = r["outcomes"]["market_target"]
        r["arms"]["C1"]["prediction"] = MARKET_CLASSES[(MARKET_CLASSES.index(actual)+1) % 3]
    result = acceptance("market", records, protocol, calendar)
    assert not result["gates"]["C1"]["checks"]["each_fixed_40_session_window_not_lower"]
    assert result["selected_arm"] == "C2"


def test_relay_ranking_does_not_make_negative_top1_positive(calendar):
    records = relay_records(calendar, 1)
    for arm in records[0]["arms"].values():
        arm["candidates"][0]["raw_score"] = .4
        arm["candidates"][0]["predicted_positive"] = False
        arm["candidates"][1]["raw_score"] = .3
        arm["candidates"][1]["predicted_positive"] = False
    result = summarize_records("relay", records, calendar)
    assert result["arms"]["BASE"]["primary_accuracy"] == 1
    assert result["arms"]["BASE"]["positive_top1_called_dates"] == 0
    assert result["arms"]["BASE"]["positive_top1_precision"] is None


def test_tie_uses_current_v280_code_rule_even_at_different_boards(calendar):
    records = relay_records(calendar, 1)
    for arm in records[0]["arms"].values():
        for c in arm["candidates"]:
            c["raw_score"] = .5
            c["predicted_positive"] = False
    result = summarize_records("relay", records, calendar)
    assert result["arms"]["BASE"]["rank_successes"] == 1


def test_unknown_identity_does_not_complete_group_or_disappear(calendar):
    records = relay_records(calendar, 2)
    for arm in records[1]["arms"].values():
        arm["candidates"][1]["actual_one_word_daily"] = None
    result = summarize_records("relay", records, calendar)
    assert result["confirmed_nonword_dates"] == 1
    assert result["unknown_eligibility_dates"] == 1
    assert result["effective_eligible_dates"] == 1
    assert result["primary_coverage_denominator"] == 2
    assert result["arms"]["C1"]["primary_scoring_coverage"] == .5


def test_incomplete_original_outcome_does_not_rank_around_missing_code(calendar):
    records = relay_records(calendar, 1)
    records[0]["outcomes"]["candidates"]["000002"] = None
    result = summarize_records("relay", records, calendar)
    assert result["confirmed_nonword_dates"] == 1
    assert result["effective_eligible_dates"] == 0
    assert result["arms"]["C1"]["rank_scored_dates"] == 0
    assert result["arms"]["C1"]["all_candidate"]["known_outcome_rows"] == 1


def test_no_nonword_day_is_not_a_failure(calendar):
    records = relay_records(calendar, 2)
    for arm in records[1]["arms"].values():
        for c in arm["candidates"]:
            c["actual_one_word_daily"] = 1
    result = summarize_records("relay", records, calendar)
    assert result["known_no_nonword_dates"] == 1
    assert result["primary_coverage_denominator"] == 1
    assert result["arms"]["C1"]["primary_accuracy"] == 1
    assert result["arms"]["C1"]["primary_scoring_coverage"] == 1


def test_known_forecast_cannot_drop_an_original_candidate(calendar):
    records = relay_records(calendar, 1)
    records[0]["arms"]["C1"]["candidates"].pop()
    with pytest.raises(ValueError, match="complete original"):
        summarize_records("relay", records, calendar)


def test_relay_forward_candidate_risk_and_primary_gate(calendar, protocol):
    result = acceptance("relay", relay_records(calendar), protocol, calendar)
    assert result["selected_arm"] == "C1"
    assert result["gates"]["C1"]["passed"]
    assert result["summary"]["arms"]["C1"]["all_candidate"]["accuracy"] == 1
    assert result["summary"]["arms"]["C1"]["raw_scores"]["raw_score_Brier"] == pytest.approx(.01)
    assert result["summary"]["arms"]["C1"]["by_board"]["two_board"]["all_candidate"]["registered_rows"] == 120


def test_relay_call_coverage_risk_can_block_improved_ranking(calendar, protocol):
    records = relay_records(calendar)
    for r in records:
        for c in r["arms"]["C1"]["candidates"]:
            c["predicted_positive"] = False
    result = acceptance("relay", records, protocol, calendar)
    assert not result["gates"]["C1"]["checks"]["positive_top1_call_coverage_not_lower"]
    assert not result["gates"]["C1"]["passed"]
    assert result["selected_arm"] == "C2"


def test_better_relay_rank_cannot_hide_worse_raw_brier(calendar, protocol):
    records = relay_records(calendar)
    for r in records:
        candidates = r["arms"]["C1"]["candidates"]
        candidates[0]["raw_score"] = .51
        candidates[0]["predicted_positive"] = True
        candidates[1]["raw_score"] = .5
        candidates[1]["predicted_positive"] = False
    result = acceptance("relay", records, protocol, calendar)
    assert result["summary"]["arms"]["C1"]["primary_accuracy"] == 1
    assert not result["gates"]["C1"]["checks"]["raw_score_Brier_not_worse"]
    assert not result["gates"]["C1"]["passed"]


def test_absent_market_classes_block_perfect_accuracy_admission(calendar, protocol):
    records = market_records(calendar)
    for r in records:
        r["outcomes"]["market_target"] = "CONTINUE"
        r["arms"]["BASE"]["prediction"] = "IMPROVE"
        r["arms"]["C1"]["prediction"] = "CONTINUE"
        r["arms"]["C2"]["prediction"] = "CONTINUE"
    result = acceptance("market", records, protocol, calendar)
    assert result["summary"]["arms"]["C1"]["accuracy"] == 1
    assert not result["gates"]["C1"]["checks"]["each_class_support_at_least20"]
    assert result["selected_arm"] == "BASE"


def test_market_fixed_60_node_rolls_back_adverse_recall(calendar, protocol):
    records = market_records(calendar, 60)
    for r in records:
        actual = r["outcomes"]["market_target"]
        r["arms"]["BASE"]["prediction"] = actual
        r["arms"]["C1"]["prediction"] = "CONTINUE" if actual == "WORSEN" else actual
    result = rollback_check("market", records, protocol, calendar)
    assert result["rollback"]
    assert result["selected_arm"] == "BASE"
    assert result["triggers"]["recall_drop_at_least10pp"]
    assert result["recall_difference"] == -1


def test_relay_fixed_60_node_rolls_back_complete_recall(calendar, protocol):
    records = relay_records(calendar, 60)
    for r in records:
        r["arms"]["C1"] = deepcopy(r["arms"]["BASE"])
        for c in r["arms"]["C1"]["candidates"]:
            c["predicted_positive"] = False
    result = rollback_check("relay", records, protocol, calendar)
    assert result["rollback"]
    assert result["triggers"]["recall_drop_at_least10pp"]


def test_rollback_missing_sources_is_hold_not_model_failure(calendar, protocol):
    records = market_records(calendar, 60)
    for r in records:
        r["outcomes"] = {}
    result = rollback_check("market", records, protocol, calendar)
    assert result["checks"]["mature_fixed_60_node"]
    assert result["comparison"]["paired_dates"] == 0
    assert not result["rollback"]
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["hold_due_to_missing_evidence"]


def test_protocol_threshold_search_and_duplicate_dates_rejected(calendar, protocol):
    bad = deepcopy(protocol)
    bad["min_improvement"] = .01
    with pytest.raises(ValueError, match="approved fixed gate"):
        acceptance("market", market_records(calendar), bad, calendar)
    records = market_records(calendar, 1)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_records("market", records+records, calendar)


def test_market_rollback_any_class_complete_recall_without_primary_drop(calendar, protocol):
    records = market_records(calendar, 60)
    for r in records:
        actual = r['outcomes']['market_target']
        r['arms']['BASE']['prediction'] = actual
        r['arms']['C1']['prediction'] = actual
    improve = [r for r in records if r['outcomes']['market_target'] == 'IMPROVE']
    continuing = [r for r in records if r['outcomes']['market_target'] == 'CONTINUE']
    for r in improve[:2]:
        r['arms']['C1']['prediction'] = 'CONTINUE'
    for r in continuing[:2]:
        r['arms']['BASE']['prediction'] = 'IMPROVE'
    result = rollback_check('market', records, protocol, calendar)
    assert result['comparison']['primary_difference'] == 0
    assert not result['triggers']['primary_drop_at_least5pp']
    assert result['recall_differences']['WORSEN'] == 0
    assert result['recall_differences']['IMPROVE'] == pytest.approx(-.1)
    assert result['recall_differences']['CONTINUE'] == pytest.approx(.1)
    assert result['triggers']['recall_drop_at_least10pp']
    assert result['rollback'] and result['selected_arm'] == 'BASE'


def test_missing_120th_record_real_closed_certificate_still_matures(calendar, protocol):
    records = market_records(calendar)
    records.pop()
    certified = deepcopy(protocol)
    certified['closed_through_date'] = calendar[120]
    result = acceptance('market', records, certified, calendar)
    assert result['status'] == 'MATURE_FIXED_NODE'
    assert result['summary']['registered_dates'] == 120
    assert result['summary']['source_record_dates'] == 119
    assert result['gates']['C1']['comparison']['paired_dates'] == 119
    assert result['gates']['C1']['comparison']['pair_coverage'] == pytest.approx(119/120)
    assert result['gates']['C1']['checks']['mature_fixed_120_node']


def test_sparse_true_nonword_60_day_primary_deterioration_still_rolls_back(calendar, protocol):
    records = relay_records(calendar, 60)
    for i, record in enumerate(records):
        for arm in record['arms'].values():
            for candidate in arm['candidates']:
                candidate['actual_one_word_daily'] = 0 if i < 4 else None if i < 9 else 1
        if i < 4:
            rows = record['arms']['BASE']['candidates']
            rows[0].update(raw_score=.9, predicted_positive=True)
            rows[1].update(raw_score=.1, predicted_positive=False)
        record['arms']['C1'] = deepcopy(record['arms']['BASE'])
    records[0]['arms']['C1']['candidates'][0].update(raw_score=.1, predicted_positive=False)
    records[0]['arms']['C1']['candidates'][1].update(raw_score=.9, predicted_positive=True)
    result = rollback_check('relay', records, protocol, calendar)
    assert result['checks']['mature_fixed_60_node']
    assert result['comparison']['paired_dates'] == 4
    assert result['comparison']['pair_coverage'] == pytest.approx(4/9)
    assert result['comparison']['primary_difference'] == pytest.approx(-.25)
    assert result['triggers']['primary_drop_at_least5pp']
    assert result['rollback'] and result['selected_arm'] == 'BASE'
