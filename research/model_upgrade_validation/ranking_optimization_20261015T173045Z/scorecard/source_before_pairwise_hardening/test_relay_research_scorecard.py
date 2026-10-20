import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import relay_research_scorecard as scorecard


def fixture():
    return pd.DataFrame([
        dict(date="2026-10-15", code="000003", name="C", boards=2, arm="BASE", score=.2, ranking_score=.2, call="FAIL", current_price_shape=0, daily_one_word=0, target=1, label_valid=True, rank=3),
        dict(date="2026-10-15", code="000001", name="A", boards=2, arm="BASE", score=.4, ranking_score=.4, call="FAIL", current_price_shape=0, daily_one_word=0, target=0, label_valid=True, rank=1),
        dict(date="2026-10-15", code="000002", name="B", boards=3, arm="BASE", score=.4, ranking_score=.4, call="UNKNOWN", current_price_shape=np.nan, daily_one_word=0, target=np.nan, label_valid=False, rank=2),
        dict(date="2026-10-15", code="000004", name="D", boards=2, arm="BASE", score=np.nan, ranking_score=np.nan, call="UNKNOWN", current_price_shape=np.nan, daily_one_word=np.nan, target=0, label_valid=True, rank=np.nan),
        dict(date="2026-10-15", code="000005", name="E", boards=3, arm="BASE", score=.8, ranking_score=.8, call="PROMOTE", current_price_shape=1, daily_one_word=1, target=1, label_valid=True, rank=4),
    ])


def test_complete_pool_unknown_scores_and_inputs_preserved_without_mutation():
    data = fixture()
    before = data.copy(deep=True)
    result = scorecard.build_scorecard(data)
    pdt.assert_frame_equal(data, before)
    pdt.assert_frame_equal(result[data.columns], before)
    assert len(result) == len(data)
    assert result.pool_pool_size.eq(5).all()
    assert result.pool_scored_count.eq(4).all()
    assert result.pool_unknown_score_count.eq(1).all()
    unknown = result[result.code.eq("000004")].iloc[0]
    assert pd.isna(unknown.pool_rank) and pd.isna(unknown.pool_rank_percentile)
    assert unknown.pool_relative_first is None
    assert pd.isna(unknown.absolute_event_score)
    assert unknown.absolute_event_call == "UNKNOWN"
    assert unknown.price_qualification == "UNKNOWN_PRICE_SHAPE_QUALIFICATION"


def test_targets_labels_and_future_prices_cannot_change_scores_or_ranks():
    data = fixture()
    baseline = scorecard.build_scorecard(data)
    altered = data.copy()
    altered["target"] = [0, 1, 1, 1, 0]
    altered["label_valid"] = ~altered.label_valid
    altered["next_day_close"] = [999, 1, 100, 0, 25]
    altered["actual_gain"] = [100, 500, -100, 1, 10]
    result = scorecard.build_scorecard(altered)
    derived = [field for field in baseline if field not in data.columns]
    pdt.assert_frame_equal(result[derived], baseline[derived])


def test_ties_keep_same_rank_with_code_stable_positions_and_layer():
    data = fixture()
    result = scorecard.build_scorecard(data).set_index("code")
    assert result.loc["000001", "pool_rank"] == result.loc["000002", "pool_rank"] == 2
    assert result.loc["000001", "pool_position"] == 2
    assert result.loc["000002", "pool_position"] == 3
    assert result.loc["000001", "pool_tie_count"] == 2
    assert result.loc["000003", "pool_rank"] == 4
    assert result.loc["000003", "pool_tie_layer"] == 3
    assert result.loc["000001", "pool_gap_to_first"] == pytest.approx(.4)
    assert result.loc["000001", "pool_gap_next_distinct"] == pytest.approx(.2)
    shuffled = scorecard.build_scorecard(data.sample(frac=1, random_state=71)).set_index("code")
    for field in ["pool_rank", "pool_position", "pool_tie_layer", "pool_gap_to_first"]:
        pdt.assert_series_equal(result[field].sort_index(), shuffled[field].sort_index())


def test_relative_first_and_top_negative_are_not_promoted():
    result = scorecard.build_scorecard(fixture()).set_index("code")
    a, b = result.loc["000001"], result.loc["000002"]
    assert a.non_one_word_pool_relative_first and b.non_one_word_pool_relative_first
    assert a.absolute_event_call == "FAIL"
    assert a.attention_state == "OBSERVATION_NEGATIVE_CLASSIFICATION"
    assert b.absolute_event_call == "UNKNOWN"
    assert b.attention_state == "UNKNOWN_ABSTAIN_OR_NO_RECORD"
    assert not a.actionable and not b.actionable


def test_non_one_word_members_and_same_height_counts_are_complete():
    result = scorecard.build_scorecard(fixture()).set_index("code")
    assert result.loc["000001", "non_one_word_pool_pool_size"] == 3
    assert result.loc["000001", "non_one_word_same_height_pool_size"] == 2
    assert result.loc["000002", "non_one_word_same_height_pool_size"] == 1
    assert pd.isna(result.loc["000005", "non_one_word_pool_rank"])
    assert pd.isna(result.loc["000004", "non_one_word_pool_rank"])
    assert result.loc["000002", "original_non_one_word_pool_member"]
    assert result.loc["000002", "price_qualification"] == "UNKNOWN_PRICE_SHAPE_QUALIFICATION"


def test_singleton_has_no_fabricated_percentile_or_neighbor_gap():
    result = scorecard.build_scorecard(fixture().iloc[[0]]).iloc[0]
    assert result.pool_rank == result.pool_position == result.pool_tie_layer == 1
    assert result.pool_singleton and result.pool_small_pool
    assert pd.isna(result.pool_rank_percentile)
    assert pd.isna(result.pool_gap_previous_distinct)
    assert pd.isna(result.pool_gap_next_distinct)


def test_all_missing_scores_keep_all_candidates_without_zero_rank_or_gap():
    data = fixture().iloc[:3].copy()
    data[["score", "ranking_score"]] = np.nan
    result = scorecard.build_scorecard(data)
    assert result.pool_pool_size.eq(3).all()
    assert result.pool_scored_count.eq(0).all()
    assert result.pool_unknown_score_count.eq(3).all()
    assert result.research_ranking_score.isna().all()
    assert result.pool_rank.isna().all() and result.pool_gap_to_first.isna().all()
    assert result.base_score_input_original.isna().all()


def test_pairwise_has_no_absolute_score_or_direction_even_if_input_is_positive():
    data = fixture().iloc[:3].copy()
    data.arm = scorecard.PAIRWISE_ARM
    data["ranking_score"] = [2.4, 8.1, 8.1]
    data["call"] = "PROMOTE"
    result = scorecard.build_scorecard(data)
    assert result.absolute_event_score.isna().all()
    assert result.absolute_event_call.isna().all()
    assert result.call_input_original.eq("PROMOTE").all()
    assert result.research_ranking_score.tolist() == [2.4, 8.1, 8.1]
    assert result.attention_state.eq("OBSERVATION_RELATIVE_PREFERENCE_ONLY").all()
    assert not result.event_probability_validated.any()
    assert result.base_score_input_original.isna().all()


def test_baseline_keeps_its_original_score_direction_and_supplied_ranking():
    baseline = fixture()
    other = baseline.copy()
    other["arm"] = "NEW"
    other["score"] = .9
    other["ranking_score"] = .9
    other["call"] = "PROMOTE"
    result = scorecard.build_scorecard(pd.concat([baseline, other], ignore_index=True))
    new = result[result.arm.eq("NEW")].set_index("code")
    old = result[result.arm.eq("BASE")].set_index("code")
    for left, right in [("base_score_input_original", "score"),
                        ("base_call_input_original", "call"), ("base_input_rank_original", "rank"),
                        ("base_pool_rank", "pool_rank")]:
        pdt.assert_series_equal(new[left].sort_index(), old[right].sort_index(), check_names=False)
    assert not new.production_eligible.any()


def test_unknown_board_height_does_not_mix_with_known_height():
    data = fixture().iloc[:3].copy()
    data.loc[data.index[:2], "boards"] = np.nan
    result = scorecard.build_scorecard(data)
    assert result.same_height_pool_size.tolist() == [2, 2, 1]
    assert result.same_height_known.tolist() == [False, False, True]
    assert result.same_height_group.tolist() == ["UNKNOWN", "UNKNOWN", "3.0"]


def test_duplicates_fail_before_output_and_missing_required_columns_rejected():
    with pytest.raises(ValueError, match="duplicate date/code/arm"):
        scorecard.build_scorecard(pd.concat([fixture(), fixture().iloc[[0]]]))
    with pytest.raises(ValueError, match="date and code"):
        scorecard.build_scorecard(pd.DataFrame({"code": ["000001"]}))


def test_write_new_complete_html_json_csv_and_reject_existing_directory(tmp_path):
    data = fixture()
    data.loc[0, "name"] = "</script><img src=x onerror=alert(1)>"
    target = tmp_path / "view"
    manifest = scorecard.write_scorecard(data, target)
    assert manifest["complete_input_rows"] == manifest["complete_output_rows"] == 5
    rows = json.loads((target / "complete_scorecard.json").read_text(encoding="utf-8"))["rows"]
    assert len(rows) == 5 and rows[3]["pool_rank"] is None
    page = (target / "complete_scorecard.html").read_text(encoding="utf-8")
    assert "<img src=x" not in page
    assert 'id="date"' in page and 'id="arm"' in page
    assert "展开全部字段" in page and "event_probability_validated" in page
    assert "TARGET" not in page  # no target-driven selection formula
    before = (target / "complete_scorecard.csv").read_bytes()
    with pytest.raises(FileExistsError):
        scorecard.write_scorecard(data.iloc[:1], target)
    assert (target / "complete_scorecard.csv").read_bytes() == before


def test_empty_complete_pool_remains_empty():
    result = scorecard.build_scorecard(pd.DataFrame(columns=["date", "code"]))
    assert result.empty
    assert "absolute_event_score" in result and "base_pool_rank" in result


def test_local_html_javascript_filters_complete_rows_and_expands_without_network():
    node = shutil.which("node")
    if node is None:
        pytest.skip("local Node runtime unavailable")
    first = fixture()
    second = first.iloc[:2].copy()
    second.date = "2026-10-16"
    second.arm = "PAIRWISE_DAY_RANK"
    page = scorecard.render_html(scorecard.build_scorecard(pd.concat([first, second], ignore_index=True)))
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const page=fs.readFileSync(0,'utf8');
const script=page.match(/<script>([\s\S]*?)<\/script>/)[1];
const nodes=Object.fromEntries(['date','arm','count','rows'].map(k=>[k,{value:'',innerHTML:'',textContent:''}]));
const context=vm.createContext({document:{getElementById:k=>nodes[k]}});
vm.runInContext(script,context);
assert.equal(nodes.date.value,'2026-10-16');
assert.equal((nodes.rows.innerHTML.match(/<tr class=/g)||[]).length,2);
assert(nodes.count.textContent.includes('2 / 完整 7'));
nodes.date.value='';nodes.date.onchange();
assert.equal((nodes.rows.innerHTML.match(/<tr class=/g)||[]).length,7);
nodes.arm.value='BASE';nodes.arm.onchange();
assert.equal((nodes.rows.innerHTML.match(/<tr class=/g)||[]).length,5);
assert(nodes.rows.innerHTML.includes('展开全部字段'));
assert(nodes.rows.innerHTML.includes('UNKNOWN_PRICE_SHAPE_QUALIFICATION'));
nodes.arm.value='PAIRWISE_DAY_RANK';nodes.arm.onchange();
assert.equal((nodes.rows.innerHTML.match(/<tr class=/g)||[]).length,2);
assert(nodes.rows.innerHTML.includes('PAIRWISE_PREFERENCE_WITHIN_DAY_ONLY'));
console.log('local HTML filtering and full row expansion verified');
"""
    result = subprocess.run([node, "-e", script], input=page, text=True, encoding="utf-8", capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "verified" in result.stdout
