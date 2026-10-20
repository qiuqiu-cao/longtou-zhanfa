"""Independent display coverage/rank checks; never fits or changes predictions."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
import relay_research_scorecard as view

ROOT = Path("research/model_upgrade_validation/ranking_optimization_20261015T173045Z")
SCORE = ROOT / "scorecard"
SOURCE = ROOT / "fixed_comparison/all22662_predictions_BEFORE_EVALUATION.csv"
OUTPUT = SCORE / "full22662_research_view"
STATUS = "FAILED_FIXED_SCREEN_NOT_ADOPTED"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal_values(left, right):
    pd.testing.assert_series_equal(left.reset_index(drop=True), right.reset_index(drop=True),
                                   check_dtype=False, check_names=False, rtol=0, atol=0)


def check_group(group, prefix):
    # Independent implementation from records; no call to the view rank helper.
    values = [(idx, float(row.research_ranking_score), row.code)
              for idx, row in group.iterrows() if pd.notna(row.research_ranking_score)]
    values.sort(key=lambda item: (-item[1], item[2]))
    unique = sorted({value for _, value, _ in values}, reverse=True)
    counts = {value: sum(v == value for _, v, _ in values) for value in unique}
    first_rank = {value: 1 + sum(v > value for _, v, _ in values) for value in unique}
    for position, (idx, value, _) in enumerate(values, 1):
        row = group.loc[idx]
        assert row[prefix + "_position"] == position
        assert row[prefix + "_rank"] == first_rank[value]
        assert row[prefix + "_tie_layer"] == unique.index(value) + 1
        assert row[prefix + "_tie_count"] == counts[value]
        assert row[prefix + "_relative_first"] == (value == unique[0])
        assert row[prefix + "_gap_to_first"] == unique[0] - value
    assert group[prefix + "_pool_size"].eq(len(group)).all()
    assert group[prefix + "_scored_count"].eq(len(values)).all()
    assert group[prefix + "_unknown_score_count"].eq(len(group) - len(values)).all()
    missing = group.research_ranking_score.isna()
    assert group.loc[missing, prefix + "_rank"].isna().all()
    assert group.loc[missing, prefix + "_rank_percentile"].isna().all()
    if len(group) == 1:
        assert group[prefix + "_sole_candidate_position"].eq(1).all()
    else:
        assert group[prefix + "_sole_candidate_position"].isna().all()


def main():
    started = time.monotonic()
    source = pd.read_csv(SOURCE, dtype={"code": str}, low_memory=False)
    packet = json.loads((OUTPUT / "complete_scorecard.json").read_text(encoding="utf-8"))
    table = pd.DataFrame(packet["rows"])
    assert len(source) == len(table) == 22662
    assert source.groupby("arm").size().eq(3777).all() and source.arm.nunique() == 6
    assert not source.duplicated(["date", "code", "arm"]).any()
    for column in source:
        equal_values(table[column], source[column])
    print("all22662 supplied inputs preserved; each arm3777", flush=True)

    baseline = source[source.arm.eq("BASE")].set_index(["date", "code"])
    base_reference = baseline.loc[pd.MultiIndex.from_frame(table[["date", "code"]])]
    equal_values(table.base_score_input_original, base_reference.score)
    equal_values(table.base_call_input_original, base_reference["call"])
    np.testing.assert_array_equal(table.base_rating_100.to_numpy(), 100 * base_reference.score.to_numpy())
    assert not table.event_probability_validated.any()
    assert not table.actionable.any() and not table.production_eligible.any()

    pair = table[table.arm.eq("PAIRWISE_DAY_RANK")]
    assert pair.absolute_event_score.isna().all() and pair.absolute_event_call.isna().all()
    assert pair.absolute_event_rating_100.isna().all()
    equal_values(pair.research_ranking_score, pair.ranking_score)
    equal_values(pair.frozen_classifier_absolute_score, pair.absolute_classifier_score)
    equal_values(pair.frozen_classifier_absolute_call, pair.absolute_classifier_call)
    assert pair.research_validation_status.eq(STATUS).all()
    no_preference = pair.ranking_score.isna()
    assert pair.loc[no_preference, "research_ranking_score"].isna().all()
    assert pair.loc[no_preference, "pool_rank"].isna().all()

    groups = 0
    for _, group in table.groupby(["date", "arm"], sort=False):
        check_group(group, "pool"); groups += 1
        for _, same in group.groupby("same_height_group", sort=False):
            check_group(same, "same_height"); groups += 1
        members = group[group.original_non_one_word_pool_member]
        if not members.empty:
            check_group(members, "non_one_word_pool"); groups += 1
            for _, same in members.groupby("same_height_group", sort=False):
                check_group(same, "non_one_word_same_height"); groups += 1
    print(f"independently checked {groups} complete ranking groups", flush=True)

    changed = source.copy()
    changed["target"] = np.arange(len(source)) % 2
    changed["label_valid"] = False
    changed["next_day_close"] = np.arange(len(source))[::-1]
    changed["actual_gain"] = 9999
    rescored = view.build_scorecard(changed, pairwise_validation_status=STATUS)
    derived = [column for column in table if column not in source]
    for column in derived:
        equal_values(table[column], rescored[column])
    print("all22662 target/label/future changes leave derived scores/ranks exact", flush=True)

    js = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const page=fs.readFileSync(0,'utf8');const nodes={};
const clock=Date.now();const context=vm.createContext({document:{getElementById:k=>(nodes[k]??={value:'',innerHTML:'',textContent:''})}});
vm.runInContext(page.match(/<script>([\s\S]*?)<\/script>/)[1],context);
const count=()=>((nodes.rows.innerHTML.match(/<tr class=/g)||[]).length);
assert.equal(nodes.date.value,'2026-10-15');assert.equal(count(),36);
nodes.date.value='';nodes.date.onchange();assert.equal(count(),200);
assert(nodes.count.textContent.includes('筛选 22662 / 完整 22662'));
nodes.last.onclick();assert.equal(count(),62);
nodes.first.onclick();assert.equal(count(),200);
nodes.arm.value='BASE';nodes.arm.onchange();assert.equal(count(),200);
assert(nodes.count.textContent.includes('筛选 3777 / 完整 22662'));
nodes.arm.value='PAIRWISE_DAY_RANK';nodes.arm.onchange();assert.equal(count(),200);
assert(nodes.rows.innerHTML.includes('FAILED_FIXED_SCREEN_NOT_ADOPTED'));
assert(nodes.rows.innerHTML.includes('FROZEN_CORRECTED_CAUSAL_CLASSIFIER_APPENDIX_NOT_PAIRWISE_DIRECTION'));
assert(nodes.rows.innerHTML.includes('非概率'));
const rows=vm.runInContext('data',context);assert.equal(rows.length,22662);
const armCounts={};for(const row of rows){armCounts[row.arm]=(armCounts[row.arm]||0)+1;}
assert(Object.values(armCounts).every(x=>x===3777));
const coverage=[];let filtered=rows.filter(x=>x.arm==='PAIRWISE_DAY_RANK');
for(let pageIndex=0;pageIndex<Math.ceil(filtered.length/200);pageIndex++){
  if(pageIndex)nodes.next.onclick();
  let shown=nodes.rows.innerHTML;let matches=[...shown.matchAll(/<td>(\d{6}) /g)].map(x=>x[1]);
  assert.equal(matches.length,Math.min(200,filtered.length-pageIndex*200));coverage.push(...matches);
}
assert.equal(coverage.length,3777);
console.log(JSON.stringify({initial_rows:36,complete_rows:rows.length,rows_per_arm:armCounts,
 maximum_DOM_rows:200,last_full_page_rows:62,all3777_pairwise_rows_reachable:true,elapsed_ms:Date.now()-clock}));
"""
    node = shutil.which("node")
    assert node, "local Node required to execute full table filters"
    page = (OUTPUT / "complete_scorecard.html").read_text(encoding="utf-8")
    run = subprocess.run([node, "-e", js], input=page, text=True, encoding="utf-8", capture_output=True, timeout=60)
    assert run.returncode == 0, run.stderr
    html_result = json.loads(run.stdout)
    print("full HTML default36/pagination/complete filters executed", flush=True)

    prior_manifest_path = SCORE / "archive81_research_view/manifest.json"
    prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    preservation = json.loads((SCORE / "source_before_pairwise_hardening/preservation.json").read_text(encoding="utf-8"))
    for row in preservation["source_preimages"]:
        assert sha(row["preserved_path"]) == row["sha256"]
    old_module = SCORE / "source_before_pairwise_hardening/relay_research_scorecard.py"
    assert sha(old_module) == prior_manifest["source_bindings"][str(Path(view.__file__).resolve())]
    for filename, digest in prior_manifest["artifact_hashes"].items():
        assert sha(SCORE / "archive81_research_view" / filename) == digest

    new_archive = SCORE / "archive405_after_pairwise_hardening"
    view.write_scorecard(pd.read_csv(SCORE / "archive81_scores_all5arms.csv", dtype={"code": str}, low_memory=False),
                         new_archive, SCORE / "archive81_scores_all5arms.csv")
    for folder in [OUTPUT, new_archive]:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        for path, digest in manifest["source_bindings"].items():
            assert sha(path) == digest
        for filename, digest in manifest["artifact_hashes"].items():
            assert sha(folder / filename) == digest
    receipt = {
        "view_version": view.VERSION, "input_rows": len(source), "output_rows": len(table),
        "unique_complete_candidates": len(source[["date", "code"]].drop_duplicates()),
        "rows_per_arm": source.groupby("arm").size().to_dict(),
        "all_original_input_columns_exact": True, "all_BASE_original_scores_calls_exact": True,
        "BASE_rating100_exactly_100_times_original_score_not_probability": True,
        "independent_complete_groups_checked": groups,
        "all22662_target_label_future_changes_leave_derived_columns_exact": True,
        "pairwise_no_absolute_event_score_rating_or_direction": True,
        "pairwise_missing_preference_never_filled_from_classifier": True,
        "pairwise_frozen_classifier_appendix_separate": True,
        "pairwise_failed_research_status_rows": len(pair),
        "pairwise_no_preference_rows": int(no_preference.sum()),
        "pairwise_full_pool_singleton_structural_rows": int(pair.sole_candidate_position.eq(1).sum()),
        "HTML_execution": html_result, "HTML_visual_browser_screenshot": False,
        "prior_v1_page_and14_tests_preserved": True,
        "prior_source_claims_verified_against_preserved_preimage_not_current_v2": True,
        "new_fits": 0, "existing_review_gate_unchanged": True,
        "elapsed_seconds": time.monotonic() - started,
        "source_bindings": {str(p.resolve()): sha(p) for p in [SOURCE, Path(view.__file__),
            Path("tests/test_relay_research_scorecard.py"), Path(__file__),
            SCORE / "test_results_24_final.xml", OUTPUT / "manifest.json", new_archive / "manifest.json",
            prior_manifest_path, old_module, SCORE / "source_before_pairwise_hardening/preservation.json"]},
    }
    target = SCORE / "full22662_independent_verification.json"
    assert not target.exists()
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ["input_rows", "unique_complete_candidates", "independent_complete_groups_checked", "pairwise_no_preference_rows", "elapsed_seconds"]}))


if __name__ == "__main__":
    main()
