"""Validate new seven-arm view against all prior six-arm values and ranks."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
import relay_research_scorecard as view
from verify_full22662_scorecard import equal_values, check_group

SCORE = Path("research/model_upgrade_validation/ranking_optimization_20261015T173045Z/scorecard")
NEW_ARM = "SOURCE_PATCHED_FEATURE_CLASSIFIER"
FAILED = "FAILED_FIXED_SCREEN_NOT_ADOPTED"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_bindings(receipt):
    for path, digest in receipt.get("source_bindings", {}).items():
        assert sha(path) == digest, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-receipt", required=True)
    args = parser.parse_args()
    start = time.monotonic()
    source_path, source_receipt = Path(args.source), Path(args.source_receipt)
    original = pd.read_csv(source_path, dtype={"code": str}, low_memory=False)
    display_path = SCORE / "source_patched26439_display_status_overlay.csv"
    display_input = pd.read_csv(display_path, dtype={"code": str}, low_memory=False)
    output = SCORE / "final_verified26439_research_view"
    packet = json.loads((output / "complete_scorecard.json").read_text(encoding="utf-8"))
    actual = pd.DataFrame(packet["rows"])
    old_path = SCORE / "final_verified22662_research_view/complete_scorecard.json"
    old = pd.DataFrame(json.loads(old_path.read_text(encoding="utf-8"))["rows"])
    assert view.VERSION == packet["version"]
    assert len(actual) == len(original) == len(display_input) == 26439
    assert actual.groupby("arm").size().eq(3777).all() and actual.arm.nunique() == 7
    assert actual[["date", "code", "arm"]].duplicated().sum() == 0
    for column in original:
        equal_values(display_input[column], original[column])
    for column in display_input:
        equal_values(actual[column], display_input[column])

    key = ["date", "code", "arm"]
    before = old.set_index(key).sort_index()
    remaining = actual[actual.arm.ne(NEW_ARM)].set_index(key).sort_index()
    assert before.index.equals(remaining.index)
    for column in before:
        equal_values(remaining[column], before[column])
    print("original6arm all raw columns, v3 derived fields/ranks and BASE values exact", flush=True)

    new = actual[actual.arm.eq(NEW_ARM)]
    equal_values(new.absolute_event_score, new.score)
    equal_values(new.absolute_event_call, new["call"])
    assert new.research_validation_status.eq(FAILED).all()
    assert not new.event_probability_validated.any()
    assert not new.production_eligible.any() and not new.actionable.any()
    expected_rating = 100 * pd.to_numeric(new.score, errors="coerce")
    equal_values(new.absolute_event_rating_100, expected_rating)
    # The classified head's source score/call is supplied by the frozen routed
    # output, never reconstructed or thresholded by this presentation module.
    groups = 0
    for _, group in new.groupby(["date", "arm"], sort=False):
        check_group(group, "pool"); groups += 1
        for _, subgroup in group.groupby("same_height_group", sort=False):
            check_group(subgroup, "same_height"); groups += 1
        members = group[group.original_non_one_word_pool_member]
        if len(members):
            check_group(members, "non_one_word_pool"); groups += 1
            for _, subgroup in members.groupby("same_height_group", sort=False):
                check_group(subgroup, "non_one_word_same_height"); groups += 1
    pair = actual[actual.arm.eq("PAIRWISE_DAY_RANK")]
    assert pair.absolute_event_score.isna().all() and pair.absolute_event_call.isna().all()
    assert pair.absolute_event_rating_100.isna().all()
    assert pair.recovered_classifier_research_call.isna().all()
    assert pair.research_validation_status.eq(FAILED).all()
    print(f"new head3777 and its {groups} complete ranking groups checked", flush=True)

    altered = display_input.copy()
    altered["target"] = np.arange(len(altered)) % 2
    altered["label_valid"] = False
    altered["next_day_high"] = 100000 + np.arange(len(altered))
    altered["actual_profit"] = -10000
    rebuilt = view.build_scorecard(altered, pairwise_validation_status=FAILED)
    for column in [c for c in actual if c not in display_input]:
        equal_values(actual[column], rebuilt[column])
    print("all26439 target/label/future perturbations leave derived score/rank fields exact", flush=True)

    js = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const page=fs.readFileSync(0,'utf8');const nodes={};const clock=Date.now();
const context=vm.createContext({document:{getElementById:k=>(nodes[k]??={value:'',innerHTML:'',textContent:''})}});
vm.runInContext(page.match(/<script>([\s\S]*?)<\/script>/)[1],context);
const count=()=>((nodes.rows.innerHTML.match(/<tr class=/g)||[]).length);
assert.equal(nodes.date.value,'2026-10-15');assert.equal(count(),42);
const initial_ms=Date.now()-clock;
nodes.date.value='';nodes.date.onchange();assert.equal(count(),200);
assert(nodes.count.textContent.includes('筛选 26439 / 完整 26439'));
nodes.last.onclick();assert.equal(count(),39);nodes.first.onclick();
nodes.arm.value='SOURCE_PATCHED_FEATURE_CLASSIFIER';nodes.arm.onchange();
assert(nodes.count.textContent.includes('筛选 3777 / 完整 26439'));
assert(nodes.rows.innerHTML.includes('FAILED_FIXED_SCREEN_NOT_ADOPTED'));
assert(nodes.rows.innerHTML.includes('原模型评分，非概率'));
const rows=vm.runInContext('data',context);
assert.equal(rows.length,26439);const arms={};for(const r of rows){arms[r.arm]=(arms[r.arm]||0)+1;}
assert.equal(Object.keys(arms).length,7);assert(Object.values(arms).every(v=>v===3777));
let reachable=0;for(let p=0;p<Math.ceil(3777/200);p++){if(p)nodes.next.onclick();reachable+=count();}
assert.equal(reachable,3777);
nodes.date.value='2026-10-15';nodes.date.onchange();assert.equal(count(),6);
nodes.arm.value='PAIRWISE_DAY_RANK';nodes.arm.onchange();assert.equal(count(),6);
assert(rows.filter(r=>r.arm==='PAIRWISE_DAY_RANK').every(r=>r.absolute_event_score===null&&r.absolute_event_call===null&&r.absolute_event_rating_100===null));
console.log(JSON.stringify({initial_rows:42,complete_rows:rows.length,rows_per_arm:arms,maximum_DOM_rows:200,
last_full_page_rows:39,all_newhead3777_rows_reachable:true,latest_each_arm_rows:6,
initial_script_execution_ms:initial_ms,total_script_execution_ms:Date.now()-clock,pairwise_absolute_fields_empty:true}));
"""
    run = subprocess.run([shutil.which("node"), "-e", js],
        input=(output / "complete_scorecard.html").read_text(encoding="utf-8"),
        text=True, encoding="utf-8", capture_output=True, timeout=60)
    assert run.returncode == 0, run.stderr
    html_result = json.loads(run.stdout)

    overlay_manifest_path = SCORE / "source_patched26439_display_status_overlay_manifest.json"
    overlay_manifest = json.loads(overlay_manifest_path.read_text(encoding="utf-8"))
    verify_bindings(overlay_manifest)
    assert overlay_manifest["display_overlay_csv_sha256"] == sha(display_path)
    verify_bindings(json.loads(source_receipt.read_text(encoding="utf-8")))
    for folder in [output, SCORE / "final_verified22662_research_view"]:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        verify_bindings(manifest)
        for filename, digest in manifest["artifact_hashes"].items():
            assert sha(folder / filename) == digest
    previous_verification = SCORE / "final_verified22662_independent_verification.json"
    verify_bindings(json.loads(previous_verification.read_text(encoding="utf-8")))

    receipt = {
        "view_version": view.VERSION, "scoring_module_changed": False, "rows": len(actual),
        "rows_per_arm": actual.groupby("arm").size().to_dict(),
        "original6arm_all_original_and_v3_derived_columns_exact": True,
        "BASE_original_scores_calls_ranks_and_rating100_unchanged": True,
        "new_classifier_score_and_call_exactly_supplied_qualified_A_routed_output": True,
        "new_classifier_validation_status": FAILED,
        "new_classifier_independent_complete_ranking_groups": groups,
        "PAIRWISE_absolute_fields_and_recovered_directions_remain_empty": True,
        "all26439_target_label_future_changes_leave_derived_fields_exact": True,
        "HTML_execution": html_result, "actual_browser_visual_verified": False,
        "browser_limitation": "Existing browser file-protocol denial remains; no browser or local-server bypass attempted. Node local script and DOM-function checks only.",
        "old22662_outputs_and_receipts_preserved": True, "new_fits_by_scorecard": 0,
        "existing_author_review_gate_unchanged": True,
        "elapsed_seconds": time.monotonic() - start,
        "source_bindings": {str(p.resolve()): sha(p) for p in [source_path, source_receipt, display_path,
            overlay_manifest_path, Path(__file__), SCORE / "verify_full22662_scorecard.py",
            Path(view.__file__), Path("tests/test_relay_research_scorecard.py"),
            SCORE / "test_results_28_with_source_overlay.xml", output / "manifest.json", old_path,
            previous_verification]},
    }
    target = SCORE / "final_verified26439_independent_verification.json"
    assert not target.exists()
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ["rows", "rows_per_arm", "new_classifier_independent_complete_ranking_groups", "elapsed_seconds"]}))


if __name__ == "__main__":
    main()
