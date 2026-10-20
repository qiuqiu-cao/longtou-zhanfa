"""Verify final qualified overlay view without fitting or modifying any input."""
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
from verify_full22662_scorecard import equal_values

ROOT = Path("research/model_upgrade_validation/ranking_optimization_20261015T173045Z")
SCORE = ROOT / "scorecard"
SOURCE = ROOT / "source_repair/verified_22662_final/all_predictions_with_isolated_qualification.csv"
RECEIPT = SOURCE.parent / "source_qualification_repair_receipt.json"
ORIGINAL = ROOT / "fixed_comparison/all22662_predictions_BEFORE_EVALUATION.csv"
OUTPUT = SCORE / "final_verified22662_research_view"
STATUS = "FAILED_FIXED_SCREEN_NOT_ADOPTED"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    start = time.monotonic()
    source = pd.read_csv(SOURCE, dtype={"code": str}, low_memory=False)
    original = pd.read_csv(ORIGINAL, dtype={"code": str}, low_memory=False)
    actual = pd.DataFrame(json.loads((OUTPUT / "complete_scorecard.json").read_text(encoding="utf-8"))["rows"])
    old = pd.DataFrame(json.loads((SCORE / "full22662_research_view/complete_scorecard.json").read_text(encoding="utf-8"))["rows"])
    assert len(actual) == len(source) == len(original) == 22662
    assert source.groupby("arm").size().eq(3777).all() and source.arm.nunique() == 6
    for column in original:
        equal_values(source[column], original[column])
    for column in source:
        equal_values(actual[column], source[column])
    for column in old:
        equal_values(actual[column], old[column])
    print("all22662 original inputs and all v2 ranks/directions/display columns remain exact", flush=True)
    assert int(actual.source_repair_applied.sum()) == 282
    recomputed = actual.recovered_B_call_basis.eq("SAME_SIGNAL_EXISTING_ABSOLUTE_B_SCORE_STRICT_GT_0P5_RETROSPECTIVE")
    assert recomputed.sum() == 76
    equal_values(actual.loc[recomputed, "recovered_classifier_research_call"], actual.loc[recomputed, "recovered_B_call"])
    assert actual.loc[~recomputed, "recovered_classifier_research_call"].isna().all()
    assert actual.loc[recomputed, "arm"].isin(["CORRECTED_CAUSAL", "CORRECTED_CAUSAL_COST1P5"]).all()
    pair = actual.arm.eq("PAIRWISE_DAY_RANK")
    assert actual.loc[pair, "absolute_event_score"].isna().all()
    assert actual.loc[pair, "absolute_event_call"].isna().all()
    assert actual.loc[pair, "absolute_event_rating_100"].isna().all()
    assert actual.loc[pair, "recovered_classifier_research_call"].isna().all()
    assert actual.loc[pair, "research_validation_status"].eq(STATUS).all()
    assert not actual.qualification_PIT_eligible.any()
    assert not actual.actionable.any() and not actual.production_eligible.any()
    assert not actual.event_probability_validated.any()

    changed = source.copy()
    changed["target"] = np.arange(len(changed)) % 2
    changed["label_valid"] = False
    changed["next_day_high"] = 100000 + np.arange(len(changed))
    changed["actual_profit"] = -99999
    rebuilt = view.build_scorecard(changed, pairwise_validation_status=STATUS)
    for column in [c for c in actual if c not in source]:
        equal_values(actual[column], rebuilt[column])
    print("all22662 qualified-overlay target/label/future perturbations leave derived fields exact", flush=True)

    js = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const page=fs.readFileSync(0,'utf8');const nodes={};const clock=Date.now();
const context=vm.createContext({document:{getElementById:k=>(nodes[k]??={value:'',innerHTML:'',textContent:''})}});
vm.runInContext(page.match(/<script>([\s\S]*?)<\/script>/)[1],context);
const count=()=>((nodes.rows.innerHTML.match(/<tr class=/g)||[]).length);
assert.equal(nodes.date.value,'2026-10-15');assert.equal(count(),36);
const initial_ms=Date.now()-clock;
nodes.date.value='';nodes.date.onchange();assert.equal(count(),200);
assert(nodes.count.textContent.includes('筛选 22662 / 完整 22662'));
nodes.last.onclick();assert.equal(count(),62);nodes.first.onclick();
nodes.arm.value='BASE';nodes.arm.onchange();assert(nodes.count.textContent.includes('筛选 3777 / 完整 22662'));
assert(nodes.rows.innerHTML.includes('原模型评分，非概率'));
const all=vm.runInContext('data',context);
const example=all.find(r=>r.recovered_classifier_research_call==='PROMOTE'&&r.call==='UNKNOWN');
assert(example);nodes.date.value=example.date;nodes.arm.value=example.arm;nodes.date.onchange();
assert(nodes.rows.innerHTML.includes('新核验附层'));
assert(nodes.rows.innerHTML.includes('已存分类头补资料后研究方向'));
assert(nodes.rows.innerHTML.includes('不是原预测或PAIRWISE方向'));
assert(nodes.rows.innerHTML.includes('qualification_receipt_time_unknown'));
nodes.arm.value='PAIRWISE_DAY_RANK';nodes.date.onchange();
assert(nodes.rows.innerHTML.includes('FAILED_FIXED_SCREEN_NOT_ADOPTED'));
assert(!nodes.rows.innerHTML.includes('POSITIVE_CLASSIFICATION_RESEARCH_ONLY'));
assert(all.filter(r=>r.arm==='PAIRWISE_DAY_RANK').every(r=>r.absolute_event_call===null&&r.recovered_classifier_research_call===null));
console.log(JSON.stringify({initial_rows:36,complete_rows:all.length,maximum_DOM_rows:200,all_filter_rows:22662,
each_arm_rows:3777,last_full_page_rows:62,initial_script_execution_ms:initial_ms,total_script_execution_ms:Date.now()-clock,
example_overlay_date:example.date,example_overlay_code:example.code,example_overlay_arm:example.arm,
original_overlay_calls_separate:true,pairwise_direction_always_empty:true}));
"""
    run = subprocess.run([shutil.which("node"), "-e", js],
        input=(OUTPUT / "complete_scorecard.html").read_text(encoding="utf-8"),
        text=True, encoding="utf-8", capture_output=True, timeout=60)
    assert run.returncode == 0, run.stderr
    html_execution = json.loads(run.stdout)

    prior_source_resolution = []
    old_receipt_path = SCORE / "full22662_independent_verification.json"
    old_receipt = json.loads(old_receipt_path.read_text(encoding="utf-8"))
    preservation_path = SCORE / "source_before_qualification_overlay/preservation.json"
    preservation = json.loads(preservation_path.read_text(encoding="utf-8"))
    preimages = {(row["original_path"], row["sha256"]): row["preserved_path"] for row in preservation["source_preimages"]}
    for path, digest in old_receipt["source_bindings"].items():
        if sha(path) == digest:
            resolved = path
        else:
            resolved = preimages[(path, digest)]
            assert sha(resolved) == digest
        prior_source_resolution.append({"original_path": path, "resolved_version_path": resolved, "sha256": digest})
    for folder in [SCORE / "full22662_research_view", SCORE / "archive405_after_pairwise_hardening", OUTPUT]:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        for filename, digest in manifest["artifact_hashes"].items():
            assert sha(folder / filename) == digest
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    for path, digest in manifest["source_bindings"].items():
        assert sha(path) == digest
    qualification_receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    for path, digest in qualification_receipt["source_bindings"].items():
        assert sha(path) == digest

    receipt = {
        "version": view.VERSION, "input_rows": len(source), "output_rows": len(actual),
        "rows_per_arm": source.groupby("arm").size().to_dict(),
        "original26_columns_and_all_prior_v2_derived_columns_exact": True,
        "BASE_original_score_call_ranking_and_100point_display_unchanged": True,
        "source_overlay_applied_rows": int(actual.source_repair_applied.sum()),
        "source_overlay_distinct_repaired_candidates": len(actual[actual.source_repair_applied][["date", "code"]].drop_duplicates()),
        "isolated_existing_classifier_recomputation_rows": int(recomputed.sum()),
        "PAIRWISE_has_no_absolute_event_or_recovered_direction": True,
        "full22662_target_label_future_changes_leave_all_derived_columns_exact": True,
        "HTML_execution": html_execution, "actual_browser_visual_verified": False,
        "actual_browser_limitation": "Parent attempted browser QA; browser safety policy denied file protocol and alternative local-server surface. No bypass attempted. Local Node DOM-function execution only.",
        "prior_full_v2_source_resolution": prior_source_resolution,
        "no_external_network_or_credentials_read": True, "new_fits": 0,
        "existing_author_review_gate_unchanged": True,
        "elapsed_seconds": time.monotonic() - start,
        "source_bindings": {str(p.resolve()): sha(p) for p in [SOURCE, RECEIPT, ORIGINAL,
            Path(view.__file__), Path("tests/test_relay_research_scorecard.py"), Path(__file__),
            SCORE / "verify_full22662_scorecard.py", SCORE / "test_results_28_with_source_overlay.xml",
            OUTPUT / "manifest.json", old_receipt_path, preservation_path]},
    }
    target = SCORE / "final_verified22662_independent_verification.json"
    assert not target.exists()
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ["input_rows", "source_overlay_applied_rows", "isolated_existing_classifier_recomputation_rows", "actual_browser_visual_verified", "elapsed_seconds"]}))


if __name__ == "__main__":
    main()
