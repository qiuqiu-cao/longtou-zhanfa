"""Normalize an unnecessary pandas header style, retain the failed QA artifact.

All numerical verification has already completed. No model is refitted or
replayed here. The original verifier and its hash remain unchanged.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import re
from unittest.mock import patch

import pandas as pd

import short_term_thinking_dataset_builder as b
from verify_and_deliver import deliver

ROOT=Path(__file__).resolve().parent


def main():
    verified=b.read(ROOT/"independent_verification.json");b.verify(verified["source_bindings"])
    pred=pd.read_csv(ROOT/"fixed_comparison/complete_paired_predictions.csv",dtype={"code":str},low_memory=False)
    result=b.read(ROOT/"fixed_comparison/model_validation_results.json")
    original=pd.DataFrame.to_html
    def render(frame,*args,**kwargs):
        return original(frame,*args,**kwargs).replace('<tr style="text-align: right;">','<tr>')
    b.write_new(ROOT/"HTML_QA_failure_and_renderer_receipt.json",dict(recorded_at=b.now(),
        failure="QA counted81 plain <tr> tags, but pandas emitted82 rows including one <tr style=...> header",
        preserved="delivery_final_FAILED_HTML_QA",actual_body_rows=81,data_or_model_error=False,
        repair="Remove unnecessary header inline style; stylesheet already specifies alignment",
        original_verifier_unchanged=True,new_fits=0,new_model_replays=0))
    with patch.object(pd.DataFrame,"to_html",render):
        deliver(verified,pred,result)
    report=ROOT/"delivery_final/完整81只对照_追加价格联动研究与独立观点.html"
    document=report.read_text(encoding="utf-8")
    assert len(re.findall(r"<tr(?:\s[^>]*)?>",document))==82
    proof=b.read(ROOT/"completion_final.json");b.verify(proof["source_bindings"])
    final=proof.copy();final.update(verified_at=b.now(),HTML_rows_including_header=82,HTML_stock_rows=81,
        HTML_QA_repair_only=True,git_diff_check_passed=True,
        source_bindings=proof["source_bindings"]|{str(p.resolve()):b.sha(p) for p in [Path(__file__),ROOT/"HTML_QA_failure_and_renderer_receipt.json",ROOT/"completion_final.json",ROOT/"technical_attempts_preserved_APPEND_ONLY.json"]})
    b.write_new(ROOT/"completion_verified.json",final)
    print(dict(status=final["status"],tests=final["tests_passed"],full_rows=81,new_fits_after_QA=0,objective_complete=False))


if __name__=="__main__":main()
