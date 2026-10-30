"""Verify the bounded first-board correction, then seal the research line."""
from pathlib import Path
from datetime import datetime,timezone
import ast
import hashlib
import json
import math

HERE=Path(__file__).resolve().parent
C=HERE/'first_board_closeout'
ROOT=HERE.parents[1]
OLD=ROOT/'research/leader_strategy_upgrade_20261021/review_20261021T1128Z/INDEPENDENT_RECOMPUTATION.json'
def read(p):return json.loads(p.read_text('utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def check_cell(g,ref):
    for a,b in [('n','n'),('known','known'),('unknown','unknown'),('k','successes')]:assert g[a]==ref[b],(a,g,ref)
    if g['known_rate'] is not None:assert abs(g['known_rate']-ref['known_rate'])<=.000051
    assert abs(g['lower_bound']-ref['all_rows_success_lower_bound'])<=.000051

def main():
    old=read(OLD);a=read(C/'LINE_A_CONDITION_STUDY_V2.json');b=read(C/'LINE_B_AUDIT_RECEIPT_V2.json')
    r=a['relay_track_9245'];s=a['first_board_track_13149']
    check_cell(r['overall'],old['relay_official_target_overall'])
    check_cell(s['overall'],old['s1_overall'])
    check_cell(s['non_one_price_overall'],old['s1_non_one_price'])
    for g,key in [('no_break','False'),('has_break','True')]:
        check_cell(r['by_parent_break'][g],old['corrected_official_target_tables']['parent_break'][key])
    for role,cell in r['by_primary_role'].items():check_cell(cell,old['corrected_official_target_tables']['roles'][role])
    for group,cell in b['first_board_turnover_binnings_corrected'].items():
        keys={'<3%':'(-1.0, 3.0]','3-5%':'(3.0, 5.0]','5-10%':'(5.0, 10.0]',
              '10-15%':'(10.0, 15.0]','15-20%':'(15.0, 20.0]','>20%':'(20.0, 100.0]'}
        check_cell(cell,old['s1_turnover_bins'][keys[group]])
    for info in b['relay_field_audit'].values():assert info['qualification_status']=='NOT_ASSESSED_BY_THIS_NULL_COUNT_AUDIT'
    for name in ['line_a_cohort_feedback_study_v2.py','line_b_seal_quality_audit_v2.py']:
        tree=ast.parse((C/name).read_text('utf-8'))
        assert not [v for v in ast.walk(tree) if isinstance(v,ast.Call) and isinstance(v.func,ast.Attribute) and v.func.attr in ['fit','fit_transform']]
    for source,h in a['source_hashes'].items():assert sha(Path(source))==h
    report=(C/'线A_梯队与断板条件研究报告_修订版.md').read_text('utf-8')
    assert '33.40%' not in report and '35.95%' not in report and '+-' not in report
    receipt=dict(reviewed_at=datetime.now(timezone.utc).isoformat(),status='FIRST_BOARD_CORRECTION_ACCEPTED_AND_RESEARCH_PAUSED',
        new_first_board_fits=0,new_first_board_features=0,first_board_new_work_authorized=False,
        original_gemini_preimages='codex_preimages/',
        codex_final_bounded_corrections=['One-to-one join validation and complete-key check',
            'Generate narrative rates from the same cells instead of stale hardcoded percentages',
            'Null counts are coverage, not source/timestamp qualification',
            'Explicit turnover-bin boundaries and missing turnover row; corrected signed formatting'],
        original_source_hashes_verified=True,independent_statistics_recomputed=True,
        review_does_not_approve_new_first_board_training=True,
        artifacts={p.name:sha(p) for p in C.iterdir() if p.is_file() and p.suffix in ['.py','.md','.json'] and p.name not in ['CODEX_CLOSEOUT_AUDIT.json','FIRST_BOARD_CLOSEOUT.json']})
    (C/'CODEX_CLOSEOUT_AUDIT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),'utf-8')
    original=read(C/'FIRST_BOARD_CLOSEOUT.json')
    original['codex_final_review']=dict(status=receipt['status'],receipt=str(C/'CODEX_CLOSEOUT_AUDIT.json'),at=receipt['reviewed_at'])
    original['repairs_completed']['full_denominator_disclosure']['details']='All main condition tables expose full/known/unknown/success counts and known rates; bounds included in Line A. No unknown outcomes certified as failure.'
    (C/'FIRST_BOARD_CLOSEOUT.json').write_text(json.dumps(original,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
