"""Record the completed review and correct the stale current-batch pointer."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
POINTER=ROOT/'research/gemini_codex_joint_training_20261020/CURRENT_JOINT_EXPERIMENT.json'

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    data=json.loads((HERE/'INDEPENDENT_RECOMPUTATION.json').read_text('utf-8'))
    for rel, expected in data['sources'].items():
        assert digest(ROOT/rel)==expected, f'Reviewed original changed during review: {rel}'
    now=datetime.now(timezone.utc).isoformat()
    status=dict(updated_at=now,status='REQUIRES_CORRECTIONS_BEFORE_ANY_NEW_FREEZE',
        review=str(HERE/'研究报告与双臂方案审查.md'),independent_evidence=str(HERE/'INDEPENDENT_RECOMPUTATION.json'),
        next_peer_work_order=str(HERE/'RESPONSE_TO_GEMINI.md'),
        reviewed_new_arms=['COHORT_TURNOVER_74','REGIME_DYNAMIC_STATUS_76'],
        new_arms_training_freeze_approved=False,models_fitted_in_review=0,http_requests_in_review=0,
        old_lifecycle_status='BOTH_FIXED_ARMS_REJECTED',old_lifecycle_saved_scores_exactly_reproduced=True,
        old_lifecycle_root=str(ROOT/'research/leader_strategy_upgrade_20261021/model_comparison'),
        actual_peer='Antigravity Gemini 3.8 Flash High Turbo',
        communication=dict(direct_ui_message_sent=False,reason='Human unsent draft exists; preserved.',
                           file_is_peer_read_acknowledgement=False),
        original_files_preserved=True,base_old_predictions_ledger_trades_schedules_unchanged=True)
    (HERE/'REVIEW_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),'utf-8')
    new_pointer=dict(updated_at=now,status='NEW_REPORTS_REVIEW_REQUIRES_CORRECTIONS_NO_NEW_FREEZE',
        experiment_root=str(HERE),actual_peer=status['actual_peer'],
        original_pointer_snapshot=str(HERE/'originals/research/gemini_codex_joint_training_20261020/CURRENT_JOINT_EXPERIMENT.json'),
        review_receipt=str(HERE/'REVIEW_STATUS.json'),review_report=status['review'],
        current_new_proposal=dict(arms=status['reviewed_new_arms'],training_freeze_approved=False,
                                 status='CODEX_REQUESTED_CHANGES_NOT_MUTUALLY_FROZEN'),
        last_completed_experiment=dict(root=status['old_lifecycle_root'],status=status['old_lifecycle_status'],
            report=str(ROOT/'research/leader_strategy_upgrade_20261021/model_comparison/report.json'),
            primary_hits={'THEME_LIFECYCLE_75':47,'THEME_LIFECYCLE_STATUS_79':47},
            non_one_price_hits={'THEME_LIFECYCLE_75':40,'THEME_LIFECYCLE_STATUS_79':39},
            selected_dates=117,monthly_fits=0,production_deployment_allowed=False),
        baseline=dict(arm='ORIGINAL62',primary_hits=52,non_one_price_hits=48,selected_dates=117),
        constraints=dict(scope='THIS_REVIEW_ONLY',max_arms=2,models_fit=0,http_requests=0,
                         labels_loaded_for_audit=True,new_training_freeze_approved=False,
                         goal='FIRST_CLOSE_TOP1_SUCCESSES_PER_ALL_SIGNAL_DATES_GE_60_PERCENT'),
        next_peer_work_order=status['next_peer_work_order'],communication=status['communication'],
        active_workstream=dict(status='REVIEW_COMPLETED_PEER_CORRECTION_REQUIRED',root=str(HERE),
                              actual_executor='Antigravity Gemini 3.8 Flash High Turbo',director='Codex',
                              new_fits_authorized_by_this_review=False))
    POINTER.write_text(json.dumps(new_pointer,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(status,ensure_ascii=False))

if __name__=='__main__': main()
