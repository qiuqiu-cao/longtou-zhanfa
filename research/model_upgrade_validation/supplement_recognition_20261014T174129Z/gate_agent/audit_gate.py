"""Read-only source audit and contract; writes only this agent's output folder."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8-sig'))

sources = {
    'label_counts': 'research/model_upgrade_validation/leader_role_historical_review_20261012T142646Z/actual_independent_label_counts.json',
    'training_truth': 'research/model_upgrade_validation/leader_role_training_truth_20261013T054938Z/actual_training_truth/original_promotion_training_truth.json',
    'mature_complete10': 'results/deep_review_20261014T115540Z/原10只判断与反例.csv',
    'jiuyang_baiyang': 'results/deep_review_20261014T115540Z/jiuyang_baiyang_relation_correction_20261014T121615Z/relation_review.json',
    'sunshine_relation_note': 'results/sunshine_jiuyang_relation_20261014T152814Z/阳光九阳补涨关系与候选比较.md',
    'current15': 'results/sunshine_jiuyang_relation_20261014T152814Z/15日完整15只角色补充输出.json',
    'current_name_cluster': 'results/daily_review_20261014T1118Z/name_character_context_original_rules.json',
    'engine': 'leader_role_engine.py',
    'research': 'leader_role_research.py',
}
counts = read(sources['label_counts'])
cluster = read(sources['current_name_cluster'])
current = read(sources['current15'])
with (ROOT / sources['mature_complete10']).open(encoding='utf-8-sig', newline='') as fh:
    mature = list(csv.DictReader(fh))

def metrics(rows, independent=False):
    c = dict(TP=0, TN=0, FP=0, FN=0, POSITIVE_ABSTENTION=0, NEGATIVE_ABSTENTION=0)
    for r in rows:
        actual = r['supplemental_promoted'] == 'True'
        call = r['original_independent_call'] if independent else ('PROMOTE' if r['original_model_positive'] == 'True' else 'FAIL')
        if call == 'UNKNOWN':
            c['POSITIVE_ABSTENTION' if actual else 'NEGATIVE_ABSTENTION'] += 1
        else:
            c['TP' if actual and call == 'PROMOTE' else 'FN' if actual else 'FP' if call == 'PROMOTE' else 'TN'] += 1
    judged = c['TP'] + c['TN'] + c['FP'] + c['FN']
    positives = c['TP'] + c['FN'] + c['POSITIVE_ABSTENTION']
    return c | {'pool': len(rows), 'judged': judged, 'Accuracy': (c['TP']+c['TN'])/judged if judged else None,
        'correct_over_all_pool': (c['TP']+c['TN'])/len(rows), 'Recall_complete': c['TP']/positives if positives else None,
        'Precision': c['TP']/(c['TP']+c['FP']) if c['TP']+c['FP'] else None,
        'F1_complete': 2*c['TP']/(2*c['TP']+c['FP']+c['FN']+c['POSITIVE_ABSTENTION']) if positives or c['FP'] else None,
        'prediction_coverage': judged/len(rows)}

report = {
    'formed_at': datetime.now(timezone.utc).isoformat(),
    'purpose': 'READ_ONLY_EVALUATION_AND_PREDICTION_GATE_FOR_PARENT_CHILD_RECOGNIZER',
    'source_bindings': {str((ROOT / rel).resolve()): sha(ROOT / rel) for rel in sources.values()},
    'role_truth_counts': counts['by_role'],
    'usable_supervised_supplement_pair_truth': 0,
    'promotion_metrics_original13to14_supplemental10': {'BASE': metrics(mature), 'independent': metrics(mature, True)},
    'non_one_price_signal_metrics': {'BASE': metrics([r for r in mature if r['signal_actual_one_word'] == '0.0']),
        'independent': metrics([r for r in mature if r['signal_actual_one_word'] == '0.0'], True)},
    'mature_BASE_false_negatives': [{k: r[k] for k in ('code','name','original_display_score','original_independent_call','signal_actual_one_word','supplemental_promoted')}
        for r in mature if r['original_model_positive'] == 'False' and r['supplemental_promoted'] == 'True'],
    'current15_unmatured': [{'code': r['code'], 'name': r['name'], 'score': r['score_display'], 'model_positive': r['predicted_positive'],
        'independent': r['original_independent_direction'], 'next_day_opening_status': r['next_day_opening_status']}
        for r in current['rows']],
    'structural_diagnosis': {
        'legacy_dependency_requires_same_specific_theme': True,
        'legacy_SUPPLEMENT_requires_theme_and_dependency': True,
        'name_cluster_requires_min_members': cluster['thresholds']['min_member_count'],
        'name_cluster_actual_count': cluster['cluster_count'],
        'scope_limitation': 'Current closing/failed pool does not retain a broken older anchor as an explicit temporal parent.',
        'do_not_forge_same_specific_theme_for_name_relation': True},
    'evaluation_contract': {
        'unit': 'signal_date + scope_id + anchor_code + child_code + relation_type + source_knowledge_cutoff',
        'candidate_discovery': 'Preserve complete declared child pool, explicit archived prior-anchor scope, every generated edge, no-edge/unknown stocks, excluded branches and all alternatives.',
        'evidence_targets': ['PUBLIC_NARRATIVE_LINK_EXISTS_IN_BOUND_BODY', 'ANCHOR_PRECEDES_CHILD_START_IN_VERIFIED_SESSION_TIMELINE', 'CHILD_CONTINUES_AFTER_OBSERVED_ANCHOR_BREAK', 'WITHIN_CHAIN_CURRENT_RELATIVE_LEADERSHIP'],
        'identity_targets_separate': ['SUPPLEMENT_CANDIDATE', 'SUPPLEMENT_CORE_OR_LEADER'],
        'unknown_is_not_negative': True,
        'pair_truth_source': 'Independent actual bounded evidence review; current audited records are hypotheses, not supervised identity truth.',
        'factual_edge_recall': 'Only against a separately audited complete edge set within explicitly fixed source and pool scope; searched-but-not-mentioned is not ground-truth negative.',
        'identity_metrics': ['full known-scope Accuracy','complete-positive Recall','Precision','F1','abstention and evidence coverage','complete-chain Top1/Top3 with ties and missing edges retained'],
        'predictive_metrics': 'Exact original date/code/label full cohort vs frozen BASE; Accuracy, complete Recall, Precision, F1, coverage, board/actual one-price/role-evidence strata; no calibrated Brier claim.',
        'settlement': 'D+1 exact-session promotion truth separate from D+3/D+5 identity persistence; observed break/reacquisition is not exact intraday time; leave absent identity labels unknown.',
        'historical_status': 'Already-seen historical source and outcomes: diagnostic replay only. Late-acquired relation cannot retroactively change forecast knowledge cutoff.',
        'deployment_gate': 'Existing original fixed effect and 120-session gates remain. No deployment from 2 case demonstrations or increased test counts.',
        'numerical_fit_now': 'Not justified for SUPPLEMENT supervised head or existing all11 C2. Promotion factual-feature challenger is not universally prohibited, but complete comparable pre-cutoff pair-feature coverage is absent, so this batch should not fit it.',
    },
    'new_fits': 0,
    'new_labels_created': 0,
    'production_modified': False,
    'new_formal_forward_samples': 0,
}
OUT.mkdir(parents=True, exist_ok=True)
dest = OUT / 'read_only_gate_findings.json'
dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'output': str(dest), 'mature_metrics': report['promotion_metrics_original13to14_supplemental10'], 'role_pairs_supervised': 0}, ensure_ascii=False))
