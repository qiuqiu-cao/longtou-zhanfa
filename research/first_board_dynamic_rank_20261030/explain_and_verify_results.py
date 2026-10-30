"""Replay stored coefficients and explain every swap. Never fits or changes picks."""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pairwise_residual import transform, score, choose
from dynamic_features import FIELDS, KEYS
from run_experiment import HERE, DATA, ROOT, read, sha, save_json, verify_freeze, slots


def main():
    verify_freeze()
    results=HERE/'results'; out=HERE/'postrun_review';out.mkdir(exist_ok=False)
    x=read(DATA/'FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv')
    f=read(HERE/'prepared_v2/FULL_POOL_THEME_FEATURES_NO_LABELS.csv')
    full=x.merge(f,on=KEYS,validate='one_to_one',suffixes=('','_theme'))
    scores=read(results/'ALL_CANDIDATE_SCORES_NO_LABELS.csv')
    scores=scores.sort_values(['arm','signal_date','research_score','code'],ascending=[True,True,False,True])
    scores['new_rank']=scores.groupby(['arm','signal_date']).cumcount()+1
    x=x.sort_values(['signal_date','s1_score','code'],ascending=[True,False,True])
    x['original_S1_rank']=x.groupby('signal_date').cumcount()+1
    rank_lookup=x.set_index(KEYS).original_S1_rank.to_dict()
    feature_lookup=full.set_index(KEYS)
    score_lookup=scores.set_index(['arm']+KEYS)
    models={}; checks=[]
    for path in sorted(results.glob('*MODEL.json')):
        m=json.loads(path.read_text(encoding='utf-8'));models[(m['arm'],m['month'])]=m
        month_rows=full[full.signal_date.str.startswith(m['month'])]
        before=read(results/f'{m["arm"]}_PICKS_BEFORE_OUTCOME_JOIN.csv')
        recomputed=[]; max_error=0.0
        for day,g in month_rows.groupby('signal_date'):
            g=g.copy();v,delta,status=score(g,m);g['replayed']=v
            saved=score_lookup.loc[(m['arm'],day)].reindex(g.code)
            max_error=max(max_error,float(np.max(np.abs(v-saved.research_score.to_numpy()))))
            np.testing.assert_allclose(v,saved.research_score.to_numpy(),atol=1e-14,rtol=0)
            recomputed.append(slots(choose(g,'replayed')))
        replay=pd.concat(recomputed,ignore_index=True)
        expected=before[before.signal_date.str.startswith(m['month'])].reset_index(drop=True)
        pd.testing.assert_frame_equal(replay,expected)
        members=read(results/f'{m["arm"]}_{m["month"]}_TRAIN_MEMBERS.csv')
        if len(members):
            assert members.signal_date.lt(month_rows.signal_date.min()).all()
            assert members.label_end.dropna().le(m['cutoff']).all()
            assert members.signal_date.nunique()<=252
        checks.append({'arm':m['arm'],'month':m['month'],'scores_checked':len(month_rows),
            'score_max_error':max_error,'picks_exact':True,'mature_training_window_passed':True,'fit_status':m['status']})
    swaps=pd.read_csv(results/'BIDIRECTIONAL_SWAP_LEDGER.csv',
        dtype={'displaced_code':str,'promoted_code':str},float_precision='round_trip')
    records=[]
    for row in swaps.to_dict('records'):
        m=models[(row['arm'],row['signal_date'][:7])]
        for side in ['promoted','displaced']:
            code=row[side+'_code'];key=(row['signal_date'],code)
            observation=feature_lookup.loc[key]; scored=score_lookup.loc[(row['arm'],*key)]
            frame=observation.to_frame().T
            contributions=transform(frame,m['fields'],m)[0]*np.array(m['coefficients'])
            details=json.loads(observation.theme_details_json)
            row.update({side+'_name':observation['name'],side+'_original_S1_rank':int(rank_lookup[key]),
                side+'_new_rank':int(scored.new_rank),side+'_S1_score':float(observation.s1_score),
                side+'_research_score':float(scored.research_score),side+'_residual':float(scored.residual),
                side+'_factor_contributions':json.dumps(dict(zip(m['fields'],contributions.tolist())),ensure_ascii=False),
                side+'_missing_feature_count':int(observation[m['fields']].isna().sum()),
                side+'_theme_evidence':observation.theme_details_json,
                side+'_candidate_theme_source':observation.source_path_theme,
                side+'_candidate_theme_source_sha256':observation.source_sha256_theme,
                side+'_prior_anchor_codes':'|'.join(sorted(set(c for d in details for c in d['prior_anchor_codes'])))})
        row['explanation_scope']='LINEAR_SCORE_COMPARISON_WITH_REPORTED_THEME_EVIDENCE_NOT_CAUSAL_LEADERSHIP'
        records.append(row)
    enriched=pd.DataFrame(records)
    enriched.to_csv(out/'ALL_SWAPS_WITH_SCORE_CONTRIBUTIONS_AND_EVIDENCE.csv',index=False,encoding='utf-8-sig')
    rank_diagnostics={}
    for arm,g in enriched.groupby('arm'):
        rank_diagnostics[arm]={'promoted_original_rank_min':int(g.promoted_original_S1_rank.min()),
            'promoted_original_rank_max':int(g.promoted_original_S1_rank.max()),
            'promoted_from_beyond_Top10':int(g.promoted_original_S1_rank.gt(10).sum()),
            'promoted_unknown_context':int(g.promoted_missing_feature_count.eq(len(models[(arm,'2026-08')]['fields'])).sum())}
    receipt={'created_at':datetime.now(timezone.utc).isoformat(),'new_fits':0,'new_selections':0,
        'model_replay_checks':checks,'all_14508_candidate_scores_reproduced':sum(c['scores_checked'] for c in checks)==14508,
        'all_stored_top3_picks_reproduced':True,'swap_rows_enriched':len(enriched),
        'rank_diagnostics':rank_diagnostics,'hypothesis_or_weights_changed_after_results':False,
        'protocol_sha256':sha(HERE/'PROTOCOL.json')}
    save_json(out/'REPLAY_VERIFICATION.json',receipt)
    print(json.dumps({'replayed_candidate_scores':sum(c['scores_checked'] for c in checks),
        'enriched_swaps':len(enriched),'rank_diagnostics':rank_diagnostics},ensure_ascii=False))


if __name__=='__main__':main()
