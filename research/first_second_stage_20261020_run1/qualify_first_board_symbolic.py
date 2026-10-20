"""Qualify existing dated-name diffusion for first boards; never fit models."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import first_board_leader_model as base
import historical_symbolic_context as symbolic

OUT=ROOT/'research/first_board_symbolic_qualification_20261020'
if OUT.exists():raise FileExistsError(OUT)
OUT.mkdir()
first,bars,sessions,audit=base.build_dataset()
candidates=first[['signal_date','code','name']].rename(columns={'signal_date':'date'})
candidates['boards']=1
closing=bars[bars.limit][['date','code','name','height']].rename(columns={'date':'signal_date','height':'board_height'})
closing['pool_assignment']='CLOSING_LIMIT_UP'
complete=sorted(bars.date.unique())
master=pd.read_csv(base.INPUTS['security_intervals'],dtype={'code':str}).fillna('')
facts,clusters,name_changes=symbolic.build_causal(candidates,closing,master,sessions,complete)
# The shared old engine was written for >=2-board relay candidates. A first
# board is not itself a relay, so it must not be subtracted from relay support.
facts.loc[facts.primary_symbolic_token.notna(),'symbolic_primary_relay_support_ex_self']+=1
facts.to_csv(OUT/'complete28503_first_board_dated_symbolic_features.csv',index=False,encoding='utf-8-sig')
base.write_json(OUT/'fixed_policy.json',symbolic.POLICY)
base.write_json(OUT/'all_qualified_name_clusters_NOT_VERIFIED_THEMES.json',clusters)
base.write_json(OUT/'all_dated_name_differences.json',name_changes)
checks=[]
for cutoff in ('2025-06-30','2025-12-31'):
    old=facts[facts.date.le(cutoff)].set_index(['date','code']).sort_index()
    prefix,_,_=symbolic.build_causal(candidates[candidates.date.le(cutoff)],closing[closing.signal_date.le(cutoff)],master,sessions,[d for d in complete if d<=cutoff])
    prefix.loc[prefix.primary_symbolic_token.notna(),'symbolic_primary_relay_support_ex_self']+=1
    new=prefix.set_index(['date','code']).sort_index()
    assert old.index.equals(new.index)
    np.testing.assert_allclose(old[symbolic.FIELDS].to_numpy(float),new[symbolic.FIELDS].to_numpy(float),rtol=1e-12,atol=1e-12,equal_nan=True)
    checks.append(dict(cutoff=cutoff,rows=len(old),same_all17=True))
receipt=dict(formed_at=base.now(),status='QUALIFIED_TYPED_RETROSPECTIVE_PROXY_ONLY',rows=len(facts),known_name_intervals=int(facts.symbolic_name_interval_observed.sum()),
             complete_within_declared_archive=int(facts.symbolic_scope_complete.sum()),candidate_rows_in_qualified_clusters=int(facts.primary_symbolic_token.notna().sum()),
             qualified_cluster_rows=len(clusters),source_name_differences=len(name_changes),prefix_causality=checks,independent_theme_or_role_truth=0,
             fixed_old_policy=True,first_board_adapter_correction='Do not subtract self from relay supporters when current board=1',
             source_complete_meaning='Complete only within declared reconstructed archive; current-universe survivorship and cold-start height lower bounds remain, not exchange-wide completeness',
             source_retrieved_later_not_backdated=True,source_hashes={str(p.resolve()):base.sha(p) for p in base.INPUTS.values()},
             engine_sha256=base.sha(symbolic.__file__),adapter_sha256=base.sha(__file__),fit_count=0,goal_60_not_verified=True,
             next='New evidence qualified; one separate preregistered first-stage feature-increment comparison may use this artifact, same labels/windows/algorithm, second head unchanged. Do not rerun rejected pairwise challenge.')
base.write_json(OUT/'qualification.json',receipt)
print(json.dumps({k:receipt[k] for k in ['rows','known_name_intervals','candidate_rows_in_qualified_clusters','source_name_differences','fit_count']},ensure_ascii=False))
