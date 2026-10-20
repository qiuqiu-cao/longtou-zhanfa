"""Verify saved-rank diagnosis and same-stock second-close rechecks; no fits."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_saved_rank_audit as audit

OUT=Path(__file__).resolve().parent
protocol=json.loads((OUT/"protocol_BEFORE_AUDIT.json").read_text("utf-8"))
report=json.loads((OUT/"report.json").read_text("utf-8"))
for path,digest in protocol["inputs"].items():
    assert base.sha(path)==digest,path
assert base.sha(audit.__file__)==protocol["source_sha256"]
full=pd.read_csv(OUT/"COMPLETE_SAVED_POOLS_UNCHANGED_LABELS_SCORES_AND_SCREEN.csv",dtype={"code":str},float_precision="round_trip")
assert len(full)==17873 and not full.duplicated(["signal_date","code"]).any()
original=pd.read_csv(audit.REF/"all_first_and_second_boards_WITH_SEPARATE_LABELS.csv",dtype={"code":str},float_precision="round_trip")
original=original[original.stage.eq(1)]
for period,file in [("validation","validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv"),("evaluation","S1_ALL_PREDICTIONS.csv")]:
    saved=pd.read_csv(audit.REF/file,dtype={"code":str},float_precision="round_trip")
    our=full[full.source_period.eq(period)].set_index(["signal_date","code"]).sort_index()
    frozen=saved.set_index(["signal_date","code"]).sort_index()
    assert our.index.tolist()==frozen.index.tolist()
    np.testing.assert_allclose(our[stage.VARIANTS["STAGE_CYCLE_GENE"]+[stage.TARGET,"model_score","calibrated_score"]],frozen[stage.VARIANTS["STAGE_CYCLE_GENE"]+[stage.TARGET,"model_score","calibrated_score"]],atol=0,rtol=0,equal_nan=True)
    old=original[original.signal_date.isin(saved.signal_date)].set_index(["signal_date","code"]).sort_index()
    np.testing.assert_allclose(our[stage.TARGET],old[stage.TARGET],atol=0,rtol=0,equal_nan=True)

counts=0
for period,mask in [("validation2025H1",full.source_period.eq("validation")),("evaluation_all",full.source_period.eq("evaluation")),
                    ("2025H2",full.signal_date.between("2025-07-01","2025-12-31")),("2026",full.signal_date.ge("2026-01-01"))]:
    period_rows=full[mask]
    for universe,frame in [("all",period_rows),("non_one_price",period_rows[period_rows.one_price.eq(0)])]:
        assert frame.signal_date.nunique()==period_rows.signal_date.nunique()
        for screen_name,apply in [("unfiltered",False),("fixed_screen",True)]:
            key=universe+"_"+screen_name
            metric=report["periods"][period][key]
            daily=pd.read_csv(OUT/f"{period}_{key}_DAILY_ORACLE_LABEL_ONLY.csv",dtype={"top1_code":str},float_precision="round_trip").set_index("signal_date")
            top_counts={k:dict(selected=0,known=0,hits=0,days=0) for k in (1,3,5)}
            oracle=0;removed_dates=0;categories={}
            ranked_winners=[]
            for day,complete in frame.groupby("signal_date",sort=True):
                candidates=complete[complete.research_selection_allowed] if apply else complete
                # NumPy lexicographic order independently verifies the saved rank.
                order=np.lexsort((candidates.code.to_numpy(),-candidates.model_score.to_numpy()))
                selected=candidates.iloc[order]
                truth=selected[stage.TARGET].to_numpy()
                expected=daily.loc[day]
                assert expected.complete_unknown==int(complete[stage.TARGET].isna().sum())
                assert expected.admissible_unknown==int(np.isnan(truth).sum())
                assert expected.top1_code==selected.code.iloc[0]
                np.testing.assert_allclose(expected.top1_outcome_LABEL_ONLY,truth[0],equal_nan=True,atol=0,rtol=0)
                positive_ranks=np.flatnonzero(truth==1)+1
                oracle+=int(len(positive_ranks)>0)
                removed_dates+=int(complete[stage.TARGET].eq(1).any() and len(positive_ranks)==0)
                best=float(positive_ranks.min()) if len(positive_ranks) else np.nan
                np.testing.assert_allclose(expected.oracle_best_known_winner_rank_LABEL_ONLY,best,equal_nan=True,atol=0,rtol=0)
                for k in (1,3,5):
                    values=truth[:k];c=top_counts[k]
                    c["selected"]+=len(values);c["known"]+=int(np.isfinite(values).sum())
                    c["hits"]+=int((values==1).sum());c["days"]+=int((values==1).any())
                    assert expected[f"top{k}_positive_LABEL_ONLY"]==int((values==1).sum())
                for rank in positive_ranks:
                    ranked_winners.append((day,selected.code.iloc[rank-1],int(rank)))
                category=expected.retrospective_error_category_LABEL_ONLY
                categories[category]=categories.get(category,0)+1
                counts+=1
            assert metric["rows"]==len(frame) and metric["dates"]==len(daily)
            assert metric["complete_unknown"]==int(frame[stage.TARGET].isna().sum())
            assert metric["complete_positive_pool"]==int(frame[stage.TARGET].eq(1).sum())
            assert metric["known_winner_dates_LABEL_ONLY"]==oracle
            assert metric["filter_removes_all_known_winner_dates_LABEL_ONLY"]==removed_dates
            assert categories==metric["error_categories_LABEL_ONLY"]
            assert sum(categories.values())==len(daily)
            for k,c in top_counts.items():
                m=metric["top"][str(k)]
                assert m["selected"]==c["selected"] and m["known"]==c["known"]
                assert m["positive_stocks"]==c["hits"] and m["dates_containing_known_winner"]==c["days"]
                assert m["unknown"]==c["selected"]-c["known"]
                np.testing.assert_allclose(m["stock_precision"],c["hits"]/c["known"],atol=0,rtol=0)
                np.testing.assert_allclose(m["date_any_winner_fraction"],c["days"]/len(daily),atol=0,rtol=0)
            saved_winners=pd.read_csv(OUT/f"{period}_{key}_ALL_KNOWN_WINNER_RANKS_LABEL_ONLY.csv",dtype={"code":str})
            assert set(map(tuple,saved_winners[["signal_date","code","saved_score_rank"]].to_numpy()))==set(ranked_winners)

# Bind the second-close extension before using its predictions. Original first
# choices and labels remain unchanged; stage2 cannot revise first-close metrics.
second_file=audit.REF/"S2_ALL_PREDICTIONS.csv"
extension=dict(formed_at=base.now(),task="Same-code original second-close head after fixed-screen first pick promotes",first_selection="Exactly saved baseline scores and unchanged original fixed screen",second_source=str(second_file.resolve()),second_sha256=base.sha(second_file),fits=0)
extension_path=OUT/"second_recheck_extension_BEFORE_COMPUTE.json"
if extension_path.exists():
    old=json.loads(extension_path.read_text("utf-8"));assert old["second_sha256"]==extension["second_sha256"]
else:
    base.write_json(extension_path,extension)
evaluation=full[full.source_period.eq("evaluation")]
second=pd.read_csv(second_file,dtype={"code":str},float_precision="round_trip")
chain,chain_stats=stage.chain_recheck(evaluation[evaluation.research_selection_allowed],second)
assert len(chain)==286
assert chain.first_promotion_LABEL_ONLY.eq(1).sum()==106
# Independent exact code and date join using only genuinely promoted selections.
next_rows=second.set_index(["signal_date","code"])
known=0;thirds=0;available=0
for row in chain.itertuples():
    if row.first_promotion_LABEL_ONLY!=1:
        assert pd.isna(row.second_model_score) and pd.isna(row.second_promotion_LABEL_ONLY)
        continue
    ref=next_rows.loc[(row.second_signal_date,row.code)]
    assert row.second_model_score==ref.model_score
    np.testing.assert_allclose(row.second_promotion_LABEL_ONLY,ref[stage.TARGET],atol=0,rtol=0,equal_nan=True)
    available+=1
    known+=int(pd.notna(ref[stage.TARGET]));thirds+=int(ref[stage.TARGET]==1)
assert chain_stats["second_rechecks_available"]==available
assert chain_stats["second_outcomes_known"]==known and chain_stats["continued_to_third"]==thirds
chain.to_csv(OUT/"FIXED_SCREEN_SAME_STOCK_SECOND_RECHECK_ALL286.csv",index=False,encoding="utf-8-sig")
base.write_json(OUT/"fixed_screen_second_recheck.json",chain_stats)
result=dict(status="PASS",verified_at=base.now(),complete_rows=len(full),complete_signal_dates=int(full.signal_date.nunique()),
    independent_day_universe_checks=counts,all16_tables_counts_unknowns_ranks_and_topk_checked=True,
    saved62_features_and_scores_exact=True,screen_redefined=False,oracle_labels_not_model_inputs=True,
    evaluation_fixed_screen_top1="106/284,2unknown",same_stock_second_recheck=chain_stats,
    new_fits=0,new_forward_results=0,actual_trades=0,goal_60_passed=False,
    source_sha256=base.sha(audit.__file__),verifier_sha256=base.sha(__file__))
for path,digest in protocol["inputs"].items():
    assert base.sha(path)==digest
base.write_json(OUT/"verification.json",result)
print(json.dumps(result,ensure_ascii=True,indent=2))
