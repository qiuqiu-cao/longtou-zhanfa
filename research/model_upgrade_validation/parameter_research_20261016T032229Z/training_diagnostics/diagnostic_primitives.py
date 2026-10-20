"""Pure read-only model diagnostics. No fitting, fetching or parameter selection."""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def eligible_B_metrics(frame, scores, original_B, verified_shape):
    """The same strict B route for each configuration; unknown is never zero."""
    score=pd.Series(np.asarray(scores,dtype=float),index=frame.index)
    original_B=pd.Series(np.asarray(original_B),index=frame.index).eq(0)
    shape=pd.Series(np.asarray(verified_shape),index=frame.index)
    known=frame.label_valid.eq(True)&frame.target.notna()
    valid=original_B&shape.eq(0)&known&score.notna()
    p=score[valid];y=frame.loc[valid,'target'].astype(int)
    predict=p.gt(.5)
    tp=int((predict&y.eq(1)).sum());fp=int((predict&y.eq(0)).sum())
    fn=int((~predict&y.eq(1)).sum());tn=int((~predict&y.eq(0)).sum())
    pr=tp/(tp+fp) if tp+fp else None;re=tp/(tp+fn) if tp+fn else None
    return dict(rows=len(frame),original_B_rows=int(original_B.sum()),
      original_B_unverified_price_rows=int((original_B&~shape.eq(0)).sum()),
      original_B_verified0_unknown_outcome_rows=int((original_B&shape.eq(0)&~known).sum()),
      eligible_known_B_rows=int(valid.sum()),TP=tp,FP=fp,FN=fn,TN=tn,
      precision=pr,recall=re,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
      F0_5=1.25*tp/(1.25*tp+fp+.25*fn) if 1.25*tp+fp+.25*fn else None,
      positive_call_rate=float(predict.mean()) if len(predict) else None,
      raw_uncalibrated_logloss=float(log_loss(y,p,labels=[0,1])) if len(y) else None,
      scores_are_validated_event_probabilities=False,threshold=.5,strict_greater_than=True)


def model_tree_structure(saved, fields):
    model=saved['model'] if isinstance(saved,dict) else saved
    hgb=model.named_steps['hgb']
    empty=model.named_steps['empty_training_columns'].empty_columns_
    trees=[];uses={field:dict(splits=0,trees=0,positive_gain_splits=0,gain_sum=0.) for field in fields}
    for iteration,predictors in enumerate(hgb._predictors):
        for class_index,predictor in enumerate(predictors):
            nodes=predictor.nodes;leaf=nodes['is_leaf'].astype(bool);internal=nodes[~leaf]
            gains=internal['gain'];feature_indices=internal['feature_idx'].astype(int)
            for idx in np.unique(feature_indices):
                if idx>=len(fields):raise ValueError('split feature outside frozen schema')
                use=uses[fields[idx]];group=internal[feature_indices==idx]
                use['splits']+=len(group);use['trees']+=1
                use['positive_gain_splits']+=int((group['gain']>0).sum())
                use['gain_sum']+=float(group['gain'].sum())
            trees.append(dict(iteration=iteration,class_index=class_index,nodes=len(nodes),
              leaves=int(leaf.sum()),internal_nodes=int((~leaf).sum()),
              max_actual_depth=int(nodes['depth'].max()),effective_positive_gain_splits=int((gains>0).sum()),
              unsplit_tree=bool(leaf.sum()==1),minimum_leaf_training_rows=int(nodes['count'][leaf].min()),
              maximum_leaf_training_rows=int(nodes['count'][leaf].max()),
              total_gain=float(gains.sum())))
    tree_frame=pd.DataFrame(trees)
    summary=dict(trees=len(trees),iterations=int(hgb.n_iter_),
      mean_leaves=float(tree_frame.leaves.mean()),maximum_leaves=int(tree_frame.leaves.max()),
      maximum_actual_depth=int(tree_frame.max_actual_depth.max()),
      effective_positive_gain_splits=int(tree_frame.effective_positive_gain_splits.sum()),
      unsplit_trees=int(tree_frame.unsplit_tree.sum()),
      minimum_leaf_training_rows=int(tree_frame.minimum_leaf_training_rows.min()),
      used_feature_count=sum(v['splits']>0 for v in uses.values()),
      training_all_missing_guard_columns=list(empty),
      configured_max_depth=hgb.max_depth,configured_max_leaf_nodes=hgb.max_leaf_nodes,
      configured_min_samples_leaf=hgb.min_samples_leaf,configured_l2_regularization=hgb.l2_regularization,
      configured_learning_rate=hgb.learning_rate,configured_max_iter=hgb.max_iter,
      trees_at_configured_leaf_cap=int(tree_frame.leaves.eq(hgb.max_leaf_nodes).sum()))
    return summary,tree_frame,pd.DataFrame([dict(field=k,**v) for k,v in uses.items()])


def feature_profiles(train,oof,fields):
    result=[]
    for field in fields:
        t=train[field];v=oof[field]
        distinct=set(t.dropna().unique())
        binary=bool(distinct and distinct<={0.,1.})
        observed=int(t.notna().sum());ones=int(t.eq(1).sum()) if binary else None
        tags=[]
        if observed==0:tags.append('ALL_MISSING_TRAIN_GUARD_CONSTANT')
        elif t.nunique(dropna=True)<2:tags.append('FEWER_THAN_TWO_OBSERVED_VALUES')
        if observed<=50:tags.append('OBSERVED_AT_MOST_50_TRAIN_ROWS')
        if binary and ones<=50:tags.append('BINARY_ONE_AT_MOST_50_TRAIN_ROWS')
        result.append(dict(field=field,train_rows=len(t),oof_rows=len(v),
          train_observed_rows=observed,train_missing_rows=int(t.isna().sum()),
          oof_observed_rows=int(v.notna().sum()),oof_missing_rows=int(v.isna().sum()),
          train_observed_unique=int(t.nunique()),oof_observed_unique=int(v.nunique()),
          binary_train=binary,train_binary_one_rows=ones,
          oof_binary_one_rows=int(v.eq(1).sum()) if binary else None,
          diagnostic_tags='|'.join(tags),diagnostic_only_not_filter=True))
    return pd.DataFrame(result)
