"""Frozen pairwise ridge formulation with an explicit, uniform logit-scale offset."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('old_pairwise_helpers',ROOT/'research/first_board_dynamic_rank_20261030/pairwise_residual.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
scale_fit, transform, choose = old.scale_fit, old.transform, old.choose
KEYS=old.KEYS
TARGET=old.TARGET


def fit(train_pool,mature_labels,fields):
    if train_pool.duplicated(KEYS).any() or mature_labels.duplicated(KEYS).any():
        raise ValueError('duplicate training identity')
    if any('LABEL' in str(c) or c in {'outcome','label_end','target'} for c in train_pool):
        raise ValueError('labels must be separate')
    mean,scale,active=scale_fit(train_pool,fields)
    model={'fields':fields,'status':'FALLBACK','ridge_lambda':1.0,'intercept':None,
           'means':mean.tolist(),'scales':scale.tolist(),'active':active.tolist()}
    offset=train_pool.fixed_offset.to_numpy(float)
    if not np.isfinite(offset).all():raise ValueError('invalid fixed offset')
    if not active.any():return {**model,'reason':'NO_NONCONSTANT_FEATURES'}
    z=transform(train_pool,fields,model)
    indexed=train_pool[KEYS].reset_index(drop=True).reset_index(names='_i')
    known=indexed.merge(mature_labels[KEYS+[TARGET]],on=KEYS,how='left',validate='one_to_one')
    pairs, offsets, weights=[],[],[]
    pair_days=0
    for _,group in known.groupby('signal_date'):
        pos=group.loc[group[TARGET].eq(1),'_i'].to_numpy(int)
        neg=group.loc[group[TARGET].eq(0),'_i'].to_numpy(int)
        if not len(pos) or not len(neg):continue
        a,b=np.repeat(pos,len(neg)),np.tile(neg,len(pos))
        pairs.append(z[a]-z[b]);offsets.append(offset[a]-offset[b])
        weights.append(np.full(len(a),1/len(a)));pair_days+=1
    if not pair_days:return {**model,'reason':'NO_MATURE_POSITIVE_NEGATIVE_DAYS'}
    x,o,w=np.vstack(pairs),np.concatenate(offsets),np.concatenate(weights)/pair_days
    def objective(beta):
        delta=o+x@beta
        return np.dot(w,np.logaddexp(0,-delta))+.5*np.dot(beta,beta), x.T@(-w*expit(-delta))+beta
    result=minimize(objective,np.zeros(len(fields)),jac=True,method='L-BFGS-B',
                    options={'maxiter':300,'ftol':1e-12,'gtol':1e-8})
    model.update(pair_days=pair_days,pairs=len(o),solver_success=bool(result.success),
                 solver_status=int(result.status),solver_message=str(result.message),iterations=int(result.nit))
    if not result.success or not np.isfinite(result.x).all():return {**model,'reason':'SOLVER_FAILED_NO_RETRY'}
    return {**model,'status':'FIT','coefficients':result.x.tolist(),'loss':float(result.fun)}


def score(frame,model):
    offset=frame.fixed_offset.to_numpy(float)
    if not np.isfinite(offset).all():raise ValueError('invalid offset')
    if model['status']!='FIT':return offset.copy(),np.zeros(len(frame)),'FIXED_BASE_FALLBACK'
    delta=transform(frame,model['fields'],model)@np.asarray(model['coefficients'])
    return offset+delta,delta,'UNCALIBRATED_THS_OBSERVED_SUBSET_RESIDUAL'
