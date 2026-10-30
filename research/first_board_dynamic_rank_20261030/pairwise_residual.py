"""Fixed ridge pairwise residual; never a calibrated event probability."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

KEYS = ['signal_date', 'code']
TARGET = 'promotion_next_LABEL_ONLY'


def scale_fit(frame, fields):
    values = frame[fields].to_numpy(float)
    valid = np.isfinite(values)
    counts = valid.sum(axis=0)
    means = np.divide(np.where(valid, values, 0).sum(axis=0), counts,
                      out=np.zeros(len(fields)), where=counts > 0)
    centered = np.where(valid, values - means, 0)
    variance = np.divide((centered**2).sum(axis=0), counts,
                         out=np.zeros(len(fields)), where=counts > 0)
    active = (counts > 0) & (variance > 0)
    scales = np.where(active, np.sqrt(variance), 1)
    return means, scales, active


def transform(frame, fields, model):
    x = frame[fields].to_numpy(float)
    z = np.where(np.isfinite(x), (x - np.array(model['means'])) / np.array(model['scales']), 0)
    z[:, ~np.array(model['active'], dtype=bool)] = 0
    return z


def fit(train_pool, mature_labels, fields):
    if train_pool.duplicated(KEYS).any() or mature_labels.duplicated(KEYS).any():
        raise ValueError('duplicate training identity')
    if any('LABEL' in str(c) or c in {'outcome','label_end','target'} for c in train_pool):
        raise ValueError('labels must be supplied separately')
    model = {'fields': fields, 'status': 'FALLBACK', 'ridge_lambda': 1.0, 'intercept': None}
    mean, scale, active = scale_fit(train_pool, fields)
    model.update(means=mean.tolist(), scales=scale.tolist(), active=active.tolist())
    scores = train_pool.s1_score.to_numpy(float)
    if not (np.isfinite(scores).all() and ((scores > 0) & (scores < 1)).all()):
        return {**model, 'reason': 'INVALID_S1_TRAINING_OFFSET'}
    if not active.any(): return {**model, 'reason': 'NO_NONCONSTANT_FEATURES'}
    z = transform(train_pool, fields, model)
    offset = logit(scores)
    indexed = train_pool[KEYS].reset_index(drop=True).reset_index(names='_i')
    known = indexed.merge(mature_labels[KEYS+[TARGET]], on=KEYS, how='left', validate='one_to_one')
    pairs, offsets, weights = [], [], []
    pair_days = 0
    for _, group in known.groupby('signal_date'):
        pos = group.loc[group[TARGET].eq(1), '_i'].to_numpy(int)
        neg = group.loc[group[TARGET].eq(0), '_i'].to_numpy(int)
        if not len(pos) or not len(neg): continue
        a, b = np.repeat(pos, len(neg)), np.tile(neg, len(pos))
        pairs.append(z[a]-z[b]); offsets.append(offset[a]-offset[b])
        weights.append(np.full(len(a), 1/len(a))); pair_days += 1
    if not pair_days: return {**model, 'reason': 'NO_MATURE_POSITIVE_NEGATIVE_DAYS'}
    x, o, w = np.vstack(pairs), np.concatenate(offsets), np.concatenate(weights)/pair_days
    def objective(beta):
        delta = o + x @ beta
        loss = np.dot(w, np.logaddexp(0, -delta)) + .5 * np.dot(beta, beta)
        gradient = x.T @ (-w * expit(-delta)) + beta
        return loss, gradient
    result = minimize(objective, np.zeros(len(fields)), jac=True, method='L-BFGS-B',
                      options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
    model.update(pair_days=pair_days, pairs=len(o), solver_success=bool(result.success),
        solver_status=int(result.status), solver_message=str(result.message), iterations=int(result.nit))
    if not result.success or not np.isfinite(result.x).all():
        return {**model, 'reason': 'SOLVER_FAILED_NO_RETRY'}
    return {**model, 'status': 'FIT', 'coefficients': result.x.tolist(), 'loss': float(result.fun)}


def score(frame, model):
    values = frame.s1_score.to_numpy(float)
    if not (np.isfinite(values).all() and ((values > 0) & (values < 1)).all()):
        return values.copy(), np.zeros(len(frame)), 'INVALID_S1_DAY_FALLBACK'
    if model['status'] != 'FIT': return values.copy(), np.zeros(len(frame)), 'S1_FALLBACK'
    delta = transform(frame, model['fields'], model) @ np.array(model['coefficients'])
    return logit(values)+delta, delta, 'PAIRWISE_RESIDUAL_UNCALIBRATED'


def choose(frame, score_column='s1_score'):
    return frame.sort_values(['signal_date', score_column, 'code'], ascending=[True, False, True],
        kind='stable').groupby('signal_date', sort=True).head(3)[KEYS].reset_index(drop=True)
