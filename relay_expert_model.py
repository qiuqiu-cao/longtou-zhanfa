"""Frozen boosting challengers; proxy routing never certifies true one-word identity."""
import warnings
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from current_model_diagnostics import RELAY_FIELDS

PARAMETERS = {"learning_rate": .05, "max_iter": 160, "max_leaf_nodes": 15,
              "max_depth": 3, "min_samples_leaf": 15, "l2_regularization": 1.,
              "early_stopping": False, "class_weight": None, "random_state": 271,
              "max_bins": 255, "categorical_features": None}


class RelayExpertClassifier:
    def __init__(self, arm):
        if arm not in ("H1", "H2"):
            raise ValueError("unregistered boosting arm")
        self.arm = arm

    def _inputs(self, frame):
        if list(frame.columns) != RELAY_FIELDS:
            raise ValueError("original 29 fields/order required")
        proxy = frame["signal_one_word"].to_numpy(float)
        if not np.isin(proxy[np.isfinite(proxy)], [0., 1.]).all():
            raise ValueError("legacy proxy must be zero, one or unknown")
        return proxy

    def fit(self, frame, labels, sample_weight):
        proxy = self._inputs(frame)
        labels, weights = np.asarray(labels, dtype=int), np.asarray(sample_weight, dtype=float)
        if set(labels) != {0, 1} or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("both known classes and original positive daily weights required")
        self.preprocessor_ = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                                           StandardScaler())
        prepared = self.preprocessor_.fit_transform(frame)
        self.classes_ = np.asarray([0, 1])
        self.n_features_in_ = len(RELAY_FIELDS)
        self.members_ = len(labels)
        self.models_, self.members_by_branch_ = {}, {}
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            warnings.simplefilter("error", ConvergenceWarning)
            self.global_ = HistGradientBoostingClassifier(**PARAMETERS)
            self.global_.fit(prepared, labels, sample_weight=weights)
            if self.global_.n_iter_ != PARAMETERS["max_iter"]:
                raise ValueError("unexpected early termination")
            if self.arm == "H2":
                for group in (0., 1.):
                    mask = proxy == group
                    self.members_by_branch_[str(int(group))] = int(mask.sum())
                    if not mask.any() or set(labels[mask]) != {0, 1}:
                        raise ValueError("routing branch missing members or either class; no synthetic completion")
                    model = HistGradientBoostingClassifier(**PARAMETERS)
                    model.fit(prepared[mask], labels[mask], sample_weight=weights[mask])
                    if model.n_iter_ != PARAMETERS["max_iter"]:
                        raise ValueError("branch early termination")
                    self.models_[group] = model
        self.unknown_proxy_training_members_ = int(np.isnan(proxy).sum())
        return self

    def predict_proba(self, frame):
        proxy = self._inputs(frame)
        prepared = self.preprocessor_.transform(frame)
        with threadpool_limits(limits=1):
            scores = self.global_.predict_proba(prepared)
            if self.arm == "H2":
                for group, model in self.models_.items():
                    mask = proxy == group
                    if mask.any():
                        scores[mask] = model.predict_proba(prepared[mask])
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite internal score")
        return scores

    def predict(self, frame):
        return self.classes_[self.predict_proba(frame).argmax(axis=1)]
