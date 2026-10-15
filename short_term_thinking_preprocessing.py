"""Train-only adapter for the installed histogram binning empty-array bug.

A column with no observed training value cannot supply a learned split. Keep
it constant on both train and inference, even if a later test value appears.
Partially observed columns retain every NaN and use native missing routing.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class EmptyTrainingColumnGuard(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        self.empty_columns_ = list(X.columns[X.isna().all()])
        return self

    def transform(self, X):
        if list(X.columns) != self.columns_:
            raise ValueError("frozen feature schema changed")
        result = X.copy()
        for column in self.empty_columns_:
            result[column] = 0.
        return result
