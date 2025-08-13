import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from numpy.random import RandomState
import pandas as pd


class PJSTree(BaseEstimator, RegressorMixin):
    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        self.js_estimators = {}

    def get_params(self, deep=True):
        # Return a dictionary of all parameters
        return {
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf
        }

    def set_params(self, **params):
        # Set parameters and reinitialize components as necessary
        self.max_depth = params.get('max_depth', self.max_depth)
        self.min_samples_leaf = params.get('min_samples_leaf', self.min_samples_leaf)
        self.tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        return self

    def fit(self, X, y):
        self.tree.fit(X, y)
        self.n_outputs_ = y.shape[1]
        self._compute_js_estimators(X, y)
        return self

    def _compute_js_estimator(self, y_true, y_pred_sa):

        delta_squared = np.var(y_true - y_pred_sa, axis=0, ddof=1)
        delta_squared += np.where(delta_squared == 0, 1e-6, 0)
        Lambda_hat = np.diag(delta_squared)
        eta_max = np.max(delta_squared)
        y_bar_sa = np.mean(y_pred_sa)
        trace_Lambda = np.sum(delta_squared)
        diff = y_pred_sa - y_bar_sa

        # Check if diff is a zero vector
        if np.all(diff == 0):
            # If diff is zero, then there is no adjustment needed, return y_pred_sa directly
            return y_pred_sa

        adjustment_factor = (trace_Lambda / eta_max - 3) / np.dot(diff.T, np.linalg.inv(Lambda_hat).dot(diff))
        y_pred_js = y_bar_sa + np.maximum(0, 1 - adjustment_factor) * diff
        return y_pred_js

    def _compute_js_estimators(self, X, y):
        leaf_indices = self.tree.apply(X)
        for leaf in np.unique(leaf_indices):
            mask = leaf_indices == leaf
            y_true_leaf = y[mask]
            y_pred_sa_leaf = np.mean(y_true_leaf, axis=0)
            self.js_estimators[leaf] = self._compute_js_estimator(y_true_leaf, y_pred_sa_leaf)

    def predict(self, X):
        leaf_indices = self.tree.apply(X)
        predictions = np.array([self.js_estimators[leaf] for leaf in leaf_indices])
        return predictions

    def score(self, X, y):
        y_pred = self.predict(X)
        mse_vals = mean_squared_error(y, y_pred, multioutput='uniform_average')
        return -mse_vals  # Scoring function should return higher values for better models

class PJSRF(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, dt_params=None, max_features='sqrt', bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        # Ensure dt_params is a dictionary
        self.dt_params = dt_params if dt_params is not None else {}
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        X= np.array(X)
        y= np.array(y)
        self.trees = []
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        # print(y)
        for _ in range(self.n_estimators):
            if self.bootstrap:
                indices = rng.choice(n_samples, size=n_samples, replace=True)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                X_sample, y_sample = X, y

            if self.max_features == 'sqrt':
                max_features = int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                max_features = int(np.log2(n_features))
            elif isinstance(self.max_features, float):
                max_features = int(self.max_features * n_features)
            elif self.max_features is None:
                max_features = n_features
            else:
                max_features = self.max_features

            features = rng.choice(n_features, max_features, replace=False)


            tree = PJSTree(**self.dt_params)
            tree.fit(X_sample[:, features], y_sample)

            self.trees.append((tree, features))

        return self

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.trees[0][0].n_outputs_))
        X = np.array(X)
        for tree, features in self.trees:
            X_subset = X[:, features]
            predictions += tree.predict(X_subset)
        return predictions / self.n_estimators
