import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from numpy.random import RandomState
import pandas as pd

class PSITree(BaseEstimator, RegressorMixin):
    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        self.SI_estimators = {}

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
        """
        Fits the decision tree model to the training data and computes the MWS* estimators for each leaf node.
        """
        self.tree.fit(X, y)
        self.n_outputs_ = y.shape[1]
        self._compute_si_estimators(X, y)

    def _compute_si_estimator(self, y_true, y_pred_sa):
        """
        Computes the MWS* estimator for a single node.
        """
        N_m = y_true.shape[0]  # Number of samples in the node

        squared_diff_sum = sum(
            (y_pred_sa[j] - y_pred_sa[j_prime]) ** 2
            for j in range(self.n_outputs_ - 1) for j_prime in range(j + 1, self.n_outputs_)
        )

        # Calculate sum of variances
        variance_sum = np.sum(np.var(y_true - y_pred_sa, axis=0, ddof=1))

        # Calculate the denominator
        denominator = N_m * squared_diff_sum + (self.n_outputs_ - 1) * variance_sum

        if denominator == 0:
            return y_pred_sa

        # Calculate the adjustment matrix
        adjustment_matrix = N_m * squared_diff_sum * np.identity(self.n_outputs_) +  (self.n_outputs_ - 1) * variance_sum * np.ones((self.n_outputs_, self.n_outputs_)) / self.n_outputs_

        # Calculate the MWS* estimator
        y_pred_mws = (1 / denominator) * adjustment_matrix.dot(y_pred_sa)

        return y_pred_mws

    def _compute_si_estimators(self, X, y):
        """
        Computes and stores the MWS* estimator for each leaf node in the tree.
        """
        leaf_indices = self.tree.apply(X)
        for leaf in np.unique(leaf_indices):
            mask = leaf_indices == leaf
            y_true_leaf = y[mask]
            y_pred_sa_leaf = np.mean(y_true_leaf, axis=0)
            self.SI_estimators[leaf] = self._compute_si_estimator(y_true_leaf, y_pred_sa_leaf)

    def predict(self, X):
        """
        Predicts target values for the given feature matrix using the pre-computed MWS* estimators.
        """
        leaf_indices = self.tree.apply(X)
        predictions = np.array([self.SI_estimators[leaf] for leaf in leaf_indices])
        return predictions

    def score(self, X, y):
        y_pred = self.predict(X)
        mse_vals = mean_squared_error(y, y_pred, multioutput='uniform_average')
        return -mse_vals  # Scoring function should return higher values for better models

class PSIRF(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, dt_params=None, max_features='sqrt', bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        # Ensure dt_params is a dictionary
        self.dt_params = dt_params if dt_params is not None else {}
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []
        self.random_state = random_state

    def fit(self, X, y):
        self.trees = []
        X=np.array(X)
        y=np.array(y)
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_estimators):
            if self.bootstrap:
                # Adjusted to handle both DataFrame and numpy array
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

            tree = PSITree(**self.dt_params)
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

