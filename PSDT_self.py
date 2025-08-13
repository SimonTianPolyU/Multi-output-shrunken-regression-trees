from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from sklearn.metrics import mean_squared_error
from collections import deque
import copy
from numpy.random import RandomState

class PSDTree:
    def __init__(self, max_depth=1, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = {}
        self.n_outputs_ = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_outputs_ = y.shape[1] if y.ndim == 2 else 1
        self._initialize_tree()
        self._build_tree()
        return self

    def _initialize_tree(self):
        self.tree = {
                0: {'depth': 0, 'samples': list(range(len(self.X)))}
            }
        self._compute_node_value(0)

    def _build_tree(self):
        stack = deque([0])
        while stack:
            node_id = stack.pop()
            node = self.tree[node_id]
            if self._is_leaf(node):
                continue
            best_split = self._find_best_split(node_id)
            if best_split:
                left_id, right_id = self._split_node(node_id, best_split)
                stack.extend([left_id, right_id])

    def _is_leaf(self, node):
        return (node['depth'] >= self.max_depth or
                len(node['samples']) <= 2 * self.min_samples_leaf - 1)

    def _compute_node_value(self, node_id):
    #   Compute the value of the node as the JS estimators of the targets
        samples = self.tree[node_id]['samples']
        y_m = [self.y[i] for i in samples]
        self.tree[node_id]['value'] = self.compute_SD_estimators(y_m)

    def compute_SD_estimators(self, y_m):
        y_true = np.array(y_m)  # Get the true labels of the samples
        y_pred_sa = np.mean(y_true, axis=0)  # Compute the simple average prediction
        N_m = y_true.shape[0]  # Get the number of samples in the node

        squared_diff_sum = sum(
            (y_pred_sa[j] - y_pred_sa[j_prime]) ** 2
            for j in range(self.n_outputs_-1) for j_prime in range(j+1, self.n_outputs_)
        )

        # Calculate sum of variances
        residuals = y_true - y_pred_sa
        covariance = np.cov(residuals, rowvar=False, ddof=1)
        trace_cov_matrix = np.trace(covariance)
        sum_cov_matrix = np.sum(covariance)

        # if trace_cov_matrix <= sum_cov_matrix:
        #     return y_pred_sa

        # Calculate the denominator according to the formula
        denominator = N_m * squared_diff_sum + self.n_outputs_ * trace_cov_matrix - sum_cov_matrix

        if denominator == 0:
            return y_pred_sa

        # Calculate the adjustment matrix according to the formula
        adjustment_matrix = (N_m * squared_diff_sum * np.identity(self.n_outputs_) +
                    (self.n_outputs_ * trace_cov_matrix - sum_cov_matrix) *
                        np.ones((self.n_outputs_, self.n_outputs_)) / self.n_outputs_)

        # Calculate the SC estimator
        y_pred_SC = (1 / denominator) * adjustment_matrix.dot(y_pred_sa)

        return y_pred_SC  # Return the JS prediction

    def predict(self, X):
        X = np.array(X)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _find_best_split(self, node_id):
        node = self.tree[node_id]
        samples = node['samples']
        best_impurity = float('inf')
        best_split = None

        for feature_index in range(self.X.shape[1]):
            values = np.unique(self.X[samples, feature_index])

            # If the feature is one-hot encoded
            if np.array_equal(values, [0, 1]):
                # theta_p = 0.5  # split point between 0 and 1
                unique_values = [0.5]
            else:
                # If not one-hot encoded, use the binning technique
                min_val, max_val = values[0], values[-1]
                bin_edges = np.linspace(min_val, max_val, 11)  # creates 10 split points
                # Skip the first and the last edge since they represent the min and max values.
                unique_values = bin_edges[1:-1]

            for split_value in unique_values:
                left_samples = [i for i in samples if self.X[i, feature_index] <= split_value]
                right_samples = [i for i in samples if self.X[i, feature_index] > split_value]

                if len(left_samples) < self.min_samples_leaf or len(right_samples) < self.min_samples_leaf:
                    continue  # Skip if the split doesn't respect the minimum sample constraint

                impurity = self._calculate_mean_impurity(left_samples, right_samples)

                if impurity < best_impurity:
                    best_impurity = impurity
                    best_split = {'feature_index': feature_index, 'split_value': split_value,
                                      'left_samples': left_samples, 'right_samples': right_samples}

        return best_split

    def _split_node(self, node_id, split_params):
        feature_index = split_params['feature_index']
        split_value = split_params['split_value']
        left_samples = split_params['left_samples']
        right_samples = split_params['right_samples']

        left_id = len(self.tree)
        right_id = left_id + 1

        self.tree[node_id]['split'] = (feature_index, split_value)
        self.tree[node_id]['left_child'] = left_id
        self.tree[node_id]['right_child'] = right_id
        self.tree[left_id] = {'depth': self.tree[node_id]['depth'] + 1, 'samples': left_samples}
        self.tree[right_id] = {'depth': self.tree[node_id]['depth'] + 1, 'samples': right_samples}

        self._compute_node_value(left_id)
        self._compute_node_value(right_id)

        return left_id, right_id

    def _calculate_mean_impurity(self, left_samples, right_samples):
        # Extract the actual y values for left and right splits
        y_left = self.y[left_samples]
        y_right = self.y[right_samples]

        # Calculate the mean estimator predictions for left and right splits
        left_mean = np.mean(y_left, axis=0)
        right_mean = np.mean(y_right, axis=0)

        # Expand the mean estimators to match the shape of y_left and y_right for broadcasting
        left_mean_expanded = np.tile(left_mean, (len(left_samples), 1))
        right_mean_expanded = np.tile(right_mean, (len(right_samples), 1))

        # Calculate the total sum of squared errors (SSE) for each split
        # Note: np.sum() is used here instead of np.mean() to get the total SSE
        SSE_left = np.sum((y_left - left_mean_expanded) ** 2)
        SSE_right = np.sum((y_right - right_mean_expanded) ** 2)

        # Sum the SSE of both splits to get the total impurity decrease
        total_SSE = SSE_left + SSE_right

        return total_SSE

    def _predict_single(self, x):
        node_id = 0

        while True:
            children = self.tree[node_id].get('left_child', None), self.tree[node_id].get('right_child', None)

            if not children or children == (None, None):
                return self.tree[node_id]['value']

            # Ensure that children[0] is not None
            if children[0] is None:
                raise ValueError(f"Invalid child node for parent node {node_id}.")

            # Ensure feature_index is treated as an integer
            feature_index = int(self.tree[node_id]['split'][0])  # Get feature for current node

            # Ensure threshold is treated as a float
            threshold = float(self.tree[node_id]['split'][1])  # Get threshold for that feature

            # Ensure test_sample's feature value is treated as a float
            feature_value = float(x[feature_index])

            # Based on the test sample's value for the feature, choose the next node
            if feature_value <= threshold:
                node_id = children[0]
            else:
                node_id = children[1]

    def get_params(self):
        """Get parameters for this estimator."""
        return {'max_depth': self.max_depth, 'min_samples_leaf': self.min_samples_leaf}

    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class PSDRF(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, dt_params=None, max_features='sqrt', bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        # Ensure dt_params is a dictionary
        self.dt_params = dt_params if dt_params is not None else {}
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_estimators):
            if self.bootstrap:
                indices = rng.choice(n_samples, size=n_samples, replace=True)
                X_sample, y_sample = X[indices], y[indices]
            else:
                X_sample, y_sample = X, y

            if self.max_features == 'sqrt':
                max_features = int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                max_features = int(np.log2(n_features))
            elif isinstance(self.max_features, float):
                max_features = int(self.max_features * n_features)
            else:
                max_features = self.max_features

            features = rng.choice(n_features, max_features, replace=False)

            tree = PSDTree(**self.dt_params)
            tree.fit(X_sample[:, features], y_sample)
            self.trees.append((tree, features))

        return self

    def predict(self, X):
        # Ensure the initial predictions array is correctly shaped
        # Assuming all trees have the same number of outputs, use the first tree to determine this
        if len(self.trees) > 0:
            n_outputs_ = self.trees[0][0].n_outputs_
        else:
            raise ValueError("No trees in the forest.")

        predictions = np.zeros((X.shape[0], n_outputs_))
        for tree, features in self.trees:
            # Aggregate predictions
            predictions += tree.predict(X[:, features])
        # Average the predictions from all trees
        return predictions / self.n_estimators


