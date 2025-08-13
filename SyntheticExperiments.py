
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from TMRT_self import TMRTree
from PJST_self import PJSTree
from PSIT_self import PSITree
from PSDT_self import PSDTree
from AJST import AJSTree
from ASIT import ASITree
from ASDT import ASDTree
import time
import numpy as np

n_features_list = [10, 15, 20]
n_targets_list = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
n_samples_list = [2000, 2500, 3000]

def generate_and_split_data(n_samples, n_features, n_targets, shuffle_random_state=42):
    X, Y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=int(n_features * 0.8),
                           n_targets=n_targets, noise=0.3, effective_rank=int(n_features * 0.8),
                           tail_strength=0.5, random_state=shuffle_random_state)

    # Initialize a RandomState object with a fixed seed
    rng = np.random.RandomState(shuffle_random_state)

    # Shuffle X and Y together using the controlled RandomState object
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    return X_shuffled, Y_shuffled


def split_data(X, Y, test_size=0.2, random_state=42):
    """
    Split data into training, validation, and testing sets. Normalize the target variables based on the training set.

    Args:
    X: Features dataframe.
    Y: Target dataframe.
    test_size: Proportion of the dataset to include in the test split.
    random_state: Controls the shuffling applied to the data before applying the split.

    Returns:
    Tuple containing training, validation, and testing datasets, with targets normalized: (X_train, X_val, X_test, Y_train_normalized, Y_val_normalized, Y_test_normalized).
    """
    # Split the dataset into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    # Further split the training set for validation purposes
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25,
                                                      random_state=random_state)  # Adjust the test_size as needed for your validation split

    # Normalize targets using StandardScaler
    scaler = StandardScaler()
    Y_train_normalized = scaler.fit_transform(Y_train)
    Y_val_normalized = scaler.transform(Y_val)
    Y_test_normalized = scaler.transform(Y_test)  # Apply the same transformation to Y_test

    return X_train, X_val, X_test, Y_train_normalized, Y_val_normalized, Y_test_normalized

def evaluate_and_retrain_models(X_train, X_val, X_test, y_train, y_val, y_test, models, param_options):
    results = {}
    for name, model in models.items():
        best_mse = np.inf
        best_params = None
        # Evaluate to find best hyperparameters
        for params in param_options[name]:
            model.set_params(**params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            mse = mean_squared_error(y_val, predictions)
            if mse < best_mse:
                best_mse = mse
                best_params = params
        # Retrain with the best parameters on the combined training and validation set
        print(f"Validation for {name} done. Best params: {best_params}")

        X_combined = np.vstack([X_train, X_val])
        y_combined = np.vstack([y_train, y_val])

        start_time = time.time()
        model.set_params(**best_params)
        model.fit(X_combined, y_combined)
        training_time = time.time() - start_time
        # Evaluate on the test set
        start_time = time.time()
        test_predictions = model.predict(X_test)
        prediction_time = time.time() - start_time
        test_mse = mean_squared_error(y_test, test_predictions)
        # Store results
        results[name] = {
            'best_params': best_params,
            'test_mse': test_mse,
            'training_time': training_time,
            'prediction_time': prediction_time
        }
    return results

models = {
'TMRTree': TMRTree(),
'PJSTree': PJSTree(),
'PSITree': PSITree(),
'PSDTree': PSDTree(),
'AJSTree': AJSTree(),
'ASITree': ASITree(),
'ASDTree': ASDTree(),
}

param_options = {
'TMRTree': [{'max_depth': d, 'min_samples_leaf': l} for d in [4, 6, 8, 10, 12, 14, 16] for l in [5, 10, 15, 20, 25, 30, 35]],
'PJSTree': [{'max_depth': d, 'min_samples_leaf': l} for d in [4, 6, 8, 10, 12, 14, 16] for l in [5, 10, 15, 20, 25, 30, 35]],
'PSITree': [{'max_depth': d, 'min_samples_leaf': l} for d in [4, 6, 8, 10, 12, 14, 16] for l in [5, 10, 15, 20, 25, 30, 35]],
'PSDTree': [{'max_depth': d, 'min_samples_leaf': l} for d in [4, 6, 8, 10, 12, 14, 16] for l in [5, 10, 15, 20, 25, 30, 35]],
'AJSTree': [{'max_depth': d, 'min_samples_leaf': l} for d in [4, 6, 8, 10, 12, 14, 16] for l in [5, 10, 15, 20, 25, 30, 35]],
'ASITree': [{'max_depth': d, 'min_samples_leaf': l} for d in [4, 6, 8, 10, 12, 14, 16] for l in [5, 10, 15, 20, 25, 30, 35]],
'ASDTree': [{'max_depth': d, 'min_samples_leaf': l} for d in [4, 6, 8, 10, 12, 14, 16] for l in [5, 10, 15, 20, 25, 30, 35]],
}

# dataset_details = pd.read_csv("C:/Users/Lenovo/OneDrive - The Hong Kong Polytechnic University/Desktop/multioutput regression/Code/Data/candidate/Datasets Details.csv")

all_datasets_results = []
# group_counter = 0
total_groups = len(n_features_list) * len(n_targets_list) * len(n_samples_list)
index = 0

for feature_count in n_features_list:
    for target_count in n_targets_list:
        for sample_size in n_samples_list:
            print(f"Processing dataset with {feature_count} features, {target_count} targets, and {sample_size} samples")
            # Load and preprocess the dataset

            # Store results for each dataset

            # Placeholder for results of the current dataset
            dataset_results = {
                'feature_count': feature_count,
                'target_count': target_count,
                'sample_size': sample_size,
                'split_results': [],
                'average_mse': {},
                'ranked_models': [],
                'average_training_time': {},
                'average_prediction_time': {}
            }

            # Temporary storage for accumulating MSE values for each model across splits
            mse_values = {model_name: [] for model_name in models.keys()}
            prediction_times = {model_name: [] for model_name in models.keys()}
            training_times = {model_name: [] for model_name in models.keys()}

            for split in range(5):
                X, Y = generate_and_split_data(sample_size, feature_count,target_count,shuffle_random_state=split)

                # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=split)
                X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, Y, test_size=0.2, random_state=split)

                split_results = evaluate_and_retrain_models(X_train, X_val, X_test, y_train, y_val, y_test, models,
                                                            param_options)

                # Collect MSE for each model
                for model_name, result in split_results.items():
                    mse_values[model_name].append(result['test_mse'])
                    training_times[model_name].append(result['training_time'])
                    prediction_times[model_name].append(result['prediction_time'])
                    dataset_results['split_results'].append({
                        'split': split,
                        'model': model_name,
                        'best_params': result['best_params'],
                        'test_mse': result['test_mse'],
                        'training_time': result['training_time'],
                        'prediction_time': result['prediction_time']
                    })

            for model_name, mses in mse_values.items():
                dataset_results['average_mse'][model_name] = np.mean(mses)

            for model_name, times in training_times.items():
                dataset_results['average_training_time'][model_name] = np.mean(times)

            for model_name, times in prediction_times.items():
                dataset_results['average_prediction_time'][model_name] = np.mean(times)

            ranked_models = sorted(dataset_results['average_mse'].items(), key=lambda x: x[1])
            dataset_results['ranked_models'] = ranked_models

            all_datasets_results.append(dataset_results)

            # Save results periodically or after processing all datasets
            if (index + 1) % 5 == 0 or index == total_groups - 1:
                df_results = pd.DataFrame([{
                    **res,
                    'average_mse': res['average_mse'],
                    'average_training_time': res['average_training_time'],
                    'average_prediction_time': res['average_prediction_time'],
                    'ranked_models': ', '.join([f"{model[0]}: {model[1]}" for model in res['ranked_models']])
                } for res in all_datasets_results])
                output_file = f'model_results_synthetic_{index + 1}.xlsx'
                df_results.to_excel(output_file, index=False)
                print(f"Results saved to {output_file}")
                # Reset for the next group of datasets, if batching
                all_datasets_results = []

            index += 1

# Check if there are remaining results to write after the loop
if all_datasets_results:
    df_results = pd.DataFrame([{
        **res,
        'average_mse': res['average_mse'],
        'average_training_time': res['average_training_time'],
        'average_prediction_time': res['average_prediction_time'],
        'ranked_models': ', '.join([f"{model[0]}: {model[1]}" for model in res['ranked_models']])
    } for res in all_datasets_results])
    output_file = f'model_results_remaining_synthetic.xlsx'
    df_results.to_excel(output_file, index=False)
    print(f"Remaining results saved to {output_file}")




# Example usage

#
# datasets = [generate_and_split_data(S, F, T) for S in n_samples_list for F in n_features_list for T in n_targets_list]






