
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
# from TMRT_self import TMRTree
from PJSRF import PJSTree, PJSRF
from PSIRF import PSITree, PSIRF
from PSDRF import PSDTree, PSDRF
# from AJST import AJSTree
# from ASIT import ASITree
# from ASDT import ASDTree
import time
import numpy as np

# Load the dataset details from the provided Excel file
file_path = "/Users/simonsmac/Desktop/Code/Data/candidate/Datasets Details.csv"
datasets_details = pd.read_csv(file_path)

def convert_parentheses_to_negative(value):
    """Convert string values formatted as '(number)' to negative numbers."""
    try:
        if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
            return -float(value[1:-1])
        return float(value)
    except ValueError:
        return value  # Return as is if conversion is not applicable

def load_dataset(dataset_details_row):
    """
    Load and preprocess the dataset based on details provided in dataset_details_row.

    Args:
    dataset_details_row: A dict or Series containing details about the dataset.

    Returns:
    Tuple containing file_name, features (X), targets (Y), feature_count, target_count, sample_size.
    """
    # Extract dataset details
    file_name = dataset_details_row['file name']
    feature_cols = dataset_details_row['features'].strip("'").split("','")
    target_cols = dataset_details_row['targets'].strip("'").split("','")
    feature_count = dataset_details_row['feature number']
    target_count = dataset_details_row['target number']
    sample_size = dataset_details_row['sample size']

    # Assuming datasets are stored in a predefined accessible directory
    file_path = f"/Users/simonsmac/Desktop/Code/Data/candidate/{file_name}"

    # Load the dataset
    df = pd.read_csv(file_path)

    # Apply conversion for negative values
    for col in df.columns:
        df[col] = df[col].apply(convert_parentheses_to_negative)

    # Split into features (X) and targets (Y)
    X = df[feature_cols]
    Y = df[target_cols]

    return file_name, X, Y, feature_count, target_count, sample_size


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
        test_predictions = model.predict(X_test.to_numpy())
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
'RF': RandomForestRegressor(),
'PJSRF': PJSRF(),
'PSIRF': PSIRF(),
'PSDRF': PSDRF()
}

param_options = {
    'RF': [
        {'n_estimators': n, 'max_depth': d, 'min_samples_leaf': l, 'max_features': None, 'bootstrap': b}
        for n in [100, 200]
        for d in [4, 6, 8, 10, 12, 14, 16]
        for l in [5, 10, 15, 20, 25, 30, 35]
        for b in [True]
    ],
    'PJSRF': [
        {'n_estimators': n, 'max_features': None, 'bootstrap': b, 'dt_params': {'max_depth': d, 'min_samples_leaf': l}}
        for n in [100, 200]
        for d in [4, 6, 8, 10, 12, 14, 16]
        for l in [5, 10, 15, 20, 25, 30, 35]
        for b in [True]
    ],
    'PSIRF': [
        {'n_estimators': n, 'max_features': None, 'bootstrap': b, 'dt_params': {'max_depth': d, 'min_samples_leaf': l}}
        for n in [100, 200]
        for d in [4, 6, 8, 10, 12, 14, 16]
        for l in [5, 10, 15, 20, 25, 30, 35]
        for b in [True]
    ],
    'PSDRF': [
        {'n_estimators': n, 'max_features': None, 'bootstrap': b, 'dt_params': {'max_depth': d, 'min_samples_leaf': l}}
        for n in [100, 200]
        for d in [4, 6, 8, 10, 12, 14, 16]
        for l in [5, 10, 15, 20, 25, 30, 35]
        for b in [True]
    ]
}

dataset_details = pd.read_csv("/Users/simonsmac/Desktop/Code/Data/candidate/Datasets Details.csv")

all_datasets_results = []

for index, row in dataset_details.iterrows():

    print(row)
    file_name, X, Y, feature_count, target_count, sample_size = load_dataset(row)

    # Store results for each dataset

    # Placeholder for results of the current dataset
    dataset_results = {
        'dataset_name': file_name,
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

    for model_name, mses  in mse_values.items():
        dataset_results['average_mse'][model_name] = np.mean(mses)

    for model_name, times in training_times.items():
        dataset_results['average_training_time'][model_name] = np.mean(times)

    for model_name, times in prediction_times.items():
        dataset_results['average_prediction_time'][model_name] = np.mean(times)

    ranked_models = sorted(dataset_results['average_mse'].items(), key=lambda x: x[1])
    dataset_results['ranked_models'] = ranked_models

    all_datasets_results.append(dataset_results)

    # Save results periodically or after processing all datasets
    if (index + 1) % 5 == 0 or index == len(dataset_details) - 1:
        df_results = pd.DataFrame([{
            **res,
            'average_mse': res['average_mse'],
            'average_training_time': res['average_training_time'],
            'average_prediction_time': res['average_prediction_time'],
            'ranked_models': ', '.join([f"{model[0]}: {model[1]}" for model in res['ranked_models']])
        } for res in all_datasets_results])
        output_file = f'model_results_{index + 1}_realworld_forest.xlsx'
        df_results.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
        # Reset for the next group of datasets, if batching
        all_datasets_results = []

# Check if there are remaining results to write after the loop
if all_datasets_results:
    df_results = pd.DataFrame([{
        **res,
        'average_mse': res['average_mse'],
        'average_training_time': res['average_training_time'],
        'average_prediction_time': res['average_prediction_time'],
        'ranked_models': ', '.join([f"{model[0]}: {model[1]}" for model in res['ranked_models']])
    } for res in all_datasets_results])
    output_file = f'model_results_remaining_realworld_forest.xlsx'
    df_results.to_excel(output_file, index=False)
    print(f"Remaining results saved to {output_file}")


