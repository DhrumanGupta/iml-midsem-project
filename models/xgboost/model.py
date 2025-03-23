import numpy as np
from xgboost import XGBRegressor
from dataset.dataset import (
    FEATURE_COLS_SIR,
    FEATURE_COLS_INTERVENTIONS,
    FEATURE_COLS_STATIC,
    LABEL_COLS,
)
from sklearn.model_selection import GridSearchCV
import os
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed

IS_PYTORCH = False
AUTOREGRESSIVE = False
# GRID SEARCH
# Lr ++ seems to be good
# subsample ++
# colsample_bytree --
# gamma ++ (at 0.01) (bad at 0.1)
# reg_alpha -- (try only at low value, > 0.1 is bad)
# reg_lambda invariant
# min_child_weight invariant
# max_depth to be explored

# Hyperparameter search space for grid search (sklearn format)
PARAM_GRID = {
    "n_estimators": [100, 300, 500],
    # "learning_rate": [0.01, 0.1, 0.2, 0.3],
    # "subsample": [0.5, 0.75, 1.0],
    # "colsample_bytree": [0.5, 0.75, 1.0],
    # "gamma": [0.001, 0.01, 0.1],
    # "reg_alpha": [0.001, 0.01, 0.1],
    # "reg_lambda": [0.001, 0.01, 0.1, 1.0],
    # "max_depth": [6, 9, 12],
    # "min_child_weight": [3, 5, 7],
}



class Model:
    def __init__(
        self,
        input_size,
        is_deltas,
        config={
            "colsample_bytree": 0.75,
            "learning_rate": 0.1,
            "max_depth": 10,
            "n_estimators": 300,
        },
        n_jobs=2,
    ):
        self.is_deltas = is_deltas

        self.model = XGBRegressor(
            **config,
            objective="reg:squarederror",
            n_jobs=n_jobs,
        )
        self.is_fitted = False


def train_model(model, train_data, val_data, num_epochs, loss_fn):
    # Get features and labels from DataFrames
    X_train = np.concatenate(
        [
            train_data[FEATURE_COLS_SIR].values,
            train_data[FEATURE_COLS_INTERVENTIONS].values,
            train_data[FEATURE_COLS_STATIC].values,
        ],
        axis=1,
    )

    y_train = train_data[LABEL_COLS].values

    # Train models
    model.model.fit(X_train, y_train)
    model.is_fitted = True

    # Calculate training and validation losses
    train_loss = calculate_loss_df(model, train_data)
    val_loss = calculate_loss_df(model, val_data)

    # Yield losses for each "epoch" (though XGBoost doesn't use epochs in the same way)
    # for epoch in range(num_epochs):
    yield train_loss, val_loss, 0


def normalize_predictions(predictions, is_deltas=False):
    # Split predictions into students and adults
    students = predictions[:, :3]
    adults = predictions[:, 3:]

    if is_deltas:
        # For deltas, ensure each group sums to 0
        students = students - (students.sum(axis=1, keepdims=True) / 3)
        adults = adults - (adults.sum(axis=1, keepdims=True) / 3)
    else:
        # For absolute values, normalize to sum to 1
        students = students / students.sum(axis=1, keepdims=True)
        adults = adults / adults.sum(axis=1, keepdims=True)

    # Concatenate back together
    return np.concatenate([students, adults], axis=1)


def predict(model, x_sir, x_interventions, x_static) -> np.ndarray:
    if not model.is_fitted:
        raise RuntimeError("Model must be trained before prediction")

    # Combine features
    X = np.concatenate([x_sir, x_interventions, x_static], axis=1)

    # Get predictions for both groups
    pred = model.model.predict(X)

    if len(pred.shape) == 1:
        pred = pred.reshape(1, -1)

    normalized_pred = normalize_predictions(pred, model.is_deltas)

    # Combine predictions
    return normalized_pred


def save_model(model, path):
    import joblib

    # Save the XGBoost model and relevant attributes
    model_data = {
        "xgb_model": model.model,  # Save the actual XGBoost model
        "is_fitted": model.is_fitted,
        "is_deltas": model.is_deltas,  # Also save is_deltas attribute
    }
    joblib.dump(model_data, path)


def load_model(model, path):
    import joblib

    # Load the saved model data
    model_data = joblib.load(path)

    # Update the passed model's attributes
    model.model = model_data["xgb_model"]
    model.is_fitted = model_data["is_fitted"]
    model.is_deltas = model_data["is_deltas"]

    # Return the updated model for convenience
    return model


def calculate_loss_df(model, data):
    # Get features and labels from DataFrames
    X = np.concatenate(
        [
            data[FEATURE_COLS_SIR].values,
            data[FEATURE_COLS_INTERVENTIONS].values,
            data[FEATURE_COLS_STATIC].values,
        ],
        axis=1,
    )

    labels = data[LABEL_COLS].values

    # Get predictions
    pred = model.model.predict(X)

    # Calculate MSE loss
    loss = np.mean((pred - labels) ** 2)

    return loss


def grid_search(
    train_data,
    val_data,
    is_deltas,
    loss_fn,
    n_jobs=4,  # Number of parallel jobs
    top_n=10,  # Number of top models to return
):
    """
    Perform hyperparameter optimization by training individual models.

    Args:
        train_data: Training data DataFrame
        val_data: Validation data DataFrame
        is_deltas: Whether the model predicts deltas or absolute values
        loss_fn: Function to evaluate model performance on validation data
        n_jobs: Number of parallel jobs (default: 4)
        top_n: Number of top models to return (default: 10)

    Returns:
        sorted_models: List of dictionaries containing models and their configs,
                       sorted from best to worst performance
    """

    # Function to train and evaluate a single model configuration
    def train_and_evaluate(config):
        # Create model with the configuration
        model = Model(
            input_size=train_data[FEATURE_COLS_SIR].shape[1],
            is_deltas=is_deltas,
            config=config,
        )

        # Prepare training data
        X_train = np.concatenate(
            [
                train_data[FEATURE_COLS_SIR].values,
                train_data[FEATURE_COLS_INTERVENTIONS].values,
                train_data[FEATURE_COLS_STATIC].values,
            ],
            axis=1,
        )
        y_train = train_data[LABEL_COLS].values

        # Train the model
        model.model.fit(X_train, y_train)
        model.is_fitted = True

        # Evaluate the model using the provided loss_fn
        val_loss = loss_fn(model)

        return {"model": model, "config": config, "val_loss": val_loss}

    # Generate all parameter combinations
    param_combinations = list(ParameterGrid(PARAM_GRID))
    print(f"Testing {len(param_combinations)} parameter combinations")

    # Train and evaluate models in parallel
    print(f"Starting parallel training with {n_jobs} jobs")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(train_and_evaluate)(config) for config in param_combinations
    )

    # Sort models by validation loss (ascending)
    sorted_models = sorted(results, key=lambda x: x["val_loss"])

    # Keep only the top N models
    sorted_models = sorted_models[:top_n]

    # Print results
    print(f"\nTop {len(sorted_models)} hyperparameter configurations found:")
    for i, result in enumerate(sorted_models):
        config = result["config"]
        val_loss = result["val_loss"]

        print(f"\nRank {i+1} (val_loss: {val_loss:.6f}):")
        for param, value in config.items():
            print(f"  {param}: {value}")

    return sorted_models
