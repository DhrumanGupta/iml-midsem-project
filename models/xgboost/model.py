import numpy as np
from xgboost import XGBRegressor
from dataset.dataset import (
    FEATURE_COLS_SIR,
    FEATURE_COLS_INTERVENTIONS,
    FEATURE_COLS_STATIC,
    LABEL_COLS,
)
from ray import train, tune
from ray.tune.search import ConcurrencyLimiter, BasicVariantGenerator
from ray.tune.schedulers import ASHAScheduler
import os

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

# Hyperparameter search space for grid search
SEARCH_SPACE = {
    "n_estimators": tune.grid_search([50, 100, 300, 500]),
    # "learning_rate": tune.grid_search([0.01, 0.1, 0.3]),
    # "subsample": tune.grid_search([0.5, 0.75, 1.0]),
    # "colsample_bytree": tune.grid_search([0.5, 0.75, 1.0]),
    # "gamma": tune.grid_search([0.001, 0.01, 0.1]),
    # "reg_alpha": tune.grid_search([0.001, 0.01, 0.1]),
    # "reg_lambda": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
    # "max_depth": tune.grid_search([3, 5, 7, 9, 11]),
    # "min_child_weight": tune.grid_search([1, 3, 5, 7]),
}


class Model:
    def __init__(
        self,
        input_size,
        is_deltas,
        config={
            "n_estimators": 100,
            "learning_rate": 0.15,
            "subsample": 0.8,
        },
    ):
        self.is_deltas = is_deltas

        self.model = XGBRegressor(
            **config,
            objective="reg:squarederror",
            n_jobs=-1,  # Use all CPU cores
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


# Training function for a single trial
def train_xgb_tune(config, train_data, val_data, is_deltas, loss_fn):
    # Create model with the trial's hyperparameters
    model = Model(
        input_size=train_data[FEATURE_COLS_SIR].shape[1],
        is_deltas=is_deltas,
        config=config,
    )

    # Train model
    X_train = np.concatenate(
        [
            train_data[FEATURE_COLS_SIR].values,
            train_data[FEATURE_COLS_INTERVENTIONS].values,
            train_data[FEATURE_COLS_STATIC].values,
        ],
        axis=1,
    )
    y_train = train_data[LABEL_COLS].values

    model.model.fit(X_train, y_train)
    model.is_fitted = True

    val_loss = loss_fn(model)

    # Report metrics to Ray Tune
    train.report({"val_loss": val_loss})


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
    max_concurrent_trials=8,
    top_n=10,  # Number of top models to return
):
    """
    Perform hyperparameter optimization using Ray Tune.

    Args:
        train_data: Training data DataFrame
        val_data: Validation data DataFrame
        is_deltas: Whether the model predicts deltas or absolute values
        loss_fn: Function to evaluate model performance
        max_concurrent_trials: Maximum number of trials to run in parallel
        top_n: Number of top models to return (default: 10)

    Returns:
        sorted_models: List of dictionaries containing models and their configs,
                       sorted from best to worst performance
    """
    # Set up the trainable function with parameters
    trainable_func = tune.with_parameters(
        train_xgb_tune,
        train_data=train_data,
        val_data=val_data,
        is_deltas=is_deltas,
        loss_fn=loss_fn,
    )

    # Configure the tuner
    tuner = tune.Tuner(
        tune.with_resources(trainable_func, {"cpu": 1}),
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            max_concurrent_trials=max_concurrent_trials,
        ),
        param_space=SEARCH_SPACE,
    )

    # Execute the optimization
    results = tuner.fit()

    # Get the top N configurations and results
    top_results = results.get_dataframe().sort_values("val_loss").head(top_n)
    columns = [f"config/{key}" for key in SEARCH_SPACE.keys()]
    top_configs = [dict(row) for _, row in top_results[columns].iterrows()]
    val_losses = top_results["val_loss"].tolist()

    print(f"Top {top_n} hyperparameter configurations found:")
    for i, (config, val_loss) in enumerate(zip(top_configs, val_losses)):
        print(f"\nRank {i+1} (val_loss: {val_loss:.6f}):")
        for param, value in config.items():
            print(f"  {param}: {value}")

    # Create a list to store models with their configs
    sorted_models = []

    # Train models with the top configurations and store with their configs
    for i, (config, val_loss) in enumerate(zip(top_configs, val_losses)):
        model = Model(
            input_size=train_data[FEATURE_COLS_SIR].shape[1],
            is_deltas=is_deltas,
            config=config,
        )

        X_train = np.concatenate(
            [
                train_data[FEATURE_COLS_SIR].values,
                train_data[FEATURE_COLS_INTERVENTIONS].values,
                train_data[FEATURE_COLS_STATIC].values,
            ],
            axis=1,
        )
        y_train = train_data[LABEL_COLS].values

        model.model.fit(X_train, y_train)
        model.is_fitted = True

        # Store model and its config together
        sorted_models.append({"model": model, "config": config, "val_loss": val_loss})

    return sorted_models
