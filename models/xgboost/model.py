import numpy as np
from xgboost import XGBRegressor
from dataset.dataset import (
    FEATURE_COLS_SIR,
    FEATURE_COLS_INTERVENTIONS,
    FEATURE_COLS_STATIC,
    LABEL_COLS,
)

IS_PYTORCH = False

# GRID SEARCH
# Lr ++ seems to be good
# subsample ++
# colsample_bytree --
# gamma ++ (at 0.01) (bad at 0.1)
# reg_alpha -- (try only at low value, > 0.1 is bad)
# reg_lambda invariant
# min_child_weight invariant
# max_depth to be explored


class Model:
    def __init__(self, input_size, is_deltas):
        self.is_deltas = is_deltas

        self.model = XGBRegressor(
            n_estimators=100,  # Reduced from 10000 to prevent overfitting
            learning_rate=0.15,  # Slower learning rate for better generalization
            subsample=0.8,  # Use 80% of data per tree to prevent overfitting
            # gamma=0.01,  # Minimum loss reduction for split
            # reg_alpha=0.005,  # L1 regularization
            # reg_lambda=1,  # L2 regularization
            # min_child_weight=1,  # Minimum sum of instance weight in a child
            # max_depth=15,  # Control tree depth to prevent overfitting
            objective="reg:squarederror",
            n_jobs=-1,  # Use all CPU cores
        )
        self.is_fitted = False


def train_model(model, train_data, val_data, num_epochs):
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

    # column_mapping = {}

    # for col in LABEL_COLS:
    #     column_mapping[col] = col.split("Label_")[1]

    # y_train = y_train.rename(columns=column_mapping)
    # y_train = (y_train - train_data[FEATURE_COLS_SIR]).values

    # print()
    # print(y_train - train_data[FEATURE_COLS_SIR])

    # exit()

    # Split targets for students and adults
    # y_train_students = y_train[:, :3]
    # y_train_adults = y_train[:, 3:]

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

    model_data = {
        "model": model,
        "is_fitted": model.is_fitted,
    }
    joblib.dump(model_data, path)


def load_model(model, path):
    import joblib

    model_data = joblib.load(path)
    model = model_data["model"]
    model.is_fitted = model_data["is_fitted"]


def calculate_loss_df(model, data):
    # Get features and labels from DataFrame
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
