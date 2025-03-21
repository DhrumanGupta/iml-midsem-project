import os
import sys

import numpy as np
import pandas as pd
from dataset.data_loader import load_data
from dataset.dataset import FEATURE_SIZE, SimulationDataset
from logger import logger
import importlib

TRAIN_MODEL = False
IS_DELTAS = True
MODEL_TO_LOAD = "model_1.pth"
EPOCHS = 50
BATCH_SIZE = 256

PLOT_TEST = False
PLOT_FOR_ALL = False

IS_TEST_MODE = True


INITIAL_CONFIG = {
    "sir": [0.999, 0.001, 0, 0.999, 0.001, 0],
    "static": [0.3, 0.7, 6, 1500, 300, 0.3, 1 / 14],
    "school_lockdown_days": [35],
    "school_lockdown_intensities": [0.9],
    "school_lockdown_durations": [21],
    "office_lockdown_days": [35],
    "office_lockdown_intensities": [0.8],
    "office_lockdown_durations": [21],
}

models_dict = {}
for item in os.listdir(os.path.dirname(__file__)):
    model_dir = os.path.join(os.path.dirname(__file__), item)
    init_file = os.path.join(model_dir, "__init__.py")

    # Check if it's a directory with an __init__.py file (a proper module)
    if os.path.isdir(model_dir) and os.path.isfile(init_file) and item != "__pycache__":
        try:
            # Import the module dynamically
            model_module = importlib.import_module(f"models.{item}")
            # Check if it has the required attributes
            required_attrs = [
                "Model",
                "IS_PYTORCH",
                "save_model",
                "load_model",
                "train_model",
                "predict",
            ]
            if all(hasattr(model_module, attr) for attr in required_attrs):
                models_dict[item] = model_module
                # print(f"Discovered model: {item}")
        except (ImportError, AttributeError) as e:
            print(f"Skipping {item}: {e}")


def load_model(model_name, path):
    model = models_dict[model_name]
    model_instance = model.Model(input_size=FEATURE_SIZE, is_deltas=IS_DELTAS)
    model.load_model(model_instance, path)
    return model_instance


def run_test(model, model_instance, initial_config):
    if (
        not "sir" in initial_config
        or len(initial_config["sir"]) != 6
        or sum(initial_config["sir"][:3]) != 1
        or sum(initial_config["sir"][3:]) != 1
    ):
        raise ValueError(
            "initial_config must contain a 'sir' key with a 6-element array"
        )

    if (
        not "static" in initial_config
        or len(initial_config["static"]) != 7
        or sum(initial_config["static"][:2]) != 1
    ):
        raise ValueError(
            "initial_config must contain a 'static' key with a 7-element array"
        )
    sir = [np.array(initial_config["sir"])]

    school_lockdown_days = initial_config.get("school_lockdown_days", [])
    school_lockdown_intensities = initial_config.get("school_lockdown_intensities", [])
    school_lockdown_durations = initial_config.get("school_lockdown_durations", [])

    office_lockdown_days = initial_config.get("office_lockdown_days", [])
    office_lockdown_intensities = initial_config.get("office_lockdown_intensities", [])
    office_lockdown_durations = initial_config.get("office_lockdown_durations", [])

    static = np.array(initial_config["static"])

    for i in range(150):
        sir_last = sir[-1]
        school_intensity = 0
        office_intensity = 0

        # Check if there are active school lockdowns for the current day
        for day, duration, intensity in zip(
            school_lockdown_days, school_lockdown_durations, school_lockdown_intensities
        ):
            if day <= i < day + duration:
                school_intensity = intensity
                break

        # Check if there are active office lockdowns for the current day
        for day, duration, intensity in zip(
            office_lockdown_days, office_lockdown_durations, office_lockdown_intensities
        ):
            if day <= i < day + duration:
                office_intensity = intensity
                break

        interventions_curr = [school_intensity, office_intensity]

        predictions = model.predict(
            model_instance,
            np.array([sir_last]),
            np.array([interventions_curr]),
            np.array([static]),
        )

        if IS_DELTAS:
            predictions = predictions + np.array([sir_last])

        sir.append(predictions[0])

    # Plot and show this

    sir_array = np.vstack(sir)

    return sir_array

    # import matplotlib.pyplot as plt

    # plt.cla()
    # plt.clf()
    # plt.close()

    # plt.figure(figsize=(21, 10))

    # plt.plot(sir_array[:, 0], label="Students Susceptible")
    # plt.plot(sir_array[:, 1], label="Students Infected")
    # plt.plot(sir_array[:, 2], label="Students Recovered")
    # plt.plot(sir_array[:, 3], label="Adults Susceptible")
    # plt.plot(sir_array[:, 4], label="Adults Infected")
    # plt.plot(sir_array[:, 5], label="Adults Recovered")

    # susceptible = sir_array[:, 0] * static[0] + sir_array[:, 3] * static[1]
    # infected = sir_array[:, 1] * static[0] + sir_array[:, 4] * static[1]
    # recovered = sir_array[:, 2] * static[0] + sir_array[:, 5] * static[1]

    # plt.title(f"Beta = {static[5]}, Gamma = {static[6]}")

    # plt.plot(susceptible, label="Susceptible", color="blue")
    # plt.plot(infected, label="Infected", color="red")
    # plt.plot(recovered, label="Recovered", color="green")

    # plt.show()


def plot_for_model(model, model_instance):
    split_info = np.load(
        os.path.join("dataset", "split_indices.npy"), allow_pickle=True
    ).item()
    for type in ["train", "test"]:
        print(f'\nPlotting for "{type}"')
        test_idx = np.array(split_info[type])[0]

        # Load the test data
        test_df = pd.read_csv(
            os.path.join("dataset", "processed_data", f"{test_idx + 1}.csv")
        )

        # print(test_df.head())

        data_loader = SimulationDataset(test_df)
        # dynamic, fixed
        sir = [data_loader[0][0]]

        last_school = 0
        last_work = 0

        for i in range(150):
            _, x_intervention, x_static, label = data_loader[i]
            x_sir = sir[-1]
            curr_school = x_intervention[0]
            curr_work = x_intervention[1]

            if curr_school != last_school:
                print(f"[{i}] School changed from {last_school} to {curr_school}")
                last_school = curr_school

            if curr_work != last_work:
                print(f"[{i}] Work changed from {last_work} to {curr_work}")
                last_work = curr_work

            outputs: np.ndarray = model.predict(
                model_instance,
                np.array([x_sir]),
                np.array([x_intervention]),
                np.array([x_static]),
            )

            if IS_DELTAS:
                outputs = outputs + np.array([x_sir])

            sir.append(outputs[0])

        sir.pop(0)
        sir_array = np.vstack(sir)

        # print(sir_array)

        # Plot the data
        import matplotlib.pyplot as plt

        # Plot each column of the SIR data
        # Sir is a list of numpy arrays
        # Each numpy array has shape (6)
        # Plot each column as a separate line
        # students_infected = 1
        # adults_infected = 4

        # print(sir_array[:, 0])

        plt.figure(figsize=(10, 10))

        plt.subplot(2, 1, 1)
        plt.plot(
            sir_array[:, 0],
            label="Susceptible - Predicted",
            color="blue",
            alpha=0.5,
            linestyle="--",
        )
        plt.plot(
            test_df["S_Students"],
            label="Susceptible - Real",
            color="blue",
            alpha=0.5,
        )

        plt.plot(
            sir_array[:, 1],
            label="Infected - Predicted",
            color="red",
            alpha=0.5,
            linestyle="--",
        )
        plt.plot(
            test_df["I_Students"],
            label="Infected - Real",
            color="red",
            alpha=0.5,
        )

        plt.plot(
            sir_array[:, 2],
            label="Recovered - Predicted",
            color="green",
            alpha=0.5,
            linestyle="--",
        )
        plt.plot(
            test_df["R_Students"],
            label="Recovered - Real",
            color="darkgreen",
            alpha=0.5,
        )

        plt.title("Student Population SIR Model")
        plt.xlabel("Time Steps")
        plt.ylabel("Ratio")
        # plt.legend()
        plt.grid(True)

        # Plot adult ratios
        plt.subplot(2, 1, 2)
        plt.plot(
            sir_array[:, 3],
            label="Susceptible - Predicted",
            color="blue",
            alpha=0.5,
            linestyle="--",
        )
        plt.plot(
            test_df["S_Adults"],
            label="Susceptible - Real",
            color="blue",
            alpha=0.5,
        )

        plt.plot(
            sir_array[:, 4],
            label="Infected - Predicted",
            color="red",
            alpha=0.5,
            linestyle="--",
        )
        plt.plot(
            test_df["I_Adults"],
            label="Infected - Real",
            color="red",
            alpha=0.5,
        )

        plt.plot(
            sir_array[:, 5],
            label="Recovered - Predicted",
            color="green",
            alpha=0.5,
            linestyle="--",
        )
        plt.plot(
            test_df["R_Adults"],
            label="Recovered - Real",
            color="green",
            alpha=0.5,
        )

        plt.title("Adult Population SIR Model")
        plt.xlabel("Time Steps")
        plt.ylabel("Ratio")
        # plt.legend()
        plt.tight_layout()
        plt.grid(True)

        # Get next index which is not taken
        idx = 0
        postfix = "deltas" if IS_DELTAS else "absolute"
        directory = f"plots/{model}/{postfix}"
        os.makedirs(directory, exist_ok=True)
        while os.path.exists(f"{directory}/plot_{idx}_{type}.png"):
            idx += 1

        plt.savefig(f"{directory}/plot_{idx}_{type}.png")

        plt.cla()
        plt.clf()

        # plt.show()


# Add these global variables for caching
_cached_test_ids = None
_cached_test_dfs = None
_cached_data_loaders = None


def evaluate_model(model, model_instance):
    global _cached_test_ids, _cached_test_dfs, _cached_data_loaders

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Use cached values if available, otherwise load from disk
    if _cached_test_ids is None:

        # Load the test indices.
        _cached_test_ids = np.load(
            os.path.join(project_root, "dataset", "split_indices.npy"),
            allow_pickle=True,
        ).item()["test"]

    if _cached_test_dfs is None or _cached_data_loaders is None:
        # Load all test datasets and create a list of SimulationDataset instances.
        _cached_data_loaders = []
        _cached_test_dfs = []  # Store dataframes for plotting
        for test_id in _cached_test_ids:
            test_df = pd.read_csv(
                os.path.join(
                    project_root, "dataset", "processed_data", f"{test_id + 1}.csv"
                )
            )
            _cached_test_dfs.append(test_df)
            _cached_data_loaders.append(SimulationDataset(test_df))

    # Use the cached values
    test_ids = _cached_test_ids
    test_dfs = _cached_test_dfs
    data_loaders = _cached_data_loaders

    # Assume all datasets have the same length (150 steps)
    T = 150
    n_tests = len(data_loaders)

    # Initialize current SIR states for each test dataset.
    current_sirs = []
    for loader in data_loaders:
        # loader[0] returns a tuple; the first element is the initial SIR.
        s, _, _, _ = loader[0]
        current_sirs.append(s)
    current_sirs = np.array(current_sirs)  # shape: (n_tests, sir_dim)

    total_loss = 0

    # For plotting all simulations
    if PLOT_FOR_ALL:
        all_predictions = [[] for _ in range(n_tests)]
        for i in range(n_tests):
            all_predictions[i].append(current_sirs[i])

    # Run the simulation for T steps.
    for i in range(T):
        interventions = []
        statics = []
        labels = []
        for loader in data_loaders:
            # Each loader[i] returns: (current SIR, x_intervention, x_static, label)
            _, x_intervention, x_static, label = loader[i]
            interventions.append(x_intervention)
            statics.append(x_static)
            labels.append(label)

        # Convert lists to numpy arrays for batch processing.
        interventions = np.array(interventions)
        statics = np.array(statics)

        # Predict one step for all test datasets at once.
        outputs = model.predict(
            model_instance,
            np.array(current_sirs),  # current SIR states for the batch
            interventions,
            statics,
        )

        # If working with deltas, add the current state.
        if IS_DELTAS:
            outputs = outputs + current_sirs

        # Compute the MSE for this step across all test datasets.
        labels_array = np.array(labels)
        mse = np.sum((outputs - labels_array) ** 2)
        total_loss += mse

        # Save predictions for plotting
        if PLOT_FOR_ALL:
            for j in range(n_tests):
                all_predictions[j].append(outputs[j])

        # Update the current states.
        current_sirs = outputs

    # Plot all simulations if requested
    if PLOT_FOR_ALL:
        import matplotlib.pyplot as plt

        postfix = "deltas" if IS_DELTAS else "absolute"
        directory = f"plots/{model}/{postfix}/all_simulations"
        os.makedirs(directory, exist_ok=True)

        for test_idx in range(n_tests):
            # Convert list of arrays to a 2D array
            predictions = np.array(all_predictions[test_idx])

            # Plot the data
            plt.figure(figsize=(10, 10))

            # Students plot
            plt.subplot(2, 1, 1)
            plt.plot(
                predictions[1:, 0],  # Skip initial state
                label="Susceptible - Predicted",
                color="blue",
                alpha=0.5,
                linestyle="--",
            )
            plt.plot(
                test_dfs[test_idx]["S_Students"],
                label="Susceptible - Real",
                color="blue",
                alpha=0.5,
            )

            plt.plot(
                predictions[1:, 1],
                label="Infected - Predicted",
                color="red",
                alpha=0.5,
                linestyle="--",
            )
            plt.plot(
                test_dfs[test_idx]["I_Students"],
                label="Infected - Real",
                color="red",
                alpha=0.5,
            )

            plt.plot(
                predictions[1:, 2],
                label="Recovered - Predicted",
                color="green",
                alpha=0.5,
                linestyle="--",
            )
            plt.plot(
                test_dfs[test_idx]["R_Students"],
                label="Recovered - Real",
                color="green",
                alpha=0.5,
            )

            plt.title(f"Student Population SIR Model - Test {test_ids[test_idx] + 1}")
            plt.xlabel("Time Steps")
            plt.ylabel("Ratio")
            plt.legend()
            plt.grid(True)

            # Adults plot
            plt.subplot(2, 1, 2)
            plt.plot(
                predictions[1:, 3],
                label="Susceptible - Predicted",
                color="blue",
                alpha=0.5,
                linestyle="--",
            )
            plt.plot(
                test_dfs[test_idx]["S_Adults"],
                label="Susceptible - Real",
                color="blue",
                alpha=0.5,
            )

            plt.plot(
                predictions[1:, 4],
                label="Infected - Predicted",
                color="red",
                alpha=0.5,
                linestyle="--",
            )
            plt.plot(
                test_dfs[test_idx]["I_Adults"],
                label="Infected - Real",
                color="red",
                alpha=0.5,
            )

            plt.plot(
                predictions[1:, 5],
                label="Recovered - Predicted",
                color="green",
                alpha=0.5,
                linestyle="--",
            )
            plt.plot(
                test_dfs[test_idx]["R_Adults"],
                label="Recovered - Real",
                color="green",
                alpha=0.5,
            )

            plt.title(f"Adult Population SIR Model - Test {test_ids[test_idx] + 1}")
            plt.xlabel("Time Steps")
            plt.ylabel("Ratio")
            plt.legend()
            plt.tight_layout()
            plt.grid(True)

            plt.savefig(f"{directory}/test_{test_ids[test_idx] + 1}.png")
            plt.close()

    return total_loss / (150 * len(data_loaders) * 6)


def grid_search_model(model_name):
    """Perform hyperparameter grid search for the specified model"""
    if model_name not in models_dict:
        print(f"Model {model_name} not found")
        sys.exit(1)

    model = models_dict[model_name]

    # Check if the model supports grid search
    if not hasattr(model, "grid_search"):
        print(f"Model {model_name} does not support grid search")
        sys.exit(1)

    # Load data for training and validation
    data = load_data(
        batch_size=BATCH_SIZE,
        pytorch=model.IS_PYTORCH,
        is_deltas=IS_DELTAS,
        sequence_length=150 if model.AUTOREGRESSIVE else 1,
    )

    # Handle different return types from load_data based on pytorch flag
    if model.IS_PYTORCH:
        train_loader, val_loader, _ = data
        train_df = train_loader.dataset.df
        val_df = val_loader.dataset.df
    else:
        train_df, val_df, _ = data

    # Run grid search
    logger.info(f"Starting grid search for {model_name}...")
    best_config, best_model = model.grid_search(
        train_df, val_df, IS_DELTAS, loss_fn=lambda x: evaluate_model(model, x)
    )

    # Save the best model
    if not os.path.exists(f"models/{model_name}/checkpoints"):
        os.makedirs(f"models/{model_name}/checkpoints")

    model.save_model(
        best_model, f"models/{model_name}/checkpoints/best_model_grid_search.pth"
    )

    # Log the best configuration
    logger.info(f"Best configuration for {model_name}: {best_config}")

    # Evaluate the best model
    loss = evaluate_model(model, best_model)
    logger.info(
        f"Test Loss for {model_name} ({'deltas' if IS_DELTAS else 'absolute'}) with grid search: {loss}"
    )

    return best_model


def main(model_name):
    if model_name not in models_dict:
        print(f"Model {model_name} not found")
        sys.exit(1)
    model = models_dict[model_name]

    # Add grid search option
    if len(sys.argv) > 2 and sys.argv[2] == "--grid-search":
        return grid_search_model(model_name)

    model_instance = model.Model(input_size=FEATURE_SIZE, is_deltas=IS_DELTAS)

    if not os.path.exists(f"models/{model_name}/checkpoints"):
        os.makedirs(f"models/{model_name}/checkpoints")

    if not TRAIN_MODEL:
        model.load_model(
            model_instance, f"models/{model_name}/checkpoints/{MODEL_TO_LOAD}"
        )
    else:
        train_loader, val_loader, _ = load_data(
            batch_size=BATCH_SIZE,
            pytorch=model.IS_PYTORCH,
            is_deltas=IS_DELTAS,
            sequence_length=150 if model.AUTOREGRESSIVE else 1,
        )
        for avg_train_loss, avg_val_loss, epoch in model.train_model(
            model_instance, train_loader, val_loader, EPOCHS
        ):
            logger.info(
                f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
            )

            model.save_model(
                model_instance, f"models/{model_name}/checkpoints/model_{epoch+1}.pth"
            )

    if PLOT_TEST:
        plot_for_model(model, model_instance)

    loss = evaluate_model(model, model_instance)
    logger.info(
        f"Test Loss for {model_name} ({'deltas' if IS_DELTAS else 'absolute'}): {loss}"
    )

    if IS_TEST_MODE:
        run_test(model, model_instance, INITIAL_CONFIG)
