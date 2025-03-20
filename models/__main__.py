import os
import sys

import numpy as np
import pandas as pd
from dataset.data_loader import load_data
from dataset.dataset import FEATURE_SIZE, SimulationDataset
from logger import logger
import importlib

TRAIN_MODEL = True
IS_DELTAS = False
MODEL_TO_LOAD = "model_5.pth"
EPOCHS = 50
AUTOREGRESSIVE = True


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
        directory = f"plots/{model_name}/{postfix}"
        os.makedirs(directory, exist_ok=True)
        while os.path.exists(f"{directory}/plot_{idx}_{type}.png"):
            idx += 1

        plt.savefig(f"{directory}/plot_{idx}_{type}.png")

        plt.cla()
        plt.clf()

        # plt.show()


def evaluate_model(model, model_instance):
    # Load the test indices.
    test_ids = np.load(
        os.path.join("dataset", "split_indices.npy"), allow_pickle=True
    ).item()["test"]

    # Load all test datasets and create a list of SimulationDataset instances.
    data_loaders = []
    for test_id in test_ids:
        test_df = pd.read_csv(
            os.path.join("dataset", "processed_data", f"{test_id + 1}.csv")
        )
        data_loaders.append(SimulationDataset(test_df))

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

        # Update the current states.
        current_sirs = outputs

    return total_loss / (150 * len(data_loaders) * 6)


def main(model_name):
    model = models_dict[model_name]

    train_loader, val_loader, test_loader = load_data(
        batch_size=256,
        pytorch=model.IS_PYTORCH,
        is_deltas=IS_DELTAS,
        sequence_length=150 if AUTOREGRESSIVE else 1,
    )

    model_instance = model.Model(input_size=FEATURE_SIZE, is_deltas=IS_DELTAS)

    if not os.path.exists(f"models/{model_name}/checkpoints"):
        os.makedirs(f"models/{model_name}/checkpoints")

    if not TRAIN_MODEL:
        model.load_model(
            model_instance, f"models/{model_name}/checkpoints/{MODEL_TO_LOAD}"
        )
    else:
        for avg_train_loss, avg_val_loss, epoch in model.train_model(
            model_instance, train_loader, val_loader, EPOCHS
        ):
            logger.info(
                f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
            )

            model.save_model(
                model_instance, f"models/{model_name}/checkpoints/model_{epoch+1}.pth"
            )

    plot_for_model(model, model_instance)

    loss = evaluate_model(model, model_instance)
    logger.info(
        f"Test Loss for {model_name} ({'deltas' if IS_DELTAS else 'absolute'}): {loss}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python __main__.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    if model_name not in models_dict:
        print(f"Model {model_name} not found")
        sys.exit(1)

    main(model_name)
