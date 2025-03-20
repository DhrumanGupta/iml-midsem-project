import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from dataset.data_loader import load_sequence_data
from dataset.dataset import FEATURE_SIZE, SimulationDataset
from logger import logger
import importlib

TRAIN_MODEL = True
IS_DELTAS = False
MODEL_TO_LOAD = "model_3.pth"
EPOCHS = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

# Import the LSTM sequence model
from models.lstm import (
    Model,
    IS_PYTORCH,
    save_model,
    load_model,
    train_model,
    predict,
)


def plot_sequence_predictions(model, test_loader, model_name="lstm_seq"):
    """Plot predictions for a few test sequences"""
    model.eval()

    # Get a few test sequences
    for idx, (x_sir, x_interventions, x_static, labels) in enumerate(test_loader):
        if idx >= 3:  # Only plot first 3 sequences
            break

        # Convert to numpy for plotting
        x_sir_np = x_sir.numpy()[0]  # First batch item
        x_int_np = x_interventions.numpy()[0]
        x_static_np = x_static.numpy()[0]
        labels_np = labels.numpy()[0]

        # Make predictions
        with torch.no_grad():
            # Move inputs to device
            x_sir_tensor = x_sir.to(device)
            x_int_tensor = x_interventions.to(device)
            x_static_tensor = x_static.to(device)

            # Get predictions for the whole sequence
            preds = model(x_sir_tensor, x_int_tensor, x_static_tensor)
            preds_np = preds.cpu().numpy()[0]  # First batch item

        # Plot the results
        plot_epidemic_curves(preds_np, labels_np, x_int_np, idx, model_name)


def plot_epidemic_curves(predictions, ground_truth, interventions, index, model_name):
    """
    Plot the predicted vs actual epidemic curves and intervention strategies
    """
    plt.figure(figsize=(15, 10))

    # Plot student population
    plt.subplot(3, 1, 1)
    plt.title("Student Population SIR Model")

    # Plot S, I, R for students
    plt.plot(predictions[:, 0], label="S (Predicted)", color="blue", linestyle="-")
    plt.plot(ground_truth[:, 0], label="S (Actual)", color="blue", linestyle="--")

    plt.plot(predictions[:, 1], label="I (Predicted)", color="red", linestyle="-")
    plt.plot(ground_truth[:, 1], label="I (Actual)", color="red", linestyle="--")

    plt.plot(predictions[:, 2], label="R (Predicted)", color="green", linestyle="-")
    plt.plot(ground_truth[:, 2], label="R (Actual)", color="green", linestyle="--")

    plt.xlabel("Time Steps")
    plt.ylabel("Ratio")
    plt.legend()
    plt.grid(True)

    # Plot adult population
    plt.subplot(3, 1, 2)
    plt.title("Adult Population SIR Model")

    # Plot S, I, R for adults
    plt.plot(predictions[:, 3], label="S (Predicted)", color="blue", linestyle="-")
    plt.plot(ground_truth[:, 3], label="S (Actual)", color="blue", linestyle="--")

    plt.plot(predictions[:, 4], label="I (Predicted)", color="red", linestyle="-")
    plt.plot(ground_truth[:, 4], label="I (Actual)", color="red", linestyle="--")

    plt.plot(predictions[:, 5], label="R (Predicted)", color="green", linestyle="-")
    plt.plot(ground_truth[:, 5], label="R (Actual)", color="green", linestyle="--")

    plt.xlabel("Time Steps")
    plt.ylabel("Ratio")
    plt.legend()
    plt.grid(True)

    # Plot intervention strategies
    plt.subplot(3, 1, 3)
    plt.title("Intervention Strategies")

    plt.plot(interventions[:, 0], label="School Lockdown", color="purple")
    plt.plot(interventions[:, 1], label="Office Lockdown", color="orange")

    plt.xlabel("Time Steps")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Create directory if it doesn't exist
    postfix = "deltas" if IS_DELTAS else "absolute"
    directory = f"plots/{model_name}/{postfix}"
    os.makedirs(directory, exist_ok=True)

    # Save the plot
    plt.savefig(f"{directory}/sequence_plot_{index}.png")
    plt.close()


def evaluate_sequence_model(model, test_loader):
    """Evaluate the sequence model on test data"""
    model.eval()
    total_loss = 0
    total_sequences = 0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for x_sir, x_interventions, x_static, labels in test_loader:
            # Move to device
            x_sir = x_sir.to(device)
            x_interventions = x_interventions.to(device)
            x_static = x_static.to(device)
            labels = labels.to(device)

            # Get predictions
            preds = model(x_sir, x_interventions, x_static)

            # Calculate loss
            loss = criterion(preds, labels)

            total_loss += loss.item()
            total_sequences += 1

    return total_loss / total_sequences


def main():
    # Load data as sequences
    train_loader, val_loader, test_loader = load_sequence_data(
        batch_size=4,
        is_deltas=IS_DELTAS,
    )

    # Initialize model
    model = Model(input_size=FEATURE_SIZE, is_deltas=IS_DELTAS)

    # Create checkpoint directory
    model_name = "lstm_seq"
    if not os.path.exists(f"models/{model_name}/checkpoints"):
        os.makedirs(f"models/{model_name}/checkpoints")

    if not TRAIN_MODEL:
        # Load pre-trained model
        load_model(model, f"models/{model_name}/checkpoints/{MODEL_TO_LOAD}")
    else:
        # Train the model
        for avg_train_loss, avg_val_loss, epoch in train_model(
            model, train_loader, val_loader, EPOCHS
        ):
            logger.info(
                f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
            )

            # Save model checkpoint
            save_model(model, f"models/{model_name}/checkpoints/model_{epoch+1}.pth")

    # Evaluate and plot
    plot_sequence_predictions(model, test_loader, model_name)
    test_loss = evaluate_sequence_model(model, test_loader)
    logger.info(
        f"Test Loss for {model_name} ({'deltas' if IS_DELTAS else 'absolute'}): {test_loss}"
    )


if __name__ == "__main__":
    main()
