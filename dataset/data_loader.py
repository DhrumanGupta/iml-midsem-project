import os
from logger import logger
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch

from dataset.dataset import FEATURE_COLS_SIR, LABEL_COLS, SimulationDataset
from dataset.sequence_dataset import SequenceDataset


def load_data(
    batch_size: int = 32,
    pytorch: bool = True,
    seed: int = 1234,
    is_deltas: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_directory = os.path.join("dataset", "processed_data")
    files = sorted(os.listdir(data_directory), key=lambda x: int(x.split(".")[0]))
    logger.info(f"Found {len(files)} files in {data_directory}")

    split_indices_path = os.path.join("dataset", "split_indices.npy")

    if os.path.exists(split_indices_path) and False:
        # Load existing split indices
        split_info = np.load(split_indices_path, allow_pickle=True).item()
        train_indices = np.array(split_info["train"])
        val_indices = np.array(split_info["val"])
        test_indices = np.array(split_info["test"])
        logger.info("Using existing split indices")
    else:
        # Set random seed for reproducibility
        np.random.seed(seed)

        # Randomly shuffle file indices
        indices = np.random.permutation(len(files))

        # Calculate split points
        train_size = int(0.75 * len(files))
        val_size = int(0.05 * len(files))

        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        # Save indices for reproducibility
        split_info = {
            "train": train_indices.tolist(),
            "val": val_indices.tolist(),
            "test": test_indices.tolist(),
        }
        np.save(split_indices_path, split_info)
        logger.info("Created and saved new split indices")

    # Create DataFrames for each split
    train_dfs = [
        pd.read_csv(os.path.join(data_directory, files[i])) for i in train_indices
    ]
    val_dfs = [pd.read_csv(os.path.join(data_directory, files[i])) for i in val_indices]
    test_dfs = [
        pd.read_csv(os.path.join(data_directory, files[i])) for i in test_indices
    ]

    # Concatenate the splits
    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_df = pd.concat(test_dfs)

    if is_deltas:
        for label_col, feature_col in zip(LABEL_COLS, FEATURE_COLS_SIR):
            train_df[label_col] = train_df[label_col] - train_df[feature_col]
            val_df[label_col] = val_df[label_col] - val_df[feature_col]
            test_df[label_col] = test_df[label_col] - test_df[feature_col]

    if pytorch:
        train_dataset = SimulationDataset(train_df)
        val_dataset = SimulationDataset(val_df)
        test_dataset = SimulationDataset(test_df)
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        )
    else:
        return (
            train_df,
            val_df,
            test_df,
        )


def load_sequence_data(
    batch_size: int = 4,
    seed: int = 1234,
    is_deltas: bool = False,
):
    data_directory = os.path.join("dataset", "processed_data")
    files = sorted(os.listdir(data_directory), key=lambda x: int(x.split(".")[0]))
    logger.info(f"Found {len(files)} files in {data_directory}")

    split_indices_path = os.path.join("dataset", "split_indices.npy")

    if os.path.exists(split_indices_path):
        # Load existing split indices
        split_info = np.load(split_indices_path, allow_pickle=True).item()
        train_indices = np.array(split_info["train"])
        val_indices = np.array(split_info["val"])
        test_indices = np.array(split_info["test"])
        logger.info("Using existing split indices")
    else:
        # Set random seed for reproducibility
        np.random.seed(seed)

        # Randomly shuffle file indices
        indices = np.random.permutation(len(files))

        # Calculate split points
        train_size = int(0.75 * len(files))
        val_size = int(0.05 * len(files))

        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        # Save indices for reproducibility
        split_info = {
            "train": train_indices.tolist(),
            "val": val_indices.tolist(),
            "test": test_indices.tolist(),
        }
        np.save(split_indices_path, split_info)
        logger.info("Created and saved new split indices")

    # Create dataset instances for each split
    train_dataset = SequenceDataset(train_indices, data_directory, is_deltas)
    val_dataset = SequenceDataset(val_indices, data_directory, is_deltas)
    test_dataset = SequenceDataset(test_indices, data_directory, is_deltas)

    # Create DataLoaders with padding
    def collate_fn(batch):
        # Sort batch by sequence length (descending)
        batch.sort(key=lambda x: len(x[0]), reverse=True)

        # Get sequence lengths
        lengths = [len(x[0]) for x in batch]
        max_len = lengths[0]

        # Prepare padded tensors
        sir_batch = []
        int_batch = []
        static_batch = []
        label_batch = []

        for sir, interventions, static, labels in batch:
            # Pad sequences
            sir_padded = torch.nn.functional.pad(sir, (0, 0, 0, max_len - len(sir)))
            int_padded = torch.nn.functional.pad(
                interventions, (0, 0, 0, max_len - len(interventions))
            )
            static_padded = torch.nn.functional.pad(
                static, (0, 0, 0, max_len - len(static))
            )
            label_padded = torch.nn.functional.pad(
                labels, (0, 0, 0, max_len - len(labels))
            )

            sir_batch.append(sir_padded)
            int_batch.append(int_padded)
            static_batch.append(static_padded)
            label_batch.append(label_padded)

        return (
            torch.stack(sir_batch),
            torch.stack(int_batch),
            torch.stack(static_batch),
            torch.stack(label_batch),
            torch.tensor(lengths),
        )

    return (
        DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        ),
        DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        ),
        DataLoader(test_dataset, batch_size=1, shuffle=False),
    )
