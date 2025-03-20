import os
from logger import logger
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from dataset.dataset import FEATURE_COLS_SIR, LABEL_COLS, SimulationDataset


def load_data(
    batch_size: int = 32,
    pytorch: bool = True,
    seed: int = 123456,
    is_deltas: bool = False,
    sequence_length: int = 1,
) -> tuple:
    data_directory = os.path.join("dataset", "processed_data")
    files = sorted(os.listdir(data_directory), key=lambda x: int(x.split(".")[0]))
    logger.info(f"Found {len(files)} files in {data_directory}")

    split_indices_path = os.path.join("dataset", "split_indices.npy")

    if os.path.exists(split_indices_path) and False:
        split_info = np.load(split_indices_path, allow_pickle=True).item()
        train_indices = np.array(split_info["train"])
        val_indices = np.array(split_info["val"])
        test_indices = np.array(split_info["test"])
        logger.info("Using existing split indices")
    else:
        np.random.seed(seed)
        indices = np.random.permutation(len(files))
        train_size = int(0.75 * len(files))
        val_size = int(0.05 * len(files))
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]
        split_info = {
            "train": train_indices.tolist(),
            "val": val_indices.tolist(),
            "test": test_indices.tolist(),
        }
        np.save(split_indices_path, split_info)
        logger.info("Created and saved new split indices")

    train_dfs = [
        pd.read_csv(os.path.join(data_directory, files[i])) for i in train_indices
    ]
    val_dfs = [pd.read_csv(os.path.join(data_directory, files[i])) for i in val_indices]
    test_dfs = [
        pd.read_csv(os.path.join(data_directory, files[i])) for i in test_indices
    ]

    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_df = pd.concat(test_dfs)

    if is_deltas:
        for label_col, feature_col in zip(LABEL_COLS, FEATURE_COLS_SIR):
            train_df[label_col] = train_df[label_col] - train_df[feature_col]
            val_df[label_col] = val_df[label_col] - val_df[feature_col]
            test_df[label_col] = test_df[label_col] - test_df[feature_col]

    if pytorch:
        train_dataset = SimulationDataset(train_df, sequence_length=sequence_length)
        val_dataset = SimulationDataset(val_df, sequence_length=sequence_length)
        test_dataset = SimulationDataset(test_df, sequence_length=sequence_length)
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        )
    else:
        return (train_df, val_df, test_df)
