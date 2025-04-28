import os
from logger import logger
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from multiprocessing.pool import ThreadPool # Use ThreadPool for map interface
import time # For timing
from tqdm import tqdm # Import tqdm

from dataset.dataset import FEATURE_COLS_SIR, LABEL_COLS, SimulationDataset


def load_data(
    batch_size: int = 32,
    pytorch: bool = True,
    seed: int = 123456,
    is_deltas: bool = False,
    sequence_length: int = 1,
    is_large: bool = False,
    max_workers: int = None, # Allow controlling max workers for ThreadPoolExecutor
) -> tuple:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if is_large:
        # Read combined config files from processed_data_large
        data_directory = os.path.join(project_root, "dataset", "processed_data_large")
        all_entries = os.listdir(data_directory)
        # Filter for combined CSV files (e.g., 0.csv, 1.csv) and sort numerically
        csv_files = [f for f in all_entries if f.endswith('.csv') and f.split('.')[0].isdigit()]
        files = sorted(csv_files, key=lambda x: int(x.split(".")[0]))
        logger.info(f"Identified {len(files)} combined config files in {data_directory}")
    else:
        # Original logic: Read averaged files from processed_data
        data_directory = os.path.join(project_root, "dataset", "processed_data")
        all_entries = os.listdir(data_directory)
        csv_files = [f for f in all_entries if f.endswith('.csv') and f.split('.')[0].isdigit()]
        files = sorted(csv_files, key=lambda x: int(x.split(".")[0]))
        logger.info(f"Identified {len(files)} averaged files in {data_directory}")

    if not files:
        raise FileNotFoundError(f"No data files found in {data_directory}. Check the path and if preprocessing was successful.")

    split_indices_path = os.path.join(project_root, "dataset", f"split_indices{'_large' if is_large else ''}.npy")

    if os.path.exists(split_indices_path):
        split_info = np.load(split_indices_path, allow_pickle=True).item()
        train_indices = np.array(split_info["train"])
        val_indices = np.array(split_info["val"])
        test_indices = np.array(split_info["test"])
        max_index = len(files) - 1
        train_indices = train_indices[train_indices <= max_index]
        val_indices = val_indices[val_indices <= max_index]
        test_indices = test_indices[test_indices <= max_index]
        logger.info(f"Using existing split indices from {split_indices_path}")
        if len(train_indices) + len(val_indices) + len(test_indices) < len(files) * 0.9: # Check if loaded split covers most files
            logger.warning("Loaded split indices cover significantly fewer files than found. Consider regenerating the split.")

    else:
        logger.info(f"Split file {split_indices_path} not found. Creating new split.")
        np.random.seed(seed)
        indices = np.random.permutation(len(files))
        train_size = int(0.75 * len(files))
        val_size = int(0.05 * len(files))
        # Ensure validation set has at least one sample if possible
        val_size = max(1, val_size) if len(files) > train_size else 0
        # Ensure test set has at least one sample if possible
        test_size = len(files) - train_size - val_size
        test_size = max(1, test_size) if len(files) > train_size + val_size else 0
        # Adjust train_size if test_size needed adjustment
        train_size = len(files) - val_size - test_size

        if train_size <= 0 or (val_size <=0 and test_size <=0):
            raise ValueError(f"Not enough files ({len(files)}) to create a meaningful train/val/test split.")

        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]
        split_info = {
            "train": train_indices.tolist(),
            "val": val_indices.tolist(),
            "test": test_indices.tolist(),
        }
        np.save(split_indices_path, split_info)
        logger.info(f"Created and saved new split indices to {split_indices_path}")

    # --- Parallel DataFrame Loading & Splitting --- 
    start_time = time.time()
    logger.info(f"Starting parallel read & split of {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test config files...")

    def read_and_split_wrapper(file_index):
        """Reads a combined CSV and splits it into a list of DFs by run_id."""
        file_path = os.path.join(data_directory, files[file_index])
        try:
            combined_df = pd.read_csv(file_path)
            # Split the combined DataFrame into a list of DataFrames based on 'run_id'
            # Ensure run_id exists if is_large is True
            if is_large and 'run_id' not in combined_df.columns:
                logger.error(f"'run_id' column missing in {file_path}. Cannot split. Skipping file.")
                return [] # Return empty list on critical error
            if is_large:
                 # Group by run_id and return a list of the group DataFrames
                 # .copy() ensures slices are independent DataFrames
                 grouped = combined_df.groupby('run_id')
                 split_dfs = [group.copy() for _, group in grouped]
                 return split_dfs
            else:
                 # If not is_large, return the single DataFrame in a list
                 return [combined_df]
        except Exception as e:
            logger.error(f"Error reading/splitting file {file_path}: {e}")
            return [] # Return empty list on error

    if max_workers is None:
        # For ThreadPool, often good to use slightly more workers than cores for I/O
        max_workers = (os.cpu_count() or 1) * 2

    train_dfs = []
    val_dfs = []
    test_dfs = []

    # Use ThreadPool with imap_unordered for incremental processing
    with ThreadPool(processes=max_workers) as pool:
        # Process Train files
        logger.info(f"Reading & Splitting {len(train_indices)} train files...")
        train_results_iter = pool.imap_unordered(read_and_split_wrapper, train_indices)
        for df_list in tqdm(train_results_iter, total=len(train_indices), desc="Processing train configs"):
            train_dfs.extend(df_list) # Extend with the list of split DFs

        # Process Validation files
        logger.info(f"Reading & Splitting {len(val_indices)} val files...")
        val_results_iter = pool.imap_unordered(read_and_split_wrapper, val_indices)
        for df_list in tqdm(val_results_iter, total=len(val_indices), desc="Processing val configs"):
            val_dfs.extend(df_list)

        # Process Test files
        logger.info(f"Reading & Splitting {len(test_indices)} test files...")
        test_results_iter = pool.imap_unordered(read_and_split_wrapper, test_indices)
        for df_list in tqdm(test_results_iter, total=len(test_indices), desc="Processing test configs"):
            test_dfs.extend(df_list)

    end_time = time.time()
    total_runs_loaded = len(train_dfs) + len(val_dfs) + len(test_dfs)
    logger.info(f"Finished reading/splitting {total_runs_loaded} total runs in {end_time - start_time:.2f} seconds.")

    # Check if any set is empty
    if not train_dfs:
        raise ValueError("No training data could be loaded/split. Check CSV files and errors.")
    if not val_dfs:
        logger.warning("No validation data could be loaded/split. Proceeding without validation set.")
    if not test_dfs:
        logger.warning("No test data could be loaded/split. Proceeding without test set.")

    # --- Concatenate Individual Run DataFrames --- 
    logger.info("Concatenating individual run dataframes...")
    start_concat_time = time.time()
    # Concatenation logic remains the same, now operates on individual run DFs
    train_df = pd.concat(train_dfs) if train_dfs else pd.DataFrame()
    val_df = pd.concat(val_dfs) if val_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs) if test_dfs else pd.DataFrame()
    end_concat_time = time.time()
    logger.info(f"Finished concatenating run dataframes in {end_concat_time - start_concat_time:.2f} seconds.")

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
