import os
import sys
import numpy as np
import pandas as pd
import toml
import matplotlib.pyplot as plt
import shutil
import multiprocessing

from logger import logger


def process_individual_df(df: pd.DataFrame, config: dict, school_intensity: np.ndarray, office_intensity: np.ndarray, run_id: str) -> pd.DataFrame:
    """Processes a single DataFrame for one simulation run."""
    # Parse INPUT config
    input_parts = config["INPUT"].split("_")
    adult_ratio = float(input_parts[1])
    student_ratio = float(input_parts[3])
    home_size = int(input_parts[5])
    school_size = int(input_parts[7])
    work_size = int(input_parts[9])

    # Calculate total population sizes for this specific DataFrame
    student_size = (
        pd.to_numeric(df["Students - Susceptible - Mumbai"])
        + pd.to_numeric(df["Students - Infected - Mumbai"])
        + pd.to_numeric(df["Students - Recovered - Mumbai"])
    )
    adult_size = (
        pd.to_numeric(df["Adults - Susceptible - Mumbai"])
        + pd.to_numeric(df["Adults - Infected - Mumbai"])
        + pd.to_numeric(df["Adults - Recovered - Mumbai"])
    )

    # Avoid division by zero if a population group is empty
    student_size = student_size.replace(0, 1)
    adult_size = adult_size.replace(0, 1)

    # Create the processed dataframe
    processed_df = pd.DataFrame(
        {
            "run_id": run_id,
            "S_Students": pd.to_numeric(df["Students - Susceptible - Mumbai"])
            / student_size,
            "I_Students": pd.to_numeric(df["Students - Infected - Mumbai"])
            / student_size,
            "R_Students": pd.to_numeric(df["Students - Recovered - Mumbai"])
            / student_size,
            "S_Adults": pd.to_numeric(df["Adults - Susceptible - Mumbai"])
            / adult_size,
            "I_Adults": pd.to_numeric(df["Adults - Infected - Mumbai"])
            / adult_size,
            "R_Adults": pd.to_numeric(df["Adults - Recovered - Mumbai"])
            / adult_size,
            "Adult_Ratio": adult_ratio,
            "Student_Ratio": student_ratio,
            "Home_Size": home_size,
            "School_Size": school_size,
            "Work_Size": work_size,
            "Beta": float(config["BETA"]),
            "Gamma": float(config["GAMMA"]),
            "School_Lockdown_Intensity": school_intensity,
            "Office_Lockdown_Intensity": office_intensity,
            "Label_S_Students": pd.to_numeric(
                df["Students - Susceptible - Mumbai"]
            ).shift(-1)
            / student_size,
            "Label_I_Students": pd.to_numeric(
                df["Students - Infected - Mumbai"]
            ).shift(-1)
            / student_size,
            "Label_R_Students": pd.to_numeric(
                df["Students - Recovered - Mumbai"]
            ).shift(-1)
            / student_size,
            "Label_S_Adults": pd.to_numeric(
                df["Adults - Susceptible - Mumbai"]
            ).shift(-1)
            / adult_size,
            "Label_I_Adults": pd.to_numeric(df["Adults - Infected - Mumbai"]).shift(
                -1
            )
            / adult_size,
            "Label_R_Adults": pd.to_numeric(
                df["Adults - Recovered - Mumbai"]
            ).shift(-1)
            / adult_size,
        }
    )

    # Drop the last row since it won't have labels
    processed_df = processed_df.dropna()
    return processed_df


def process_config(name: str):
    config_base_name = name.split('.')[0]
    # Build file paths
    config_path = f"dataset/configs/{name}"
    raw_data_dir = f"dataset/raw_data/{config_base_name}.toml"
    output_path_avg = f"dataset/processed_data/{config_base_name}.csv"
    # Output path for the combined large file
    output_path_large = f"dataset/processed_data_large/{config_base_name}.csv"

    # Remove the old directory structure if it exists to avoid confusion
    old_output_dir_large = f"dataset/processed_data_large/{config_base_name}"
    if os.path.isdir(old_output_dir_large):
        shutil.rmtree(old_output_dir_large)
        logger.info(f"Removed old directory: {old_output_dir_large}")

    # Load config
    config = toml.load(config_path)

    if not os.path.exists(raw_data_dir):
        logger.error(f"Raw data directory {raw_data_dir} does not exist")
        return

    csv_files = [os.path.join(raw_data_dir, f) for f in os.listdir(raw_data_dir) if f.endswith('.csv')] # Ensure only CSVs are read

    dfs_for_avg = [] # To store raw dfs for averaging
    processed_dfs_large = [] # To store processed dfs for the large file

    # --- Precompute items common to all files in this config ---
    try:
        # Load one file to determine n_days reliably
        temp_df_for_days = pd.read_csv(csv_files[0])
        n_days = len(temp_df_for_days)
        if n_days != 151: # Check length from actual data
             logger.warning(f"Expected 151 days, found {n_days} in {csv_files[0]}. Using {n_days}.")
        del temp_df_for_days # Free memory
    except IndexError:
        logger.error(f"No CSV files found in {raw_data_dir}")
        return
    except Exception as e:
        logger.error(f"Error reading first file {csv_files[0]} to determine n_days: {e}")
        return

    # Parse INPUT config only once
    input_parts = config["INPUT"].split("_")
    adult_ratio = float(input_parts[1])
    student_ratio = float(input_parts[3])
    home_size = int(input_parts[5])
    school_size = int(input_parts[7])
    work_size = int(input_parts[9])

    # Lockdown configuration
    school_lockdowns = list(
        zip(
            config["SCHOOL_CLOSED_DAYS"],
            config["SCHOOL_CLOSED_DURATIONS"],
            config["SCHOOL_CLOSED_STRENGTHS"],
        )
    )
    office_lockdowns = list(
        zip(
            config["OFFICE_CLOSED_DAYS"],
            config["OFFICE_CLOSED_DURATIONS"],
            config["OFFICE_CLOSED_STRENGTHS"],
        )
    )

    # Precompute lockdown intensities
    school_intensity = np.zeros(n_days)
    office_intensity = np.zeros(n_days)

    for day, duration, intensity in school_lockdowns:
        start = max(0, day - 1) # Ensure start index is non-negative
        end = min(day + duration - 1, n_days)
        if start < end: # Ensure valid range
            school_intensity[start:end] = intensity
    for day, duration, intensity in office_lockdowns:
        start = max(0, day - 1) # Ensure start index is non-negative
        end = min(day + duration - 1, n_days)
        if start < end: # Ensure valid range
            office_intensity[start:end] = intensity
    # --- End of precomputation ---

    if not csv_files:
        logger.warning(f"No CSV files found in {raw_data_dir} for config {name}. Skipping.")
        return

    logger.info(f"Processing {len(csv_files)} run files for config {name}...")
    for f in csv_files:
        try:
            df = pd.read_csv(f).drop(columns=["Day"])

            if len(df) != n_days: # Check against determined n_days
                logger.error(f"Inconsistent number of days in {f}. Expected {n_days}, got {len(df)}. Skipping.")
                continue # Skip this file

            # Get run_id from filename
            run_id = os.path.basename(f).split('.')[0]

            # Process this individual dataframe
            processed_df_individual = process_individual_df(df.copy(), config, school_intensity, office_intensity, run_id)
            processed_dfs_large.append(processed_df_individual)

            # Add the raw df (without Day col) to list for averaging later
            dfs_for_avg.append(df)

        except Exception as e:
            logger.error(f"Error processing file {f}: {e}")
            continue # Continue to next file

    # --- Save the combined large file ---
    if processed_dfs_large:
        logger.info(f"Concatenating and saving {len(processed_dfs_large)} processed runs for config {name}...")
        combined_large_df = pd.concat(processed_dfs_large, ignore_index=True)
        combined_large_df.to_csv(output_path_large, index=False)
        logger.info(f"Saved combined large data for {name} to {output_path_large}")
    else:
        logger.warning(f"No valid runs processed for large dataset for config {name}. No large file saved.")

    # --- Compute and save the average ---
    if not dfs_for_avg:
        logger.warning(f"No valid data files processed for config {name}. Skipping average calculation.")
        return # Exit if no files were successfully processed for averaging either

    logger.info(f"Calculating average for config {name} from {len(dfs_for_avg)} files.")
    avg_values = np.mean([df.values for df in dfs_for_avg], axis=0)
    avg_df = pd.DataFrame(avg_values, index=dfs_for_avg[0].index, columns=dfs_for_avg[0].columns)

    # Process the averaged dataframe
    processed_df_avg = process_individual_df(avg_df, config, school_intensity, office_intensity, "")

    # Save the averaged dataframe
    processed_df_avg.to_csv(output_path_avg, index=False)
    logger.info(f"Saved averaged data for {name} to {output_path_avg}")


def get_all_configs():
    configs = []
    for file in sorted(
        os.listdir("dataset/configs"), key=lambda x: int(x.split(".")[0])
    ):
        configs.append(file)
    return configs


def main():
    configs = get_all_configs()
    logger.info(f"Found {len(configs)} configs")

    os.makedirs("dataset/processed_data", exist_ok=True)
    os.makedirs("dataset/processed_data_large", exist_ok=True) # Create the new base directory

    # Determine the number of processes to use (e.g., number of CPU cores, capped at 10)
    num_processes = min(multiprocessing.cpu_count()//4, 10)
    logger.info(f"Using {num_processes} processes for parallel execution.")

    # Use a Pool to parallelize the processing of configs
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the process_config function to the list of configs
        # pool.map will distribute the configs among the worker processes
        pool.map(process_config, configs)

    logger.info("Finished processing all configurations.")


if __name__ == "__main__":
    main()
