import os
import sys
import numpy as np
import pandas as pd
import toml
from loguru import logger
import matplotlib.pyplot as plt
import shutil

POPULATION_SIZE = 100_000


logger.remove()  # Remove default handler
logger.add(
    sink=sys.stderr,
    format="<green>{time:HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)


def process_config(name: str):
    # Build file paths
    config_path = f"dataset/configs/{name}"
    raw_data_dir = f"dataset/raw_data/{name}"
    output_path = f"dataset/processed_data/{name.split('.')[0]}.csv"

    # Load config
    config = toml.load(config_path)

    if not os.path.exists(raw_data_dir):
        # Optionally, log an error here
        logger.error(f"Raw data directory {raw_data_dir} does not exist")
        return

    # Read CSV files using a list comprehension
    csv_files = [os.path.join(raw_data_dir, f) for f in os.listdir(raw_data_dir)]

    dfs = []

    for f in csv_files:
        try:
            df = pd.read_csv(f).drop(columns=["Day"])
            dfs.append(df)

            if len(df) != 151:
                logger.error(f"Invalid number of days in {f}")
                exit()
        except Exception as e:
            logger.error(f"Error reading {f}: {e}")
            exit()

    # Compute average of each cell from the list of DataFrames
    avg_values = np.mean([df.values for df in dfs], axis=0)
    avg_df = pd.DataFrame(avg_values, index=dfs[0].index, columns=dfs[0].columns)

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

    # Precompute lockdown intensities for each day
    n_days = len(avg_df)
    school_intensity = np.zeros(n_days)
    office_intensity = np.zeros(n_days)

    # Assign intensities; assumes intervals are nonoverlapping
    for day, duration, intensity in school_lockdowns:
        start, end = day, min(day + duration, n_days)
        school_intensity[start:end] = intensity
    for day, duration, intensity in office_lockdowns:
        start, end = day, min(day + duration, n_days)
        office_intensity[start:end] = intensity

    # Build the processed dataframe using vectorized operations
    processed_df = pd.DataFrame(
        {
            "S_Students": pd.to_numeric(avg_df["Students - Susceptible - Mumbai"])
            / POPULATION_SIZE,
            "I_Students": pd.to_numeric(avg_df["Students - Infected - Mumbai"])
            / POPULATION_SIZE,
            "R_Students": pd.to_numeric(avg_df["Students - Recovered - Mumbai"])
            / POPULATION_SIZE,
            "S_Adults": pd.to_numeric(avg_df["Adults - Susceptible - Mumbai"])
            / POPULATION_SIZE,
            "I_Adults": pd.to_numeric(avg_df["Adults - Infected - Mumbai"])
            / POPULATION_SIZE,
            "R_Adults": pd.to_numeric(avg_df["Adults - Recovered - Mumbai"])
            / POPULATION_SIZE,
            "Adult_Ratio": adult_ratio,
            "Student_Ratio": student_ratio,
            "Home_Size": home_size,
            "School_Size": school_size,
            "Work_Size": work_size,
            "Beta": float(config["BETA"]),
            "Gamma": float(config["GAMMA"]),
            "School_Lockdown_Intensity": school_intensity,
            "Office_Lockdown_Intensity": office_intensity,
        }
    )

    processed_df.to_csv(output_path, index=False)


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

    for config in configs:
        process_config(config)


if __name__ == "__main__":
    main()
