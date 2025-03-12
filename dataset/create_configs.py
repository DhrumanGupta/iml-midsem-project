import itertools
import os
import toml
from pathlib import Path

lockdown_days_config = [[], [20], [40], [60], [20, 60], [20, 70], [40, 70]]

lockdown_durations = [21]
strengths = [0.7, 0.9]

beta_gamma_config = [
    (0.3, 1 / 7),
    (0.3, 1 / 14),
    (0.35, 1 / 14),
    (0.35, 1 / 7),
]

age_dist_config = [(0.8, 0.2), (0.7, 0.3)]

population_dist_config = [(5, 1500, 300), (6, 300, 50)]

configs = []

# Precompute the possible (duration, strength) pairs
pairs = list(itertools.product(lockdown_durations, strengths))

# Loop through each lockdown_days_config and its corresponding beta_gamma
for days in lockdown_days_config:
    # For each day in the current config, assign a (duration, strength) pair.
    # The number of combinations is 4^len(days) (since there are 4 possible pairs per day).
    for combo in itertools.product(pairs, repeat=len(days)):
        # print(combo)
        config = {
            "lockdown_days": days,
            "lockdown_params": list(combo),  # each tuple corresponds to a day in "days"
        }
        configs.append(config)

config_pairs = list(itertools.product(configs, repeat=2))

final_configs = []

for config_pair in config_pairs:
    for age_dist in age_dist_config:
        for population_dist in population_dist_config:
            for beta, gamma in beta_gamma_config:
                final_configs.append(
                    {
                        # "age_dist": age_dist,
                        # "population_dist": population_dist,
                        "population_name": f"adult_{age_dist[0]}_kid_{age_dist[1]}_home_{population_dist[0]}_school_{population_dist[1]}_work_{population_dist[2]}",
                        "school_lockdown": config_pair[0],
                        "office_lockdown": config_pair[1],
                        "beta": beta,
                        "gamma": gamma,
                    }
                )


def create_config_file(config, template_path, output_dir):
    # Create output directory if it doesn't exist
    Path(output_dir).parent.mkdir(parents=True, exist_ok=True)

    # Read template
    with open(template_path, "r") as f:
        template_content = toml.load(f)

    # Extract values from config dictionary
    beta = config["beta"]
    gamma = config["gamma"]
    school_lockdown = config["school_lockdown"]
    office_lockdown = config["office_lockdown"]

    # Update template values
    template_content["BETA"] = beta
    template_content["GAMMA"] = gamma
    template_content["INPUT"] = config["population_name"]

    # Update school lockdown parameters
    template_content["SCHOOL_CLOSED_DAYS"] = school_lockdown["lockdown_days"]
    template_content["SCHOOL_CLOSED_DURATIONS"] = [
        params[0] for params in school_lockdown["lockdown_params"]
    ]
    template_content["SCHOOL_CLOSED_STRENGTHS"] = [
        params[1] for params in school_lockdown["lockdown_params"]
    ]

    template_content["SCHOOL_CLOSED"] = len(school_lockdown["lockdown_days"]) > 0
    template_content["OFFICE_CLOSED"] = len(office_lockdown["lockdown_days"]) > 0

    # Update office lockdown parameters
    template_content["OFFICE_CLOSED_DAYS"] = office_lockdown["lockdown_days"]
    template_content["OFFICE_CLOSED_DURATIONS"] = [
        params[0] for params in office_lockdown["lockdown_params"]
    ]
    template_content["OFFICE_CLOSED_STRENGTHS"] = [
        params[1] for params in office_lockdown["lockdown_params"]
    ]

    # Write config file
    with open(output_dir, "w") as f:
        toml.dump(template_content, f)


# Replace print(config) with config file creation
for i, config in enumerate(final_configs):
    create_config_file(
        config, "dataset/config.template.toml", f"dataset/configs/{i+1}.toml"
    )
