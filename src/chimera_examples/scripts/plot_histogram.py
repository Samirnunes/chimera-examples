import json
from pathlib import Path

import matplotlib.pyplot as plt


def create_histograms(time_json_path: str, folder: str) -> None:
    """
    Creates histograms from a JSON file containing worker and master times.

    Args:
        time_json_path: Path to the JSON file.
    """
    try:
        with open(time_json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {time_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {time_json_path}")
        return

    for section, endpoints in data.items():
        for endpoint, times in endpoints.items():
            plt.figure(figsize=(8, 6))
            plt.hist(times)
            plt.xlabel("Time (seconds)")
            plt.ylabel("Count")
            plt.title(f"Times Histogram for {endpoint}")
            plt.savefig(
                f"{time_json_path.split('/')[-1].split('.')[0]}_{folder}_{section}_{endpoint.replace('//', '/').replace('/', '_').replace(':', '_').replace('-', '_')}.png"
            )
            plt.close()


folders = [
    "ex_chimera_bagging_classification_rand_forest",
    "ex_chimera_bagging_regression_rand_forest",
    "ex_chimera_sgd_classification_log_reg",
    "ex_chimera_sgd_regression_lin_reg",
]

for folder in folders:
    folderpath = Path(f"src/chimera_examples/{folder}")
    for time_json in folderpath.iterdir():
        if time_json.is_file():
            create_histograms(str(time_json), str(folder))
