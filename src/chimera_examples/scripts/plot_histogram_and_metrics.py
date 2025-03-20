import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def create_histograms_and_metrics(time_json_path: str, folder: str) -> None:
    """
    Creates histograms and calculate metrics from a JSON file containing worker and master times.

    Args:
        time_json_path: Path to the JSON file.
    """
    with open(time_json_path, "r") as f:
        data = json.load(f)

    for section, endpoints in data.items():
        for endpoint, times in endpoints.items():
            plt.figure(figsize=(8, 6))
            plt.hist(times)
            plt.xlabel("Time (seconds)")
            plt.ylabel("Count")
            plt.title(f"Times Histogram for {endpoint}")
            filename_base = time_json_path.split("/")[-1].split(".")[0]
            filename = f"{filename_base}_{folder}_{section}_{endpoint.replace('://', '/').replace('/', '_').replace(':', '_').replace('-', '_')}.png"
            plt.savefig(filename)
            plt.close()

            mean_time = np.mean(times)
            std_time = np.std(times)
            mean_filename = filename.replace(".png", ".txt")
            with open(mean_filename, "w") as mean_file:
                mean_file.write(
                    f"Mean Time (s) = {mean_time:.4f}\nStd Deviation Time (s) = {std_time:.4f}"
                )


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
            create_histograms_and_metrics(str(time_json), str(folder))
