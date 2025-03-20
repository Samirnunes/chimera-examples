from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def sklearn_plot_histogram_and_metrics(filepath: str, folder: str) -> None:
    """Processes a CSV file containing time data, creates a histogram, and calculates statistics using pandas."""
    try:
        df = pd.read_csv(filepath)  # Read the CSV using pandas

        times = df["latency(s)"]

        if times.empty:
            print(f"Warning: No valid time data found in {filepath}")
            return

        plt.figure(figsize=(8, 6))
        plt.hist(times)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Count")
        plt.title("Times Histogram")

        filename_base = Path(filepath).stem
        filename = f"{folder}_{filename_base}.png"
        plt.savefig(filename)
        plt.close()

        mean_time = times.mean()
        std_time = times.std()
        mean_filename = filename.replace(".png", ".txt")
        with open(mean_filename, "w") as outfile:
            outfile.write(
                f"Mean Time (s) = {mean_time:.4f}\nStd Deviation Time (s) = {std_time:.4f}"
            )

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except pd.errors.EmptyDataError:
        print(f"Error: Empty CSV file at {filepath}")
    except pd.errors.ParserError:
        print(f"Error: Could not parse CSV file at {filepath}")
    except KeyError as e:
        print(f"Error: Column 'latency(s)' not found in {filepath}.  Error: {e}")
    except Exception as e:
        print(f"An error occurred processing {filepath}: {e}")


folders = [
    "ex_sklearn_bagging_classification_rand_forest",
    "ex_sklearn_bagging_regression_rand_forest",
    "ex_sklearn_sgd_classification_log_reg",
    "ex_sklearn_sgd_regression_lin_reg",
]

for folder in folders:
    folderpath = Path(f"src/chimera_examples/{folder}")
    for time_csv in folderpath.glob("*.csv"):
        if "X_train" not in str(time_csv) and "y_train" not in str(time_csv):
            sklearn_plot_histogram_and_metrics(str(time_csv), str(folder))
