# import kagglehub

# path = kagglehub.dataset_download(
#    "vinayakshanawad/heart-rate-prediction-to-monitor-stress-level"
# )

# print("Path to dataset files:", path)

# ---------

# from pathlib import Path

# import pandas as pd

# data = pd.read_csv(Path(__file__).resolve().parent / "data.csv")

# X = data.drop("HR", axis="columns")
# y = data[["HR"]]

# -----

from pathlib import Path

import pandas as pd


def load_heart_rate(num_partitions: int) -> None:
    X = pd.read_csv(Path(__file__).resolve().parent / "X_train.csv")
    y = pd.read_csv(Path(__file__).resolve().parent / "y_train.csv")

    X.drop("uuid", axis="columns", inplace=True)

    data_len = len(X)
    partition_size = data_len // num_partitions
    remainder = data_len % num_partitions

    start_index = 0
    for i in range(1, num_partitions + 1):
        end_index = start_index + partition_size
        if i <= remainder:
            end_index += 1

        X_part = pd.DataFrame(X[start_index:end_index])
        y_part = pd.DataFrame(y[start_index:end_index])

        X_part.to_csv(f"X_train_{i}.csv", index=False)
        y_part.to_csv(f"y_train_{i}.csv", index=False)

        start_index = end_index
