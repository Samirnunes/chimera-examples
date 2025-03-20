import json
from pathlib import Path

from chimera.utils import parse_times_file

folders = [
    "ex_chimera_bagging_classification_rand_forest",
    "ex_chimera_bagging_regression_rand_forest",
    "ex_chimera_sgd_classification_log_reg",
    "ex_chimera_sgd_regression_lin_reg",
]

for folder in folders:
    folderpath = Path(f"src/chimera_examples/{folder}")
    for worker_folder in folderpath.iterdir():
        if worker_folder.is_dir():
            filepath = worker_folder / "chimera_time.log"
            with open(str(worker_folder) + "_" + "time" + ".json", "w") as fp:
                json.dump(parse_times_file(filepath), fp)
