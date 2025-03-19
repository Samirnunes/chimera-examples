import time
from logging import INFO, FileHandler, getLogger
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

SAMPLES = 30


for n_estimators in range(1, 5):
    logger = getLogger(__name__ + str(n_estimators))
    logger.setLevel(INFO)
    logger.addHandler(
        FileHandler(
            str(
                Path(__file__).resolve().parent
                / f"time_{n_estimators}_estimators.txt"
            )
        )
    )

    logger.info("sample,latency(s)")

    for i in range(0, SAMPLES):
        begin = time.time()

        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=5, max_leaf_nodes=15, n_jobs=1
        )

        X_train = pd.read_csv(str(Path(__file__).resolve().parent / "X_train.csv"))
        y_train = pd.read_csv(str(Path(__file__).resolve().parent / "y_train.csv"))

        model.fit(X_train, np.array(y_train).ravel())

        end = time.time()

        logger.info(f"{i + 1},{end - begin}")
