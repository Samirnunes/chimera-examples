import time
from logging import INFO, FileHandler, getLogger
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logger = getLogger(__name__)
logger.setLevel(INFO)
logger.addHandler(FileHandler(str(Path(__file__).resolve().parent / "time.txt")))

SAMPLES = 30

logger.info("sample,latency(s)")

for i in range(0, SAMPLES):
    begin = time.time()

    model = RandomForestClassifier(
        n_estimators=4, max_depth=5, max_leaf_nodes=15, n_jobs=1
    )

    X_train = pd.read_csv(str(Path(__file__).resolve().parent / "X_train.csv"))
    y_train = pd.read_csv(str(Path(__file__).resolve().parent / "y_train.csv"))

    model.fit(X_train, np.array(y_train).ravel())

    end = time.time()

    logger.info(f"{i + 1},{end - begin}")
