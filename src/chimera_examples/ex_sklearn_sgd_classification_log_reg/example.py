import time
from logging import INFO, FileHandler, getLogger
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

logger = getLogger(__name__)
logger.setLevel(INFO)
logger.addHandler(FileHandler(str(Path(__file__).resolve().parent / "time.txt")))

SAMPLES = 30

logger.info("sample,latency(s)")

for i in range(0, SAMPLES):
    begin = time.time()

    model = SGDClassifier(max_iter=200, epsilon=1e-11, eta0=1e-7)

    X_train = pd.read_csv(str(Path(__file__).resolve().parent / "X_train.csv"))
    y_train = pd.read_csv(str(Path(__file__).resolve().parent / "y_train.csv"))

    model.fit(X_train, np.array(y_train).ravel())

    end = time.time()

    logger.info(f"{i+1},{end - begin}")
