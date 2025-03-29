import json
from typing import Tuple

import numpy as np
import pandas as pd
import requests  # type: ignore
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def _holdout(
    X_test_path: str, y_test_path: str, fit_endpoint: str, predict_endpoint: str
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"One or both of the files were not found: {X_test_path}, {y_test_path}"
        )
    except pd.errors.EmptyDataError:
        raise ValueError(
            f"One or both of the files are empty: {X_test_path}, {y_test_path}"
        )
    except Exception as e:
        raise Exception(f"An error occurred while reading the files: {e}")

    print("Fitting the model...")
    try:
        fit_response = requests.post(fit_endpoint)
        fit_response.raise_for_status()
        print("Model fitted successfully.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error during model fitting: {e}")

    X_pred_columns = list(X_test.columns)
    X_pred_rows = X_test.values.tolist()

    print("Making predictions...")
    predict_data = {
        "X_pred_columns": X_pred_columns,
        "X_pred_rows": X_pred_rows,
    }
    headers = {"Content-type": "application/json"}
    try:
        predict_response = requests.post(
            predict_endpoint, data=json.dumps(predict_data), headers=headers
        )
        predict_response.raise_for_status()
        predictions = predict_response.json()
        y_pred = predictions["y_pred_rows"]
        print("Predictions received successfully.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error during prediction: {e}")
    except json.JSONDecodeError as e:
        raise Exception(f"Error decoding JSON response: {e}")
    except KeyError as e:
        raise Exception(f"Key error in JSON response: {e}")

    return np.array(y_pred).ravel(), np.array(y_test).ravel()


def holdout_classification_test(
    X_test_path: str, y_test_path: str, fit_endpoint: str, predict_endpoint: str
) -> None:
    """
    Performs a holdout classification test using a remote model via API endpoints.

    Args:
        X_test_path (str): Path to the X_test CSV file.
        y_test_path (str): Path to the y_test CSV file.
        fit_endpoint (str): URL endpoint for model fitting.
        predict_endpoint (str): URL endpoint for model prediction.
    """

    print("Evaluating the model...")
    y_pred, y_true = _holdout(
        X_test_path, y_test_path, fit_endpoint, predict_endpoint
    )
    metrics = {
        "report": classification_report(
            y_true, np.array(y_pred) > 0.5, output_dict=True
        )
    }

    with open("classification_metrics.json", "w") as f:
        json.dump(metrics, f)


def holdout_regression_test(
    X_test_path: str, y_test_path: str, fit_endpoint: str, predict_endpoint: str
) -> None:
    """
    Performs a holdout regression test using a remote model via API endpoints.

    Args:
        X_test_path (str): Path to the X_test CSV file.
        y_test_path (str): Path to the y_test CSV file.
        fit_endpoint (str): URL endpoint for model fitting.
        predict_endpoint (str): URL endpoint for model prediction.
    """
    print("Evaluating the model...")
    y_pred, y_true = _holdout(
        X_test_path, y_test_path, fit_endpoint, predict_endpoint
    )

    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

    with open("regression_metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    holdout_regression_test(
        "X_test.csv",
        "y_test.csv",
        "http://localhost:8082/v1/chimera-aggregation/fit",
        "http://localhost:8082/v1/chimera-aggregation/predict",
    )
