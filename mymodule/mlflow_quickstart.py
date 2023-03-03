"""This module is used to test the mlflow quickstart"""
import os
from mlflow import log_metric, log_param, log_artifacts


def mlflow_quickstart(parameter: float, metric: float) -> None:
    """This function is used to test the mlflow quickstart

    Args:
        parameter (float): This is a parameter to log in mlflow.
        metric (float): This is a metric to log in mlflow.
    """

    # Log a parameter
    log_param("param_name", parameter)

    # Log a metric
    log_metric("metric_name", metric)

    # Log artifacts, it can be any file or model related data (weights, ...)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

        with open("outputs/test.txt", "w") as file:
            file.write("hello world!")
        log_artifacts("outputs")
