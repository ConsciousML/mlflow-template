"""This module is used to test the mlflow quickstart"""
import os
from mlflow import log_metric, log_param, log_artifacts

from mymodule.utils import set_mlflow_tracking_uri


def mlflow_quickstart(
    parameter: float, metric: float, remote_server_uri: str | None = None
) -> None:
    """This function is used to test the mlflow quickstart

    Args:
        parameter (float): This is a parameter to log in mlflow.
        metric (float): This is a metric to log in mlflow.
        remote_server_uri (str | None, optional): This is the URI of the remote MLFlow server.
    """

    if remote_server_uri is not None:
        set_mlflow_tracking_uri(remote_server_uri)

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
