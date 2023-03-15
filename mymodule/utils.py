"""This module contains utility functions for mymodule"""
import mlflow
import requests


def set_mlflow_tracking_uri(remote_server_uri: str) -> None:
    """Sets the mlflow tracking uri to the remote server uri

    Args:
        remote_server_uri (str): The uri of the remote server
    """
    response = requests.get(f"{remote_server_uri}/version", timeout=60)
    if response.text != mlflow.__version__:
        raise ValueError(
            f'The version of the remote server {remote_server_uri} is not the same as the '
            f'version of the local client {mlflow.__version__}.'
        )

    mlflow.set_tracking_uri(remote_server_uri)
