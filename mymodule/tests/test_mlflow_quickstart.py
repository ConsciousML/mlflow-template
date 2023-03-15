"""Dummy module description"""
import mlflow
from mymodule.mlflow_quickstart import mlflow_quickstart


def test_mlflow_quickstart() -> None:
    """Tests the hello world function"""
    mlflow.end_run()
    mlflow_quickstart(1.0, 2.0)
    assert True
