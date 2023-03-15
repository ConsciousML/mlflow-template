"""Tests the mnist pytorch training function
Just a quick training on only 100 samples with 1 epoch to make sure the function runs
"""
import mlflow
from mymodule.mnist_pytorch_training import mnist_pytorch_training


def test_mnist_pytorch_training() -> None:
    """Tests the mnist pytorch training function"""
    mlflow.end_run()
    mnist_pytorch_training(epochs=1, batch_size=1, train_on_first_n=2)
    assert True
