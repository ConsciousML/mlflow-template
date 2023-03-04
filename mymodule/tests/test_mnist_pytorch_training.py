"""Tests the mnist pytorch training function
Just a quick training on only 100 samples with 1 epoch to make sure the function runs
"""
from mymodule.mnist_pytorch_training import mnist_pytorch_training


def test_mnist_pytorch_training() -> None:
    """Tests the mnist pytorch training function"""
    mnist_pytorch_training(epochs=1, batch_size=64, train_on_first_n=100)
    assert True
