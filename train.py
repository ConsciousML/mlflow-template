"""Train a model on MNIST data."""
from mymodule.mnist_pytorch_training import mnist_pytorch_training

if __name__ == '__main__':
    mnist_pytorch_training(epochs=1, batch_size=64, train_on_first_n=126)
