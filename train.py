"""Train a model on MNIST data."""
from argparse import ArgumentParser

from mymodule.mnist_pytorch_training import mnist_pytorch_training

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size to use for training')
    parser.add_argument(
        '--train_on_first_n',
        type=int,
        default=0,
        help='Number of samples to train on. If 0 train on all samples',
    )
    args = parser.parse_args()
    mnist_pytorch_training(
        epochs=args.epochs, batch_size=args.batch_size, train_on_first_n=args.train_on_first_n
    )
