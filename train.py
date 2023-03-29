"""Train a model on MNIST data."""
import os
from argparse import ArgumentParser

from mymodule.mnist_pytorch_training import mnist_pytorch_training
from mymodule.oauth_mlflow import OAuthMLflow


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
    parser.add_argument('--tracking_uri', type=str, default=None, help='MLflow tracking URI')
    args = parser.parse_args()

    tracking_uri_env = os.environ.get('MLFLOW_TRACKING_URI')
    tracking_uri = tracking_uri_env if tracking_uri_env is not None else args.tracking_uri

    # Authenticate with MLflow on GCP and downloads the required service account key
    OAuthMLflow(tracking_uri)

    mnist_pytorch_training(
        epochs=args.epochs, batch_size=args.batch_size, train_on_first_n=args.train_on_first_n
    )
