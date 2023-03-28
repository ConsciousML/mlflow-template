"""Train a model on MNIST data."""
import os
from argparse import ArgumentParser

import mlflow
import six
import requests
from google.auth.transport.requests import Request
from google.oauth2 import id_token

from mymodule.mnist_pytorch_training import mnist_pytorch_training


def _get_oauth_client_id(service_uri: str) -> str:
    """Get the OAuth client ID for the given service URI.

    This is a workaround for the fact that the MLflow client does not support
    authentication using OpenID Connect tokens. This function gets the OAuth
    client ID from the service's redirect URI, which is required to obtain an
    OpenID Connect token.

    Args:
        service_uri (str): The URI of the service to get the OAuth client ID for.
    Returns:
        The OAuth client ID for the given service URI.
    """
    redirect_response = requests.get(service_uri, allow_redirects=False, timeout=60)
    if redirect_response.status_code != 302:
        raise ValueError(f"The URI {service_uri} does not seem to be a valid AppEngine endpoint.")

    redirect_location = redirect_response.headers.get("location")
    if not redirect_location:
        raise ValueError(f"No redirect location for request to {service_uri}")

    parsed = six.moves.urllib.parse.urlparse(redirect_location)
    query_string = six.moves.urllib.parse.parse_qs(parsed.query)
    return query_string["client_id"][0]


def _get_oauth_token(service_uri: str) -> str:
    """Get an OpenID Connect token for the given MLflow tracking URI.

    Args:
        service_uri (str): The MLflow tracking URI to get the token for.
    Returns:
        The OpenID Connect token for the given MLflow tracking URI.
    """
    client_id = _get_oauth_client_id(service_uri)
    open_id_connect_token = id_token.fetch_id_token(Request(), client_id)
    return open_id_connect_token


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

    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri_env)

        if 'mlflow-log-pusher-key.json' in os.listdir():
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(
                os.getcwd(), 'mlflow-log-pusher-key.json'
            )
            os.environ["MLFLOW_TRACKING_TOKEN"] = _get_oauth_token(tracking_uri)

    mnist_pytorch_training(
        epochs=args.epochs, batch_size=args.batch_size, train_on_first_n=args.train_on_first_n
    )
