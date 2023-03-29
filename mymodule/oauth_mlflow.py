"""
Class for authenticating with MLflow on GCP
For more information see:
https://blog.axelmendoza.fr/posts/2023-03-27-mlflow-gcp.html
"""
import os
from subprocess import check_output  # nosec B404
import six
import mlflow
import requests
from google.auth.transport.requests import Request
from google.oauth2 import id_token


class OAuthMLflow:
    """Class for authenticating with MLflow on GCP

    This class will search for a service account key in the current working directory.
    If not found it will download the key from GCP and save it in the current working directory.
    """

    def __init__(
        self,
        tracking_uri: str,
        sa_name: str = 'mlflow-log-pusher',
        sa_key_name: str = 'mlflow-log-pusher-key.json',
    ) -> None:
        """
        Args:
            tracking_uri (str): The MLflow tracking URI to authenticate with.
            sa_key_name (str, optional): The name of the service account key to use.
            sa_name (str, optional): The name of the service account to use.
        """
        self.tracking_uri = tracking_uri
        self.sa_name = sa_name
        self.sa_key_name = sa_key_name
        if self.tracking_uri is None:
            return

        mlflow.set_tracking_uri(self.tracking_uri)

        service_account_key_path = os.path.join(os.getcwd(), self.sa_key_name)
        if not os.path.exists(service_account_key_path):
            self.sa_key_interactive_download()
            print(f'Service account key downloaded in: {service_account_key_path}')

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_key_path
        os.environ["MLFLOW_TRACKING_TOKEN"] = self.get_oauth_token(self.tracking_uri)

    def get_oauth_token(self, tracking_uri: str) -> str:
        """Get an OpenID Connect token for the given MLflow tracking URI.

        Args:
            tracking_uri (str): The MLflow tracking URI to get the token for.
        Returns:
            The OpenID Connect token for the given MLflow tracking URI.
        """
        client_id = self.get_oauth_client_id(tracking_uri)
        open_id_connect_token = id_token.fetch_id_token(Request(), client_id)
        return open_id_connect_token

    def get_oauth_client_id(self, tracking_uri: str) -> str:
        """Get the OAuth client ID for the given service URI.

        This is a workaround for the fact that the MLflow client does not support
        authentication using OpenID Connect tokens. This function gets the OAuth
        client ID from the service's redirect URI, which is required to obtain an
        OpenID Connect token.

        Args:
            tracking_uri (str): The URI of the service to get the OAuth client ID for.
        Returns:
            The OAuth client ID for the given service URI.
        """
        redirect_response = requests.get(tracking_uri, allow_redirects=False, timeout=60)
        if redirect_response.status_code != 302:
            raise ValueError(
                f"The URI {tracking_uri} does not seem to be a valid AppEngine endpoint."
            )

        redirect_location = redirect_response.headers.get("location")
        if not redirect_location:
            raise ValueError(f"No redirect location for request to {tracking_uri}")

        parsed = six.moves.urllib.parse.urlparse(redirect_location)
        query_string = six.moves.urllib.parse.parse_qs(parsed.query)
        return query_string["client_id"][0]

    def ask_user(self, question: str, yes_no_prompt: bool = True) -> bool | str:
        """Ask the user a question.

        Args:
            question (str): The question to ask the user.
            yes_no_prompt (bool): Whether to prompt the user with a yes/no question.
        Returns:
            True if the user answered yes, False otherwise.
        """
        if yes_no_prompt:
            answer = input(f'{question} (Y/n): ')
            return answer.lower() == 'y'

        answer = input(f'{question}: ')
        return answer

    def sa_key_interactive_download(self) -> None:
        """Download the service account key from GCP in interactive mode."""
        if not self.ask_user(
            'Service account for MLflow authentication not found.\n'
            'Are you using an MLflow tracking URI on GCP?'
        ):
            print('Continuing without OAuth authentication.')
            return

        if not self.ask_user(
            f'Do you have a `{self.sa_key_name}` service account with the editor role on:\n'
            '- Compute Engine\n'
            '- Google APIs Service Agent\n'
            '- App Engine'
        ):
            print(
                'Create the service account or copy the key from an existing one in the root'
                f'directory of your project and name it `{self.sa_key_name}`.\n'
                'You are continuing without OAuth authentication.\n'
            )
            return

        project_id = self.ask_user(
            'What is the project id of your GCP project ?', yes_no_prompt=False
        )

        print(
            'The Google Cloud SDK might ask you your login credentials.\n'
            'Downloading service account key from GCP..'
        )
        try:
            check_output(
                [
                    'gcloud',
                    'iam',
                    'service-accounts',
                    'keys',
                    'create',
                    f'./{self.sa_key_name}',
                    '--iam-account',
                    f'{self.sa_name}@{project_id}.iam.gserviceaccount.com',
                ],
                shell=False,
            )  # nosec B603 B607
        except Exception as exception:
            print('Unable to download service account key.')
            raise exception
