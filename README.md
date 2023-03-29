# MLFlow Continuous Integration Template
This repository was made with the [Python Code Quality Continuous Integration (CI)](https://github.com/ConsciousML/python-code-quality-ci) template. For more information on how to use this template, please refer the previous link.

This template to provide the following features:
- A GitHub Actions Continuous Integration (CI) from MLflow (pytest, pylint, black, mypy, bandit).
- Pre-commit hooks to ensure that the code is always clean before commiting
- MLflow and Pytorch example code to track your experiments
- Guideliens to customize the CI to your needs

The main purpose of this repository is to prevent programmers to waste time re-creating their MLFflow
CI over and over each time they have to create a new machine learning project.<br>

The [GitHub Flow] workflow is advised to this template:
- Create a branch for your feature
- Create Pull Request once your feature is ready
- The PR will trigger the CI
- Once the CI passed and the PR is resolved, merge your branch to `main`.

This workflow ensure that the CI is the authority enforcing the code quality and that your production code will always pass the tests.

## Installation
Install Python3.11 and VirutalEnvWrapper:
```bash
sudo apt-get install python3.10 python3.10-venv python3-venv
sudo apt-get install python3-virtualenv

pip install virtualenvwrapper
python3.11 -m pip install virtualenvwrapper

echo "export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/Devel
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source ~/.local/bin/virtualenvwrapper.sh" >> ~/.bashrc
```

Create Python environment:
```bash
mkvirtualenv myenv -p python3.10
```

Install dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install your custom package in your environment:
```bash
python setup.py install --force
```

Install pre-commit hooks:
```bash
pre-commit install
```

## Usage
The process is quite simple:
- Create a project from template pointing to this repository. Follow this [link](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template) for further explanation on how to create a project from template.
- Follow the [Protected Branch Configuration](##-Protected-Branch-Configuration) section.
- Follow the [Installation](##-Installation) section.
- Test the CI by running the `main-ci` manually. To run a workflow manually follow this [link](https://docs.github.com/en/actions/managing-workflow-runs/manually-running-a-workflow).
- Create a branch from `main`.
- Remove the `mymodule` example package and create your own at the root of the repository.
- Create your own package with the approriate tests.
- Edit the `setup.py` file, under the `packages` argument add your module path.
- Create a Pull Request and merge your code on `main`.
- By default, each time you try to merge a PR the CI will be triggered and the user will be allowed to merge if all the tests pass.

Congratulation ! You have now a repository with a functional CI to maintain your code standard to a stellar level ;)

## Protected Branch Configuration
- Go to the main Github page of your project created from template.
- Go to Settings > Branches.
- Click on `Add branch protection rule`
- Tick the following boxes:
    - Require a pull request before merging
    - Require status checks to pass before merging
    - Require branches to be up to date before merging
    - Do not allow bypassing the above settings
- Untick the Require Approval box if you are the only maintainer of the project.


## Features
### Overview
The CI by default is triggered on merge on the `main` branch. <br>
If any of the following job fail, the push/merge will be rejected. <br>
This ensures that the code meets a certain level of quality. <br>

The CI uses following tools:
- `Virtualenv`: creates a virtual environment with the necessary dependencies.
- `Pylint`: check for any linting error. If any is found, the CI triggers and error.
- `Black` to check for any formatting error in the Python code.
- `MyPy`: to check for type hints errors.
- `Bandit` to check for any security vulnerability in the code.
- `Pytest` to run the test suite.

### Customization
This repository comes with multiple configuration file that you can modify as you see fit:
| Package | Description | Configuration file | Job name |
|---------|-------------|--------------------|----------|
| Pylint  | Static code analyzer to enforce best practices | `.pylintrc` | `check-linting` |
| Black   | Code formatter that ensures that every contributor uses the same coding style | `pyproject.toml` under `[tool.black]` section | `check-coding-style` |
| MyPy | Check type hints to improve further code readability | `pyproject.toml` under `[tool.mypy]` section | `check-static-types` |
| Bandit | Checks for security vulnerability in the code | `.bandit` | `check-security-vulnerability` |
| PyTest | Runs a test suite | `pyproject.toml` under `[tool.pytest.ini_options]` section | `run-tests` |

To change the Python version of the CI, edit the `github/workflows/main-ci.yml` file. <br>
Change the value of the `PYTHON_VERSION` env variable to suit your needs.

### Using a Tracking Server on GCP
If you want to host your tracking server on GCP, you can follow this [link](https://blog.axelmendoza.fr/posts/2023-03-27-mlflow-gcp.html).
Otherwise, if you already have your own GCP infrastructure, you need to have:
- A service account named `mlflow-log-pusher` with the Editor role on:
    - Compute Engine
    - Google APIs Service Agent
    - App Engine
- A OAuth 2.0 set up to authenticate your clients to the tracking server.

First, add your tracking server URI to your environment variables:
```bash
export MLFLOW_TRACKING_URI=<your/tracking/uri>
```

Or add it to your `~/.bashrc` file:
```bash
echo "export MLFLOW_TRACKING_URI=<your/tracking/uri>" >> ~/.bashrc
```

Then run the following command to authenticate to the MLflow tracking server:
```python
from mymodule.oauth_mlflow import OAuthMLflow

tracking_uri = '<your/tracking/uri>'
OAuthMLflow(tracking_uri)

# Your MLflow code here
```
The first time you run your code, you will be asked to authenticate to the Tracking Server by providing the `project_id` of your GCP project as well as the password of your GCP admin account. A `mlflow-log-pusher-key.json` file will be created in at the root of this repository. This file is used to authenticate you to the tracking server.

Once the `mlflow-log-pusher-key.json` file is created, you can run your code without being asked to authenticate.

The `train.py` authenticates you by default.



