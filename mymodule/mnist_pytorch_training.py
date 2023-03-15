"""Train a simple neural network on the MNIST dataset using PyTorch.
For more information, see https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html"""
import torch
import mlflow
import requests
from torch import nn
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchvision import datasets
from torchvision.transforms import ToTensor


class NeuralNetwork(nn.Module):
    """This class is used to create a simple neural network."""

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This function is used to forward propagate the input."""
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: str = 'cuda',
) -> None:
    """This function trains a neural network.

    Args:
        dataloader (DataLoader): This is the dataloader to use for training.
        model (nn.Module): This is the model to train.
        loss_fn (nn.Module): This is the loss function to use.
        optimizer (Optimizer): This is the optimizer to use.
        device (str, optional): This is the device to use for training. Defaults to 'cuda'.
    """
    size = len(dataloader.dataset)  # type: ignore

    if dataloader.batch_size is None:
        raise ValueError('The batch size must be specified.')

    nb_batches = int(size / dataloader.batch_size)
    for batch, (input_data, ground_truth) in enumerate(dataloader):
        # Compute prediction error
        input_data = input_data.to(device)
        ground_truth = ground_truth.to(device)

        pred = model(input_data)
        loss = loss_fn(pred, ground_truth)

        # Calling optimizer.zero_grad() sets all the gradients to zero for all the parameters that
        # the optimizer is keeping track of. This allows us to compute the gradients of the current
        # iteration without affecting the gradients of the previous iterations.
        optimizer.zero_grad()

        # Computes gradients of the loss function with respect to all the parameters of the
        # learnable model.
        loss.backward()

        optimizer.step()

        loss = loss.item()
        mlflow.log_metric('train_loss', loss, step=batch)
        print(f"loss: {loss:>7f}  [{batch:>5d}/{nb_batches:>5d}]")


def test(
    epoch: int, dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, device: str = 'cuda'
) -> None:
    """This function tests a neural network on the mnist test dataset."""
    size = float(len(dataloader.dataset))  # type: ignore
    num_batches = float(len(dataloader))
    model.eval()
    test_loss, correct = 0.0, 0.0
    with torch.no_grad():
        for input_data, ground_truth in dataloader:
            input_data, ground_truth = input_data.to(device), ground_truth.to(device)
            pred = model(input_data)
            test_loss += loss_fn(pred, ground_truth).item()
            correct += (pred.argmax(1) == ground_truth).type(torch.float).sum().item()
    test_loss /= num_batches

    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    mlflow.log_metric('test_loss', test_loss, step=epoch)
    mlflow.log_metric('accuracy', correct, step=epoch)


def mnist_pytorch_training(
    epochs: int = 5, batch_size: int = 64, train_on_first_n: int = 0
) -> None:
    """This function trains a simple neural network on the MNIST dataset using PyTorch.

    Args:
        epochs (int, optional): This is the number of epochs to train the model. Defaults to 5.
        batch_size (int, optional): This is the batch size to use for training. Defaults to 64.
        train_on_first_n (int, optional): This is the number of samples to train on. Defaults to 0.
    """

    remote_server_uri = 'http://127.0.0.1:5000'
    response = requests.get(f"{remote_server_uri}/version", timeout=60)
    if response.text != mlflow.__version__:
        raise ValueError(
            f'The version of the remote server {remote_server_uri} is not the same as the '
            f'version of the local client {mlflow.__version__}.'
        )

    mlflow.set_tracking_uri(remote_server_uri)

    mlflow.set_experiment('mnist_pytorch_training')

    # Set MLFlow run name according to time and date
    with mlflow.start_run(run_name=datetime.now().strftime("%m/%d/%Y, %H:%M:%S")):
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('train_on_first_n', train_on_first_n)
        mlflow.log_param('dataset', 'FashionMNIST')

        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )

        if train_on_first_n != 0:
            training_data = torch.utils.data.Subset(training_data, range(train_on_first_n))
            test_data = torch.utils.data.Subset(test_data, range(train_on_first_n))

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

        print(f"Using {device} device")

        model = NeuralNetwork().to(device)
        print(model)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer, device=device)
            test(epoch, test_dataloader, model, loss_fn, device=device)
        print("Done!")


if __name__ == '__main__':
    mnist_pytorch_training(epochs=1, batch_size=64, train_on_first_n=126)
