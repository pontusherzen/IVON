"""
Functions for evaluating LOO loss
"""

import torch
from torch import nn
from torch.utils.data import DataLoader

from ivon import IVON

import pathlib

from sensitivity import get_variance_batch

from typing import Tuple


@torch.no_grad()
def evaluate_average_loss(model: nn.Module,
                          loader: DataLoader,
                          device: torch.device) -> float:
    """
    Iterate through a full data loader and compute average loss.
    """
    running_loss = 0.0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        batch_size = X.shape[0]

        with torch.no_grad():
            logits = model(X)
            running_loss += nn.functional.cross_entropy(logits, y).item() * batch_size / len(loader.dataset)

    return running_loss


def evaluate_loss_and_loo(model: nn.Module, 
                          optimizer: torch.optim.Optimizer,
                          loader: DataLoader, 
                          device: torch.device,
                          train_set_size: int) -> Tuple[float, float]:
    """
    Iterate through a full data loader and compute average
      - loss (mean NLL over dataset)
      - loo loss
    """

    model.eval()
    running_loss = 0.0
    running_loo = 0.0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        batch_size = X.shape[0]

        with torch.no_grad():
            logits = model(X)
            probs = nn.functional.softmax(logits, dim=1)

        # Compute the predictive variance
        n_classes = logits.shape[-1]
        var = get_variance_batch(x=X, model=model, optimizer=optimizer, n_classes=n_classes, dataset_size=train_set_size)

        with torch.no_grad():
            # Get ordinary cross-entropy loss
            running_loss += nn.functional.cross_entropy(logits, y).item() * batch_size / len(loader.dataset)

            # The model's mistake direction, we get a vector that tells us
            # how much the model should increase the correct class probability
            # and decrease the wrong ones.
            y_onehot = torch.nn.functional.one_hot(y, num_classes=n_classes).float().to(device)
            errors = (probs - y_onehot)                 # [B, C]

            # Now we scale the correction by how sensitive the model is.
            # var - tells us how much the model would change if we removed this
            # example from training.
            # If a single example really influenced training we adjust the prediction a lot.
            adjusted_logits = logits + (var @ errors[:, :, None]).squeeze(-1)  # [B, C]

            # Here we check how wrong the model becomes with this "pretend change"
            running_loo += nn.functional.cross_entropy(adjusted_logits, y).item() * batch_size / len(loader.dataset)

    return running_loss, running_loo


def compare_loo_and_test_loss(args):
    import matplotlib.pyplot as plt
    import matplotlib
    from utils import load_dataset, load_checkpoint
    from torch.utils.data import Subset
    from tqdm import tqdm

    # Set rcParams for unified figures
    matplotlib.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "text.latex.preamble": r"\usepackage{amsmath}",
        "figure.titlesize": 16,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.title_fontsize": 12,
    })


    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        from warnings import warn
        warn(f"Found no CUDA device. Running on CPU.")
        device = torch.device("cpu")

    # Initialize dataset
    full_train_set, _, test_set = load_dataset(args.dataset, root=args.data_root)
    if args.limit_train is not None:
        # Use only a subset of training data to evaluate LOO (since Jacobian is expensive to compute)
        idx = torch.randperm(len(full_train_set))[:args.limit_train]
        train_set = Subset(full_train_set, indices=idx.tolist())
    else:
        train_set = full_train_set

    train_loader = DataLoader(train_set, shuffle=False, batch_size=20)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=50)

    # First, dry run each epoch to check that the checkpoints exist before running any expensive computations
    for epoch in args.epochs:
        _ = load_checkpoint(
            optimizer_name=args.optimizer,
            dataset_name=args.dataset,
            model_name=args.model,
            epoch=epoch,
            device=torch.device("cpu")
        )

    # Then do the actual evaluation
    train_losses = []
    train_loos = []
    test_losses = []
    for epoch in tqdm(args.epochs):

        # Load the partially trained checkpoint
        model, optimizer, config = load_checkpoint(
            optimizer_name=args.optimizer,
            dataset_name=args.dataset,
            model_name=args.model,
            epoch=epoch,
            device=device
        )
        model.to(device)
        model.eval()

        # Get LOO estimate from train set
        train_loss, train_loo = evaluate_loss_and_loo(model, optimizer, loader=train_loader, device=device, train_set_size=len(full_train_set))
        train_losses.append(train_loss)
        train_loos.append(train_loo)

        # Get true test loss
        test_loss = evaluate_average_loss(model, loader=test_loader, device=device)
        test_losses.append(test_loss)

    figdir = pathlib.Path("fig/sensitivity")
    if not figdir.exists():
        print(f"Creating directory {figdir} for saving figures.")
        figdir.mkdir(parents=True)

    plt.figure(figsize=(5,5), layout="tight")
    plt.plot(args.epochs, test_losses, label="Test loss", linestyle=":")
    plt.plot(args.epochs, train_loos, label="Train LOO")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    path_jpg = figdir / f"loo__{args.optimizer}__{args.dataset}__{args.model}.jpg"
    path_svg = path_jpg.with_suffix(".svg")
    plt.savefig(path_jpg, dpi=300)
    plt.savefig(path_svg)
    plt.show()


def _test_loo():
    """
    Tests the LOO computation on a small dataloader.
    """
    from utils import load_dataset, load_checkpoint
    from torch.utils.data import Subset

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Create a small subset of the train and test sets of cifar10
    full_train_set, val_set, test_set = load_dataset("cifar10")
    train_set = Subset(full_train_set, indices=list(range(500)))
    test_set = Subset(test_set, indices=list(range(500)))

    train_loader = DataLoader(train_set, batch_size=20, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=20, shuffle=False)

    # Load a trained model
    model, optimizer, config = load_checkpoint("ivon", "cifar10", "resnet18", epoch=200, device=device)
    model.to(device)
    assert isinstance(optimizer, IVON)

    # Get the train and test losses and LOO losses
    train_loss, train_loo = evaluate_loss_and_loo(model, optimizer, train_loader, device=device, train_set_size=len(full_train_set))
    print(f"Train set; loss={train_loss:.4g}; loo={train_loo}")
    test_loss, test_loo= evaluate_loss_and_loo(model, optimizer, test_loader, device=device, train_set_size=len(full_train_set))
    print(f"Train set; loss={test_loss:.4g}; loo={test_loo}")



if __name__ == '__main__':
    import argparse
    from utils import SUPPORTED_ARCHITECTURES, SUPPORTED_DATASETS, SUPPORTED_OPTIMIZERS

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--optimizer", type=str, default="ivon", help="Name of optimizer to use for training (default=%(default)s)", choices=SUPPORTED_OPTIMIZERS)
    parser.add_argument("-m", "--model", type=str, default="resnet18", help="Name of model architecture to train. (default=%(default)s)", choices=SUPPORTED_ARCHITECTURES)
    parser.add_argument("-d", "--dataset", type=str, default="cifar10", help="Name of dataset to train. (default=%(default)s)", choices=SUPPORTED_DATASETS)
    parser.add_argument("-e", "--epochs", type=int, default=[1, 2, 5, 10, 25, 50, 100, 150, 200], help="Epochs to evaluate at. (default=%(default)s)", nargs="+")
    parser.add_argument("--limit-train", type=int, default=None, help="Limit the LOO evaluation to a random subset of the training set of this size (to save computation). (default=%(default)s)")
    parser.add_argument("--data-root", type=pathlib.Path, default=pathlib.Path("./data"), help="Directory where datasets are downloaded to. (default = '%(default)s')")

    args = parser.parse_args()
    compare_loo_and_test_loss(args)


