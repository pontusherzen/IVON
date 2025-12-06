
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ivon import IVON

import pathlib
from tqdm import tqdm


@torch.no_grad()
def get_error_batch(x: torch.Tensor, y: torch.Tensor, model: nn.Module, n_classes: int) -> torch.Tensor:
    """
    Get the error estimate on a batch of example inputs x of shape (batch, *) with corresponding true labels y of shape (batch,)
    Returns a tensor of shape (batch, n_classes)

    Follows definitions in appendix C.5. of the IVON paper.
    """

    logits = model(x)
    p = nn.functional.softmax(logits, dim=-1)
    error = p - nn.functional.one_hot(y, num_classes=n_classes)
    return error


def get_variance_batch(x: torch.Tensor, model: nn.Module, ivon: IVON, n_classes: int) -> torch.Tensor:
    """
    Get the variance estimate as described in appendix C.5. of the IVON paper for a batch of inputs x of shape (batch, *).
    Uses the optimizer state of the IVON optimizer to retrieve the relevant variances.
    Returns a tensor of shape (batch, n_classes, n_classes).
    """

    # The implementation here just calls get_variance_sincle on each data point in x and concatenates the results
    sub_results = []
    for i in range(x.size(0)):
        sub_results.append(get_variance_single(x=x[i], model=model, ivon=ivon, n_classes=n_classes))
    return torch.stack(sub_results, dim=0)


def get_variance_single(x: torch.Tensor, model: nn.Module, ivon: IVON, n_classes: int) -> torch.Tensor:
    """
    Get the variance estimate described in appendix C.5. of the IVON paper for a single input x.
    Uses the optimizer state of the IVON optimizer to retrieve the relevant variances.
    Returns a tensor of shape (n_classes, n_classes).
    """

    # Initialize buffers to hold the output jacobian of shape (n_params, n_classes)
    jacobian = torch.zeros(0, n_classes, dtype=x.dtype, device=x.device)

    # Initialize buffer to hold the variance (squared sigma) for each parameter as a vector of shape (n_params,)
    sigma2 = torch.zeros(0, dtype=x.dtype, device=x.device)

    # Loop through all parameters to fill the jacobian and sigma buffers
    for group in ivon.param_groups:
        for param in group["params"]:
            this_param_jacobian_shape = (param.numel(), n_classes)
            jacobian = torch.cat((jacobian, torch.zeros(this_param_jacobian_shape, dtype=x.dtype, device=x.device)), dim=0)

        this_sigma2 = 1.0 / group["ess"] * (group["hess"] + group["weight_decay"])
        sigma2 = torch.cat((sigma2, this_sigma2))

    # Number of parameters should be consistent with the size of the buffers
    n_parameters = jacobian.shape[0]
    assert sigma2.shape == (n_parameters,)

    # Compute jacobian
    for i in range(n_classes):
        output = model(x[None, ...]) 

        # Get i:th row of jacobian = derivative of i:th output logit
        metric = output.squeeze(0)[i]
        assert metric.shape == ()

        ivon.zero_grad()
        metric.backward()

        # Walk through the gradient parameter by parameter and fill the relevant sections of the jacobian
        param_i = 0
        for param in model.parameters():
            assert param.grad is not None
            flattened_grad = param.grad.flatten()
            jacobian[param_i:param_i + flattened_grad.numel(), i] = flattened_grad
            param_i += flattened_grad.numel()
        assert param_i == n_parameters

    # Do the contraction.
    # We use einsum for the sake of memory efficiency since the matrices are large.
    # V = J^T @ diag(sigma²) @ J   <-->   einsum("pc, pq, dq", jacobian, diag(sigma2), jacobian)   <-->   einsum("pc,p,pd", jacobian, sigma2, jacobian)
    v = torch.einsum("pc,p,pd", jacobian, sigma2, jacobian)

    # Sanity check for the shape
    assert v.shape == (n_classes, n_classes)

    return v


def eval_session(args):
    """
    Runn a full sensitivity eval session based on passed user args
    """
    from utils import load_checkpoint, load_dataset, denormalize
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib

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
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Initialize dataset
    _, _, test_set = load_dataset(args.dataset, root=args.data_root)

    # Create a subset containing only the class of interest
    single_category_set = Subset(test_set, [i for i in range(len(test_set)) if test_set[i][1] == args.showcase_category])

    # Create dataloaders
    single_category_loader = DataLoader(single_category_set, shuffle=False, batch_size=10)

    # Load the model, optimizer and hyperparameter config from file
    model, optimizer, config = load_checkpoint(
        optimizer_name="ivon",
        dataset_name=args.dataset,
        model_name=args.model,
        epoch=args.epoch
    )
    assert isinstance(optimizer, IVON)
    model.to(device)
    model.eval()

    # Run through the data for all items of the showcase category and compute sensitivies
    all_sensitivities = []
    for i, (x, y) in enumerate(tqdm(single_category_loader, desc="Evaluating sensitivities")):
        x = x.to(device)
        y = y.to(device)
        v = get_variance_batch(x=x, model=model, ivon=optimizer, n_classes=config.n_classes)
        e = get_error_batch(x=x, y=y, model=model, n_classes=config.n_classes)

        sensitivity = v @ e[:, :, None]  # Shape (batch, nc, nc) @ (batch, nc, 1) -> (batch, nc, 1)
        sensitivity = sensitivity.squeeze(-1)  # Shape (batch, nc)
        sensitivity = sensitivity.norm(2, dim=-1)  # Shape (batch,)
        all_sensitivities += sensitivity.cpu().numpy().tolist()

    # Create a directory for outputting figures
    figdir = pathlib.Path("fig/sensitivity")
    if not figdir.exists():
        print(f"Creating directory '{figdir}' for saving sensitivity figures")
        figdir.mkdir(parents=True)

    # Make a histogram showing the distribution of sensitivities
    fig = plt.figure(figsize=(4,2), layout="tight")
    plt.hist(all_sensitivities)
    plt.xlabel("Sensitivity")
    plt.ylabel("Count")
    plt.yscale("log")
    path_jpg = figdir / f"histogram__{args.dataset}__category{args.showcase_category}__{args.model}__epoch{args.epoch}.jpg"
    path_svg = path_jpg.with_suffix(".svg")
    plt.savefig(path_jpg, dpi=300)
    plt.savefig(path_svg)
    plt.show()

    # Sorte the sensitivities to find the highest and lowest sensitivity images
    sorted_sensitivities = np.argsort(all_sensitivities)
    least_sensitive_i = int(np.argmin(all_sensitivities))
    most_sensitive_i = int(np.argmax(all_sensitivities))
    print(f"{least_sensitive_i=}")
    print(f"{most_sensitive_i=}")
    print(f"sorted_idx={sorted_sensitivities}")

    # Show the images with highest and lowest sensitivities
    fig, axs = plt.subplots(4, 2, figsize=(3,6), layout="tight")
    for i in range(axs.size):
        ax = axs.flatten()[i]
        img_idx = sorted_sensitivities[i]
        norm_img = single_category_set[img_idx][0]
        ax.imshow(
            denormalize(norm_img, dataset_name=args.dataset).squeeze(0)
        )
        ax.axis("off")
    path_jpg = figdir / f"least_sensitive__{args.dataset}__category{args.showcase_category}__{args.model}__epoch{args.epoch}.jpg"
    path_svg = path_jpg.with_suffix(".svg")
    plt.savefig(path_jpg, dpi=300)
    plt.savefig(path_svg)
    plt.suptitle("Least sensitive")
    plt.show()

    fig, axs = plt.subplots(4, 2, figsize=(3,6), layout="tight")
    for i in range(axs.size):
        ax = axs.flatten()[i]
        img_idx = sorted_sensitivities[-i-1]
        norm_img = single_category_set[img_idx][0]
        ax.imshow(
            denormalize(norm_img, dataset_name=args.dataset).squeeze(0)
        )
        ax.axis("off")
    path_jpg = figdir / f"most_sensitive__{args.dataset}__category{args.showcase_category}__{args.model}__epoch{args.epoch}.jpg"
    path_svg = path_jpg.with_suffix(".svg")
    plt.savefig(path_jpg, dpi=300)
    plt.savefig(path_svg)
    plt.suptitle("Most sensitive")
    plt.show()


if __name__ == '__main__':
    # Example arguments

    # To eval on CIFAR10 (default) checkpoint at epoch 50 showcasing the first class (airplane)
    # > python3 sensitivity.py -m cifar10 -e 50 --showcase-category 0

    import argparse
    from utils import SUPPORTED_ARCHITECTURES, SUPPORTED_DATASETS

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="resnet18", help="Name of model architecture to train. (default=%(default)s)", choices=SUPPORTED_ARCHITECTURES)
    parser.add_argument("-d", "--dataset", type=str, default="cifar10", help="Name of dataset to train. (default=%(default)s)", choices=SUPPORTED_DATASETS)
    parser.add_argument("-e", "--epoch", type=int, default=200, help="Epoch of model checkpoint to load. (default=%(default)s)")
    parser.add_argument("--data-root", type=pathlib.Path, default=pathlib.Path("./data"), help="Directory where datasets are downloaded to. (default = '%(default)s')")
    parser.add_argument("--showcase-category", type=int, default=9, help="Index of category to use when getting example images with high/low sensitivity. (default=%(default)s)")

    args = parser.parse_args()
    eval_session(args)

