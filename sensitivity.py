
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Normalize

from torch.optim.lr_scheduler import LinearLR

import torchvision
from torchvision.datasets import CIFAR10

from ivon import IVON

from omegaconf import OmegaConf, DictConfig

import pathlib
from tqdm import tqdm

from typing import Tuple



CIFAR10_CONFIG = OmegaConf.create({
    "lr": 0.2,
    "batch": 50,
    "epochs": 200,
    "warmup_epochs": 5,
    "optim": {
        "lambda_": 50000,
        "h0": 0.5,
        "beta1": 0.9,
        "beta2": 1.0 - 1e-5,
        "weight_decay": 0.0002
    },
    "n_classes": 10
})

IMAGENET_CONFIG = OmegaConf.create({
    "lr": 3,
    "batch": 50,
    "epochs": 200,
    "warmup_epochs": 5,
    "optim": {
        "lambda_": 14_000_000, # TODO: Check
        "h0": 0.05,
        "beta1": 0.9,
        "beta2": 1.0 - 1e-6,
        "weight_decay": 5e-5
    },
    "n_classes": 1000
})

CONFIGS = {"cifar10": CIFAR10_CONFIG}

NORM_MEAN = torch.tensor([0.485, 0.456, 0.406])
NORM_STD = torch.tensor([0.229, 0.224, 0.225])


def make_dataset(name: str, root: pathlib.Path = pathlib.Path("data")) -> Tuple[Dataset, Dataset]:
    if name == "cifar10":
        transform = Compose([
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_set = CIFAR10(root=root, train=True, download=True, transform=transform)
        val_set = CIFAR10(root=root, train=False, download=True, transform=transform)
        return train_set, val_set
    else:
        raise NotImplementedError()

def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    img_01 = img_tensor * NORM_STD[None, :, None, None] + NORM_MEAN[None, :, None, None]
    img_01 = img_01.clamp(min=0.0, max=1.0).permute(0, 2, 3, 1)
    return img_01


def init_model(name: str, n_classes: int) -> nn.Module:
    if name == "resnet18":
        model = torchvision.models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model
    else:
        raise NotImplementedError(f"Unsupported model name '{name}'")


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                    dataset_name: str, model_name: str, epoch: int, checkpoint_dir: pathlib.Path = pathlib.Path("checkpoints/sensitivity/")):
    """Saves a model checkpoint to file."""
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir()
    path = checkpoint_dir / f"{model_name}__{dataset_name}__{epoch}.pt"
    print(f"Saving checkpoint to '{path}' ...")
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                    dataset_name: str, model_name: str, epoch: int, checkpoint_dir: pathlib.Path = pathlib.Path("checkpoints/sensitivity/")):
    """Loads a model checkpoint trom file onto 'model'."""
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir()
    path = checkpoint_dir / f"{model_name}__{dataset_name}__{epoch}.pt"

    state = torch.load(path)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])


def train(args, model: nn.Module, optimizer: IVON, train_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]], config: DictConfig):
    device = torch.device("cuda:0")

    schedule = LinearLR(optimizer, start_factor = 1/config.warmup_epochs, total_iters=config.warmup_epochs)
    for epoch in range(config.epochs):
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            with optimizer.sampled_params(train=True):
                optimizer.zero_grad()
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
                loss.backward()

            optimizer.step()
            running_loss += loss.item() * x.size(0) / len(train_loader.dataset)
        schedule.step()
        print(f"Epoch {epoch}; running_loss={running_loss:.4g}")

        if (epoch + 1) in [1, 2, 5, 10, 50, 100, 150, 200]:
            save_checkpoint(model, optimizer, dataset_name=args.dataset_name, model_name=args.model_name, epoch=epoch + 1)


@torch.no_grad()
def get_error_batch(x: torch.Tensor, y: torch.Tensor, model: nn.Module, n_classes: int) -> torch.Tensor:
    logits = model(x)
    p = nn.functional.softmax(logits, dim=-1)
    error = p - nn.functional.one_hot(y, num_classes=n_classes)
    return error

def get_sensitivity_batch(x: torch.Tensor, model: nn.Module, ivon: IVON, n_classes: int) -> torch.Tensor:
    sub_results = []
    for i in range(x.size(0)):
        sub_results.append(get_sensitivity_single(x=x[i], model=model, ivon=ivon, n_classes=n_classes))
    return torch.stack(sub_results, dim=0)


def get_sensitivity_single(x: torch.Tensor, model: nn.Module, ivon: IVON, n_classes: int) -> torch.Tensor:

    jacobian = torch.zeros(0, n_classes, dtype=x.dtype, device=x.device)
    sigma2 = torch.zeros(0, dtype=x.dtype, device=x.device)

    for group in ivon.param_groups:
        for param in group["params"]:
            this_param_jacobian_shape = (param.numel(), n_classes)
            jacobian = torch.cat((jacobian, torch.zeros(this_param_jacobian_shape, dtype=x.dtype, device=x.device)), dim=0)

        this_sigma2 = 1.0 / group["ess"] * (group["hess"] + group["weight_decay"])
        sigma2 = torch.cat((sigma2, this_sigma2))

    n_parameters = jacobian.shape[0]
    assert sigma2.shape == (n_parameters,)

    # Compute jacobian
    for i in range(n_classes):
        output = model(x[None, ...]) 

        # Get i:th row of jacobian
        metric = output[0, i]
        assert metric.shape == ()

        ivon.zero_grad()
        metric.backward()

        param_i = 0
        for param in model.parameters():
            assert param.grad is not None
            flattened_grad = param.grad.flatten()
            jacobian[param_i:param_i + flattened_grad.numel(), i] = flattened_grad
            param_i += flattened_grad.numel()
        assert param_i == n_parameters

    # V = J^T @ diag(sigma²) @ J   <--> einsum("pc, pq, dq", jacobian, diag(sigma2), jacobian)  <--> einsum("pc,p,pd", jacobian, sigma2, jacobian)
    v = torch.einsum("pc,p,pd", jacobian, sigma2, jacobian)
    assert v.shape == (n_classes, n_classes)
    return v


def train_session(args):
    device = torch.device("cuda:0")
    train_set, val_set = make_dataset(args.dataset_name, root=pathlib.Path("/ml/data/"))
    train_set = Subset(train_set, list(range(10000)))

    config = CONFIGS[args.dataset_name]
    train_loader = DataLoader(train_set, batch_size=config.batch, shuffle=True)

    model = init_model(args.model_name, n_classes=config.n_classes).to(device)
    optimizer = IVON(model.parameters(), lr=config.lr, ess=config.optim.lambda_, beta1=config.optim.beta1, beta2=config.optim.beta2, weight_decay=config.optim.weight_decay, hess_init=config.optim.h0)

    train(args, model, optimizer, train_loader, config)

def eval_session(args):
    device = torch.device("cuda:0")
    train_set, val_set = make_dataset(args.dataset_name, root=pathlib.Path("/ml/data/"))

    single_category = 9
    single_category_val_set = Subset(val_set, list(i for i in range(len(val_set)) if val_set[i][1] == single_category))

    config = CONFIGS[args.dataset_name]
    val_loader = DataLoader(val_set, batch_size=10)
    single_category_loader = DataLoader(single_category_val_set, shuffle=False, batch_size=10)

    model = init_model(args.model_name, n_classes=config.n_classes).to(device)
    optimizer = IVON(model.parameters(), lr=config.lr, ess=config.optim.lambda_, beta1=config.optim.beta1, beta2=config.optim.beta2, weight_decay=config.optim.weight_decay, hess_init=config.optim.h0)

    import numpy as np
    import matplotlib.pyplot as plt

    epoch = 100
    load_checkpoint(model, optimizer, dataset_name=args.dataset_name, model_name=args.model_name, epoch=epoch)
    model.eval()

    all_sensitivities = []
    for i, (x, y) in enumerate(tqdm(single_category_loader, desc="Evaluating sensitivities")):
        x = x.to(device)
        y = y.to(device)
        v = get_sensitivity_batch(x=x, model=model, ivon=optimizer, n_classes=config.n_classes)
        e = get_error_batch(x=x, y=y, model=model, n_classes=config.n_classes)

        sensitivity = v @ e[:, :, None]  # Shape (batch, nc, nc) @ (batch, nc, 1) -> (batch, nc, 1)
        sensitivity = sensitivity.squeeze(-1)  # Shape (batch, nc)
        sensitivity = sensitivity.norm(2, dim=-1)  # Shape (batch,)
        all_sensitivities += sensitivity.cpu().numpy().tolist()


    plt.hist(all_sensitivities)
    plt.xlabel("Sensitivity")
    plt.ylabel("Count")
    plt.savefig(f"fig/sensitivity_distribution__{args.dataset_name}__{single_category}__{args.model_name}.jpg", dpi=300)
    plt.show()

    sorted_sensitivities = np.argsort(all_sensitivities)
    least_sensitive_i = int(np.argmin(all_sensitivities))
    most_sensitive_i = int(np.argmax(all_sensitivities))
    print(f"{least_sensitive_i=}")
    print(f"{most_sensitive_i=}")
    print(f"sorted_idx={sorted_sensitivities}")

    fig, axs = plt.subplots(4, 4, figsize=(6,6))
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            flat_idx = i*axs.shape[1] + j
            idx = sorted_sensitivities[flat_idx]
            print("low", idx, "category", single_category_val_set[idx][1])
            axs[i,j].imshow(denormalize(single_category_val_set[idx][0]).squeeze(0))
            axs[i,j].axis("off")
    plt.savefig(f"fig/least_sensitive__{args.dataset_name}__{single_category}__{args.model_name}.jpg", dpi=300)
    plt.suptitle("Least sensitive")
    plt.show()

    fig, axs = plt.subplots(4, 4, figsize=(6,6))
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            flat_idx = -(i*axs.shape[1] + j + 1)
            idx = sorted_sensitivities[flat_idx]
            print("high", idx, "category", single_category_val_set[idx][1])
            axs[i,j].imshow(denormalize(single_category_val_set[idx][0]).squeeze(0))
            axs[i,j].axis("off")
    plt.savefig(f"fig/most_sensitive__{args.dataset_name}__{single_category}__{args.model_name}.jpg", dpi=300)
    plt.suptitle("Most sensitive")
    plt.show()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", default=False, help="Flag to train the model.")
    parser.add_argument("--no-eval", action="store_true", default=False, help="Flag to only train, with no evaluation.")
    parser.add_argument("-m", "--model-name", type=str, default="resnet18", help="Name of model architecture to train.")
    parser.add_argument("-d", "--dataset-name", type=str, default="cifar10", help="Name of dataset to train.")

    args = parser.parse_args()
    if args.train:
        train_session(args)
    if not args.no_eval:
        eval_session(args)
