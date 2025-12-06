import torch
from torch import nn
from torchvision.transforms.v2 import Compose, RandomCrop, RandomHorizontalFlip, ToImage, ToDtype, Normalize
from torch.utils.data import Dataset

from torchvision.datasets import CIFAR10, CIFAR100

from omegaconf import DictConfig, OmegaConf
from ivon import IVON

import pathlib

from typing import Tuple, Optional


# ImageNet statistics for normalization 
NORM_MEAN = torch.tensor([0.485, 0.456, 0.406])
NORM_STD = torch.tensor([0.229, 0.224, 0.225])


# Valid names of optimizers used in constructors
SUPPORTED_OPTIMIZERS = ["adam", "adamw", "sgd", "ivon"]

# Valid names of model architectures used in constructors
SUPPORTED_ARCHITECTURES = [f"resnet{number}" for number in [18, 34, 50, 101, 152]]

# Valid dataset names
SUPPORTED_DATASETS = ["cifar10", "cifar100"]


def load_dataset(name: str, root: pathlib.Path = pathlib.Path("data")) -> Tuple[Dataset, Dataset]:
    """
    Loads the dataset with name 'name' and returns a tuple (train_set, val_set)
    """
    transform_train = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=NORM_MEAN.tolist(), std=NORM_STD.tolist())
    ])
    transform_val = Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=NORM_MEAN.tolist(), std=NORM_STD.tolist())
    ])

    if name == "cifar10":
        train_set = CIFAR10(root=root, train=True, download=True, transform=transform_train)
        val_set = CIFAR10(root=root, train=False, download=True, transform=transform_val)
    elif name == "cifar100":
        train_set = CIFAR100(root=root, train=True, download=True, transform=transform_train)
        val_set = CIFAR100(root=root, train=False, download=True, transform=transform_val)
    else:
        raise NotImplementedError(f"Unrecognized dataset name '{name}'.")

    return train_set, val_set


def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Takes in a normalized image tensor of shape 
    (batch, 3, height, width)
    and returns a denormalized image tensor of shape
    (batch, height, width, 3)
    """
    img_01 = img_tensor * NORM_STD[None, :, None, None] + NORM_MEAN[None, :, None, None]
    img_01 = img_01.clamp(min=0.0, max=1.0).permute(0, 2, 3, 1)
    return img_01


def init_model(name: str, n_classes: int) -> nn.Module:
    """
    Initializes a torch model based on the name of the architecture and the number of ouput classes.
    """
    from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

    if name not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Provided architecture name '{name}' not supported. Must be one of {SUPPORTED_ARCHITECTURES}.")

    resnet_builders = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152
    }

    assert name in resnet_builders

    builder = resnet_builders[name]
    model = builder()

    # Change the number of output classes according to dataset
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def init_optimizer(model: nn.Module, name: str, config: DictConfig):
    if name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"Provided optimizer name '{name}' not supported. Must be one of {SUPPORTED_OPTIMIZERS}.")

    if name == "ivon":
        return IVON(
            model.parameters(),
            lr=config.lr,
            ess=config.optim.lambda_,
            beta1 = config.optim.beta1,
            beta2 = config.optim.beta2,
            weight_decay=config.optim.weight_decay,
            hess_init=config.optim.h0
        )
    else:
        raise NotImplementedError(f"Initializing optimizer {name} not implemented yet")  # TODO


def save_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer, 
                    dataset_name: str,
                    model_name: str,
                    epoch: int, 
                    config: DictConfig,
                    checkpoint_dir: pathlib.Path = pathlib.Path("checkpoints")):
    """
    Takes a trained model and optimizer config and saves it to file.

    The state is saved to path
        {checkpoint_dir}/{dataset_name}/{model_name}__{optimizer_name}__{epoch}.pt

    The save file contains a state dictionary on the form
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metadata": {"epoch": epoch, "model": model_name, "dataset": dataset_name, "optimizer": optimizer_name},
            "config": config
        }
    """

    directory = checkpoint_dir / dataset_name 
    if not directory.exists():
        print(f"Creating a checkpoint save directory at '{directory}'.")
        directory.mkdir(parents=True)

    # Extract the name of the optimizer from the optimizer object
    if isinstance(optimizer, torch.optim.Adam):
        optimizer_name = "adam"
    elif isinstance(optimizer, torch.optim.AdamW):
        optimizer_name = "adamw"
    elif isinstance(optimizer, torch.optim.SGD):
        optimizer_name = "sgd"
    elif isinstance(optimizer, IVON):
        optimizer_name = "ivon"
    else:
        raise NotImplementedError(f"Unrecognized optimizer type: {type(optimizer)}.")
    assert optimizer_name in SUPPORTED_OPTIMIZERS

    path = directory / f"{model_name}__{optimizer_name}__{epoch}.pt"
    print(f"Saving checkpoint to '{path}' ...")
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": {"epoch": epoch, "model": model_name, "dataset": dataset_name, "optimizer": optimizer_name},
        "config": OmegaConf.to_container(config)
    }

    # Finally save the state
    torch.save(state, path)


def load_checkpoint(optimizer_name: str,
                    dataset_name: str, 
                    model_name: str, 
                    epoch: int, 
                    checkpoint_dir: pathlib.Path = pathlib.Path("checkpoints")) -> Tuple[nn.Module, torch.optim.Optimizer, DictConfig]:
    """
    Reads in a checkpoint and constructs a model and optimizer according to the loaded states.
    Also retrieves hyperparameter config as a omegaconf DictConfig.
    Returns tuple (model, optimizer, config)
    """
    # Check that the optimizer asked for is valid
    if not optimizer_name in SUPPORTED_OPTIMIZERS:
        raise ValueError("Provided optimizer name '{optimizer_name}' not supported. Must be one of {SUPPORTED_OPTIMIZERS}.")

    # Check that save directory exists
    directory = checkpoint_dir / dataset_name
    if not directory.exists():
        raise FileNotFoundError("Found no saved checkpoints with dataset '{dataset_name}'. Directory '{directory}' does not exist.")

    # Create the path based on passed params
    path = directory / f"{model_name}__{optimizer_name}__{epoch}.pt"

    # Check that the specific path exists
    if not path.exists():
        raise FileNotFoundError(f"Found no saved checkpoints with provided areguments {dataset_name=}, {optimizer_name=}, {epoch=}. File '{path}' does not exist.")

    # If the filename is validated, we load the state dictionary
    state = torch.load(path)

    # Extract the metadata
    metadata = state["metadata"]
    config = DictConfig(state["config"])
    print("Loaded config:", config)

    # Initialize model with given hyperparameters, and then load the params
    model = init_model(name=metadata["model"], n_classes=config.n_classes)
    model.load_state_dict(state["model"])

    # If the optimizer is IVON, we initialize an appropriate optimizer setup and load its state
    optimizer = init_optimizer(model, name=optimizer_name, config=config)
    optimizer.load_state_dict(state["optimizer"])

    return model, optimizer, config

