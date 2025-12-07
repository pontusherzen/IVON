import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ChainedScheduler, CosineAnnealingLR, LinearLR

from ivon import IVON

from omegaconf import DictConfig, OmegaConf

from utils import save_checkpoint

from typing import Tuple, List, Optional


def train(dataset_name: str,
          model_name: str,
          model: nn.Module,
          optimizer: torch.optim.Optimizer,
          train_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
          config: DictConfig,
          val_loader: Optional[DataLoader[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Tuple[List[float], Optional[List[float]]]:
    """
    Runs a training loop with hyperparameters 'config'.
    Checkpoints are regularly saved to file, and loss statistics are printed while training
    """

    # Initialize device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        from warnings import warn
        device = torch.device("cpu")
        warn("Could not find CUDA device, running on CPU")

    # Create a linear warmup schedule
    warmup = LinearLR(optimizer, start_factor=1.0/config.warmup_epochs, total_iters=config.warmup_epochs)
    annealing = CosineAnnealingLR(optimizer, T_max=config.epochs)
    schedule = ChainedScheduler([warmup, annealing], optimizer)

    # Do the training
    train_losses = []
    val_losses = []
    for epoch in range(config.epochs):
        running_loss = 0.0

        # Training epoch
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # IVON and normal optimizers need to be handled separately, since IVON requires sampling
            if isinstance(optimizer, IVON):
                # Do n_samples training MC samples
                for _ in range(config.optim.n_samples):
                    with optimizer.sampled_params(train=True):
                        optimizer.zero_grad()
                        logits = model(x)
                        loss = nn.functional.cross_entropy(logits, y)
                        loss.backward()
            else:
                optimizer.zero_grad()
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
                loss.backward()

            optimizer.step()
            running_loss += loss.item() * x.size(0) / len(train_loader.dataset)

        # Eval epoch
        if val_loader is not None:
            running_loss_val = 0.0
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                with torch.no_grad():
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits, y)
                running_loss_val += loss.item() * x.size(0) / len(val_loader.dataset)
        else:
            running_loss_val = None

        schedule.step()
        print(f"Epoch {epoch}; train_loss={running_loss:.4g}" 
              + (f"; val_loss={running_loss_val:.4g}" if running_loss_val is not None else ""))

        train_losses.append(running_loss)
        if running_loss_val is not None:
            val_losses.append(running_loss_val)

        # Save model checkpoint
        if (epoch + 1) in [1, 2, 5, 10] or (epoch + 1) % 25 == 0:
            save_checkpoint(model, optimizer, dataset_name=dataset_name, model_name=model_name, epoch=epoch + 1, config=config)

    # Return the losses
    return train_losses, (val_losses if val_losses else None)


def train_session(args):
    """
    Run a full training run based on passed user args
    """
    from utils import load_dataset, init_model, init_optimizer

    # Initialize device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        from warnings import warn
        device = torch.device("cpu")
        warn("Could not find CUDA device, running on CPU")

    # Load relevant dataset
    train_set, val_set, _test_set = load_dataset(args.dataset, root=args.data_root)

    # Get hyperaparameter config
    if args.config_path is None:
        args.config_path = pathlib.Path("configs") / f"{args.dataset}__{args.optimizer}.yaml"
    if not args.config_path.exists():
        raise FileNotFoundError(f"Could not find hyperparameter config file '{args.config_path}'.")
    config = OmegaConf.load(args.config_path)
    assert isinstance(config, DictConfig)

    print(f"Will train with config: {config}")

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=config.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch, shuffle=False, drop_last=False)

    # Initialize model and optimizer
    model = init_model(args.model, n_classes=config.n_classes).to(device)
    optimizer = init_optimizer(model=model, name=args.optimizer, config=config)

    train(dataset_name=args.dataset,
          model_name=args.model,
          model=model,
          optimizer=optimizer,
          train_loader=train_loader,
          config=config,
          val_loader=val_loader
    )


if __name__ == '__main__':
    import argparse
    from utils import SUPPORTED_ARCHITECTURES, SUPPORTED_DATASETS, SUPPORTED_OPTIMIZERS
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="resnet18", help="Name of model architecture to train. (default=%(default)s)", choices=SUPPORTED_ARCHITECTURES)
    parser.add_argument("-d", "--dataset", type=str, default="cifar10", help="Name of dataset to train. (default=%(default)s)", choices=SUPPORTED_DATASETS)
    parser.add_argument("-o", "--optimizer", type=str, default="ivon", help="Name of optimizer to use for training (default=%(default)s)", choices=SUPPORTED_OPTIMIZERS)
    parser.add_argument("-c", "--config-path", type=pathlib.Path, help="Path to config YAML file overriding the default for the (dataset,optimizer) combination.")
    parser.add_argument("--data-root", type=pathlib.Path, default=pathlib.Path("./data"), help="Directory where datasets are downloaded to. (default = '%(default)s')")

    args = parser.parse_args()
    train_session(args)

