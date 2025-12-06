# Reproducing IVON experiments

## Project Goal
The objective of this project is to reproduce the main results of the paper introducing the IVON optimizer
([Shen et al., 2024](https://proceedings.mlr.press/v235/shen24b.html)).
IVON is an optimizer that allows for efficiently training deep learning models in a Bayesian framework, where a diagonal covariance Gaussian posterior is recovered using an Adam-like optimizer

IVON presents a refined and elegant synthesis of several influential prior works in adaptive Bayesian optimization and natural gradient methods, including:
- [**Khan et al.**](https://dl.acm.org/doi/abs/10.5555/3648699.3648980)
- [**Khan & Rue**](https://proceedings.mlr.press/v80/khan18a.html)
- [**Lin et al.**](https://proceedings.mlr.press/v119/lin20d.html)
- [**Zhang et al.**](https://dl.acm.org/doi/10.5555/3454287.3454673)

This project aims to carefully reconstruct the experimental setup, methodology, and quantitative findings presented in the paper to ensure a faithful replication.

For deeper context, motivations, and additional explanations, consult:,

- <https://ysngshn.github.io/research/why-ivon/>
- <https://adaptive-bayesian.ai/blog/ivon/>
- <https://github.com/team-approx-bayes/ivon-experiments>

We also introduce a deterministic fine-tuning model called DIVON which aims to summarize the predictive posterior learned by IVON into a deterministic parameter setting that maintains some of the good calibration and uncertainty estimation of the full posterior, while being less expensive at inference time due to the lack of posterior MC sampling.

## Navigating the code
Our experiments focus on the CIFAR-10 dataset, in order to limit the computational demand. Some utility functions for loading data, data preprocessing, initializing of optimizers etc is provided in [utils.py](./utils.py).

The general experimental setup proceeds by first training model checkpoints with the various hyperparameters, and then loading them for the purposes of experimental evaluation.

### Training models
To train models, the script [train.py](./train.py) can be used. Running
```bash
python3 train.py
```
will train a default ResNet18 model on CIFAR-10 using the IVON optimizer. While training, checkpoints are regularly saved to the [checkpoints/](./checkpoints) directory.

To use non-default settings, these can be passed as command-line arguments, such as
```bash
python3 train.py --dataset cifar100 --optimizer adamw --model resnet34 
```
By default, hyperparameters are loaded from the `config` directory according to the dataset/optimizer combination, but a different config file can be passed using the `--config` flag.

### Sensitivity
To run experiments where high/low sensitivity images are computed, the script [sentitivity.py](./sensitivity.py) can be used. For example, to find the highest and lowest sensitivity images of the cat class (index 3) on CIFAR-10, run
```bash
python3 sensitivity.py --dataset cifar10 --optimizer ivon --epoch 50 --showcase-category 3
```
This assumes that the relevant model (ResNet-18 by default) has been trained and saved as a checkpoint. The relevant plots will be shown to the user and also saved to the [fig/](./fig) directory.

### LOO
To run experiments where the LOO estimate of the generalization loss is compared against the true test loss, run [loo.py](./loo.py). For example, to run on a model trained with AdamW, run
```bash
python3 loo.py --optimizer adamw
```
Evaluating the LOO estimate over the full train set is very slow, so to use a random subset of the train set as an estimate, pass the `--limit-train` argument:
```bash
python3 loo.py --optimizer adamw --limit-train 1000
```

### Other experiments
The remaining experiments, including performance on in-distribution and out-of-distribution data, as well as the DIVON extension, is given in the relevant Jupyter notebooks.

