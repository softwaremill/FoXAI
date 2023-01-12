# AutoXAI

AutoXAI simplifies the application of e**X**plainable **AI** algorithms to explain the
performance of neural network models during training. The library acts as an
aggregator of existing libraries with implementations of various XAI algorithms and
seeks to facilitate and popularize their use in machine learning projects.

Currently, only algorithms related to computer vision are supported, but we plan to
add support for text, tabular and multimodal data problems in the future.

## Table of content:
* [Installation](#installation)
    * [GPU acceleration](#gpu-acceleration)
    * [Manual installation](#manual-installation)
* [Getting started](#getting-started)
* [Development](#development)
    * [Requirements](#requirements)
    * [CUDA](#cuda)
    * [Poetry](#poetry)
    * [pyenv](#pyenv)
        * [Installation errors](#installation-errors)
    * [pre-commit hooks](#pre-commit-hooks-setup)
    * [Note](#note)
        * [Artifacts directory structure](#artifacts-directory-structure)
        * [Examples](#examples)

# Installation

Installation requirements:
* `Python` >= 3.8 & < 4.0

## GPU acceleration

In order to use the torch library with GPU acceleration, you need to install
a dedicated version of torch with support for the installed version of CUDA
drivers in the version supported by the library, at the moment `torch==1.12.1`.
List of `torch` wheels with CUDA support can be found at
[https://download.pytorch.org/whl/torch/](https://download.pytorch.org/whl/torch/).

## Manual installation

If you would like to install from source you can build `wheel` package using `poetry`.
The assumption is that the `poetry` package is installed. You can find how to install
`poetry` [here](#poetry). To build `wheel` package run:

```bash
git clone https://github.com/softwaremill/AutoXAI.git
cd AutoXAI/
poetry install
poetry build
```

As a result you will get `wheel` file inside `dist/` directory that you can install
via `pip`:
```bash
pip install dist/autoxai-0.3.1-py3-none-any.whl
```

# Getting started

To use the AutoXAI library in your ML project, simply add an additional object of type
`WandBCallback` to the `Trainer`'s callback list from the `pytorch-lightning` library.
Currently, only the Weights and Biases tool for tracking experiments is supported.

Below is a code snippet from the example (`example/mnist_wandb.py`):

```python
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import wandb
from autoxai.callbacks.wandb_callback import WandBCallback
from autoxai.explainer.gradient_shap import GradientSHAPCVExplainer
from autoxai.explainer.integrated_gradients import IntegratedGradientsCVExplainer

    ...
    wandb.login()
    wandb_logger = WandbLogger(project=project_name, log_model="all")
    callback = WandBCallback(
        wandb_logger=wandb_logger,
        explainers=[
            IntegratedGradientsCVExplainer(),
            GradientSHAPCVExplainer(),
        ],
        idx_to_label={index: index for index in range(0, 10)},
    )
    model = LitMNIST()
    trainer = Trainer(
        accelerator="gpu",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[callback],
    )
    trainer.fit(model)
```

## CLI

A CLI tool is available to update the artifacts of an experiment tracked in
Weights and Biases. Allows you to create XAI explanations and send them to
W&B offline. To check options type:

```bash
autoxai-wandb-updater --help
```

# Development

## Requirements

The project was tested using Python version `3.8`.

## CUDA

Recommended version of CUDA is `10.2` as it is supported since version
`1.5.0` of `torch`. You can check compatilibity of Your CUDA version
with current version of `torch`:
https://pytorch.org/get-started/previous-versions/.

## Poetry

To separate runtime environments for different services and repositories, it is
recommended to use a virtual Python environment. You can configure `Poetry` to
create new virtual environment in project directory of every repository. To
install `Poetry` follow instruction at https://python-poetry.org/docs/#installing-with-the-official-installer. We are using `Poetry` in version
`1.2.1`. To install specific version You have to provide desired package
version:
```bash
curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.2.1 python3 -
```

After installation configure creation of virtual environments in directory
of project.
```bash
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true
```

The final step is to install all the dependencies defined in the
`pyproject.toml` file.

```bash
poetry install
```

Once all the steps have been completed, the environment is ready to go.
Virtual environment by default will be created with name `.venv` inside
project directory.

## pyenv

`pyenv` is a tool used to manage multiple version of Python. To install
this package follow instructions on project repository page:
https://github.com/pyenv/pyenv#installation. After installation You can
install desired Python version, e.g. `3.8.16`:
```bash
pyenv install 3.8.16
```

The next step is required to be able to use desired version of Python with
`poetry`. To activate specific version of Python interpreter execute command:
```bash
pyenv local 3.8.16 # or `pyenv global 3.8.16`
```

Inside repository with `poetry` You can select specific version of Python
interpreter with command:
```bash
poetry env use 3.8.16
```

After changing interpreter version You have to once again install all
dependencies:
```bash
poetry install
```

### Installation errors

If You encounter errors during dependencies installation You can disable
parallel installer, remove current virtual environment and remove `artifacts`
and `cache` directories from `poetry` root directory (by default is under
`/home/<user>/.cache/pypoetry/`). To disable parallel installer run:
```bash
poetry config installer.parallel false
```

## Pre-commit hooks setup

In order to improve the development experience, please make sure to install
our [pre-commit][https://pre-commit.com/] hooks as the very first step after
cloning the repository:

```bash
poetry run pre-commit install
```

## Note
---
At the moment only explainable algorithms for image classification are
implemented.. In future more algorithms and more computer vision tasks will
be introduces. In the end module should work with all types of tasks (NLP, etc.).

### Examples

In `example/notebooks/` directory You can find notebooks with example usage of this
framework. Scripts in `example/` directory contain samples of training models using
different callbacks.
