# FoXAI

FoXAI simplifies the application of e**X**plainable **AI** algorithms to explain the
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
    * [pyenv](#pyenv)
    * [Poetry](#poetry)
    * [pre-commit hooks](#pre-commit-hooks-setup)
    * [Note](#note)
        * [Artifacts directory structure](#artifacts-directory-structure)
        * [Examples](#examples)
* [Star history](#star-history)

# Installation

Installation requirements:
* `Python` >=3.7.2,<3.11

**Important**: For any problems regarding installation we advise to refer first to our [FAQ](FAQ.md).

## GPU acceleration

To use the torch library with GPU acceleration, you need to install
a dedicated version of torch with support for the installed version of CUDA
drivers in the version supported by the library, at the moment `torch>=1.12.1,<2.0.0`.
A list of `torch` wheels with CUDA support can be found at
[https://download.pytorch.org/whl/torch/](https://download.pytorch.org/whl/torch/).

## Manual installation

If you would like to install from the source you can build a `wheel` package using `poetry`.
The assumption is that the `poetry` package is installed. You can find how to install
`poetry` [here](#poetry). To build `wheel` package run:

```bash
git clone https://github.com/softwaremill/FoXAI.git
cd FoXAI/
poetry install
poetry build
```

As a result you will get `wheel` file inside `dist/` directory that you can install
via `pip`:
```bash
pip install dist/foxai-x.y.z-py3-none-any.whl
```

# Getting started

To use the FoXAI library in your ML project, simply add an additional object of type
`WandBCallback` to the `Trainer`'s callback list from the `pytorch-lightning` library.
Currently, only the Weights and Biases tool for tracking experiments is supported.

Below is a code snippet from the example (`example/mnist_wandb.py`):

```python
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import wandb
from foxai.callbacks.wandb_callback import WandBCallback
from foxai.context_manager import CVClassificationExplainers, ExplainerWithParams

    ...
    wandb.login()
    wandb_logger = WandbLogger(project=project_name, log_model="all")
    callback = WandBCallback(
        wandb_logger=wandb_logger,
        explainers=[
            ExplainerWithParams(
                explainer_name=CVClassificationExplainers.CV_INTEGRATED_GRADIENTS_EXPLAINER
            ),
            ExplainerWithParams(
                explainer_name=CVClassificationExplainers.CV_GRADIENT_SHAP_EXPLAINER
            ),
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
W&B offline. This tool is using `hydra` to handle the configuration of `yaml` files.
To check options type:

```bash
foxai-wandb-updater --help
```

Typical usage with configuration in `config/config.yaml`:
```bash
foxai-wandb-updater --config-dir config/ --config-name config
```

Content of `config.yaml`:
```bash
username: <WANDB_USERANEM>
experiment: <WANDB_EXPERIMENT>
run_id: <WANDB_RUN_ID>
classifier: # model class to explain
  _target_: example.streamlit_app.mnist_model.LitMNIST
  batch_size: 1
  data_dir: "."
explainers: # list of explainers to use
 - explainer_with_params:
    explainer_name: CV_GRADIENT_SHAP_EXPLAINER
    kwargs:
      n_steps: 1000
```


## Examples

In `example/notebooks/` directory You can find notebooks with example usage of this
framework. Scripts in `example/` directory contain samples of training models using
different callbacks.

| Tutorial description | Notebook | Google Colab |
|----------------------|----------|--------------|
| Basic usage          | [Notebook](https://github.com/softwaremill/FoXAI/blob/develop/example/notebooks/basic_usage.ipynb)     | <sub>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_Wc-ci3cyYwuSbpbwMSbLlQIiL-VVaXo?usp=sharing)</sub> |


## Explainers list

The table below presents a list of available explainers:

|              Explainer name             |           Is deterministic           |
|:---------------------------------------:|:------------------------------------:|
| CV_OCCLUSION_EXPLAINER                  | yes, if non-random baseline provided |
| CV_INTEGRATED_GRADIENTS_EXPLAINER       | yes, if non-random baseline provided |
| CV_NOISE_TUNNEL_EXPLAINER               | no                                   |
| CV_GRADIENT_SHAP_EXPLAINER              | no                                   |
| CV_LRP_EXPLAINER                        | yes                                  |
| CV_FULL_GRADIENTS_EXPLAINER             | yes                                  |
| CV_GUIDEDGRADCAM_EXPLAINER              | yes                                  |
| CV_LAYER_INTEGRATED_GRADIENTS_EXPLAINER | yes, if non-random baseline provided |
| CV_LAYER_NOISE_TUNNEL_EXPLAINER         | no                                   |
| CV_LAYER_GRADIENT_SHAP_EXPLAINER        | no                                   |
| CV_LAYER_LRP_EXPLAINER                  | yes                                  |
| CV_LAYER_GRADCAM_EXPLAINER              | yes                                  |
| CV_INPUT_X_GRADIENT_EXPLAINER           | yes                                  |
| CV_LAYER_INPUT_X_GRADIENT_EXPLAINER     | yes                                  |
| CV_DEEPLIFT_EXPLAINER                   | yes, if non-random baseline provided |
| CV_LAYER_DEEPLIFT_EXPLAINER             | yes, if non-random baseline provided |
| CV_DEEPLIFT_SHAP_EXPLAINER              | yes, if non-random baseline provided |
| CV_LAYER_DEEPLIFT_SHAP_EXPLAINER        | yes, if non-random baseline provided |
| CV_DECONVOLUTION_EXPLAINER              | yes                                  |
| CV_LAYER_CONDUCTANCE_EXPLAINER          | yes                                  |
| CV_SALIENCY_EXPLAINER                   | yes                                  |
| CV_GUIDED_BACKPOPAGATION_EXPLAINER      | yes                                  |
| CV_XRAI_EXPLAINER                       | no                                   |

# Development

## Requirements

The project was tested using Python version `3.8`.

## CUDA

The recommended version of CUDA is `10.2` as it is supported since version
`1.5.0` of `torch`. You can check the compatibility of your CUDA version
with the current version of `torch`:
https://pytorch.org/get-started/previous-versions/.

As our starting Docker image we were using the one provided by Nvidia: ``nvidia/cuda:10.2-devel-ubuntu18.04``.

If you wish an easy to use docker image we advise to use our ``Dockerfile``.

## pyenv
Optional step, but probably one of the easiest way to actually get Python version with all the needed aditional tools (e.g. pip).

`pyenv` is a tool used to manage multiple versions of Python. To install
this package follow the instructions on the project repository page:
https://github.com/pyenv/pyenv#installation. After installation You can
install desired Python version, e.g. `3.8.16`:
```bash
pyenv install 3.8.16
```

The next step is required to be able to use a desired version of Python with
`poetry`. To activate a specific version of Python interpreter execute the command:
```bash
pyenv local 3.8.16 # or `pyenv global 3.8.16`
```

Inside the repository with `poetry` You can select a specific version of Python
interpreter with the command:
```bash
poetry env use 3.8.16
```

After changing the interpreter version You have to once again install all
dependencies:
```bash
poetry install
```

## Poetry

To separate runtime environments for different services and repositories, it is
recommended to use a virtual Python environment. You can configure `Poetry` to
create a new virtual environment in the project directory of every repository. To
install `Poetry` follow the instruction at https://python-poetry.org/docs/#installing-with-the-official-installer. We are using `Poetry` in version
`1.4.2`. To install a specific version You have to provide desired package
version:
```bash
curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.4.2 python3 -
```
Add poetry to PATH:
```bash
export PATH="/home/ubuntu/.local/bin:$PATH"
echo 'export PATH="/home/ubuntu/.local/bin:$PATH"' >> ~/.bashrc
```

After installation, configure the creation of virtual environments in the directory of the project.
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
A virtual environment by default will be created with the name `.venv` inside
the project directory.

## Pre-commit hooks setup

To improve the development experience, please make sure to install
our [pre-commit](https://pre-commit.com/) hooks as the very first step after
cloning the repository:

```bash
poetry run pre-commit install
```

## Note
---
At the moment only explainable algorithms for image classification are
implemented. In the future more algorithms and more computer vision tasks will
be introduced. In the end, the module should work with all types of tasks (NLP, etc.).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=softwaremill/FoXAI&type=Date)](https://star-history.com/#softwaremill/FoXAI&Date)
