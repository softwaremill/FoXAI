# AutoXAI

## Requirements

To separate runtime environments for different services and repositories, it is
recommended to use a virtual Python environment, e.g. `virtualenv`. After
installing it, create a new environment and activate it. The project uses Python
version `3.10`.

In the example below, the `-p` parameter specifies the Python version
and the second parameter the name of the virtual environment, for
example `env`.

```bash
virtualenv -p python3.10 .venv
source .venv/bin/activate
```

The project uses the `poetry` package, which manages the dependencies in the
project. To install it first update the `pip` package and then install `poetry`
version `1.2.1`.

```bash
python -m pip install --upgrade pip
```

Instructions for installation of `poetry`:
https://python-poetry.org/docs/#installing-with-the-official-installer.

The final step is to install all the dependencies defined in the
`pyproject.toml` file.

```bash
poetry install
```

Once all the steps have been completed, the environment is ready to go.

## Pre-commit hooks setup

In order to improve the development experience, please make sure to install
our [pre-commit][https://pre-commit.com/] hooks as the very first step after
cloning the repository:

```bash
poetry install
poetry run pre-commit install
```

pre-commit: https://pre-commit.com/


# Note

At the moment only explainable algorithms for image classification are
implemented to test design of the architecture. In future more algorithms
and more computer vision tasks will be introduces. In the end module should
work with all types of tasks (NLP, etc.).

## Architecture

Module is designed to operate in two modes: offline and online. In offline
mode user can explain already trained model against test data. In online
mode user can attach callback to training framework to perform explanations
of predictions during training at the end of each validation epochs.

Module is using cache directory to store artifacts and configuration similar
to `Tensorboard`. There are many levels of directory structure:
```bash
cache_directory/
└── <date>
    ├── <uuid>
    │   ├── data
    │   │   ├── input_data
    │   │   │   └── <data>.pkl
    │   │   ├── normalized
    │   │   │   └── <data>.pkl
    │   │   ├── original
    │   │   │   └── <data>.pkl
    │   │   └── predictions
    │   │       └── <data>.pkl
    │   ├── explanations
    │   │   ├── <method1>
    │   │   │   └── figures
    │   │   │       ├── attributes.npy
    │   │   │       └── params.json.pkl
    │   │   ├── <method2>
    │   │   │   └── figures
    │   │   │       ├── attributes.npy
    │   │   │       └── params.json.pkl
    │   │   ├── ...
    └── ...
```

Another part of this module is GUI interface to view explanations and
modify parameters of explainable algorithms. As a PoC application in
`streamlit` is developed.
