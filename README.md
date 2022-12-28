# AutoXAI

# Requirements

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
    │   │   ├── <data>.pkl
    |   |   └─── ...
    │   ├── labels
    │   │   └── idx_to_label.json.pkl
    |   └── training
    |       ├── <epoch>
    |       |   └── model.pt
    ...     ...
```

## Examples

In `example/streamlit_app/` directory You can find sample application with
simple GUI to present interactive explanations of given models.
Scripts in `example/` directory contain samples of training models using
different callbacks.
