# AutoXAI

## Requirements

This package was tested using `Python3.8`.

## Environment

To prepare virtual environment You can use `virtualenv`.
```bash
virtualenv -p python3.8 <ENV_NAME>
```

Activate new virtual environment:
```bash
source <ENV_NAME>/bin/activate
```

Firstly upgrade `pip` package:
```bash
python3 -m pip install --upgrade pip
```

Now You can install all dependencies from `requirements.txt` file:
```bash
python3 -m pip install -r requirements.txt
```

And finally, You can install all dependencies from `poetry.lock` file.
```bash
poetry install
```

## Development

We are using `Makefile` to perform automatic linting, style and type checks.
To perform them just type:
```bash
make all
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
