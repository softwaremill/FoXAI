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