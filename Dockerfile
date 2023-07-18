FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
# from https://hub.docker.com/r/yahwang/ubuntu-pyenv/dockerfile
ARG BUILD_PYTHON_DEPS=" \
        make \
        build-essential \
        libbz2-dev \
        libffi-dev \
        libgdbm-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libnss3-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        xz-utils \
        zlib1g-dev \
        liblzma-dev \
        "
ARG BUILD_OPT_DEPS=" \
        sudo \
        locales \
        git \
        ca-certificates \
        curl\
        "
# basic update & locale setting
RUN apt-get update \
    && apt-get upgrade -yqq \
    && apt-get install -y --no-install-recommends \
        ${BUILD_PYTHON_DEPS} \
        ${BUILD_OPT_DEPS} \
        ffmpeg libsm6 libxext6 wget\
    && localedef -f UTF-8 -i en_US en_US.UTF-8 \
    && useradd -m -s /bin/bash ubuntu \
    && echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# instead of copy you can fetch repository
COPY foxai/ ./foxai
COPY poetry.lock .
COPY pyproject.toml .
COPY README.md .
COPY mypy.ini .
# setup poetry, install deps and build wheel
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.4.2 python3 - \
    && export PATH="/root/.local/bin:$PATH" \
    && echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc \
    && poetry config virtualenvs.create true \
    && poetry config virtualenvs.in-project true \
    && poetry install --no-root \
    && poetry build
RUN python3 -m pip install dist/foxai-0.6.0-py3-none-any.whl
RUN python3 -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN python3 -m pip install jupyterlab==3.1.0
