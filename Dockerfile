FROM nvidia/cuda:10.2-devel-ubuntu18.04
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
        ffmpeg libsm6 libxext6\
    && localedef -f UTF-8 -i en_US en_US.UTF-8 \
    && useradd -m -s /bin/bash ubuntu \
    && echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
USER ubuntu
WORKDIR /home/ubuntu
ENV LANG=en_US.UTF-8 \
    PYENV_ROOT="/home/ubuntu/.pyenv" \
    PATH="/home/ubuntu/.pyenv/bin:/home/ubuntu/.pyenv/shims:$PATH"
# install pyenv & python
ARG PYTHON_VERSION=3.8.16
RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash \
 && pyenv install ${PYTHON_VERSION} \
 && pyenv global ${PYTHON_VERSION} \
 && pip install --upgrade pip
# instead of copy you can fetch repository
# RUN git clone https://github.com/softwaremill/AutoXAI.git && cd AutoXAI/
COPY autoxai/ ./autoxai
COPY poetry.lock .
COPY pyproject.toml .
COPY README.md .
COPY mypy.ini .
RUN pyenv local ${PYTHON_VERSION}
# setup poetry, install deps and build wheel
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.2.1 python3 - \
    && export PATH="/home/ubuntu/.local/bin:$PATH" \
    && echo 'export PATH="/home/ubuntu/.local/bin:$PATH"' >> ~/.bashrc \
    && poetry config virtualenvs.create true \
    && poetry config virtualenvs.in-project true \
    && poetry install --no-root \
    && poetry build
RUN python3 -m pip install dist/autoxai-0.5.1-py3-none-any.whl
RUN python3 -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
