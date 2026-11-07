FROM debian:bookworm-slim

ENV PYENV_ROOT="/opt/pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:$PATH"
ENV PYTHON_BUILD_MIRROR_URL="https://mirrors.huaweicloud.com/python"
ENV PYTHON_BUILD_CACHE_PATH="/tmp/python-build-cache"
ENV PYTHON_BUILD_HTTP_CLIENT="curl"
ENV PYTHON_BUILD_CURL_OPTS="--retry 5 --retry-delay 5 --retry-all-errors"

RUN apt-get update --quiet && \
    apt-get install --yes --no-install-recommends \
        build-essential \
        bzip2 \
        ca-certificates \
        curl \
        git \
        bash \
        gcc \
        gdb \
        g++ \
        lcov \
        libbz2-dev \
        libffi-dev \
        libgdbm-compat-dev \
        libgdbm-dev \
        liblzma-dev \
        libmemcached-dev \
        libncurses-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        lzma-dev \
        make \
        pkg-config \
        tk-dev \
        uuid-dev \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*_dists_*

RUN git clone https://github.com/pyenv/pyenv.git "${PYENV_ROOT}" && \
    mkdir -p "${PYENV_ROOT}/plugins" && \
    git clone https://github.com/pyenv/pyenv-update.git \
        "${PYENV_ROOT}/plugins/pyenv-update" && \
    pyenv --version

WORKDIR /opt
RUN mkdir /opt/logs

COPY swebench_docker swebench_docker
COPY --chmod=0755 docker/pyenv/entrypoint.sh .

ENTRYPOINT ["/opt/entrypoint.sh"]