# syntax=docker/dockerfile:1
#see https://stackoverflow.com/questions/58018300/using-a-pip-cache-directory-in-docker-builds
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

# Update package lists and install dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libjpeg62 \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    libglx-mesa0 \
    libegl1 \
    libgles2 \
    mesa-utils \
    libglib2.0-0 \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    wget \
    git \
    ca-certificates \
    curl \
    openssl \
    cron \
    rclone

RUN wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz  && \
    tar -xzf Python-3.12.0.tgz && \
    cd Python-3.12.0 && \
    ./configure --enable-shared --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    ln -s /usr/local/bin/python3.12 /usr/local/bin/python && \
    echo "/usr/local/lib" > /etc/ld.so.conf.d/python3.12.conf && \
    ldconfig

WORKDIR /

# Keep the container running indefinitely
CMD ["sleep", "infinity"]