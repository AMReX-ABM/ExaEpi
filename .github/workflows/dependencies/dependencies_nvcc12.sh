#!/usr/bin/env bash
#
# Copyright 2020 Axel Huebl
#
# License: BSD-3-Clause-LBNL

set -eu -o pipefail

sudo apt-get -qqq update
sudo apt-get install -y \
    build-essential     \
    ca-certificates     \
    cmake               \
    g++                 \
    gnupg               \
    libopenmpi-dev      \
    openmpi-bin         \
    pkg-config          \
    wget                \
    libhdf5-openmpi-dev

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" \
    | sudo tee /etc/apt/sources.list.d/cuda.list
sudo apt-get update
sudo apt-get install -y \
    cuda-command-line-tools-12-6 \
    cuda-compiler-12-6           \
    cuda-cupti-dev-12-6          \
    cuda-minimal-build-12-6      \
    cuda-nvml-dev-12-6           \
    cuda-nvtx-12-6               \
    libcurand-dev-12-6
sudo ln -s cuda-12.6 /usr/local/cuda

