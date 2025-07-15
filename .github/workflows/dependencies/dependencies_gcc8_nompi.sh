#!/usr/bin/env bash
#
# Copyright 2020-2022 The AMReX Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

set -eu -o pipefail

sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update

sudo cat <<EOF | sudo tee /etc/apt/sources.list.d/gcc-8.list
deb http://old-releases.ubuntu.com/ubuntu/ impish main restricted universe multiverse
EOF

sudo apt update
apt-cache policy g++-8
apt-cache show g++-8
sudo apt install g++-8
apt-cache policy gfortran-8
apt-cache show gfortran-8
sudo apt install gfortran-8

sudo apt-get install -y --no-install-recommends \
    build-essential    \
    libopenmpi-dev     \
    openmpi-bin        \
    libhdf5-dev
