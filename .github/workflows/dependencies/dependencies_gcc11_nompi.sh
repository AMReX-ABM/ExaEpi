#!/usr/bin/env bash
#
# Copyright 2020-2022 The AMReX Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

set -eu -o pipefail

sudo apt-get update

sudo apt-get install -y --no-install-recommends \
    build-essential       \
    gcc-11 g++-11         \
    gfortran-11           \
    libopenmpi-dev        \
    openmpi-bin           \
    libhdf5-dev
