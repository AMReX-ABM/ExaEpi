#!/usr/bin/env bash

set -eu -o pipefail

sudo apt-get update -y -qq
sudo apt-get -qq -y install clang-format

