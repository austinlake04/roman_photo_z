#!/bin/bash

if ! conda -V; then
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
    bash Mambaforge-$(uname)-$(uname -m).sh -b -p "${HOME}/conda"
    rm -rf Mambaforge-$(uname)-$(uname -m).sh
    source "${HOME}/conda/etc/profile.d/conda.sh"
    source "${HOME}/conda/etc/profile.d/mamba.sh"
    conda activate
fi

conda env create -f env.yaml

cwd=$PWD
cd $(cd "$(dirname "$0")"; pwd)

git submodule update --init --recursive
source env.sh
cd LEPHARE
python setup.py install
cd source
make

cd ${cwd}