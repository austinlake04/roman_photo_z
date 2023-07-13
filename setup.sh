#!/bin/bash

if ! conda -V; then
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
    bash Mambaforge-$(uname)-$(uname -m).sh -b -p "${HOME}/conda"
    rm -rf Mambaforge-$(uname)-$(uname -m).sh
    source "${HOME}/conda/etc/profile.d/conda.sh"
fi

conda env create -f environment.yml

conda activate roman_photo_z

cd $cwd