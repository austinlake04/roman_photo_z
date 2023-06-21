#!/bin/bash

conda activate roman_photo_z

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
cd ${SCRIPT_DIR}

export LEPHAREDIR=${SCRIPT_DIR}/LEPHARE
export LEPHAREWORK=${SCRIPT_DIR}/py
export BPZDATAPATH=${SCRIPT_DIR}/DESC_BPZ/src/desc_bpz/data_files/

git submodule update --init --recursive
cd -