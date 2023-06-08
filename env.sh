#!/bin/bash

conda activate roman_filters

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"

export LEPHAREDIR=${SCRIPT_DIR}/LEPHARE
export LEPHAREWORK=${SCRIPT_DIR}/lepharework