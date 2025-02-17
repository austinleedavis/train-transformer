#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/train.sh

nvidia-smi
python src/train.py "$@"
