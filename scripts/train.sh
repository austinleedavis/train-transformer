#!/bin/bash

nvidia-smi
python src/train.py "$@"
