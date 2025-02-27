#!/bin/bash
#SBATCH --job-name=train-transformer
#SBATCH --time=24:00:00
#SBATCH --partition=highgpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --output=train_transformer_%j.out
#SBATCH --error=train_transformer_%j.err

# Load required modules
module load apptainer
module load cuda/cuda-12.4.0

# Run all commands inside the Apptainer container
apptainer exec --nv ~/containers/train-transformer_latest.sif bash <<zzzRunHEREinTheContainer
nvidia-smi
cd ~/git/train-trainsformer
python src/train.py
exit
zzzRunHEREinTheContainer

scancel $SLURM_JOB_ID