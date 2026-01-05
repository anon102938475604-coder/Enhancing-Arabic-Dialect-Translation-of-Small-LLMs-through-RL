#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:3
#SBATCH --constrain=80g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --time=4-00:00:00
#SBATCH -J GRPO_3GPU_training
#SBATCH -o logs/%x.out

set -euo pipefail

# --- Project dir & logs ---
cd /path/to/project
mkdir -p logs

# --- Load required modules ---
module load gcc

# --- Clean & activate the right Python ---
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# Proper conda init for non-interactive shells
###
###

# --- Runtime env (minimal) ---
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=warn

# --- Compiler settings for PyTorch/Triton ---
export CC=gcc
export CXX=g++

# Hugging Face caches (avoid deprecated TRANSFORMERS_CACHE)
export HF_HOME="/path/to/hf_home"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
unset TRANSFORMERS_CACHE || true

# --- Launch the GRPO training with 3 GPUs ---
bash main_grpo_3gpu.sh
