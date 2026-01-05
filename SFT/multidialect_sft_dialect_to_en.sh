#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH -J qwen_multidialect_dialect_to_en
#SBATCH -o logs/%x.out

set -euo pipefail

# --- Project dir & logs ---
cd /path/to/project/dir
mkdir -p logs

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

# Hugging Face caches (avoid deprecated TRANSFORMERS_CACHE)
export HF_HOME="/path/to/hf_home"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
unset TRANSFORMERS_CACHE || true

# --- Launch (2 GPUs on this node) ---
python -m torch.distributed.run --nproc_per_node=2 multidialect_sft_dialect_to_en.py