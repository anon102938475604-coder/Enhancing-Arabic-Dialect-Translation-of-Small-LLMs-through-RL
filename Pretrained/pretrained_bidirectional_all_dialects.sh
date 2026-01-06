#!/bin/bash
#SBATCH -p nvidia                  # partition/queue name
#SBATCH --gres=gpu:a100:1          # request 1 A100 GPUs
#SBATCH --nodes=1                  # 1 node
#SBATCH --ntasks=1                 # 1 task
#SBATCH --cpus-per-task=12         # CPUs for data loading
#SBATCH --mem=256G                 # RAM (adjust up if needed)
#SBATCH --time=1-00:00:00          # walltime (24h here)
#SBATCH -J qwen_zeroshot_bidirectional_all_dialects      # job name
#SBATCH -o %x.out             # save stdout/stderrt

cd /script/path

# --- Load envs/modules  ---
###
###
###

# Optional: performance/env tuning
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=warn

# --- Launch training ---
CUDA_VISIBLE_DEVICES=0 python zero_shot_bidirectional_all_dialects.py

