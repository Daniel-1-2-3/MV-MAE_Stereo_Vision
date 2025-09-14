#!/bin/bash
#SBATCH --job-name=training_mvmae_rl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_output/%x-%j.out

# --- Load Apptainer ---
module load apptainer

# --- Paths (all relative to this folder) ---
IMG=training.sif
WORKDIR=$PWD   # current directory = MV_MAE_Implementation

# --- Optional: HuggingFace / W&B tokens ---
# export HF_TOKEN=...
# export WANDB_API_KEY=...

# --- Install requirements inside container ---
apptainer exec --nv -B $WORKDIR:/workspace $IMG \
    bash -lc "pip install -r /workspace/requirements.txt"

# --- Run training script ---
apptainer exec --nv -B $WORKDIR:/workspace $IMG \
    python /workspace/trainer_pipeline.py --batch_size 256 --buffer_size 200000
