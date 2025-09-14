#!/bin/bash
#SBATCH --job-name=training_mvmae_rl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# Run from the directory you submitted from
cd "$SLURM_SUBMIT_DIR"

# Load Apptainer
module load apptainer/1.3.5 || module load apptainer

# Image must sit next to this script
IMG="$SLURM_SUBMIT_DIR/training.sif"
if [[ ! -f "$IMG" ]]; then
  echo "ERROR: $IMG not found"; exit 2
fi

# Make both /opt/src and /opt/src/MV_MAE_Implementation visible to Python inside the container
export APPTAINERENV_PYTHONPATH="/opt/src:/opt/src/MV_MAE_Implementation:${PYTHONPATH:-}"

# Run the image's %runscript (starts Xvfb and launches trainer_pipeline.py in the image)
apptainer run --nv "$IMG" \
  --batch_size 64 --buffer_size 200000 --render_mode rgb_array
