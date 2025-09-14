#!/bin/bash
#SBATCH --job-name=training_mvmae_rl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

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

# Run the trainer directly (no %runscript), with CWD = submit dir (writable)
apptainer exec --nv \
  --bind "$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR" \
  --pwd  "$SLURM_SUBMIT_DIR" \
  "$IMG" \
  bash -lc 'export MUJOCO_GL=egl; python -m MV_MAE_Implementation.trainer_pipeline --learning_starts 25000 --batch_size 64 --buffer_size 200000 --render_mode rgb_array'
