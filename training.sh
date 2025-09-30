#!/bin/bash
#SBATCH --job-name=training_mvmae_rl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=50G
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# --------------------------- 0) Environment ---------------------------
# Run from the directory you submitted from
cd "$SLURM_SUBMIT_DIR"

# Load Apptainer
module load apptainer/1.3.5 || module load apptainer

# --------------------------- 1) Paths --------------------------------
IMG="$SLURM_SUBMIT_DIR/training.sif"
if [[ ! -f "$IMG" ]]; then
  echo "ERROR: $IMG not found"
  exit 2
fi

echo "SUBMIT_DIR : $SLURM_SUBMIT_DIR"
echo "IMAGE      : $IMG"
echo

# --------------------------- 2) Run ----------------------------------
# We keep your submit dir binding and run from it inside the container.
# Make your repo importable inside the container via PYTHONPATH.
apptainer exec --nv --cleanenv \
  --bind "$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR" \
  --pwd  "$SLURM_SUBMIT_DIR" \
  --env  PYTHONUNBUFFERED=1 \
  --env  MUJOCO_GL=egl \
  --env  PYTHONPATH="$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/MV_MAE_Implementation:${PYTHONPATH:-}" \
  "$IMG" \
  bash -lc 'stdbuf -oL -eL python -u -m MV_MAE_Implementation.trainer_pipeline \
              --learning_starts 15000 \
              --batch_size 64 \
              --buffer_size 200000 \
              --render_mode rgb_array 2>&1'

ret=$?  # capture exit code
if [[ $ret -ne 0 ]]; then
  echo "Error: job failed with exit code $ret"
  exit $ret
fi

echo "Finished   : $(date)"