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

# Make your code visible to Python inside the container
export APPTAINERENV_PYTHONPATH="/opt/src:/opt/src/MV_MAE_Implementation:${PYTHONPATH:-}"

# Make sure EGL uses the NVIDIA driver inside the container
export APPTAINERENV_MUJOCO_GL=egl
export APPTAINERENV_PYOPENGL_PLATFORM=egl
export APPTAINERENV_DISPLAY=
export APPTAINERENV_LIBGL_ALWAYS_SOFTWARE=0
export APPTAINERENV_MESA_LOADER_DRIVER_OVERRIDE=
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]]; then
  export APPTAINERENV__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
fi

# Verify OpenGL is on the GPU (should say NVIDIA)
apptainer exec --nv \
  --bind "$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR" \
  --pwd  "$SLURM_SUBMIT_DIR" \
  "$IMG" \
  bash -lc '
    python - << "PY"
import torch, mujoco, OpenGL.GL as gl
print("torch cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
ctx = mujoco.GLContext(64, 64); ctx.make_current()
s = lambda b: b.decode("utf-8","ignore") if b else None
print("OpenGL vendor  :", s(gl.glGetString(gl.GL_VENDOR)))
print("OpenGL renderer:", s(gl.glGetString(gl.GL_RENDERER)))
ctx.free()
PY
  '

# Training run
apptainer exec --nv \
  --bind "$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR" \
  --pwd  "$SLURM_SUBMIT_DIR" \
  "$IMG" \
  bash -lc '
    export PYTHONUNBUFFERED=1
    # (Optional) re-export for clarity; APPTAINERENV_* already applied
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export DISPLAY=
    export LIBGL_ALWAYS_SOFTWARE=0
    export MESA_LOADER_DRIVER_OVERRIDE=
    stdbuf -oL -eL python -u -m MV_MAE_Implementation.trainer_pipeline \
      --learning_starts 10_000 \
      --batch_size 64 \
      --buffer_size 100_000 \
      --total_timesteps 500_000 \
      --coef_mvmae 0.005 \
      --render_mode rgb_array \
    2>&1
  '
