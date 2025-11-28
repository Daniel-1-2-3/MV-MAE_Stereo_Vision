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
cd "$SLURM_SUBMIT_DIR"
module load apptainer/1.3.5 || module load apptainer

# Apptainer image
IMG="$SLURM_SUBMIT_DIR/training.sif"
if [[ ! -f "$IMG" ]]; then
  echo "ERROR: $IMG not found"; exit 2
fi

export APPTAINERENV_PYTHONPATH="/opt/src:/opt/src/MV_MAE_Implementation:${PYTHONPATH:-}"

# EGL on GPU: env + binds
# Environment
export APPTAINERENV_MUJOCO_GL=egl
export APPTAINERENV_PYOPENGL_PLATFORM=egl
export APPTAINERENV_MUJOCO_PLATFORM=egl
export APPTAINERENV_DISPLAY=
export APPTAINERENV_LIBGL_ALWAYS_SOFTWARE=0
export APPTAINERENV_MESA_LOADER_DRIVER_OVERRIDE=
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export APPTAINERENV_IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg

# Paths on HOST we want visible inside the container
VENDOR_JSON="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if [[ ! -f "$VENDOR_JSON" ]]; then
  echo "FATAL: $VENDOR_JSON not found on host (NVIDIA EGL ICD missing). Ask admins to install GLVND/EGL for NVIDIA."
  exit 3
fi
# Tell EGL (inside the container) to use the NVIDIA vendor JSON we bind 1:1
export APPTAINERENV__EGL_VENDOR_LIBRARY_FILENAMES="$VENDOR_JSON"

# Try to locate the directory that contains libEGL_nvidia.so
NV_EGL_DIR="$(ldconfig -p | awk '/libEGL_nvidia\.so/{print $NF; exit}' | xargs -r dirname || true)"
for d in /usr/lib/x86_64-linux-gnu/nvidia /usr/lib/nvidia /usr/lib64/nvidia /usr/lib/x86_64-linux-gnu; do
  [[ -z "$NV_EGL_DIR" && -e "$d/libEGL_nvidia.so.0" ]] && NV_EGL_DIR="$d"
done
if [[ -z "${NV_EGL_DIR:-}" || ! -d "$NV_EGL_DIR" ]]; then
  echo "FATAL: Could not find libEGL_nvidia.so* on host. Ask admins to install NVIDIA EGL libs."
  exit 4
fi

# GLVND client lib directory
GLVND_DIR="/usr/lib/x86_64-linux-gnu"
[[ -e "$GLVND_DIR/libEGL.so.1" ]] || GLVND_DIR="/usr/lib64"

# Build bind flags
BIND_FLAGS=( --bind "$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR" )
BIND_FLAGS+=( --bind "/usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d" )
BIND_FLAGS+=( --bind "$NV_EGL_DIR:$NV_EGL_DIR" )
BIND_FLAGS+=( --bind "$GLVND_DIR:$GLVND_DIR" )

# Probe to see if EGL on GPU
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$SLURM_SUBMIT_DIR" \
  "$IMG" \
  bash -lc '
python - << "PY"
import torch, mujoco, OpenGL.GL as gl
print("torch cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
ctx = mujoco.GLContext(64, 64); ctx.make_current()
to_s = lambda b: b.decode("utf-8","ignore") if b else None
print("OpenGL vendor  :", to_s(gl.glGetString(gl.GL_VENDOR)))
print("OpenGL renderer:", to_s(gl.glGetString(gl.GL_RENDERER)))
ctx.free()
PY
  '
# Training
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$SLURM_SUBMIT_DIR" \
  "$IMG" \
  bash -lc '
    export PYTHONUNBUFFERED=1
    # (MuJoCo/PyOpenGL EGL env is already set via APPTAINERENV_*)
    stdbuf -oL -eL python -u -m MV_MAE_Implementation.train_drqv2_mujoco \
      --render_mode rgb_array \
      --mvmae_patch_size 16 \
      --mvmae_encoder_embed_dim 192 \
      --mvmae_decoder_embed_dim 96 \
      --mvmae_encoder_heads 8 \
      --mvmae_decoder_heads 8 \
      --hidden_dim 512 \
    2>&1
  '