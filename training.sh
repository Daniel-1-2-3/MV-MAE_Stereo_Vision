#!/bin/bash
#SBATCH --job-name=mjxs_mvmae
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=8:00:00
#SBATCH --account=aip-aspuru-ab
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load apptainer/1.3.5 || module load apptainer

IMG="$SLURM_SUBMIT_DIR/training.sif"
[[ -f "$IMG" ]] || { echo "ERROR: $IMG not found"; exit 2; }

HOST_PROJECT_ROOT="$SLURM_SUBMIT_DIR"
WORKDIR_IN_CONTAINER="/workspace"

[[ -f "$HOST_PROJECT_ROOT/execute.py" ]] || { echo "FATAL: execute.py not found"; exit 10; }

# ---- forward env into container ----
export APPTAINERENV_PYTHONPATH="/workspace:/opt/src:/opt/src/MV_MAE_Implementation"
export APPTAINERENV_MUJOCO_GL=egl
export APPTAINERENV_PYOPENGL_PLATFORM=egl
export APPTAINERENV_MUJOCO_PLATFORM=egl
export APPTAINERENV_DISPLAY=
export APPTAINERENV_LIBGL_ALWAYS_SOFTWARE=0
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export APPTAINERENV_IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg

VENDOR_JSON="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
[[ -f "$VENDOR_JSON" ]] || { echo "FATAL: $VENDOR_JSON not found"; exit 3; }
export APPTAINERENV__EGL_VENDOR_LIBRARY_FILENAMES="$VENDOR_JSON"

NV_EGL_DIR="$(ldconfig -p | awk '/libEGL_nvidia\.so/{print $NF; exit}' | xargs -r dirname || true)"
for d in /usr/lib/x86_64-linux-gnu/nvidia /usr/lib/nvidia /usr/lib64/nvidia /usr/lib/x86_64-linux-gnu; do
  [[ -z "$NV_EGL_DIR" && -e "$d/libEGL_nvidia.so.0" ]] && NV_EGL_DIR="$d"
done
[[ -n "${NV_EGL_DIR:-}" && -d "$NV_EGL_DIR" ]] || { echo "FATAL: Could not find libEGL_nvidia.so*"; exit 4; }

GLVND_DIR="/usr/lib/x86_64-linux-gnu"
[[ -e "$GLVND_DIR/libEGL.so.1" ]] || GLVND_DIR="/usr/lib64"

HOST_MJP_DEPS="$SLURM_SUBMIT_DIR/mujoco_playground_external_deps"
mkdir -p "$HOST_MJP_DEPS"
MJP_DEPS_IN_CONTAINER="/opt/mvmae_venv/lib/python3.12/site-packages/mujoco_playground/external_deps"

BIND_FLAGS=(
  --bind "$HOST_PROJECT_ROOT:$WORKDIR_IN_CONTAINER"
  --bind "/usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d"
  --bind "$NV_EGL_DIR:$NV_EGL_DIR"
  --bind "$GLVND_DIR:$GLVND_DIR"
  --bind "$HOST_MJP_DEPS:$MJP_DEPS_IN_CONTAINER"
)

apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  "$IMG" \
  bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1

echo "=== GPU ==="
nvidia-smi || true
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "=== JAX ==="
python - <<PY
import jax, jaxlib
print("jax", jax.__version__, "jaxlib", jaxlib.__version__)
print("devices:", jax.devices())
print("default_backend:", jax.default_backend())
PY

echo "=== MuJoCo ==="
python - <<PY
import mujoco
print("MuJoCo:", mujoco.__version__)
PY

echo "=== Run ==="
python -u execute.py
'
