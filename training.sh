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

# Don’t let host PYTHONPATH leak into the container (you want site-packages resolution).
unset PYTHONPATH || true

# ---------------- Apptainer image ----------------
IMG="$SLURM_SUBMIT_DIR/training.sif"
if [[ ! -f "$IMG" ]]; then
  echo "ERROR: $IMG not found"
  exit 2
fi

# ---------------- Project layout ----------------
HOST_PROJECT_ROOT="$SLURM_SUBMIT_DIR"
WORKDIR_IN_CONTAINER="/workspace"

if [[ ! -f "$HOST_PROJECT_ROOT/execute.py" ]]; then
  echo "FATAL: execute.py not found at:"
  echo "  $HOST_PROJECT_ROOT/execute.py"
  exit 10
fi

# ---------------- Shell inside container (NO bash) ----------------
CONTAINER_SH="/bin/sh"
if ! apptainer exec "$IMG" "$CONTAINER_SH" -c 'true' >/dev/null 2>&1; then
  if apptainer exec "$IMG" /usr/bin/sh -c 'true' >/dev/null 2>&1; then
    CONTAINER_SH="/usr/bin/sh"
  elif apptainer exec "$IMG" sh -c 'true' >/dev/null 2>&1; then
    CONTAINER_SH="sh"
  else
    echo "FATAL: No usable sh found in container (tried /bin/sh, /usr/bin/sh, sh)."
    exit 11
  fi
fi
echo "Using container shell: $CONTAINER_SH"

# ---------------- Force site-packages imports ----------------
# Critical: DO NOT include /opt/madrona_mjx/src or /opt/src in PYTHONPATH,
# otherwise Python will import madrona_mjx from source-tree.
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_PYTHONPATH="/workspace"

# ---------------- EGL / MuJoCo GL setup ----------------
export APPTAINERENV_MUJOCO_GL=egl
export APPTAINERENV_PYOPENGL_PLATFORM=egl
export APPTAINERENV_MUJOCO_PLATFORM=egl
export APPTAINERENV_DISPLAY=
export APPTAINERENV_LIBGL_ALWAYS_SOFTWARE=0
export APPTAINERENV_MESA_LOADER_DRIVER_OVERRIDE=
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export APPTAINERENV_IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg

# NVIDIA EGL vendor JSON on the HOST
VENDOR_JSON="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if [[ ! -f "$VENDOR_JSON" ]]; then
  echo "FATAL: $VENDOR_JSON not found on host (NVIDIA EGL ICD missing)."
  exit 3
fi
export APPTAINERENV__EGL_VENDOR_LIBRARY_FILENAMES="$VENDOR_JSON"

# Locate libEGL_nvidia.so on HOST
NV_EGL_DIR="$(ldconfig -p | awk '/libEGL_nvidia\.so/{print $NF; exit}' | xargs -r dirname || true)"
for d in /usr/lib/x86_64-linux-gnu/nvidia /usr/lib/nvidia /usr/lib64/nvidia /usr/lib/x86_64-linux-gnu; do
  [[ -z "$NV_EGL_DIR" && -e "$d/libEGL_nvidia.so.0" ]] && NV_EGL_DIR="$d"
done
if [[ -z "${NV_EGL_DIR:-}" || ! -d "$NV_EGL_DIR" ]]; then
  echo "FATAL: Could not find libEGL_nvidia.so* on host."
  exit 4
fi

# GLVND client lib directory
GLVND_DIR="/usr/lib/x86_64-linux-gnu"
[[ -e "$GLVND_DIR/libEGL.so.1" ]] || GLVND_DIR="/usr/lib64"

# ---------------- Binds ----------------
# ---- mujoco_playground external_deps fix (site-packages is read-only) ----
HOST_MJP_DEPS="$SLURM_SUBMIT_DIR/mujoco_playground_external_deps"
mkdir -p "$HOST_MJP_DEPS"
MJP_DEPS_IN_CONTAINER="/opt/mvmae_venv/lib/python3.12/site-packages/mujoco_playground/external_deps"

# ---- Hide Madrona build-tree to avoid accidental source-tree / nanobind double-load ----
HOST_EMPTY_MADRONA_BUILD="$SLURM_SUBMIT_DIR/empty_madrona_build"
mkdir -p "$HOST_EMPTY_MADRONA_BUILD"
MADRONA_BUILD_IN_CONTAINER="/opt/madrona_mjx/build"

BIND_FLAGS=( --bind "$HOST_PROJECT_ROOT:$WORKDIR_IN_CONTAINER" )
BIND_FLAGS+=( --bind "/usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d" )
BIND_FLAGS+=( --bind "$NV_EGL_DIR:$NV_EGL_DIR" )
BIND_FLAGS+=( --bind "$GLVND_DIR:$GLVND_DIR" )
BIND_FLAGS+=( --bind "$HOST_MJP_DEPS:$MJP_DEPS_IN_CONTAINER" )
BIND_FLAGS+=( --bind "$HOST_EMPTY_MADRONA_BUILD:$MADRONA_BUILD_IN_CONTAINER" )

# ---------------- Quick EGL + GPU probe (NO bash) ----------------
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  "$IMG" \
  "$CONTAINER_SH" -c '
set -eu
python - << "PY"
import os
try:
  import torch
  print("torch cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
except Exception as e:
  print("torch probe failed:", repr(e))

import mujoco
try:
  import OpenGL.GL as gl
  ctx = mujoco.GLContext(64, 64); ctx.make_current()
  to_s = lambda b: b.decode("utf-8","ignore") if b else None
  print("OpenGL vendor  :", to_s(gl.glGetString(gl.GL_VENDOR)))
  print("OpenGL renderer:", to_s(gl.glGetString(gl.GL_RENDERER)))
  ctx.free()
except Exception as e:
  print("OpenGL probe failed:", repr(e))
PY
'

# ---------------- Training ----------------
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  "$IMG" \
  "$CONTAINER_SH" -c '
set -eu

# Prefer venv python if present (keeps you on site-packages by default)
PYBIN="python"
if [ -x /opt/mvmae_venv/bin/python ]; then
  PYBIN="/opt/mvmae_venv/bin/python"
fi

echo "=== MuJoCo version ==="
"$PYBIN" - << "PY"
import mujoco
print("MuJoCo version:", mujoco.__version__)
PY
echo "======================"

export PYTHONUNBUFFERED=1

# Hard-force site-packages behavior
export PYTHONNOUSERSITE=1
export PYTHONPATH="/workspace"

# JAX / XLA tuning (optional)
export JAX_TRACEBACK_FILTERING=off
export JAX_DISABLE_CUSOLVER=1
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=.60
export PICK_ENV_DEBUG=1

# Force CUDA first, then CPU fallback
export JAX_PLATFORMS="cuda,cpu"

# ---------------- Madrona cache integration ----------------
GPU_MODEL="unknown"
if command -v nvidia-smi >/dev/null 2>&1; then
  ACTUAL_GPU=$(nvidia-smi -L 2>/dev/null | head -1 || true)
  echo "Actual GPU: ${ACTUAL_GPU:-<none>}"
  GPU_MODEL=$(echo "${ACTUAL_GPU:-}" | grep -o "H100\|L40S\|A100\|V100\|RTX" | head -1 || true)
  [ -n "$GPU_MODEL" ] || GPU_MODEL="unknown"
else
  echo "WARNING: nvidia-smi not found in container; using generic GPU tag"
fi
GPU_MODEL_LOWER=$(echo "$GPU_MODEL" | tr "[:upper:]" "[:lower:]")

ENV_CONFIG="default"
CACHE_BUILD_DIR="'"$SLURM_SUBMIT_DIR"'/build_${GPU_MODEL_LOWER}_${ENV_CONFIG}"
mkdir -p "$CACHE_BUILD_DIR/kernel_cache" "$CACHE_BUILD_DIR/bvh_cache"

export MADRONA_MWGPU_KERNEL_CACHE="$CACHE_BUILD_DIR/kernel_cache/kernel.cache"
export MADRONA_BVH_KERNEL_CACHE="$CACHE_BUILD_DIR/bvh_cache/bvh.cache"

echo "Madrona cache configuration:"
echo "  GPU_MODEL_LOWER = $GPU_MODEL_LOWER"
echo "  ENV_CONFIG      = $ENV_CONFIG"
echo "  MADRONA_MWGPU_KERNEL_CACHE = $MADRONA_MWGPU_KERNEL_CACHE"
echo "  MADRONA_BVH_KERNEL_CACHE   = $MADRONA_BVH_KERNEL_CACHE"
if [ -f "$MADRONA_MWGPU_KERNEL_CACHE" ] && [ -f "$MADRONA_BVH_KERNEL_CACHE" ]; then
  echo "  Cache files found (no recompile expected)."
else
  echo "  No cache files yet; first run will compile and populate them."
fi
echo

# ---------------- Verify madrona_mjx comes from site-packages ----------------
echo "=== madrona_mjx import origin check ==="
"$PYBIN" - << "PY"
import sys, importlib.util, os

def spec(name):
    s = importlib.util.find_spec(name)
    return None if s is None else getattr(s, "origin", None)

print("python:", sys.executable)
print("spec madrona_mjx:", spec("madrona_mjx"))
print("spec madrona_mjx._madrona_mjx_batch_renderer:", spec("madrona_mjx._madrona_mjx_batch_renderer"))
print("spec top-level _madrona_mjx_batch_renderer:", spec("_madrona_mjx_batch_renderer"))

import madrona_mjx
print("madrona_mjx.__file__:", madrona_mjx.__file__)

# Hard fail if we accidentally import from source-tree
if "/site-packages/" not in madrona_mjx.__file__:
    print("FATAL: madrona_mjx is NOT from site-packages. sys.path follows:")
    for p in sys.path[:30]:
        print(" ", p)
    raise SystemExit(42)

print("OK: madrona_mjx is from site-packages.")
PY
echo "======================================="
echo

# ---------------- Runtime Python deps (persist on host via $SLURM_SUBMIT_DIR) ----------------
DEPS_PREFIX="'"$SLURM_SUBMIT_DIR"'/.pydeps_prefix"

PY_MM=$("$PYBIN" - << "PY"
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)

SITE_PKGS="${DEPS_PREFIX}/lib/python${PY_MM}/site-packages"
BIN_DIR="${DEPS_PREFIX}/bin"
mkdir -p "$DEPS_PREFIX"

# Keep it clean: do NOT append any inherited PYTHONPATH
export PYTHONPATH="/workspace:${SITE_PKGS}"
export PATH="${BIN_DIR}:$PATH"

echo "=== Ensuring TensorBoard is available in ${DEPS_PREFIX} ==="
if "$PYBIN" - << "PY"
import importlib.util
ok = importlib.util.find_spec("tensorboard") is not None
print("tensorboard already importable:", ok)
raise SystemExit(0 if ok else 1)
PY
then
  echo "TensorBoard already installed."
else
  echo "Installing tensorboard into persistent prefix..."
  "$PYBIN" -m pip install --upgrade --no-cache-dir --prefix "$DEPS_PREFIX" tensorboard
fi

"$PYBIN" - << "PY"
import tensorboard
print("TensorBoard version:", getattr(tensorboard, "__version__", "unknown"))
PY
echo "============================================"

echo "========================================="
echo "Starting MV-MAE training with MJX + Madrona"
echo "========================================="

# ---------------- Run training ----------------
stdbuf -oL -eL "$PYBIN" -u /workspace/execute.py 2>&1

echo "Training completed."
'

echo "Finished at $(date)"
