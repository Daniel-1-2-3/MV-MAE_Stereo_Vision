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

# ---------------- Apptainer image ----------------
IMG="$SLURM_SUBMIT_DIR/training.sif"
if [[ ! -f "$IMG" ]]; then
  echo "ERROR: $IMG not found"
  exit 2
fi

# ---------------- Project layout ----------------
HOST_PROJECT_ROOT="$SLURM_SUBMIT_DIR"
WORKDIR_IN_CONTAINER="/workspace"

# execute.py must exist at the project root
if [[ ! -f "$HOST_PROJECT_ROOT/execute.py" ]]; then
  echo "FATAL: execute.py not found at:"
  echo "  $HOST_PROJECT_ROOT/execute.py"
  exit 10
fi

# ---------------- Python path inside container ----------------
# Prefer host-mounted code under /workspace so edits/additions require no SIF rebuild.
export APPTAINERENV_PYTHONPATH="/workspace:/opt/src:/opt/src/MV_MAE_Implementation:${PYTHONPATH:-}"

# ---------------- EGL / MuJoCo GL setup ----------------
export APPTAINERENV_MUJOCO_GL=egl
export APPTAINERENV_PYOPENGL_PLATFORM=egl
export APPTAINERENV_MUJOCO_PLATFORM=egl
export APPTAINERENV_DISPLAY=
export APPTAINERENV_LIBGL_ALWAYS_SOFTWARE=0
export APPTAINERENV_MESA_LOADER_DRIVER_OVERRIDE=
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export APPTAINERENV_IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg

# ---- JAX backend selection: IMPORTANT ----
# JAX "gpu" is NOT the platform name you want here; use CUDA PJRT plugin backend.
export APPTAINERENV_JAX_PLATFORMS="cuda,cpu"

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

BIND_FLAGS=( --bind "$HOST_PROJECT_ROOT:$HOST_PROJECT_ROOT" )
BIND_FLAGS+=( --bind "/usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d" )
BIND_FLAGS+=( --bind "$NV_EGL_DIR:$NV_EGL_DIR" )
BIND_FLAGS+=( --bind "$GLVND_DIR:$GLVND_DIR" )
BIND_FLAGS+=( --bind "$HOST_MJP_DEPS:$MJP_DEPS_IN_CONTAINER" )

# Critical bind: mount the entire project to /workspace
BIND_FLAGS+=( --bind "$HOST_PROJECT_ROOT:$WORKDIR_IN_CONTAINER" )

# ---------------- Quick EGL + GPU probe ----------------
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
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

# ---------------- Training with Madrona cache integration ----------------
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  "$IMG" \
  bash -lc '
set -e

echo "=== MuJoCo version ==="
python - <<'"'"'PY'"'"'
import mujoco
print("MuJoCo version:", mujoco.__version__)
PY
echo "======================"

export PYTHONUNBUFFERED=1

# JAX / XLA tuning (optional)
export JAX_TRACEBACK_FILTERING=off
export JAX_DISABLE_CUSOLVER=1
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=.60
export PICK_ENV_DEBUG=1

# Force CUDA PJRT plugin backend path
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"

echo "=== Madrona + GPU detection (inside container) ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  ACTUAL_GPU=$(nvidia-smi -L 2>/dev/null | head -1)
  echo "Actual GPU: $ACTUAL_GPU"
  GPU_MODEL=$(echo "$ACTUAL_GPU" | grep -o "H100\|L40S\|A100\|V100\|RTX" | head -1)
  if [ -z "$GPU_MODEL" ]; then
    GPU_MODEL="unknown"
  fi
else
  echo "WARNING: nvidia-smi not found in container; using generic GPU tag"
  GPU_MODEL="unknown"
fi
GPU_MODEL_LOWER=$(echo "$GPU_MODEL" | tr "[:upper:]" "[:lower:]")

ENV_CONFIG="default"

# Cache build dir lives in your submit directory, shared host<->container
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

echo "========================================="
echo "Starting MV-MAE training with MJX + Madrona"
echo "Watch for Compiling /opt/madrona_mjx/... only on first run."
echo "========================================="

# ---------------- Runtime Python deps (persist on host via $SLURM_SUBMIT_DIR) ----------------
DEPS_PREFIX="'"$SLURM_SUBMIT_DIR"'/.pydeps_prefix"

PY_MM=$(python - <<'"'"'PY'"'"'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)

SITE_PKGS="${DEPS_PREFIX}/lib/python${PY_MM}/site-packages"
BIN_DIR="${DEPS_PREFIX}/bin"

mkdir -p "$DEPS_PREFIX"

# Make prefix visible for imports + CLI entrypoints (prefix must come BEFORE SIF venv packages)
# Also, prefer Madrona build dir so we don’t load two different nanobind modules from different paths.
export PYTHONPATH="/opt/madrona_mjx/build:/workspace:${SITE_PKGS}:${PYTHONPATH:-}"
export PATH="${BIN_DIR}:${PATH}"

# Ensure Madrona runtime shared libs can be found
export LD_LIBRARY_PATH="/opt/madrona_mjx/build:${LD_LIBRARY_PATH:-}"

# ---------------- Ensure JAX is actually CUDA-capable (fixes CPU-only jaxlib in /opt/mvmae_venv) ----------------
echo "=== Ensuring JAX CUDA runtime is installed in ${DEPS_PREFIX} (override CPU-only jaxlib in SIF) ==="

python - <<'"'"'PY'"'"'
import os, sys, importlib.util

os.environ["JAX_PLATFORMS"] = os.environ.get("JAX_PLATFORMS", "cuda,cpu")

def gpu_ok():
    import jax
    devs = jax.devices()
    plats = [getattr(d, "platform", "unknown") for d in devs]
    print("jax.__version__:", jax.__version__)
    print("JAX_PLATFORMS:", os.environ.get("JAX_PLATFORMS"))
    print("JAX devices:", devs)
    print("device platforms:", plats)
    return any(p in ("cuda", "gpu") for p in plats)

# Quick pre-check: do we already see CUDA?
try:
    ok = gpu_ok()
except Exception as e:
    print("pre-check failed:", type(e).__name__, e)
    ok = False

if ok:
    print("[ok] JAX sees CUDA already.")
    raise SystemExit(0)

print("[fix] JAX does NOT see CUDA. Installing/overriding into persistent prefix...")

PY

# Install exact versions (same as your SIF: 0.4.36) into the persistent prefix.
# This is what actually overrides the CPU-only jaxlib/xla_extension in /opt/mvmae_venv.
python -m pip install --upgrade --no-cache-dir --prefix "$DEPS_PREFIX" \
  "jax[cuda12_local]==0.4.36"

# Re-check after install and force a tiny GPU compute
python - <<'"'"'PY'"'"'
import os
os.environ["JAX_PLATFORMS"] = os.environ.get("JAX_PLATFORMS", "cuda,cpu")
import jax, jax.numpy as jnp

print("jax:", jax.__version__)
print("default backend:", jax.default_backend())
print("devices:", jax.devices())

# Hard fail if still CPU-only
plats = [d.platform for d in jax.devices()]
if not any(p in ("cuda", "gpu") for p in plats):
    import jaxlib, jaxlib.xla_extension as xe
    print("FATAL: still no CUDA devices.")
    print("jaxlib:", getattr(jaxlib, "__version__", None), "at", getattr(jaxlib, "__file__", None))
    print("xla_extension:", getattr(xe, "__file__", None))
    print("has GpuAllocatorConfig:", hasattr(xe, "GpuAllocatorConfig"))
    raise SystemExit(42)

x = jnp.ones((1024, 1024), dtype=jnp.float32)
y = (x @ x).block_until_ready()
print("[ok] matmul ran on:", y.device())
PY
echo "============================================"

# ---------------- Install TensorBoard (persistently) ----------------
echo "=== Ensuring TensorBoard is available in ${DEPS_PREFIX} ==="

if python - <<'"'"'PY'"'"'
import importlib.util
ok = importlib.util.find_spec("tensorboard") is not None
print("tensorboard already importable:", ok)
raise SystemExit(0 if ok else 1)
PY
then
  echo "TensorBoard already installed."
else
  echo "Installing tensorboard into persistent prefix..."
  python -m pip install --upgrade --no-cache-dir --prefix "$DEPS_PREFIX" tensorboard
fi

python - <<'"'"'PY'"'"'
import tensorboard
print("TensorBoard version:", getattr(tensorboard, "__version__", "unknown"))
PY
echo "============================================"

# ---------------- Run training ----------------
stdbuf -oL -eL python -u execute.py 2>&1

echo "Training completed."
'

echo "Finished at $(date)"
