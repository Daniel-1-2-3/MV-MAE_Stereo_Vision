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

if [[ ! -f "$HOST_PROJECT_ROOT/execute.py" ]]; then
  echo "FATAL: execute.py not found at:"
  echo "  $HOST_PROJECT_ROOT/execute.py"
  exit 10
fi

# ---------------- EGL / MuJoCo GL setup ----------------
# NOTE: Don't rely on APPTAINERENV_PYTHONPATH forwarding (your container already sets PYTHONPATH).
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
# mujoco_playground external_deps fix (site-packages is read-only)
HOST_MJP_DEPS="$SLURM_SUBMIT_DIR/mujoco_playground_external_deps"
mkdir -p "$HOST_MJP_DEPS"
MJP_DEPS_IN_CONTAINER="/opt/mvmae_venv/lib/python3.12/site-packages/mujoco_playground/external_deps"

# Hide Madrona build-tree extensions to avoid nanobind double-import crashes
# (we will provide our own /workspace/madrona_mjx shim that imports the *site-packages* extensions)
HOST_EMPTY_MADRONA_BUILD="$SLURM_SUBMIT_DIR/empty_madrona_build"
mkdir -p "$HOST_EMPTY_MADRONA_BUILD"
MADRONA_BUILD_IN_CONTAINER="/opt/madrona_mjx/build"

BIND_FLAGS=( )
BIND_FLAGS+=( --bind "$HOST_PROJECT_ROOT:$WORKDIR_IN_CONTAINER" )
BIND_FLAGS+=( --bind "/usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d" )
BIND_FLAGS+=( --bind "$NV_EGL_DIR:$NV_EGL_DIR" )
BIND_FLAGS+=( --bind "$GLVND_DIR:$GLVND_DIR" )
BIND_FLAGS+=( --bind "$HOST_MJP_DEPS:$MJP_DEPS_IN_CONTAINER" )
BIND_FLAGS+=( --bind "$HOST_EMPTY_MADRONA_BUILD:$MADRONA_BUILD_IN_CONTAINER" )

# ---------------- Quick EGL + GPU probe ----------------
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  --env SUBMIT_DIR="$SLURM_SUBMIT_DIR" \
  "$IMG" \
  /bin/sh -s <<'EOS'
set -e

PYTHON="/opt/mvmae_venv/bin/python"
if [ ! -x "$PYTHON" ]; then PYTHON="python"; fi

"$PYTHON" - <<'PY'
import torch, mujoco
import OpenGL.GL as gl

print("Using container shell:", "/bin/sh")
print("torch cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

ctx = mujoco.GLContext(64, 64); ctx.make_current()
to_s = lambda b: b.decode("utf-8","ignore") if b else None
print("OpenGL vendor  :", to_s(gl.glGetString(gl.GL_VENDOR)))
print("OpenGL renderer:", to_s(gl.glGetString(gl.GL_RENDERER)))
ctx.free()
PY
EOS

# ---------------- Training ----------------
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  --env SUBMIT_DIR="$SLURM_SUBMIT_DIR" \
  "$IMG" \
  /bin/sh -s <<'EOS'
set -e

PYTHON="/opt/mvmae_venv/bin/python"
if [ ! -x "$PYTHON" ]; then PYTHON="python"; fi

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# ---- JAX / XLA tuning ----
export JAX_TRACEBACK_FILTERING=off
export JAX_DISABLE_CUSOLVER=1
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=.60
export JAX_PLATFORMS="cuda,cpu"

# If you want pick_env extra prints:
export PICK_ENV_DEBUG=1

echo "=== MuJoCo version ==="
"$PYTHON" - <<'PY'
import mujoco
print("MuJoCo version:", mujoco.__version__)
PY
echo "======================"

# ---------------- Madrona cache paths (host-persistent) ----------------
# Put caches under $SUBMIT_DIR so they persist across runs.
if command -v nvidia-smi >/dev/null 2>&1; then
  ACTUAL_GPU=$(nvidia-smi -L 2>/dev/null | head -1)
  echo "Actual GPU: $ACTUAL_GPU"
  GPU_MODEL=$(echo "$ACTUAL_GPU" | grep -o "H100\|L40S\|A100\|V100\|RTX" | head -1)
  [ -z "$GPU_MODEL" ] && GPU_MODEL="unknown"
else
  echo "WARNING: nvidia-smi not found in container; using generic GPU tag"
  GPU_MODEL="unknown"
fi
GPU_MODEL_LOWER=$(echo "$GPU_MODEL" | tr "[:upper:]" "[:lower:]")
ENV_CONFIG="default"

CACHE_BUILD_DIR="${SUBMIT_DIR}/build_${GPU_MODEL_LOWER}_${ENV_CONFIG}"
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

# ---------------- Persistent python prefix for extra deps (optional) ----------------
DEPS_PREFIX="${SUBMIT_DIR}/.pydeps_prefix"

PY_MM=$("$PYTHON" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)

SITE_PKGS="${DEPS_PREFIX}/lib/python${PY_MM}/site-packages"
BIN_DIR="${DEPS_PREFIX}/bin"
mkdir -p "$DEPS_PREFIX"

# Make sure /workspace is first
export PYTHONPATH="/workspace:${SITE_PKGS}:${PYTHONPATH:-}"
export PATH="${BIN_DIR}:${PATH}"

# ---------------- CRITICAL FIX: provide /workspace/madrona_mjx shim ----------------
# Reason:
# - You hide /opt/madrona_mjx/build (empty bind) to avoid nanobind double-imports.
# - But the source-tree /opt/madrona_mjx/src/madrona_mjx tries to import madrona_mjx._madrona_mjx_batch_renderer from build.
# - This shim makes `import madrona_mjx` resolve to /workspace and uses the *site-packages* extension module instead.
mkdir -p /workspace/madrona_mjx

cat > /workspace/madrona_mjx/__init__.py <<'PY'
from .renderer import BatchRenderer
__all__ = ["BatchRenderer"]
PY

cat > /workspace/madrona_mjx/renderer.py <<'PY'
try:
    # Site-packages installs these as TOP-LEVEL extension modules:
    #   _madrona_mjx_batch_renderer.so
    from _madrona_mjx_batch_renderer import MadronaBatchRenderer as BatchRenderer  # type: ignore
    from _madrona_mjx_batch_renderer import MadronaBatchRenderer  # type: ignore
except Exception as e:
    raise ImportError(
        "Failed to import site-packages Madrona extension `_madrona_mjx_batch_renderer`.\n"
        "This usually means the extension is missing from site-packages or cannot load its CUDA deps."
    ) from e

__all__ = ["BatchRenderer", "MadronaBatchRenderer"]
PY

cat > /workspace/madrona_mjx/visualizer.py <<'PY'
try:
    from _madrona_mjx_visualizer import MadronaVisualizer  # type: ignore
except Exception:
    MadronaVisualizer = None
__all__ = ["MadronaVisualizer"]
PY

# ---------------- Diagnostics: prove we're importing the correct thing ----------------
echo "=== madrona_mjx import origin check ==="
"$PYTHON" - <<'PY'
import importlib.util
import sys

print("python:", sys.executable)
print("spec madrona_mjx:", importlib.util.find_spec("madrona_mjx").origin)
print("spec _madrona_mjx_batch_renderer:", importlib.util.find_spec("_madrona_mjx_batch_renderer").origin)
print("spec madrona_mjx._madrona_mjx_batch_renderer:", importlib.util.find_spec("madrona_mjx._madrona_mjx_batch_renderer"))
PY
echo

# ---------------- Optional: TensorBoard into persistent prefix ----------------
echo "=== Ensuring TensorBoard is available in ${DEPS_PREFIX} ==="
if "$PYTHON" - <<'PY'
import importlib.util
ok = importlib.util.find_spec("tensorboard") is not None
print("tensorboard already importable:", ok)
raise SystemExit(0 if ok else 1)
PY
then
  echo "TensorBoard already installed."
else
  echo "Installing tensorboard into persistent prefix..."
  "$PYTHON" -m pip install --upgrade --no-cache-dir --prefix "$DEPS_PREFIX" tensorboard
fi

"$PYTHON" - <<'PY'
import tensorboard
print("TensorBoard version:", getattr(tensorboard, "__version__", "unknown"))
PY
echo "============================================"
echo

# ---------------- Run training ----------------
echo "========================================="
echo "Starting MV-MAE training with MJX + Madrona"
echo "========================================="

stdbuf -oL -eL "$PYTHON" -u execute.py 2>&1

echo "Training completed."
EOS

echo "Finished at $(date)"
