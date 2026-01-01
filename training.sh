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
WORKDIR_IN_CONTAINER="/workspace"

if [[ ! -f "$IMG" ]]; then
  echo "FATAL: Apptainer image not found: $IMG"
  exit 2
fi

if [[ ! -f "$SLURM_SUBMIT_DIR/execute.py" ]]; then
  echo "FATAL: execute.py not found in submit dir: $SLURM_SUBMIT_DIR"
  exit 3
fi

# ---------------- Host EGL/NVIDIA ----------------
VENDOR_JSON="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if [[ ! -f "$VENDOR_JSON" ]]; then
  echo "FATAL: Missing NVIDIA EGL vendor JSON on host: $VENDOR_JSON"
  exit 4
fi

NV_EGL_DIR="$(ldconfig -p | awk '/libEGL_nvidia\.so/{print $NF; exit}' | xargs -r dirname || true)"
for d in /usr/lib/x86_64-linux-gnu/nvidia /usr/lib/nvidia /usr/lib64/nvidia /usr/lib/x86_64-linux-gnu; do
  [[ -z "$NV_EGL_DIR" && -e "$d/libEGL_nvidia.so.0" ]] && NV_EGL_DIR="$d"
done
if [[ -z "${NV_EGL_DIR:-}" || ! -d "$NV_EGL_DIR" ]]; then
  echo "FATAL: Could not locate libEGL_nvidia.so* directory on host."
  exit 5
fi

GLVND_DIR="/usr/lib/x86_64-linux-gnu"
[[ -e "$GLVND_DIR/libEGL.so.1" ]] || GLVND_DIR="/usr/lib64"

# ---------------- Persistent python prefix (host) ----------------
DEPS_PREFIX="$SLURM_SUBMIT_DIR/.pydeps_prefix"
mkdir -p "$DEPS_PREFIX"

# mujoco_playground external_deps workaround (host dir bind)
HOST_MJP_DEPS="$SLURM_SUBMIT_DIR/mujoco_playground_external_deps"
mkdir -p "$HOST_MJP_DEPS"
MJP_DEPS_IN_CONTAINER="/opt/mvmae_venv/lib/python3.12/site-packages/mujoco_playground/external_deps"

# ---------------- Container environment (EGL/MuJoCo/JAX) ----------------
export APPTAINERENV_MUJOCO_GL=egl
export APPTAINERENV_PYOPENGL_PLATFORM=egl
export APPTAINERENV_MUJOCO_PLATFORM=egl
export APPTAINERENV_DISPLAY=
export APPTAINERENV__EGL_VENDOR_LIBRARY_FILENAMES="$VENDOR_JSON"

export APPTAINERENV_JAX_TRACEBACK_FILTERING=off
export APPTAINERENV_JAX_DISABLE_CUSOLVER=1
export APPTAINERENV_XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
export APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
export APPTAINERENV_XLA_PYTHON_CLIENT_ALLOCATOR=platform
export APPTAINERENV_XLA_PYTHON_CLIENT_MEM_FRACTION=.60
export APPTAINERENV_JAX_PLATFORMS="cuda,cpu"

export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_PICK_ENV_DEBUG=1

# ---------------- Binds ----------------
# IMPORTANT CHANGE: we NO LONGER bind an empty dir over /opt/madrona_mjx/build
# because the site-packages extension depends on runtime libs in that directory.
BIND_FLAGS=( )
BIND_FLAGS+=( --bind "$SLURM_SUBMIT_DIR:$WORKDIR_IN_CONTAINER" )
BIND_FLAGS+=( --bind "/usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d" )
BIND_FLAGS+=( --bind "$NV_EGL_DIR:$NV_EGL_DIR" )
BIND_FLAGS+=( --bind "$GLVND_DIR:$GLVND_DIR" )
BIND_FLAGS+=( --bind "$HOST_MJP_DEPS:$MJP_DEPS_IN_CONTAINER" )

# ---------------- 0) GPU/EGL probe ----------------
apptainer exec --nv "${BIND_FLAGS[@]}" --pwd "$WORKDIR_IN_CONTAINER" --env SUBMIT_DIR="$SLURM_SUBMIT_DIR" \
  "$IMG" /bin/sh -lc '
set -eu
PY="/opt/mvmae_venv/bin/python"; [ -x "$PY" ] || PY="python"
$PY - <<'"'"'PY'"'"'
import torch, mujoco
import OpenGL.GL as gl
print("torch cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
ctx = mujoco.GLContext(64, 64); ctx.make_current()
to_s = lambda b: b.decode("utf-8","ignore") if b else None
print("OpenGL vendor  :", to_s(gl.glGetString(gl.GL_VENDOR)))
print("OpenGL renderer:", to_s(gl.glGetString(gl.GL_RENDERER)))
ctx.free()
print("MuJoCo version:", mujoco.__version__)
PY
'

# ---------------- 1) Main run ----------------
apptainer exec --nv "${BIND_FLAGS[@]}" --pwd "$WORKDIR_IN_CONTAINER" --env SUBMIT_DIR="$SLURM_SUBMIT_DIR" \
  "$IMG" /bin/sh -lc '
set -eu

PY="/opt/mvmae_venv/bin/python"
[ -x "$PY" ] || PY="python"

# Force -S so container .pth/site hacks can’t inject /opt/madrona_mjx/src
VENV_SITE="/opt/mvmae_venv/lib/python3.12/site-packages"

DEPS_PREFIX="${SUBMIT_DIR}/.pydeps_prefix"
PY_MM=$($PY - <<'"'"'PY'"'"'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
PREFIX_SITE="${DEPS_PREFIX}/lib/python${PY_MM}/site-packages"
BIN_DIR="${DEPS_PREFIX}/bin"
mkdir -p "$DEPS_PREFIX"

export PYTHONNOUSERSITE=1
export PYTHONPATH="/workspace:${PREFIX_SITE}:${VENV_SITE}"
export PATH="${BIN_DIR}:${PATH}"
PYRUN="$PY -S"

# Madrona caches (host-persistent)
if command -v nvidia-smi >/dev/null 2>&1; then
  ACTUAL_GPU=$(nvidia-smi -L 2>/dev/null | head -1 || true)
  echo "Actual GPU: ${ACTUAL_GPU:-<none>}"
  GPU_MODEL=$(echo "${ACTUAL_GPU:-}" | grep -o "H100\|L40S\|A100\|V100\|RTX" | head -1 || true)
  [ -n "$GPU_MODEL" ] || GPU_MODEL="unknown"
else
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

# Ensure tensorboard in persistent prefix (optional)
echo "=== Ensuring TensorBoard is available in ${DEPS_PREFIX} ==="
if $PYRUN - <<'"'"'PY'"'"'
import importlib.util
ok = importlib.util.find_spec("tensorboard") is not None
print("tensorboard already importable:", ok)
raise SystemExit(0 if ok else 1)
PY
then
  echo "TensorBoard already installed."
else
  echo "Installing tensorboard into persistent prefix..."
  $PY -m pip install --upgrade --no-cache-dir --prefix "$DEPS_PREFIX" tensorboard
fi
$PYRUN - <<'"'"'PY'"'"'
import tensorboard
print("TensorBoard version:", getattr(tensorboard, "__version__", "unknown"))
PY
echo "============================================"
echo

# Create /workspace/madrona_mjx shim that uses the SITE-PACKAGES extension
mkdir -p /workspace/madrona_mjx
cat > /workspace/madrona_mjx/__init__.py <<'"'"'PY'"'"'
from .renderer import BatchRenderer
__all__ = ["BatchRenderer"]
PY

cat > /workspace/madrona_mjx/renderer.py <<'"'"'PY'"'"'
try:
    from _madrona_mjx_batch_renderer import MadronaBatchRenderer as BatchRenderer  # type: ignore
    from _madrona_mjx_batch_renderer import MadronaBatchRenderer  # type: ignore
except Exception as e:
    raise ImportError(
        "Failed to import `_madrona_mjx_batch_renderer` from site-packages. "
        "Run `ldd` on the extension to see missing libs."
    ) from e
__all__ = ["BatchRenderer", "MadronaBatchRenderer"]
PY

# Proof + ldd check (fast)
echo "=== madrona_mjx import origin check (python -S) ==="
$PYRUN - <<'"'"'PY'"'"'
import importlib.util
print("spec madrona_mjx:", importlib.util.find_spec("madrona_mjx").origin)
print("spec _madrona_mjx_batch_renderer:", importlib.util.find_spec("_madrona_mjx_batch_renderer").origin)
PY
echo "==============================================="
echo

SO=$($PYRUN - <<'"'"'PY'"'"'
import importlib.util as u
print(u.find_spec("_madrona_mjx_batch_renderer").origin)
PY
)
echo "=== ldd on extension (first 60 lines) ==="
ldd "$SO" | sed -n "1,60p"
echo "========================================"
echo

echo "========================================="
echo "Starting MV-MAE training with MJX + Madrona"
echo "========================================="
stdbuf -oL -eL $PYRUN -u execute.py 2>&1
'

echo "Finished at $(date)"
