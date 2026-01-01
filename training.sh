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

# ---------------- Basics ----------------
unset PYTHONPATH PYTHONHOME

IMG="$SLURM_SUBMIT_DIR/training.sif"
[[ -f "$IMG" ]] || { echo "ERROR: $IMG not found"; exit 2; }

HOST_PROJECT_ROOT="$SLURM_SUBMIT_DIR"
WORKDIR_IN_CONTAINER="/workspace"
[[ -f "$HOST_PROJECT_ROOT/execute.py" ]] || { echo "FATAL: execute.py not found at $HOST_PROJECT_ROOT/execute.py"; exit 10; }

# ---------------- Container env ----------------
export APPTAINERENV_PYTHONPATH="/workspace:/opt/src:/opt/src/MV_MAE_Implementation"
export APPTAINERENV_JAX_PLATFORMS="cuda,cpu"
export APPTAINERENV_SUBMIT_DIR="$SLURM_SUBMIT_DIR"

# EGL / MuJoCo GL
export APPTAINERENV_MUJOCO_GL=egl
export APPTAINERENV_PYOPENGL_PLATFORM=egl
export APPTAINERENV_MUJOCO_PLATFORM=egl
export APPTAINERENV_DISPLAY=
export APPTAINERENV_LIBGL_ALWAYS_SOFTWARE=0
export APPTAINERENV_MESA_LOADER_DRIVER_OVERRIDE=
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export APPTAINERENV_IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg

# NVIDIA EGL vendor JSON (HOST)
VENDOR_JSON="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
[[ -f "$VENDOR_JSON" ]] || { echo "FATAL: $VENDOR_JSON not found on host (NVIDIA EGL ICD missing)."; exit 3; }
export APPTAINERENV__EGL_VENDOR_LIBRARY_FILENAMES="$VENDOR_JSON"

# Locate libEGL_nvidia.so on HOST
NV_EGL_DIR="$(ldconfig -p | awk '/libEGL_nvidia\.so/{print $NF; exit}' | xargs -r dirname || true)"
for d in /usr/lib/x86_64-linux-gnu/nvidia /usr/lib/nvidia /usr/lib64/nvidia /usr/lib/x86_64-linux-gnu; do
  [[ -z "${NV_EGL_DIR:-}" && -e "$d/libEGL_nvidia.so.0" ]] && NV_EGL_DIR="$d"
done
[[ -n "${NV_EGL_DIR:-}" && -d "$NV_EGL_DIR" ]] || { echo "FATAL: Could not find libEGL_nvidia.so* on host."; exit 4; }

GLVND_DIR="/usr/lib/x86_64-linux-gnu"
[[ -e "$GLVND_DIR/libEGL.so.1" ]] || GLVND_DIR="/usr/lib64"

# ---------------- Binds ----------------
HOST_MJP_DEPS="$SLURM_SUBMIT_DIR/mujoco_playground_external_deps"
mkdir -p "$HOST_MJP_DEPS"
MJP_DEPS_IN_CONTAINER="/opt/mvmae_venv/lib/python3.12/site-packages/mujoco_playground/external_deps"

BIND_FLAGS=(
  --bind "$HOST_PROJECT_ROOT:$HOST_PROJECT_ROOT"
  --bind "$HOST_PROJECT_ROOT:$WORKDIR_IN_CONTAINER"
  --bind "/usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d"
  --bind "$NV_EGL_DIR:$NV_EGL_DIR"
  --bind "$GLVND_DIR:$GLVND_DIR"
  --bind "$HOST_MJP_DEPS:$MJP_DEPS_IN_CONTAINER"
)

# ---------------- Quick EGL + GPU probe ----------------
PROBE_SCRIPT=$(cat <<'EOS'
python - <<'PY'
import torch, mujoco
import OpenGL.GL as gl
print("torch cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
ctx = mujoco.GLContext(64, 64); ctx.make_current()
to_s = lambda b: b.decode("utf-8","ignore") if b else None
print("OpenGL vendor  :", to_s(gl.glGetString(gl.GL_VENDOR)))
print("OpenGL renderer:", to_s(gl.glGetString(gl.GL_RENDERER)))
ctx.free()
PY
EOS
)

apptainer exec --nv "${BIND_FLAGS[@]}" --pwd "$WORKDIR_IN_CONTAINER" "$IMG" bash -lc "$PROBE_SCRIPT"

# ---------------- Training ----------------
RUN_SCRIPT=$(cat <<'EOS'
set -euo pipefail
. /opt/mvmae_venv/bin/activate

export PYTHONUNBUFFERED=1
export JAX_TRACEBACK_FILTERING=off
export JAX_DISABLE_CUSOLVER=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=.60
export PICK_ENV_DEBUG=1

# Force CUDA runtime libs
export CUDA_HOME=/usr/local/cuda
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda ${XLA_FLAGS:-}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/usr/local/cuda/lib64/libnvJitLink.so.12:${LD_PRELOAD:-}"

echo "=== MuJoCo version ==="
python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"
echo "======================"

# Prefer CUDA, fall back to CPU
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"

echo "=== Madrona + GPU detection (inside container) ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  ACTUAL_GPU="$(nvidia-smi -L 2>/dev/null | head -1 || true)"
  echo "Actual GPU: ${ACTUAL_GPU:-unknown}"
  GPU_MODEL="$(echo "${ACTUAL_GPU:-}" | grep -o "H100\|L40S\|A100\|V100\|RTX" | head -1 || true)"
  [[ -z "${GPU_MODEL:-}" ]] && GPU_MODEL="unknown"
else
  GPU_MODEL="unknown"
fi
GPU_MODEL_LOWER="$(echo "$GPU_MODEL" | tr '[:upper:]' '[:lower:]')"
ENV_CONFIG="default"

CACHE_BUILD_DIR="${SUBMIT_DIR}/build_${GPU_MODEL_LOWER}_${ENV_CONFIG}"
mkdir -p "${CACHE_BUILD_DIR}/kernel_cache" "${CACHE_BUILD_DIR}/bvh_cache"
export MADRONA_MWGPU_KERNEL_CACHE="${CACHE_BUILD_DIR}/kernel_cache/kernel.cache"
export MADRONA_BVH_KERNEL_CACHE="${CACHE_BUILD_DIR}/bvh_cache/bvh.cache"

# Put Madrona build FIRST so top-level _madrona_* resolves to the .so
DEPS_PREFIX="${SUBMIT_DIR}/.pydeps_prefix"
PY_MM="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
SITE_PKGS="${DEPS_PREFIX}/lib/python${PY_MM}/site-packages"
BIN_DIR="${DEPS_PREFIX}/bin"
mkdir -p "$DEPS_PREFIX"

export PYTHONPATH="/opt/madrona_mjx/build:/workspace:${SITE_PKGS}:${PYTHONPATH:-}"
export PATH="${BIN_DIR}:${PATH}"
export LD_LIBRARY_PATH="/opt/madrona_mjx/build:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

echo "=== nvJitLink resolution ==="
ldd /opt/madrona_mjx/build/libmadmjx_mgr.so | grep -i nvjitlink || true
ldconfig -p | grep -i nvjitlink || true
echo "==========================="

echo "=== nvJitLink symbol check ==="
python - <<'PY'
import ctypes
p="/usr/local/cuda/lib64/libnvJitLink.so.12"
lib=ctypes.CDLL(p)
print("loaded:", p)
print("has __nvJitLinkCreate_12_5:", hasattr(lib, "__nvJitLinkCreate_12_5"))
PY
echo "============================="

# Remove any shadowing python files from /workspace (host bind)
rm -f /workspace/_madrona_mjx_batch_renderer.py /workspace/_madrona_mjx_visualizer.py
rm -rf /workspace/__pycache__ || true

echo "=== Verify top-level Madrona modules resolve to .so ==="
python - <<'PY'
import _madrona_mjx_batch_renderer as br
import _madrona_mjx_visualizer as vz
print("_madrona_mjx_batch_renderer:", br.__file__)
print("_madrona_mjx_visualizer   :", vz.__file__)
if br.__file__.endswith(".py") or vz.__file__.endswith(".py"):
    raise SystemExit("FATAL: _madrona_mjx_* resolved to .py; shadowing still present")
print("[ok] _madrona_mjx_* resolved to compiled .so")
PY
echo "======================================================"

echo "=== GPU sanity check (Torch + JAX + Madrona import) ==="
python - <<'PY'
import os
import torch
print("[torch] version:", torch.__version__)
print("[torch] cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("[torch] FATAL: torch.cuda.is_available() is False")
print("[torch] gpu:", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda")
(y := x @ x)
torch.cuda.synchronize()
print("[torch] matmul ok")

os.environ["JAX_PLATFORMS"] = os.environ.get("JAX_PLATFORMS", "cuda,cpu")
import jax, jax.numpy as jnp
print("[jax] version:", jax.__version__)
print("[jax] devices:", jax.devices())
if not any(d.platform in ("cuda","gpu") for d in jax.devices()):
    raise SystemExit("[jax] FATAL: no CUDA/GPU device visible to JAX")
a = jnp.ones((1024,1024), dtype=jnp.float32)
b = (a @ a).block_until_ready()
dev_attr = getattr(b, "device", None)
dev = dev_attr() if callable(dev_attr) else dev_attr
print("[jax] matmul ok on:", dev)

from madrona_mjx.renderer import BatchRenderer
print("[madrona] imported BatchRenderer ok")
PY
echo "======================================================="

echo "=== BatchRenderer signature probe ==="
python -c "import inspect; from madrona_mjx.renderer import BatchRenderer; \
print('BatchRenderer.init :', inspect.signature(BatchRenderer.init)); \
print('BatchRenderer.render:', inspect.signature(BatchRenderer.render))"
echo "===================================="

# Run training
stdbuf -oL -eL python -u execute.py 2>&1
echo "Training completed."
EOS
)

apptainer exec --nv "${BIND_FLAGS[@]}" --pwd "$WORKDIR_IN_CONTAINER" "$IMG" bash -lc "$RUN_SCRIPT"

echo "Finished at $(date)"
