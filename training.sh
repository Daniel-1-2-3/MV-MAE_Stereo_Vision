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

# Avoid interactive module selection
module load StdEnv/2023 apptainer/1.3.5 || module load apptainer

unset PYTHONPATH
unset PYTHONHOME

IMG="$SLURM_SUBMIT_DIR/training.sif"
[[ -f "$IMG" ]] || { echo "ERROR: $IMG not found"; exit 2; }

HOST_PROJECT_ROOT="$SLURM_SUBMIT_DIR"
WORKDIR_IN_CONTAINER="/workspace"
[[ -f "$HOST_PROJECT_ROOT/execute.py" ]] || { echo "FATAL: execute.py not found"; exit 10; }

# Container env
export APPTAINERENV_PYTHONPATH="/workspace:/opt/src:/opt/src/MV_MAE_Implementation"
export APPTAINERENV_JAX_PLATFORMS="cuda,cpu"

# EGL / MuJoCo
export APPTAINERENV_MUJOCO_GL=egl
export APPTAINERENV_PYOPENGL_PLATFORM=egl
export APPTAINERENV_MUJOCO_PLATFORM=egl
export APPTAINERENV_DISPLAY=
export APPTAINERENV_LIBGL_ALWAYS_SOFTWARE=0
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export APPTAINERENV_IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg

# Host EGL vendor json
VENDOR_JSON="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
[[ -f "$VENDOR_JSON" ]] || { echo "FATAL: $VENDOR_JSON missing on host"; exit 3; }
export APPTAINERENV__EGL_VENDOR_LIBRARY_FILENAMES="$VENDOR_JSON"

# Locate host NVIDIA EGL lib dir
NV_EGL_DIR="$(ldconfig -p | awk '/libEGL_nvidia\.so/{print $NF; exit}' | xargs -r dirname || true)"
for d in /usr/lib/x86_64-linux-gnu/nvidia /usr/lib/nvidia /usr/lib64/nvidia /usr/lib/x86_64-linux-gnu; do
  [[ -z "${NV_EGL_DIR:-}" && -e "$d/libEGL_nvidia.so.0" ]] && NV_EGL_DIR="$d"
done
[[ -n "${NV_EGL_DIR:-}" && -d "$NV_EGL_DIR" ]] || { echo "FATAL: libEGL_nvidia.so* not found on host"; exit 4; }

GLVND_DIR="/usr/lib/x86_64-linux-gnu"
[[ -e "$GLVND_DIR/libEGL.so.1" ]] || GLVND_DIR="/usr/lib64"

# Mujoco-playground external deps bind
HOST_MJP_DEPS="$SLURM_SUBMIT_DIR/mujoco_playground_external_deps"
mkdir -p "$HOST_MJP_DEPS"
MJP_DEPS_IN_CONTAINER="/opt/mvmae_venv/lib/python3.12/site-packages/mujoco_playground/external_deps"

BIND_FLAGS=( --bind "$HOST_PROJECT_ROOT:$HOST_PROJECT_ROOT" )
BIND_FLAGS+=( --bind "$HOST_PROJECT_ROOT:$WORKDIR_IN_CONTAINER" )
BIND_FLAGS+=( --bind "/usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d" )
BIND_FLAGS+=( --bind "$NV_EGL_DIR:$NV_EGL_DIR" )
BIND_FLAGS+=( --bind "$GLVND_DIR:$GLVND_DIR" )
BIND_FLAGS+=( --bind "$HOST_MJP_DEPS:$MJP_DEPS_IN_CONTAINER" )

echo "=== [HOST] job context ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "=========================="

# Quick EGL probe
apptainer exec --nv "${BIND_FLAGS[@]}" --pwd "$WORKDIR_IN_CONTAINER" "$IMG" bash -lc '
set -euo pipefail
. /opt/mvmae_venv/bin/activate
echo "=== [CONTAINER] EGL + Torch probe ==="
python - << "PY"
import os, torch, mujoco, OpenGL.GL as gl
print("torch:", torch.__version__)
print("torch cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("MUJOCO_GL:", os.environ.get("MUJOCO_GL"))
ctx = mujoco.GLContext(64, 64); ctx.make_current()
to_s = lambda b: b.decode("utf-8","ignore") if b else None
print("OpenGL vendor  :", to_s(gl.glGetString(gl.GL_VENDOR)))
print("OpenGL renderer:", to_s(gl.glGetString(gl.GL_RENDERER)))
ctx.free()
PY
echo "====================================="
'

# Training
apptainer exec --nv "${BIND_FLAGS[@]}" --pwd "$WORKDIR_IN_CONTAINER" "$IMG" bash -lc '
set -euo pipefail
. /opt/mvmae_venv/bin/activate

# Force CUDA 12.5 nvJitLink (fixes undefined symbol __nvJitLinkCreate_12_5)
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/opt/madrona_mjx/build:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/usr/local/cuda/lib64/libnvJitLink.so.12:${LD_PRELOAD:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

echo "=== [CONTAINER] preflight env snapshot ==="
echo "PATH=$PATH"
echo "PYTHONPATH=${PYTHONPATH:-}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
echo "JAX_PLATFORMS=${JAX_PLATFORMS:-}"
echo "XLA_FLAGS=${XLA_FLAGS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "=========================================="

# Keep CUDA find paths sane
export CUDA_HOME=/usr/local/cuda
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=.60
export PYTHONUNBUFFERED=1
export JAX_TRACEBACK_FILTERING=off
export PICK_ENV_DEBUG=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"

# Caches
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== [CONTAINER] nvidia-smi ==="
  nvidia-smi -L || true
  nvidia-smi || true
  echo "============================="
  ACTUAL_GPU=$(nvidia-smi -L 2>/dev/null | head -1 || true)
else
  ACTUAL_GPU=""
fi
GPU_MODEL=$(echo "$ACTUAL_GPU" | grep -o "H100\|L40S\|A100\|V100\|RTX" | head -1 || true)
[[ -z "${GPU_MODEL:-}" ]] && GPU_MODEL="unknown"
GPU_MODEL_LOWER=$(echo "$GPU_MODEL" | tr "[:upper:]" "[:lower:]")
CACHE_BUILD_DIR="'"$SLURM_SUBMIT_DIR"'/build_${GPU_MODEL_LOWER}_default"
mkdir -p "$CACHE_BUILD_DIR/kernel_cache" "$CACHE_BUILD_DIR/bvh_cache"
export MADRONA_MWGPU_KERNEL_CACHE="$CACHE_BUILD_DIR/kernel_cache/kernel.cache"
export MADRONA_BVH_KERNEL_CACHE="$CACHE_BUILD_DIR/bvh_cache/bvh.cache"
echo "MADRONA_MWGPU_KERNEL_CACHE=$MADRONA_MWGPU_KERNEL_CACHE"
echo "MADRONA_BVH_KERNEL_CACHE=$MADRONA_BVH_KERNEL_CACHE"

# IMPORTANT: do NOT put /opt/madrona_mjx/build on PYTHONPATH
# (prevents top-level _madrona_* imports & nanobind double-load).
# Rely on editable install (madrona_mjx package) + ldconfig for libs.
export LD_LIBRARY_PATH="/opt/madrona_mjx/build:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# Kill any accidental shadowing python files from /workspace
rm -f /workspace/_madrona_mjx_batch_renderer.py /workspace/_madrona_mjx_visualizer.py
rm -rf /workspace/__pycache__ || true

echo "=== [CONTAINER] version inventory ==="
python - <<'"'"'PY'"'"'
import sys, pkgutil
import jax, jaxlib
import mujoco
print("python:", sys.version)
print("mujoco:", mujoco.__version__)
print("jax:", jax.__version__)
print("jaxlib:", jaxlib.__version__)
PY
echo "--- pip (filtered) ---"
python -m pip list | egrep -i "jax|cuda12|pjrt|plugin|mujoco|nanobind|madrona" || true
echo "==============================="

echo "=== [CONTAINER] CUDA lib resolution (ctypes + ldd) ==="
python - <<'"'"'PY'"'"'
import os, ctypes.util
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH",""))
for name in ["cuda","cudart","nvJitLink","cublas","cudnn"]:
    print(name, "->", ctypes.util.find_library(name))
PY
echo "--- ldd madrona mgr ---"
ldd /opt/madrona_mjx/build/libmadmjx_mgr.so | egrep -i "cuda|cudart|nvjitlink|cublas|cudnn|stdc\+\+|gcc_s" || true
echo "==============================================="

echo "=== [CONTAINER] JAX GPU plugin presence + devices ==="
python - <<'"'"'PY'"'"'
import importlib.util, jax
print("devices:", jax.devices())
print("spec jax_plugins.xla_cuda12:", importlib.util.find_spec("jax_plugins.xla_cuda12"))
print("spec jax_cuda12_pjrt       :", importlib.util.find_spec("jax_cuda12_pjrt"))
print("spec jax_cuda12_plugin     :", importlib.util.find_spec("jax_cuda12_plugin"))
PY
echo "==============================================="

echo "=== [CONTAINER] JAX matmul smoke ==="
python - <<'"'"'PY'"'"'
import jax, jax.numpy as jnp
x = jnp.ones((1024,1024), dtype=jnp.float32)
y = (x @ x).block_until_ready()
dev_attr = getattr(y, "device", None)
dev = dev_attr() if callable(dev_attr) else dev_attr
print("matmul ok on:", dev)
PY
echo "==================================="

echo "=== [CONTAINER] Madrona import path check (package-only) ==="
python - <<'"'"'PY'"'"'
import importlib.util, madrona_mjx, madrona_mjx.renderer
spec_br = importlib.util.find_spec("madrona_mjx._madrona_mjx_batch_renderer")
spec_vz = importlib.util.find_spec("madrona_mjx._madrona_mjx_visualizer")
print("madrona_mjx pkg:", madrona_mjx.__file__)
print("madrona_mjx.renderer:", madrona_mjx.renderer.__file__)
print("madrona_mjx._madrona_mjx_batch_renderer:", None if spec_br is None else spec_br.origin)
print("madrona_mjx._madrona_mjx_visualizer   :", None if spec_vz is None else spec_vz.origin)
PY
echo "==========================================================="

echo "=== [CONTAINER] starting training ==="
stdbuf -oL -eL python -u execute.py 2>&1
'

echo "Finished at $(date)"
