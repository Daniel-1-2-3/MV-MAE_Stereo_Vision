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

# Prevent host python env from shadowing container
unset PYTHONPATH
unset PYTHONHOME

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

# ---------------- Python path inside container ----------------
export APPTAINERENV_PYTHONPATH="/workspace:/opt/src:/opt/src/MV_MAE_Implementation"

# ---- JAX backend selection (CUDA PJRT plugin) ----
export APPTAINERENV_JAX_PLATFORMS="cuda,cpu"

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
HOST_MJP_DEPS="$SLURM_SUBMIT_DIR/mujoco_playground_external_deps"
mkdir -p "$HOST_MJP_DEPS"
MJP_DEPS_IN_CONTAINER="/opt/mvmae_venv/lib/python3.12/site-packages/mujoco_playground/external_deps"

BIND_FLAGS=( --bind "$HOST_PROJECT_ROOT:$HOST_PROJECT_ROOT" )
BIND_FLAGS+=( --bind "/usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d" )
BIND_FLAGS+=( --bind "$NV_EGL_DIR:$NV_EGL_DIR" )
BIND_FLAGS+=( --bind "$GLVND_DIR:$GLVND_DIR" )
BIND_FLAGS+=( --bind "$HOST_MJP_DEPS:$MJP_DEPS_IN_CONTAINER" )
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

# ---------------- Training ----------------
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  "$IMG" \
  bash -lc '
set -euo pipefail
. /opt/mvmae_venv/bin/activate

# --- Force CUDA runtime libs to win over any injected host libs ---
export CUDA_HOME=/usr/local/cuda
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda ${XLA_FLAGS:-}"

# Put CUDA libs first (nvJitLink lives here)
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

# Also ensure the dynamic linker cache sees these first in-process
export LD_PRELOAD="/usr/local/cuda/lib64/libnvJitLink.so.12:${LD_PRELOAD:-}"

echo "=== MuJoCo version ==="
python - <<'"'"'PY'"'"'
import mujoco
print("MuJoCo version:", mujoco.__version__)
PY
echo "======================"

export PYTHONUNBUFFERED=1
export JAX_TRACEBACK_FILTERING=off
export JAX_DISABLE_CUSOLVER=1
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=.60
export PICK_ENV_DEBUG=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"

echo "=== Madrona + GPU detection (inside container) ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  ACTUAL_GPU=$(nvidia-smi -L 2>/dev/null | head -1)
  echo "Actual GPU: $ACTUAL_GPU"
  GPU_MODEL=$(echo "$ACTUAL_GPU" | grep -o "H100\|L40S\|A100\|V100\|RTX" | head -1 || true)
  [[ -z "${GPU_MODEL:-}" ]] && GPU_MODEL="unknown"
else
  GPU_MODEL="unknown"
fi
GPU_MODEL_LOWER=$(echo "$GPU_MODEL" | tr "[:upper:]" "[:lower:]")
ENV_CONFIG="default"

CACHE_BUILD_DIR="'"$SLURM_SUBMIT_DIR"'/build_${GPU_MODEL_LOWER}_${ENV_CONFIG}"
mkdir -p "$CACHE_BUILD_DIR/kernel_cache" "$CACHE_BUILD_DIR/bvh_cache"
export MADRONA_MWGPU_KERNEL_CACHE="$CACHE_BUILD_DIR/kernel_cache/kernel.cache"
export MADRONA_BVH_KERNEL_CACHE="$CACHE_BUILD_DIR/bvh_cache/bvh.cache"

DEPS_PREFIX="'"$SLURM_SUBMIT_DIR"'/.pydeps_prefix"
PY_MM=$(python - <<'"'"'PY'"'"'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
SITE_PKGS="${DEPS_PREFIX}/lib/python${PY_MM}/site-packages"
BIN_DIR="${DEPS_PREFIX}/bin"
mkdir -p "$DEPS_PREFIX"

# Put Madrona build FIRST so top-level _madrona_* resolves to the .so
export PYTHONPATH="/opt/madrona_mjx/build:/workspace:${SITE_PKGS}:${PYTHONPATH:-}"
export PATH="${BIN_DIR}:${PATH}"
export LD_LIBRARY_PATH="/opt/madrona_mjx/build:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

echo "=== nvJitLink resolution ==="
ldd /opt/madrona_mjx/build/libmadmjx_mgr.so | grep -i nvjitlink || true
ldconfig -p | grep -i nvjitlink || true
echo "==========================="

echo "=== nvJitLink symbol check (the one Madrona needs) ==="
python - <<'PY'
import ctypes
p = "/usr/local/cuda/lib64/libnvJitLink.so.12"
lib = ctypes.CDLL(p)
print("loaded:", p)
print("has __nvJitLinkCreate_12_5:", hasattr(lib, "__nvJitLinkCreate_12_5"))
PY
echo "====================================================="

# Hard delete any shadowing python files from /workspace (host bind)
rm -f /workspace/_madrona_mjx_batch_renderer.py /workspace/_madrona_mjx_visualizer.py
rm -rf /workspace/__pycache__ || true

echo "=== Verify top-level Madrona modules resolve to .so (no nanobind double-load) ==="
python - <<'"'"'PY'"'"'
import _madrona_mjx_batch_renderer as br
import _madrona_mjx_visualizer as vz
print("_madrona_mjx_batch_renderer:", br.__file__)
print("_madrona_mjx_visualizer   :", vz.__file__)
if br.__file__.endswith(".py") or vz.__file__.endswith(".py"):
    raise SystemExit("FATAL: _madrona_mjx_* resolved to .py; shadowing still present")
print("[ok] _madrona_mjx_* resolved to compiled .so")
PY
echo "==============================================================="

# ---------------- NEW: GPU sanity check (Torch + JAX + Madrona init) ----------------
echo "=== GPU sanity check (Torch + JAX + Madrona) ==="
python - <<'"'"'PY'"'"'
import os, sys

# ---- Torch CUDA smoke ----
import torch
print("[torch] version:", torch.__version__)
print("[torch] cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("[torch] FATAL: torch.cuda.is_available() is False")
print("[torch] gpu:", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("[torch] matmul ok")

# ---- JAX CUDA smoke ----
os.environ["JAX_PLATFORMS"] = os.environ.get("JAX_PLATFORMS", "cuda,cpu")
import jax, jax.numpy as jnp
print("[jax] version:", jax.__version__)
print("[jax] devices:", jax.devices())
if not any(d.platform in ("cuda", "gpu") for d in jax.devices()):
    raise SystemExit("[jax] FATAL: no CUDA/GPU device visible to JAX")
a = jnp.ones((1024, 1024), dtype=jnp.float32)
b = (a @ a).block_until_ready()
dev_attr = getattr(b, "device", None)
dev = dev_attr() if callable(dev_attr) else dev_attr
print("[jax] matmul ok on:", dev)

# ---- Madrona init smoke (catches the custom-call / cuModuleGetFunction failure early) ----
# This does NOT run your env; it only tries to import and initialize renderer pieces.
import mujoco
from madrona_mjx.renderer import BatchRenderer

print("[madrona] MADRONA_MWGPU_KERNEL_CACHE:", os.environ.get("MADRONA_MWGPU_KERNEL_CACHE"))
print("[madrona] MADRONA_BVH_KERNEL_CACHE  :", os.environ.get("MADRONA_BVH_KERNEL_CACHE"))

# Minimal tiny model: use mujoco’s built-in small XML if available; otherwise a 1-body trivial MJCF.
xml = """
<mujoco>
  <worldbody>
    <body name="b" pos="0 0 0">
      <geom type="sphere" size="0.05" rgba="0.7 0.2 0.2 1"/>
      <camera name="cam" pos="0 0 0.5" quat="1 0 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(xml)

# One world, one cam, tiny render
renderer = BatchRenderer(
    m,
    gpu_id=0,
    num_worlds=1,
    batch_render_view_width=8,
    batch_render_view_height=8,
    use_rasterizer=False,  # raytracer path
)
echo "==============================================="

# Run training
stdbuf -oL -eL python -u execute.py 2>&1
echo "Training completed."
'

echo "Finished at $(date)"
