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

# Prefer host-mounted code under /workspace so edits require no SIF rebuild.
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

# Critical bind: mount the entire project to /workspace
BIND_FLAGS+=( --bind "$HOST_PROJECT_ROOT:$WORKDIR_IN_CONTAINER" )

# ============================================================
# 0) HOST-SIDE SNAPSHOT (proves which node/driver you landed on)
# ============================================================
echo "================ HOST SNAPSHOT ================"
date
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_NODELIST=$SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo

echo "---- nvidia-smi (host) ----"
nvidia-smi -L || true
nvidia-smi || true
echo

echo "---- Host GL / EGL vendor json ----"
echo "VENDOR_JSON=$VENDOR_JSON"
echo "NV_EGL_DIR=$NV_EGL_DIR"
echo "GLVND_DIR=$GLVND_DIR"
echo "================================================"
echo

# ============================================================
# 1) QUICK CONTAINER PROBE (GPU + EGL + basic imports)
# ============================================================
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  "$IMG" \
  bash -lc '
set -euo pipefail

echo "================ CONTAINER PROBE ================"
date
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
echo

echo "---- nvidia-smi (container) ----"
nvidia-smi -L || true
nvidia-smi || true
echo

echo "---- glibc / gcc ----"
ldd --version | head -n 2 || true
which g++ || true
g++ --version | head -n 2 || true
echo

echo "---- Python + key packages ----"
python -V
python - << "PY"
import sys, platform
print("python:", sys.version)
print("platform:", platform.platform())

def show(mod):
    try:
        m = __import__(mod)
        print(f"{mod}: OK  version={getattr(m, '__version__', 'unknown')}  file={getattr(m, '__file__', None)}")
        return m
    except Exception as e:
        print(f"{mod}: FAIL {e!r}")
        return None

torch = show("torch")
jax   = show("jax")
jaxlib= show("jaxlib")
muj   = show("mujoco")
mm    = show("madrona_mjx")

if torch and torch.cuda.is_available():
    print("torch.cuda:", True, torch.cuda.get_device_name(0))
else:
    print("torch.cuda:", False)

if jax:
    try:
        print("jax.devices():", jax.devices())
        print("jax.default_backend():", jax.default_backend())
    except Exception as e:
        print("jax backend query FAIL:", e)

# Show where XLA extension comes from (helps spot wrong wheel)
if jaxlib:
    try:
        from jaxlib import xla_extension
        print("jaxlib.xla_extension:", xla_extension.__file__)
    except Exception as e:
        print("xla_extension FAIL:", e)
PY
echo

echo "---- EGL context sanity (MuJoCo + OpenGL) ----"
python - << "PY"
import mujoco
import OpenGL.GL as gl
ctx = mujoco.GLContext(64, 64)
ctx.make_current()
to_s = lambda b: b.decode("utf-8","ignore") if b else None
print("OpenGL vendor  :", to_s(gl.glGetString(gl.GL_VENDOR)))
print("OpenGL renderer:", to_s(gl.glGetString(gl.GL_RENDERER)))
ctx.free()
PY
echo "================================================"
'

# ============================================================
# 2) DEEP DIVE: SHARED LIB / ABI CHECKS FOR madrona_mjx + JAX
#    This is the #1 way to catch "scary cuda errors" culprits:
#    - missing libs (NOT FOUND)
#    - libstdc++ / GLIBCXX mismatches
#    - linking against unexpected libcuda/libcudart
# ============================================================
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  "$IMG" \
  bash -lc '
set -euo pipefail

echo "================ ABI / LDD CHECKS ================"
MADDIR="$(python - <<'"'"'PY'"'"'
import os, madrona_mjx
print(os.path.dirname(madrona_mjx.__file__))
PY
)"
echo "madrona_mjx dir: $MADDIR"
echo

echo "---- List madrona_mjx *.so ----"
ls -lah "$MADDIR" || true
find "$MADDIR" -maxdepth 2 -name "*.so" -print || true
echo

echo "---- ldd on madrona_mjx shared libs (look for NOT FOUND) ----"
shopt -s nullglob
for so in "$MADDIR"/*.so "$MADDIR"/*/*.so; do
  echo "----- $so -----"
  ldd "$so" | egrep -i "not found|libcuda|libcudart|libnvrtc|libnvidia|libstdc\+\+|libgcc_s|GLIBC|GLIBCXX|CXXABI" || true
done
echo

echo "---- ldd on jaxlib xla_extension ----"
python - <<'"'"'PY'"'"'
from jaxlib import xla_extension
print(xla_extension.__file__)
PY
XLA_SO="$(python - <<'"'"'PY'"'"'
from jaxlib import xla_extension
print(xla_extension.__file__)
PY
)"
echo "xla_extension: $XLA_SO"
ldd "$XLA_SO" | egrep -i "not found|libcuda|libcudart|libnvrtc|libnvidia|libstdc\+\+|libgcc_s|GLIBC|GLIBCXX|CXXABI" || true
echo

echo "---- CUDA library resolution (where are libs coming from?) ----"
python - <<'"'"'PY'"'"'
import ctypes.util
for name in ["cuda", "cudart", "nvrtc"]:
    print(name, "->", ctypes.util.find_library(name))
PY
echo "=================================================="
'

# ============================================================
# 3) MADRONA CACHE SETUP + JAX FLAGS + MINIMAL REPRO
#    This tries to fail FAST at the exact culprit:
#    - first env reset
#    - renderer.init
#    - one render call
# ============================================================
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  "$IMG" \
  bash -lc '
set -euo pipefail

echo "================ MADRONA / JAX SETUP ================"
export PYTHONUNBUFFERED=1

# Make JAX print real stack traces
export JAX_TRACEBACK_FILTERING=off

# Reduce async confusion during debugging:
# (set to 1 only while diagnosing; it will slow you down)
export CUDA_LAUNCH_BLOCKING=1

# Optional allocator tweaks
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=.60

# Keep your cuda data dir flag if needed
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

echo "XLA_FLAGS=$XLA_FLAGS"
echo "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo

echo "---- GPU identity ----"
nvidia-smi -L || true
echo

# ---------------- Madrona cache integration ----------------
echo "---- Madrona cache selection ----"
ACTUAL_GPU="$(nvidia-smi -L 2>/dev/null | head -1 || true)"
echo "Actual GPU: $ACTUAL_GPU"
GPU_MODEL="$(echo "$ACTUAL_GPU" | grep -o "H100\|L40S\|A100\|V100\|RTX" | head -1 || true)"
GPU_MODEL="${GPU_MODEL:-unknown}"
GPU_MODEL_LOWER="$(echo "$GPU_MODEL" | tr "[:upper:]" "[:lower:]")"
ENV_CONFIG="default"

CACHE_BUILD_DIR="'"$SLURM_SUBMIT_DIR"'/build_${GPU_MODEL_LOWER}_${ENV_CONFIG}"
mkdir -p "$CACHE_BUILD_DIR/kernel_cache" "$CACHE_BUILD_DIR/bvh_cache"
export MADRONA_MWGPU_KERNEL_CACHE="$CACHE_BUILD_DIR/kernel_cache/kernel.cache"
export MADRONA_BVH_KERNEL_CACHE="$CACHE_BUILD_DIR/bvh_cache/bvh.cache"

echo "MADRONA_MWGPU_KERNEL_CACHE=$MADRONA_MWGPU_KERNEL_CACHE"
echo "MADRONA_BVH_KERNEL_CACHE=$MADRONA_BVH_KERNEL_CACHE"
ls -lah "$CACHE_BUILD_DIR" || true
echo

echo "---- Versions (inside container) ----"
python - <<'"'"'PY'"'"'
import jax, jaxlib, mujoco, madrona_mjx
print("jax", jax.__version__)
print("jaxlib", jaxlib.__version__)
print("mujoco", mujoco.__version__)
print("madrona_mjx", madrona_mjx.__file__)
print("jax.devices()", jax.devices())
print("jax.default_backend()", jax.default_backend())
PY
echo "======================================================"
echo

# ============================================================
# 3a) MINIMAL SMOKE TEST: import env + do one reset
#     If this fails, it isolates to env reset / renderer.init.
# ============================================================
echo "================ MINIMAL ENV SMOKE TEST ================"
python - <<'"'"'PY'"'"'
import os
import jax
import jax.numpy as jnp

# Make failures happen where they occur (not later)
from jax import config
config.update("jax_debug_nans", False)

# Import your env
from Mujoco_Sim.pick_env import StereoPickCube

# IMPORTANT: match your training batch size here (or start with 1 then 128)
B = int(os.environ.get("SMOKE_B", "128"))

env = StereoPickCube(config_overrides={
    "vision": True,
    "vision_config.render_batch_size": B,
    # keep your resolution the same as training
    "vision_config.render_width": 64,
    "vision_config.render_height": 64,
    "vision_config.use_rasterizer": False,
})

key = jax.random.PRNGKey(0)
keys = jax.random.split(key, B)

print("Resetting...")
st = env.reset(keys)

# Force sync immediately so errors point to the real op
jax.block_until_ready(st.obs)

print("Reset OK. obs shape:", getattr(st.obs, "shape", None))
PY
echo "========================================================"
echo

# ============================================================
# 4) RUN TRAINING
# ============================================================
echo "================ TRAINING RUN ================"

# Runtime deps prefix (persist on host)
DEPS_PREFIX="'"$SLURM_SUBMIT_DIR"'/.pydeps_prefix"
PY_MM=$(python - <<'"'"'PY'"'"'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
SITE_PKGS="${DEPS_PREFIX}/lib/python${PY_MM}/site-packages"
BIN_DIR="${DEPS_PREFIX}/bin"
mkdir -p "$DEPS_PREFIX"
export PYTHONPATH="/workspace:${SITE_PKGS}:${PYTHONPATH:-}"
export PATH="${BIN_DIR}:${PATH}"

echo "=== Ensuring TensorBoard in ${DEPS_PREFIX} ==="
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

echo "---- Final version echo ----"
python -c "import jax, jaxlib; print('jax', jax.__version__); print('jaxlib', jaxlib.__version__)"
python -c "import madrona_mjx; print('madrona_mjx', madrona_mjx.__file__)"
echo

stdbuf -oL -eL python -u execute.py 2>&1

echo "Training completed."
'

echo "Finished at $(date)"
