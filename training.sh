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

# Silence Apptainer PYTHONPATH forwarding warnings; we set it explicitly
unset PYTHONPATH || true
export APPTAINERENV_PYTHONPATH="/workspace:/opt/src:/opt/src/MV_MAE_Implementation"

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

# ============================================================
# HOST SNAPSHOT
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
echo "---- Host EGL bits ----"
echo "VENDOR_JSON=$VENDOR_JSON"
echo "NV_EGL_DIR=$NV_EGL_DIR"
echo "GLVND_DIR=$GLVND_DIR"
echo "================================================"
echo

# ============================================================
# SINGLE apptainer exec for EVERYTHING (prevents teardown hangs)
# ============================================================
echo "[HOST] launching single apptainer exec..."
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  "$IMG" \
  bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1

echo "================ CONTAINER START ================"
date
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
echo

echo "---- nvidia-smi (container) ----"
nvidia-smi -L || true
nvidia-smi || true
echo

# IMPORTANT: no MuJoCo GLContext probe here (it can hang on teardown)
echo "---- Basic GPU + imports (no EGL context) ----"
python - << "PY"
import torch, jax
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch device:", torch.cuda.get_device_name(0), "torch.version.cuda:", torch.version.cuda)
print("jax.default_backend():", jax.default_backend())
print("jax.devices():", jax.devices())
PY
echo "================================================"
echo

echo "=== MuJoCo version ==="
python - <<'"'"'PY'"'"'
import mujoco
print("MuJoCo version:", mujoco.__version__)
PY
echo "======================"
echo

# ---------------- JAX / XLA tuning ----------------
export JAX_TRACEBACK_FILTERING=off
export JAX_DISABLE_CUSOLVER=1
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_gpu_force_compilation_parallelism=1"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=.60

# Reduce toolchain parallelism (avoid toolchain deadlocks)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---------------- Determine GPU tag ----------------
ACTUAL_GPU="$(nvidia-smi -L 2>/dev/null | head -1 || true)"
GPU_MODEL="$(echo "$ACTUAL_GPU" | grep -o "H100\|L40S\|A100\|V100\|RTX" | head -1 || true)"
GPU_MODEL="${GPU_MODEL:-unknown}"
GPU_MODEL_LOWER="$(echo "$GPU_MODEL" | tr "[:upper:]" "[:lower:]")"
ENV_CONFIG="default"

# ---------------- Persistent cache root on /scratch ----------------
CACHE_BUILD_DIR="'"$SLURM_SUBMIT_DIR"'/build_${GPU_MODEL_LOWER}_${ENV_CONFIG}"
mkdir -p "$CACHE_BUILD_DIR/kernel_cache" "$CACHE_BUILD_DIR/bvh_cache"

export MADRONA_MWGPU_KERNEL_CACHE="$CACHE_BUILD_DIR/kernel_cache/kernel.cache"
export MADRONA_BVH_KERNEL_CACHE="$CACHE_BUILD_DIR/bvh_cache/bvh.cache"

# ---------------- Force ALL caches/temp to same writable area ----------------
export TMPDIR="$CACHE_BUILD_DIR/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export XDG_CACHE_HOME="$CACHE_BUILD_DIR/xdg_cache"
export CUDA_CACHE_PATH="$CACHE_BUILD_DIR/cuda_cache"
export CUDA_CACHE_MAXSIZE=$((2*1024*1024*1024))
export HOME="$CACHE_BUILD_DIR/home"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$HOME"

# Flip to 1 only if you suspect driver JIT cache locking causes the hang
export CUDA_CACHE_DISABLE=${CUDA_CACHE_DISABLE:-0}

echo "=== CACHE CONFIG ==="
echo "CACHE_BUILD_DIR=$CACHE_BUILD_DIR"
echo "MADRONA_MWGPU_KERNEL_CACHE=$MADRONA_MWGPU_KERNEL_CACHE"
echo "MADRONA_BVH_KERNEL_CACHE=$MADRONA_BVH_KERNEL_CACHE"
echo "TMPDIR=$TMPDIR"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "CUDA_CACHE_PATH=$CUDA_CACHE_PATH"
echo "CUDA_CACHE_DISABLE=$CUDA_CACHE_DISABLE"
echo "HOME=$HOME"
echo

echo "---- Versions ----"
python - <<'"'"'PY'"'"'
import jax, jaxlib, mujoco, madrona_mjx, torch
print("torch.cuda.is_available()", torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch device", torch.cuda.get_device_name(0), "torch.version.cuda", torch.version.cuda)
print("jax", jax.__version__)
print("jaxlib", jaxlib.__version__)
print("mujoco", mujoco.__version__)
print("madrona_mjx", madrona_mjx.__file__)
print("jax.devices()", jax.devices())
print("jax.default_backend()", jax.default_backend())
PY
echo

# ---------------- Runtime deps prefix (tensorboard) ----------------
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
echo

echo "========================================="
echo "Starting MV-MAE training with MJX + Madrona"
echo "========================================="
echo

# ---------------- LOCKED Madrona mini init + timeout ----------------
echo "=========== MADRONA MINI INIT (LOCKED) ==========="
LOCKFILE="$CACHE_BUILD_DIR/madrona_compile.lock"
COMPILE_TIMEOUT_SEC="${MADRONA_COMPILE_TIMEOUT_SEC:-1800}"  # 30 min

need_compile=0
if [[ ! -f "$MADRONA_MWGPU_KERNEL_CACHE" || ! -f "$MADRONA_BVH_KERNEL_CACHE" ]]; then
  need_compile=1
fi
echo "need_compile=$need_compile  lockfile=$LOCKFILE  timeout=${COMPILE_TIMEOUT_SEC}s"
echo

if [[ "$need_compile" -eq 1 ]]; then
  echo "Caches missing -> acquire lock and compile once..."
  flock -x "$LOCKFILE" bash -lc "
    set -euo pipefail
    echo '[LOCK ACQUIRED] compiling via mini init...'
    timeout ${COMPILE_TIMEOUT_SEC}s stdbuf -oL -eL python -u - <<'PY'
import faulthandler, signal, threading, time
faulthandler.enable()
faulthandler.register(signal.SIGUSR1)

def heartbeat():
    t0 = time.time()
    while True:
        dt = int(time.time() - t0)
        print(f'[heartbeat] still alive... t={dt}s', flush=True)
        time.sleep(10)

threading.Thread(target=heartbeat, daemon=True).start()

import jax, jax.numpy as jnp
import mujoco
from mujoco import mjx
from madrona_mjx.renderer import BatchRenderer

xml = \"\"\"
<mujoco>
  <worldbody>
    <geom type='plane' size='5 5 0.1' rgba='0.2 0.2 0.2 1'/>
  </worldbody>
</mujoco>
\"\"\"

m = mujoco.MjModel.from_xml_string(xml)
mx = mjx.put_model(m, impl='jax')
d = mjx.make_data(mx)

print('Creating BatchRenderer (raytracer)...', flush=True)
r = BatchRenderer(
    m=mx,
    gpu_id=0,
    num_worlds=1,
    batch_render_view_width=64,
    batch_render_view_height=64,
    enabled_geom_groups=jnp.array([0], dtype=jnp.int32),
    enabled_cameras=None,
    add_cam_debug_geo=False,
    use_rasterizer=False,
    viz_gpu_hdls=None,
)

print('Calling renderer.init...', flush=True)
tok, rgb, depth = r.init(d, mx)
jax.block_until_ready(rgb)
print('Init OK, rgb shape:', rgb.shape, 'dtype:', rgb.dtype, flush=True)
PY
    echo '[LOCK HELD] mini init completed.'
  "
else
  echo "Caches present -> skipping mini init."
fi

echo "Cache dir listing:"
ls -lah "$CACHE_BUILD_DIR/kernel_cache" "$CACHE_BUILD_DIR/bvh_cache" || true
echo "=========== MADRONA MINI INIT DONE ==========="
echo

# ---------------- Run training ----------------
echo "================ TRAINING RUN ================"
stdbuf -oL -eL python -u execute.py 2>&1
echo "Training completed."
echo "================ DONE ================"
'
echo "[HOST] apptainer exec returned."
echo "Finished at $(date)"
