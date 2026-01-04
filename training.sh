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

# Prevent host python module leakage into apptainer env forwarding
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
# Keep your logic, but avoid inheriting host PYTHONPATH that triggers apptainer warnings
export APPTAINERENV_PYTHONPATH="/workspace:/opt/src:/opt/src/MV_MAE_Implementation"
export APPTAINERENV_PYTHONHOME=

# ---------------- EGL / MuJoCo GL setup ----------------
export APPTAINERENV_MUJOCO_GL=egl
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_JAX_LOG_COMPILES=0

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

# ---------------- Hang-focused runtime knobs ----------------
# 1) Put caches on local disk (network FS stalls are common)
export XDG_CACHE_HOME="/tmp/${SLURM_JOB_ID}_xdg_cache"
export CUDA_CACHE_PATH="/tmp/${SLURM_JOB_ID}_cuda_cache"
export CUDA_CACHE_MAXSIZE=2147483648
mkdir -p "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"

# 2) Make CUDA module loads deterministic; LAZY can look like "hang at first call"
export CUDA_MODULE_LOADING=EAGER

# 3) Faulthandler: periodic Python stack dumps (shows where it is stuck)
export PYTHONFAULTHANDLER=1
export FAULTHANDLER_TIMEOUT=60

# JAX / XLA tuning (keep yours; avoid clobbering XLA_FLAGS)
export JAX_TRACEBACK_FILTERING=off
export JAX_DISABLE_CUSOLVER=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=${DUMP_DIR} --xla_gpu_cuda_data_dir=/usr/local/cuda"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=.60

echo "=== Madrona + GPU detection (inside container) ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  ACTUAL_GPU=$(nvidia-smi -L 2>/dev/null | head -1)
  echo "Actual GPU: $ACTUAL_GPU"
  GPU_MODEL=$(echo "$ACTUAL_GPU" | grep -o "H100\|L40S\|A100\|V100\|RTX" | head -1)
  [[ -z "$GPU_MODEL" ]] && GPU_MODEL="unknown"
else
  echo "WARNING: nvidia-smi not found in container; using generic GPU tag"
  GPU_MODEL="unknown"
fi
GPU_MODEL_LOWER=$(echo "$GPU_MODEL" | tr "[:upper:]" "[:lower:]")
ENV_CONFIG="default"

# Madrona caches: move to /tmp to avoid /scratch hiccups on first render/init
CACHE_BUILD_DIR="/tmp/${SLURM_JOB_ID}_madrona_build_${GPU_MODEL_LOWER}_${ENV_CONFIG}"
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

python - <<'"'"'PY'"'"'
import tensorboard
print("TensorBoard version:", getattr(tensorboard, "__version__", "unknown"))
PY
echo "============================================"

# --- XLA dump (writes into the directory you're running in) ---
DUMP_DIR="${SLURM_SUBMIT_DIR}/xla_dump_${SLURM_JOB_ID}"
mkdir -p "$DUMP_DIR"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=${DUMP_DIR}"

# --- launch python with periodic stack dumps (every 60s) ---
set +e
stdbuf -oL -eL python -u - <<'PY' 2>&1 &
import faulthandler, runpy
faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)  # <-- this is the missing piece
runpy.run_path("execute.py", run_name="__main__")
PY
PY_PID=$!

# --- heartbeat AFTER PY_PID is known ---
heartbeat () {
  while kill -0 "$PY_PID" 2>/dev/null; do
    ts=$(date +"%H:%M:%S")
    cpu=$(ps -p "$PY_PID" -o %cpu=,rss=,etime=,stat= 2>/dev/null | tr -s " " | sed "s/^ //")
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "na")
    echo "[hb $ts] py(pid=$PY_PID) cpu(%CPU,RSSKB,ETIME,STAT)=${cpu:-na} | gpu(util%,memutil%,memMB)=${gpu:-na}"
    sleep 5
  done
}
heartbeat & HB_PID=$!

wait $PY_PID
RC=$?

kill $HB_PID >/dev/null 2>&1 || true
wait $HB_PID >/dev/null 2>&1 || true
set -e

echo "Training completed (exit_code=$RC)."
exit $RC
'

echo "Finished at $(date)"
