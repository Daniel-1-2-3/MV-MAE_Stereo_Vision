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
[[ -f "$IMG" ]] || { echo "ERROR: $IMG not found"; exit 2; }

HOST_PROJECT_ROOT="$SLURM_SUBMIT_DIR"
WORKDIR_IN_CONTAINER="/workspace"
[[ -f "$HOST_PROJECT_ROOT/execute.py" ]] || { echo "FATAL: execute.py not found"; exit 10; }

# -------- container env --------
export APPTAINERENV_PYTHONPATH="/workspace:/opt/src:/opt/src/MV_MAE_Implementation:${PYTHONPATH:-}"
export APPTAINERENV_MUJOCO_GL=egl
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_JAX_LOG_COMPILES=0

export APPTAINERENV_PYOPENGL_PLATFORM=egl
export APPTAINERENV_MUJOCO_PLATFORM=egl
export APPTAINERENV_DISPLAY=
export APPTAINERENV_LIBGL_ALWAYS_SOFTWARE=0
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export APPTAINERENV_IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg

# -------- EGL host bits --------
VENDOR_JSON="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
[[ -f "$VENDOR_JSON" ]] || { echo "FATAL: missing $VENDOR_JSON"; exit 3; }
export APPTAINERENV__EGL_VENDOR_LIBRARY_FILENAMES="$VENDOR_JSON"

NV_EGL_DIR="$(ldconfig -p | awk '/libEGL_nvidia\.so/{print $NF; exit}' | xargs -r dirname || true)"
for d in /usr/lib/x86_64-linux-gnu/nvidia /usr/lib/nvidia /usr/lib64/nvidia /usr/lib/x86_64-linux-gnu; do
  [[ -z "$NV_EGL_DIR" && -e "$d/libEGL_nvidia.so.0" ]] && NV_EGL_DIR="$d"
done
[[ -n "${NV_EGL_DIR:-}" && -d "$NV_EGL_DIR" ]] || { echo "FATAL: could not find libEGL_nvidia.so on host"; exit 4; }

GLVND_DIR="/usr/lib/x86_64-linux-gnu"
[[ -e "$GLVND_DIR/libEGL.so.1" ]] || GLVND_DIR="/usr/lib64"

# -------- binds --------
HOST_MJP_DEPS="$SLURM_SUBMIT_DIR/mujoco_playground_external_deps"
mkdir -p "$HOST_MJP_DEPS"
MJP_DEPS_IN_CONTAINER="/opt/mvmae_venv/lib/python3.12/site-packages/mujoco_playground/external_deps"

BIND_FLAGS=( --bind "$HOST_PROJECT_ROOT:$HOST_PROJECT_ROOT" )
BIND_FLAGS+=( --bind "/usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d" )
BIND_FLAGS+=( --bind "$NV_EGL_DIR:$NV_EGL_DIR" )
BIND_FLAGS+=( --bind "$GLVND_DIR:$GLVND_DIR" )
BIND_FLAGS+=( --bind "$HOST_MJP_DEPS:$MJP_DEPS_IN_CONTAINER" )
BIND_FLAGS+=( --bind "$HOST_PROJECT_ROOT:$WORKDIR_IN_CONTAINER" )

# -------- quick probe --------
apptainer exec --nv "${BIND_FLAGS[@]}" --pwd "$WORKDIR_IN_CONTAINER" "$IMG" bash -lc '
python - << "PY"
import torch, mujoco, OpenGL.GL as gl
print("torch:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
ctx = mujoco.GLContext(64, 64); ctx.make_current()
to_s = lambda b: b.decode("utf-8","ignore") if b else None
print("GL:", to_s(gl.glGetString(gl.GL_VENDOR)), "|", to_s(gl.glGetString(gl.GL_RENDERER)))
ctx.free()
PY
'

# -------- run --------
apptainer exec --nv "${BIND_FLAGS[@]}" --pwd "$WORKDIR_IN_CONTAINER" "$IMG" bash -lc '
set -e
python - <<'"'"'PY'"'"'
import mujoco
print("MuJoCo:", mujoco.__version__)
PY

# ---- fast local caches ----
export XDG_CACHE_HOME="/tmp/${SLURM_JOB_ID}_cache"
export CUDA_CACHE_PATH="/tmp/${SLURM_JOB_ID}_cuda_cache"
export CUDA_CACHE_MAXSIZE=2147483648
export JAX_CACHE_DIR="/tmp/${SLURM_JOB_ID}_jax_cache"
mkdir -p "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$JAX_CACHE_DIR"
export CUDA_MODULE_LOADING=LAZY

# ---- LD_LIBRARY_PATH test: drop /usr/local/cuda* from front, keep madrona build ----
ORIG_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
LD_LIBRARY_PATH_CLEAN="$(echo "${ORIG_LD_LIBRARY_PATH}" | tr ":" "\n" | \
  grep -v "^/usr/local/cuda" | tr "\n" ":" | sed "s/:$//")"
export LD_LIBRARY_PATH="/opt/madrona_mjx/build${LD_LIBRARY_PATH_CLEAN:+:${LD_LIBRARY_PATH_CLEAN}}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# ---- XLA flags: append, don’t overwrite ----
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_cuda_data_dir=/usr/local/cuda"
# XLA dump (persists on host via /workspace mount)
XLA_DUMP_DIR="/workspace/xla_dump_${SLURM_JOB_ID}"
mkdir -p "$XLA_DUMP_DIR"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=${XLA_DUMP_DIR}"
echo "XLA_DUMP_DIR=${XLA_DUMP_DIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=.60
export JAX_TRACEBACK_FILTERING=off
export JAX_DISABLE_CUSOLVER=1

# ---- madrona caches on /tmp (avoid FS stalls) ----
GPU_LINE="$(nvidia-smi -L 2>/dev/null | head -1 || true)"
GPU_MODEL="$(echo "$GPU_LINE" | grep -o "H100\|L40S\|A100\|V100\|RTX" | head -1)"
GPU_MODEL="${GPU_MODEL:-unknown}"
GPU_MODEL_LOWER="$(echo "$GPU_MODEL" | tr "[:upper:]" "[:lower:]")"
ENV_CONFIG="default"

CACHE_BUILD_DIR="/tmp/${SLURM_JOB_ID}_madrona_${GPU_MODEL_LOWER}_${ENV_CONFIG}"
mkdir -p "$CACHE_BUILD_DIR/kernel_cache" "$CACHE_BUILD_DIR/bvh_cache"
export MADRONA_MWGPU_KERNEL_CACHE="$CACHE_BUILD_DIR/kernel_cache/kernel.cache"
export MADRONA_BVH_KERNEL_CACHE="$CACHE_BUILD_DIR/bvh_cache/bvh.cache"

echo "GPU=${GPU_LINE:-unknown}"
echo "MADRONA_CACHE=${CACHE_BUILD_DIR}"
echo "CUDA_CACHE=${CUDA_CACHE_PATH}"
echo "JAX_CACHE=${JAX_CACHE_DIR}"

# ---- persistent prefix (keep your tensorboard logic, quieter) ----
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

python - <<'"'"'PY'"'"'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("tensorboard") else 1)
PY
|| python -m pip install --upgrade --no-cache-dir --prefix "$DEPS_PREFIX" tensorboard

# ---- watchdog: prints every 5s while execute.py runs ----
watchdog () {
  while true; do
    ts="$(date +%H:%M:%S)"
    cpu="$(ps -o %cpu= -p $$ | tr -d " " || true)"
    gpu="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
    echo "[hb ${ts}] gpu(util%,memMB)=${gpu:-na}"
    sleep 5
  done
}
watchdog & WD_PID=$!

# ---- run ----
set +e
stdbuf -oL -eL python -u execute.py 2>&1
RC=$?
set -e

kill $WD_PID >/dev/null 2>&1 || true
wait $WD_PID >/dev/null 2>&1 || true

echo "exit_code=${RC}"
exit $RC
'

echo "Finished at $(date)"
