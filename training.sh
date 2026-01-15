#!/bin/bash
#SBATCH --job-name=mjxs_mvmae
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=1:00:00
#SBATCH --account=aip-aspuru-ab
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load apptainer/1.3.5 || module load apptainer

# Prevent host python env from shadowing container
unset PYTHONPATH
unset PYTHONHOME

# ---------------- Save node used by this job ----------------
LAST_NODE_FILE="$SLURM_SUBMIT_DIR/.last_node"
NODE="${SLURMD_NODENAME:-$(hostname)}"
echo "$NODE" > "$LAST_NODE_FILE" || true
echo "NODE=$NODE"

# ---------------- Per-job cache isolation + reset ----------------
JOB_TMP="${SLURM_TMPDIR:-/tmp/$USER/slurm_$SLURM_JOB_ID}"
mkdir -p "$JOB_TMP"

export XDG_CACHE_HOME="$JOB_TMP/xdg_cache"
export TMPDIR="$JOB_TMP/tmp"
export JAX_COMPILATION_CACHE_DIR="$JOB_TMP/jax_cache"
export CUDA_CACHE_PATH="$JOB_TMP/cuda_cache"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-2147483648}"  # 2GB

# Hard reset job-local caches every run
rm -rf "$XDG_CACHE_HOME" "$TMPDIR" "$JAX_COMPILATION_CACHE_DIR" "$CUDA_CACHE_PATH" || true
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" "$JAX_COMPILATION_CACHE_DIR" "$CUDA_CACHE_PATH"

# Optional: wipe global user caches (set CLEAN_GLOBAL_CACHE=1 when you want it)
if [[ "${CLEAN_GLOBAL_CACHE:-0}" == "1" ]]; then
  rm -rf ~/.cache/jax ~/.cache/xla ~/.cache/torch_extensions \
         ~/.cache/mesa_shader_cache ~/.cache/glcache 2>/dev/null || true
  rm -rf ~/.nv/ComputeCache 2>/dev/null || true
fi

# Optional: disable CUDA caching entirely (debug only; slower compiles)
if [[ "${DISABLE_CUDA_CACHE:-0}" == "1" ]]; then
  export CUDA_CACHE_DISABLE=1
fi

# --------------- Apptainer image ---------------
IMG="$SLURM_SUBMIT_DIR/training.sif"
[[ -f "$IMG" ]] || { echo "ERROR: $IMG not found"; exit 2; }

HOST_PROJECT_ROOT="$SLURM_SUBMIT_DIR"
WORKDIR_IN_CONTAINER="/workspace"

[[ -f "$HOST_PROJECT_ROOT/execute.py" ]] || { echo "FATAL: execute.py not found"; exit 10; }

# ---------------- Python path inside container ----------------
export APPTAINERENV_PYTHONPATH="/workspace:/opt/src:/opt/src/MV_MAE_Implementation"

# ---- JAX backend selection (CUDA PJRT plugin) ----
export APPTAINERENV_JAX_PLATFORMS="cuda,cpu"

# ---------------- Pass cache isolation into container ----------------
export APPTAINERENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"
export APPTAINERENV_TMPDIR="$TMPDIR"
export APPTAINERENV_JAX_COMPILATION_CACHE_DIR="$JAX_COMPILATION_CACHE_DIR"
export APPTAINERENV_CUDA_CACHE_PATH="$CUDA_CACHE_PATH"
export APPTAINERENV_CUDA_CACHE_MAXSIZE="$CUDA_CACHE_MAXSIZE"
export APPTAINERENV_CUDA_CACHE_DISABLE="${CUDA_CACHE_DISABLE:-0}"

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
[[ -f "$VENDOR_JSON" ]] || { echo "FATAL: $VENDOR_JSON not found on host."; exit 3; }
export APPTAINERENV__EGL_VENDOR_LIBRARY_FILENAMES="$VENDOR_JSON"

# Locate libEGL_nvidia.so on HOST
NV_EGL_DIR="$(ldconfig -p | awk '/libEGL_nvidia\.so/{print $NF; exit}' | xargs -r dirname || true)"
for d in /usr/lib/x86_64-linux-gnu/nvidia /usr/lib/nvidia /usr/lib64/nvidia /usr/lib/x86_64-linux-gnu; do
  [[ -z "${NV_EGL_DIR:-}" && -e "$d/libEGL_nvidia.so.0" ]] && NV_EGL_DIR="$d"
done
[[ -n "${NV_EGL_DIR:-}" && -d "$NV_EGL_DIR" ]] || { echo "FATAL: Could not find libEGL_nvidia.so* on host."; exit 4; }

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
BIND_FLAGS+=( --bind "$JOB_TMP:$JOB_TMP" )

# ---------------- Run ----------------
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  "$IMG" \
  bash -lc '
set -euo pipefail
. /opt/mvmae_venv/bin/activate

# Ensure caches are job-local INSIDE container too
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
export TMPDIR="${TMPDIR:-/tmp}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/jax_cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/tmp/cuda_cache}"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-2147483648}"
export CUDA_CACHE_DISABLE="${CUDA_CACHE_DISABLE:-0}"

rm -rf "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR" "$CUDA_CACHE_PATH" \
       "$TMPDIR/jax-*" "$TMPDIR/xla-*" 2>/dev/null || true
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" "$JAX_COMPILATION_CACHE_DIR" "$CUDA_CACHE_PATH"

# Force container CUDA runtime libs to win over injected host libs
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/usr/local/cuda/lib64/libnvJitLink.so.12:${LD_PRELOAD:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda ${XLA_FLAGS:-}"

# Runtime knobs
export PYTHONUNBUFFERED=1
export JAX_TRACEBACK_FILTERING=off
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=.60
export PICK_ENV_DEBUG=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"

# Put Madrona build FIRST
export PYTHONPATH="/opt/madrona_mjx/build:/workspace:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/opt/madrona_mjx/build:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

stdbuf -oL -eL python -u execute.py 2>&1
'
