#!/bin/bash
# =========================
# mjxs_mvmae.sh
# =========================
#SBATCH --job-name=mjxs_mvmae
#SBATCH --nodes=1
#SBATCH --exclude=kn117
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

module load StdEnv/2023 apptainer/1.3.5 || module load apptainer

unset PYTHONPATH
unset PYTHONHOME

IMG="$SLURM_SUBMIT_DIR/training.sif"
[[ -f "$IMG" ]] || { echo "ERROR: $IMG not found"; exit 2; }

HOST_PROJECT_ROOT="$SLURM_SUBMIT_DIR"
WORKDIR_IN_CONTAINER="/workspace"
[[ -f "$HOST_PROJECT_ROOT/execute.py" ]] || { echo "FATAL: execute.py not found"; exit 10; }

export APPTAINERENV_PYTHONPATH="/workspace:/opt/src:/opt/src/MV_MAE_Implementation"
export APPTAINERENV_JAX_PLATFORMS="cuda,cpu"

export APPTAINERENV_MUJOCO_GL=egl
export APPTAINERENV_PYOPENGL_PLATFORM=egl
export APPTAINERENV_MUJOCO_PLATFORM=egl
export APPTAINERENV_DISPLAY=
export APPTAINERENV_LIBGL_ALWAYS_SOFTWARE=0
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export APPTAINERENV_IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg

# ----------------------------
# NEW: enable in-code probes
# ----------------------------
export APPTAINERENV_PICK_ENV_PROBE="${PICK_ENV_PROBE:-1}"
export APPTAINERENV_WRAPPER_PROBE="${WRAPPER_PROBE:-1}"
export APPTAINERENV_PROBE_CUDA_CLEAR_MODE="${PROBE_CUDA_CLEAR_MODE:-peek}"   # peek (don't clear) or get (clear)
export APPTAINERENV_PROBE_DUMP_MAPS="${PROBE_DUMP_MAPS:-1}"
export APPTAINERENV_PROBE_FAIL_FAST="${PROBE_FAIL_FAST:-0}"
export APPTAINERENV_PICK_ENV_PROBE_SYNC="${PICK_ENV_PROBE_SYNC:-1}"         # block_until_ready(rgb) inside pick_env render
export APPTAINERENV_WRAPPER_PROBE_MAX_STEPS="${WRAPPER_PROBE_MAX_STEPS:-1}"  # detailed wrapper probes for first N steps
export APPTAINERENV_WRAPPER_PROBE_EXIT_AFTER_RESET="${WRAPPER_PROBE_EXIT_AFTER_RESET:-0}"

VENDOR_JSON="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
[[ -f "$VENDOR_JSON" ]] || { echo "FATAL: $VENDOR_JSON missing on host"; exit 3; }
export APPTAINERENV__EGL_VENDOR_LIBRARY_FILENAMES="$VENDOR_JSON"

NV_EGL_DIR="$(ldconfig -p | awk '/libEGL_nvidia\.so/{print $NF; exit}' | xargs -r dirname || true)"
for d in /usr/lib/x86_64-linux-gnu/nvidia /usr/lib/nvidia /usr/lib64/nvidia /usr/lib/x86_64-linux-gnu; do
  [[ -z "${NV_EGL_DIR:-}" && -e "$d/libEGL_nvidia.so.0" ]] && NV_EGL_DIR="$d"
done
[[ -n "${NV_EGL_DIR:-}" && -d "$NV_EGL_DIR" ]] || { echo "FATAL: libEGL_nvidia.so* not found on host"; exit 4; }

GLVND_DIR="/usr/lib/x86_64-linux-gnu"
[[ -e "$GLVND_DIR/libEGL.so.1" ]] || GLVND_DIR="/usr/lib64"

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

# Probe 1: EGL + Torch (kept)
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

# Training (with in-code probes)
apptainer exec --nv "${BIND_FLAGS[@]}" --pwd "$WORKDIR_IN_CONTAINER" "$IMG" bash -lc '
set -euo pipefail
. /opt/mvmae_venv/bin/activate

export PYTHONNOUSERSITE=1

if [[ -d /usr/local/cuda-12.4 ]]; then
  export CUDA_HOME=/usr/local/cuda-12.4
else
  export CUDA_HOME=/usr/local/cuda
fi

SYSTEM_CUDA_LD="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:/opt/madrona_mjx/build:/.singularity.d/libs:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

PIP_CUDA_LD="$(python - << "PY"
import os, sys
root=None
for p in sys.path:
  if p and "site-packages" in p:
    n=os.path.join(p,"nvidia")
    if os.path.isdir(n):
      root=n
      break
dirs=[]
if root:
  for sub in ["cuda_runtime","cuda_nvrtc","nvjitlink","cublas","cudnn","cufft","curand","cusolver","cusparse","nccl"]:
    d=os.path.join(root,sub,"lib")
    if os.path.isdir(d):
      dirs.append(d)
print(":".join(dirs))
PY
)"

# Choose LD mode for the actual training process (override with MAD_LD_MODE=pip)
LD_MODE="${MAD_LD_MODE:-system}"
if [[ "$LD_MODE" == "pip" ]]; then
  export LD_LIBRARY_PATH="${PIP_CUDA_LD}${PIP_CUDA_LD:+:}$SYSTEM_CUDA_LD"
else
  export LD_LIBRARY_PATH="$SYSTEM_CUDA_LD"
fi

export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=.60
export PYTHONUNBUFFERED=1
export JAX_TRACEBACK_FILTERING=off
export PICK_ENV_DEBUG=1

echo "=== [CONTAINER] preflight env snapshot ==="
echo "CUDA_HOME=$CUDA_HOME"
echo "LD_MODE=$LD_MODE"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "XLA_FLAGS=$XLA_FLAGS"
echo "PICK_ENV_PROBE=${PICK_ENV_PROBE:-}"
echo "WRAPPER_PROBE=${WRAPPER_PROBE:-}"
echo "PROBE_CUDA_CLEAR_MODE=${PROBE_CUDA_CLEAR_MODE:-}"
echo "PROBE_DUMP_MAPS=${PROBE_DUMP_MAPS:-}"
echo "PICK_ENV_PROBE_SYNC=${PICK_ENV_PROBE_SYNC:-}"
echo "WRAPPER_PROBE_MAX_STEPS=${WRAPPER_PROBE_MAX_STEPS:-}"
echo "WRAPPER_PROBE_EXIT_AFTER_RESET=${WRAPPER_PROBE_EXIT_AFTER_RESET:-}"
echo "=========================================="

echo "=== [CONTAINER] nvidia-smi ==="
nvidia-smi -L || true
nvidia-smi || true
echo "============================="

# Cache paths (Madrona compile)
ACTUAL_GPU=$(nvidia-smi -L 2>/dev/null | head -1 || true)
GPU_MODEL=$(echo "$ACTUAL_GPU" | grep -o "H100\|L40S\|A100\|V100\|RTX" | head -1 || true)
[[ -z "${GPU_MODEL:-}" ]] && GPU_MODEL="unknown"
GPU_MODEL_LOWER=$(echo "$GPU_MODEL" | tr "[:upper:]" "[:lower:]")
CACHE_BUILD_DIR="'"$SLURM_SUBMIT_DIR"'/build_${GPU_MODEL_LOWER}_default"
mkdir -p "$CACHE_BUILD_DIR/kernel_cache" "$CACHE_BUILD_DIR/bvh_cache"
export MADRONA_MWGPU_KERNEL_CACHE="$CACHE_BUILD_DIR/kernel_cache/kernel.cache"
export MADRONA_BVH_KERNEL_CACHE="$CACHE_BUILD_DIR/bvh_cache/bvh.cache"
echo "MADRONA_MWGPU_KERNEL_CACHE=$MADRONA_MWGPU_KERNEL_CACHE"
echo "MADRONA_BVH_KERNEL_CACHE=$MADRONA_BVH_KERNEL_CACHE"

echo "=== [CONTAINER] starting training (probes live in pick_env + wrapper) ==="
stdbuf -oL -eL python -u execute.py 2>&1
'

echo "Finished at $(date)"
