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

# --------------- Apptainer image ---------------
IMG="$SLURM_SUBMIT_DIR/training.sif"
if [[ ! -f "$IMG" ]]; then
  echo "ERROR: $IMG not found"
  exit 2
fi

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

echo "=== Host sanity ==="
echo "Host apptainer: $(command -v apptainer || true)"
apptainer --version || true
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
echo "Host CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Host SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "==================="

# ---------------- Run (single exec) ----------------
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  "$IMG" \
  bash -lc '
set -euo pipefail
. /opt/mvmae_venv/bin/activate

# --- Force container CUDA runtime libs to win over injected host libs ---
export CUDA_HOME=/usr/local/cuda
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda ${XLA_FLAGS:-}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/usr/local/cuda/lib64/libnvJitLink.so.12:${LD_PRELOAD:-}"

# Training/runtime knobs
export PYTHONUNBUFFERED=1
export JAX_TRACEBACK_FILTERING=off
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=.60
export PICK_ENV_DEBUG=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"

# Put Madrona build FIRST so top-level _madrona_* resolves to the .so
export PYTHONPATH="/opt/madrona_mjx/build:/workspace:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/opt/madrona_mjx/build:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

echo "=== Container env (high signal) ==="
echo "whoami=$(whoami)  pwd=$(pwd)"
echo "python=$(command -v python)"
python -V
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "LD_PRELOAD=$LD_PRELOAD"
echo "PYTHONPATH=$PYTHONPATH"
echo "XLA_FLAGS=$XLA_FLAGS"
echo "JAX_PLATFORMS=$JAX_PLATFORMS"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
echo "=================================="

echo "=== System / driver info (inside container) ==="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
echo "--- /usr/local/cuda/version.json (if exists) ---"
cat /usr/local/cuda/version.json 2>/dev/null || true
echo "--- nvcc --version (if exists) ---"
command -v nvcc >/dev/null 2>&1 && nvcc --version || true
echo "==============================================="

echo "=== Dynamic loader cache (GPU libs) ==="
ldconfig -p | egrep -i "libcuda\.so|libcudart\.so|libnvrtc\.so|libnvjitlink\.so|libcublas\.so|libcusolver\.so|libcudnn\.so|libcusparse\.so|libcurand\.so" || true
echo "======================================"

echo "=== pip freeze (focused) ==="
python -m pip show jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-nvjitlink-cu12 nvidia-cudnn-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-curand-cu12 nvidia-cufft-cu12 2>/dev/null || true
echo "========================================"

echo "=== Version sanity (mujoco / jax / jaxlib) ==="
python - <<'"'"'PY'"'"'
import os, mujoco, jax, jaxlib
print("MuJoCo:", mujoco.__version__)
print("jax   :", jax.__version__)
print("jaxlib:", jaxlib.__version__)
print("JAX_PLATFORMS:", os.environ.get("JAX_PLATFORMS"))
print("devices:", jax.devices())
try:
    backend = jax.lib.xla_bridge.get_backend()
    print("backend:", backend.platform, "|", getattr(backend, "platform_version", None))
except Exception as e:
    print("backend: <failed>", e)
PY
echo "============================================="

echo "=== nvJitLink resolution + symbol check ==="
python - <<'"'"'PY'"'"'
import ctypes, os, subprocess, sys
def sh(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"<failed {cmd}: {e}>"
print("ldconfig nvjitlink:")
print(sh(["bash","-lc","ldconfig -p | grep -i nvjitlink || true"]))
paths = [
    "/usr/local/cuda/lib64/libnvJitLink.so.12",
    "/usr/local/cuda/targets/x86_64-linux/lib/libnvJitLink.so.12",
]
p = next((x for x in paths if os.path.exists(x)), None)
print("using:", p)
if not p:
    raise SystemExit("FATAL: libnvJitLink.so.12 not found in expected locations")
lib = ctypes.CDLL(p)
for sym in ["__nvJitLinkCreate_12_5", "__nvJitLinkCreate_12_4", "__nvJitLinkCreate_12_3"]:
    print(sym, "->", hasattr(lib, sym))
PY
echo "========================================="

# Hard delete any shadowing python files from /workspace (host bind)
rm -f /workspace/_madrona_mjx_batch_renderer.py /workspace/_madrona_mjx_visualizer.py
rm -rf /workspace/__pycache__ || true

echo "=== Where Madrona resolves + file hashes (prove exact .so) ==="
python - <<'"'"'PY'"'"'
import importlib.util, hashlib, os
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()
mods = ["_madrona_mjx_batch_renderer", "_madrona_mjx_visualizer",
        "madrona_mjx._madrona_mjx_batch_renderer", "madrona_mjx._madrona_mjx_visualizer"]
for name in mods:
    spec = importlib.util.find_spec(name)
    origin = getattr(spec, "origin", None)
    print(name, "->", origin)
    if origin and os.path.exists(origin) and origin.endswith(".so"):
        print("  sha256:", sha256(origin))
PY
echo "=============================================================="

echo "=== Link check: full ldd (unfiltered, for exact paths) ==="
for so in /opt/madrona_mjx/build/_madrona_mjx_batch_renderer*.so /opt/madrona_mjx/build/_madrona_mjx_visualizer*.so /opt/madrona_mjx/build/libmadmjx_mgr.so; do
  [[ -e "$so" ]] || continue
  echo "----- ldd $so -----"
  ldd "$so" || true
done
echo "========================================================"

echo "=== Runtime loader trace (import-only) ==="
# Full trace is huge; we also emit a filtered summary you can read quickly.
LD_DEBUG=libs python - <<'"'"'PY'"'"' 2>&1 | tee /tmp/ld_debug_madrona.txt >/dev/null || true
import _madrona_mjx_batch_renderer as br
print("import ok:", br.__file__)
PY
echo "--- LD_DEBUG filtered summary ---"
egrep -i "(_madrona_mjx|libcuda|libcudart|libnvjitlink|libnvrtc|libcublas|libcusolver|libcudnn|libcusparse|libcurand)" /tmp/ld_debug_madrona.txt || true
echo "================================="

echo "=== Process map after import (shows actual loaded .so paths) ==="
python - <<'"'"'PY'"'"'
import _madrona_mjx_batch_renderer  # force load
with open("/proc/self/maps","r") as f:
    lines = [ln.strip() for ln in f if ".so" in ln and ("cuda" in ln.lower() or "nvjit" in ln.lower() or "nvrtc" in ln.lower() or "madrona" in ln.lower())]
print("\n".join(lines))
PY
echo "=============================================================="

echo "=== Minimal GPU sanity: torch + jax ==="
python - <<'"'"'PY'"'"'
import torch
print("[torch] version:", torch.__version__)
print("[torch] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[torch] gpu:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("[torch] matmul ok")
import jax, jax.numpy as jnp
a = jnp.ones((1024,1024), dtype=jnp.float32)
b = (a @ a).block_until_ready()
dev_attr = getattr(b, "device", None)
dev = dev_attr() if callable(dev_attr) else dev_attr
print("[jax] matmul ok on:", dev)
PY
echo "======================================"

echo "=== JAX compilation logging (only for crashes around custom calls) ==="
# These can help if XLA is generating malformed backend_config for a custom call.
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text --xla_dump_disable_metadata"
mkdir -p /tmp/xla_dump || true
echo "XLA dump dir: /tmp/xla_dump"
echo "==============================================================="

echo "=== Run training ==="
stdbuf -oL -eL python -u execute.py 2>&1
echo "Training completed."
'

echo "Finished at $(date)"
