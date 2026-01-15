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

# ---------------- Per-job cache isolation + reset ----------------
# Use node-local scratch if available; otherwise fall back to /tmp.
JOB_TMP="${SLURM_TMPDIR:-/tmp/$USER/slurm_$SLURM_JOB_ID}"
mkdir -p "$JOB_TMP"

# Put *all* caches in job-local dirs so nothing persists across runs
export XDG_CACHE_HOME="$JOB_TMP/xdg_cache"
export TMPDIR="$JOB_TMP/tmp"
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"

# JAX/XLA per-job compilation cache (prevents cross-job contamination)
export JAX_COMPILATION_CACHE_DIR="$JOB_TMP/jax_cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# CUDA per-job kernel cache (prevents stale/bad kernels persisting)
export CUDA_CACHE_PATH="$JOB_TMP/cuda_cache"
mkdir -p "$CUDA_CACHE_PATH"

# Hard reset the per-job caches every run of this script
rm -rf "$JAX_COMPILATION_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$TMPDIR"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$TMPDIR"

# Optional: also wipe global user caches (set CLEAN_GLOBAL_CACHE=1 when you want it)
if [[ "${CLEAN_GLOBAL_CACHE:-0}" == "1" ]]; then
  echo "=== WIPING GLOBAL USER CACHES (CLEAN_GLOBAL_CACHE=1) ==="
  rm -rf ~/.cache/jax ~/.cache/xla ~/.cache/torch_extensions ~/.cache/mesa_shader_cache ~/.cache/glcache 2>/dev/null || true
  rm -rf ~/.nv/ComputeCache 2>/dev/null || true
  echo "======================================================="
fi

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

# ---------------- Pass cache isolation into container ----------------
export APPTAINERENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"
export APPTAINERENV_TMPDIR="$TMPDIR"
export APPTAINERENV_JAX_COMPILATION_CACHE_DIR="$JAX_COMPILATION_CACHE_DIR"
export APPTAINERENV_CUDA_CACHE_PATH="$CUDA_CACHE_PATH"

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
echo "JOB_TMP=$JOB_TMP"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR"
echo "CUDA_CACHE_PATH=$CUDA_CACHE_PATH"
echo "Host apptainer: $(command -v apptainer || true)"
apptainer --version || true
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
  nvidia-smi || true
fi
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

# --- Ensure caches are job-local INSIDE container too ---
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
export TMPDIR="${TMPDIR:-/tmp}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/jax_cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/tmp/cuda_cache}"
rm -rf "$XDG_CACHE_HOME" "$TMPDIR/jax-*" "$TMPDIR/xla-*" "$JAX_COMPILATION_CACHE_DIR" "$CUDA_CACHE_PATH" 2>/dev/null || true
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" "$JAX_COMPILATION_CACHE_DIR" "$CUDA_CACHE_PATH"

# --- Force container CUDA runtime libs to win over injected host libs ---
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/usr/local/cuda/lib64/libnvJitLink.so.12:${LD_PRELOAD:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda ${XLA_FLAGS:-}"

# Training/runtime knobs (keep minimal)
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
echo "pwd=$(pwd)"
python -V
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "TMPDIR=$TMPDIR"
echo "JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR"
echo "CUDA_CACHE_PATH=$CUDA_CACHE_PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "LD_PRELOAD=$LD_PRELOAD"
echo "PYTHONPATH=$PYTHONPATH"
echo "XLA_FLAGS=$XLA_FLAGS"
echo "JAX_PLATFORMS=$JAX_PLATFORMS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "=================================="

echo "=== Driver/GPU (inside container) ==="
command -v nvidia-smi >/dev/null 2>&1 && (nvidia-smi -L || true) && (nvidia-smi || true)
command -v nvcc >/dev/null 2>&1 && nvcc --version || true
echo "===================================="

echo "=== JAX/MuJoCo versions + backend ==="
python - <<'"'"'PY'"'"'
import os, mujoco, jax, jaxlib
print("MuJoCo:", mujoco.__version__)
print("jax   :", jax.__version__)
print("jaxlib:", jaxlib.__version__)
print("JAX_PLATFORMS:", os.environ.get("JAX_PLATFORMS"))
print("devices:", jax.devices())
backend = jax.lib.xla_bridge.get_backend()
print("backend:", backend.platform, "|", getattr(backend, "platform_version", None))
PY
echo "===================================="

echo "=== nvJitLink exact path + symbols ==="
python - <<'"'"'PY'"'"'
import ctypes, os
cands = [
  "/usr/local/cuda/lib64/libnvJitLink.so.12",
  "/usr/local/cuda/targets/x86_64-linux/lib/libnvJitLink.so.12",
]
p = next((x for x in cands if os.path.exists(x)), None)
print("libnvJitLink:", p)
if not p: raise SystemExit("FATAL: libnvJitLink.so.12 not found")
lib = ctypes.CDLL(p)
for sym in ["__nvJitLinkCreate_12_5", "__nvJitLinkCreate_12_4", "__nvJitLinkCreate_12_3"]:
  print(sym, "->", hasattr(lib, sym))
PY
echo "====================================="

echo "=== Madrona extension resolution (must be .so) ==="
rm -f /workspace/_madrona_mjx_batch_renderer.py /workspace/_madrona_mjx_visualizer.py
rm -rf /workspace/__pycache__ || true
python - <<'"'"'PY'"'"'
import importlib.util
names = ["_madrona_mjx_batch_renderer","_madrona_mjx_visualizer",
         "madrona_mjx._madrona_mjx_batch_renderer","madrona_mjx._madrona_mjx_visualizer"]
for n in names:
    spec = importlib.util.find_spec(n)
    print(n, "->", getattr(spec, "origin", None))
PY
echo "=============================================="

echo "=== ldd (critical CUDA libs) ==="
for so in /opt/madrona_mjx/build/_madrona_mjx_batch_renderer*.so \
          /opt/madrona_mjx/build/_madrona_mjx_visualizer*.so \
          /opt/madrona_mjx/build/libmadmjx_mgr.so; do
  [[ -e "$so" ]] || continue
  echo "----- $so -----"
  ldd "$so" | egrep -i "libcuda\.so|libcudart\.so|libnvrtc\.so|libnvjitlink\.so" || true
done
echo "================================"

echo "=== Minimal GPU sanity ==="
python - <<'"'"'PY'"'"'
import jax, jax.numpy as jnp
a = jnp.ones((512,512), dtype=jnp.float32)
(a @ a).block_until_ready()
print("[jax] matmul ok")
try:
    import torch
    print("[torch] cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        x = torch.randn(512, 512, device="cuda")
        (x @ x).sum().item()
        torch.cuda.synchronize()
        print("[torch] matmul ok")
except Exception as e:
    print("[torch] skipped/failed:", type(e).__name__, e)
PY
echo "=========================="

echo "=== smalltest-REPRO (Madrona init/render) ==="
python - <<'"'"'PY'"'"'
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from madrona_mjx.renderer import BatchRenderer
import numpy as np

xml = r"""
<mujoco model="mini">
  <option timestep="0.01"/>
  <worldbody>
    <light name="sun" pos="0 0 1" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.3 0.3 0.3 1"/>
    <body name="b" pos="0 0 0.20">
      <joint name="hinge" type="hinge" axis="0 0 1" limited="false"/>
      <geom type="sphere" size="0.05" rgba="0.8 0.2 0.2 1"/>
    </body>
    <camera name="cam0" pos="-0.15 -0.35 0.20" xyaxes="1 0 0 0 0 1" fovy="60"/>
    <camera name="cam1" pos=" 0.15 -0.35 0.20" xyaxes="1 0 0 0 0 1" fovy="60"/>
  </worldbody>
</mujoco>
"""
mjm = mujoco.MjModel.from_xml_string(xml)
m = mjx.put_model(mjm)
B = 1
data = jax.vmap(lambda _: mjx.make_data(m))(jnp.arange(B))
data = data.replace(qpos=data.qpos.at[:, 0].set(jnp.array(0.25, dtype=jnp.float32)))
data = jax.vmap(lambda d: mjx.forward(m, d))(data)

r = BatchRenderer(
    m=m, gpu_id=0, num_worlds=B,
    batch_render_view_width=64, batch_render_view_height=64,
    enabled_geom_groups=np.asarray([0,1,2], np.int32, order="C"),
    enabled_cameras=np.asarray([0,1], np.int32, order="C"),
    add_cam_debug_geo=False, use_rasterizer=False, viz_gpu_hdls=None,
)
tok, _, _ = r.init(data, mjm)
jax.block_until_ready(tok)
_, rgb, _ = r.render(tok, data, mjm)
jax.block_until_ready(rgb)
print("[smalltest] render ok:", rgb.shape, rgb.dtype)
PY
echo "==================================="

echo "=== Run training ==="
stdbuf -oL -eL python -u execute.py 2>&1
echo "Training completed."
'

echo "Finished at $(date)"
