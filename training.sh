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

# ============================================================
# 0) Paths / image
# ============================================================
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

# Silence Apptainer PYTHONPATH forwarding warnings:
# (we explicitly set APPTAINERENV_PYTHONPATH below)
unset PYTHONPATH || true

# Prefer host-mounted code under /workspace so edits require no SIF rebuild.
export APPTAINERENV_PYTHONPATH="/workspace:/opt/src:/opt/src/MV_MAE_Implementation"

# ============================================================
# 1) EGL / MuJoCo GL setup (working settings)
# ============================================================
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

# ============================================================
# 2) Bind mounts (working layout)
# ============================================================
# mujoco_playground external_deps fix (site-packages is read-only)
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
# 3) Host snapshot (useful when debugging)
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
# 4) Quick container probe (GPU + EGL sanity)
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
echo "---- EGL sanity (MuJoCo + OpenGL) ----"
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
# 5) Run training inside container
#    - Madrona caches: JOB-LOCAL + UNIQUE filenames
#    - Keep your existing JAX flags (from working script)
#    - Add a minimal smoke test (B=1) before full training
# ============================================================
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  "$IMG" \
  bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1

# ---------- JAX / XLA tuning (keep from your working script) ----------
export JAX_TRACEBACK_FILTERING=off
export JAX_DISABLE_CUSOLVER=1
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=.60

echo "=== MuJoCo version ==="
python - <<'"'"'PY'"'"'
import mujoco
print("MuJoCo version:", mujoco.__version__)
PY
echo "======================"
echo

# ---------- Madrona cache: node-local + unique files to avoid deadlocks ----------
echo "=== Madrona + GPU detection (inside container) ==="
ACTUAL_GPU="$(nvidia-smi -L 2>/dev/null | head -1 || true)"
echo "Actual GPU: $ACTUAL_GPU"
GPU_MODEL="$(echo "$ACTUAL_GPU" | grep -o "H100\|L40S\|A100\|V100\|RTX" | head -1 || true)"
GPU_MODEL="${GPU_MODEL:-unknown}"
GPU_MODEL_LOWER="$(echo "$GPU_MODEL" | tr "[:upper:]" "[:lower:]")"
ENV_CONFIG="default"

CACHE_ROOT="${SLURM_TMPDIR:-/tmp}"
CACHE_BUILD_DIR="${CACHE_ROOT}/madrona_cache_${GPU_MODEL_LOWER}_${ENV_CONFIG}_${SLURM_JOB_ID}"
mkdir -p "$CACHE_BUILD_DIR/kernel_cache" "$CACHE_BUILD_DIR/bvh_cache"

export MADRONA_MWGPU_KERNEL_CACHE="$CACHE_BUILD_DIR/kernel_cache/kernel_${SLURM_JOB_ID}.cache"
export MADRONA_BVH_KERNEL_CACHE="$CACHE_BUILD_DIR/bvh_cache/bvh_${SLURM_JOB_ID}.cache"

echo "Madrona cache configuration:"
echo "  CACHE_ROOT      = $CACHE_ROOT"
echo "  GPU_MODEL_LOWER = $GPU_MODEL_LOWER"
echo "  ENV_CONFIG      = $ENV_CONFIG"
echo "  MADRONA_MWGPU_KERNEL_CACHE = $MADRONA_MWGPU_KERNEL_CACHE"
echo "  MADRONA_BVH_KERNEL_CACHE   = $MADRONA_BVH_KERNEL_CACHE"
ls -lah "$CACHE_BUILD_DIR" || true
echo

echo "========================================="
echo "Starting MV-MAE training with MJX + Madrona"
echo "Raytracer is expected; first run will compile."
echo "========================================="

# ---------- Runtime Python deps (persist on host via $SLURM_SUBMIT_DIR) ----------
DEPS_PREFIX="'"$SLURM_SUBMIT_DIR"'/.pydeps_prefix"
PY_MM=$(python - <<'"'"'PY'"'"'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
SITE_PKGS="${DEPS_PREFIX}/lib/python${PY_MM}/site-packages"
BIN_DIR="${DEPS_PREFIX}/bin"
mkdir -p "$DEPS_PREFIX"

# Make prefix visible for imports + CLI entrypoints (keep /workspace first)
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

python -c "import jax, jaxlib; print('jax', jax.__version__); print('jaxlib', jaxlib.__version__)"
python -c "import madrona_mjx; print('madrona_mjx', madrona_mjx.__file__)"
echo "============================================"
echo

# ---------- Minimal env smoke test (B=1) to ensure Madrona init does not wedge ----------
# Only enable CUDA_LAUNCH_BLOCKING for this tiny smoke step (better tracebacks without wedging init)
echo "=========== SMOKE TEST (B=1) ==========="
SMOKE_B=1 CUDA_LAUNCH_BLOCKING=1 stdbuf -oL -eL python -u - <<'"'"'PY'"'"'
import os
import jax
from Mujoco_Sim.pick_env import StereoPickCube

B = int(os.environ.get("SMOKE_B", "1"))
print("SMOKE_B =", B)

env = StereoPickCube(config_overrides={
    "vision_config.gpu_id": 0,
    "vision_config.use_rasterizer": False,  # raytracer
    "vision_config.enabled_geom_groups": [0, 1, 2],
})

key = jax.random.PRNGKey(0)
keys = jax.random.split(key, B)

print("Calling env.reset(keys)...")
st = env.reset(keys)

# Force sync so any GPU init error surfaces at the true op
jax.block_until_ready(st)

print("Reset OK.")
print("obs shape:", getattr(st.obs, "shape", None), "dtype:", getattr(st.obs, "dtype", None))
print("info keys:", sorted(list(st.info.keys())))

if "render_token" not in st.info:
    raise RuntimeError("Renderer did not run during reset: no render_token in state.info")
print("render_token present; type:", type(st.info["render_token"]))
jax.block_until_ready(st.info["render_token"])
print("render_token sync OK")
PY
echo "=========== SMOKE TEST PASSED ==========="
echo

# ---------- Run the real training ----------
stdbuf -oL -eL python -u execute.py 2>&1

echo "Training completed."
'

echo "Finished at $(date)"
