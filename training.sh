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
# 0) HOST SNAPSHOT (node, GPU, driver)
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
# 1) CONTAINER PROBE (GPU, EGL, Python imports, JAX backend)
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

echo "---- Python + imports (paths only) ----"
python -V
python - << "PY"
import sys, platform
print("python:", sys.version)
print("platform:", platform.platform())

def imp(name):
    try:
        m = __import__(name)
        print(f"{name}: OK  file={getattr(m, '__file__', None)}")
        return m
    except Exception as e:
        print(f"{name}: FAIL {e!r}")
        return None

torch = imp("torch")
jax   = imp("jax")
jaxlib= imp("jaxlib")
muj   = imp("mujoco")
mm    = imp("madrona_mjx")

if torch:
    try:
        import torch as T
        print("torch.cuda.is_available():", T.cuda.is_available())
        if T.cuda.is_available():
            print("torch GPU:", T.cuda.get_device_name(0))
            print("torch.version.cuda:", T.version.cuda)
    except Exception as e:
        print("torch cuda query FAIL:", e)

if jax:
    try:
        import jax as J
        print("jax.__version__:", J.__version__)
        print("jax.devices():", J.devices())
        print("jax.default_backend():", J.default_backend())
    except Exception as e:
        print("jax backend query FAIL:", e)

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
# 2) ABI / LDD CHECKS
#    - checks jaxlib xla_extension.so
#    - finds and checks ANY madrona-related .so on disk
# ============================================================
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  "$IMG" \
  bash -lc '
set -euo pipefail

echo "================ ABI / LDD CHECKS ================"

echo "---- jaxlib xla_extension ----"
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

echo "---- Find madrona-related shared libs ----"
# 1) under /opt/madrona_mjx (common for your setup)
find /opt/madrona_mjx -maxdepth 6 -name "*.so" -print 2>/dev/null || true
echo

# 2) under site-packages (if installed as wheels/egg)
python - <<'"'"'PY'"'"'
import site, glob
paths = site.getsitepackages() + [site.getusersitepackages()]
cands = []
for p in paths:
    cands += glob.glob(p + "/**/*madrona*.so", recursive=True)
    cands += glob.glob(p + "/**/*mjx*.so", recursive=True)
print("\n".join(sorted(set(cands))) if cands else "No madrona*.so or *mjx*.so found under site-packages")
PY
echo

echo "---- ldd on ALL discovered madrona/mjx .so ----"
# Collect candidates (keep it simple)
CANDS=$( (find /opt/madrona_mjx -maxdepth 6 -name "*.so" -print 2>/dev/null || true) ; \
         (python - <<'"'"'PY'"'"'
import site, glob
paths = site.getsitepackages() + [site.getusersitepackages()]
cands = []
for p in paths:
    cands += glob.glob(p + "/**/*madrona*.so", recursive=True)
    cands += glob.glob(p + "/**/*mjx*.so", recursive=True)
print("\n".join(sorted(set(cands))))
PY
) | awk "NF" | sort -u )

if [ -z "$CANDS" ]; then
  echo "WARNING: no madrona/mjx .so candidates found to ldd."
else
  while IFS= read -r so; do
    echo "----- $so -----"
    ldd "$so" | egrep -i "not found|libcuda|libcudart|libnvrtc|libnvidia|libstdc\+\+|libgcc_s|GLIBC|GLIBCXX|CXXABI" || true
  done <<< "$CANDS"
fi
echo

echo "---- CUDA library resolution (ctypes) ----"
python - <<'"'"'PY'"'"'
import ctypes.util
for name in ["cuda", "cudart", "nvrtc"]:
    print(name, "->", ctypes.util.find_library(name))
PY

echo "=================================================="
'

# ============================================================
# 3) MADRONA CACHE + JAX FLAGS + MINIMAL REPRO
#    - forces sync to surface the real failing op
#    - verifies renderer actually ran (render_token in info)
# ============================================================
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  "$IMG" \
  bash -lc '
set -euo pipefail

echo "================ MADRONA / JAX SETUP ================"
export PYTHONUNBUFFERED=1
export JAX_TRACEBACK_FILTERING=off

# Make GPU failures surface at the correct line (debug only; slows down)
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

echo "================ MINIMAL ENV SMOKE TEST ================"
python - <<'"'"'PY'"'"'
import os
import jax

from Mujoco_Sim.pick_env import StereoPickCube

B = int(os.environ.get("SMOKE_B", "128"))
print("SMOKE_B =", B)

# Only override keys that exist in your schema
env = StereoPickCube(config_overrides={
    "vision_config.gpu_id": 0,
    "vision_config.use_rasterizer": False,
    "vision_config.enabled_geom_groups": [0, 1, 2],
})

key = jax.random.PRNGKey(0)
keys = jax.random.split(key, B)

print("Calling env.reset(keys)...")
st = env.reset(keys)

# Force sync so any GPU error surfaces at the true op
jax.block_until_ready(st)

print("Reset OK.")
print("obs shape:", getattr(st.obs, "shape", None), "dtype:", getattr(st.obs, "dtype", None))
print("info keys:", sorted(list(st.info.keys())))

# Make sure renderer actually ran; adjust key if your env uses a different name
if "render_token" not in st.info:
    raise RuntimeError("Renderer did not run during reset: no 'render_token' in state.info")

print("render_token type:", type(st.info["render_token"]))
print("render_token ready sync...")
jax.block_until_ready(st.info["render_token"])
print("render_token sync OK")
PY
echo "========================================================"
echo
'

# ============================================================
# 4) RUN TRAINING (unchanged from your original, but keep it after smoke test)
# ============================================================
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  "$IMG" \
  bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1

# Keep helpful traceback + allocator behavior in real run too
export JAX_TRACEBACK_FILTERING=off
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=.60

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
