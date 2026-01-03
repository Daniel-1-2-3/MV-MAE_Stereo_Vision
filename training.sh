#!/bin/bash
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

# Probe 1: EGL + Torch (fast, catches GL issues early)
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

# Probe 2 + Training
apptainer exec --nv "${BIND_FLAGS[@]}" --pwd "$WORKDIR_IN_CONTAINER" "$IMG" bash -lc '
set -euo pipefail
. /opt/mvmae_venv/bin/activate

# ---- CUDA / nvJitLink: make dynamic linking unambiguous ----
export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/opt/madrona_mjx/build:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/usr/local/cuda/lib64/libnvJitLink.so.12"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
# -----------------------------------------------------------

# JAX runtime knobs
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=.60
export PYTHONUNBUFFERED=1
export JAX_TRACEBACK_FILTERING=off
export PICK_ENV_DEBUG=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"

# Optional: turn on a single sync after render inside your wrapper (NOT launch blocking)
# (only useful if your Python code reads this env var)
export RENDER_SYNC="${RENDER_SYNC:-0}"

echo "=== [CONTAINER] preflight env snapshot ==="
echo "PATH=$PATH"
echo "PYTHONPATH=${PYTHONPATH:-}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
echo "LD_PRELOAD=${LD_PRELOAD:-}"
echo "JAX_PLATFORMS=${JAX_PLATFORMS:-}"
echo "XLA_FLAGS=${XLA_FLAGS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "RENDER_SYNC=${RENDER_SYNC:-}"
echo "=========================================="

echo "=== [CONTAINER] nvidia-smi ==="
nvidia-smi -L || true
nvidia-smi || true
echo "============================="

# Cache paths (good for Madrona compile)
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

echo "=== [CONTAINER] JAX device check ==="
python - <<'"'"'PY'"'"'
import jax
print("jax:", jax.__version__)
print("devices:", jax.devices())
PY
echo "==================================="

echo "=== [CONTAINER] JAX/JAXLIB/plugin version probe ==="
python - <<'"'"'PY'"'"'
import jax, jaxlib
print("jax   =", jax.__version__)
print("jaxlib=", jaxlib.__version__)
try:
    import jax_cuda12_plugin, jax_cuda12_pjrt
    print("jax_cuda12_plugin =", jax_cuda12_plugin.__version__)
    print("jax_cuda12_pjrt   =", jax_cuda12_pjrt.__version__)
except Exception as e:
    print("cuda12 plugin/pjrt not importable:", e)

print("\nENV INFO:")
try:
    jax.print_environment_info()
except Exception as e:
    print("print_environment_info failed:", e)
PY
echo "===================================================="

echo "=== [CONTAINER] Madrona import + extension origins (must be consistent) ==="
python - <<'"'"'PY'"'"'
import os, sys, glob, site, importlib.util

def origin(name):
    spec = importlib.util.find_spec(name)
    return None if spec is None else spec.origin

# Show base package origin (helps catch “wrong madrona_mjx” on sys.path)
try:
    import madrona_mjx
    print("madrona_mjx.__file__:", madrona_mjx.__file__)
except Exception as e:
    print("FATAL: import madrona_mjx failed:", e)
    raise SystemExit(90)

# Extension module origins
top_br = origin("_madrona_mjx_batch_renderer")
pkg_br = origin("madrona_mjx._madrona_mjx_batch_renderer")
top_vz = origin("_madrona_mjx_visualizer")
pkg_vz = origin("madrona_mjx._madrona_mjx_visualizer")

print("top-level batch:", top_br)
print("pkg batch     :", pkg_br)
print("top-level viz :", top_vz)
print("pkg viz       :", pkg_vz)

# Hard fail if any top-level extension exists (double-load risk)
if top_br is not None or top_vz is not None:
    raise SystemExit("FATAL: top-level _madrona_* exists -> double-load risk. Remove stray .so from site-packages.")

# Hard fail if package extension missing (your current symptom)
if pkg_br is None:
    raise SystemExit("FATAL: madrona_mjx._madrona_mjx_batch_renderer is NOT importable (pkg_br=None). Wrong install or missing .so.")
if pkg_vz is None:
    raise SystemExit("FATAL: madrona_mjx._madrona_mjx_visualizer is NOT importable (pkg_vz=None). Wrong install or missing .so.")

# Look for duplicates on disk (common cause of “None” or mismatched imports)
roots = [r for r in set(site.getsitepackages() + [site.getusersitepackages()]) if r and os.path.isdir(r)]
pat_batch = "*_madrona_mjx_batch_renderer*.so"
pat_viz   = "*_madrona_mjx_visualizer*.so"
hits_batch, hits_viz = [], []
for r in roots:
    hits_batch += glob.glob(os.path.join(r, "**", pat_batch), recursive=True)
    hits_viz   += glob.glob(os.path.join(r, "**", pat_viz), recursive=True)

print("\nDisk scan (site-packages) hits:")
print("batch hits:", len(hits_batch))
for h in hits_batch: print("  ", h)
print("viz hits  :", len(hits_viz))
for h in hits_viz: print("  ", h)

# Warn/fail if duplicates exist
if len(hits_batch) > 1:
    raise SystemExit("FATAL: multiple batch_renderer .so files found in site-packages -> ambiguous/double-load.")
if len(hits_viz) > 1:
    raise SystemExit("FATAL: multiple visualizer .so files found in site-packages -> ambiguous/double-load.")

# Also search for any top-level _madrona_* .so (these are especially bad)
pat_top = "_madrona_*.so"
hits_top = []
for r in roots:
    hits_top += glob.glob(os.path.join(r, "**", pat_top), recursive=True)
hits_top = [h for h in hits_top if "madrona_mjx" not in os.path.basename(h)] + [h for h in hits_top if os.path.basename(h).startswith("_madrona_")]
hits_top = sorted(set(hits_top))
if hits_top:
    print("\nTop-level _madrona_*.so present on disk (DANGER):")
    for h in hits_top: print("  ", h)
    raise SystemExit("FATAL: top-level _madrona_*.so exists on disk -> remove it.")

# Print the chosen origins so bash can capture them if needed
print("\nCHOSEN_PKG_BATCH_SO=" + str(pkg_br))
print("CHOSEN_PKG_VIZ_SO=" + str(pkg_vz))
PY
echo "=========================================================================="

# Extract chosen .so paths for ldd checks
CHOSEN_PKG_BATCH_SO="$(python - <<'"'"'PY'"'"'
import importlib.util
spec = importlib.util.find_spec("madrona_mjx._madrona_mjx_batch_renderer")
print("" if spec is None else spec.origin)
PY
)"
CHOSEN_PKG_VIZ_SO="$(python - <<'"'"'PY'"'"'
import importlib.util
spec = importlib.util.find_spec("madrona_mjx._madrona_mjx_visualizer")
print("" if spec is None else spec.origin)
PY
)"

echo "=== [CONTAINER] ldd check: extension .so linkage ==="
if [[ -n "$CHOSEN_PKG_BATCH_SO" && -f "$CHOSEN_PKG_BATCH_SO" ]]; then
  echo "--- ldd batch_renderer: $CHOSEN_PKG_BATCH_SO"
  ldd -v "$CHOSEN_PKG_BATCH_SO" | grep -E "libcuda|libcudart|nvJitLink|libstdc\+\+|libgcc_s|libmadmjx|Version" || true
else
  echo "FATAL: batch_renderer .so path missing -> cannot ldd"; exit 91
fi

if [[ -n "$CHOSEN_PKG_VIZ_SO" && -f "$CHOSEN_PKG_VIZ_SO" ]]; then
  echo "--- ldd visualizer: $CHOSEN_PKG_VIZ_SO"
  ldd -v "$CHOSEN_PKG_VIZ_SO" | grep -E "libcuda|libcudart|nvJitLink|libstdc\+\+|libgcc_s|libmadmjx|Version" || true
else
  echo "FATAL: visualizer .so path missing -> cannot ldd"; exit 92
fi
echo "===================================================="

echo "=== [CONTAINER] nvJitLink symbol check ==="
python - <<'"'"'PY'"'"'
import ctypes
p = "/usr/local/cuda/lib64/libnvJitLink.so.12"
lib = ctypes.CDLL(p)
print("loaded:", p)
print("has __nvJitLinkCreate_12_5:", hasattr(lib, "__nvJitLinkCreate_12_5"))
PY
echo "========================================="

echo "=== [CONTAINER] ldd check for what libmadmjx_mgr.so actually binds ==="
ldd -v /opt/madrona_mjx/build/libmadmjx_mgr.so | grep -E "nvJitLink|Version|libmadmjx_mgr|libcuda|libcudart" || true
echo "==============================================================="

echo "=== [CONTAINER] starting training ==="
stdbuf -oL -eL python -u execute.py 2>&1
'

echo "Finished at $(date)"
