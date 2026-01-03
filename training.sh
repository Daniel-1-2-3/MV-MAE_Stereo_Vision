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

# --------------------------------------------------------------------
# Probe 1: EGL + Torch (fast, catches GL issues early)
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# Probe 2 + Training
# --------------------------------------------------------------------
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

echo "=== [CONTAINER] preflight env snapshot ==="
echo "PATH=$PATH"
echo "PYTHONPATH=${PYTHONPATH:-}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
echo "LD_PRELOAD=${LD_PRELOAD:-}"
echo "JAX_PLATFORMS=${JAX_PLATFORMS:-}"
echo "XLA_FLAGS=${XLA_FLAGS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
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

# ------------------------------
# (1) JAX/jaxlib + backend string
# ------------------------------
echo "=== [CONTAINER] (1) JAX/JAXLIB + backend platform_version ==="
python - <<'"'"'PY'"'"'
import jax, jaxlib
from jax.lib import xla_bridge
print("jax   =", jax.__version__)
print("jaxlib=", jaxlib.__version__)
b = xla_bridge.get_backend()
print("backend.platform        =", b.platform)
print("backend.platform_version=", b.platform_version)
PY
echo "============================================================="

# ------------------------------
# (2) Which JAX CUDA plugin modules exist + their origins
# ------------------------------
echo "=== [CONTAINER] (2) JAX CUDA plugin module origins ==="
python - <<'"'"'PY'"'"'
import importlib.util
mods = [
  "jax_cuda12_plugin",
  "jax_cuda12_pjrt",
  "jaxlib.cuda_plugin_extension",
  "jaxlib.xla_extension",
]
for m in mods:
  s = importlib.util.find_spec(m)
  print(f"{m:28s} ->", None if s is None else s.origin)
PY
echo "======================================================"

# ------------------------------
# Madrona import + extension origins (must be consistent) + duplicate scan
# ------------------------------
echo "=== [CONTAINER] Madrona import + extension origins (must be consistent) ==="
python - <<'"'"'PY'"'"'
import os, glob, site, importlib.util

def origin(name):
    spec = importlib.util.find_spec(name)
    return None if spec is None else spec.origin

import madrona_mjx
print("madrona_mjx.__file__:", madrona_mjx.__file__)

top_br = origin("_madrona_mjx_batch_renderer")
pkg_br = origin("madrona_mjx._madrona_mjx_batch_renderer")
top_vz = origin("_madrona_mjx_visualizer")
pkg_vz = origin("madrona_mjx._madrona_mjx_visualizer")

print("top-level batch:", top_br)
print("pkg batch     :", pkg_br)
print("top-level viz :", top_vz)
print("pkg viz       :", pkg_vz)

if top_br is not None or top_vz is not None:
    raise SystemExit("FATAL: top-level _madrona_* exists -> double-load risk.")
if pkg_br is None:
    raise SystemExit("FATAL: madrona_mjx._madrona_mjx_batch_renderer not importable (None).")
if pkg_vz is None:
    raise SystemExit("FATAL: madrona_mjx._madrona_mjx_visualizer not importable (None).")

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

print("\nCHOSEN_PKG_BATCH_SO=" + str(pkg_br))
print("CHOSEN_PKG_VIZ_SO=" + str(pkg_vz))
PY
echo "=========================================================================="


# ------------------------------
# (3) ldd on the exact imported madrona .so (link targets)
# ------------------------------
echo "=== [CONTAINER] (3) ldd link targets for imported madrona .so ==="
CHOSEN_PKG_BATCH_SO="$(python - <<'"'"'PY'"'"'
import importlib.util
s = importlib.util.find_spec("madrona_mjx._madrona_mjx_batch_renderer")
print("" if s is None else s.origin)
PY
)"
CHOSEN_PKG_VIZ_SO="$(python - <<'"'"'PY'"'"'
import importlib.util
s = importlib.util.find_spec("madrona_mjx._madrona_mjx_visualizer")
print("" if s is None else s.origin)
PY
)"

echo "--- batch so: $CHOSEN_PKG_BATCH_SO"
ldd -v "$CHOSEN_PKG_BATCH_SO" | egrep "libcuda|libcudart|nvJitLink|nvrtc|libstdc\+\+|libgcc_s|libmadmjx|Version" || true

echo "--- viz so:   $CHOSEN_PKG_VIZ_SO"
ldd -v "$CHOSEN_PKG_VIZ_SO" | egrep "libcuda|libcudart|nvJitLink|nvrtc|libstdc\+\+|libgcc_s|libmadmjx|Version" || true
echo "================================================================="

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
ldd -v /opt/madrona_mjx/build/libmadmjx_mgr.so | egrep "nvJitLink|nvrtc|libcuda|libcudart|Version|libmadmjx_mgr" || true
echo "==============================================================="

# ------------------------------
# (4) DECISIVE poisoning probe:
#     baseline JAX -> Madrona render -> JAX kernel -> Torch kernel -> DLPack crossings
#     PLUS LD_PRELOAD A/B to isolate nvJitLink preload issues
# ------------------------------
echo "=== [CONTAINER] (4) DECISIVE probe: isolate Madrona vs JAX vs Torch vs DLPack ==="

run_probe () {
  local label="$1"
  shift
  echo "---- PROBE RUN: $label ----"
  "$@" python - <<'"'"'PY'"'"'
import os, importlib.util
import jax, jax.numpy as jp
import torch
from torch.utils import dlpack as tpack

B = int(os.environ.get("RENDER_PROBE_B", "32"))
DBG = os.environ.get("RENDER_PROBE_DEBUG", "0") == "1"
print("[probe] B =", B, "| debug =", DBG)

print("\n[probe] === versions ===")
print("torch:", torch.__version__)
print("jax:", jax.__version__)
import jaxlib
print("jaxlib:", jaxlib.__version__)

print("\n[probe] === baseline JAX (no Madrona) ===")
x = jax.jit(lambda: (jp.arange(4096, dtype=jp.float32) * 2).sum())()
jax.block_until_ready(x)
print("[probe] baseline JAX OK")

print("\n[probe] === baseline Torch (no Madrona) ===")
torch.empty((1024,), device="cuda", dtype=torch.float32).fill_(1.0)
torch.cuda.synchronize()
print("[probe] baseline Torch OK")

print("\n[probe] === load env + render ===")
path = "/workspace/Mujoco_Sim/pick_env.py"
spec = importlib.util.spec_from_file_location("pick_env_mod", path)
m = importlib.util.module_from_spec(spec); assert spec and spec.loader
spec.loader.exec_module(m)
StereoPickCube = getattr(m, "StereoPickCube")
env = StereoPickCube(render_batch_size=B)

keys = jax.random.split(jax.random.PRNGKey(0), B)
st_b = jax.vmap(env.reset_physics)(keys)
data_b = st_b.data

# Avoid extra smoke-render unless explicitly testing it
env._ensure_render_token(data_b, debug=DBG)
rgb = env.render_rgba(data_b)
jax.block_until_ready(rgb)
print("[probe] render OK; rgb:", rgb.shape, rgb.dtype)

print("\n[probe] === (A1) JAX kernel AFTER render on fresh array (global poison check) ===")
post = jax.jit(lambda t: t + jp.array(0, dtype=t.dtype))
fresh = jp.zeros((B, 2, 64, 64, 4), dtype=jp.uint8)
y0 = post(fresh); jax.block_until_ready(y0)
print("[probe] PASS A1")

print("\n[probe] === (A2) JAX kernel AFTER render on rgb (buffer/path check) ===")
y = post(rgb); jax.block_until_ready(y)
print("[probe] PASS A2")

print("\n[probe] === (B) Torch CUDA AFTER render (no DLPack) ===")
torch.empty((1024,), device="cuda", dtype=torch.uint8).zero_()
torch.cuda.synchronize()
print("[probe] PASS B")

print("\n[probe] === (C) DLPack JAX->Torch + clone AFTER render ===")
t = tpack.from_dlpack(rgb)          # match wrapper
t2 = t.contiguous()
t3 = t2.clone()
torch.cuda.synchronize()
print("[probe] PASS C")

print("\n[probe] === (D) DLPack Torch->JAX + JAX op ===")
tj = jax.dlpack.from_dlpack(t3)
z = jax.jit(lambda a: a[0,0,0,0,0].astype(jp.int32) + 1)(tj)
jax.block_until_ready(z)
print("[probe] PASS D")

print("\n[probe] ALL PASSED")
PY
  echo "---- END PROBE RUN: $label ----"
}

# Run 1: WITH LD_PRELOAD (current environment)
RENDER_PROBE_DEBUG=0 run_probe "WITH LD_PRELOAD, debug=0" env

# Run 2: WITHOUT LD_PRELOAD (critical A/B)
RENDER_PROBE_DEBUG=0 run_probe "NO LD_PRELOAD, debug=0" env -u LD_PRELOAD

# Optional: test the debug-smoke-render path too (your pick_env does extra work when debug=True)
RENDER_PROBE_DEBUG=1 run_probe "WITH LD_PRELOAD, debug=1" env
RENDER_PROBE_DEBUG=1 run_probe "NO LD_PRELOAD, debug=1" env -u LD_PRELOAD

echo "==========================================================================="

echo "=== [CONTAINER] starting training ==="
stdbuf -oL -eL python -u execute.py 2>&1
'

echo "Finished at $(date)"
