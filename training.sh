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

# Probe 1: EGL + Torch
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

# ============================================================
# CUDA/JAX/Madrona loader hardening: SYSTEM CUDA TOOLKIT ONLY
# ============================================================

export PYTHONNOUSERSITE=1

if [[ -d /usr/local/cuda-12.4 ]]; then
  export CUDA_HOME=/usr/local/cuda-12.4
else
  export CUDA_HOME=/usr/local/cuda
fi

# -------------------- ADDED: quarantine pip nvidia CUDA libs --------------------
# JAX is still pulling libcudart from venv site-packages/nvidia/* even when
# LD_LIBRARY_PATH prefers /usr/local/cuda. This block makes those libs un-loadable.
PY_NVIDIA_ROOT="$(python - << "PY"
import os, sys
for p in sys.path:
  if p and "site-packages" in p:
    n = os.path.join(p, "nvidia")
    if os.path.isdir(n):
      print(n)
      break
PY
)"
echo "PY_NVIDIA_ROOT=${PY_NVIDIA_ROOT:-}"

QUAR_DIR="/tmp/pip_cuda_quarantine"
mkdir -p "$QUAR_DIR"
RESTORE_LIST="$QUAR_DIR/restore_paths.txt"
: > "$RESTORE_LIST"

quarantine_one () {
  local d="$1"
  if [[ -d "$d" ]]; then
    local bn
    bn="$(echo "$d" | sed "s|/|_|g")"
    local dst="$QUAR_DIR/$bn"
    echo "[quarantine] moving $d -> $dst"
    mv "$d" "$dst"
    echo "$d|$dst" >> "$RESTORE_LIST"
  fi
}

# Most important offenders:
quarantine_one "$PY_NVIDIA_ROOT/cuda_runtime/lib"
quarantine_one "$PY_NVIDIA_ROOT/cuda_nvrtc/lib"
quarantine_one "$PY_NVIDIA_ROOT/nvjitlink/lib"
# Common extras (safe to quarantine if present):
quarantine_one "$PY_NVIDIA_ROOT/cublas/lib"
quarantine_one "$PY_NVIDIA_ROOT/cudnn/lib"
quarantine_one "$PY_NVIDIA_ROOT/cufft/lib"
quarantine_one "$PY_NVIDIA_ROOT/curand/lib"
quarantine_one "$PY_NVIDIA_ROOT/cusolver/lib"
quarantine_one "$PY_NVIDIA_ROOT/cusparse/lib"
quarantine_one "$PY_NVIDIA_ROOT/nccl/lib"

restore_quarantine () {
  if [[ -f "$RESTORE_LIST" ]]; then
    echo "[quarantine] restoring pip CUDA dirs..."
    while IFS="|" read -r orig dst; do
      if [[ -d "$dst" ]]; then
        mkdir -p "$(dirname "$orig")"
        mv "$dst" "$orig"
        echo "[quarantine] restored $orig"
      fi
    done < "$RESTORE_LIST"
  fi
}
trap restore_quarantine EXIT
# ------------------------------------------------------------------------------

unset LD_PRELOAD
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:/opt/madrona_mjx/build:/.singularity.d/libs:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
echo "USING SYSTEM CUDA TOOLKIT LIBS (pip nvidia/* CUDA dirs quarantined)"

export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"

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
echo "PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-}"
echo "CUDA_HOME=${CUDA_HOME:-}"
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
python - << "PY"
import jax
print("jax:", jax.__version__)
print("devices:", jax.devices())
PY
echo "==================================="

echo "=== [CONTAINER] Madrona .so origins ==="
python - << "PY"
import importlib.util
def origin(name):
  s = importlib.util.find_spec(name)
  return None if s is None else s.origin
print("madrona_mjx._madrona_mjx_batch_renderer ->", origin("madrona_mjx._madrona_mjx_batch_renderer"))
print("madrona_mjx._madrona_mjx_visualizer     ->", origin("madrona_mjx._madrona_mjx_visualizer"))
print("_madrona_mjx_batch_renderer (top-level) ->", origin("_madrona_mjx_batch_renderer"))
print("_madrona_mjx_visualizer (top-level)     ->", origin("_madrona_mjx_visualizer"))
PY
echo "======================================"

echo "=== [CONTAINER] loader sanity (maps, system-only) ==="
python - << "PY"
import re, sys
import jax
import jax.numpy as jnp

jax.block_until_ready(jax.jit(lambda: (jnp.arange(8192, dtype=jnp.float32) * 2).sum())())

pats = [
  r".*libcudart\.so(\.[0-9]+)*$",
  r".*libnvrtc\.so(\.[0-9]+)*$",
  r".*libnvJitLink\.so(\.[0-9]+)*$",
  r".*libcuda\.so(\.[0-9]+)*$",
]
hits=[]
with open("/proc/self/maps","r") as f:
  for line in f:
    parts=line.strip().split()
    if not parts: continue
    path=parts[-1]
    if any(re.match(p, path) for p in pats):
      hits.append(path)

uniq=[]
seen=set()
for h in hits:
  if h not in seen:
    seen.add(h); uniq.append(h)

print("CUDA libs in this process:")
for h in uniq:
  print("  ", h)

bad=[h for h in uniq if "/site-packages/nvidia/" in h and "libcuda.so" not in h]
if bad:
  print("FATAL: pip nvidia/* CUDA user-mode libs loaded, but this run is configured for system CUDA toolkit libs only:")
  for b in bad:
    print("  ", b)
  sys.exit(99)

print("OK: system CUDA toolkit libs only (pip nvidia/* not loaded).")
PY
echo "========================================"

echo "=== [CONTAINER] (4) DECISIVE probe variants ==="

run_probe () {
  local TORCH_IMPORT="$1"

  echo
  echo "---- PROBE RUN: torch_import=$TORCH_IMPORT ----"

  TORCH_IMPORT="$TORCH_IMPORT" python - << "PY"
import os, importlib.util, ctypes, re, sys
import jax
import jax.numpy as jp

torch_import = os.environ.get("TORCH_IMPORT","0") == "1"

def dump_loaded_cuda_libs():
  pats = [
    r".*libcudart\.so(\.[0-9]+)*$",
    r".*libnvrtc\.so(\.[0-9]+)*$",
    r".*libnvJitLink\.so(\.[0-9]+)*$",
    r".*libcuda\.so(\.[0-9]+)*$",
  ]
  seen = set()
  hits = []
  with open("/proc/self/maps","r") as f:
    for line in f:
      parts = line.strip().split()
      if not parts:
        continue
      path = parts[-1]
      for p in pats:
        if re.match(p, path):
          if path not in seen:
            seen.add(path)
            hits.append(path)
  print("[probe] /proc/self/maps CUDA libs:")
  for h in hits:
    print("   ", h)

def load_cudart():
  for name in ("libcudart.so.12","libcudart.so"):
    try:
      return ctypes.CDLL(name)
    except Exception:
      pass
  return None

def cuda_sync_and_last_error(lib):
  if lib is None:
    return None
  lib.cudaDeviceSynchronize.restype = ctypes.c_int
  lib.cudaGetLastError.restype = ctypes.c_int
  lib.cudaGetErrorString.restype = ctypes.c_char_p
  rc = lib.cudaDeviceSynchronize()
  err = lib.cudaGetLastError()
  msg = lib.cudaGetErrorString(err)
  msg = msg.decode("utf-8","ignore") if msg else ""
  return rc, err, msg

print("[probe] jax:", jax.__version__)
if torch_import:
  import torch
  print("[probe] torch:", torch.__version__, "| torch.version.cuda:", getattr(torch.version,"cuda",None))
else:
  print("[probe] torch: (not imported)")

dump_loaded_cuda_libs()

jax.block_until_ready(jax.jit(lambda: (jp.arange(8192, dtype=jp.float32) * 2).sum())())
print("[probe] baseline JAX OK")

path = "/workspace/Mujoco_Sim/pick_env.py"
spec = importlib.util.spec_from_file_location("pick_env_mod", path)
m = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(m)
StereoPickCube = getattr(m, "StereoPickCube")

B = 32
env = StereoPickCube(render_batch_size=B)

keys = jax.random.split(jax.random.PRNGKey(0), B)
st_b = jax.vmap(env.reset_physics)(keys)
data_b = st_b.data

lib = load_cudart()

env._ensure_render_token(data_b, debug=False)
print("[probe] renderer.init OK")

try:
  jax.block_until_ready(jp.zeros((128,), dtype=jp.float32) + 1)
  print("[probe] PASS: JAX op after init OK")
except Exception as e:
  print("[probe] FAIL: JAX op after init FAILED -> poisoning at init:", repr(e))
  sys.exit(12)

rgb = env.render_rgba(data_b)
jax.block_until_ready(rgb)
print("[probe] render OK; rgb:", getattr(rgb,"shape",None), getattr(rgb,"dtype",None))

triple = cuda_sync_and_last_error(lib)
if triple is None:
  print("[probe] libcudart ctypes load failed; skipping cudaDeviceSynchronize/cudaGetLastError")
else:
  rc, err, msg = triple
  print("[probe] cudaDeviceSynchronize rc:", rc)
  print("[probe] cudaGetLastError:", (err, msg))

try:
  jax.block_until_ready(jp.zeros((B,2,64,64,4), dtype=jp.uint8))
  print("[probe] PASS: post-render fresh JAX alloc OK")
except Exception as e:
  print("[probe] FAIL: post-render fresh JAX alloc FAILED -> poisoning after render:", repr(e))
  sys.exit(13)

if torch_import:
  import torch
  torch.empty((1024,), device="cuda", dtype=torch.float32).fill_(1.0)
  torch.cuda.synchronize()
  print("[probe] PASS: Torch CUDA op after render OK")

print("[probe] ALL GOOD in this variant")
PY
}

set +e
run_probe 0; rc0=$?
run_probe 1; rc1=$?
set -e

echo
echo "=== [probe summary] exit codes ==="
echo "noTorch : $rc0"
echo "Torch   : $rc1"
echo "Codes: 0=OK, 12=poison at init, 13=poison after render"

if [[ "$rc0" != "0" || "$rc1" != "0" ]]; then
  echo "FATAL: decisive probe failed (see above). Not starting training."
  exit 20
fi

echo "=== [CONTAINER] starting training ==="
stdbuf -oL -eL python -u execute.py 2>&1
'

echo "Finished at $(date)"
