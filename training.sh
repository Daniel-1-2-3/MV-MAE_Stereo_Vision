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

# ---- CUDA / nvJitLink: keep consistent & explicit ----
export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/opt/madrona_mjx/build:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
# ------------------------------------------------------

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
echo "JAX_PLATFORMS=${JAX_PLATFORMS:-}"
echo "XLA_FLAGS=${XLA_FLAGS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
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

# ------------------------------
# (4) DECISIVE poisoning probe, variants:
#   - no torch import
#   - torch imported first
#   - optional LD_PRELOAD on/off
# ------------------------------
echo "=== [CONTAINER] (4) DECISIVE probe variants ==="

run_probe () {
  local TORCH_IMPORT="$1"
  local PRELOAD="$2"
  local LAUNCH_BLOCKING="$3"

  if [[ "$PRELOAD" == "1" ]]; then
    export LD_PRELOAD="/usr/local/cuda/lib64/libnvJitLink.so.12"
  else
    unset LD_PRELOAD || true
  fi

  if [[ "$LAUNCH_BLOCKING" == "1" ]]; then
    export CUDA_LAUNCH_BLOCKING=1
  else
    unset CUDA_LAUNCH_BLOCKING || true
  fi

  echo
  echo "---- PROBE RUN: torch_import=$TORCH_IMPORT preload=$PRELOAD launch_blocking=$LAUNCH_BLOCKING ----"

  TORCH_IMPORT="$TORCH_IMPORT" python - << "PY"
import os, importlib.util, ctypes, re

import jax
import jax.numpy as jp

torch_import = os.environ.get("TORCH_IMPORT","0") == "1"

def print_loaded_cuda_libs():
  pats = [
    r"/.*libcudart\.so(\.[0-9]+)*",
    r"/.*libnvrtc\.so(\.[0-9]+)*",
    r"/.*libnvJitLink\.so(\.[0-9]+)*",
    r"/.*libcuda\.so(\.[0-9]+)*",
  ]
  seen = set()
  hits = []
  with open("/proc/self/maps","r") as f:
    for line in f:
      path = line.strip().split()[-1] if line.strip() else ""
      for p in pats:
        if re.match(p, path):
          if path not in seen:
            seen.add(path)
            hits.append(path)
  print("[probe] loaded CUDA libs:")
  for h in hits:
    print("   ", h)

def cudart():
  # try common names
  for name in ("libcudart.so.12","libcudart.so"):
    try:
      return ctypes.CDLL(name)
    except Exception:
      pass
  return None

def cuda_last_error_str(lib):
  if lib is None:
    return None
  lib.cudaGetLastError.restype = ctypes.c_int
  lib.cudaGetErrorString.restype = ctypes.c_char_p
  err = lib.cudaGetLastError()
  msg = lib.cudaGetErrorString(err)
  return err, (msg.decode("utf-8","ignore") if msg else "")

def cuda_sync(lib):
  if lib is None:
    return None
  lib.cudaDeviceSynchronize.restype = ctypes.c_int
  return lib.cudaDeviceSynchronize()

print("[probe] jax:", jax.__version__)
if torch_import:
  import torch
  print("[probe] torch:", torch.__version__, "| torch.version.cuda:", getattr(torch.version,"cuda",None))
else:
  print("[probe] torch: (not imported)")

print_loaded_cuda_libs()

# Baseline JAX should work
baseline = jax.jit(lambda: (jp.arange(8192, dtype=jp.float32) * 2).sum())
jax.block_until_ready(baseline())
print("[probe] baseline JAX OK")

# Load env module without touching torch unless requested
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

lib = cudart()

# ---- Stage 1: init token only (no render) ----
env._ensure_render_token(data_b, debug=False)
print("[probe] renderer.init OK; token dtype/shape:", getattr(env, "_render_token", None).dtype, getattr(env, "_render_token", None).shape)

# If init already poisons, this will fail
try:
  jax.block_until_ready(jp.zeros((128,), dtype=jp.float32) + 1)
  print("[probe] PASS: JAX op after init OK")
except Exception as e:
  print("[probe] FAIL: JAX op after init FAILED -> poisoning happens at init")
  raise

# ---- Stage 2: render once ----
rgb = env.render_rgba(data_b)
jax.block_until_ready(rgb)
print("[probe] render OK; rgb:", getattr(rgb,"shape",None), getattr(rgb,"dtype",None))

# Check CUDA runtime error state immediately
if lib is not None:
  sync_rc = cuda_sync(lib)
  err = cuda_last_error_str(lib)
  print("[probe] cudaDeviceSynchronize rc:", sync_rc)
  print("[probe] cudaGetLastError:", err)
else:
  print("[probe] libcudart not loadable via ctypes; skipping cudaGetLastError")

# ---- Stage 3: first unrelated JAX op after render (this is your A1) ----
def try_post_jax(tag):
  try:
    jax.block_until_ready(jp.zeros((B,2,64,64,4), dtype=jp.uint8))
    print(f"[probe] PASS: post-render JAX fresh alloc OK ({tag})")
    return True
  except Exception as e:
    print(f"[probe] FAIL: post-render JAX fresh alloc FAILED ({tag}) ->", repr(e))
    return False

ok = try_post_jax("no-extra-sync")

# If it failed, try one more hard device sync + retry (detects stream/ordering bugs)
if not ok and lib is not None:
  print("[probe] retrying after extra cudaDeviceSynchronize() ...")
  cuda_sync(lib)
  # clear last error read
  _ = cuda_last_error_str(lib)
  ok2 = try_post_jax("after-extra-device-sync")
  if not ok2:
    raise SystemExit(2)
  else:
    raise SystemExit(3)  # special code: sync workaround helped

# Optional: torch kernel AFTER render (only if imported)
if torch_import:
  import torch
  torch.empty((1024,), device="cuda", dtype=torch.float32).fill_(1.0)
  torch.cuda.synchronize()
  print("[probe] PASS: Torch CUDA op after render OK")

print("[probe] ALL GOOD: no poisoning detected in this variant")
PY
}

# Run variants (separate python processes => clean isolation)
#   - If torch_import=0 still fails: Madrona/JAX/XLA side.
#   - If torch_import=0 passes but torch_import=1 fails: CUDA runtime lib collision involving torch.
set +e
run_probe 0 0 1; rc00=$?
run_probe 1 0 1; rc10=$?
run_probe 0 1 1; rc01=$?
run_probe 1 1 1; rc11=$?
set -e

echo
echo "=== [probe summary] exit codes ==="
echo "noTorch noPreload : $rc00"
echo "Torch   noPreload : $rc10"
echo "noTorch Preload   : $rc01"
echo "Torch   Preload   : $rc11"
echo "Codes: 0=OK, 2=poison persists even after extra sync, 3=extra sync fixed (stream ordering bug)"

# If any variant shows poisoning (exit 2), stop before training
if [[ "$rc00" == "2" || "$rc10" == "2" || "$rc01" == "2" || "$rc11" == "2" ]]; then
  echo "FATAL: renderer poisoning confirmed in at least one variant (exit=2). Not starting training."
  exit 20
fi

echo "=== [CONTAINER] starting training ==="
stdbuf -oL -eL python -u execute.py 2>&1
'

echo "Finished at $(date)"
