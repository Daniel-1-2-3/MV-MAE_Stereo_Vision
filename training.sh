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

# ------------------------------------------------------------
# Probe 1: EGL + Torch (unchanged)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Probe 2 + Training
# ------------------------------------------------------------
apptainer exec --nv "${BIND_FLAGS[@]}" --pwd "$WORKDIR_IN_CONTAINER" "$IMG" bash -lc '
set -euo pipefail
. /opt/mvmae_venv/bin/activate

export PYTHONNOUSERSITE=1
export PICK_ENV_DEBUG=1
export PYTHONUNBUFFERED=1
export JAX_TRACEBACK_FILTERING=off
unset LD_PRELOAD

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

export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=.60
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"

echo "=== [CONTAINER] preflight env snapshot ==="
echo "PATH=$PATH"
echo "PYTHONPATH=${PYTHONPATH:-}"
echo "PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-}"
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "SYSTEM_CUDA_LD=$SYSTEM_CUDA_LD"
echo "PIP_CUDA_LD=$PIP_CUDA_LD"
echo "LD_PRELOAD=${LD_PRELOAD:-}"
echo "JAX_PLATFORMS=${JAX_PLATFORMS:-}"
echo "XLA_FLAGS=${XLA_FLAGS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "=========================================="

echo "=== [CONTAINER] nvidia-smi ==="
nvidia-smi -L || true
nvidia-smi || true
echo "============================="

# Madrona kernel cache dirs (same as before)
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

echo "=== [CONTAINER] Madrona .so origins (importlib only) ==="
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

# ------------------------------------------------------------
# Helper: loader-only JAX init to record which CUDA libs load
# ------------------------------------------------------------
loader_probe () {
  local MODE="$1"
  echo
  echo "=== [LOADER PROBE] mode=$MODE ==="

  if [[ "$MODE" == "system" ]]; then
    export LD_LIBRARY_PATH="$SYSTEM_CUDA_LD"
  else
    export LD_LIBRARY_PATH="${PIP_CUDA_LD}${PIP_CUDA_LD:+:}$SYSTEM_CUDA_LD"
  fi

  python - << "PY"
import os, re
import jax, jax.numpy as jnp
print("LD_LIBRARY_PATH=", os.environ.get("LD_LIBRARY_PATH",""))
jax.block_until_ready(jax.jit(lambda: (jnp.arange(4096, dtype=jnp.float32) * 3).sum())())
pats=[r".*libcudart\.so(\.[0-9]+)*$", r".*libnvrtc\.so(\.[0-9]+)*$", r".*libnvJitLink\.so(\.[0-9]+)*$", r".*libcuda\.so(\.[0-9]+)*$"]
seen=set(); uniq=[]
with open("/proc/self/maps","r") as f:
  for line in f:
    parts=line.strip().split()
    if not parts: continue
    path=parts[-1]
    if any(re.match(p, path) for p in pats):
      if path not in seen:
        seen.add(path); uniq.append(path)
print("CUDA libs in this process:")
for h in uniq: print("  ", h)
pip=[h for h in uniq if "/site-packages/nvidia/" in h]
syscuda=[h for h in uniq if "/usr/local/cuda" in h]
print("pip_cuda_count:", len(pip))
print("sys_cuda_count:", len(syscuda))
PY
}

loader_probe system
loader_probe pip

# ------------------------------------------------------------
# Decisive probe that mirrors YOUR wrapper + failing line
#   render -> block_until_ready -> (rgb+0) -> device_put -> dlpack -> torch ops
#
# It also tests CUDA error state via:
#   - cudaPeekAtLastError() : does NOT clear
#   - cudaGetLastError()    : clears
#
# And it can run multi-frame loops to catch "poison after N renders".
# ------------------------------------------------------------

echo
echo "=== [CONTAINER] DECISIVE probes: render->consume(rgb)->dlpack ==="

run_decisive () {
  local LD_MODE="$1"        # system | pip
  local TORCH_IMPORT="$2"   # 0 | 1  (env imports torch anyway, but we keep symmetry)
  local CLEAR_MODE="$3"     # peek | get
  local LOOPS="$4"          # number of renders/consumes

  echo
  echo "---- DECISIVE: ld=$LD_MODE torch_import=$TORCH_IMPORT clear=$CLEAR_MODE loops=$LOOPS ----"

  if [[ "$LD_MODE" == "system" ]]; then
    export LD_LIBRARY_PATH="$SYSTEM_CUDA_LD"
  else
    export LD_LIBRARY_PATH="${PIP_CUDA_LD}${PIP_CUDA_LD:+:}$SYSTEM_CUDA_LD"
  fi

  TORCH_IMPORT="$TORCH_IMPORT" CLEAR_MODE="$CLEAR_MODE" LOOPS="$LOOPS" python - << "PY"
import os, re, sys, ctypes, importlib.util
import jax
import jax.numpy as jp

torch_import = os.environ.get("TORCH_IMPORT","0") == "1"
clear_mode   = os.environ.get("CLEAR_MODE","peek")
loops        = int(os.environ.get("LOOPS","1"))

print("[probe] jax:", jax.__version__)
print("[probe] LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH",""))

# --- (A) Torch import control (for symmetry) ---
if torch_import:
  import torch
  print("[probe] torch:", torch.__version__, "| torch.version.cuda:", getattr(torch.version,"cuda",None))
else:
  print("[probe] torch: (not imported yet)")

# --- (B) dump loaded CUDA libs ---
def dump_loaded_cuda_libs(tag):
  pats=[r".*libcudart\.so(\.[0-9]+)*$", r".*libnvrtc\.so(\.[0-9]+)*$", r".*libnvJitLink\.so(\.[0-9]+)*$", r".*libcuda\.so(\.[0-9]+)*$"]
  seen=set(); uniq=[]
  with open("/proc/self/maps","r") as f:
    for line in f:
      parts=line.strip().split()
      if not parts: continue
      path=parts[-1]
      if any(re.match(p, path) for p in pats):
        if path not in seen:
          seen.add(path); uniq.append(path)
  print(f"[probe] CUDA libs ({tag}):")
  for h in uniq: print("   ", h)

dump_loaded_cuda_libs("start")

# --- (C) load libcudart and implement peek/get last error ---
def load_cudart():
  for name in ("libcudart.so.12","libcudart.so"):
    try:
      return ctypes.CDLL(name)
    except Exception:
      pass
  return None

lib = load_cudart()
if lib is None:
  print("[probe] WARN: could not ctypes-load libcudart; CUDA error probing disabled")
else:
  lib.cudaPeekAtLastError.restype = ctypes.c_int
  lib.cudaGetLastError.restype    = ctypes.c_int
  lib.cudaGetErrorString.restype  = ctypes.c_char_p

def cuda_err(where):
  if lib is None:
    return
  if clear_mode == "get":
    err = lib.cudaGetLastError()
  else:
    err = lib.cudaPeekAtLastError()
  msg = lib.cudaGetErrorString(err)
  msg = msg.decode("utf-8","ignore") if msg else ""
  print(f"[probe] {where}: cuda{('Get' if clear_mode=='get' else 'Peek')}AtLastError ->", (err, msg))

# --- (D) baseline JAX ---
jax.block_until_ready(jax.jit(lambda: (jp.arange(8192, dtype=jp.float32) * 2).sum())())
print("[probe] baseline JAX OK")
dump_loaded_cuda_libs("after baseline JAX")
cuda_err("after baseline JAX")

# --- (E) import env exactly like you do ---
path="/workspace/Mujoco_Sim/pick_env.py"
spec=importlib.util.spec_from_file_location("pick_env_mod", path)
m=importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(m)
StereoPickCube=getattr(m,"StereoPickCube")

B=32
env=StereoPickCube(render_batch_size=B)

# batched reset_physics like wrapper
keys=jax.random.split(jax.random.PRNGKey(0), B)
st_b=jax.vmap(env.reset_physics)(keys)
data_b=st_b.data

# ensure token (init)
cuda_err("pre _ensure_render_token")
env._ensure_render_token(data_b, debug=False)
print("[probe] renderer.init OK")
cuda_err("post _ensure_render_token")

# --- (F) define the exact operations your wrapper performs ---
def rebuffer_xla(rgb):
  z = jp.array(0, dtype=rgb.dtype)
  return rgb + z

def jax_device_put(rgb):
  return jax.device_put(rgb)

def dlpack_to_torch(rgb2):
  import torch
  from torch.utils import dlpack as tpack
  # your _jax_to_torch: capsule path
  return tpack.from_dlpack(jax.dlpack.to_dlpack(rgb2))

def torch_pack_stereo(rgb_t):
  import torch
  left  = rgb_t[:, 0, :, :, :3]
  right = rgb_t[:, 1, :, :, :3]
  B, H, W, C = left.shape
  obs_t = torch.empty((B, H, 2*W, C), device=rgb_t.device, dtype=left.dtype)
  obs_t[:, :, :W, :].copy_(left)
  obs_t[:, :,  W:, :].copy_(right)
  return obs_t

# --- (G) loop renders to catch "poison after N frames" ---
for i in range(loops):
  print(f"[probe] === iter {i} ===")
  cuda_err("pre render_rgba")
  rgb = env.render_rgba(data_b)
  jax.block_until_ready(rgb)
  print("[probe] render_rgba OK:", getattr(rgb,"shape",None), getattr(rgb,"dtype",None))
  cuda_err("post render_rgba")

  # 1) jax.device_put(rgb)
  try:
    rgb_dp = jax_device_put(rgb)
    jax.block_until_ready(rgb_dp)
    print("[probe] device_put(rgb) OK")
  except Exception as e:
    print("[probe] FAIL: device_put(rgb) ->", repr(e))
    cuda_err("after FAIL device_put(rgb)")
    sys.exit(41)

  # 2) rgb + 0  (THIS is where your real run dies)
  try:
    rgb2 = rebuffer_xla(rgb)
    jax.block_until_ready(rgb2)
    print("[probe] rebuffer_xla(rgb) (rgb+0) OK")
  except Exception as e:
    print("[probe] FAIL: rebuffer_xla(rgb) ->", repr(e))
    cuda_err("after FAIL rebuffer_xla(rgb)")
    sys.exit(42)

  # 3) DLPack to torch + torch op
  try:
    import torch
    rgb_t = dlpack_to_torch(rgb2)
    if rgb_t.is_cuda:
      torch.cuda.synchronize()
    print("[probe] dlpack_to_torch OK:", rgb_t.shape, rgb_t.dtype, rgb_t.device)
    obs_t = torch_pack_stereo(rgb_t)
    if obs_t.is_cuda:
      torch.cuda.synchronize()
    print("[probe] torch_pack_stereo OK:", obs_t.shape, obs_t.dtype, obs_t.device)
  except Exception as e:
    print("[probe] FAIL: dlpack/torch consume ->", repr(e))
    cuda_err("after FAIL dlpack/torch consume")
    sys.exit(43)

  # 4) now do another JAX op that *depends on rgb2* (stronger than allocating zeros)
  try:
    y = (rgb2.astype(jp.uint32).sum())
    jax.block_until_ready(y)
    print("[probe] JAX consume(rgb2) OK")
  except Exception as e:
    print("[probe] FAIL: JAX consume(rgb2) ->", repr(e))
    cuda_err("after FAIL JAX consume(rgb2)")
    sys.exit(44)

print("[probe] ALL GOOD in this decisive variant")
PY
}

set +e
# 4-way LD/Torch × 2-way clear-mode × 2 loop counts (1 and 5)
# Clear-mode=peek is the “no masking” mode; clear-mode=get tests whether clearing hides issues.
run_decisive system 0 peek 1; rc_a=$?
run_decisive system 1 peek 1; rc_b=$?
run_decisive pip    0 peek 1; rc_c=$?
run_decisive pip    1 peek 1; rc_d=$?

run_decisive system 1 peek 5; rc_e=$?
run_decisive system 1 get  5; rc_f=$?
set -e

echo
echo "=== [decisive summary] exit codes ==="
echo "system/noTorch peek loops1 : $rc_a"
echo "system/Torch   peek loops1 : $rc_b"
echo "pip/noTorch    peek loops1 : $rc_c"
echo "pip/Torch      peek loops1 : $rc_d"
echo "system/Torch   peek loops5 : $rc_e"
echo "system/Torch   get  loops5 : $rc_f"
echo "Codes: 0=OK, 41=device_put fail, 42=rebuffer(rgb+0) fail, 43=dlpack/torch fail, 44=JAX consume(rgb2) fail"

# If any decisive probe fails, stop before training
if [[ "$rc_a" != "0" || "$rc_b" != "0" || "$rc_c" != "0" || "$rc_d" != "0" || "$rc_e" != "0" || "$rc_f" != "0" ]]; then
  echo "FATAL: decisive probe failed. Not starting training."
  exit 20
fi

echo
echo "=== [CONTAINER] starting training ==="
stdbuf -oL -eL python -u execute.py 2>&1
'

echo "Finished at $(date)"
