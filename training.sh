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

IMG="$SLURM_SUBMIT_DIR/training.sif"
[[ -f "$IMG" ]] || { echo "ERROR: $IMG not found"; exit 2; }

HOST_PROJECT_ROOT="$SLURM_SUBMIT_DIR"
WORKDIR_IN_CONTAINER="/workspace"
[[ -f "$HOST_PROJECT_ROOT/execute.py" ]] || { echo "FATAL: execute.py not found at $HOST_PROJECT_ROOT/execute.py"; exit 10; }

# ---------- Persistent user prefix on HOST (NOT inside /opt) ----------
DEPS_PREFIX_HOST="$SLURM_SUBMIT_DIR/.pydeps_prefix"
mkdir -p "$DEPS_PREFIX_HOST"

# ---------- EGL / MuJoCo GL ----------
export APPTAINERENV_MUJOCO_GL=egl
export APPTAINERENV_PYOPENGL_PLATFORM=egl
export APPTAINERENV_MUJOCO_PLATFORM=egl
export APPTAINERENV_DISPLAY=
export APPTAINERENV_LIBGL_ALWAYS_SOFTWARE=0
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export APPTAINERENV_IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg

VENDOR_JSON="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
[[ -f "$VENDOR_JSON" ]] || { echo "FATAL: $VENDOR_JSON not found on host"; exit 3; }
export APPTAINERENV__EGL_VENDOR_LIBRARY_FILENAMES="$VENDOR_JSON"

NV_EGL_DIR="$(ldconfig -p | awk '/libEGL_nvidia\.so/{print $NF; exit}' | xargs -r dirname || true)"
for d in /usr/lib/x86_64-linux-gnu/nvidia /usr/lib/nvidia /usr/lib64/nvidia /usr/lib/x86_64-linux-gnu; do
  [[ -z "${NV_EGL_DIR:-}" && -e "$d/libEGL_nvidia.so.0" ]] && NV_EGL_DIR="$d"
done
[[ -n "${NV_EGL_DIR:-}" && -d "$NV_EGL_DIR" ]] || { echo "FATAL: Could not find libEGL_nvidia.so* on host"; exit 4; }

GLVND_DIR="/usr/lib/x86_64-linux-gnu"
[[ -e "$GLVND_DIR/libEGL.so.1" ]] || GLVND_DIR="/usr/lib64"

# ---------- Bind mujoco_playground external_deps override ----------
HOST_MJP_DEPS="$SLURM_SUBMIT_DIR/mujoco_playground_external_deps"
mkdir -p "$HOST_MJP_DEPS"
MJP_DEPS_IN_CONTAINER="/opt/mvmae_venv/lib/python3.12/site-packages/mujoco_playground/external_deps"

BIND_FLAGS=(
  --bind "$HOST_PROJECT_ROOT:$WORKDIR_IN_CONTAINER"
  --bind "/usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d"
  --bind "$NV_EGL_DIR:$NV_EGL_DIR"
  --bind "$GLVND_DIR:$GLVND_DIR"
  --bind "$HOST_MJP_DEPS:$MJP_DEPS_IN_CONTAINER"
  --bind "$DEPS_PREFIX_HOST:$DEPS_PREFIX_HOST"
)

# ---------- One container run ----------
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  "$IMG" \
  bash -lc "
set -euo pipefail

echo '=== Prefix ==='
echo 'DEPS_PREFIX=$DEPS_PREFIX_HOST'

# Make user prefix visible (pip installs go here)
PY_MM=\$(python - <<'PY'
import sys
print(f'{sys.version_info.major}.{sys.version_info.minor}')
PY
)
SITE_PKGS=\"$DEPS_PREFIX_HOST/lib/python\${PY_MM}/site-packages\"
BIN_DIR=\"$DEPS_PREFIX_HOST/bin\"
mkdir -p \"$DEPS_PREFIX_HOST\"
export PYTHONPATH=\"$WORKDIR_IN_CONTAINER:\$SITE_PKGS:\${PYTHONPATH:-}\"
export PATH=\"\$BIN_DIR:\$PATH\"

echo '=== Ensure cmake+ninja in prefix (avoids librhash.so.0 issues) ==='
python -m pip install --upgrade --no-cache-dir --prefix \"$DEPS_PREFIX_HOST\" cmake ninja >/dev/null
hash -r
cmake --version || true
ninja --version || true

echo '=== GPU/EGL probe ==='
python - <<'PY'
import torch, mujoco
import OpenGL.GL as gl
print('torch cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
ctx = mujoco.GLContext(64, 64); ctx.make_current()
to_s = lambda b: b.decode('utf-8','ignore') if b else None
print('OpenGL vendor  :', to_s(gl.glGetString(gl.GL_VENDOR)))
print('OpenGL renderer:', to_s(gl.glGetString(gl.GL_RENDERER)))
ctx.free()
PY

echo '=== Versions ==='
python - <<'PY'
import mujoco, jax, jaxlib
print('MuJoCo:', mujoco.__version__)
print('jax:', jax.__version__, 'jaxlib:', jaxlib.__version__)
print('devices:', jax.devices())
print('default_backend:', jax.default_backend())
PY

# ---------------- Madrona caches ----------------
ACTUAL_GPU=\$(nvidia-smi -L 2>/dev/null | head -1 || true)
GPU_MODEL=\$(echo \"\$ACTUAL_GPU\" | grep -o 'H100\\|L40S\\|A100\\|V100\\|RTX' | head -1 || true)
GPU_MODEL=\${GPU_MODEL:-unknown}
GPU_MODEL_LOWER=\$(echo \"\$GPU_MODEL\" | tr '[:upper:]' '[:lower:]')
ENV_CONFIG=default

CACHE_BUILD_DIR=\"$DEPS_PREFIX_HOST/../build_\${GPU_MODEL_LOWER}_\${ENV_CONFIG}\"
mkdir -p \"\$CACHE_BUILD_DIR/kernel_cache\" \"\$CACHE_BUILD_DIR/bvh_cache\"
export MADRONA_MWGPU_KERNEL_CACHE=\"\$CACHE_BUILD_DIR/kernel_cache/kernel.cache\"
export MADRONA_BVH_KERNEL_CACHE=\"\$CACHE_BUILD_DIR/bvh_cache/bvh.cache\"

echo '=== Madrona caches ==='
echo \"MADRONA_MWGPU_KERNEL_CACHE=\$MADRONA_MWGPU_KERNEL_CACHE\"
echo \"MADRONA_BVH_KERNEL_CACHE=\$MADRONA_BVH_KERNEL_CACHE\"

# ---------------- JAX/XLA knobs (keep raytracing) ----------------
export PYTHONUNBUFFERED=1
export JAX_TRACEBACK_FILTERING=off
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=.60
export XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_gpu_enable_triton_gemm=false'
export NVIDIA_TF32_OVERRIDE=0

# ---------------- Build madrona_mjx (NO pip install into /opt) ----------------
# We DO NOT 'pip install' it. We just put its 'src' on PYTHONPATH so imports resolve to your build.
echo '=== Build madrona_mjx_user (if needed) ==='
cd \"$HOST_PROJECT_ROOT\"
if [[ ! -d madrona_mjx_user ]]; then
  git clone --branch geom_quat https://github.com/shacklettbp/madrona_mjx.git madrona_mjx_user
fi
cd madrona_mjx_user
git submodule update --init --recursive

mkdir -p build
cd build
cmake .. -DLOAD_VULKAN=OFF -G Ninja
ninja -j\$(nproc)

# Put the built package ahead of everything else
export PYTHONPATH=\"$HOST_PROJECT_ROOT/madrona_mjx_user/src:\$PYTHONPATH\"

python - <<'PY'
import madrona_mjx, inspect
print('madrona_mjx imported from:', inspect.getfile(madrona_mjx))
PY

# ---------------- Smoke test: force renderer failure NOW ----------------
# This is the key: if Madrona is the culprit, it should die HERE, not later at PRNGKey.
echo '=== Smoke test: construct env + force one render (no JIT) ==='
export JAX_DISABLE_JIT=1
python - <<'PY'
import jax
# import your env module path as you actually use it:
from Mujoco_Sim.pick_env import StereoPickCube

env = StereoPickCube(render_batch_size=2, render_width=64, render_height=64)
print('Env constructed OK')

# Force a reset -> should call _init_renderer already, and obs path should render pixels.
key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 2)
state = env.reset(keys)
print('Reset OK; obs shape:', getattr(state.obs, 'shape', None), 'dtype:', getattr(state.obs, 'dtype', None))
PY
unset JAX_DISABLE_JIT

echo '=== Start training ==='
cd \"$WORKDIR_IN_CONTAINER\"
stdbuf -oL -eL python -u execute.py 2>&1
"

echo "Finished at $(date)"
