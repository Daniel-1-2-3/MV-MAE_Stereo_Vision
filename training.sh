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
cd "${SLURM_SUBMIT_DIR:?}"

module load apptainer/1.3.5 || module load apptainer

# ============================================================
# CONFIG (ONLY EDIT THIS SECTION IF NEEDED)
# ============================================================
IMG="${SLURM_SUBMIT_DIR}/training.sif"
WORKDIR_IN_CONTAINER="/workspace"
HOST_PROJECT_ROOT="${SLURM_SUBMIT_DIR}"

DEPS_PREFIX="${SLURM_SUBMIT_DIR}/.pydeps_prefix"
MADRONA_SRC_DIR="${SLURM_SUBMIT_DIR}/madrona_mjx_user"
MADRONA_BRANCH="geom_quat"

REBUILD_MADRONA_EVERY_RUN="true"
WIPE_CACHES_EVERY_RUN="true"
# ============================================================

[[ -f "$IMG" ]] || { echo "ERROR: IMG not found: $IMG"; exit 2; }
[[ -f "$HOST_PROJECT_ROOT/execute.py" ]] || { echo "ERROR: execute.py not found at $HOST_PROJECT_ROOT/execute.py"; exit 10; }

# Set PYTHONPATH explicitly so Apptainer won't merge in cluster PYTHONPATH junk
export APPTAINERENV_PYTHONPATH="/workspace:/opt/src:/opt/src/MV_MAE_Implementation"

# EGL / MuJoCo GL
export APPTAINERENV_MUJOCO_GL=egl
export APPTAINERENV_PYOPENGL_PLATFORM=egl
export APPTAINERENV_MUJOCO_PLATFORM=egl
export APPTAINERENV_DISPLAY=
export APPTAINERENV_LIBGL_ALWAYS_SOFTWARE=0
export APPTAINERENV_MESA_LOADER_DRIVER_OVERRIDE=
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export APPTAINERENV_IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg

# NVIDIA EGL vendor JSON on host
VENDOR_JSON="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
[[ -f "$VENDOR_JSON" ]] || { echo "FATAL: $VENDOR_JSON not found on host"; exit 3; }
export APPTAINERENV__EGL_VENDOR_LIBRARY_FILENAMES="$VENDOR_JSON"

# Locate libEGL_nvidia.so on host
NV_EGL_DIR="$(ldconfig -p | awk '/libEGL_nvidia\.so/{print $NF; exit}' | xargs -r dirname || true)"
for d in /usr/lib/x86_64-linux-gnu/nvidia /usr/lib/nvidia /usr/lib64/nvidia /usr/lib/x86_64-linux-gnu; do
  [[ -z "$NV_EGL_DIR" && -e "$d/libEGL_nvidia.so.0" ]] && NV_EGL_DIR="$d"
done
[[ -n "${NV_EGL_DIR:-}" && -d "$NV_EGL_DIR" ]] || { echo "FATAL: Could not find libEGL_nvidia.so* on host"; exit 4; }

GLVND_DIR="/usr/lib/x86_64-linux-gnu"
[[ -e "$GLVND_DIR/libEGL.so.1" ]] || GLVND_DIR="/usr/lib64"

# mujoco_playground external_deps bind (writable host dir)
HOST_MJP_DEPS="${SLURM_SUBMIT_DIR}/mujoco_playground_external_deps"
mkdir -p "$HOST_MJP_DEPS"
MJP_DEPS_IN_CONTAINER="/opt/mvmae_venv/lib/python3.12/site-packages/mujoco_playground/external_deps"

# Binds
BIND_FLAGS=( --bind "$HOST_PROJECT_ROOT:$WORKDIR_IN_CONTAINER" )
BIND_FLAGS+=( --bind "/usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d" )
BIND_FLAGS+=( --bind "$NV_EGL_DIR:$NV_EGL_DIR" )
BIND_FLAGS+=( --bind "$GLVND_DIR:$GLVND_DIR" )
BIND_FLAGS+=( --bind "$HOST_MJP_DEPS:$MJP_DEPS_IN_CONTAINER" )

apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  "$IMG" \
  bash -lc "
set -euo pipefail
export PYTHONUNBUFFERED=1

echo '=== Prefix ==='
echo 'DEPS_PREFIX=$DEPS_PREFIX'
mkdir -p '$DEPS_PREFIX'

# pip should never target /opt (read-only)
export PIP_PREFIX='$DEPS_PREFIX'
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1

PY_MM=\$(python - <<'PY'
import sys
print(f\"{sys.version_info.major}.{sys.version_info.minor}\")
PY
)
SITE_PKGS='$DEPS_PREFIX'/lib/python\${PY_MM}/site-packages
BIN_DIR='$DEPS_PREFIX'/bin
mkdir -p \"\$SITE_PKGS\" \"\$BIN_DIR\"

# Make our prefix win for imports + entrypoints
export PYTHONPATH=\"\$SITE_PKGS:/workspace:/opt/src:/opt/src/MV_MAE_Implementation\"
export PATH=\"\$BIN_DIR:\$PATH\"

echo '=== Ensure cmake+ninja in prefix (avoids librhash.so.0 issues) ==='
python -m pip install --upgrade --prefix '$DEPS_PREFIX' cmake ninja
hash -r
cmake --version
ninja --version || true

echo '=== GPU/EGL probe ==='
python - <<'PY'
import torch, mujoco, OpenGL.GL as gl
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

# JAX/XLA knobs (optional)
export JAX_TRACEBACK_FILTERING=off
export JAX_DISABLE_CUSOLVER=1
export XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_gpu_enable_triton_gemm=false'
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=.60
export NVIDIA_TF32_OVERRIDE=0

# Madrona caches
ACTUAL_GPU=\$(nvidia-smi -L 2>/dev/null | head -1 || true)
GPU_MODEL=\$(echo \"\$ACTUAL_GPU\" | grep -o 'H100\\|L40S\\|A100\\|V100\\|RTX' | head -1 || true)
GPU_MODEL=\${GPU_MODEL:-unknown}
GPU_MODEL_LOWER=\$(echo \"\$GPU_MODEL\" | tr '[:upper:]' '[:lower:]')
CACHE_BUILD_DIR='${SLURM_SUBMIT_DIR}'/build_\${GPU_MODEL_LOWER}_default
mkdir -p \"\$CACHE_BUILD_DIR/kernel_cache\" \"\$CACHE_BUILD_DIR/bvh_cache\"
export MADRONA_MWGPU_KERNEL_CACHE=\"\$CACHE_BUILD_DIR/kernel_cache/kernel.cache\"
export MADRONA_BVH_KERNEL_CACHE=\"\$CACHE_BUILD_DIR/bvh_cache/bvh.cache\"

echo '=== Madrona caches ==='
echo \"MADRONA_MWGPU_KERNEL_CACHE=\$MADRONA_MWGPU_KERNEL_CACHE\"
echo \"MADRONA_BVH_KERNEL_CACHE=\$MADRONA_BVH_KERNEL_CACHE\"

if [[ '$WIPE_CACHES_EVERY_RUN' == 'true' ]]; then
  echo '=== Wiping caches ==='
  rm -rf ~/.cache/jax || true
  rm -rf ~/.nv/ComputeCache || true
  rm -f \"\$MADRONA_MWGPU_KERNEL_CACHE\" \"\$MADRONA_BVH_KERNEL_CACHE\" || true
fi

if [[ '$REBUILD_MADRONA_EVERY_RUN' == 'true' ]]; then
  echo '=== Rebuild madrona_mjx (source in host dir) ==='
  cd '${SLURM_SUBMIT_DIR}'

  if [[ ! -d '$MADRONA_SRC_DIR' ]]; then
    git clone --branch '$MADRONA_BRANCH' https://github.com/shacklettbp/madrona_mjx.git '$MADRONA_SRC_DIR'
  fi

  cd '$MADRONA_SRC_DIR'
  git submodule update --init --recursive

  rm -rf build
  mkdir -p build
  cd build
  cmake .. -G Ninja -DLOAD_VULKAN=OFF
  ninja -j\"\$(nproc)\"
  cd ..

  # IMPORTANT:
  # pip non-editable fails because madrona_py_build doesn't implement build_wheel.
  # Use editable install (works for this repo) and install INTO OUR PREFIX (writable).
  echo '=== Install madrona_mjx as EDITABLE into prefix (no wheel) ==='
  if command -v uv >/dev/null 2>&1; then
    uv pip install -e . --no-deps --prefix '$DEPS_PREFIX'
  else
    # fallback: pip editable (still uses build backend, but calls editable path)
    python -m pip install -e . --no-deps --prefix '$DEPS_PREFIX'
  fi
fi

echo '=== Confirm madrona_mjx import path ==='
python - <<'PY'
import madrona_mjx, inspect
print('madrona_mjx imported from:', inspect.getfile(madrona_mjx))
PY

echo '=== Start training ==='
cd /workspace
stdbuf -oL -eL python -u execute.py 2>&1
"

echo "Finished at $(date)"
