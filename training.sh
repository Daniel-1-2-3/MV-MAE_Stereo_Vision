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
  echo "FATAL: execute.py not found at: $HOST_PROJECT_ROOT/execute.py"
  exit 10
fi

# ---------------- Python path inside container ----------------
export APPTAINERENV_PYTHONPATH="/workspace:/opt/src:/opt/src/MV_MAE_Implementation:${PYTHONPATH:-}"

# ---------------- EGL / MuJoCo GL setup ----------------
export APPTAINERENV_MUJOCO_GL=egl
export APPTAINERENV_PYOPENGL_PLATFORM=egl
export APPTAINERENV_MUJOCO_PLATFORM=egl
export APPTAINERENV_DISPLAY=
export APPTAINERENV_LIBGL_ALWAYS_SOFTWARE=0
export APPTAINERENV_MESA_LOADER_DRIVER_OVERRIDE=
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
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

# ---------------- Quick EGL + GPU probe ----------------
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  "$IMG" \
  bash -lc '
python - << "PY"
import torch, mujoco, OpenGL.GL as gl
print("torch cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
ctx = mujoco.GLContext(64, 64); ctx.make_current()
to_s = lambda b: b.decode("utf-8","ignore") if b else None
print("OpenGL vendor  :", to_s(gl.glGetString(gl.GL_VENDOR)))
print("OpenGL renderer:", to_s(gl.glGetString(gl.GL_RENDERER)))
ctx.free()
PY
'

# ---------------- Training ----------------
apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  --pwd "$WORKDIR_IN_CONTAINER" \
  "$IMG" \
  bash -lc "
set -euo pipefail
export PYTHONUNBUFFERED=1

echo '=== MuJoCo version ==='
python - <<'PY'
import mujoco
print('MuJoCo version:', mujoco.__version__)
PY
echo '======================'

# ---------------- Persistent python prefix (host-backed) ----------------
DEPS_PREFIX='$SLURM_SUBMIT_DIR/.pydeps_prefix'
PY_MM=\$(python - <<'PY'
import sys
print(f\"{sys.version_info.major}.{sys.version_info.minor}\")
PY
)
SITE_PKGS=\"\$DEPS_PREFIX/lib/python\${PY_MM}/site-packages\"
BIN_DIR=\"\$DEPS_PREFIX/bin\"
mkdir -p \"\$DEPS_PREFIX\"

export PYTHONPATH=\"/workspace:\$SITE_PKGS:\${PYTHONPATH:-}\"
export PATH=\"\$BIN_DIR:\$PATH\"

# ---------------- Working cmake (avoids missing librhash.so.0) ----------------
echo '=== Ensuring working cmake + ninja in prefix ==='
python -m pip install --upgrade --no-cache-dir --prefix \"\$DEPS_PREFIX\" cmake ninja
hash -r
cmake --version

# ---------------- JAX / XLA tuning ----------------
export JAX_TRACEBACK_FILTERING=off
export JAX_DISABLE_CUSOLVER=1
export XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_gpu_enable_triton_gemm=false'
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=.60
export NVIDIA_TF32_OVERRIDE=0

echo '=== Driver + visibility ==='
nvidia-smi || true
echo \"CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES\"

echo '=== JAX backend probe ==='
python - <<'PY'
import jax, jaxlib
print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)
print('devices()', jax.devices())
print('default_backend()', jax.default_backend())
PY

# ---------------- Ensure tensorboard (optional) ----------------
if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec('tensorboard') else 1)
PY
then
  python -m pip install --upgrade --no-cache-dir --prefix \"\$DEPS_PREFIX\" tensorboard
fi

# ---------------- Madrona cache dir (per GPU model) ----------------
ACTUAL_GPU=\$(nvidia-smi -L 2>/dev/null | head -1 || true)
GPU_MODEL=\$(echo \"\$ACTUAL_GPU\" | grep -o 'H100\\|L40S\\|A100\\|V100\\|RTX' | head -1 || true)
GPU_MODEL=\${GPU_MODEL:-unknown}
GPU_MODEL_LOWER=\$(echo \"\$GPU_MODEL\" | tr '[:upper:]' '[:lower:]')
ENV_CONFIG='default'

CACHE_BUILD_DIR='$SLURM_SUBMIT_DIR/build_'\${GPU_MODEL_LOWER}'_'\${ENV_CONFIG}
mkdir -p \"\$CACHE_BUILD_DIR/kernel_cache\" \"\$CACHE_BUILD_DIR/bvh_cache\"
export MADRONA_MWGPU_KERNEL_CACHE=\"\$CACHE_BUILD_DIR/kernel_cache/kernel.cache\"
export MADRONA_BVH_KERNEL_CACHE=\"\$CACHE_BUILD_DIR/bvh_cache/bvh.cache\"

echo '=== Madrona caches ==='
echo \"  GPU_MODEL_LOWER=\$GPU_MODEL_LOWER\"
echo \"  MADRONA_MWGPU_KERNEL_CACHE=\$MADRONA_MWGPU_KERNEL_CACHE\"
echo \"  MADRONA_BVH_KERNEL_CACHE=\$MADRONA_BVH_KERNEL_CACHE\"

# ---------------- HARD RESET caches (raytracer stability) ----------------
echo '=== Clearing JAX + CUDA + Madrona caches (raytracer) ==='
rm -rf ~/.cache/jax || true
rm -rf ~/.nv/ComputeCache || true
rm -f \"\$MADRONA_MWGPU_KERNEL_CACHE\" \"\$MADRONA_BVH_KERNEL_CACHE\" || true

# ---------------- Rebuild madrona_mjx into prefix (overrides /opt/madrona_mjx) ----------------
echo '=== Rebuilding madrona_mjx into prefix (no SIF changes) ==='
cd '$SLURM_SUBMIT_DIR'
if [[ ! -d madrona_mjx_user ]]; then
  git clone --branch geom_quat https://github.com/shacklettbp/madrona_mjx.git madrona_mjx_user
fi
cd madrona_mjx_user
git submodule update --init --recursive
rm -rf build
mkdir -p build && cd build
cmake .. -G Ninja -DLOAD_VULKAN=OFF
ninja -j\"\$(nproc)\"
cd ..
python -m pip install -e . --prefix \"\$DEPS_PREFIX\"

python - <<'PY'
import madrona_mjx, inspect
print('madrona_mjx imported from:', inspect.getfile(madrona_mjx))
PY

# ---------------- Run training ----------------
cd /workspace
echo '=== Starting training ==='
stdbuf -oL -eL python -u execute.py 2>&1
echo 'Training completed.'
"

echo "Finished at $(date)"
