#!/bin/bash
#SBATCH --job-name=mvmae_drqv2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load apptainer/1.3.5 || module load apptainer

IMG="$SLURM_SUBMIT_DIR/training.sif"
if [[ ! -f "$IMG" ]]; then
  echo "ERROR: $IMG not found"; exit 2
fi

# Optional Hydra overrides, e.g.: sbatch run_training.sh "save_video=false agent.device=cuda"
EXTRA_ARGS="${1:-}"

# ---- Export env into container (don’t set PYTHONPATH here; image already does) ----
export APPTAINERENV_MUJOCO_GL="egl"
export APPTAINERENV_PYOPENGL_PLATFORM="egl"
export APPTAINERENV_MESA_EGL_NO_X11="1"
export APPTAINERENV_DISPLAY=""
export APPTAINERENV_LIBGL_ALWAYS_SOFTWARE="0"
export APPTAINERENV_OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ---- Bind host EGL vendor JSON + NVIDIA libs (needed for headless EGL) ----
VENDOR_JSON="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if [[ ! -f "$VENDOR_JSON" ]]; then
  echo "FATAL: $VENDOR_JSON missing on host. Ask admins to install GLVND NVIDIA EGL ICD."; exit 3
fi
export APPTAINERENV__EGL_VENDOR_LIBRARY_FILENAMES="$VENDOR_JSON"

NV_EGL_DIR="$(ldconfig -p | awk '/libEGL_nvidia\.so/{print $NF; exit}' | xargs -r dirname || true)"
for d in /usr/lib/x86_64-linux-gnu/nvidia /usr/lib/nvidia /usr/lib64/nvidia /usr/lib/x86_64-linux-gnu; do
  [[ -z "$NV_EGL_DIR" && -e "$d/libEGL_nvidia.so.0" ]] && NV_EGL_DIR="$d"
done
if [[ -z "${NV_EGL_DIR:-}" || ! -d "$NV_EGL_DIR" ]]; then
  echo "FATAL: Could not find libEGL_nvidia.so* on host. Ask admins to install NVIDIA EGL libs."; exit 4
fi

GLVND_DIR="/usr/lib/x86_64-linux-gnu"
[[ -e "$GLVND_DIR/libEGL.so.1" ]] || GLVND_DIR="/usr/lib64"

BIND_FLAGS=( --bind "$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR" )
BIND_FLAGS+=( --bind "/usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d" )
BIND_FLAGS+=( --bind "$NV_EGL_DIR:$NV_EGL_DIR" )
BIND_FLAGS+=( --bind "$GLVND_DIR:$GLVND_DIR" )

echo "[INFO] Using binds:"; printf '  %s\n' "${BIND_FLAGS[@]}"

# ---- Probe CUDA + EGL inside the container ----
apptainer exec --nv --cleanenv \
  "${BIND_FLAGS[@]}" \
  --pwd "$SLURM_SUBMIT_DIR" \
  "$IMG" \
  bash -lc 'set -euo pipefail
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

# ---- Run training (container %runscript runs drqv2_architecture.py by default) ----
echo "[RUN] apptainer run --nv --cleanenv training.sif ${EXTRA_ARGS}"
apptainer run --nv --cleanenv \
  "${BIND_FLAGS[@]}" \
  --pwd "$SLURM_SUBMIT_DIR" \
  "$IMG" ${EXTRA_ARGS}
