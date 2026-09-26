#!/usr/bin/env bash
# Install everything into the *active* Python 3.11 environment:
#   Isaac Sim 5.0.0, Isaac Lab v2.3.0 (from source, into third_party/IsaacLab),
#   PyTorch 2.7.0 + CUDA 12.8, the Isaac for Healthcare asset helper, and this
#   project's Python dependencies.
#
#   conda create -n mvmae python=3.11 -y && conda activate mvmae
#   export OMNI_KIT_ACCEPT_EULA=YES      # after reading https://docs.omniverse.nvidia.com/eula
#   bash install.sh
#
# Mirrors Isaac for Healthcare v0.4.0's tools/env_setup_robot_surgery.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python -c 'import sys; assert sys.version_info[:2] == (3, 11), f"Python 3.11 is required, found {sys.version.split()[0]}"'
if [ "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" ]; then
    echo "Isaac Sim needs the NVIDIA Omniverse License Agreement accepted: read"
    echo "https://docs.omniverse.nvidia.com/eula, then run: export OMNI_KIT_ACCEPT_EULA=YES"
    exit 1
fi
if ! command -v nvidia-smi > /dev/null; then
    echo "nvidia-smi not found: an NVIDIA GPU with drivers is required"; exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# System libraries Isaac Sim needs on minimal (e.g. cloud container) images.
if command -v apt-get > /dev/null && [ "$(id -u)" = "0" ]; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git wget libglvnd0 libglx0 libegl1 libvulkan1 libglu1-mesa libgl1 libxt6 libxrender1 libxi6 libxext6 libsm6 libice6 libxrandr2 vulkan-tools
fi

python -m pip install --upgrade pip
python -m pip install setuptools==75.8.0 toml==0.10.2

# 1) Isaac Sim + Isaac for Healthcare asset helper (same command as Isaac for Healthcare's installer)
python -m pip install "isaacsim[all,extscache]==5.0.0" \
    "i4h_asset_helper @ git+https://github.com/isaac-for-healthcare/i4h-asset-catalog.git@v0.3.0" \
    --extra-index-url https://pypi.nvidia.com
# 2) PyTorch built for CUDA 12.8 (needed for Blackwell GPUs; isaaclab.sh also checks this)
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

ISAACLAB_DIR="third_party/IsaacLab"
if [ ! -d "$ISAACLAB_DIR" ]; then
    git clone --depth 1 --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi
# 3) Isaac Lab from source
(cd "$ISAACLAB_DIR" && ./isaaclab.sh --install none)

# Same patch as Isaac for Healthcare's installer: import omni.log lazily in
# isaaclab/utils/math.py so the module also imports cleanly with Isaac Sim 5.0.
MATH_PY="$ISAACLAB_DIR/source/isaaclab/isaaclab/utils/math.py"
sed -i '/^[[:space:]]*import omni\.log[[:space:]]*$/d' "$MATH_PY"
sed -i -E 's/^([[:space:]]*)omni\.log\.warn\(/\1import omni.log\
\1omni.log.warn(/g' "$MATH_PY"

# 4) Project dependencies last (see requirements.txt for why they are a separate step)
python -m pip install -r requirements.txt

python - <<'EOF'
import importlib.metadata as md
import torch, wandb, imageio, yaml  # fail loudly if anything is missing
import isaaclab
for pkg in ("isaacsim", "isaaclab", "torch", "i4h_asset_helper", "wandb"):
    print(f"{pkg:18s} {md.version(pkg)}")
print("CUDA available:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
EOF
if command -v vulkaninfo > /dev/null; then
    vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverVersion" || echo "WARNING: vulkaninfo found no GPU; Isaac Sim cameras will not render"
fi
echo "Done. Next: python test_stereo.py"
