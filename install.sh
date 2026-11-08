#!/usr/bin/env bash
# Install everything into the *active* Python 3.12 environment:
#   Isaac Sim 6.0.1, Isaac Lab v3.0.0-beta2.patch1 (from source, into third_party/IsaacLab),
#   PyTorch 2.10.0 + CUDA 12.8, and this project's Python dependencies.
#
#   conda create -n mvmae python=3.12 -y && conda activate mvmae
#   export OMNI_KIT_ACCEPT_EULA=YES      # after reading https://docs.omniverse.nvidia.com/eula
#   bash install.sh
#
# Isaac Sim 6.0 instead of 5.1: Isaac Sim 5.1 crashes at start-up (librtx.scenedb.plugin.so) on
# 595-series NVIDIA drivers (isaac-sim/IsaacSim issue #677). Isaac Lab v3.0.0-beta2.patch1 is the
# Isaac Lab release built for Isaac Sim 6.0.1 (its pins: Python 3.12, torch 2.10.0 / torchvision 0.25.0
# on CUDA 12.8); the later v3.0.0 release moved on to Isaac Sim 6.2.
# Originally mirrored Isaac for Healthcare v0.4.0's tools/env_setup_robot_surgery.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python -c 'import sys; assert sys.version_info[:2] == (3, 12), f"Python 3.12 is required, found {sys.version.split()[0]}"'
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
python -m pip install "setuptools<82" wheel toml==0.10.2

# 1) Isaac Sim 6.0.1 (same command as Isaac Lab v3.0.0-beta2.patch1's installer).
#    The Isaac for Healthcare asset helper (i4h_asset_helper) is no longer installed: nothing imports
#    it - surgical_env/assets.py downloads the same robot / needle USDs (catalog v0.3.0) over plain
#    HTTPS - and its unpinned "isaacsim[all,extscache]" dependency could pull in another Isaac Sim.
python -m pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com
# 2) PyTorch built for CUDA 12.8 (the build Isaac Lab 3.0 pins on x86_64; has Blackwell / sm_120 kernels).
#    The explicit "+cu128" local version forces pip to replace whatever torch Isaac Sim pulled in
#    instead of treating "2.10.0" as already satisfied. torchaudio is not used (Isaac Lab 3.0 dropped it).
python -m pip uninstall -y torchaudio > /dev/null 2>&1 || true
python -m pip install "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128

ISAACLAB_DIR="third_party/IsaacLab"
ISAACLAB_TAG="v3.0.0-beta2.patch1"
if [ ! -d "$ISAACLAB_DIR" ]; then
    git clone --depth 1 --branch "$ISAACLAB_TAG" https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi
# 3) Isaac Lab core package plus the PhysX backend (assets, cameras and the RTX renderer moved to
# isaaclab_physx in 3.0). Installed with pip directly rather than `isaaclab.sh --install`, which hides
# pip failures and pulls in Newton / RL frameworks this project does not use.
python -m pip install --editable "$ISAACLAB_DIR/source/isaaclab" h5py
# Isaac Sim also auto-loads Isaac Lab's extensions from the source tree (isaaclab_physx, isaaclab_newton,
# isaaclab_assets, isaaclab_tasks, ...); make them importable without their optional heavy dependencies.
python -m pip install --no-deps \
    --editable "$ISAACLAB_DIR/source/isaaclab_ppisp" \
    --editable "$ISAACLAB_DIR/source/isaaclab_physx" \
    --editable "$ISAACLAB_DIR/source/isaaclab_newton" \
    --editable "$ISAACLAB_DIR/source/isaaclab_assets" \
    --editable "$ISAACLAB_DIR/source/isaaclab_tasks"
# (The omni.log patch of isaaclab/utils/math.py used with Isaac Lab 2.3 is gone: 3.0's math.py no
# longer imports omni.log.)

# 4) Project dependencies last, constrained to the exact versions Isaac Sim pins
python tools/isaacsim_constraints.py > .isaacsim_constraints.txt
echo "versions pinned by Isaac Sim: $(wc -l < .isaacsim_constraints.txt)"
python -m pip install -r requirements.txt -c .isaacsim_constraints.txt

python - <<'EOF'
import importlib.metadata as md
import torch, wandb, imageio, yaml  # fail loudly if anything is missing
import isaaclab
import isaaclab_physx
for pkg in ("isaacsim", "isaaclab", "isaaclab_physx", "torch", "torchvision", "wandb"):
    print(f"{pkg:18s} {md.version(pkg)}")
print("CUDA available:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
assert torch.version.cuda and torch.version.cuda.startswith("12.8"), f"expected a CUDA 12.8 torch build, got {torch.__version__}"
if torch.cuda.is_available():
    import re
    major, minor = torch.cuda.get_device_capability(0)
    # CUDA kernels built for sm_XY run on any GPU of the same major architecture X with an equal or
    # higher minor version (e.g. an L40S, sm_89, runs the sm_86 kernels): exact matches are not needed.
    built = [tuple(int(g) for g in m.groups()) for a in torch.cuda.get_arch_list()
             if (m := re.fullmatch(r"sm_(\d+)(\d)[a-z]?", a))]
    usable = [f"sm_{M}{m}" for M, m in built if M == major and m <= minor]
    assert usable, f"torch has no kernels for this GPU (sm_{major}{minor}); it was built for {torch.cuda.get_arch_list()}"
    print(f"GPU sm_{major}{minor} runs torch's {usable[-1]} kernels")
    print("torch GPU check:", (torch.ones(4, device="cuda") * 2).sum().item() == 8.0)
EOF
if command -v vulkaninfo > /dev/null; then
    vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverVersion" || echo "WARNING: vulkaninfo found no GPU; Isaac Sim cameras will not render"
fi
echo "Done. Next: python test_stereo.py"
