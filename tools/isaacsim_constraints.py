"""Print pip constraints for the exact versions Isaac Sim pins (used by install.sh).

Installing the project's own dependencies with these constraints stops pip from
replacing packages Isaac Sim depends on (click, psutil, plotly, ...). Pillow and
torch* are skipped: Isaac Lab 2.3.0 itself needs a different Pillow than Isaac Sim
pins, and torch is installed separately from the CUDA 12.8 index.
"""

import importlib.metadata as md

from packaging.requirements import Requirement

pins = {}
for dist in md.distributions():
    name = (dist.metadata["Name"] or "").lower()
    if not name.startswith("isaacsim"):
        continue
    for line in dist.requires or []:
        req = Requirement(line)
        if req.marker is not None or req.name.lower().startswith(("isaacsim", "torch", "pillow")):
            continue
        spec = str(req.specifier)
        if spec.startswith("==") and "," not in spec:
            pins[req.name.lower()] = spec
for name, spec in sorted(pins.items()):
    print(f"{name}{spec}")
