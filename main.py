import jax
from enum import Enum

# ---- Orbax / JAX compatibility patch ----
if not hasattr(jax.sharding, "AxisType"):
    class AxisType(Enum):
        REPLICATED = "replicated"
        SHARDED = "sharded"
        AUTO = "auto"

    jax.sharding.AxisType = AxisType
# ----------------------------------------

from pathlib import Path
from absl import app
import train_drqv2_mujoco
def main():
    train_drqv2_mujoco.main()