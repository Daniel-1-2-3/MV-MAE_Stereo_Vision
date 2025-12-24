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
from train_drqv2_mujoco import Workshop, get_args, save_agent
#from Custom_Mujoco_Playground.learning import train_jax_ppo
from Custom_Mujoco_Playground.learning import train_rsl_rl

def main():
    #app.run(train_jax_ppo.main)
    app.run(train_rsl_rl.main)