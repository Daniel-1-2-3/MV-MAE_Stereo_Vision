import jax

# ---- Orbax / JAX compatibility patch (tree API shim) ----
# Some Orbax versions expect `jax.tree.map_with_path`, while some JAX versions
# only provide this under `jax.tree_util.map_with_path`.
if not hasattr(jax, "tree"):
    # Provide a namespace object to hang attributes on.
    class _JaxTreeNamespace:
        pass
    jax.tree = _JaxTreeNamespace()

if not hasattr(jax.tree, "map_with_path"):
    # Prefer the canonical location if available.
    if hasattr(jax, "tree_util") and hasattr(jax.tree_util, "map_with_path"):
        jax.tree.map_with_path = jax.tree_util.map_with_path
    else:
        # Last-resort fallback: emulate map_with_path using tree flatten/unflatten.
        # This is not perfect, but is usually enough for checkpoint traversal.
        from jax import tree_util as _tu

        def _map_with_path(f, pytree, *rest):
            leaves, treedef = _tu.tree_flatten(pytree)
            rest_leaves = []
            for r in rest:
                rl, rt = _tu.tree_flatten(r)
                if rt != treedef:
                    raise ValueError("map_with_path fallback requires matching tree structure.")
                rest_leaves.append(rl)

            # Best-effort "path": use leaf index as a stand-in.
            out_leaves = [
                f((i,), leaf, *[rl[i] for rl in rest_leaves])
                for i, leaf in enumerate(leaves)
            ]
            return _tu.tree_unflatten(treedef, out_leaves)

        jax.tree.map_with_path = _map_with_path
# ----------------------------------------

from enum import Enum

# ---- Orbax / JAX compatibility patch (sharding AxisType shim) ----
if not hasattr(jax.sharding, "AxisType"):
    class AxisType(Enum):
        REPLICATED = "replicated"
        SHARDED = "sharded"
        AUTO = "auto"

    jax.sharding.AxisType = AxisType
# ----------------------------------------

from absl import app
from Custom_Mujoco_Playground.learning import train_jax_ppo

def main():
    app.run(train_jax_ppo.main)