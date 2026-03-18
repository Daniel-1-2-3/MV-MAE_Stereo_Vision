---
hide-toc: true
firstpage:
lastpage:
---

```{project-logo} ../Metaworld_Stereo-text-banner.svg
:alt: Metaworld_Stereo Logo
```

```{project-heading}
Meta-World is an open-source simulated benchmark for meta-reinforcement learning and multi-task learning consisting of 50 distinct robotic manipulation tasks.
```

```{figure} _static/mt10.gif
   :alt: REPLACE ME
   :width: 500
```

**Basic example:**

```{code-block} python
import gymnasium as gym
import Metaworld_Stereo

env = gym.make('Meta-World/MT1', env_name='reach-v3')

obs = env.reset()
a = env.action_space.sample()
next_obs, reward, terminate, truncate, info = env.step(a)

```

```{toctree}
:hidden:
:caption: Introduction

introduction/basic_usage
evaluation/evaluation
installation/installation
rendering/rendering
```

```{toctree}
:hidden:
:caption: Benchmark Information
benchmark/environment_creation
benchmark/action_space
benchmark/state_space
benchmark/benchmark_descriptions
benchmark/task_descriptions.md
benchmark/reward_functions
benchmark/expert_trajectories
benchmark/resetting
```

```{toctree}
:hidden:
:caption: Development

Github <https://github.com/Farama-Foundation/Metaworld_Stereo>
citation
release_notes/index
Contribute to the Docs <https://github.com/Farama-Foundation/Metaworld_Stereo/blob/main/docs/README.md>
```
