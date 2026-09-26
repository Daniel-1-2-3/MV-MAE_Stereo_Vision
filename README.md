# MV-MAE + DrQ-v2: picking a suture needle off soft tissue with a stereo endoscope

A da Vinci surgical arm (dVRK PSM) learns to pick up a suture needle lying on soft
tissue and lift it to a target point, **using only the two images of a stereo
camera**, the way a real da Vinci sees through its stereo endoscope.

* **Simulator:** NVIDIA Isaac Sim 5.1 + Isaac Lab 2.3.0. The needle-lift task and the
  robot come from [Isaac for Healthcare](https://github.com/isaac-for-healthcare/i4h-workflows)
  v0.4.0 (originally [ORBIT-Surgical](https://github.com/orbit-surgical/orbit-surgical)).
  This project replaces the table with a soft tissue pad (a finite-element soft body)
  and adds the stereo camera.
* **Vision:** MV-MAE (multi-view masked autoencoder, [Seo et al. 2023](https://arxiv.org/abs/2302.02408)).
  It learns to understand both camera views by redrawing one eye's picture from the
  other eye and from the previous frames.
* **Control:** DrQ-v2 ([Yarats et al. 2021](https://arxiv.org/abs/2107.09645)), an
  actor-critic method in the SAC family built for learning from images.

## What's in the repo

| Path | What it does |
|---|---|
| `surgical_env/env_cfg.py` | The task: PSM arm, needle, soft tissue pad, stereo cameras, rewards |
| `surgical_env/stereo.py` | Stereo camera geometry: placement, projection, expected disparity |
| `surgical_env/env.py` | Wraps the Isaac Lab env so it returns stereo images `(N, 2, 3, 96, 96)` |
| `surgical_env/scripted.py` | Scripted pick-and-lift controller (used for demonstrations and in the test) |
| `mvmae/model.py` | MV-MAE: conv stem, view masking, ViT encoder/decoder, reward head |
| `agent/drqv2.py` | DrQ-v2 actor/critic on top of MV-MAE, with optional learning from demonstrations |
| `agent/replay.py` | Replay memory on the GPU with n-step returns and frame stacking |
| `trainer.py` | Training loop, evaluation and wandb logging |
| `test_stereo.py` | **Checks the stereo rig and the tissue scene** (run this first) |
| `record_demos.py` | Records successful runs of the scripted controller |
| `train.py` / `eval.py` | Train / evaluate |
| `config.py`, `configs/needle_tissue.yaml` | All settings |
| `tests/` | CPU unit tests for everything that doesn't need the simulator |

## Requirements

* Linux x86_64, Ubuntu 22.04 or 24.04 (for example a RunPod pod).
* An NVIDIA **RTX** GPU (with ray-tracing cores) and at least 24 GB of memory.
  L40S, RTX 6000 Ada, RTX A6000, RTX 4500/PRO 4500 and RTX 4090 work. **A100/H100/H200 do
  not render cameras**, because they have no RT cores.
* NVIDIA driver 535 or newer (Isaac Lab recommends 580+). Blackwell cards such as the
  RTX PRO 4500 need 570 or newer.
* About 40 GB of disk space, and at least 16 CPU cores for a comfortable run.
* Python 3.11. Isaac Sim 5.x requires exactly this version.

Before installing, check that the GPU can render. Isaac Sim needs Vulkan, not just CUDA:

```bash
nvidia-smi                                   # GPU name and driver version
sudo apt-get install -y vulkan-tools && vulkaninfo --summary   # must list your GPU
```

If `vulkaninfo` does not list the GPU on RunPod, add `NVIDIA_DRIVER_CAPABILITIES=all` to
the pod template's environment variables, or pick a different machine.

## Install

```bash
git clone https://github.com/Daniel-1-2-3/MV-MAE_Stereo_Vision.git
cd MV-MAE_Stereo_Vision

# Python 3.11 environment (conda shown; a python3.11 venv also works)
conda create -n mvmae python=3.11 -y
conda activate mvmae

# Isaac Sim asks you to accept the NVIDIA Omniverse license:
# read https://docs.omniverse.nvidia.com/eula, then
export OMNI_KIT_ACCEPT_EULA=YES

bash install.sh        # 20-40 min: Isaac Sim 5.1.0, Isaac Lab v2.3.0, PyTorch 2.7.0+cu128, ...
```

`install.sh` installs into the active environment, clones Isaac Lab into
`third_party/IsaacLab`, and prints the installed versions. When run as root on apt-based
images (for example RunPod), it also installs the few system libraries Isaac Sim needs.

## Step 1: check the stereo setup

```bash
python test_stereo.py
```

The first run downloads the robot and needle models (from NVIDIA's public asset
bucket, into `~/.cache/i4h-assets`). Isaac Sim also compiles its shaders the first
time, so allow 5–10 minutes before anything happens. Later runs take about a minute.

It builds 4 copies of the scene, lets the needle settle on the tissue, and prints
`PASS`/`FAIL` for each check:

| Check | What it verifies |
|---|---|
| `images/*` | Images have the right shape, aren't blank, and left and right are similar but not identical |
| `rig/*` | The cameras are where they were designed to be: 5 mm apart, facing the same way, right camera on the right, correct focal length |
| `projection/*` | The needle and tool tip appear in **both** images, on the **same pixel row**, shifted sideways by exactly the amount the geometry predicts, and nothing blocks the view of them |
| `stereo/depth_warp` | An image-only check: shifting the right image by the disparity that the depth map predicts reproduces the left image better than no shift, the opposite shift, or a vertical shift |
| `tissue/*` | The needle rests on the tissue (doesn't sink through or bounce), the bottom of the pad stays pinned, and the tissue dents by more than 1 mm when the tool presses into it |

It then writes to `outputs/stereo_check/`:

* `stereo_check.png`: one row per environment showing left | right | red-cyan 3D
  overlay | depth. The needle is marked yellow and the tool tip cyan. Look at the
  framing here.
* `stereo_check.mp4`: environment 0 through the whole test (settling, then the tool
  pressing into the tissue).
* `results.json`: every number behind the checks.

The exit code is 0 only if everything passes. Useful options: `--gui` watches the run in
the Isaac Sim window (on a machine with a display), `--num_envs 2`, and any config
override, such as `camera.baseline=0.008`.

## Step 2 (optional but recommended): record demonstrations

```bash
python record_demos.py --num_demos 50 --out demos/needle_demos.pt
```

This runs the scripted controller in every environment and keeps only successful
episodes. It prints the controller's success rate and saves a video of one demo next to
the `.pt` file. The paper pre-fills its replay memory with demonstrations in the same
way; for a thin needle grabbed from images, this helps a lot.

## Step 3: train

```bash
wandb login                                   # once
python train.py train.demo_path=demos/needle_demos.pt log.run_name=first_run
```

Without demonstrations, run just `python train.py`. Settings can be overridden as
`section.key=value` (see `config.py` for every option). Examples:
`env.num_envs=16`, `train.replay_capacity=60000` (if GPU memory runs out),
`log.wandb_mode=offline` (no internet; upload later with `wandb sync`).

Checkpoints, evaluation videos and MV-MAE reconstruction images go to `runs/<run_name>/`.

### What gets logged to wandb

* **train/**: episode return and length, success (final state), success at any point in
  the episode, final needle-to-goal distance, and fraction of episodes that ended by
  dropping the needle. `train/isaac/*` has Isaac Lab's per-reward-term episode sums and
  termination counts; `train/isaac/Episode_Termination/physics_blowup` counts episodes
  cut short because the simulation glitched (should stay at or near 0). Also the exploration noise, encoder learning rate, replay size,
  and env steps / updates.
* **eval/** (every 50k steps): the same episode metrics from a deterministic policy,
  plus `eval/video` with the stereo view (left | right) of one episode.
* **critic/**, **actor/**: Q-values, target Q, critic loss, actor loss, and
  behaviour-cloning loss when demos are used.
* **mvmae/**: reconstruction loss, reward-prediction loss, and a
  `mvmae/reconstruction` image. Each row of the image is one frame; the columns are
  truth L, truth R, what the encoder saw (grey = hidden), and reconstruction L/R.
* **grad/**: gradient norms of the encoder, critic and actor. **pretrain/**: the MV-MAE
  warm-up phase.
* **speed/**, **time/**, **system/**: env steps and updates per second, the time split
  between simulation, acting and learning, and GPU memory.

## Step 4: evaluate

```bash
python eval.py --checkpoint runs/first_run/ckpt_final.pt --rounds 2
```

This writes one stereo video per round and a `metrics.json` to `outputs/eval/`.

## The task

| | |
|---|---|
| Robot | dVRK Patient Side Manipulator (one da Vinci instrument arm) with a gripper |
| Observation | Last 3 frames from both cameras, 96×96 RGB each: `(3 frames, 2 views, 3, 96, 96)` |
| Action | 7 numbers in [-1, 1]: move the tool tip up to 5 mm and rotate it up to 0.05 rad per step (relative to where it is), plus open (≥ 0) or close (< 0) the gripper |
| Control rate | 25 Hz; episodes last 5 s (125 steps) |
| Scene | 10 × 10 × 1 cm soft tissue pad (Young's modulus 50 kPa, bottom pinned) on a rigid platform. The needle is dropped onto it at a random spot within ±3 cm |
| Goal | Lift the needle to a fixed point 4 cm above the centre of the pad |
| Rewards | Unchanged from Isaac for Healthcare: reach the needle (×1), needle lifted 2 cm (×15), needle near the goal (×16 coarse, ×5 fine), and small penalties for jerky actions and fast joints that are ramped up after 10k steps |
| Success | Needle lifted and within 2 cm of the goal |
| Cameras | Stereo pair with a 5 mm gap (like a da Vinci endoscope), 17 cm from a point 3.5 cm above the tissue centre, 35° above horizontal, 47° field of view. The tool's start position, the goal and the whole needle area are in view; disparity is about 2.5–4 px |

**Changes from the original Isaac for Healthcare task, and why:**

* The needle lies on **soft tissue** instead of a table, with "lifted" measured from the
  tissue surface.
* **Fixed goal.** The original goal moves every second, but the policy only sees
  images, and a moving goal would be invisible to it.
* **25 Hz control with small per-step motions** (5 mm / 0.05 rad) instead of 50 Hz with
  up to 0.5 m per step. The original works for PPO with exact state inputs; exploring
  from images needs millimetre-scale motions.
* **5 s episodes** instead of 2 s, so there is time to find and grab the needle from
  images.

## The learning method

1. **MV-MAE.** A shared convolutional stem turns each 96×96 image into a 6×6 grid of
   feature "tokens". Each token is tagged with where it is in the image, which camera it
   came from, and which frame. During training, for every frame one camera's view is
   hidden entirely and most of the other camera's tokens are hidden too (95% of all
   tokens). The transformer then has to redraw everything and predict the reward. That
   forces it to learn what the scene looks like from both eyes, and how things move
   between frames.
2. **DrQ-v2.** The encoder's output for the full, unhidden stereo clip is the robot's
   "state". Two critics estimate how good an action is, and an actor picks actions. The
   images are randomly shifted a few pixels during training, with the same shift for
   both eyes so their depth relationship is kept. Returns look 3 steps ahead.
3. **Training the encoder.** The critic's error and the MV-MAE losses both update the
   encoder. Before reinforcement learning starts, the MV-MAE warms up for 2,000 updates
   on the initial random-action data.
4. **Demonstrations (optional).** 25% of every batch comes from recorded demos, and the
   actor gets an extra behaviour-cloning term on those samples, balanced against the
   critic as in TD3+BC.

## Troubleshooting

| Symptom | Try |
|---|---|
| `tissue/needle_on_surface` or `needle_settled` fails (the needle sinks or jitters) | `env.needle_asset=mesh` (plain-mesh needle), or a finer tissue mesh: `env.tissue_hex_resolution=16` |
| `tissue/dents_when_pressed` fails | Softer tissue: `env.tissue_youngs_modulus=2e4`. Check the reported tool-tip distance from its target |
| `projection/*_not_occluded` or `*_in_view` fails | Move the cameras: `camera.azimuth_deg`, `camera.elevation_deg`, `camera.distance` |
| `tissue/bottom_pinned` fails | Run with `env.pin_tissue_bottom=false` and see whether `tissue/no_sliding` still passes |
| CUDA out of memory | Lower `train.replay_capacity` (each transition is ~55 KB), `env.num_envs`, or `agent.batch_size` |
| "A camera was spawned without the --enable_cameras flag" | The scripts turn this on themselves; make sure you run them from this directory with `python <script>.py` |
| Nothing renders / black images on a cloud GPU | `vulkaninfo --summary` must list the GPU (see Requirements) |

## Tests (no GPU needed)

```bash
pytest          # rotations, stereo geometry, scripted controller, MV-MAE, replay memory, agent, full training loop on a fake env
```

## Credits and licences

* Needle-lift task, dVRK PSM configuration, reward functions and scripted controller:
  Isaac for Healthcare v0.4.0 / ORBIT-Surgical (BSD-3-Clause, Apache-2.0). The license
  headers are kept in the adapted files.
* DrQ-v2 components: [facebookresearch/drqv2](https://github.com/facebookresearch/drqv2) (MIT).
* MV-MAE: Seo, Kim, James, Lee, Shin, Abbeel, *Multi-View Masked World Models for Visual
  Robotic Manipulation*, ICML 2023.
