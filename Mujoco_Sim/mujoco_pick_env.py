from Custom_Mujoco_Playground._src.manipulation.franka_emika_panda.pick import (
    PandaPickCube,
    PandaPickCubeOrientation,
)
from Custom_Mujoco_Playground._src.manipulation.franka_emika_panda.pick import (
    default_config,
)
from ml_collections import config_dict
from typing import Any, Dict, Optional, Union
import numpy as np
import numpy.typing as npt
from dm_env import StepType
import cv2
import time
import torch
from MAE_Model.prepare_input import Prepare
from gymnasium.spaces import Box
import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco.mjx._src import math
from Custom_Mujoco_Playground._src import mjx_env

# Madrona MJX batch renderer
from madrona_mjx.renderer import BatchRenderer # type: ignore
_GLOBAL_MADRONA_RENDERER = None
_GLOBAL_MJX_MODEL = None

"""
mjx_panda.xml has the content of panda_updated_robotiq_2f85.xml
"""

class StereoPickCube(PandaPickCube):
    def __init__(
        self,
        render_mode: str = "human",
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        sample_orientation: bool = False,
        img_h_size: int = 64,
        img_w_size: int = 64,
        discount: float = 0.99,
        max_path_length: int = 300,  # Horizon
    ):
        self.img_h_size = img_h_size
        self.img_w_size = img_w_size
        self.render_mode = render_mode
        self.discount = discount
        self.config = self.custom_config(max_path_length)

        # Let the base PandaPickCube build its mj_model / mjx_model etc.
        super().__init__(self.config, config_overrides, sample_orientation)

        # Vision config
        self._vision_config = config_dict.create(
            gpu_id=0,
            render_batch_size=1,
            render_width=self.img_h_size,
            render_height=self.img_w_size,
            use_rasterizer=True,      # Change this from False to True for faster
            enabled_geom_groups=[0, 1, 2],
        )

        # Observation / action spaces as before
        self.observation_space = Box(
            low=np.float32(-4.0),
            high=np.float32(4.0),
            shape=(self.img_h_size, 2 * self.img_w_size, 3),
            dtype=np.float32,
        )
        n_ctrl = int(self._init_ctrl.shape[0])
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(n_ctrl,),
            dtype=np.float32,
        )

        # ---------- Global shared MJX model + renderer ----------
        global _GLOBAL_MJX_MODEL, _GLOBAL_MADRONA_RENDERER

        if _GLOBAL_MJX_MODEL is None:
            # First env: register its mjx.Model as the global one
            _GLOBAL_MJX_MODEL = self._mjx_model
        else:
            # Subsequent envs: force them to use the shared model
            # (optionally, you could check xml_path here)
            self._mjx_model = _GLOBAL_MJX_MODEL

        # JAX MJX step uses the (possibly overridden) shared model
        self._mjx_step = self.make_mjx_step(self._mjx_model, self.n_substeps)

        # Create Madrona BatchRenderer only once, on the shared MJX model
        if _GLOBAL_MADRONA_RENDERER is None:
            _GLOBAL_MADRONA_RENDERER = BatchRenderer(
                m=_GLOBAL_MJX_MODEL,
                gpu_id=self._vision_config.gpu_id,
                num_worlds=self._vision_config.render_batch_size,
                batch_render_view_width=self._vision_config.render_width,
                batch_render_view_height=self._vision_config.render_height,
                enabled_geom_groups=np.asarray(
                    self._vision_config.enabled_geom_groups, dtype=np.int32
                ),
                enabled_cameras=None,  # use all cameras
                add_cam_debug_geo=False,
                use_rasterizer=self._vision_config.use_rasterizer,
                viz_gpu_hdls=None,
            )

        # Per-env handle + per-env token/data
        self.renderer = _GLOBAL_MADRONA_RENDERER
        self._latest_data = None
        self._render_token = None

        # These are just used for naming / visualization
        self.left_cam_name, self.right_cam_name = "left1", "right1"

        self.step_count = 0
        self.max_episode_length = max_path_length

    def reset(self, rng: jax.Array) -> dict:
        rng, rng_box, rng_target = jax.random.split(rng, 3)

        # initialize box position
        box_pos = (
            jax.random.uniform(
                rng_box,
                (3,),
                minval=jp.array([-0.2, -0.2, 0.0]),
                maxval=jp.array([0.2, 0.2, 0.0]),
            )
            + self._init_obj_pos
        )

        # initialize target position
        target_pos = (
            jax.random.uniform(
                rng_target,
                (3,),
                minval=jp.array([-0.2, -0.2, 0.2]),
                maxval=jp.array([0.2, 0.2, 0.4]),
            )
            + self._init_obj_pos
        )

        target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        if self._sample_orientation:
            # sample a random direction
            rng, rng_axis, rng_theta = jax.random.split(rng, 3)
            perturb_axis = jax.random.uniform(rng_axis, (3,), minval=-1, maxval=1)
            perturb_axis = perturb_axis / math.norm(perturb_axis)
            perturb_theta = jax.random.uniform(rng_theta, maxval=np.deg2rad(45))
            target_quat = math.axis_angle_to_quat(perturb_axis, perturb_theta)

        # initialize data
        init_q = (
            jp.array(self._init_q)
            .at[self._obj_qposadr : self._obj_qposadr + 3]
            .set(box_pos)
        )
        data = mjx_env.make_data(
            self._mjx_model,
            qpos=init_q,
            qvel=jp.zeros(self._mjx_model.nv, dtype=float),
            ctrl=self._init_ctrl,
            impl=self._mjx_model.impl.value,
        )

        # set target mocap position
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
            mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat),
        )

        # initialize env state and info
        metrics = {
            "out_of_bounds": jp.array(0.0, dtype=float),
            **{k: 0.0 for k in self._config.reward_config.scales.keys()},
        }
        info = {
            "rng": rng,
            "target_pos": target_pos,
            "reached_box": 0.0,
        }

        # Reset Madrona token & cache latest data for rendering
        self._latest_data = data
        self._render_token = None

        obs = self._get_img_obs()  # uses Madrona on GPU

        reward, done = jp.zeros(2)
        self.step_type = StepType.FIRST
        self.step_count = 0

        return {
            "data": data,
            "metrics": metrics,
            "observation": obs,
            "info": info,
            "step_type": self.step_type,
            "action": None,
            "done": done,
            "reward": reward,
            "discount": 0.0 if self.step_type == StepType.LAST else self.discount,
        }

    def step(self, time_step, action: jax.Array) -> dict:
        t0 = time.perf_counter()
        self.step_count += 1
        # time_step can be an ExtendedTimeStep OR a dict.
        # Both support string indexing (ExtendedTimeStep via __getitem__).
        delta = action * self._action_scale
        ctrl = time_step["data"].ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        data = self._mjx_step(time_step["data"], ctrl)

        raw_rewards = self._get_reward(data, time_step["info"])
        sanitized_raw = {}  # Sanitize rewards to take out NaN
        for k, v in raw_rewards.items():
            v = jp.where(jp.isnan(v) | jp.isinf(v), 0.0, v)
            sanitized_raw[k] = v

        rewards = {
            k: sanitized_raw[k] * self._config.reward_config.scales[k]
            for k in sanitized_raw
        }
        reward_sum = sum(rewards.values())
        reward_sum = jp.where(
            jp.isnan(reward_sum) | jp.isinf(reward_sum), 0.0, reward_sum
        )
        reward = jp.clip(reward_sum, -1e4, 1e4)

        box_pos = data.xpos[self._obj_body]
        out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
        out_of_bounds |= box_pos[2] < 0.0
        truncated = self.step_count >= self.max_episode_length
        terminated = (
            out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        )
        done = float(terminated or truncated)
        self.step_type = StepType.LAST if bool(done) else StepType.MID

        time_step["metrics"].update(
            **raw_rewards, out_of_bounds=out_of_bounds.astype(float)
        )

        # Cache latest MJX data for GPU rendering
        self._latest_data = data

        t1 = time.perf_counter()
        with open("debugs.txt", "a", encoding="utf-8") as f:
            f.write(f"Take action and update physics: {t1 - t0} s \n")

        obs = self._get_img_obs()

        return {
            "data": data,
            "metrics": time_step["metrics"],
            "observation": obs,
            "info": time_step["info"],
            "step_type": self.step_type,
            "action": None,
            "done": done,
            "reward": reward,
            "discount": 0.0 if self.step_type == StepType.LAST else self.discount,
        }

    @staticmethod
    def make_mjx_step(mjx_model, n_substeps: int):
        """
        Build a jitted step function for MJX.
        We pass model and n_substeps via closure so the JAX signature is just (data, ctrl).
        """
        devices = jax.devices()
        print("[make_mjx_step] JAX devices:", devices)
        if not any(d.platform == "gpu" for d in devices):
            print("[make_mjx_step][WARNING] No GPU detected; MJX will run on CPU.")

        @jax.jit
        def _step(data, ctrl):
            # Pure: no side effects, only depends on inputs & closed-over constants
            return mjx_env.step(mjx_model, data, ctrl, n_substeps)

        return _step

    def _get_img_obs(self) -> np.ndarray:
        """
        Returns the image, both views fused, in numpy array format
            np.ndarray, shape (H, 2*W, 3), dtype float32 roughly [-4, 4]

        Uses Madrona MJX BatchRenderer, keeping rendering on GPU and only
        pulling the final RGB frames back to host.
        """
        if self._latest_data is None:
            raise RuntimeError(
                "_get_img_obs called before _latest_data was set. "
                "Make sure reset/step set self._latest_data."
            )

        t0 = time.perf_counter()

        # Initialize or render with Madrona depending on whether we have a token
        if self._render_token is None:
            # First time: initialize Madrona with data + model
            render_token, rgb, _ = self.renderer.init(
                self._latest_data, self._mjx_model
            )
            self._render_token = render_token
        else:
            # Subsequent calls: render requires (token, data, model)
            _, rgb, _ = self.renderer.render(
                self._render_token, self._latest_data, self._mjx_model
            )
        t_render = time.perf_counter()

        # rgb is a JAX array on device; bring to host for PyTorch / OpenCV
        rgb_np = np.asarray(rgb)

        # Heuristic handling of possible shapes:
        #  - (num_worlds, H, W, 4)            -> single camera
        #  - (num_worlds, num_cams, H, W, 4)  -> multi-camera, we use first two
        if rgb_np.ndim == 4:
            # Assume (num_worlds, H, W, 4)
            world0 = rgb_np[0]  # (H, W, 4)
            img0 = world0[..., :3]  # RGB
            # No true stereo views from renderer -> duplicate
            left_u8 = img0.astype(np.uint8)
            right_u8 = img0.astype(np.uint8)
        elif rgb_np.ndim == 5:
            # Assume (num_worlds, num_cams, H, W, 4)
            world0 = rgb_np[0]  # (num_cams, H, W, 4)
            num_cams = world0.shape[0]
            if num_cams >= 2:
                left_u8 = world0[0, ..., :3].astype(np.uint8)
                right_u8 = world0[1, ..., :3].astype(np.uint8)
            else:
                img0 = world0[0, ..., :3]
                left_u8 = img0.astype(np.uint8)
                right_u8 = img0.astype(np.uint8)
        else:
            raise ValueError(
                f"Unexpected Madrona RGB shape {rgb_np.shape}; expected 4D or 5D."
            )

        # Convert to tensors exactly like before, fuse + normalize with your helper
        left_t = (
            torch.from_numpy(left_u8).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        right_t = (
            torch.from_numpy(right_u8).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )

        stereo_t = Prepare.fuse_normalize([left_t, right_t])  # (1, H, 2W, C)
        stereo_np = stereo_t[0].detach().cpu().numpy().astype(np.float32, copy=False)

        if getattr(self, "render_mode", None) == "human":
            stereo_image = np.concatenate([left_u8, right_u8], axis=1)
            enlarged = cv2.resize(
                stereo_image,
                None,
                fx=5.0,
                fy=5.0,
                interpolation=cv2.INTER_NEAREST,
            )
            bgr = cv2.cvtColor(enlarged, cv2.COLOR_RGB2BGR)
            cv2.imshow("Stereo view", bgr)
            cv2.waitKey(1)

        t1 = time.perf_counter()
        with open("debugs.txt", "a", encoding="utf-8") as f:
            f.write(f"Entire rendering (Madrona + preprocessing): {t1 - t0} s \n")
            f.write(f"Madrona render call: {t_render - t0} s \n")

        return stereo_np

    def custom_config(self, max_path_length):
        """Returns the default config for bring_to_target tasks."""
        config = config_dict.create(
            ctrl_dt=0.02,
            sim_dt=0.005,
            episode_length=max_path_length,
            action_repeat=1,
            action_scale=0.04,
            reward_config=config_dict.create(
                scales=config_dict.create(
                    gripper_box=4.0,  # Gripper goes to the box.
                    box_target=8.0,  # Box goes to the target mocap.
                    no_floor_collision=0.25,  # Do not collide the gripper with the floor.
                    robot_target_qpos=0.3,  # Arm stays close to target pose.
                )
            ),
            impl="jax",
            nconmax=24 * 2048,
            njmax=128,
        )
        return config