"""Base classes for all the envs."""

from __future__ import annotations

from functools import cached_property
from typing import Any, Literal, SupportsFloat

import torch
import torch.nn.functional as F
import cv2
import time
import mujoco
import random
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from gymnasium.utils import seeding
from gymnasium.utils.ezpickle import EzPickle
from typing_extensions import TypeAlias
from dm_env import StepType

from Outdated.CustomMetaworld.metaworld.types import XYZ, ObservationDict
from Outdated.CustomMetaworld.metaworld.sawyer_xyz_env import SawyerMocapBase
from Outdated.CustomMetaworld.metaworld.utils import reward_utils
from MAE_Model.prepare_input import Prepare

RenderMode: TypeAlias = Literal['human', 'rgb_array', 'depth_array']

class SawyerXYZEnv(SawyerMocapBase, EzPickle):
    """The base environment for all Sawyer Mujoco envs that use mocap for XYZ control."""

    _HAND_SPACE = Box( # Constraints for hand positions
        np.array([-0.525, 0.348, -0.0525]),
        np.array([+0.525, 1.025, 0.7]),
        dtype=np.float32,
    )
    TARGET_RADIUS: float = 0.05 # Upper bound for distance from the target when checking for task completion
    max_path_length: int = 500 # Maximum episode length (task horizon)

    def __init__(
        self,
        frame_skip: int = 5,
        hand_low: XYZ = (-0.2, 0.55, 0.05),
        hand_high: XYZ = (0.2, 0.75, 0.3),
        mocap_low: XYZ | None = None,
        mocap_high: XYZ | None = None,
        action_scale: float = 1.0 / 100,
        action_rot_scale: float = 1.0,
        render_mode: RenderMode | None = None,
        camera_id: int | None = None,
        camera_pairs: list | None = None,
        img_width: int = 84,
        img_height: int = 84,
        discount: float = 0.99,
    ):
        self.action_scale = action_scale
        self.action_rot_scale = action_rot_scale
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)
        if mocap_low is None:
            mocap_low = hand_low
        if mocap_high is None:
            mocap_high = hand_high
        self.mocap_low = np.hstack(mocap_low)
        self.mocap_high = np.hstack(mocap_high)
        self.curr_path_length: int = 0
        self.seeded_rand_vec: bool = False
        self._last_rand_vec: npt.NDArray[Any] | None = None
        self.num_resets: int = 0
        self.current_seed: int | None = None
        self.obj_init_pos: npt.NDArray[Any] | None = None
        
        self._partially_observable = False

        self.width = img_width
        self.height = img_height
        self.discount = discount
        
        self._obs_obj_max_len = 14
        
        super().__init__(
            self.model_name,
            frame_skip=frame_skip,
            render_mode=render_mode,
            camera_name=None,
            camera_id=None,
            width=img_width,
            height=img_height,
        )

        mujoco.mj_forward(self.model, self.data)  # DO NOT REMOVE: EZPICKLE WON'T WORK

        self._did_see_sim_exception: bool = False # If sim hits an unstable physics state
        self.init_left_pad: npt.NDArray[Any] = self.get_body_com("leftpad") # Initial center of mass
        self.init_right_pad: npt.NDArray[Any] = self.get_body_com("rightpad")

        self.observation_space = Box(
            low=np.float32(-4.0), 
            high=np.float32(4.0), 
            shape=(self.height, 2 * self.width, 3), 
            dtype=np.float32
        )
        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([+1, +1, +1, +1]),
            dtype=np.float32,
        )
        
        self.hand_init_pos: npt.NDArray[Any] | None = None  # OVERRIDE ME
        self._target_pos: npt.NDArray[Any] | None = None  # OVERRIDE ME
        self._random_reset_space: Box | None = None  # OVERRIDE ME
        self.goal_space: Box | None = None  # OVERRIDE ME
        self._last_stable_obs = None

        self.init_qpos = np.copy(self.data.qpos)
        self.init_qvel = np.copy(self.data.qvel)
        self._prev_obs = self._get_curr_obs_combined_no_goal()
        
        EzPickle.__init__(
            self,
            self.model_name,
            frame_skip,
            hand_low,
            hand_high,
            mocap_low,
            mocap_high,
            action_scale,
            action_rot_scale,
        )
        
        self.camera_pairs = camera_pairs
        self.left_cam_name, self.right_cam_name = random.choice(self.camera_pairs)
        self.render_mode = render_mode

        # Fast offscreen renderer
        self._mjv_cam = mujoco.MjvCamera()
        self._mjv_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._mjv_opt = mujoco.MjvOption()
        self._mjv_scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self._gl_ctx = None
        if hasattr(mujoco, "GLContext") and mujoco.GLContext is not None:
            self._gl_ctx = mujoco.GLContext(int(self.width), int(self.height))
            self._gl_ctx.make_current()
        else:
            raise RuntimeError(
                "MuJoCo GLContext unavailable. Ensure MUJOCO_GL is set (e.g., 'egl' or 'osmesa') "
                "BEFORE importing mujoco."
            )
        self._mjr_ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._mjr_ctx)
        self._mjr_ctx.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR
        self._mjr_rect = mujoco.MjrRect(0, 0, int(self.width), int(self.height))
        # Preallocate CPU buffers used by mjr_readPixels (uint8, HxWx3)
        self._rgb_left  = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self._rgb_right = np.empty((self.height, self.width, 3), dtype=np.uint8)

    def seed(self, seed: int) -> list[int]:
        """Seeds the environment.
        Args:
            seed: The seed to use.
        Returns:
            The seed used inside a 1 element list.
        """
        assert seed is not None
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        assert self.goal_space
        self.goal_space.seed(seed)
        return [seed]

    def set_xyz_action(self, action: npt.NDArray[Any]) -> None:
        """Adjusts the position of the mocap body from the given action.
        Moves each body axis in XYZ by the amount described by the action.
        Args:
            action: The action to apply (in offsets between :math:`[-1, 1]` for each axis in XYZ).
        """
        action = np.clip(action, -1, 1)
        pos_delta = action * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.mocap_pos = new_mocap_pos
        self.data.mocap_quat = np.array([1, 0, 1, 0])

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        """Sets the position of the object.
        Args:
            pos: The position to set as a numpy array of 3 elements (XYZ value).
        """
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _get_site_pos(self, site_name: str) -> npt.NDArray[np.float32]:
        """Gets the position of a given site.
        Args:
            site_name: The name of the site to get the position of.
        Returns:
            Flat, 3 element array indicating site's location.
        """
        return self.data.site(site_name).xpos.copy()

    def _set_pos_site(self, name: str, pos: npt.NDArray[Any]) -> None:
        """Sets the position of a given site.
        Args:
            name: The site's name
            pos: Flat, 3 element array indicating site's location
        """
        assert isinstance(pos, np.ndarray)
        assert pos.ndim == 1
        self.data.site(name).xpos = pos[:3]

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        """Retrieves site name(s) and position(s) corresponding to env targets."""
        assert self._target_pos is not None
        return [("goal", self._target_pos)]

    @property
    def touching_main_object(self) -> bool:
        """Calls `touching_object` for the ID of the env's main object.

        Returns:
            Whether the gripper is touching the object
        """
        return self.touching_object(self._get_id_main_object())

    def touching_object(self, object_geom_id: int) -> bool:
        """Determines whether the gripper is touching the object with given id.
        Args:
            object_geom_id: the ID of the object in question
        Returns:
            Whether the gripper is touching the object
        """

        leftpad_geom_id = self.data.geom("leftpad_geom").id
        rightpad_geom_id = self.data.geom("rightpad_geom").id

        leftpad_object_contacts = [
            x
            for x in self.data.contact
            if (
                leftpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2)
            )
        ]
        rightpad_object_contacts = [
            x
            for x in self.data.contact
            if (
                rightpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2)
            )
        ]

        leftpad_object_contact_force = sum(
            self.data.efc_force[x.efc_address] for x in leftpad_object_contacts)
        rightpad_object_contact_force = sum(
            self.data.efc_force[x.efc_address] for x in rightpad_object_contacts)
        
        return 0 < leftpad_object_contact_force and 0 < rightpad_object_contact_force

    def _get_id_main_object(self) -> int:
        return self.data.geom("objGeom").id

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        """Retrieves object position(s) from mujoco properties or instance vars.

        Returns:
            Flat array (usually 3 elements) representing the object(s)' position(s)
        """
        # Throw error rather than making this an @abc.abstractmethod so that
        # V1 environments don't have to implement it
        raise NotImplementedError

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        """Retrieves object quaternion(s) from mujoco properties.
        Returns:
            Flat array (usually 4 elements) representing the object(s)' quaternion(s)
        """
        # Throw error rather than making this an @abc.abstractmethod so that
        # V1 environments don't have to implement it
        raise NotImplementedError

    def _get_pos_goal(self) -> npt.NDArray[Any]:
        """Retrieves goal position from mujoco properties or instance vars.

        Returns:
            Flat array (3 elements) representing the goal position
        """
        assert isinstance(self._target_pos, np.ndarray)
        assert self._target_pos.ndim == 1
        return self._target_pos
    
    def _get_state_obs(self) -> npt.NDArray[np.float64]:
        """Frame stacks `_get_curr_obs_combined_no_goal()` and concatenates the goal position to form a single flat observation.

        Returns:
            The flat observation array (39 elements)
        """
        # do frame stacking
        pos_goal = self._get_pos_goal()
        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)
        curr_obs = self._get_curr_obs_combined_no_goal()
        # do frame stacking
        obs = np.hstack((curr_obs, self._prev_obs, pos_goal))
        self._prev_obs = curr_obs
        return obs

    def _get_state_obs_dict(self) -> ObservationDict:
        state_obs = self._get_state_obs()
        return dict(
            state_observation=state_obs,
            state_desired_goal=self._get_pos_goal(),
            state_achieved_goal=state_obs[3:-3],
        )

    def _get_curr_obs_combined_no_goal(self) -> npt.NDArray[np.float64]:
        """Combines the end effector's {pos, closed amount} and the object(s)' {pos, quat} into a single flat observation.

        Note: The goal's position is *not* included in this.

        Returns:
            The flat observation array (18 elements)
        """

        pos_hand = self.get_endeff_pos()

        finger_right, finger_left = (
            self.data.body("rightclaw"),
            self.data.body("leftclaw"),
        )
        # the gripper can be at maximum about ~0.1 m apart.
        # dividing by 0.1 normalized the gripper distance between
        # 0 and 1. Further, we clip because sometimes the grippers
        # are slightly more than 0.1m apart (~0.00045 m)
        # clipping removes the effects of this random extra distance
        # that is produced by mujoco

        gripper_distance_apart = np.linalg.norm(finger_right.xpos - finger_left.xpos)
        gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)

        obs_obj_padded = np.zeros(self._obs_obj_max_len)
        obj_pos = self._get_pos_objects()
        assert len(obj_pos) % 3 == 0
        obj_pos_split = np.split(obj_pos, len(obj_pos) // 3)

        obj_quat = self._get_quat_objects()
        assert len(obj_quat) % 4 == 0
        obj_quat_split = np.split(obj_quat, len(obj_quat) // 4)
        obs_obj_padded[: len(obj_pos) + len(obj_quat)] = np.hstack(
            [np.hstack((pos, quat)) for pos, quat in zip(obj_pos_split, obj_quat_split)]
        )
        return np.hstack((pos_hand, gripper_distance_apart, obs_obj_padded))
   
    def _render_one_camera(self, cam_name: str, out_buf: np.ndarray) -> np.ndarray:
        """
        Render a fixed camera into preallocated uint8 buffer, matching DeepMind Renderer output.
        out_buf: (H, W, 3) uint8; returned buffer is vertically flipped in place.
        """
        # Keep EGL/OSMesa context current before GL calls
        if getattr(self, "_gl_ctx", None) is not None:
            self._gl_ctx.make_current()
    
        cam_id = self.model.camera(cam_name).id
        self._mjv_cam.fixedcamid = int(cam_id)

        mujoco.mjv_updateScene(
            self.model, self.data, self._mjv_opt, None,
            self._mjv_cam, mujoco.mjtCatBit.mjCAT_ALL, self._mjv_scene
        )
        mujoco.mjr_render(self._mjr_rect, self._mjv_scene, self._mjr_ctx)
        mujoco.mjr_readPixels(out_buf, None, self._mjr_rect, self._mjr_ctx)

        return out_buf

    def _get_img_obs(self) -> np.ndarray:
        """
        Returns the image, both views fused, in numpy array format
            np.ndarray, shape (H, 2*W, 3), dtype float32 roughly [-4, 4]
        """
        
        # Render both views with the fast pathway
        left_u8  = self._render_one_camera(self.left_cam_name,  self._rgb_left)
        right_u8 = self._render_one_camera(self.right_cam_name, self._rgb_right)
        
        # Convert to tensors exactly like before, fuse + normalize with your helper
        left_t  = torch.from_numpy(left_u8).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        right_t = torch.from_numpy(right_u8).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        stereo_t = Prepare.fuse_normalize([left_t, right_t])  # (1, H, 2W, C)
        stereo_np = stereo_t[0].detach().cpu().numpy().astype(np.float32, copy=False)

        if getattr(self, "render_mode", None) == "human":
            stereo_image = np.concatenate([left_u8, right_u8], axis=1)
            enlarged = cv2.resize(stereo_image, None, fx=5.0, fy=5.0, interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Stereo view", enlarged)
            cv2.waitKey(1)

        return stereo_np

    def step(self, action: npt.NDArray[np.float32]):
        """
        Args:
            action: The action to take. Must be a 4 element array of floats.
        Returns:
            The (next_obs, reward, terminated, truncated, info) tuple.
        """
        
        assert len(action) == 4, f"Actions should be size 4, got {len(action)}"
        self.set_xyz_action(action[:3])
        if self.curr_path_length >= self.max_path_length:
            raise ValueError("You must reset the env manually once truncate==True")
        self.do_simulation([action[-1], -action[-1]], n_frames=self.frame_skip)
        self.curr_path_length += 1

        # Running the simulator can sometimes mess up site positions, so
        # re-position them here to make sure they're accurate
        for site in self._target_site_config:
            self._set_pos_site(*site)

        if self._did_see_sim_exception:
            self.step_type = StepType.LAST
            assert self._last_stable_obs is not None
            return (
                self._last_stable_obs,  # observation just before going unstable
                0.0,  # reward (penalize for causing instability)
                False,
                False,  # termination flag always False
                {  # info
                    "success": False,
                    "near_object": 0.0,
                    "grasp_success": False,
                    "grasp_reward": 0.0,
                    "in_place_reward": 0.0,
                    "obj_to_target": 0.0,
                    "unscaled_reward": 0.0,
                },
            )
        mujoco.mj_forward(self.model, self.data)
        
        x = self._get_img_obs().astype(np.float32, copy=False)
        low, high = self.observation_space.low, self.observation_space.high
        self._last_stable_obs = np.clip(x, a_min=low, a_max=high)

        reward, info = self.evaluate_state(None, action)
        # step will never return a terminate==True if there is a success
        # but we can return truncate=True if the current path length == max path length
        truncate = False
        if self.curr_path_length == self.max_path_length:
            truncate = True
        
        self.step_type = StepType.LAST if truncate == True else StepType.MID
        # The var info is a dict containing observation, step_type, action, reward, and discount
        return {
            "observation": self._last_stable_obs,
            "step_type": self.step_type,
            "reward": reward,
            "discount": 0.0 if self.step_type == StepType.LAST else self.discount
        }

    def evaluate_state(self) -> tuple[float, dict[str, Any]]:
        """Does the heavy-lifting for `step()` -- namely, calculating reward and populating the `info` dict with training metrics.

        Returns:
            Tuple of reward between 0 and 10 and a dictionary which contains useful metrics (success,
                near_object, grasp_success, grasp_reward, in_place_reward,
                obj_to_target, unscaled_reward)
        """
        # Throw error rather than making this an @abc.abstractmethod so that
        # V1 environments don't have to implement it
        raise NotImplementedError

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_img_obs()

    def reset_camera_placement(self):
        self.left_cam_name, self.right_cam_name = random.choice(self.camera_pairs)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        """Resets the environment.

        Args:
            seed: The seed to use. Ignored, use `seed()` instead.
            options: Additional options to pass to the environment. Ignored.

        Returns:
            The `(obs, info)` tuple.
        """
        self.curr_path_length = 0
        self.reset_camera_placement()
        obs = self.reset_model()
        self._did_see_sim_exception = False
        self.step_type = StepType.FIRST
        return {
            "observation": obs,
            "step_type": self.step_type,
            "action": None,
            "reward": None,
            "discount": 0.0 if self.step_type == StepType.LAST else self.discount
        }

    def _reset_hand(self, steps: int = 50) -> None:
        """Resets the hand position.

        Args:
            steps: The number of steps to take to reset the hand.
        """
        mocap_id = self.model.body_mocapid[self.data.body("mocap").id]
        for _ in range(steps):
            self.data.mocap_pos[mocap_id][:] = self.hand_init_pos
            self.data.mocap_quat[mocap_id][:] = np.array([1, 0, 1, 0])
            self.do_simulation([-1, 1], self.frame_skip)
        self.init_tcp = self.tcp_center

    def _get_state_rand_vec(self) -> npt.NDArray[np.float32]:
        """Gets or generates a random vector for the hand position at reset."""
        # Removed caching, simply generate a random hand position
        assert self._random_reset_space is not None
        rand_vec: npt.NDArray[np.float32] = np.random.uniform(
            self._random_reset_space.low,
            self._random_reset_space.high,
            size=self._random_reset_space.low.size,
        ).astype(np.float32)
        self._last_rand_vec = rand_vec
        return rand_vec

    def _gripper_caging_reward(
        self,
        action: npt.NDArray[np.float32],
        obj_pos: npt.NDArray[Any],
        obj_radius: float,
        pad_success_thresh: float,
        object_reach_radius: float,
        xz_thresh: float,
        desired_gripper_effort: float = 1.0,
        high_density: bool = False,
        medium_density: bool = False,
    ) -> float:
        """Reward for agent grasping obj.

        Args:
            action(np.ndarray): (4,) array representing the action
                delta(x), delta(y), delta(z), gripper_effort
            obj_pos(np.ndarray): (3,) array representing the obj x,y,z
            obj_radius(float):radius of object's bounding sphere
            pad_success_thresh(float): successful distance of gripper_pad
                to object
            object_reach_radius(float): successful distance of gripper center
                to the object.
            xz_thresh(float): successful distance of gripper in x_z axis to the
                object. Y axis not included since the caging function handles
                    successful grasping in the Y axis.
            desired_gripper_effort(float): desired gripper effort, defaults to 1.0.
            high_density(bool): flag for high-density. Cannot be used with medium-density.
            medium_density(bool): flag for medium-density. Cannot be used with high-density.

        Returns:
            the reward value
        """
        assert (
            self.obj_init_pos is not None
        ), "`obj_init_pos` must be initialized before calling this function."

        if high_density and medium_density:
            raise ValueError("Can only be either high_density or medium_density")
        # MARK: Left-right gripper information for caging reward----------------
        left_pad = self.get_body_com("leftpad")
        right_pad = self.get_body_com("rightpad")

        # get current positions of left and right pads (Y axis)
        pad_y_lr = np.hstack((left_pad[1], right_pad[1]))
        # compare *current* pad positions with *current* obj position (Y axis)
        pad_to_obj_lr = np.abs(pad_y_lr - obj_pos[1])
        # compare *current* pad positions with *initial* obj position (Y axis)
        pad_to_objinit_lr = np.abs(pad_y_lr - self.obj_init_pos[1])

        # Compute the left/right caging rewards. This is crucial for success,
        # yet counterintuitive mathematically because we invented it
        # accidentally.
        #
        # Before touching the object, `pad_to_obj_lr` ("x") is always separated
        # from `caging_lr_margin` ("the margin") by some small number,
        # `pad_success_thresh`.
        #
        # When far away from the object:
        #       x = margin + pad_success_thresh
        #       --> Thus x is outside the margin, yielding very small reward.
        #           Here, any variation in the reward is due to the fact that
        #           the margin itself is shifting.
        # When near the object (within pad_success_thresh):
        #       x = pad_success_thresh - margin
        #       --> Thus x is well within the margin. As long as x > obj_radius,
        #           it will also be within the bounds, yielding maximum reward.
        #           Here, any variation in the reward is due to the gripper
        #           moving *too close* to the object (i.e, blowing past the
        #           obj_radius bound).
        #
        # Therefore, before touching the object, this is very nearly a binary
        # reward -- if the gripper is between obj_radius and pad_success_thresh,
        # it gets maximum reward. Otherwise, the reward very quickly falls off.
        #
        # After grasping the object and moving it away from initial position,
        # x remains (mostly) constant while the margin grows considerably. This
        # penalizes the agent if it moves *back* toward `obj_init_pos`, but
        # offers no encouragement for leaving that position in the first place.
        # That part is left to the reward functions of individual environments.
        caging_lr_margin = np.abs(pad_to_objinit_lr - pad_success_thresh)
        caging_lr = [
            reward_utils.tolerance(
                pad_to_obj_lr[i],  # "x" in the description above
                bounds=(obj_radius, pad_success_thresh),
                margin=caging_lr_margin[i],  # "margin" in the description above
                sigmoid="long_tail",
            )
            for i in range(2)
        ]
        caging_y = reward_utils.hamacher_product(*caging_lr)

        # MARK: X-Z gripper information for caging reward-----------------------
        tcp = self.tcp_center
        xz = [0, 2]

        # Compared to the caging_y reward, caging_xz is simple. The margin is
        # constant (something in the 0.3 to 0.5 range) and x shrinks as the
        # gripper moves towards the object. After picking up the object, the
        # reward is maximized and changes very little
        caging_xz_margin = np.linalg.norm(self.obj_init_pos[xz] - self.init_tcp[xz])
        caging_xz_margin -= xz_thresh
        caging_xz = reward_utils.tolerance(
            np.linalg.norm(tcp[xz] - obj_pos[xz]),  # "x" in the description above
            bounds=(0, xz_thresh),
            margin=caging_xz_margin,  # "margin" in the description above
            sigmoid="long_tail",
        )

        # MARK: Closed-extent gripper information for caging reward-------------
        gripper_closed = (
            min(max(0, action[-1]), desired_gripper_effort) / desired_gripper_effort
        )

        # MARK: Combine components----------------------------------------------
        caging = reward_utils.hamacher_product(caging_y, float(caging_xz))
        gripping = gripper_closed if caging > 0.97 else 0.0
        caging_and_gripping = reward_utils.hamacher_product(caging, gripping)

        if high_density:
            caging_and_gripping = (caging_and_gripping + caging) / 2
        if medium_density:
            tcp = self.tcp_center
            tcp_to_obj = np.linalg.norm(obj_pos - tcp)
            tcp_to_obj_init = np.linalg.norm(self.obj_init_pos - self.init_tcp)
            # Compute reach reward
            # - We subtract `object_reach_radius` from the margin so that the
            #   reward always starts with a value of 0.1
            reach_margin = abs(tcp_to_obj_init - object_reach_radius)
            reach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, object_reach_radius),
                margin=reach_margin,
                sigmoid="long_tail",
            )
            caging_and_gripping = (caging_and_gripping + float(reach)) / 2

        return caging_and_gripping

    def sawyer_observation_space(self):
        pass