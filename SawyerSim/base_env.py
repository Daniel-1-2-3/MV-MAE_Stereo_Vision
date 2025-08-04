"""Base classes for all the envs."""

from __future__ import annotations

from functools import cached_property
from typing import Any, Literal, SupportsFloat

import torch
import torch.nn.functional as F
import cv2
import mujoco
import random
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box, Dict
from gymnasium.utils import seeding
from gymnasium.utils.ezpickle import EzPickle
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from typing_extensions import TypeAlias

from metaworld.types import XYZ, ObservationDict
from metaworld.sawyer_xyz_env import SawyerMocapBase
from metaworld.utils import reward_utils

from MAE_Model.prepare_input import Prepare

RenderMode: TypeAlias = "Literal['human', 'rgb_array', 'depth_array']"

class SawyerXYZEnv(SawyerMocapBase, EzPickle):
    """The base environment for all Sawyer Mujoco envs that use mocap for XYZ control."""

    _HAND_SPACE = Box( # Constraints for hand positions
        np.array([-0.525, 0.348, -0.0525]),
        np.array([+0.525, 1.025, 0.7]),
        dtype=np.float64,
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
        width: int = 480,
        height: int = 480,
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

        self.width = width
        self.height = height
        
        self._obs_obj_max_len = 14
        
        super().__init__(
            self.model_name,
            frame_skip=frame_skip,
            render_mode=render_mode,
            camera_name=None,
            camera_id=None,
            width=width,
            height=height,
        )

        mujoco.mj_forward(
            self.model, self.data
        )  # DO NOT REMOVE: EZPICKLE WON'T WORK

        self._did_see_sim_exception: bool = False # If sim hits an unstable physics state
        self.init_left_pad: npt.NDArray[Any] = self.get_body_com("leftpad") # Initial center of mass
        self.init_right_pad: npt.NDArray[Any] = self.get_body_com("rightpad")

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([+1, +1, +1, +1]),
            dtype=np.float32,
        )
        
        self.hand_init_pos: npt.NDArray[Any] | None = None  # OVERRIDE ME
        self._target_pos: npt.NDArray[Any] | None = None  # OVERRIDE ME
        self._random_reset_space: Box | None = None  # OVERRIDE ME
        self.goal_space: Box | None = None  # OVERRIDE ME
        self._last_stable_obs: npt.NDArray[np.float64] | None = None

        self.init_qpos = np.copy(self.data.qpos)
        self.init_qvel = np.copy(self.data.qvel)
        
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
        
        # Custom Mujoco renders for stereo vision
        default_camera_config = None
        max_geom = 1000
        visual_options = {}
        self.camera_pairs = camera_pairs
        chosen_pair = random.choice(self.camera_pairs)
        self.stereo_renderer_left = MujocoRenderer(
            self.model,
            self.data,
            default_camera_config,
            self.width,
            self.height,
            max_geom,
            camera_id,
            chosen_pair[0],
            visual_options,
        )
        self.stereo_renderer_right = MujocoRenderer(
            self.model,
            self.data,
            default_camera_config,
            self.width,
            self.height,
            max_geom,
            camera_id,
            chosen_pair[1],
            visual_options,
        )

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

    def _get_site_pos(self, site_name: str) -> npt.NDArray[np.float64]:
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
    
    def _get_obs(self) -> npt.NDArray[np.float64]:
        """Returns a dictionary comprised of tensor representation of the image and state observations
        Image observations: a single tensor of shape (batch, height, width_total, channels), for passing into mvmae
        State observations: a flat numpy vector of n elements

        Returns:
            The flat observation array (39 elements)
        """
        state_obs = self._get_curr_obs_combined_no_goal()
        state_obs = torch.from_numpy(state_obs.astype(np.float32)).unsqueeze(0)
        img_obs = self.render()
        print(f"[DEBUG] _get_obs() image_observation BASE_ENV: min={img_obs.min().item():.4f}, max={img_obs.max().item():.4f}")
        obs = {
            "state_observation": state_obs,
            "image_observation": img_obs
        }
        return obs
    
    def render(self):
        """Override the Mujoco render method so that I can return two views for stereo vision"""
        left_view = np.flipud(self.stereo_renderer_left.render("rgb_array"))
        right_view = np.flipud(self.stereo_renderer_right.render("rgb_array"))
        
        stereo_image = np.concatenate([left_view, right_view], axis=1)
        if self.render_mode == "human":
            cv2.imshow("Stereo view", stereo_image)
            cv2.waitKey(1)
            
        left_tensor = torch.from_numpy(np.transpose(left_view, (2, 0, 1)).copy()).unsqueeze(0).float() / 255.0
        left_tensor = F.interpolate(left_tensor, size=(128, 128), mode='bilinear', align_corners=False)

        right_tensor = torch.from_numpy(np.transpose(right_view, (2, 0, 1)).copy()).unsqueeze(0).float() / 255.0
        right_tensor = F.interpolate(right_tensor, size=(128, 128), mode='bilinear', align_corners=False)

        stereo_tensor = Prepare.fuse_normalize([left_tensor, right_tensor])
        return stereo_tensor
    
    def render_for_test_dataset(self):
        """Override the Mujoco render method so that I can return two views for stereo vision"""
        left_view = np.flipud(self.stereo_renderer_left.render("rgb_array"))
        right_view = np.flipud(self.stereo_renderer_right.render("rgb_array"))
        
        stereo_image = np.concatenate([left_view, right_view], axis=1)
        if self.render_mode == "human":
            cv2.imshow("Stereo view", stereo_image)
            cv2.waitKey(1)
        
        return left_view, right_view

    @cached_property
    def sawyer_observation_space(self) -> Dict:
        observation_space = Dict({
            "state_observation": Box(low=-np.inf, high=np.inf, shape=(3 + 1 + self._obs_obj_max_len,), dtype=np.float32),
            "image_observation": Box(low=0, high=255, shape=(128, 256, 3), dtype=np.uint8)
        })
        return observation_space
 
    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float64], SupportsFloat, bool, bool, dict[str, Any]]:
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
        self._last_stable_obs = self._get_obs()

        self._last_stable_obs["state_observation"] = np.clip(
            self._last_stable_obs["state_observation"],
            a_max=self.sawyer_observation_space["state_observation"].high,
            a_min=self.sawyer_observation_space["state_observation"].low,
            dtype=np.float64,
        )
        reward, info = self.evaluate_state(self._last_stable_obs, action)
        # step will never return a terminate==True if there is a success
        # but we can return truncate=True if the current path length == max path length
        truncate = False
        if self.curr_path_length == self.max_path_length:
            truncate = True
            
        return self._last_stable_obs, reward, False, truncate, info

    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        """Does the heavy-lifting for `step()` -- namely, calculating reward and populating the `info` dict with training metrics.

        Returns:
            Tuple of reward between 0 and 10 and a dictionary which contains useful metrics (success,
                near_object, grasp_success, grasp_reward, in_place_reward,
                obj_to_target, unscaled_reward)
        """
        # Throw error rather than making this an @abc.abstractmethod so that
        # V1 environments don't have to implement it
        raise NotImplementedError

    def reset_model(self) -> npt.NDArray[np.float64]:
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_camera_placement(self):
        camera_id = None
        default_camera_config = None
        max_geom = 1000
        visual_options = {}
        chosen_pair = random.choice(self.camera_pairs) # Switch to another random camera location
        self.stereo_renderer_left = MujocoRenderer(
            self.model,
            self.data,
            default_camera_config,
            self.width,
            self.height,
            max_geom,
            camera_id,
            chosen_pair[0],
            visual_options,
        )
        self.stereo_renderer_right = MujocoRenderer(
            self.model,
            self.data,
            default_camera_config,
            self.width,
            self.height,
            max_geom,
            camera_id,
            chosen_pair[1],
            visual_options,
        )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
        """Resets the environment.

        Args:
            seed: The seed to use. Ignored, use `seed()` instead.
            options: Additional options to pass to the environment. Ignored.

        Returns:
            The `(obs, info)` tuple.
        """
        self.curr_path_length = 0
        self.reset_camera_placement()
        self.reset_model()
        obs, info = super().reset()
        return obs, info

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

    def _get_state_rand_vec(self) -> npt.NDArray[np.float64]:
        """Gets or generates a random vector for the hand position at reset."""
        # Removed caching, simply generate a random hand position
        assert self._random_reset_space is not None
        rand_vec: npt.NDArray[np.float64] = np.random.uniform(
            self._random_reset_space.low,
            self._random_reset_space.high,
            size=self._random_reset_space.low.size,
        ).astype(np.float64)
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
