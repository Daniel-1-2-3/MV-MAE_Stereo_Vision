from collections import deque
from typing import Any, NamedTuple
import numpy as np
from dm_env import StepType
from Outdated.SawyerSim.sawyer_stereo_reach_env import SawyerReachEnvV3
class FrameStackWrapper():
    def __init__(self, env: SawyerReachEnvV3, num_frames):
        self._env = env
        self._num_frames = num_frames
        self._frames_deque = deque([], maxlen=num_frames)
    
    def _transform_observation(self, time_step):
        self._frames_deque.append(time_step["observation"])
        if len(self._frames_deque) != self._num_frames:
            last_frame: np.ndarray = self._frames_deque[-1]
            for i in range(self._num_frames - len(self._frames_deque)):
                self._frames_deque.append(last_frame.copy())
        
        assert len(self._frames_deque) == self._num_frames
        time_step["observation"] = np.concatenate(list(self._frames_deque), axis=2)
        return time_step

    def reset(self):
        self._frames_deque.clear()
        return self._transform_observation(self._env.reset())

    def step(self, action):
        return self._transform_observation(self._env.step(action))

    def __getattr__(self, name):
        return getattr(self._env, name)

class ActionRepeatWrapper:
    def __init__(self, env: FrameStackWrapper, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for _ in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step["reward"] or 0.0) * discount
            discount *= time_step["discount"]
            if time_step["step_type"] == StepType.LAST:
                break
            
        time_step["discount"] = discount
        time_step["reward"] = reward
        return time_step

    def __getattr__(self, name):
        return getattr(self._env, name)

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)

class ExtendedTimeStepWrapper:
    def __init__(self, env: ActionRepeatWrapper):
        self._env = env

    def reset(self):
        return self._augment_info(self._env.reset())

    def step(self, action):
        # The var info is a dict containing observation, step_type, action, reward, and discount
        info = self._env.step(action)
        return self._augment_info(info, action)

    def _augment_info(self, info, action=None):
        if action is None:
            action = np.zeros(self._env.action_space.shape, dtype=self._env.action_space.dtype)
        return ExtendedTimeStep(
            observation=info["observation"],
            step_type=info["step_type"],
            reward=np.array([info["reward"] if info["reward"] is not None else 0.0], dtype=np.float32),
            discount=np.array([info["discount"]], dtype=np.float32),
            action=action,
        )

    def __getattr__(self, name):
        return getattr(self._env, name)