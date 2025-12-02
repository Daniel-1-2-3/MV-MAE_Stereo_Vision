from collections import deque
from typing import Any, NamedTuple
import numpy as np
from dm_env import StepType
class FrameStackWrapper():
    def __init__(self, env, num_frames):
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

    def reset(self, rng):
        self._frames_deque.clear()
        return self._transform_observation(self._env.reset(rng))

    def step(self, time_step, action):
        return self._transform_observation(self._env.step(time_step, action))

    def __getattr__(self, name):
        return getattr(self._env, name)

class ActionRepeatWrapper:
    def __init__(self, env: FrameStackWrapper, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, time_step, action):
        reward = 0.0
        discount = 1.0
        for _ in range(self._num_repeats):
            time_step = self._env.step(time_step, action)
            reward += float(time_step["reward"]) * discount
            discount *= float(time_step["discount"])
            if time_step["step_type"] == StepType.LAST:
                break

        time_step["discount"] = discount
        time_step["reward"] = reward
        return time_step

    def __getattr__(self, name):
        return getattr(self._env, name)

class ExtendedTimeStep(NamedTuple):
    data: Any
    metrics: Any
    observation: Any
    info: Any
    step_type: Any
    action: Any
    done: Any
    reward: Any
    discount: Any

    def first(self):
        return self.step_type == StepType.FIRSTsss

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

    def reset(self, state):
        return self._augment_info(self._env.reset(state))

    def step(self, state, action):
        # The var info is a dict containing observation, step_type, action, reward, and discount
        ret = self._env.step(state, action)
        return self._augment_info(ret, action)

    def _augment_info(self, ret, action=None):
        if action is None:
            action = np.zeros(self._env.action_space.shape, dtype=self._env.action_space.dtype)
        
        reward = np.array([float(ret["reward"])], dtype=np.float32)
        discount = np.array([float(ret["discount"])], dtype=np.float32)
        return ExtendedTimeStep(
            data=ret["data"],
            metrics=ret["metrics"],
            observation=ret["observation"],
            info=ret["info"],
            step_type=ret["step_type"],
            action=action.astype(np.float32),
            done=float(ret["done"]),
            reward=reward,
            discount=discount,
        )

    def __getattr__(self, name):
        return getattr(self._env, name)