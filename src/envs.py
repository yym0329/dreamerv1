import numpy as np
import torch
from dm_control import suite
from dm_control.suite.wrappers import pixels

class NormalizeWrapper:
    def __init__(self, env):
        self.env = env
        self.action_spec = env.action_spec

    def reset(self):
        timestep = self.env.reset()
        return self._normalize(timestep)

    def step(self, action):
        timestep = self.env.step(action)
        return self._normalize(timestep)

    def _normalize(self, timestep):
        obs = timestep.observation.copy()
        obs['pixels'] = obs['pixels'].astype(np.float32) / 255.0
        return timestep._replace(observation=obs)

    def last(self):
        return self.env.last()

def init_env(env_name, task_name, height=64, width=64, camera_id=0, seed=42):
    env = suite.load(env_name, task_name, task_kwargs={'random': seed})
    env = pixels.Wrapper(env, render_kwargs={'height': height, 'width': width, 'camera_id': camera_id})
    env = NormalizeWrapper(env)
    return env
