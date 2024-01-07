import gymnasium as gym
import numpy as np
import torch

class AtariEnv():
    def __init__(self, env_name):
        self.env = gym.make(env_name, obs_type="rgb", frameskip=1)
        self.env = gym.wrappers.AtariPreprocessing(self.env, grayscale_obs=False)
        self.env = BaseWrapper(self.env)
    def get_first_state(self):
        cur, info = self.env.reset()
        return torch.from_numpy(cur), info
    def random_action(self):
        return self.env.action_space.sample()
    def step(self):
        return self.env.step()

class BaseWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.total_rewards = 0
    def step(self, action):
        obsv, reward, term, trunc, info = self.env.step(action)
        if term or trunc:
            info['episodic_return'] = self.total_rewards
            self.env.reset()
            self.total_rewards = 0
        else:
            self.total_rewards += reward
            info['episodic_return'] = None
        return torch.from_numpy(obsv), reward, term, trunc, info
