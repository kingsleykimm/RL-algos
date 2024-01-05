import gymnasium as gym
import numpy as np


def make_atari_env(env_name):
    env = gym.make(env_name, obs_type="rgb")
    env = AtariPreprocessing(env, noop_max=10, frame_skip=4, grayscale_obs=False)
    env = BaseWrapper(env)
    env.reset()
    return env

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
        return obsv, reward, term, trunc, info
