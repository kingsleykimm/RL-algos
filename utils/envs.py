import gymnasium as gym
import numpy as np


def make_atari_env(env_name):
    env = gym.make(env_name, obs_type="rgb")
    env = AtariPreprocessing(env, noop_max=10, frame_skip=4, grayscale_obs=False)

